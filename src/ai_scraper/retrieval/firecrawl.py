"""Credentialless one-URL batches on an explicitly trusted local Firecrawl.

The service owns its target connections and rendering. The adapter checks the
requested and service-reported URLs but cannot attest or pin those connections.
No hosted endpoint, inference format, actions, webhook or caller URL is followed.
A lost POST acknowledgment requires reconciliation; only GET polling is replayed.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import json
import math
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlsplit
from uuid import UUID

import httpx

from ..contracts import RetrievalOutcome, ScraperError
from ..jobs import digest
from ..policy import URLPolicy
from ..schemas import AppError, ExtractionRequest
from .brightdata import MAX_RESPONSE_BYTES, _retry_after, bounded_request
from .direct import _strip_credentials

FIRECRAWL_POLICY_VERSION = "firecrawl-trusted-service-v1"
SERVICE_WARNING = (
    "firecrawl_service_egress: target connections and rendering are delegated to the configured service; "
    "the final destination is service-reported, not locally observed or IP-pinned"
)
_RUNTIME = re.compile(r"(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})\Z")
_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,79}\Z")
FORMATS = ("rawHtml", "html", "markdown")


def validate_firecrawl_endpoint(value: str) -> str:
    """Require a literal loopback endpoint with an explicit nonprivileged port."""
    try:
        parsed = urlsplit(value)
        address = ipaddress.ip_address(parsed.hostname or "")
        valid = (
            parsed.scheme in {"http", "https"}
            and address.is_loopback
            and parsed.username is None
            and parsed.password is None
            and parsed.port is not None
            and 1024 <= parsed.port <= 65535
            and parsed.path in {"", "/"}
            and not parsed.query
            and not parsed.fragment
            and "?" not in value
            and "#" not in value
            and "%" not in value
            and not any(ord(char) < 33 or ord(char) == 127 for char in value)
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Firecrawl requires a literal loopback HTTP(S) endpoint with an explicit high port")
    return value.rstrip("/")


def validate_runtime_id(value: str) -> str:
    if not isinstance(value, str) or not _RUNTIME.fullmatch(value):
        raise ValueError("Firecrawl requires a pinned 40-hex commit or sha256 image digest")
    return value


def firecrawl_cache_options(base_url: str, runtime_id: str, trust_service_egress: bool) -> dict:
    if trust_service_egress is not True:
        raise ValueError("Explicit FIRECRAWL_TRUST_SERVICE_EGRESS=true is required")
    endpoint = validate_firecrawl_endpoint(base_url)
    runtime = validate_runtime_id(runtime_id)
    return {
        "firecrawl_policy_version": FIRECRAWL_POLICY_VERSION,
        "endpoint_sha256": hashlib.sha256(endpoint.encode()).hexdigest(),
        "runtime_id": runtime,
        "formats": list(FORMATS),
        "parsers": [],
        "only_main_content": False,
        "max_age": 0,
        "store_in_cache": False,
        "skip_tls_verification": False,
        "trust_service_egress": True,
    }


def validate_batch_id(value: Any) -> str:
    try:
        canonical = str(UUID(value)) if isinstance(value, str) else None
    except ValueError:
        canonical = None
    if canonical is None or canonical != value.lower():
        raise ScraperError("FETCH_INVALID_JOB_ID", "Firecrawl returned an invalid batch identifier")
    return canonical


def _unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON keys")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("Nonfinite JSON")


def _json(response):
    try:
        body = json.loads(response.content, object_pairs_hook=_unique, parse_constant=_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise ScraperError(
            "FETCH_INVALID_RESPONSE", "Firecrawl did not return a strict JSON object"
        ) from None
    if not isinstance(body, dict):
        raise ScraperError("FETCH_INVALID_RESPONSE", "Firecrawl did not return a JSON object")
    return body


class FirecrawlRetriever:
    retrieval_method = "firecrawl_self_hosted"

    def __init__(
        self,
        policy: URLPolicy,
        *,
        base_url: str,
        runtime_id: str,
        trust_service_egress: bool = False,
        client: httpx.AsyncClient | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_attempts: int = 3,
        poll_interval: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], float] = time.time,
    ):
        if trust_service_egress is not True:
            raise ValueError("Explicit FIRECRAWL_TRUST_SERVICE_EGRESS=true is required")
        self.base_url = validate_firecrawl_endpoint(base_url)
        self.runtime_id = validate_runtime_id(runtime_id)
        if type(max_response_bytes) is not int or not 1024 <= max_response_bytes <= 100_000_000:
            raise ValueError("Firecrawl response limit is outside supported bounds")
        if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
            raise ValueError("Firecrawl retry count is outside supported bounds")
        if not math.isfinite(poll_interval) or not 0.1 <= poll_interval <= 10:
            raise ValueError("Firecrawl poll interval is outside supported bounds")
        self.policy, self.max_response_bytes, self.max_attempts = policy, max_response_bytes, max_attempts
        self.poll_interval, self.sleep, self.clock, self.wall_clock = poll_interval, sleep, clock, wall_clock
        self._owns_client = client is None
        self.client = client or httpx.AsyncClient(trust_env=False, follow_redirects=False)
        self.client.event_hooks["request"].append(_strip_credentials)
        self.cache_options = firecrawl_cache_options(base_url, runtime_id, trust_service_egress)
        self.binding = digest(self.cache_options)

    def validate_job_identity(self, job: dict) -> None:
        binding = job.get("firecrawl_binding")
        if binding is not None and binding != self.binding:
            raise ScraperError(
                "FIRECRAWL_CONFIG_CHANGED", "Resume requires the original Firecrawl configuration"
            )
        record = job.get("record")
        if record is not None:
            service = record.get("firecrawl_service") if isinstance(record, dict) else None
            if (
                not isinstance(service, dict)
                or any(key not in service for key in self.cache_options)
                or digest({key: service[key] for key in self.cache_options}) != self.binding
            ):
                raise ScraperError(
                    "FIRECRAWL_CONFIG_CHANGED", "Saved source has a different Firecrawl configuration"
                )

    async def close(self):
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _checkpoint(job, checkpoint, **changes):
        checkpoint(changes)
        job.update(changes)

    def _outcome(self, job, status, *, record=None, error=None):
        return RetrievalOutcome(
            status=status,
            record=record,
            snapshot_id=job.get("snapshot_id"),
            retrieval_method=self.retrieval_method,
            error=error,
            warnings=[SERVICE_WARNING],
            attempts=int(job.get("attempts", 0)),
        )

    def _fail(self, job, checkpoint, code, message):
        error = AppError(code=code, message=message)
        self._checkpoint(job, checkpoint, state="failed", error=error.model_dump(mode="json"))
        return self._outcome(job, "failed", error=error)

    def _ambiguous(self, job, checkpoint):
        error = AppError(
            code="RECONCILIATION_REQUIRED",
            message="Firecrawl submission outcome is unknown; recover its batch ID before resuming",
        )
        self._checkpoint(
            job, checkpoint, state="reconciliation_required", error=error.model_dump(mode="json")
        )
        return self._outcome(job, "pending", error=error)

    async def _cooldown(self, job, checkpoint, deadline):
        delay = max(0.0, float(job.get("firecrawl_retry_not_before") or 0) - self.wall_clock())
        if delay >= deadline - self.clock():
            return False
        if delay:
            await self.sleep(delay)
            self._checkpoint(job, checkpoint, firecrawl_retry_not_before=None)
        return self.clock() < deadline

    async def _request(self, method, path, deadline, **kwargs):
        remaining = deadline - self.clock()
        return await bounded_request(
            self.client,
            method,
            self.base_url + path,
            max_response_bytes=self.max_response_bytes,
            headers={"Content-Type": "application/json"},
            timeout=httpx.Timeout(min(75.0, remaining), connect=min(10.0, remaining)),
            **kwargs,
        )

    def _delay(self, response, attempt):
        value = _retry_after(response, min(2**attempt, 10))
        return value if math.isfinite(value) and value >= 0 else 10.0

    async def retrieve(self, request: ExtractionRequest, job: dict, checkpoint, wait_seconds: float):
        if request.country is not None or request.browser_actions:
            raise ScraperError(
                "INVALID_REQUEST", "Firecrawl retrieval does not support country or browser actions"
            )
        if type(wait_seconds) not in {int, float} or not math.isfinite(wait_seconds) or wait_seconds < 0:
            raise ValueError("A finite nonnegative waiting budget is required")
        await self.policy.check(request.url)
        self.validate_job_identity(job)
        binding = job.get("firecrawl_binding")
        if binding is None:
            if job.get("state", "new") != "new" or job.get("snapshot_id"):
                raise ScraperError(
                    "FIRECRAWL_CONFIG_CHANGED", "Saved Firecrawl job has no configuration binding"
                )
            self._checkpoint(
                job, checkpoint, firecrawl_binding=self.binding, retrieval_method=self.retrieval_method
            )
        if job.get("record") is not None:
            await self.policy.check_record(job["record"], request.url)
            return self._outcome(job, "ready", record=job["record"])
        if job.get("state") == "failed":
            return self._outcome(job, "failed", error=AppError.model_validate(job["error"]))
        deadline = self.clock() + wait_seconds
        if job.get("snapshot_id"):
            identifier = validate_batch_id(job["snapshot_id"])
        elif job.get("state") in {"submitting", "pending", "reconciliation_required"}:
            return self._ambiguous(job, checkpoint)
        else:
            identifier = await self._submit(request, job, checkpoint, deadline)
            if isinstance(identifier, RetrievalOutcome):
                return identifier
        return await self._poll(request, job, checkpoint, deadline, identifier)

    async def _submit(self, request, job, checkpoint, deadline):
        for attempt in range(self.max_attempts):
            if not await self._cooldown(job, checkpoint, deadline):
                return self._outcome(job, "pending")
            self._checkpoint(job, checkpoint, state="submitting", attempts=int(job.get("attempts", 0)) + 1)
            try:
                response = await self._request(
                    "POST",
                    "/v2/batch/scrape",
                    deadline,
                    json={
                        "urls": [request.url],
                        "formats": list(FORMATS),
                        "parsers": [],
                        "onlyMainContent": False,
                        "maxAge": 0,
                        "storeInCache": False,
                        "skipTlsVerification": False,
                        "ignoreInvalidURLs": False,
                        "maxConcurrency": 1,
                    },
                )
            except (httpx.TransportError, ScraperError):
                # A bounded-body failure can also occur after the server accepted work.
                return self._ambiguous(job, checkpoint)
            if response.status_code == 429:
                delay = self._delay(response, attempt)
                self._checkpoint(
                    job, checkpoint, state="new", firecrawl_retry_not_before=self.wall_clock() + delay
                )
                if attempt + 1 == self.max_attempts:
                    return self._outcome(
                        job,
                        "pending",
                        error=AppError(
                            code="FETCH_RATE_LIMITED",
                            message="Firecrawl rejected submission with a rate limit",
                            retryable=True,
                        ),
                    )
                continue
            if response.status_code >= 500 or response.status_code == 408:
                return self._ambiguous(job, checkpoint)
            if response.status_code not in {200, 202}:
                return self._fail(
                    job, checkpoint, "FETCH_HTTP_ERROR", "Firecrawl rejected the batch submission"
                )
            try:
                body = _json(response)
                if body.get("success") is not True or body.get("invalidURLs") not in (None, []):
                    raise ScraperError(
                        "FETCH_INVALID_RESPONSE", "Firecrawl did not acknowledge exactly one URL"
                    )
                identifier = validate_batch_id(body.get("id"))
            except ScraperError:
                return self._ambiguous(job, checkpoint)
            self._checkpoint(job, checkpoint, state="pending", snapshot_id=identifier, error=None)
            return identifier
        raise AssertionError("unreachable")

    async def _safe_get(self, path, job, checkpoint, deadline):
        for attempt in range(self.max_attempts):
            if not await self._cooldown(job, checkpoint, deadline):
                return None
            try:
                response = await self._request("GET", path, deadline)
            except httpx.TransportError:
                response = None
            if response is not None and response.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                return response
            delay = self._delay(response, attempt) if response is not None else min(2**attempt, 10)
            self._checkpoint(job, checkpoint, firecrawl_retry_not_before=self.wall_clock() + delay)
        return None

    async def _terminal_error(self, request, job, checkpoint, deadline, identifier):
        response = await self._safe_get(f"/v2/batch/scrape/{identifier}/errors", job, checkpoint, deadline)
        if response is None:
            return self._outcome(job, "pending")
        if response.status_code != 200:
            return self._fail(
                job, checkpoint, "FETCH_JOB_FAILED", "Firecrawl batch failed or returned no document"
            )
        body = _json(response)
        errors, blocked = body.get("errors", []), body.get("robotsBlocked", [])
        if (
            not isinstance(errors, list)
            or not isinstance(blocked, list)
            or len(errors) > 1
            or len(blocked) > 1
        ):
            return self._fail(
                job,
                checkpoint,
                "FETCH_INVALID_RESPONSE",
                "Firecrawl error response does not match a one-URL batch",
            )
        matching = [item for item in errors if isinstance(item, dict) and item.get("url") == request.url]
        codes = [
            item["code"]
            for item in matching
            if isinstance(item.get("code"), str) and _CODE.fullmatch(item["code"])
        ]
        self._checkpoint(
            job,
            checkpoint,
            firecrawl_error_counts={
                "errors": len(errors),
                "robots_blocked": len(blocked),
                "matching_codes": codes,
            },
        )
        return self._fail(
            job,
            checkpoint,
            "FETCH_RECORD_ERROR" if errors or blocked else "FETCH_JOB_FAILED",
            "Firecrawl did not deliver the requested page",
        )

    async def _poll(self, request, job, checkpoint, deadline, identifier):
        try:
            while self.clock() < deadline:
                response = await self._safe_get(f"/v2/batch/scrape/{identifier}", job, checkpoint, deadline)
                if response is None:
                    return self._outcome(job, "pending")
                if response.status_code != 200:
                    return self._fail(
                        job, checkpoint, "FETCH_HTTP_ERROR", "Firecrawl batch status request was rejected"
                    )
                body = _json(response)
                status = body.get("status")
                if status in {"failed", "cancelled"}:
                    return await self._terminal_error(request, job, checkpoint, deadline, identifier)
                total, completed, data = body.get("total"), body.get("completed"), body.get("data")
                if (
                    body.get("success") is not True
                    or status not in {"scraping", "completed"}
                    or type(total) is not int
                    or type(completed) is not int
                    or not 0 <= completed <= total <= 1
                    or not isinstance(data, list)
                    or len(data) > 1
                ):
                    return self._fail(
                        job,
                        checkpoint,
                        "FETCH_INVALID_RESPONSE",
                        "Firecrawl status does not describe a one-URL batch",
                    )
                if status == "completed":
                    if not data:
                        return await self._terminal_error(request, job, checkpoint, deadline, identifier)
                    if total != 1 or completed != 1 or body.get("next"):
                        return self._fail(
                            job,
                            checkpoint,
                            "FETCH_RECORD_MAPPING",
                            "Firecrawl completed batch does not match one source",
                        )
                    try:
                        async with asyncio.timeout(max(0.001, deadline - self.clock())):
                            record = await self._record(data[0], request)
                    except TimeoutError:
                        return self._outcome(job, "pending")
                    self._checkpoint(job, checkpoint, state="fetched", record=record, error=None)
                    return self._outcome(job, "ready", record=record)
                pause = min(self.poll_interval, max(0.0, deadline - self.clock()))
                if pause:
                    await self.sleep(pause)
            return self._outcome(job, "pending")
        except ScraperError as error:
            self._checkpoint(job, checkpoint, state="failed", error=error.error.model_dump(mode="json"))
            return self._outcome(job, "failed", error=error.error)

    async def _record(self, data, request):
        if not isinstance(data, dict) or not isinstance(data.get("metadata"), dict):
            raise ScraperError("FETCH_INVALID_RESPONSE", "Firecrawl document metadata is missing")
        metadata = data["metadata"]
        if metadata.get("sourceURL") != request.url:
            raise ScraperError(
                "FETCH_RECORD_MAPPING", "Firecrawl source URL does not match the requested URL"
            )
        code = metadata.get("statusCode")
        errors = (metadata.get("error"), data.get("error"))
        if (
            type(code) is not int
            or not 200 <= code < 300
            or any(value is not None and (not isinstance(value, str) or value) for value in errors)
        ):
            raise ScraperError("FETCH_RECORD_ERROR", "Firecrawl returned a page error or invalid HTTP status")
        if not isinstance(metadata.get("url"), str) or not metadata["url"]:
            raise ScraperError(
                "FETCH_DESTINATION_UNVERIFIABLE", "Firecrawl did not report its engine destination"
            )
        if any(not isinstance(data.get(name), str) for name in FORMATS):
            raise ScraperError(
                "FETCH_INVALID_RESPONSE", "Firecrawl did not return all requested source representations"
            )
        record = {
            "input": {"url": request.url},
            "final_url": metadata["url"],
            "page_html": data["rawHtml"],
            "firecrawl_html": data["html"],
            "markdown": data["markdown"],
            "firecrawl_metadata": {
                key: metadata[key]
                for key in (
                    "sourceURL",
                    "url",
                    "statusCode",
                    "title",
                    "description",
                    "language",
                    "scrapeId",
                    "contentType",
                )
                if key in metadata and (isinstance(metadata[key], str) or type(metadata[key]) is int)
            },
            "firecrawl_service": {
                **self.cache_options,
                "destination_basis": "service_reported_engine_url",
                "native_connection_pinning": False,
            },
        }
        title = metadata.get("title")
        if isinstance(title, str):
            record["page_title"] = title
        chain = []
        for container in (data, metadata):
            for key in ("redirects", "redirect_chain", "redirectChain"):
                if key not in container:
                    continue
                values = container[key]
                if not isinstance(values, list) or len(values) > 20:
                    raise ScraperError(
                        "FETCH_INVALID_RESPONSE", "Firecrawl redirect metadata is invalid or oversized"
                    )
                for item in values:
                    url = item.get("url") if isinstance(item, dict) else item
                    if not isinstance(url, str):
                        raise ScraperError("FETCH_INVALID_RESPONSE", "Firecrawl redirect metadata has no URL")
                    chain.append({"url": url})
                    if len(chain) > 20:
                        raise ScraperError(
                            "FETCH_INVALID_RESPONSE", "Firecrawl redirect chain exceeds its bound"
                        )
        if chain:
            record["redirect_chain"] = chain
        if len(json.dumps(record, ensure_ascii=False, allow_nan=False).encode()) > self.max_response_bytes:
            raise ScraperError(
                "CONTENT_TOO_LARGE", "Normalized Firecrawl record exceeds the configured byte limit"
            )
        await self.policy.check_record(record, request.url)
        return record
