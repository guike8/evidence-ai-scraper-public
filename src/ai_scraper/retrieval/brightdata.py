"""Bright Data Fetch REST adapter with durable checkpoints and safe resumption.

A POST transport failure is ambiguous: it is never retried automatically. A
caller must reconcile it with the vendor and attach a recovered snapshot ID.
Safe GET requests and explicit HTTP 429 submission rejections are retryable.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import quote

import httpx

from ..contracts import RetrievalOutcome, ScraperError
from ..schemas import AppError, ExtractionRequest

Checkpoint = Callable[[dict[str, Any]], None]
FETCH_DATASET_ID = "gd_m6gjtfmeh43we6cqc"
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}
MAX_RESPONSE_BYTES = 5_000_000


async def _stream_request(
    client: httpx.AsyncClient, method: str, url: str, *, max_response_bytes: int, **kwargs: Any
) -> httpx.Response:
    """Bound the HTTP body before JSON parsing or text normalization.

    Request identity encoding and reject unexpected compression so a decompressor
    cannot expand an untrusted chunk beyond this byte limit before we count it.
    """
    headers = dict(kwargs.pop("headers", {}))
    headers["Accept-Encoding"] = "identity"
    async with client.stream(method, url, headers=headers, follow_redirects=False, **kwargs) as response:
        declared = response.headers.get("Content-Length")
        try:
            oversized = declared is not None and int(declared) > max_response_bytes
        except ValueError:
            oversized = False
        if oversized:
            raise ScraperError("CONTENT_TOO_LARGE", "Vendor response exceeds the configured byte limit")
        encoding = response.headers.get("Content-Encoding", "identity").strip().lower()
        if encoding not in {"", "identity"}:
            raise ScraperError(
                "FETCH_INVALID_RESPONSE", "Unexpected compressed response to identity-only request"
            )
        if response.is_stream_consumed:
            # Eager MockTransport fixtures are already buffered by HTTPX itself;
            # real network responses remain streamed until the loop below.
            body = response.content
            if len(body) > max_response_bytes:
                raise ScraperError("CONTENT_TOO_LARGE", "Vendor response exceeds the configured byte limit")
        else:
            chunks = []
            size = 0
            async for chunk in response.aiter_raw():
                size += len(chunk)
                if size > max_response_bytes:
                    raise ScraperError(
                        "CONTENT_TOO_LARGE", "Vendor response exceeds the configured byte limit"
                    )
                chunks.append(chunk)
            body = b"".join(chunks)
        return httpx.Response(
            response.status_code, headers=response.headers, content=body, request=response.request
        )


async def bounded_request(
    client: httpx.AsyncClient, method: str, url: str, *, max_response_bytes: int, **kwargs: Any
) -> httpx.Response:
    """Read a byte-bounded identity-encoded response under a total deadline.

    The supplied HTTPX read timeout also bounds the complete request, preventing
    a slow stream from extending its deadline with a succession of small chunks.
    """
    if type(max_response_bytes) is not int or max_response_bytes <= 0:
        raise ValueError("max_response_bytes must be a positive integer")
    timeout = kwargs.get("timeout")
    total_timeout = timeout.read if isinstance(timeout, httpx.Timeout) else timeout
    total_timeout = 75.0 if total_timeout is None else float(total_timeout)
    if not math.isfinite(total_timeout) or total_timeout <= 0:
        raise ValueError("A finite, positive total timeout is required")
    try:
        async with asyncio.timeout(total_timeout):
            return await _stream_request(client, method, url, max_response_bytes=max_response_bytes, **kwargs)
    except TimeoutError:
        raise httpx.ReadTimeout("Vendor request exceeded its total timeout") from None


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]"
            if any(s in str(key).lower() for s in ("authorization", "token", "password", "secret", "api_key"))
            else _redact(val, secret)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    return value


def _retry_after(response: httpx.Response, fallback: float) -> float:
    raw = response.headers.get("Retry-After")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            try:
                return max(0.0, (parsedate_to_datetime(raw) - datetime.now(UTC)).total_seconds())
            except (ValueError, TypeError, OverflowError):
                pass
    return fallback


def _body(response: httpx.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return {"response_excerpt": response.text[:1000]}


class BrightDataRetriever:
    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.AsyncClient | None = None,
        dataset_id: str = FETCH_DATASET_ID,
        base_url: str = "https://api.brightdata.com",
        max_attempts: int = 3,
        poll_interval: float = 1.0,
        max_poll_interval: float = 10.0,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self.api_key = api_key
        self.client = client or httpx.AsyncClient(follow_redirects=False)
        self._owns_client = client is None
        self.dataset_id = dataset_id
        self.base_url = base_url.rstrip("/")
        self.max_attempts = max_attempts
        self.poll_interval = max(0.0, poll_interval)
        self.max_poll_interval = max(0.0, max_poll_interval)
        self.max_response_bytes = max_response_bytes
        self.sleep = sleep
        self.clock = clock

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _details(self, response: httpx.Response) -> dict[str, Any]:
        return {"http_status": response.status_code, "response": _redact(_body(response), self.api_key)}

    @staticmethod
    def _input(request: ExtractionRequest) -> dict[str, str]:
        item = {"url": request.url}
        if request.country is not None:
            item["country"] = request.country
        return item

    def _outcome(self, job: dict[str, Any], status: str, **kwargs: Any) -> RetrievalOutcome:
        kwargs.setdefault("snapshot_id", job.get("snapshot_id"))
        kwargs.setdefault("attempts", int(job.get("attempts", 0)))
        kwargs.setdefault("retrieval_method", "brightdata_fetch")
        return RetrievalOutcome(status=status, **kwargs)

    def _checkpoint(self, job: dict[str, Any], checkpoint: Checkpoint, **changes: Any) -> None:
        # Persist first. If durable storage fails, never proceed to a paid call.
        checkpoint(changes)
        job.update(changes)

    def _fail(self, job: dict[str, Any], checkpoint: Checkpoint, error: AppError) -> RetrievalOutcome:
        self._checkpoint(job, checkpoint, state="failed", error=error.model_dump(mode="json"))
        return self._outcome(job, "failed", error=error)

    def _ambiguous(
        self, job: dict[str, Any], checkpoint: Checkpoint, reason: str, details: dict[str, Any] | None = None
    ) -> RetrievalOutcome:
        error = AppError(
            code="RECONCILIATION_REQUIRED",
            message="Submission outcome is unknown; reconcile with Bright Data and attach its snapshot ID before resuming.",
            vendor_details={"reason": reason, **(details or {})},
        )
        self._checkpoint(
            job, checkpoint, state="reconciliation_required", error=error.model_dump(mode="json")
        )
        return self._outcome(job, "pending", error=error)

    def _record(
        self, payload: Any, request: ExtractionRequest, job: dict[str, Any], checkpoint: Checkpoint
    ) -> RetrievalOutcome:
        if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
            return self._fail(
                job,
                checkpoint,
                AppError(
                    code="FETCH_INVALID_RESPONSE",
                    message="Fetch record response must be an array of objects.",
                ),
            )
        # Inspect every returned record, even records belonging to another input.
        errors = [row for row in payload if row.get("error_code") or row.get("error")]
        matching = [
            row
            for row in payload
            if isinstance(row.get("input"), dict) and row["input"].get("url") == request.url
        ]
        if len(matching) != 1:
            return self._fail(
                job,
                checkpoint,
                AppError(
                    code="FETCH_RECORD_MAPPING",
                    message="Response does not contain exactly one record for the input URL.",
                    vendor_details={
                        "record_count": len(payload),
                        "matching_count": len(matching),
                        "record_errors": _redact(errors, self.api_key),
                    },
                ),
            )
        record = matching[0]
        if record.get("error_code") or record.get("error"):
            code = str(record.get("error_code", "unknown"))
            retryable = any(x in code.lower() for x in ("rate", "limit", "timeout", "429", "temporar"))
            return self._fail(
                job,
                checkpoint,
                AppError(
                    code="FETCH_RATE_LIMITED" if retryable else "FETCH_RECORD_ERROR",
                    message="Bright Data returned a per-record error.",
                    vendor_details=_redact(record, self.api_key),
                    retryable=retryable,
                ),
            )
        warnings = []
        if errors:
            warnings.append(f"Vendor response also contained {len(errors)} error record(s) for other inputs.")
        self._checkpoint(
            job, checkpoint, state="fetched", record=record, error=None, retrieval_method="brightdata_fetch"
        )
        return self._outcome(job, "ready", record=record, warnings=warnings)

    async def retrieve(
        self, request: ExtractionRequest, job: dict[str, Any], checkpoint: Checkpoint, wait_seconds: float
    ) -> RetrievalOutcome:
        deadline = self.clock() + max(0.0, wait_seconds)
        if job.get("record") is not None:
            return self._outcome(job, "ready", record=job["record"])
        if job.get("state") == "failed":
            error = (
                AppError.model_validate(job["error"])
                if job.get("error")
                else AppError(code="FETCH_FAILED", message="Retrieval previously failed.")
            )
            return self._outcome(job, "failed", error=error)
        if job.get("snapshot_id"):
            return await self._poll(request, job, checkpoint, deadline)
        if job.get("state") in {"submitting", "pending", "reconciliation_required"}:
            return self._ambiguous(job, checkpoint, "No durable snapshot ID after a previous submission.")
        if not self.api_key:
            return self._fail(
                job,
                checkpoint,
                AppError(code="CREDENTIALS_MISSING", message="BRIGHTDATA_API_KEY is required."),
            )
        retry_delay = max(0.0, float(job.get("retry_not_before") or 0) - time.time())
        if retry_delay:
            if retry_delay >= deadline - self.clock():
                return self._outcome(job, "pending")
            await self.sleep(retry_delay)
            self._checkpoint(job, checkpoint, retry_not_before=None)
        if wait_seconds <= 0:
            return self._outcome(job, "pending")
        result = await self._submit([request], job, checkpoint, deadline, endpoint="scrape")
        if isinstance(result, RetrievalOutcome):
            return result
        if isinstance(result, list):
            return self._record(result, request, job, checkpoint)
        return await self._poll(request, job, checkpoint, deadline)

    async def trigger_batch(
        self,
        requests: list[ExtractionRequest],
        job: dict[str, Any],
        checkpoint: Checkpoint,
        wait_seconds: float = 180.0,
    ) -> RetrievalOutcome:
        """Durably submit a batch. Completion can be downloaded/mapped by the caller.

        Each batch must have its own job record; never reuse a single-URL job.
        Repeating this operation with that record returns the existing snapshot.
        """
        if not requests or len({r.url for r in requests}) != len(requests):
            raise ValueError("A batch requires nonempty distinct input URLs")
        if job.get("snapshot_id"):
            return self._outcome(job, "pending")
        if job.get("state") in {"submitting", "pending", "reconciliation_required"}:
            return self._ambiguous(job, checkpoint, "Batch submission has no durable snapshot ID.")
        if job.get("state") == "failed":
            return self._outcome(job, "failed", error=AppError.model_validate(job["error"]))
        if not self.api_key:
            return self._fail(
                job,
                checkpoint,
                AppError(code="CREDENTIALS_MISSING", message="BRIGHTDATA_API_KEY is required."),
            )
        if wait_seconds <= 0:
            return self._outcome(job, "pending")
        result = await self._submit(
            requests, job, checkpoint, self.clock() + wait_seconds, endpoint="trigger"
        )
        if isinstance(result, RetrievalOutcome):
            return result
        if isinstance(result, list):
            return self._fail(
                job,
                checkpoint,
                AppError(
                    code="FETCH_INVALID_RESPONSE", message="Batch trigger did not return a snapshot ID."
                ),
            )
        return self._outcome(job, "pending")

    async def _submit(
        self,
        requests: list[ExtractionRequest],
        job: dict[str, Any],
        checkpoint: Checkpoint,
        deadline: float,
        endpoint: str,
    ) -> RetrievalOutcome | list | None:
        for attempt in range(self.max_attempts):
            remaining = deadline - self.clock()
            retry_delay = max(0.0, float(job.get("retry_not_before") or 0) - time.time())
            if retry_delay:
                if retry_delay >= remaining:
                    return self._outcome(job, "pending")
                await self.sleep(retry_delay)
                self._checkpoint(job, checkpoint, retry_not_before=None)
                remaining = deadline - self.clock()
            if remaining <= 0:
                # An explicit 429 rejected the previous attempt. No vendor job exists.
                self._checkpoint(job, checkpoint, state="new")
                return self._outcome(job, "pending")
            self._checkpoint(job, checkpoint, state="submitting", attempts=int(job.get("attempts", 0)) + 1)
            try:
                response = await bounded_request(
                    self.client,
                    "POST",
                    f"{self.base_url}/datasets/v3/{endpoint}",
                    max_response_bytes=self.max_response_bytes,
                    params={"dataset_id": self.dataset_id, "format": "json", "include_errors": "true"},
                    headers=self._headers(),
                    json=[self._input(item) for item in requests],
                    timeout=httpx.Timeout(min(75.0, remaining), connect=min(10.0, remaining)),
                )
            except ScraperError as exc:
                return self._fail(job, checkpoint, exc.error)
            except httpx.TransportError as exc:
                return self._ambiguous(job, checkpoint, type(exc).__name__)
            if response.status_code == 429:
                self._checkpoint(job, checkpoint, state="new")
                if attempt + 1 == self.max_attempts:
                    return self._fail(
                        job,
                        checkpoint,
                        AppError(
                            code="FETCH_RATE_LIMITED",
                            message="Submission rejected by rate limit; retry limit exhausted.",
                            vendor_details=self._details(response),
                            retryable=True,
                        ),
                    )
                delay = _retry_after(response, min(2**attempt + random.random(), self.max_poll_interval))
                self._checkpoint(job, checkpoint, retry_not_before=time.time() + delay)
                if delay >= deadline - self.clock():
                    return self._outcome(
                        job,
                        "pending",
                        error=AppError(
                            code="FETCH_RATE_LIMITED",
                            message="Submission rejected; Retry-After exceeds waiting deadline.",
                            retryable=True,
                            vendor_details=self._details(response),
                        ),
                    )
                await self.sleep(delay)
                self._checkpoint(job, checkpoint, retry_not_before=None)
                continue
            if response.status_code >= 500 or response.status_code == 408:
                return self._ambiguous(
                    job, checkpoint, f"HTTP {response.status_code}", self._details(response)
                )
            if response.status_code not in {200, 202}:
                return self._fail(
                    job,
                    checkpoint,
                    AppError(
                        code="FETCH_HTTP_ERROR",
                        message="Fetch submission was rejected.",
                        vendor_details=self._details(response),
                    ),
                )
            payload = _body(response)
            if (
                isinstance(payload, dict)
                and isinstance(payload.get("snapshot_id"), str)
                and payload["snapshot_id"]
            ):
                self._checkpoint(
                    job, checkpoint, state="pending", snapshot_id=payload["snapshot_id"], error=None
                )
                return None
            if response.status_code == 202:
                return self._ambiguous(
                    job, checkpoint, "HTTP 202 without snapshot_id", self._details(response)
                )
            if not isinstance(payload, list):
                return self._ambiguous(
                    job,
                    checkpoint,
                    "Successful submission without records or snapshot_id",
                    self._details(response),
                )
            return payload
        raise AssertionError("unreachable")

    async def _safe_get(
        self,
        path: str,
        job: dict[str, Any],
        checkpoint: Checkpoint,
        deadline: float,
        *,
        params: dict[str, str] | None = None,
    ) -> httpx.Response | AppError | None:
        for attempt in range(self.max_attempts):
            remaining = deadline - self.clock()
            if remaining <= 0:
                return None
            # Retry-After is a vendor cooldown, not merely a delay in this
            # invocation. Keep it across waiting deadlines and process restarts.
            cooldown = max(0.0, float(job.get("poll_retry_not_before") or 0) - time.time())
            if cooldown:
                if cooldown >= remaining:
                    return None
                await self.sleep(cooldown)
                remaining = deadline - self.clock()
                if remaining <= 0:
                    return None
            if job.get("poll_retry_not_before") is not None:
                self._checkpoint(job, checkpoint, poll_retry_not_before=None)
            self._checkpoint(job, checkpoint, attempts=int(job.get("attempts", 0)) + 1)
            try:
                response = await bounded_request(
                    self.client,
                    "GET",
                    f"{self.base_url}{path}",
                    max_response_bytes=self.max_response_bytes,
                    params=params,
                    headers=self._headers(),
                    timeout=httpx.Timeout(min(75.0, remaining), connect=min(10.0, remaining)),
                )
            except ScraperError as exc:
                return exc.error
            except httpx.TransportError:
                if attempt + 1 == self.max_attempts:
                    return AppError(
                        code="FETCH_RETRY_EXHAUSTED",
                        message="Safe polling transport retries exhausted.",
                        retryable=True,
                    )
                delay = min(2**attempt + random.random(), self.max_poll_interval)
            else:
                if response.status_code not in RETRYABLE_HTTP:
                    if response.status_code != 200:
                        return AppError(
                            code="FETCH_HTTP_ERROR",
                            message="Polling or download was rejected.",
                            vendor_details=self._details(response),
                        )
                    return response
                delay = _retry_after(response, min(2**attempt + random.random(), self.max_poll_interval))
                self._checkpoint(job, checkpoint, poll_retry_not_before=time.time() + delay)
                if attempt + 1 == self.max_attempts:
                    return AppError(
                        code="FETCH_RETRY_EXHAUSTED",
                        message="Safe polling HTTP retries exhausted.",
                        vendor_details=self._details(response),
                        retryable=True,
                    )
            if delay >= deadline - self.clock():
                return None
            await self.sleep(delay)
            if job.get("poll_retry_not_before") is not None:
                self._checkpoint(job, checkpoint, poll_retry_not_before=None)
        raise AssertionError("unreachable")

    async def _poll(
        self, request: ExtractionRequest, job: dict[str, Any], checkpoint: Checkpoint, deadline: float
    ) -> RetrievalOutcome:
        snapshot_id = str(job["snapshot_id"])
        escaped_id = quote(snapshot_id, safe="")
        delay = self.poll_interval
        while self.clock() < deadline:
            progress = await self._safe_get(f"/datasets/v3/progress/{escaped_id}", job, checkpoint, deadline)
            if progress is None:
                return self._outcome(job, "pending")
            if isinstance(progress, AppError):
                if progress.retryable:
                    return self._outcome(job, "pending", error=progress)
                return self._fail(job, checkpoint, progress)
            payload = _body(progress)
            status = payload.get("status") if isinstance(payload, dict) else None
            if status in {"failed", "canceled", "cancelled"}:
                return self._fail(
                    job,
                    checkpoint,
                    AppError(
                        code="FETCH_JOB_FAILED",
                        message="Bright Data job reached a terminal failure state.",
                        vendor_details=_redact(payload, self.api_key),
                    ),
                )
            if status == "ready":
                response = await self._safe_get(
                    f"/datasets/v3/snapshot/{escaped_id}",
                    job,
                    checkpoint,
                    deadline,
                    params={"format": "json"},
                )
                if response is None:
                    return self._outcome(job, "pending")
                if isinstance(response, AppError):
                    if response.retryable:
                        return self._outcome(job, "pending", error=response)
                    return self._fail(job, checkpoint, response)
                return self._record(_body(response), request, job, checkpoint)
            if status not in {"starting", "running"}:
                return self._outcome(
                    job,
                    "pending",
                    error=AppError(
                        code="FETCH_INVALID_RESPONSE",
                        message="Unrecognized vendor progress response; job retained for reconciliation.",
                        vendor_details={"response": _redact(payload, self.api_key)},
                        retryable=True,
                    ),
                )
            if delay >= deadline - self.clock():
                return self._outcome(job, "pending")
            await self.sleep(delay)
            delay = min(max(delay * 2, 0.001), self.max_poll_interval)
        return self._outcome(job, "pending")


class WebUnlockerRenderer:
    """Experimental raw renderer. It cannot attest to the observed redirect URL."""

    supports_observed_redirects = False

    def __init__(
        self,
        api_key: str,
        zone: str,
        *,
        client: httpx.AsyncClient | None = None,
        base_url: str = "https://api.brightdata.com",
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ):
        if not zone:
            raise ValueError("A configured Web Unlocker zone is required")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        self.api_key = api_key
        self.zone = zone
        self.client = client or httpx.AsyncClient(follow_redirects=False)
        self._owns_client = client is None
        self.base_url = base_url.rstrip("/")
        self.max_response_bytes = max_response_bytes

    async def close(self) -> None:
        if self._owns_client:
            await self.client.aclose()

    async def render(self, request: ExtractionRequest) -> RetrievalOutcome:
        payload = {
            "zone": self.zone,
            "url": request.url,
            "format": "raw",
            "data_format": "markdown",
            "render": "true",
        }
        if request.country:
            payload["country"] = request.country
        try:
            response = await bounded_request(
                self.client,
                "POST",
                f"{self.base_url}/request",
                max_response_bytes=self.max_response_bytes,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=httpx.Timeout(75.0, connect=10.0),
            )
        except ScraperError as exc:
            return RetrievalOutcome(
                status="failed", attempts=1, retrieval_method="brightdata_web_unlocker", error=exc.error
            )
        except httpx.TransportError as exc:
            return RetrievalOutcome(
                status="failed",
                attempts=1,
                retrieval_method="brightdata_web_unlocker",
                error=AppError(
                    code="RENDER_OUTCOME_UNKNOWN",
                    message="Rendering transport failed; no automatic retry.",
                    vendor_details={"reason": type(exc).__name__},
                ),
            )
        if response.status_code != 200:
            return RetrievalOutcome(
                status="failed",
                attempts=1,
                retrieval_method="brightdata_web_unlocker",
                error=AppError(
                    code="RENDER_FAILED",
                    message="Rendering request was rejected.",
                    vendor_details={
                        "http_status": response.status_code,
                        "response": _redact(_body(response), self.api_key),
                    },
                ),
            )
        return RetrievalOutcome(
            status="ready",
            record={
                "input": {"url": request.url},
                "url": request.url,
                "observed_final_url": False,
                "markdown": response.text,
            },
            attempts=1,
            retrieval_method="brightdata_web_unlocker",
            warnings=[
                "Observed final URL is unavailable in raw rendering output; the application cannot attest to redirects."
            ],
        )
