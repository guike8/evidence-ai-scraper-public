"""Loopback oMLX inference with the same strict candidate parser and schema probe.

oMLX's explicit structured_outputs path rejects unavailable grammar rather than
silently downgrading response_format to prompt-only extraction. Both are sent:
https://github.com/jundot/omlx/blob/main/omlx/server.py
The local server has no extraction tools and is never allowed to redirect.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import httpx

from ..contracts import InferenceResponse, ScraperError
from ..retrieval.brightdata import bounded_request
from ..schemas import ExtractionCandidate, SourceSnapshot
from .evidence_schema import (
    SOURCE_BOUND_EVIDENCE_VERSION,
    EvidenceCatalogLimitError,
    bind_source_evidence_references,
    constrain_evidence_references,
    source_evidence_catalog,
)
from .huggingface import HuggingFaceExtractor, schema_fingerprint
from .source_quote_schema import (
    SOURCE_QUOTE_SCHEMA_VERSION,
    SourceQuoteSchemaError,
    bind_source_quote_evidence,
)

if TYPE_CHECKING:
    from ..registry import SchemaSpec
    from ..schemas import ExtractionRequest


def _local_schema_constraints(schema: dict[str, Any], candidate_model) -> dict[str, Any]:
    """Preserve supported local constraints from the trusted product model."""
    rich = candidate_model.model_json_schema()
    product = rich.get("$defs", {}).get("ProductV1", {}).get("properties", {})
    compiled_product = schema.get("$defs", {}).get("ProductV1", {}).get("properties", {})
    if "price" in product and "price" in compiled_product:
        compiled_product["price"] = copy.deepcopy(product["price"])
    return schema


def validate_local_endpoint(base_url: str) -> str:
    try:
        parsed = urlsplit(base_url)
        valid = (
            parsed.scheme in {"http", "https"}
            and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
            and parsed.username is None
            and parsed.password is None
            and "?" not in base_url
            and "#" not in base_url
            and not any(ord(character) < 33 for character in base_url)
            and (parsed.port is None or 0 < parsed.port <= 65535)
        )
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("Local inference requires a loopback HTTP(S) URL without credentials or query")
    return base_url.rstrip("/")


def _constant(_: str) -> None:
    raise ValueError("Nonfinite JSON is forbidden")


def _object(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON keys are forbidden")
        result[key] = value
    return result


class _LocalHTTPClient:
    def __init__(self, base_url: str, api_key: str | None, timeout: float, client=None):
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.client = client or httpx.AsyncClient(trust_env=False, follow_redirects=False)
        self.owns_client = client is None
        self.last_error: ScraperError | None = None

    async def chat_completion(self, **kwargs) -> dict:
        self.last_error = None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        payload = {
            **kwargs,
            # oMLX may expose configured MCP tools even if none were provided by
            # this request. Explicitly disable that merge at the server boundary.
            "tool_choice": "none",
            "chat_template_kwargs": {"enable_thinking": False},
            "structured_outputs": {"json_schema": kwargs["response_format"]["json_schema"]["schema"]},
        }
        try:
            response = await bounded_request(
                self.client,
                "POST",
                self.base_url + "/chat/completions",
                max_response_bytes=2_000_000,
                headers=headers,
                json=payload,
                timeout=httpx.Timeout(self.timeout, connect=min(10.0, self.timeout)),
            )
            if response.status_code != 200:
                raise ScraperError(
                    "LOCAL_INFERENCE_HTTP_ERROR",
                    "Local inference request was rejected",
                    vendor_details={"status_code": response.status_code},
                    retryable=response.status_code in {408, 429, 502, 503, 504},
                )
            warning = response.headers.get("Warning", "").lower()
            if any(marker in warning for marker in ("not enforced", "not schema-enforced", "best-effort")):
                raise ScraperError(
                    "LOCAL_SCHEMA_UNENFORCED", "Local server could not enforce the requested JSON schema"
                )
            try:
                envelope = json.loads(response.content, parse_constant=_constant, object_pairs_hook=_object)
            except (ValueError, RecursionError):
                raise ScraperError(
                    "INVALID_EXTRACTION", "Local inference response is not strict JSON"
                ) from None
            if not isinstance(envelope, dict):
                raise ScraperError("INVALID_EXTRACTION", "Local inference response must be an object")
            return envelope
        except httpx.TimeoutException:
            self.last_error = ScraperError(
                "INFERENCE_TIMEOUT", "Local inference exceeded its finite timeout", retryable=True
            )
            raise self.last_error from None
        except ScraperError as exc:
            self.last_error = exc
            raise

    async def close(self):
        if self.owns_client:
            await self.client.aclose()


class LocalExtractor(HuggingFaceExtractor):
    """One schema-constrained local completion; orchestration owns repair limits.

    ``client`` is an injectable httpx.AsyncClient, kept caller-owned. No HF token
    or cloud account is required. Local electricity/hardware costs are separate
    from the zero external inference fee.
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:8000/v1",
        api_key: str | None = None,
        *,
        timeout: float = 60.0,
        max_tokens: int = 4096,
        max_source_chars: int = 48000,
        client: httpx.AsyncClient | None = None,
        source_bound_evidence: bool = False,
        source_quote_evidence: bool = False,
    ):
        if not isinstance(model, str) or not model.strip() or model == "auto":
            raise ValueError("An explicit local model identifier is required")
        if type(source_bound_evidence) is not bool:
            raise ValueError("source_bound_evidence must be a boolean")
        if type(source_quote_evidence) is not bool:
            raise ValueError("source_quote_evidence must be a boolean")
        if source_bound_evidence and source_quote_evidence:
            raise ValueError("Local citation modes are mutually exclusive")
        self.source_bound_evidence = source_bound_evidence
        self.source_quote_evidence = source_quote_evidence
        self.citation_policy_version = (
            SOURCE_QUOTE_SCHEMA_VERSION
            if source_quote_evidence
            else SOURCE_BOUND_EVIDENCE_VERSION
            if source_bound_evidence
            else None
        )
        self.last_evidence_catalog: dict[str, Any] | None = None
        self.base_url = validate_local_endpoint(base_url)
        self._local_client = _LocalHTTPClient(self.base_url, api_key, timeout, client)
        super().__init__(
            "local",
            model,
            timeout=timeout,
            max_tokens=max_tokens,
            max_source_chars=max_source_chars,
            client=self._local_client,
        )
        # xgrammar supports decimal regex constraints that the conservative HF
        # provider subset omits. Derive this constraint from the trusted model.
        self.schema = constrain_evidence_references(
            _local_schema_constraints(self.schema, ExtractionCandidate)
        )
        self.schema_fingerprint = schema_fingerprint(self.schema)
        self._evidence_schema_template = copy.deepcopy(self.schema)

    def for_request(self, request: ExtractionRequest, spec: SchemaSpec) -> LocalExtractor:
        scoped = super().for_request(request, spec)
        scoped.schema = constrain_evidence_references(
            _local_schema_constraints(scoped.schema, spec.candidate_model)
        )
        scoped.schema_fingerprint = schema_fingerprint(scoped.schema)
        scoped._evidence_schema_template = copy.deepcopy(scoped.schema)
        scoped.last_evidence_catalog = None
        # The transport may be shared safely, but the shim's last_error belongs
        # to one request. An error for one schema must not taint another call.
        scoped._local_client = copy.copy(self._local_client)
        scoped._local_client.owns_client = False
        scoped._local_client.last_error = None
        scoped.client = scoped._local_client
        return scoped

    def prepare_source(self, snapshot: SourceSnapshot) -> dict[str, Any] | None:
        if self.source_bound_evidence or self.source_quote_evidence:
            # Product inference otherwise skips this optional preparation hook.
            # Check catalog limits before orchestration reserves a model call.
            self._source_content(snapshot)
            return copy.deepcopy(self.last_source_view)
        return super().prepare_source(snapshot)

    def _source_content(self, snapshot: SourceSnapshot) -> str:
        self.last_evidence_catalog = None
        if self.source_bound_evidence or self.source_quote_evidence:
            self.schema = copy.deepcopy(self._evidence_schema_template)
            self.schema_fingerprint = schema_fingerprint(self.schema)
        source = super()._source_content(snapshot)
        if self.source_quote_evidence:
            try:
                bound = bind_source_quote_evidence(
                    self._evidence_schema_template, json.loads(source, parse_constant=_constant)
                )
            except (SourceQuoteSchemaError, ValueError, TypeError, RecursionError):
                raise ScraperError(
                    "CITATION_CATALOG_UNAVAILABLE", "Cannot build the bounded source quote catalog"
                ) from None
            self.schema = bound.schema
            self.schema_fingerprint = schema_fingerprint(self.schema)
            self.last_evidence_catalog = bound.metadata()
            return source
        if not self.source_bound_evidence:
            return source
        # Always derive from the immutable request template. A previous source
        # must never narrow or widen the next snapshot's grammar.
        try:
            catalog = source_evidence_catalog(json.loads(source, parse_constant=_constant))
            schema = bind_source_evidence_references(self._evidence_schema_template, catalog)
        except EvidenceCatalogLimitError:
            raise ScraperError(
                "CONTENT_TOO_LARGE", "Source evidence catalog exceeds its bounded grammar limits"
            ) from None
        except (ValueError, TypeError, RecursionError):
            raise ScraperError("INVALID_EXTRACTION", "Source evidence catalog is invalid") from None
        self.schema = schema
        self.schema_fingerprint = schema_fingerprint(schema)
        self.last_evidence_catalog = catalog.metadata()
        return source

    async def extract(self, snapshot: SourceSnapshot, feedback: str | None = None) -> InferenceResponse:
        self._local_client.last_error = None
        try:
            response = await super().extract(snapshot, feedback)
        except ScraperError as exc:
            error = self._local_client.last_error or exc
            raise ScraperError(
                error.error.code,
                error.error.message.replace("HF ", "Local "),
                vendor_details=error.error.vendor_details,
                retryable=error.error.retryable,
            ) from None
        response.usage.update(
            cost_estimate_usd=0.0, pricing_basis="Local model; excludes electricity and hardware"
        )
        return response

    async def probe(self) -> dict[str, Any]:
        try:
            return await super().probe()
        except ScraperError as exc:
            raise ScraperError(
                exc.error.code,
                exc.error.message.replace("HF ", "Local "),
                vendor_details=exc.error.vendor_details,
                retryable=exc.error.retryable,
            ) from None

    async def close(self) -> None:
        await self._local_client.close()
