"""URL -> durable source -> extraction -> local validation -> atomic result."""

from __future__ import annotations

import copy
import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from .config import Settings
from .contracts import Inference, RetrievalOutcome, Retriever, ScraperError
from .extraction.deterministic import deterministic_extract
from .jobs import JobStore, digest
from .normalize import detect_content_problem, normalize_record
from .policy import URLPolicy
from .registry import get_schema
from .schemas import (
    PROMPT_VERSION,
    SCHEMA_ID,
    AppError,
    ExtractionCandidate,
    ExtractionRequest,
    ExtractionResult,
    Provenance,
    SourceSnapshot,
    SourceViewSummary,
    Usage,
)
from .storage import redact
from .validate import validate_candidate


def merge_semantic_gaps(payload: dict, deterministic: ExtractionCandidate) -> dict:
    """Retain established facts when a semantic call only supplies missing fields."""
    try:
        semantic = ExtractionCandidate.model_validate(payload).model_dump()
    except (ValidationError, ValueError, TypeError):
        return payload  # the unchanged local validator reports the actual errors
    base = deterministic.model_dump()

    def fill(value, trusted):
        if value is None or value == []:
            return copy.deepcopy(trusted)
        if isinstance(value, dict) and isinstance(trusted, dict):
            return {key: fill(item, trusted.get(key)) for key, item in value.items()}
        return value

    semantic["data"] = fill(semantic["data"], base["data"])
    semantic["evidence"].extend(base["evidence"])
    semantic["warnings"] = list(dict.fromkeys(semantic["warnings"] + base["warnings"]))
    return semantic


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        *,
        retriever: Retriever | None = None,
        inference: Inference | None = None,
        policy: URLPolicy | None = None,
        live: bool = True,
        renderer=None,
    ):
        self.settings = settings
        self.store = JobStore(settings.state_dir, max_snapshot_bytes=settings.max_snapshot_bytes)
        self.retriever = retriever
        self.inference = inference
        self.policy = policy or URLPolicy(settings.allowed_domains)
        self.live = live
        self.renderer = renderer
        self.secrets = tuple(
            x.get_secret_value()
            for x in (settings.brightdata_api_key, settings.hf_token, settings.local_api_key)
            if x
        )

    async def close(self):
        for adapter in (self.retriever, self.inference, self.renderer):
            if adapter is not None and hasattr(adapter, "close"):
                await adapter.close()

    def _provenance(self, request, snapshot=None, snapshot_path=None, snapshot_id=None):
        try:
            spec = get_schema(request)
        except ScraperError:
            spec = None
        return Provenance(
            requested_url=request.url,
            country=request.country,
            final_url=snapshot.final_url if snapshot else None,
            retrieved_at=snapshot.retrieved_at if snapshot else None,
            source_hash=snapshot.source_hash if snapshot else None,
            snapshot_id=snapshot.snapshot_id if snapshot else snapshot_id,
            snapshot_path=str(snapshot_path) if snapshot_path else None,
            retrieval_method=snapshot.retrieval_method if snapshot else None,
            schema_version=spec.schema_id if spec else request.schema_id,
            prompt_version=spec.prompt_version if spec else PROMPT_VERSION,
            provider=getattr(self.inference, "provider", None),
            model=getattr(self.inference, "model", None),
        )

    def _reserve_retrieval(self, job, *, rendering=False):
        if not self.live or (not rendering and self.settings.retrieval_backend == "direct"):
            return
        price = self.settings.render_record_usd if rendering else self.settings.retrieval_record_usd
        if price is None:
            raise ScraperError(
                "PRICING_UNCONFIGURED", "Configure account-specific retrieval pricing before live calls"
            )
        self.store.reserve(
            f"{job['id']}:render" if rendering else f"{job['id']}:fetch",
            price,
            self.settings.max_spend_usd,
            "configured account retrieval upper bound",
        )

    def _reserve_inference(self, job_id: str, attempt: int, extraction_key: str, inference=None):
        inference = inference or self.inference
        if not self.live or getattr(self.inference, "provider", None) == "local":
            return
        inp, out = self.settings.hf_input_million_usd, self.settings.hf_output_million_usd
        if inp is None or out is None:
            raise ScraperError(
                "PRICING_UNCONFIGURED", "Configure HF input/output token pricing before live inference"
            )
        from .extraction.huggingface import inference_input_token_bound

        input_bound = inference_input_token_bound(
            self.settings.max_source_chars,
            schema=getattr(inference, "schema", None),
            system_instruction=getattr(inference, "system_instruction", None),
        )
        maximum = (input_bound * inp + self.settings.max_output_tokens * out) / 1e6
        self.store.reserve(
            f"{job_id}:hf:{extraction_key}:{attempt}",
            maximum,
            self.settings.max_spend_usd,
            "UTF-8 byte/token upper bound incl schema, feedback and framing; configured account rates",
        )

    def _cost(self, usage: Usage, method: str | None):
        if self.settings.retrieval_backend == "direct" and getattr(self.inference, "provider", None) in {
            None,
            "local",
        }:
            usage.cost_estimate_usd = 0.0
            usage.pricing_basis = "Direct public HTTP and local inference: no vendor charge; excludes hardware, energy and connectivity"
            return
        if not self.live:
            if getattr(self.inference, "provider", None) == "local":
                usage.cost_estimate_usd = 0.0
                usage.pricing_basis = (
                    "Local inference and fixture retrieval: no vendor charge; excludes hardware and energy"
                )
            return
        price = self.settings.retrieval_record_usd
        if price is not None:
            amount = price if usage.retrieval_attempts > 0 and method is not None else 0.0
            if (
                method in {"brightdata_unlocker", "brightdata_web_unlocker"}
                and self.settings.render_record_usd is not None
            ):
                amount += self.settings.render_record_usd
            if (
                self.settings.hf_input_million_usd is not None
                and self.settings.hf_output_million_usd is not None
                and getattr(self.inference, "provider", None) != "local"
            ):
                amount += (
                    usage.input_tokens * self.settings.hf_input_million_usd
                    + usage.output_tokens * self.settings.hf_output_million_usd
                ) / 1e6
            usage.cost_estimate_usd = amount
            usage.pricing_basis = (
                "configured account rates; excludes storage/operations; reservations are conservative"
            )

    async def extract_url(self, request: ExtractionRequest) -> ExtractionResult:
        try:
            get_schema(request)
        except ScraperError as exc:
            return ExtractionResult(
                job_id=uuid.uuid4().hex,
                status="failed",
                provenance=self._provenance(request),
                error=exc.error,
            )
        try:
            await self.policy.check(request.url)
        except ScraperError as exc:
            return ExtractionResult(
                job_id=uuid.uuid4().hex,
                status="failed",
                provenance=self._provenance(request),
                error=exc.error,
            )
        # Rendering reuses the fetched source and must not split pending Fetch
        # identities when only its optional configuration changes.
        default_method = "direct_http" if self.settings.retrieval_backend == "direct" else "brightdata_fetch"
        method = getattr(self.retriever, "retrieval_method", default_method)
        options = {**getattr(self.retriever, "cache_options", {}), "method": method}
        job = self.store.get_or_create(request, options)
        return await self._run(job)

    async def resume(self, job_id: str, snapshot_id: str | None = None) -> ExtractionResult:
        job = self.store.get(job_id)
        if snapshot_id and job.get("retrieval_parent_id"):
            parent_id = job["retrieval_parent_id"]
            parent = self.store.get(parent_id)
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", snapshot_id):
                raise ScraperError("INVALID_REQUEST", "Invalid vendor snapshot ID")
            if parent["state"] not in {"submitting", "reconciliation_required", "pending"}:
                raise ScraperError("INVALID_REQUEST", "Only pending submissions can be reconciled")
            if parent.get("snapshot_id") and parent["snapshot_id"] != snapshot_id:
                raise ScraperError("INVALID_REQUEST", "Cannot replace a known vendor job ID")
            self.store.update(parent_id, snapshot_id=snapshot_id, state="pending", error=None)
            return await self._run(job)
        if snapshot_id:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", snapshot_id):
                raise ScraperError("INVALID_REQUEST", "Invalid vendor snapshot ID")
            if job["state"] not in {"submitting", "reconciliation_required", "pending"}:
                raise ScraperError("INVALID_REQUEST", "Only pending submissions can be reconciled")
            if job.get("snapshot_id") and job["snapshot_id"] != snapshot_id:
                raise ScraperError("INVALID_REQUEST", "Cannot replace a known vendor job ID")
            job = self.store.update(job_id, snapshot_id=snapshot_id, state="pending", error=None)
        return await self._run(job)

    async def _run(self, job: dict, *, retrieval_only: bool = False) -> ExtractionResult:
        start = time.monotonic()
        job_id = job["id"]
        request = ExtractionRequest.model_validate(job["request"])
        try:
            spec = get_schema(request)
        except ScraperError as exc:
            return ExtractionResult(
                job_id=job_id, status="failed", provenance=self._provenance(request), error=exc.error
            )
        inference = self.inference
        if inference is not None and hasattr(inference, "for_request"):
            inference = inference.for_request(request, spec)
        elif inference is not None and hasattr(inference, "for_schema"):
            # Keep schema-scoped third-party adapters compatible while the
            # request-scoped boundary also carries custom task metadata.
            inference = inference.for_schema(spec, request.instruction)
        if job.get("retrieval_parent_id") and not job.get("record"):
            parent = self.store.get(job["retrieval_parent_id"])
            parent_result = await self._run(parent, retrieval_only=True)
            parent = self.store.get(parent["id"])
            if parent.get("record") and parent["state"] in {"fetched", "complete"}:
                job = self.store.adopt_retrieval(job_id, parent)
            else:
                result = ExtractionResult(
                    job_id=job_id,
                    status=parent_result.status if parent_result.status == "failed" else "pending",
                    provenance=self._provenance(request, snapshot_id=parent.get("snapshot_id")),
                    error=parent_result.error,
                    warnings=["Waiting for the shared retrieval job"] + parent_result.warnings,
                )
                if result.status == "failed":
                    self.store.update(
                        job_id, state="failed", error=result.error.model_dump() if result.error else None
                    )
                    self.store.save_result(result)
                return result
        snapshot = None
        assessment = None
        warnings = []
        used_inference = False
        source_view_summary = None
        owner = uuid.uuid4().hex
        ttl = max(900, self.settings.wait_seconds + 2 * self.settings.hf_timeout + 180)
        if not self.store.acquire(job_id, owner, ttl):
            return ExtractionResult(
                job_id=job_id,
                status="pending",
                provenance=self._provenance(request),
                warnings=["Job or target-host concurrency limit is active"],
            )
        # The caller can load a job before another process completes submission
        # and releases its lease. Refresh only after ownership is established:
        # using that stale 'new' body could otherwise POST the same URL again.
        job = self.store.get(job_id)
        usage = Usage(**job.get("usage", {}))
        usage.cache_hit = False
        snapshot_path = job.get("snapshot_path")

        def checkpoint(patch):
            nonlocal job
            patch = redact(patch, self.secrets)
            job = self.store.update(job_id, **patch)
            self.store.renew(job_id, owner, ttl)

        def finish(result):
            result.provenance.inference_source = source_view_summary
            if not used_inference:
                result.provenance.provider = None
                result.provenance.model = None
            result.usage.latency_ms += (time.monotonic() - start) * 1000
            if result.status != "pending":
                self._cost(result.usage, result.provenance.retrieval_method)
                self.store.save_result(result)
                patch = {"usage": result.usage.model_dump()}
                if snapshot is not None:
                    patch["state"] = "complete"
                self.store.update(job_id, **patch)
            return result

        def prepare_inference_source():
            nonlocal source_view_summary
            if (
                inference is None
                or spec.schema_id == SCHEMA_ID
                or not getattr(inference, "source_view_version", None)
                or not hasattr(inference, "prepare_source")
            ):
                return
            try:
                inference.prepare_source(snapshot)
            finally:
                descriptor = getattr(inference, "last_source_view", None)
                if descriptor is not None:
                    required = {
                        "source_hash",
                        "schema_id",
                        "selection_version",
                        "selection_hash",
                        "view_hash",
                        "scope",
                        "coverage",
                        "original_chars",
                        "view_chars",
                        "included_section_ids",
                        "excluded_section_ids",
                        "duplicate_section_ids",
                        "out_of_scope_section_ids",
                    }
                    if not isinstance(descriptor, dict) or not required <= descriptor.keys():
                        raise ScraperError("SOURCE_VIEW_INVALID", "Inference source descriptor is incomplete")
                    if any(
                        not isinstance(descriptor[key], list)
                        or any(not isinstance(item, str) for item in descriptor[key])
                        for key in (
                            "included_section_ids",
                            "excluded_section_ids",
                            "duplicate_section_ids",
                            "out_of_scope_section_ids",
                        )
                    ):
                        raise ScraperError("SOURCE_VIEW_INVALID", "Inference source section IDs are invalid")
                    if (
                        descriptor.get("source_hash") != snapshot.source_hash
                        or descriptor.get("schema_id") != spec.schema_id
                        or descriptor.get("selection_version") != inference.source_view_version
                    ):
                        raise ScraperError("SOURCE_VIEW_INVALID", "Inference source identity is invalid")
                    section_ids = {section.id for section in snapshot.sections}
                    included = descriptor["included_section_ids"]
                    excluded = descriptor["excluded_section_ids"]
                    if (
                        len(set(included)) != len(included)
                        or len(set(excluded)) != len(excluded)
                        or set(included) & set(excluded)
                        or set(included) | set(excluded) != section_ids
                        or not set(descriptor["duplicate_section_ids"]) <= set(excluded)
                        or not set(descriptor["out_of_scope_section_ids"]) <= set(excluded)
                    ):
                        raise ScraperError("SOURCE_VIEW_INVALID", "Inference source sections are invalid")
                    path = self.store.save_source_view(job_id, descriptor)
                    source_view_summary = SourceViewSummary(
                        selection_version=descriptor["selection_version"],
                        selection_hash=descriptor["selection_hash"],
                        view_hash=descriptor["view_hash"],
                        scope=descriptor["scope"],
                        coverage=descriptor["coverage"],
                        original_chars=descriptor["original_chars"],
                        view_chars=descriptor["view_chars"],
                        included_sections=len(included),
                        excluded_sections=len(excluded),
                        duplicate_sections=len(descriptor["duplicate_section_ids"]),
                        out_of_scope_sections=len(descriptor["out_of_scope_section_ids"]),
                        descriptor_path=str(path),
                    )
                    views = dict(job.get("inference_sources", {}))
                    views[extraction_key] = source_view_summary.model_dump(mode="json")
                    checkpoint({"inference_sources": views})

        try:
            await self.policy.check(request.url)
            if (
                self.live
                and job.get("render_record")
                and job["render_record"].get("observed_final_url") is not True
            ):
                raise ScraperError(
                    "RENDER_EGRESS_UNVERIFIABLE",
                    "Persisted rendered source lacks an observed redirect destination",
                )
            if (
                job.get("render_record")
                and job.get("render_method")
                and (not snapshot_path or job.get("render_snapshot_path") != snapshot_path)
            ):
                snapshot = normalize_record(
                    job["render_record"],
                    request,
                    snapshot_id=None,
                    retrieval_method=job["render_method"],
                    max_record_bytes=self.settings.max_response_bytes,
                )
                render_time = job.get("render_retrieved_at") or job.get("snapshot_retrieved_at")
                if render_time:
                    snapshot.retrieved_at = datetime.fromisoformat(render_time)
                await self.policy.check_record(snapshot.raw_record, request.url)
                snapshot_path = self.store.save_snapshot(job_id, snapshot)
                checkpoint(
                    {
                        "render_snapshot_path": str(snapshot_path),
                        "snapshot_id": None,
                        "snapshot_retrieved_at": snapshot.retrieved_at.isoformat(),
                    }
                )
            if snapshot_path:
                if not Path(snapshot_path).resolve().is_relative_to(self.store.root):
                    raise ScraperError("SNAPSHOT_CORRUPT", "Snapshot path is outside configured storage")
                if Path(snapshot_path).stat().st_size > self.settings.max_snapshot_bytes:
                    raise ScraperError(
                        "CONTENT_TOO_LARGE", "Saved snapshot exceeds its configured byte limit"
                    )
                snapshot = SourceSnapshot.model_validate_json(Path(snapshot_path).read_text())
                if (
                    self.live
                    and snapshot.retrieval_method == "brightdata_web_unlocker"
                    and snapshot.raw_record.get("observed_final_url") is not True
                ):
                    raise ScraperError(
                        "RENDER_EGRESS_UNVERIFIABLE",
                        "Saved raw Unlocker source cannot be accepted without observed redirect metadata",
                    )
                await self.policy.check_record(snapshot.raw_record, request.url)
                # A snapshot is immutable but files can be modified externally; verify
                # its hash using the same normalizer before trusting cached evidence.
                rebuilt = normalize_record(
                    snapshot.raw_record,
                    request,
                    snapshot_id=snapshot.snapshot_id,
                    retrieval_method=snapshot.retrieval_method,
                    max_record_bytes=self.settings.max_response_bytes,
                )
                if (
                    rebuilt.source_hash != snapshot.source_hash
                    or rebuilt.sections != snapshot.sections
                    or rebuilt.ld_json != snapshot.ld_json
                    or rebuilt.excluded_sections != snapshot.excluded_sections
                ):
                    raise ScraperError("SNAPSHOT_CORRUPT", "Saved source hash or normalized evidence changed")
                if (
                    snapshot.requested_url != request.url
                    or snapshot.final_url != rebuilt.final_url
                    or snapshot.country != request.country
                    or snapshot.snapshot_id != job.get("snapshot_id")
                    or snapshot.retrieval_method
                    != (
                        job.get("render_method")
                        if job.get("render_record")
                        else job.get("retrieval_method", "brightdata_fetch")
                    )
                    or (
                        job.get("snapshot_retrieved_at")
                        and snapshot.retrieved_at.isoformat() != job["snapshot_retrieved_at"]
                    )
                ):
                    raise ScraperError("SNAPSHOT_CORRUPT", "Saved source provenance changed")
                usage.cache_hit = True
            elif job.get("state") == "failed" and not job.get("record"):
                error = AppError.model_validate(
                    job.get("error") or {"code": "FETCH_FAILED", "message": "Terminal retrieval failure"}
                )
                return finish(
                    ExtractionResult(
                        job_id=job_id,
                        status="failed",
                        provenance=self._provenance(request),
                        error=error,
                        usage=usage,
                    )
                )
            else:
                if job.get("record") and job["state"] in {"fetched", "complete"}:
                    outcome = RetrievalOutcome(
                        status="ready",
                        record=job["record"],
                        snapshot_id=job.get("snapshot_id"),
                        retrieval_method=job.get("retrieval_method", "brightdata_fetch"),
                        attempts=job.get("attempts", 0),
                    )
                    usage.cache_hit = True
                else:
                    if self.retriever is None:
                        raise ScraperError(
                            "CREDENTIALS_UNAVAILABLE",
                            "Configure Bright Data credentials or replay a saved record",
                        )
                    if job["state"] == "new":
                        self._reserve_retrieval(job)
                    outcome = await self.retriever.retrieve(
                        request, job, checkpoint, self.settings.wait_seconds
                    )
                usage.retrieval_attempts = max(outcome.attempts, job.get("attempts", 0))
                warnings.extend(outcome.warnings)
                if outcome.status != "ready":
                    result = ExtractionResult(
                        job_id=job_id,
                        status="pending" if outcome.status == "pending" else "failed",
                        provenance=self._provenance(request, snapshot_id=outcome.snapshot_id),
                        usage=usage,
                        warnings=warnings,
                        error=outcome.error,
                    )
                    checkpoint({"usage": usage.model_dump()})
                    return finish(result)
                clean_record = redact(outcome.record or {}, self.secrets)
                await self.policy.check_record(clean_record, request.url)
                snapshot = normalize_record(
                    clean_record,
                    request,
                    snapshot_id=outcome.snapshot_id,
                    retrieval_method=outcome.retrieval_method,
                    max_record_bytes=self.settings.max_response_bytes,
                )
                checkpoint({"record": clean_record, "state": "fetched"})
                if job.get("snapshot_retrieved_at"):
                    snapshot.retrieved_at = datetime.fromisoformat(job["snapshot_retrieved_at"])
                snapshot_path = self.store.save_snapshot(job_id, snapshot)
                checkpoint(
                    {
                        "snapshot_retrieved_at": snapshot.retrieved_at.isoformat(),
                        "retrieval_method": snapshot.retrieval_method,
                        "snapshot_id": snapshot.snapshot_id,
                    }
                )
            if self.live and snapshot.retrieval_method == "fixture_replay":
                raise ScraperError(
                    "SNAPSHOT_MODE_MISMATCH",
                    "Live extraction cannot accept a replay fixture as retrieved content",
                )
            if retrieval_only:
                checkpoint({"usage": usage.model_dump()})
                return ExtractionResult(
                    job_id=job_id,
                    status="pending",
                    provenance=self._provenance(request, snapshot, snapshot_path),
                    usage=usage,
                )
            problem = detect_content_problem(snapshot)
            if problem:
                if self.live and (
                    (self.renderer is None and self.settings.unlocker_zone)
                    or (
                        self.renderer is not None
                        and not getattr(self.renderer, "supports_observed_redirects", True)
                    )
                ):
                    raise ScraperError(
                        "RENDER_EGRESS_UNVERIFIABLE",
                        "Raw Unlocker rendering is disabled because its redirect destination cannot be verified",
                    )
                if self.renderer is not None and not job.get("render_attempted"):
                    # Raw Unlocker output has no observed final URL. Opt-in rendering
                    # therefore requires the deployment's vendor-side egress controls.
                    self._reserve_retrieval(job, rendering=True)
                    checkpoint({"render_attempted": True})
                    rendered = await self.renderer.render(request)
                    usage.retrieval_attempts += rendered.attempts
                    warnings.extend(rendered.warnings)
                    if rendered.status == "ready":
                        rendered.record = redact(rendered.record or {}, self.secrets)
                        if self.live and rendered.record.get("observed_final_url") is not True:
                            raise ScraperError(
                                "RENDER_EGRESS_UNVERIFIABLE",
                                "Renderer did not supply observed redirect metadata",
                            )
                        render_time = datetime.now(UTC).isoformat()
                        checkpoint(
                            {
                                "render_record": rendered.record,
                                "render_method": rendered.retrieval_method,
                                "render_retrieved_at": render_time,
                            }
                        )
                        await self.policy.check_record(rendered.record, request.url)
                        snapshot = normalize_record(
                            rendered.record,
                            request,
                            snapshot_id=rendered.snapshot_id,
                            retrieval_method=rendered.retrieval_method,
                            max_record_bytes=self.settings.max_response_bytes,
                        )
                        snapshot.retrieved_at = datetime.fromisoformat(render_time)
                        snapshot_path = self.store.save_snapshot(job_id, snapshot)
                        checkpoint(
                            {
                                "render_record": rendered.record,
                                "render_method": rendered.retrieval_method,
                                "render_snapshot_path": str(snapshot_path),
                                "snapshot_id": snapshot.snapshot_id,
                                "snapshot_retrieved_at": snapshot.retrieved_at.isoformat(),
                            }
                        )
                        problem = detect_content_problem(snapshot)
                    else:
                        raise ScraperError("CONTENT_RENDER_FAILED", "Configured rendering fallback failed")
                if problem:
                    raise ScraperError(
                        problem, "Retrieved source needs rendering, interaction, or challenge handling"
                    )

            if spec.schema_id == SCHEMA_ID:
                candidate = deterministic_extract(snapshot, request.locale)
                assess = lambda value: validate_candidate(value, snapshot, request.locale)
                merge = merge_semantic_gaps
            else:
                from .extraction.generic import deterministic_extract as general_extract
                from .extraction.generic import merge_semantic_gaps as general_merge
                from .validate_generic import validate_candidate as general_validate

                candidate = general_extract(snapshot, spec)
                assess = lambda value: general_validate(value, snapshot, spec)
                merge = lambda value, baseline: general_merge(value, baseline, spec)
            deterministic_candidate = candidate
            assessment = assess(candidate)
            cache_contract = {
                "hash": snapshot.source_hash,
                "schema": spec.schema_id,
                "identity": snapshot.final_url,
                "prompt": spec.prompt_version,
                "provider": getattr(self.inference, "provider", None),
                "model": getattr(self.inference, "model", None),
                "locale": request.locale,
                "source_limit": self.settings.max_source_chars,
                "output_limit": self.settings.max_output_tokens,
            }
            if spec.schema_id != SCHEMA_ID or request.instruction or request.fields:
                cache_contract.update(schema_fingerprint=spec.fingerprint, instruction=request.instruction)
            if spec.schema_id != SCHEMA_ID and getattr(inference, "source_view_version", None):
                cache_contract["source_view_version"] = inference.source_view_version
            extraction_key = digest(cache_contract)
            cached = self.store.cache_get(extraction_key)
            if cached:
                assessment = assess(cached.get("candidate", cached))
                used_inference = cached.get("origin") == "inference"
                if used_inference:
                    prepare_inference_source()
                usage.cache_hit = True
            elif (assessment.status != "success" or request.instruction) and inference is not None:
                feedback = None
                last_error = None
                # The count is persisted BEFORE each billable call so a crash cannot
                # reset the repair limit. Repairs only see the same saved snapshot.
                prepare_inference_source()
                used_inference = True
                counts = dict(job.get("inference_counts", {}))
                attempts = counts.get(extraction_key, 0)
                if attempts >= 2:
                    raise ScraperError(
                        "INFERENCE_BUDGET_EXHAUSTED",
                        "Initial inference and one repair already attempted for this snapshot/configuration",
                    )
                while attempts < 2:
                    attempts += 1
                    self._reserve_inference(job_id, attempts, extraction_key, inference)
                    counts[extraction_key] = attempts
                    checkpoint({"inference_attempts": attempts, "inference_counts": counts})
                    self.store.renew(job_id, owner, ttl)
                    usage.inference_attempts += 1
                    try:
                        response = await inference.extract(snapshot, feedback)
                        metrics = response.usage
                        payload = merge(response.payload, deterministic_candidate)
                        assessment = assess(payload)
                        diagnostics = dict(job.get("inference_diagnostics", {}))
                        history = list(diagnostics.get(extraction_key, []))
                        history.append(
                            {
                                "attempt": attempts,
                                "candidate": response.payload,
                                "validation_errors": assessment.errors,
                            }
                        )
                        diagnostics[extraction_key] = history[-2:]
                        checkpoint({"inference_diagnostics": diagnostics})
                        candidate = payload
                        last_error = None
                    except ScraperError as exc:
                        metrics = getattr(inference, "last_usage", {})
                        last_error = exc.error
                        assessment.errors = [exc.error.code + ": " + exc.error.message]
                    usage.input_tokens += int(metrics.get("input_tokens", 0))
                    usage.output_tokens += int(metrics.get("output_tokens", 0))
                    checkpoint({"usage": usage.model_dump()})
                    if last_error is None and not assessment.errors:
                        break
                    if last_error and last_error.code not in {
                        "INVALID_EXTRACTION",
                        "TRUNCATED_OUTPUT",
                        "EMPTY_OUTPUT",
                    }:
                        break
                    feedback = (
                        "Repair against the SAME source. Fix only these local validation errors:\n"
                        + "\n".join(assessment.errors)
                    )
                if last_error:
                    raise ScraperError(
                        last_error.code, last_error.message, vendor_details=last_error.vendor_details
                    )
                if not assessment.errors:
                    payload = candidate.model_dump() if hasattr(candidate, "model_dump") else candidate
                    # Never cache a failed HF payload as a successful deterministic fallback.
                    if last_error is None:
                        self.store.cache_put(extraction_key, {"candidate": payload, "origin": "inference"})
                        origins = dict(job.get("extraction_source", {}))
                        origins[extraction_key] = "inference"
                        checkpoint({"extraction_source": origins})
            elif inference is None and assessment.status != "success":
                warnings.append(
                    "semantic_extraction_unavailable: configure an explicitly probed HF provider/model"
                )

            if assessment.errors:
                warnings.extend(assessment.warnings)
                raise ScraperError(
                    "INVALID_EXTRACTION",
                    "Local extraction, evidence, or critical completeness checks failed",
                    vendor_details={"validation_errors": assessment.errors},
                )
            if not cached and assessment.status == "success" and hasattr(candidate, "model_dump"):
                self.store.cache_put(
                    extraction_key, {"candidate": candidate.model_dump(), "origin": "deterministic"}
                )
            result = ExtractionResult(
                job_id=job_id,
                status=assessment.status,
                data=assessment.data
                if spec.schema_id == SCHEMA_ID or isinstance(assessment.data, dict)
                else assessment.data.model_dump(mode="json"),
                missing_fields=assessment.missing_fields,
                warnings=list(dict.fromkeys(warnings + assessment.warnings)),
                evidence=assessment.evidence,
                provenance=self._provenance(request, snapshot, snapshot_path),
                usage=usage,
            )
            return finish(result)
        except ScraperError as exc:
            error = AppError.model_validate(redact(exc.error.model_dump(), self.secrets))
            return finish(
                ExtractionResult(
                    job_id=job_id,
                    status="failed",
                    error=error,
                    warnings=warnings,
                    missing_fields=assessment.missing_fields if assessment else [],
                    provenance=self._provenance(request, snapshot, snapshot_path, job.get("snapshot_id")),
                    usage=usage,
                )
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return finish(
                ExtractionResult(
                    job_id=job_id,
                    status="failed",
                    warnings=warnings,
                    usage=usage,
                    provenance=self._provenance(request, snapshot, snapshot_path),
                    error=AppError(
                        code="LOCAL_STATE_ERROR",
                        message="Saved state or source could not be read or validated",
                    ),
                )
            )
        finally:
            self.store.release(job_id, owner)


def create_pipeline(settings: Settings | None = None) -> Pipeline:
    settings = settings or Settings.from_env()
    retriever = inference = renderer = None
    if settings.retrieval_backend == "direct":
        from .retrieval.direct import DirectHTTPRetriever

        retriever = DirectHTTPRetriever(
            URLPolicy(settings.allowed_domains), max_response_bytes=settings.max_response_bytes
        )
    elif settings.brightdata_api_key:
        from .retrieval.brightdata import BrightDataRetriever

        retriever = BrightDataRetriever(
            settings.brightdata_api_key.get_secret_value(), max_response_bytes=settings.max_response_bytes
        )
    if settings.inference_backend == "local":
        if not settings.local_model:
            raise ScraperError("LOCAL_MODEL_UNCONFIGURED", "Set an explicit LOCAL_MODEL for local inference")
        from .extraction.local import LocalExtractor

        inference = LocalExtractor(
            settings.local_model,
            base_url=settings.local_base_url,
            api_key=settings.local_api_key.get_secret_value() if settings.local_api_key else None,
            timeout=settings.hf_timeout,
            max_tokens=settings.max_output_tokens,
            max_source_chars=settings.max_source_chars,
        )
    elif settings.hf_provider and settings.hf_model and settings.hf_token:
        from .extraction.huggingface import HuggingFaceExtractor

        inference = HuggingFaceExtractor(
            settings.hf_provider,
            settings.hf_model,
            settings.hf_token.get_secret_value(),
            timeout=settings.hf_timeout,
            max_tokens=settings.max_output_tokens,
            max_source_chars=settings.max_source_chars,
        )
    return Pipeline(settings, retriever=retriever, inference=inference, renderer=renderer)


async def extract_url(request: ExtractionRequest, *, pipeline: Pipeline | None = None) -> ExtractionResult:
    """Single application operation; inject a pipeline to replace adapters."""
    owned = pipeline is None
    pipeline = pipeline or create_pipeline()
    try:
        return await pipeline.extract_url(request)
    finally:
        if owned:
            await pipeline.close()
