"""CLI for extraction, durable resume, offline replay and capability checks."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import ValidationError

from .config import Settings
from .contracts import RetrievalOutcome, ScraperError
from .extraction.huggingface import HuggingFaceExtractor, compile_provider_schema, inference_input_token_bound
from .jobs import JobStore, digest
from .policy import URLPolicy
from .registry import get_schema, list_schemas
from .schemas import (
    DEFAULT_SCHEMA_ID,
    SCHEMA_ID,
    AppError,
    BrowserAction,
    CustomField,
    ExtractionRequest,
    ExtractionResult,
    Provenance,
)
from .storage import atomic_json

EXIT_CODES = {"success": 0, "partial": 2, "pending": 3, "failed": 1}


def failure_result(
    code: str,
    message: str,
    *,
    url: str = "",
    job_id: str = "",
    provenance: Provenance | None = None,
    vendor_details: dict | None = None,
) -> ExtractionResult:
    return ExtractionResult(
        job_id=job_id,
        status="failed",
        provenance=provenance or Provenance(requested_url=url),
        error=AppError(code=code, message=message, vendor_details=vendor_details),
    )


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(prog="ai-scraper", description=__doc__)
    app.add_argument("--state-dir", type=Path, help="SQLite jobs, immutable snapshots and result directory")
    commands = app.add_subparsers(dest="command", required=True)
    for name in (
        "extract",
        "resume",
        "replay",
        "schema",
        "schemas",
        "probe",
        "prune",
        "evaluate",
        "benchmark",
        "crawl",
        "resume-crawl",
        "crawl-status",
    ):
        command = commands.add_parser(name)
        command.add_argument("--state-dir", type=Path, default=argparse.SUPPRESS)
        command.add_argument("--output", type=Path, help="Atomically export JSON in addition to stdout")
        if name in {"extract", "resume", "replay", "probe", "crawl", "resume-crawl"}:
            command.add_argument(
                "--source-bound-evidence",
                action=argparse.BooleanOptionalAction,
                default=None,
                help="Constrain local citations to locations present in the saved source",
            )
        if name in {"extract", "resume", "replay", "probe", "crawl", "resume-crawl"}:
            browser = command.add_mutually_exclusive_group()
            browser.add_argument("--browser", action="store_true", help="Retrieve with anonymous Chromium")
            browser.add_argument(
                "--render-fallback", action="store_true", help="Try direct HTTP before JavaScript rendering"
            )
        if name in {"extract", "crawl"}:
            command.add_argument(
                "--browser-actions", type=Path, help="JSON list of up to 10 caller-authored browser actions"
            )
        if name in {"extract", "replay"}:
            if name == "extract":
                command.add_argument("url")
            else:
                command.add_argument("record", type=Path)
                command.add_argument("--url", required=True)
                command.add_argument(
                    "--local-inference",
                    action="store_true",
                    help="Fill semantic gaps using the configured loopback model",
                )
            command.add_argument(
                "--schema", help="Registered schema or alias; default page_v1 (custom_v1 with --fields)"
            )
            command.add_argument("--country")
            command.add_argument("--locale", choices=["en_US", "en_GB", "de_DE", "fr_FR"])
        elif name == "resume":
            command.add_argument("job_id")
            command.add_argument(
                "--snapshot-id", help="Attach a reconciled vendor snapshot without resubmitting"
            )
        elif name == "schema":
            command.add_argument("schema_id", nargs="?", help="Preset or custom schema; default page_v1")
            command.add_argument("--schema", help="Alias for the optional schema_id argument")
            command.add_argument("--provider-compatible", action="store_true")
        elif name == "probe":
            command.description = (
                "Probe the selected model; default actual nested product capability contract"
            )
            command.add_argument("--schema", default=SCHEMA_ID)
        elif name in {"evaluate", "benchmark"}:
            command.add_argument("manifest", type=Path)
        elif name == "crawl":
            command.add_argument("seeds", nargs="+", help="One to ten public seed URLs")
            command.add_argument("--schema", help="Extraction schema; default page_v1")
            command.add_argument("--country")
            command.add_argument("--locale", choices=["en_US", "en_GB", "de_DE", "fr_FR"])
            command.add_argument("--max-pages", type=int, default=20)
            command.add_argument("--max-depth", type=int, default=2)
            command.add_argument("--max-frontier", type=int, default=1000)
            command.add_argument("--max-links-per-page", type=int, default=500)
            command.add_argument("--delay-seconds", type=float, default=1.0)
            command.add_argument("--wait-seconds", type=float, default=180.0)
            command.add_argument("--scope-domain", action="append", default=[], help="Exact additional host")
        elif name in {"resume-crawl", "crawl-status"}:
            command.add_argument("crawl_id")
            if name == "resume-crawl":
                command.add_argument("--wait-seconds", type=float)
        if name in {"extract", "replay", "schema", "probe", "crawl"}:
            command.add_argument("--fields", help="Custom fields: headline,year:integer,tags:string_list")
            command.add_argument(
                "--fields-file", type=Path, help="JSON array of typed custom field descriptors (max 32)"
            )
            command.add_argument(
                "--instruction",
                "--ask",
                help="Task guidance; extracted values still need literal source evidence",
            )
    return app


def _browser_actions(args: argparse.Namespace) -> list[BrowserAction]:
    path = getattr(args, "browser_actions", None)
    if path is None:
        return []
    with path.open("rb") as source:
        contents = source.read(65_537)
    if len(contents) > 65_536:
        raise ScraperError("INVALID_REQUEST", "Browser action file exceeds the byte limit")
    value = json.loads(contents)
    if not isinstance(value, list) or len(value) > 10:
        raise ScraperError("INVALID_REQUEST", "Browser actions must be a JSON list of at most 10 actions")
    return [BrowserAction.model_validate(item) for item in value]


def build_request(
    url: str,
    schema_id: str = DEFAULT_SCHEMA_ID,
    fields: str | list[CustomField | dict] | None = None,
    instruction: str | None = None,
    **kwargs,
) -> ExtractionRequest:
    """Build the same bounded, validated contract for CLI and MCP callers."""
    if instruction is not None:
        if not isinstance(instruction, str):
            raise ScraperError("INVALID_REQUEST", "Instruction must be a string")
        instruction = instruction.strip() or None
    if isinstance(fields, str):
        parts = [part.strip() for part in fields.split(",")]
        if not fields.strip() or any(not part for part in parts):
            raise ScraperError("INVALID_REQUEST", "Custom field shorthand must contain named fields")
        fields = []
        for part in parts:
            name, separator, kind = part.partition(":")
            fields.append({"name": name.strip(), "type": kind.strip() if separator else "string"})
    if fields is not None and not isinstance(fields, list):
        raise ScraperError("INVALID_REQUEST", "Custom fields must be a descriptor list")
    general = {DEFAULT_SCHEMA_ID, "page", "auto", "general", "document", "custom", "custom_v1"}
    if fields:
        if schema_id not in general:
            raise ScraperError("INVALID_REQUEST", "Custom fields require the general or custom schema")
        schema_id = "custom_v1"
    elif instruction and schema_id in general:
        schema_id = "custom_v1"
        fields = [{"name": "matches", "type": "string_list", "description": instruction[:500]}]
    request = ExtractionRequest(
        url=url, schema_id=schema_id, fields=fields or [], instruction=instruction, **kwargs
    )
    spec = get_schema(request)
    return request.model_copy(update={"schema_id": spec.schema_id})


def _request(args: argparse.Namespace, url: str | None = None) -> ExtractionRequest:
    fields = getattr(args, "fields", None)
    path = getattr(args, "fields_file", None)
    if fields is not None and path is not None:
        raise ScraperError("INVALID_REQUEST", "Use either --fields or --fields-file")
    if path is not None:
        if path.stat().st_size > 65_536:
            raise ScraperError("INVALID_REQUEST", "Custom field definitions exceed the byte limit")
        fields = json.loads(path.read_text())
        if not isinstance(fields, list):
            raise ScraperError("INVALID_REQUEST", "Custom field definitions must be a JSON descriptor list")
    schema_id = getattr(args, "schema_id", None)
    flag = getattr(args, "schema", None)
    if schema_id is not None and flag is not None and schema_id != flag:
        raise ScraperError("INVALID_REQUEST", "Specify one schema selection")
    request = build_request(
        url or getattr(args, "url", "https://example.com/contract"),
        schema_id=schema_id or flag or DEFAULT_SCHEMA_ID,
        country=getattr(args, "country", None),
        locale=getattr(args, "locale", None),
        fields=fields,
        instruction=getattr(args, "instruction", None),
        browser_actions=_browser_actions(args),
    )
    spec = get_schema(request)
    # Persist only validated metadata. Error handling must not reread a custom
    # descriptor file or rebuild a request with potentially different contents.
    args._failure_provenance = Provenance(
        requested_url=getattr(args, "url", ""),
        schema_version=spec.schema_id,
        prompt_version=spec.prompt_version,
    )
    return request


def _settings(args: argparse.Namespace) -> Settings:
    settings = Settings.from_env()
    if args.command in {"crawl", "resume-crawl", "crawl-status"}:
        if not os.environ.get("SCRAPER_RETRIEVAL_BACKEND"):
            settings = settings.model_copy(update={"retrieval_backend": "direct"})
        if args.command == "crawl" and not os.environ.get("SCRAPER_ALLOWED_DOMAINS"):
            settings = settings.model_copy(
                update={
                    "allowed_domains": sorted(
                        {urlsplit(seed).hostname or "" for seed in args.seeds} | set(args.scope_domain)
                    )
                }
            )
    if getattr(args, "browser", False):
        settings = settings.model_copy(update={"retrieval_backend": "browser"})
    elif getattr(args, "render_fallback", False):
        settings = settings.model_copy(update={"retrieval_backend": "auto"})
    if getattr(args, "source_bound_evidence", None) is not None:
        settings = settings.model_copy(update={"local_source_bound_evidence": args.source_bound_evidence})
    if args.state_dir is not None:
        settings = settings.model_copy(update={"state_dir": args.state_dir})
    return settings


class _ReplayRetriever:
    retrieval_method = "fixture_replay"

    def __init__(self, record: dict):
        self.record = record

    @property
    def cache_options(self) -> dict:
        return {"record_hash": digest(self.record)}

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        return RetrievalOutcome(
            status="ready", record=self.record, retrieval_method="fixture_replay", attempts=0
        )

    async def close(self):
        pass


def _load_record(path: Path, url: str, *, max_bytes: int = 5_000_000) -> dict:
    if path.stat().st_size > max_bytes:
        raise ScraperError("CONTENT_TOO_LARGE", "Replay file exceeds its configured byte limit")
    body = json.loads(path.read_text())
    if isinstance(body, list):
        matching = [
            item
            for item in body
            if isinstance(item, dict)
            and (item.get("input", {}).get("url") if isinstance(item.get("input"), dict) else item.get("url"))
            == url
        ]
        if len(matching) != 1:
            raise ScraperError("INVALID_FIXTURE", "Record array must contain exactly one matching input URL")
        body = matching[0]
    if not isinstance(body, dict):
        raise ScraperError("INVALID_FIXTURE", "Replay requires a saved vendor record object")
    if body.get("error_code") or body.get("error"):
        raise ScraperError("FETCH_RECORD_ERROR", "Saved vendor record contains a retrieval error")
    return body


async def _probe(settings: Settings, request: ExtractionRequest | None = None) -> dict:
    spec = get_schema(request) if request else None

    def selected(inference):
        if spec and (spec.schema_id != SCHEMA_ID or request.instruction):
            return inference.for_request(request, spec)
        return inference

    if settings.inference_backend == "local":
        from .pipeline import create_pipeline

        pipeline = create_pipeline(settings)
        try:
            report = await selected(pipeline.inference).probe()
            report["usage"]["cost_estimate_usd"] = 0.0
            report["usage"]["pricing_basis"] = (
                "Local inference: no vendor charge; excludes hardware and energy"
            )
            return report
        finally:
            await pipeline.close()
    if not settings.hf_token or not settings.hf_provider or not settings.hf_model:
        raise ScraperError("MISSING_CREDENTIALS", "Probe requires HF_TOKEN, HF_PROVIDER and HF_MODEL")
    prices = (settings.hf_input_million_usd, settings.hf_output_million_usd)
    if settings.max_spend_usd is None or any(v is None for v in prices):
        raise ScraperError("BUDGET_UNCONFIGURED", "Probe requires a spending cap and both HF token prices")
    if not all(math.isfinite(v) and v >= 0 for v in (*prices, settings.max_spend_usd)):
        raise ScraperError("BUDGET_UNCONFIGURED", "Probe prices and cap must be finite and nonnegative")
    inference = HuggingFaceExtractor(
        settings.hf_provider,
        settings.hf_model,
        settings.hf_token.get_secret_value(),
        timeout=settings.hf_timeout,
        max_tokens=settings.max_output_tokens,
        max_source_chars=settings.max_source_chars,
    )
    try:
        adapter = selected(inference)
        input_bound = inference_input_token_bound(
            settings.max_source_chars,
            getattr(adapter, "schema", None),
            system_instruction=getattr(adapter, "system_instruction", None),
        )
        amount = (input_bound * prices[0] + settings.max_output_tokens * prices[1]) / 1_000_000
        basis = "Conservative HF probe reservation using configured input/output prices per million tokens"
        store = JobStore(settings.state_dir)
        store.reserve(f"probe:{uuid.uuid4().hex}", amount, settings.max_spend_usd, basis)
        report = await adapter.probe()
        usage = report["usage"]
        usage["cost_estimate_usd"] = (
            usage.get("input_tokens", 0) * prices[0] + usage.get("output_tokens", 0) * prices[1]
        ) / 1_000_000
        usage["pricing_basis"] = "Configured HF prices per million input and output tokens"
        report["reserved_usd"] = amount
        report["reserved_spend_total_usd"] = store.reserved_spend()
        return report
    finally:
        await inference.close()


async def _run(args: argparse.Namespace) -> dict:
    request = _request(args) if args.command in {"extract", "replay", "schema", "probe"} else None
    if args.command == "schemas":
        return {"default_schema_id": DEFAULT_SCHEMA_ID, "schemas": list_schemas()}
    if args.command == "schema":
        spec = get_schema(request)
        if args.provider_compatible:
            return compile_provider_schema(spec.candidate_model.model_json_schema())
        return {
            "schema_id": spec.schema_id,
            "required_fields": list(spec.required_fields),
            "critical_fields": list(spec.critical_fields),
            "business_schema": spec.data_model.model_json_schema(),
            "extraction_schema": spec.candidate_model.model_json_schema(),
            "fingerprint": spec.fingerprint,
        }
    settings = _settings(args)
    if args.command in {"crawl", "resume-crawl", "crawl-status"}:
        from .crawl import Crawler, CrawlRequest

        if args.command == "crawl":
            page_request = _request(args, args.seeds[0])
            if page_request.browser_actions and settings.retrieval_backend not in {"browser", "auto"}:
                raise ScraperError(
                    "INVALID_REQUEST", "Browser actions require --browser or --render-fallback"
                )
            crawl_request = CrawlRequest(
                seeds=args.seeds,
                schema_id=page_request.schema_id,
                fields=page_request.fields,
                instruction=page_request.instruction,
                country=page_request.country,
                locale=page_request.locale,
                browser_actions=page_request.browser_actions,
                max_pages=args.max_pages,
                max_depth=args.max_depth,
                max_frontier=args.max_frontier,
                max_links_per_page=args.max_links_per_page,
                delay_seconds=args.delay_seconds,
                wait_seconds=args.wait_seconds,
                scope_domains=(
                    sorted({urlsplit(seed).hostname or "" for seed in args.seeds} | set(args.scope_domain))
                    if args.scope_domain
                    else []
                ),
            )
        crawler = Crawler(settings)
        try:
            if args.command == "crawl":
                result = await crawler.start(crawl_request)
            elif args.command == "resume-crawl":
                result = await crawler.resume(args.crawl_id, wait_seconds=args.wait_seconds)
            else:
                result = crawler.status(args.crawl_id)
            return result.model_dump(mode="json")
        finally:
            await crawler.close()
    if args.command == "probe":
        return await _probe(settings, request)
    if args.command == "prune":
        return {"pruned_jobs": JobStore(settings.state_dir).prune(settings.retention_days)}
    if args.command == "evaluate":
        from .evaluation import evaluate_manifest

        return evaluate_manifest(args.manifest)
    if args.command == "benchmark":
        from .benchmark import benchmark_manifest

        return await benchmark_manifest(args.manifest, settings)

    from .pipeline import Pipeline, create_pipeline

    if (
        request is not None
        and request.browser_actions
        and settings.retrieval_backend not in {"browser", "auto"}
    ):
        raise ScraperError("INVALID_REQUEST", "Browser actions require --browser or --render-fallback")

    if args.command == "replay":
        record = _load_record(args.record, request.url, max_bytes=settings.max_response_bytes)
        allowed = settings.allowed_domains or sorted(
            {
                urlsplit(request.url).hostname or "",
                urlsplit(record.get("final_url") or request.url).hostname or "",
            }
        )
        inference = None
        if args.local_inference:
            if settings.inference_backend != "local":
                raise ScraperError(
                    "INVALID_REQUEST", "Replay inference requires SCRAPER_INFERENCE_BACKEND=local"
                )
            configured = create_pipeline(settings)
            inference = configured.inference
            if configured.retriever:
                await configured.retriever.close()
        pipeline = Pipeline(
            settings,
            retriever=_ReplayRetriever(record),
            inference=inference,
            policy=URLPolicy(allowed, resolve_dns=False),
            live=False,
        )
    else:
        pipeline = create_pipeline(settings)
    try:
        if args.command == "resume":
            result = await pipeline.resume(args.job_id, snapshot_id=args.snapshot_id)
        else:
            result = await pipeline.extract_url(request)
        return result.model_dump(mode="json")
    finally:
        await pipeline.close()


def _failure_payload(args: argparse.Namespace, code: str, message: str, *, vendor_details=None) -> dict:
    if args.command in {"crawl", "resume-crawl", "crawl-status"}:
        return {
            "crawl_id": getattr(args, "crawl_id", ""),
            "status": "failed",
            "counts": {},
            "entries": [],
            "warnings": [],
            "error": AppError(code=code, message=message).model_dump(mode="json"),
        }
    return failure_result(
        code,
        message,
        url=getattr(args, "url", ""),
        job_id=getattr(args, "job_id", ""),
        provenance=getattr(args, "_failure_provenance", None),
        vendor_details=vendor_details,
    ).model_dump(mode="json")


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = asyncio.run(_run(args))
    except ScraperError as exc:
        # Known errors already contain stable, sanitized application messages.
        payload = _failure_payload(
            args,
            exc.error.code,
            exc.error.message,
            vendor_details=(
                exc.error.vendor_details if exc.error.code == "CAPABILITY_PROBE_FAILED" else None
            ),
        )
        print(f"{exc.error.code}: {exc.error.message}", file=sys.stderr)
    except ValidationError:
        payload = _failure_payload(args, "INVALID_REQUEST", "Invalid request or configuration")
        print("INVALID_REQUEST: Invalid request or configuration", file=sys.stderr)
    except (OSError, ValueError, TypeError):
        payload = _failure_payload(args, "LOCAL_ERROR", "Could not read input, configuration or saved state")
        print("LOCAL_ERROR: Could not read input, configuration or saved state", file=sys.stderr)
    except KeyboardInterrupt:
        print("Interrupted; saved pending jobs can be resumed", file=sys.stderr)
        return 130
    if args.output:
        try:
            atomic_json(args.output, payload)
        except (OSError, ValueError, TypeError):
            print("EXPORT_FAILED: Could not atomically write JSON export", file=sys.stderr)
            return 1
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    if payload.get("status") == "not_run" or payload.get("gates", {}).get("passed") is False:
        return 1
    return EXIT_CODES.get(payload.get("status"), 0)


if __name__ == "__main__":
    sys.exit(main())
