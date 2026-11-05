"""Credentialless, schema-aware snapshot comparisons with explicit receipts.

The runner uses the real application pipeline, a fixture retriever, fresh state
and no inference adapter. The scorer also accepts independently collected tool
receipts; it never calls another scraper or treats a reported quote as verified.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import platform
import re
import sys
import tempfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

import ai_scraper
from ai_scraper.cli import build_request
from ai_scraper.config import Settings
from ai_scraper.contracts import RetrievalOutcome, ScraperError
from ai_scraper.extraction.generic import populated_fields
from ai_scraper.normalize import detect_content_problem, normalize_record
from ai_scraper.pipeline import Pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.registry import get_schema
from ai_scraper.schemas import SCHEMA_ID
from ai_scraper.storage import atomic_json
from ai_scraper.validate import validate_candidate as validate_product
from ai_scraper.validate_generic import validate_candidate as validate_general

MANIFEST_FORMAT = "snapshot-comparison-v1"
RECEIPT_FORMAT = "snapshot-receipt-v1"
MAX_JSON_BYTES = 20_000_000
_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,79}")
_FIELD = re.compile(r"[a-z][a-z0-9_]*(?:\.(?:[a-z][a-z0-9_]*|0|[1-9][0-9]*))*")
_STATUSES = {"success", "partial", "pending", "failed"}
_USAGE_COUNTS = ("input_tokens", "output_tokens", "inference_attempts", "retrieval_attempts")


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _read(path: Path) -> dict:
    if path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Benchmark JSON exceeds its byte limit")
    value = json.loads(path.read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    if not isinstance(value, dict):
        raise TypeError("Benchmark JSON must be an object")
    return value


def _finite(value: Any, *, positive: bool = False) -> bool:
    return type(value) in {int, float} and math.isfinite(value) and (value > 0 if positive else value >= 0)


def case_context(case: dict):
    """Resolve a trusted registry contract and immutable common source identity."""
    request = build_request(case["url"], **case.get("request", {}))
    spec = get_schema(request)
    snapshot = normalize_record(case["record"], request, retrieval_method="fixture_replay")
    # Model/prompt/selector changes are legitimate candidates in a comparison.
    # Bind identity to the caller's task and business contract, not to one
    # implementation's prompt version or inference response wrapper.
    request_hash = _hash(
        {
            "request": request.model_dump(mode="json"),
            "business_schema": spec.data_model.model_json_schema(),
            "required": spec.required_fields,
            "critical": spec.critical_fields,
        }
    )
    return request, spec, snapshot, request_hash


def _label_matches_schema(spec, path: str, value: Any) -> bool:
    schema = spec.data_model.model_json_schema()
    node = schema

    def resolve(item):
        while "$ref" in item:
            item = schema["$defs"][item["$ref"].rsplit("/", 1)[-1]]
        alternatives = item.get("anyOf")
        if alternatives:
            item = next(option for option in alternatives if option.get("type") != "null")
            return resolve(item)
        return item

    try:
        for part in path.split("."):
            node = resolve(node)
            if part.isdigit():
                if node.get("type") != "array":
                    return False
                node = node["items"]
            else:
                node = node["properties"][part]
        node = resolve(node)
    except (KeyError, StopIteration):
        return False
    if value is None:
        return True  # explicit absence label; never a positive recall opportunity
    kind = node.get("type")
    return (
        (kind == "string" and type(value) is str)
        or (kind == "integer" and type(value) is int)
        or (kind == "number" and type(value) is float)
        or (kind == "boolean" and type(value) is bool)
    )


def validate_manifest(manifest: dict) -> list[dict]:
    if (
        manifest.get("format") != MANIFEST_FORMAT
        or manifest.get("labels_status") != "reviewed"
        or manifest.get("labels_complete") is not True
        or not isinstance(manifest.get("corpus_id"), str)
        or not manifest["corpus_id"]
        or not isinstance(manifest.get("provenance"), str)
        or not manifest["provenance"]
    ):
        raise ValueError("A versioned corpus with reviewed, complete labels and provenance is required")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 10_000:
        raise ValueError("Manifest requires a bounded nonempty case list")
    seen = set()
    for case in cases:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("id"), str)
            or not _ID.fullmatch(case["id"])
            or case["id"] in seen
            or not isinstance(case.get("url"), str)
            or not isinstance(case.get("record"), dict)
            or not isinstance(case.get("request", {}), dict)
            or "url" in case.get("request", {})
            or not isinstance(case.get("expected_fields"), dict)
            or case.get("expected_status") not in _STATUSES
        ):
            raise ValueError("Invalid or duplicate benchmark case")
        seen.add(case["id"])
        _, spec, _, _ = case_context(case)
        roots = spec.data_model.model_fields
        for path, value in case["expected_fields"].items():
            if (
                not isinstance(path, str)
                or not _FIELD.fullmatch(path)
                or path.split(".", 1)[0] not in roots
                or (value is not None and type(value) not in {str, int, float, bool})
                or (type(value) is float and not math.isfinite(value))
                or not _label_matches_schema(spec, path, value)
            ):
                raise ValueError("Expected fields must be known scalar leaf paths")
        required = case.get("required_paths", list(spec.required_fields))
        if not isinstance(required, list) or any(
            not isinstance(path, str) or not _FIELD.fullmatch(path) or path.split(".", 1)[0] not in roots
            for path in required
        ):
            raise ValueError("Invalid required benchmark paths")
        if case.get("expected_error_code") is not None and (
            not isinstance(case["expected_error_code"], str) or not _ID.fullmatch(case["expected_error_code"])
        ):
            raise ValueError("Invalid expected error code")
    return cases


def _metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise TypeError("Receipt metadata is required")
    for field in ("candidate", "version", "mode", "runtime", "cache_state", "track"):
        if not isinstance(metadata.get(field), str) or not metadata[field].strip():
            raise ValueError("Receipt candidate, version, mode, runtime, cache and track are required")
    if metadata["track"] != "identical_snapshot":
        raise ValueError("This scorer requires identical snapshots; URL retrieval needs a separate track")
    if metadata["cache_state"] not in {"cold", "warm", "disabled", "mixed", "unknown"}:
        raise ValueError("Unknown cache state")
    if not _finite(metadata.get("deadline_seconds"), positive=True):
        raise ValueError("A finite positive completion deadline is required")
    if not isinstance(metadata.get("hardware"), dict) or any(
        not isinstance(metadata["hardware"].get(field), str) or not metadata["hardware"][field]
        for field in ("system", "machine")
    ):
        raise ValueError("Measured hardware metadata is required")
    for field in ("model", "provider", "network_calls"):
        if field not in metadata:
            raise ValueError("Model, provider and network measurement must be explicit (null if unknown)")
    if any(
        metadata[field] is not None and not isinstance(metadata[field], str)
        for field in ("model", "provider")
    ):
        raise ValueError("Model and provider must be strings or null")
    calls = metadata["network_calls"]
    if calls is not None and (type(calls) is not int or calls < 0):
        raise ValueError("Network calls must be a measured count or null")


def _counts() -> dict[str, int]:
    return {"present": 0, "predicted": 0, "correct": 0}


def _metrics(counts: dict[str, int]) -> dict:
    present, predicted, correct = (counts[key] for key in ("present", "predicted", "correct"))
    precision = correct / predicted if predicted else None
    recall = correct / present if present else None
    return {
        **counts,
        "false_positive": predicted - correct,
        "missed": present - correct,
        "precision": precision,
        "recall": recall,
        "f1": 2 * correct / (predicted + present) if predicted + present else None,
    }


def _quantiles(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "samples": len(values),
        "p50_ms": ordered[math.ceil(0.50 * len(values)) - 1] if values else None,
        "p95_ms": ordered[math.ceil(0.95 * len(values)) - 1] if values else None,
    }


def _required(path: str, required: list[str]) -> bool:
    return any(path == prefix or path.startswith(prefix + ".") for prefix in required)


def score_receipt(manifest: dict, receipt: dict) -> dict:
    """Score every labeled leaf, preserving misses from absent/late/failed work.

    Complete expected_fields are reviewed labels for populated scalar facts.
    Null and empty-string labels are negative/structural controls, not recall
    opportunities. Unlabeled populated output is a label mismatch, reported
    separately from the production source-evidence acceptance gate.
    """
    cases = validate_manifest(manifest)
    if (
        receipt.get("format") != RECEIPT_FORMAT
        or receipt.get("corpus_id") != manifest["corpus_id"]
        or receipt.get("manifest_hash") != _hash(manifest)
        or not isinstance(receipt.get("results"), list)
    ):
        raise ValueError("Receipt does not match this corpus and its exact manifest hash")
    _metadata(receipt.get("metadata"))
    known_ids = {case["id"] for case in cases}
    outcomes = {}
    for outcome in receipt["results"]:
        if (
            not isinstance(outcome, dict)
            or not isinstance(outcome.get("case_id"), str)
            or outcome.get("case_id") not in known_ids
            or outcome["case_id"] in outcomes
            or outcome.get("status") not in _STATUSES
            or (outcome.get("data") is not None and not isinstance(outcome["data"], dict))
            or (outcome["status"] in {"failed", "pending"} and outcome.get("data") is not None)
            or (outcome.get("error") is not None and not isinstance(outcome["error"], dict))
        ):
            raise ValueError("Receipt has an unknown, duplicate or invalid outcome")
        elapsed = outcome.get("wall_latency_ms")
        if elapsed is not None and not _finite(elapsed):
            raise ValueError("Wall latency must be measured, finite and nonnegative, or null")
        components = outcome.get("component_latency_ms", {})
        if not isinstance(components, dict) or any(not _finite(value) for value in components.values()):
            raise ValueError("Invalid component latency measurements")
        usage = outcome.get("usage", {})
        if not isinstance(usage, dict) or any(
            usage.get(key) is not None and (type(usage[key]) is not int or usage[key] < 0)
            for key in _USAGE_COUNTS
        ):
            raise ValueError("Usage counts must be measured nonnegative integers or unknown")
        outcomes[outcome["case_id"]] = outcome
    groups = {key: _counts() for key in ("all", "required", "optional")}
    by_schema: dict[str, dict[str, int]] = {}
    by_domain: dict[str, dict[str, int]] = {}
    statuses: Counter = Counter()
    missing_receipts = []
    unsupported = 0
    verified_count = 0
    populated_count = 0
    evidence_passes = 0
    usable = 0
    known_unusable = 0
    unknown_retrieval = 0
    status_matches = 0
    wall = []
    component_values: dict[str, list[float]] = {}
    usage_totals = {key: {"measured_cases": 0, "total": 0} for key in _USAGE_COUNTS}
    unknown_completion = []
    details = []
    deadline_ms = receipt["metadata"]["deadline_seconds"] * 1000
    for case in cases:
        request, spec, snapshot, request_hash = case_context(case)
        outcome = outcomes.get(case["id"])
        if outcome is None:
            missing_receipts.append(case["id"])
            outcome = {"status": "failed", "data": None, "error": {"code": "MISSING_RECEIPT"}}
        elif (
            outcome.get("source_hash") != snapshot.source_hash or outcome.get("request_hash") != request_hash
        ):
            raise ValueError("Receipt source/request hashes differ from the reviewed fixture")
        elapsed = outcome.get("wall_latency_ms")
        late = elapsed is not None and elapsed > deadline_ms
        status = "failed" if late else outcome["status"]
        statuses[status] += 1
        expected_error = case.get("expected_error_code")
        actual_error = "COMPLETION_DEADLINE" if late else (outcome.get("error") or {}).get("code")
        status_matches += (
            case["id"] in outcomes
            and status == case["expected_status"]
            and (expected_error is None or actual_error == expected_error)
        )
        if elapsed is not None:
            wall.append(elapsed)
        else:
            unknown_completion.append(case["id"])
        for component, value in outcome.get("component_latency_ms", {}).items():
            component_values.setdefault(component, []).append(value)
        for key in _USAGE_COUNTS:
            count = outcome.get("usage", {}).get(key)
            if count is not None:
                usage_totals[key]["measured_cases"] += 1
                usage_totals[key]["total"] += count
        # bool False and integer zero are present facts, unlike empty/null data.
        expected = {
            path: value
            for path, value in case["expected_fields"].items()
            if value is not None and value != ""
        }
        data = outcome.get("data") if status in {"success", "partial"} else None
        predicted = populated_fields(data) if isinstance(data, dict) else {}
        required = case.get("required_paths", list(spec.required_fields))
        correct = {
            path
            for path, value in predicted.items()
            if path in expected and type(value) is type(expected[path]) and value == expected[path]
        }
        unsupported += len(predicted) - len(correct)
        domain = urlsplit(request.url).hostname or "unknown"
        schema_counts = by_schema.setdefault(spec.schema_id, _counts())
        domain_counts = by_domain.setdefault(domain, _counts())
        for key, paths in (
            ("present", set(expected)),
            ("predicted", set(predicted)),
            ("correct", correct),
        ):
            groups["all"][key] += len(paths)
            schema_counts[key] += len(paths)
            domain_counts[key] += len(paths)
            for path in paths:
                groups["required" if _required(path, required) else "optional"][key] += 1
        verified = 0
        evidence_gate = False
        evidence = outcome.get("evidence")
        if data is not None and evidence is not None:
            if isinstance(evidence, dict):
                evidence = list(evidence.values())
            candidate = {"data": data, "evidence": evidence, "warnings": []}
            assessment = (
                validate_product(candidate, snapshot, request.locale)
                if spec.schema_id == SCHEMA_ID
                else validate_general(candidate, snapshot, spec)
            )
            evidence_gate = not assessment.errors
            verified = len(assessment.evidence) if evidence_gate else 0
        verified_count += verified
        populated_count += len(predicted)
        evidence_passes += evidence_gate
        usable += outcome.get("retrieval_usable") is True
        known_unusable += outcome.get("retrieval_usable") is False
        unknown_retrieval += type(outcome.get("retrieval_usable")) is not bool
        details.append(
            {
                "case_id": case["id"],
                "schema_id": spec.schema_id,
                "status": status,
                "deadline_exceeded": late,
                "present": len(expected),
                "predicted": len(predicted),
                "correct": len(correct),
                "label_mismatches": sorted(set(predicted) - correct),
                "missed_paths": sorted(set(expected) - correct),
                "evidence_gate_passed": evidence_gate,
                "verified_populated_fields": verified,
            }
        )
    return {
        "candidate": receipt["metadata"],
        "corpus_id": manifest["corpus_id"],
        "case_count": len(cases),
        **{key: _metrics(value) for key, value in groups.items()},
        "by_schema": {key: _metrics(value) for key, value in by_schema.items()},
        "by_domain": {key: _metrics(value) for key, value in by_domain.items()},
        "statuses": dict(statuses),
        "expected_status_matches": status_matches,
        "missing_receipts": missing_receipts,
        "unsupported_populated_facts": unsupported,
        "unsupported_definition": "Populated predictions differing from complete reviewed scalar labels; not a universal truth claim",
        "evidence": {
            "accepted_records": evidence_passes,
            "verified_populated_fields": verified_count,
            "populated_fields": populated_count,
            "coverage": verified_count / populated_count if populated_count else None,
            "method": "Production schema/completeness/identity/exact-source validator, independent of label scoring",
        },
        "retrieval_coverage": {
            "reported_usable": usable,
            "reported_unusable": known_unusable,
            "unknown": unknown_retrieval,
            "total": len(cases),
        },
        "wall_latency": _quantiles(wall),
        "unknown_completion_timing": unknown_completion,
        "component_latency": {key: _quantiles(value) for key, value in component_values.items()},
        "usage": {
            key: {
                "measured_cases": value["measured_cases"],
                "total": value["total"] if value["measured_cases"] else None,
            }
            for key, value in usage_totals.items()
        },
        "cases": details,
    }


class _FixtureRetriever:
    retrieval_method = "fixture_replay"

    def __init__(self, record: dict):
        self.record = record

    @property
    def cache_options(self):
        return {"record_hash": _hash(self.record)}

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        return RetrievalOutcome(status="ready", record=self.record, retrieval_method="fixture_replay")


def implementation_hash() -> str:
    """Identify imported application code, excluding this shared measurement harness."""
    root = Path(ai_scraper.__file__).parent
    files = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if path.name != "competition.py"
    }
    return _hash(files)


async def run_manifest(
    manifest: dict,
    *,
    candidate: str = "evidence-ai-scraper",
    version: str | None = None,
    deadline_seconds: float = 180,
) -> dict:
    """Run cold fixture replays with no ambient credentials, network or model.

    asyncio timeouts are cooperative; CPU-bound work is also assessed against
    the elapsed deadline on return. Late work remains in latency and recall.
    """
    cases = validate_manifest(manifest)
    if not _finite(deadline_seconds, positive=True) or deadline_seconds > 1800:
        raise ValueError("Completion deadline must be finite and within (0, 1800] seconds")
    started = perf_counter()
    results = []
    for case in cases:
        request, _, snapshot, request_hash = case_context(case)
        begin = perf_counter()
        with tempfile.TemporaryDirectory(prefix="scraper-comparison-") as directory:
            hostname = urlsplit(request.url).hostname or "invalid.test"
            pipeline = Pipeline(
                Settings(state_dir=Path(directory), allowed_domains=[hostname], retrieval_backend="direct"),
                retriever=_FixtureRetriever(case["record"]),
                policy=URLPolicy([hostname], resolve_dns=False),
                live=False,
            )
            try:
                result = await asyncio.wait_for(pipeline.extract_url(request), timeout=deadline_seconds)
                payload = result.model_dump(mode="json")
            except TimeoutError:
                payload = {"status": "failed", "data": None, "error": {"code": "COMPLETION_DEADLINE"}}
            finally:
                await pipeline.close()
        elapsed = (perf_counter() - begin) * 1000
        if elapsed > deadline_seconds * 1000:
            payload.update(status="failed", data=None, evidence={}, error={"code": "COMPLETION_DEADLINE"})
        results.append(
            {
                "case_id": case["id"],
                "source_hash": snapshot.source_hash,
                "request_hash": request_hash,
                "status": payload["status"],
                "data": payload.get("data"),
                "evidence": payload.get("evidence"),
                "error": payload.get("error"),
                "usage": {key: payload.get("usage", {}).get(key) for key in _USAGE_COUNTS},
                "wall_latency_ms": elapsed,
                "component_latency_ms": {"pipeline_reported": payload["usage"]["latency_ms"]}
                if "usage" in payload
                else {},
                "retrieval_usable": bool(payload.get("provenance", {}).get("source_hash"))
                and detect_content_problem(snapshot) is None,
            }
        )
    return {
        "format": RECEIPT_FORMAT,
        "corpus_id": manifest["corpus_id"],
        "manifest_hash": _hash(manifest),
        "harness_hash": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "checked_at": datetime.now(UTC).isoformat(),
        "metadata": {
            "candidate": candidate,
            "version": version or ai_scraper.__version__,
            "implementation_hash": implementation_hash(),
            "mode": "deterministic_offline",
            "track": "identical_snapshot",
            "runtime": f"Python {platform.python_version()}",
            "hardware": {"system": platform.system(), "machine": platform.machine()},
            "model": None,
            "provider": None,
            "cache_state": "cold",
            "deadline_seconds": deadline_seconds,
            "deadline_enforcement": "cooperative timeout plus post-completion CPU deadline assessment",
            "network_calls": 0,
        },
        "total_wall_latency_ms": (perf_counter() - started) * 1000,
        "results": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Offline application fixture replays; no model or network")
    run.add_argument("manifest", type=Path)
    run.add_argument("--candidate", default="evidence-ai-scraper")
    run.add_argument("--candidate-version")
    run.add_argument("--deadline-seconds", type=float, default=180)
    run.add_argument("--output", type=Path, required=True)
    score = commands.add_parser("score", help="Score application or independently collected receipts")
    score.add_argument("manifest", type=Path)
    score.add_argument("receipts", nargs="+", type=Path)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        manifest = _read(args.manifest)
        if args.command == "run":
            result = asyncio.run(
                run_manifest(
                    manifest,
                    candidate=args.candidate,
                    version=args.candidate_version,
                    deadline_seconds=args.deadline_seconds,
                )
            )
        else:
            result = {
                "format": "snapshot-comparison-report-v1",
                "corpus_id": manifest.get("corpus_id"),
                "manifest_hash": _hash(manifest),
                "limitations": "Reviewed authored fixtures; this is not a representative live-web or model-accuracy benchmark",
                "candidates": [score_receipt(manifest, _read(path)) for path in args.receipts],
            }
        atomic_json(args.output, result)
        print(json.dumps({"output": str(args.output), "corpus_id": manifest["corpus_id"]}))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ScraperError):
        print("INVALID_COMPARISON: Invalid benchmark input or unavailable local state", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
