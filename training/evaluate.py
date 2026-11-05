"""Compare extraction candidates against authored labels and production validators.

This measures a model's initial completion on saved sources, without deterministic
shortcuts or repair. It is separate from pipeline end-to-end measurements. Receipt
files contain hashes and counts, not source text, model responses or local paths.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

from pydantic import ValidationError

from ai_scraper.contracts import ScraperError
from ai_scraper.extraction.generic import populated_fields
from ai_scraper.extraction.local import LocalExtractor, validate_local_endpoint
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest, SourceSnapshot
from ai_scraper.storage import atomic_json
from ai_scraper.validate import validate_candidate as validate_product
from ai_scraper.validate_generic import validate_candidate as validate_general

FORMAT = "scraper-model-evaluation-v1"
MAX_ROWS = 10_000
MAX_FILE_BYTES = 100_000_000
MAX_ROW_BYTES = 2_000_000
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")
_ERROR = re.compile(r"[A-Z][A-Z0-9_]{0,79}")
_MODEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/-]{0,199}")
_FINISH = {"stop", "eos_token", "end_turn", "length", "max_tokens", "max_output_tokens"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _reject_constant(_: str):
    raise ValueError("Nonfinite JSON is forbidden")


def _unique(items: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON keys are forbidden")
        result[key] = value
    return result


def _parse(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_unique)


def read_jsonl(path: Path) -> list[dict]:
    """Read bounded strict JSONL; callers keep source/response files private."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Evaluation input exceeds its file bound")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if len(line.encode()) > MAX_ROW_BYTES or len(rows) >= MAX_ROWS:
                raise ValueError("Evaluation input exceeds its row bound")
            row = _parse(line)
            if not isinstance(row, dict):
                raise TypeError("Evaluation rows must be objects")
            rows.append(row)
    return rows


def _context(case: dict):
    request = ExtractionRequest.model_validate(case["request"])
    spec = get_schema(request)
    snapshot = SourceSnapshot.model_validate_json(_canonical(case["snapshot"]))
    rebuilt = normalize_record(snapshot.raw_record, request, retrieval_method=snapshot.retrieval_method)
    if (
        case.get("schema_id") != spec.schema_id
        or case.get("schema_fingerprint") != spec.fingerprint
        or case.get("source_hash") != snapshot.source_hash
        or rebuilt.source_hash != snapshot.source_hash
        or rebuilt.sections != snapshot.sections
        or rebuilt.ld_json != snapshot.ld_json
        or snapshot.requested_url != request.url
        or snapshot.final_url != rebuilt.final_url
    ):
        raise ValueError("Case does not match its immutable source and current schema")
    gold = spec.candidate_model.model_validate(case["expected_candidate"])
    gold_data = gold.data.model_dump()
    expected = populated_fields(gold_data)
    if case.get("expected_fields") != expected or case.get("expected_status") not in {
        "success",
        "partial",
        "failed",
    }:
        raise ValueError("Case labels do not match its authored candidate")
    assessment = _assess(gold, snapshot, request, spec)
    if assessment.status != case["expected_status"] or any(
        not error.startswith("completeness:") for error in assessment.errors
    ):
        raise ValueError("Gold labels fail production source validation")
    return request, spec, snapshot, gold


def validate_cases(cases: list[dict]) -> None:
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_ROWS:
        raise ValueError("A bounded nonempty case list is required")
    seen = set()
    for case in cases:
        case_id = case.get("case_id") if isinstance(case, dict) else None
        if not isinstance(case_id, str) or not _ID.fullmatch(case_id) or case_id in seen:
            raise ValueError("Invalid or duplicate case ID")
        seen.add(case_id)
        _context(case)


def _assess(candidate, snapshot, request, spec):
    return (
        validate_product(candidate, snapshot, request.locale)
        if spec.schema_id == "product_v1"
        else validate_general(candidate, snapshot, spec)
    )


def _model_metadata(model: dict, inference: dict) -> tuple[dict, dict]:
    if not isinstance(model, dict) or set(model) - {"id", "revision", "weights_sha256"}:
        raise ValueError("Explicit bounded model metadata is required")
    model_id, revision = model.get("id"), model.get("revision")
    if (
        not isinstance(model_id, str)
        or not _MODEL.fullmatch(model_id)
        or any(part in {"", ".", ".."} for part in model_id.split("/"))
        or not isinstance(revision, str)
        or not _ID.fullmatch(revision)
    ):
        raise ValueError("Use a public artifact identifier/revision, not local paths or URLs")
    weight_hash = model.get("weights_sha256")
    if weight_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", weight_hash):
        raise ValueError("Model weights hash must be SHA256")
    allowed = {
        "temperature",
        "max_tokens",
        "enable_thinking",
        "response_format",
        "seed",
        "repair_attempts",
        "max_source_chars",
    }
    if not isinstance(inference, dict) or set(inference) - allowed:
        raise ValueError("Unsupported inference metadata; credentials/URLs are never recorded")
    if (
        type(inference.get("temperature")) not in {int, float}
        or not math.isfinite(inference["temperature"])
        or not 0 <= inference["temperature"] <= 2
    ):
        raise ValueError("A finite temperature is required")
    if type(inference.get("max_tokens")) is not int or not 1 <= inference["max_tokens"] <= 32768:
        raise ValueError("A bounded output token count is required")
    if inference.get("enable_thinking") is not False or (
        type(inference.get("repair_attempts")) is not int or inference["repair_attempts"] != 0
    ):
        raise ValueError("This comparison uses thinking disabled and the initial completion only")
    if inference.get("response_format") not in {"json_schema", "none"}:
        raise ValueError("Response grammar mode must be explicit")
    for key in ("seed", "max_source_chars"):
        value = inference.get(key)
        if value is not None and (type(value) is not int or not 0 <= value <= 1_000_000_000):
            raise ValueError("Inference counts must be bounded integers")
    return dict(model), dict(inference)


def _validate_outcome(row: dict) -> None:
    if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
        raise TypeError("Each output needs a case ID")
    if "candidate" in row and "raw_output" in row:
        raise ValueError("Use candidate or raw_output, not both")
    if "candidate" in row and not isinstance(row["candidate"], dict):
        raise ValueError("Candidate outputs must be objects")
    if "raw_output" in row and (
        not isinstance(row["raw_output"], str) or len(row["raw_output"].encode()) > MAX_ROW_BYTES
    ):
        raise ValueError("Raw output must be a bounded string")
    if not ({"candidate", "raw_output", "error_code"} & row.keys()):
        raise ValueError("An output, or a failure code, is required")
    code = row.get("error_code")
    if code is not None and (not isinstance(code, str) or not _ERROR.fullmatch(code)):
        raise ValueError("Error codes must be identifiers, never exception messages")
    finish = row.get("finish_reason")
    if finish is not None and finish not in _FINISH:
        raise ValueError("Unknown finish reason")
    elapsed = row.get("elapsed_ms")
    if elapsed is not None and (
        type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0
    ):
        raise ValueError("Latency must be measured finite milliseconds or null")
    for key in ("input_tokens", "output_tokens"):
        value = row.get(key)
        if value is not None and (type(value) is not int or value < 0):
            raise ValueError("Token counts must be measured nonnegative integers or null")


def _negative_paths(value: Any, prefix: str = "") -> set[str]:
    if value is None or value == "" or value == []:
        return {prefix} if prefix else set()
    if isinstance(value, dict):
        return (
            set().union(
                *(_negative_paths(item, f"{prefix}.{key}" if prefix else key) for key, item in value.items())
            )
            if value
            else set()
        )
    if isinstance(value, list):
        return (
            set().union(*(_negative_paths(item, f"{prefix}.{index}") for index, item in enumerate(value)))
            if value
            else set()
        )
    return set()


def _at(value: Any, path: str):
    for part in path.split("."):
        try:
            value = value[int(part)] if isinstance(value, list) else value[part]
        except (KeyError, IndexError, TypeError, ValueError):
            return object()
    return value


def _repeats(text: str) -> bool:
    """Conservative diagnostic: three adjacent copies of a >=32-token suffix."""
    tokens = re.findall(r"\w+|[^\w\s]", text[-100_000:])[-4096:]
    for width in (32, 64, 128, 256, 512):
        if (
            len(tokens) >= width * 3
            and tokens[-width:] == tokens[-2 * width : -width] == tokens[-3 * width : -2 * width]
        ):
            return True
    return False


def _reference_matches(actual, expected) -> bool:
    return (
        actual.section_id == expected.section_id
        and actual.json_pointer == expected.json_pointer
        and expected.excerpt in actual.excerpt
    )


def _safe_path(path: str, spec) -> str:
    """Only expose trusted schema names and array indices in public receipts."""
    schema = spec.data_model.model_json_schema()
    node = schema
    try:
        for part in path.split("."):
            while True:
                if "$ref" in node:
                    node = schema["$defs"][node["$ref"].rsplit("/", 1)[-1]]
                elif "anyOf" in node:
                    node = next(item for item in node["anyOf"] if item.get("type") != "null")
                else:
                    break
            if node.get("type") == "array" and re.fullmatch(r"0|[1-9][0-9]*", part):
                node = node["items"]
            elif node.get("type") == "object":
                node = node["properties"][part]
            else:
                raise KeyError
        return path
    except (KeyError, StopIteration):
        return "unknown_field_sha256:" + _hash(path)


def _metrics(present: int, predicted: int, correct: int) -> dict:
    return {
        "present": present,
        "predicted": predicted,
        "correct": correct,
        "wrong_or_unlabeled": predicted - correct,
        "missed": present - correct,
        "precision": correct / predicted if predicted else None,
        "recall": correct / present if present else None,
        "f1": 2 * correct / (present + predicted) if present + predicted else None,
    }


def _latencies(values: list[float]) -> dict:
    values = sorted(values)
    return {
        "measured_cases": len(values),
        "p50_ms": values[math.ceil(0.50 * len(values)) - 1] if values else None,
        "p95_ms": values[math.ceil(0.95 * len(values)) - 1] if values else None,
    }


def _required(path: str, required: tuple[str, ...]) -> bool:
    return any(path == root or path.startswith(root + ".") for root in required)


def _accumulate(group: Counter, expected: set, predicted: set, correct: set, accepted: bool, required):
    group["case_count"] += 1
    for prefix, select in (
        ("", lambda _: True),
        ("required_", lambda path: _required(path, required)),
        ("optional_", lambda path: not _required(path, required)),
    ):
        for name, paths in (("present", expected), ("predicted", predicted), ("correct", correct)):
            count = sum(select(path) for path in paths)
            group[prefix + name] += count
            if name != "present":
                group[prefix + "accepted_" + name] += count if accepted else 0


def _group_metrics(group: Counter) -> dict:
    def metrics(prefix="", accepted=False):
        return _metrics(
            group[prefix + "present"],
            group[prefix + ("accepted_" if accepted else "") + "predicted"],
            group[prefix + ("accepted_" if accepted else "") + "correct"],
        )

    return {
        "case_count": group["case_count"],
        "raw_facts": metrics(),
        "accepted_facts": metrics(accepted=True),
        "required_facts": {"raw": metrics("required_"), "accepted": metrics("required_", accepted=True)},
        "optional_facts": {"raw": metrics("optional_"), "accepted": metrics("optional_", accepted=True)},
    }


def score_outputs(cases: list[dict], outputs: list[dict], *, model: dict, inference: dict) -> dict:
    """Score all cases, including omitted, malformed, truncated and failed work.

    Raw fact scoring does not give production validation credit. Accepted output
    recall is reported separately, so a well-grounded leaf in a rejected record
    does not appear as usable application output.
    """
    validate_cases(cases)
    model, inference = _model_metadata(model, inference)
    if not isinstance(outputs, list) or len(outputs) > MAX_ROWS:
        raise ValueError("Outputs must be a bounded list")
    known = {case["case_id"] for case in cases}
    by_id = {}
    for row in outputs:
        _validate_outcome(row)
        if row["case_id"] not in known or row["case_id"] in by_id:
            raise ValueError("Unknown or duplicate output case ID")
        by_id[row["case_id"]] = row
    totals: Counter = Counter()
    grouped = {"all": {"all": Counter()}, "schema": {}, "domain": {}, "field": {}}
    statuses: Counter = Counter()
    errors: Counter = Counter()
    details, elapsed_values = [], []
    tokens = {key: {"total": 0, "measured_cases": 0} for key in ("input_tokens", "output_tokens")}
    for case in cases:
        request, spec, snapshot, gold = _context(case)
        row = by_id.get(case["case_id"])
        expected = populated_fields(gold.data)
        negative = _negative_paths(gold.data.model_dump())
        totals["present"] += len(expected)
        totals["negative_paths"] += len(negative)
        candidate, parsed, assessment = None, None, None
        valid_json = False
        raw = ""
        code = "MISSING_OUTPUT" if row is None else row.get("error_code")
        token_limit = False
        repeating = False
        if row is not None:
            if "candidate" in row:
                candidate = row["candidate"]
                raw = _canonical(candidate)
                valid_json = True
            elif "raw_output" in row:
                raw = row["raw_output"]
                try:
                    candidate = _parse(raw)
                    valid_json = True
                except (ValueError, RecursionError):
                    code = code or "INVALID_JSON"
            token_limit = row.get("finish_reason") in {"length", "max_tokens", "max_output_tokens"} or (
                code == "TRUNCATED_OUTPUT"
            )
            repeating = _repeats(raw)
            if row.get("elapsed_ms") is not None:
                elapsed_values.append(row["elapsed_ms"])
            for key, counts in tokens.items():
                if row.get(key) is not None:
                    counts["total"] += row[key]
                    counts["measured_cases"] += 1
        if valid_json:
            try:
                parsed = spec.candidate_model.model_validate(candidate)
                assessment = _assess(parsed, snapshot, request, spec)
            except (ValidationError, ValueError, TypeError):
                code = code or "INVALID_SCHEMA"
        predicted_data = candidate.get("data", {}) if isinstance(candidate, dict) else {}
        predicted = populated_fields(predicted_data) if isinstance(predicted_data, dict) else {}
        correct = {
            path
            for path, value in predicted.items()
            if path in expected and type(value) is type(expected[path]) and value == expected[path]
        }
        supported: set[str] = set()
        reference_matches: set[str] = set()
        if parsed is not None:
            invalid_evidence = {
                error.split(":", 2)[1] for error in assessment.errors if error.startswith("evidence:")
            }
            supported = set(predicted) - invalid_evidence
            expected_references = {item.field_path: item for item in gold.evidence}
            for item in parsed.evidence:
                if (
                    item.field_path in supported & correct
                    and item.field_path in expected_references
                    and (_reference_matches(item, expected_references[item.field_path]))
                ):
                    reference_matches.add(item.field_path)
        status = assessment.status if assessment is not None else "failed"
        if token_limit or code:
            status = "failed"
        accepted = status in {"success", "partial"}
        for dimension, key in (
            ("all", "all"),
            ("schema", spec.schema_id),
            ("domain", urlsplit(request.url).hostname or "unknown"),
        ):
            _accumulate(
                grouped[dimension].setdefault(key, Counter()),
                set(expected),
                set(predicted),
                correct,
                accepted,
                spec.required_fields,
            )
        field_groups = {}
        for path in set(expected) | set(predicted):
            safe = _safe_path(path, spec)
            field = re.sub(r"(?<=\.)[0-9]+(?=\.|$)", "*", safe)
            field_groups.setdefault(f"{spec.schema_id}:{field}", set()).add(path)
        for field, paths in field_groups.items():
            _accumulate(
                grouped["field"].setdefault(field, Counter()),
                set(expected) & paths,
                set(predicted) & paths,
                correct & paths,
                accepted,
                spec.required_fields,
            )
        abstained = 0
        if parsed is not None and not token_limit and not code:
            actual_data = parsed.data.model_dump()
            abstained = sum(_at(actual_data, path) in (None, "", []) for path in negative)
        totals.update(
            {
                "predicted": len(predicted),
                "correct": len(correct),
                "accepted_correct": len(correct) if accepted else 0,
                "accepted_predicted": len(predicted) if accepted else 0,
                "valid_json": valid_json,
                "valid_schema": parsed is not None,
                "accepted": accepted,
                "status_matches": parsed is not None
                and assessment is not None
                and not token_limit
                and not code
                and status == case["expected_status"],
                "source_supported": len(supported),
                "correct_with_source_support": len(correct & supported),
                "authored_reference_matches": len(reference_matches),
                "correct_abstentions": abstained,
                "missing_outputs": row is None,
                "token_limit": token_limit,
                "repetition_heuristic": repeating,
            }
        )
        statuses[status] += 1
        if code:
            errors[code] += 1
        validation_codes = (
            sorted({error.split(":", 1)[0] for error in assessment.errors}) if assessment else []
        )
        details.append(
            {
                "case_id": case["case_id"],
                "schema_id": spec.schema_id,
                "case_hash": _hash(case),
                "source_hash": snapshot.source_hash,
                "output_hash": _hash(row) if row is not None else None,
                "valid_json": valid_json,
                "valid_schema": parsed is not None,
                "status": status,
                "expected_status": case["expected_status"],
                "error_code": code,
                "validation_codes": validation_codes,
                "present": len(expected),
                "predicted": len(predicted),
                "correct": len(correct),
                "wrong_paths": sorted(_safe_path(path, spec) for path in set(predicted) - correct),
                "missed_paths": sorted(set(expected) - correct),
                "source_supported": len(supported),
                "correct_with_source_support": len(correct & supported),
                "authored_reference_matches": len(reference_matches),
                "negative_paths": len(negative),
                "correct_abstentions": abstained,
                "elapsed_ms": row.get("elapsed_ms") if row else None,
                "token_limit": token_limit,
                "repetition_heuristic": repeating,
            }
        )
    count = len(cases)
    return {
        "format": FORMAT,
        "created_at": datetime.now(UTC).isoformat(),
        "model": model,
        "inference": inference,
        "evaluation_mode": "initial_model_completion_on_identical_saved_snapshot",
        "cases_hash": _hash(cases),
        "outputs_hash": _hash(outputs),
        "case_count": count,
        "missing_outputs": totals["missing_outputs"],
        "valid_json_cases": totals["valid_json"],
        "valid_schema_cases": totals["valid_schema"],
        "accepted_cases": totals["accepted"],
        "expected_status_matches": totals["status_matches"],
        "status_match_definition": "Status agreement requires a parsed candidate and production assessment without execution error or token truncation; it does not establish semantic correctness",
        "raw_facts": _metrics(totals["present"], totals["predicted"], totals["correct"]),
        "accepted_facts": _metrics(
            totals["present"], totals["accepted_predicted"], totals["accepted_correct"]
        ),
        "required_facts": _group_metrics(grouped["all"]["all"])["required_facts"],
        "optional_facts": _group_metrics(grouped["all"]["all"])["optional_facts"],
        "required_definition": "Populated scalar leaves under trusted registry required fields; null/empty controls are scored as abstention separately",
        "by_schema": {key: _group_metrics(value) for key, value in sorted(grouped["schema"].items())},
        "by_domain": {key: _group_metrics(value) for key, value in sorted(grouped["domain"].items())},
        "by_field": {key: _group_metrics(value) for key, value in sorted(grouped["field"].items())},
        "source_evidence": {
            "populated_fields": totals["predicted"],
            "source_supported_fields": totals["source_supported"],
            "correct_with_source_support": totals["correct_with_source_support"],
            "authored_reference_matches": totals["authored_reference_matches"],
        },
        "abstention": {
            "labeled_absent_paths": totals["negative_paths"],
            "correct_abstentions": totals["correct_abstentions"],
            "accuracy": totals["correct_abstentions"] / totals["negative_paths"]
            if totals["negative_paths"]
            else None,
        },
        "diagnostics": {
            "token_limit_cases": totals["token_limit"],
            "repetition_heuristic_cases": totals["repetition_heuristic"],
            "statuses": dict(statuses),
            "errors": dict(errors),
        },
        "latency": {**_latencies(elapsed_values), "unknown_cases": count - len(elapsed_values)},
        "tokens": {
            key: {**value, "total": value["total"] if value["measured_cases"] else None}
            for key, value in tokens.items()
        },
        "limitations": [
            "Synthetic authored labels are not human-reviewed real-web accuracy.",
            "Exact source occurrence and production acceptance do not establish arbitrary field meaning.",
            "Authored-reference matches require the gold locator and quote; other valid anchors may differ.",
            "Repetition is a conservative suffix heuristic, not a measured behavioral mechanism.",
            "Raw fact metrics include rejected/truncated candidates; accepted facts require application acceptance.",
        ],
        "cases": details,
    }


async def run_local(
    cases: list[dict],
    *,
    model_id: str,
    revision: str,
    base_url: str,
    max_tokens: int = 4096,
    timeout: float = 120,
    max_source_chars: int = 48000,
    output_path: Path | None = None,
) -> tuple[list[dict], dict]:
    """Run only loopback inference. Saved raw outputs are private evaluation input.

    No retrieval, credential lookup, deterministic extraction or repair is run.
    Every attempted case is checkpointed atomically; restart skips completed IDs.
    The checkpoint binds to source/case identity and model/config before reuse.
    """
    validate_cases(cases)
    validate_local_endpoint(base_url)
    if type(timeout) not in {int, float} or not math.isfinite(timeout) or not 0 < timeout <= 1800:
        raise ValueError("A bounded positive timeout is required")
    inference = {
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "response_format": "json_schema",
        "repair_attempts": 0,
        "max_source_chars": max_source_chars,
    }
    model, inference = _model_metadata({"id": model_id, "revision": revision}, inference)
    identity = _hash({"cases": cases, "model": model, "inference": inference, "timeout_seconds": timeout})
    rows = []
    if output_path is not None and output_path.exists():
        if output_path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Checkpoint exceeds its file bound")
        saved = _parse(output_path.read_text())
        if not isinstance(saved, dict) or saved.get("identity") != identity:
            raise ValueError("Existing checkpoint belongs to different model/config/cases")
        rows = saved.get("outputs")
        # Reuse full scorer validation of duplicate IDs, latencies and metadata.
        score_outputs(cases, rows, model=model, inference=inference)
    completed = {row["case_id"] for row in rows}
    extractor = LocalExtractor(
        model_id, base_url=base_url, max_tokens=max_tokens, timeout=timeout, max_source_chars=max_source_chars
    )
    try:
        for case in cases:
            if case["case_id"] in completed:
                continue
            request, spec, snapshot, _ = _context(case)
            scoped = extractor.for_request(request, spec)
            row = {"case_id": case["case_id"]}
            begin = perf_counter()
            try:
                response = await scoped.extract(snapshot)
                row.update(candidate=response.payload, finish_reason="stop")
            except ScraperError as exc:
                row["error_code"] = exc.error.code
            except Exception:  # noqa: BLE001 -- redact failures at the model/server boundary
                # Never emit third-party exception strings or server URLs.
                row["error_code"] = "LOCAL_EVALUATION_FAILED"
            row["elapsed_ms"] = (perf_counter() - begin) * 1000
            usage = getattr(scoped, "last_usage", {})
            for key in ("input_tokens", "output_tokens"):
                row[key] = usage.get(key)
            rows.append(row)
            if output_path is not None:
                atomic_json(output_path, {"identity": identity, "outputs": rows})
    finally:
        await extractor.close()
    return rows, score_outputs(cases, rows, model=model, inference=inference)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Score private raw completion JSONL offline")
    score.add_argument("cases", type=Path)
    score.add_argument("outputs", type=Path)
    local = commands.add_parser("local", help="Run a named model through a loopback schema server")
    local.add_argument("cases", type=Path)
    local.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    local.add_argument("--timeout", type=float, default=120)
    local.add_argument("--max-source-chars", type=int, default=48000)
    local.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Private raw output checkpoint; identity-checked when resuming",
    )
    for command in (score, local):
        command.add_argument("--model", required=True, help="Public artifact ID, never a local path")
        command.add_argument("--revision", required=True)
        command.add_argument("--max-tokens", type=int, default=4096)
        command.add_argument("--output", type=Path, required=True, help="Sanitized count/hash receipt")
    score.add_argument("--response-format", choices=["none", "json_schema"], required=True)
    args = parser.parse_args(argv)
    try:
        cases = read_jsonl(args.cases)
        if args.command == "local":
            _, receipt = asyncio.run(
                run_local(
                    cases,
                    model_id=args.model,
                    revision=args.revision,
                    base_url=args.base_url,
                    max_tokens=args.max_tokens,
                    timeout=args.timeout,
                    max_source_chars=args.max_source_chars,
                    output_path=args.checkpoint,
                )
            )
        else:
            receipt = score_outputs(
                cases,
                read_jsonl(args.outputs),
                model={"id": args.model, "revision": args.revision},
                inference={
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "enable_thinking": False,
                    "repair_attempts": 0,
                    "response_format": args.response_format,
                },
            )
        atomic_json(args.output, receipt)
        print(json.dumps({"case_count": receipt["case_count"], "accepted_cases": receipt["accepted_cases"]}))
        return 0
    except (OSError, ValueError, TypeError, KeyError, ValidationError, ScraperError):
        print("INVALID_EVALUATION: Invalid local evaluation input or unavailable state", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
