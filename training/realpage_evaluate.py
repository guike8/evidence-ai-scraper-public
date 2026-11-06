"""Frozen two-stage local-model comparison on reviewed saved public pages.

No URL retrieval, training, hosted inference or repair is performed. Complete
sources and raw stage outputs stay private; receipts contain hashes, trusted
field names and counts. Every case stays in the denominator.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import math
import os
import platform
import stat
import sys
import time
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from ai_scraper.contracts import ScraperError
from ai_scraper.extraction.generic import populated_fields
from ai_scraper.extraction.grounded import (
    GROUNDED_POLICY_VERSION,
    GroundedLocalExtractor,
    bind_grounding_selection,
    build_business_task,
    build_grounding_task,
)
from ai_scraper.extraction.grounded import (
    _data as validate_business,
)
from ai_scraper.extraction.grounded_view import GROUNDED_VIEW_VERSION
from ai_scraper.extraction.local import validate_local_endpoint
from ai_scraper.storage import atomic_json

from . import evaluate
from .firecrawl_benchmark import (
    _CODE,
    _SHA,
    BenchmarkDeadline,
    _read,
    _settings,
    _sha,
    checkpoint_lock,
    code_identity,
    hard_deadline,
    require,
    verify_freeze,
    write_freeze,
)

FORMAT = "realpage-two-stage-evaluation-v1"
FREEZE_FORMAT = "realpage-two-stage-freeze-v1"
CHECKPOINT_FORMAT = "realpage-two-stage-checkpoint-v1"
MAX_WALL_SECONDS = 1800
MAX_CASE_ROW_BYTES = 8_000_000


def _cases(path):
    """Read full saved representations without widening model input limits.

    A case can retain raw HTML and several normalized representations while
    its model view remains bounded to 8,000 characters. Reuse the regular-file,
    no-symlink and stable-inode checks; keep this larger row budget isolated
    from the older benchmark and generic evaluation readers.
    """
    body = _read(Path(path), evaluate.MAX_FILE_BYTES)
    cases = []
    for line in body.splitlines():
        if not line.strip():
            continue
        require(len(line) <= MAX_CASE_ROW_BYTES and len(cases) < evaluate.MAX_ROWS, "CASE_LIMIT")
        row = evaluate._parse(line.decode("utf-8"))
        require(isinstance(row, dict), "INVALID_CASE")
        cases.append(row)
    evaluate.validate_cases(cases)
    return cases, _sha(body)


def python_identity():
    """Bind the executing interpreter without exposing a local filesystem path."""
    return {
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "executable_sha256": _sha(_read(Path(sys.executable).resolve(), 100_000_000)),
    }


def _anchors(case):
    anchors = case.get("reviewed_evidence")
    require(isinstance(anchors, list) and len(anchors) <= 1024, "REVIEWED_EVIDENCE_REQUIRED")
    expected = set(case["expected_fields"])
    indexed = {section["id"]: section["text"] for section in case["snapshot"]["sections"]}
    paths = set()
    for item in anchors:
        require(
            isinstance(item, dict)
            and item.get("field_path") in expected
            and isinstance(item.get("section_id"), str)
            and item["section_id"] in indexed
            and isinstance(item.get("excerpt"), str)
            and bool(item["excerpt"])
            and item["excerpt"] in indexed[item["section_id"]]
            and isinstance(item.get("semantic_context"), str)
            and bool(item["semantic_context"]),
            "INVALID_REVIEWED_EVIDENCE",
        )
        paths.add(item["field_path"])
    require(paths == expected, "INCOMPLETE_REVIEWED_EVIDENCE")
    return anchors


def _plan(scoped, snapshot):
    """Freeze source/business inputs without deriving a catalog from gold."""
    try:
        source = scoped._included_source(snapshot)
        task = build_business_task(
            scoped.spec, source, scoped.instruction, max_source_chars=scoped.max_source_chars
        )
    except ScraperError as exc:
        require(bool(_CODE.fullmatch(exc.error.code)), "INVALID_ERROR_CODE")
        return {"error_code": exc.error.code, "source_hash": snapshot.source_hash}, None
    descriptor = copy.deepcopy(scoped.last_source_view)
    return {
        "source_hash": snapshot.source_hash,
        "source_view_sha256": evaluate._hash(source),
        "source_descriptor": descriptor,
        "business_schema_sha256": evaluate._hash(task.schema),
        "business_messages_sha256": evaluate._hash(task.messages),
        "business_metadata_sha256": evaluate._hash(task.metadata),
    }, source


def freeze_identity(
    cases,
    *,
    cases_sha256,
    model,
    base_url,
    max_tokens=768,
    timeout=45,
    max_source_chars=8000,
    wall_seconds=1800,
):
    evaluate.validate_cases(cases)
    require(isinstance(cases_sha256, str) and bool(_SHA.fullmatch(cases_sha256)), "INVALID_CASES_SHA")
    require(
        type(wall_seconds) in {int, float}
        and math.isfinite(wall_seconds)
        and 0 < wall_seconds <= MAX_WALL_SECONDS,
        "INVALID_WALL_LIMIT",
    )
    require(type(max_source_chars) is int and 1024 <= max_source_chars <= 8000, "INVALID_SOURCE_LIMIT")
    model, inference, base_url = _settings(
        model,
        base_url=base_url,
        source_policy="markdown",
        max_tokens=max_tokens,
        timeout=timeout,
        max_source_chars=max_source_chars,
        wall_seconds=min(wall_seconds, 1200),
    )
    extractor = GroundedLocalExtractor(
        model["id"],
        base_url=base_url,
        max_tokens=max_tokens,
        timeout=timeout,
        max_source_chars=max_source_chars,
        client=object(),
    )
    inputs = {}
    for case in cases:
        _anchors(case)
        request, spec, snapshot, _ = evaluate._context(case)
        inputs[case["case_id"]] = _plan(extractor.for_request(request, spec), snapshot)[0]
    return {
        "format": FREEZE_FORMAT,
        "cases_sha256": cases_sha256,
        "cases_hash": evaluate._hash(cases),
        "case_order": [case["case_id"] for case in cases],
        "case_count": len(cases),
        "model": model,
        "inference": inference,
        "timeout_seconds_per_stage": timeout,
        "wall_seconds": wall_seconds,
        "maximum_model_calls_per_case": 2,
        "citation_policy_version": GROUNDED_POLICY_VERSION,
        "source_view_version": GROUNDED_VIEW_VERSION,
        "endpoint_profile_sha256": evaluate._hash({"base_url": base_url}),
        "code": code_identity(),
        "python": python_identity(),
        "inputs": inputs,
    }


def _checkpoint(path, identity, rows, in_flight=None):
    require(not Path(path).is_symlink(), "INVALID_CHECKPOINT")
    body = {"format": CHECKPOINT_FORMAT, "identity": identity, "outputs": rows, "in_flight": in_flight}
    require(len(evaluate._canonical(body).encode()) <= evaluate.MAX_FILE_BYTES, "CHECKPOINT_LIMIT")
    atomic_json(Path(path), body)


def _stage_payload(scoped, name):
    payload = getattr(scoped, "last_business_payload" if name == "business" else "last_grounding_selection")
    if isinstance(payload, dict):
        return copy.deepcopy(payload)
    raw = getattr(scoped, f"last_{name}_response")
    if isinstance(raw, str) and not getattr(scoped, f"last_{name}_response_truncated"):
        try:
            parsed = evaluate._parse(raw)
            return parsed if isinstance(parsed, dict) else None
        except (ValueError, TypeError, RecursionError):
            pass
    return None


def _time_stages(scoped):
    """Measure actual local transport calls, including timeout/error exits."""
    original = scoped._local_client.chat_completion
    latencies = []

    async def completion(**kwargs):
        start = time.monotonic()
        try:
            return await original(**kwargs)
        finally:
            latencies.append((time.monotonic() - start) * 1000)

    scoped._local_client.chat_completion = completion
    return latencies


def _capture(row, scoped, latencies):
    """Explicit private checkpoint fields; never merge untrusted model objects."""
    raw = _stage_payload(scoped, "business")
    if raw is not None and "candidate" not in row:
        row["candidate"] = {"data": raw, "evidence": [], "warnings": []}
    row["private"] = {
        "business": raw,
        "grounding_selection": _stage_payload(scoped, "grounding"),
        "business_response": scoped.last_business_response,
        "grounding_response": scoped.last_grounding_response,
        "business_response_truncated": scoped.last_business_response_truncated,
        "grounding_response_truncated": scoped.last_grounding_response_truncated,
        "grounding_metadata": copy.deepcopy(scoped.last_grounding_metadata),
    }
    if scoped.last_grounding_task is not None:
        row["private"].update(
            grounding_schema_sha256=evaluate._hash(scoped.last_grounding_task.schema),
            grounding_messages_sha256=evaluate._hash(scoped.last_grounding_task.messages),
        )
    usage = scoped.last_usage
    row["model_calls"] = usage.get("model_calls", 0)
    row["stages"] = copy.deepcopy(usage.get("stages", []))
    for stage, elapsed in zip(row["stages"], latencies, strict=False):
        stage["elapsed_ms"] = elapsed
    for key in ("input_tokens", "output_tokens"):
        value = usage.get(key)
        row[key] = value if type(value) is int and value >= 0 else None


def _regate(row, scoped, snapshot, source):
    """A resumed record cannot bypass the current frozen binding/validators."""
    if row.get("error_code"):
        return
    private = row.get("private", {})
    try:
        task = build_grounding_task(
            scoped.spec,
            private["business"],
            source,
            scoped.instruction,
            max_source_chars=scoped.max_source_chars,
        )
        require(
            private.get("grounding_metadata") == task.metadata
            and private.get("grounding_schema_sha256") == evaluate._hash(task.schema)
            and private.get("grounding_messages_sha256") == evaluate._hash(task.messages),
            "CHECKPOINT_GROUNDING_CHANGED",
        )
        candidate = bind_grounding_selection(
            task, private["grounding_selection"], snapshot, scoped.spec, scoped.locale
        )
        require(row.get("candidate") == candidate, "CHECKPOINT_CANDIDATE_CHANGED")
    except (ScraperError, KeyError, TypeError, ValueError):
        row["error_code"] = "CHECKPOINT_GROUNDING_CHANGED"


def score_evaluation(cases, rows, identity, freeze_sha):
    receipt = evaluate.score_outputs(cases, rows, model=identity["model"], inference=identity["inference"])
    business_rows = []
    counts = Counter()
    coverage = Counter()
    stage_usage = {name: Counter() for name in ("business", "grounding")}
    stage_latencies = {name: [] for name in stage_usage}
    catalogs = {}
    offline = GroundedLocalExtractor(
        identity["model"]["id"],
        client=object(),
        max_tokens=identity["inference"]["max_tokens"],
        max_source_chars=identity["inference"]["max_source_chars"],
    )
    by_id = {row["case_id"]: row for row in rows}
    for case in cases:
        row = by_id.get(case["case_id"], {})
        private = row.get("private", {})
        raw = private.get("business")
        request, spec, snapshot, gold = evaluate._context(case)
        expected = gold.data.model_dump()
        negative = evaluate._negative_paths(expected)
        counts["null_controls"] += len(negative)
        counts["raw_json_objects"] += isinstance(raw, dict)
        business_row = {"case_id": case["case_id"], "error_code": "BUSINESS_ONLY_DIAGNOSTIC"}
        if isinstance(raw, dict):
            business_row["candidate"] = {"data": raw, "evidence": [], "warnings": []}
            counts["exact_records"] += evaluate._canonical(raw) == evaluate._canonical(expected)
            counts["correct_null_controls"] += sum(
                evaluate._at(raw, path) in (None, "", []) for path in negative
            )
            counts["grounding_populated_leaves"] += len(populated_fields(raw))
            try:
                validate_business(spec, raw)
                counts["schema_valid_records"] += 1
            except ScraperError:
                pass
        business_rows.append(business_row)
        scoped = offline.for_request(request, spec)
        plan, source = _plan(scoped, snapshot)
        require(plan == identity["inputs"][case["case_id"]], "FROZEN_INPUT_CHANGED")
        anchors = _anchors(case)
        coverage["positive_fields"] += len(case["expected_fields"])
        included = {item["id"]: item["text"] for item in source["sections"]} if source else {}
        coverage["reviewed_fields_visible"] += len(
            {
                item["field_path"]
                for item in anchors
                if item["section_id"] in included and item["excerpt"] in included[item["section_id"]]
            }
        )
        if source and isinstance(raw, dict):
            try:
                task = build_grounding_task(
                    spec, raw, source, request.instruction, max_source_chars=scoped.max_source_chars
                )
            except ScraperError:
                task = None
            if task is not None:
                catalogs[case["case_id"]] = {
                    **task.metadata,
                    "schema_sha256": evaluate._hash(task.schema),
                    "messages_sha256": evaluate._hash(task.messages),
                }
                selections = private.get("grounding_selection")
                selections = selections.get("selections", {}) if isinstance(selections, dict) else {}
                selections = selections if isinstance(selections, dict) else {}
                counts["grounding_catalogued_leaves"] += len(task.candidates)
                evidence = []
                picked = {}
                for path, candidates in task.candidates.items():
                    identifier = selections.get(path)
                    matches = [
                        item for item in candidates if type(identifier) is str and item.id == identifier
                    ]
                    if len(matches) == 1:
                        picked[path] = matches[0]
                        evidence.append(matches[0].as_evidence().model_dump())
                    elif path in selections and identifier is None:
                        counts["grounding_abstentions"] += 1
                    else:
                        counts["grounding_invalid_or_missing_ids"] += 1
                counts["grounding_valid_ids"] += len(picked)
                assessment = evaluate._assess(
                    {"data": task.data, "evidence": evidence, "warnings": []}, snapshot, request, spec
                )
                invalid = {
                    error.split(":", 2)[1]
                    for error in assessment.errors
                    if error.startswith(("evidence:", "identity:", "conflict:"))
                }
                supported = set(picked) - invalid
                if any(error.startswith("ambiguous_entity:") for error in assessment.errors):
                    supported = set()
                counts["grounding_source_valid_leaves"] += len(supported)
                actual_values = populated_fields(task.data)
                counts["reviewed_anchor_agreement"] += sum(
                    any(
                        anchor["field_path"] == path
                        and anchor["section_id"] == item.section_id
                        and anchor["excerpt"] in item.excerpt
                        for anchor in anchors
                    )
                    for path, item in picked.items()
                    if path in supported
                    and path in case["expected_fields"]
                    and type(actual_values[path]) is type(case["expected_fields"][path])
                    and actual_values[path] == case["expected_fields"][path]
                )
        calls = row.get("model_calls", 0)
        counts["model_calls"] += calls if type(calls) is int and 0 <= calls <= 2 else 0
        counts["model_calls_unknown_cases"] += (
            row.get("error_code") == "INFERENCE_INTERRUPTED" and "model_calls" not in row
        )
        counts["maximum_reserved_model_calls"] += row.get("maximum_model_calls_reserved", 0)
        for stage in row.get("stages", []):
            if isinstance(stage, dict) and stage.get("task") in stage_usage:
                target = stage_usage[stage["task"]]
                target["calls"] += 1
                for key in ("input_tokens", "output_tokens"):
                    value = stage.get(key)
                    target[key] += value if type(value) is int and value >= 0 else 0
                target["failed_calls"] += stage.get("status") != "completed"
                elapsed = stage.get("elapsed_ms")
                if type(elapsed) in {int, float} and math.isfinite(elapsed) and elapsed >= 0:
                    stage_latencies[stage["task"]].append(elapsed)
    business_score = evaluate.score_outputs(
        cases, business_rows, model=identity["model"], inference=identity["inference"]
    )
    receipt.update(
        benchmark_format=FORMAT,
        evaluation_mode="frozen_two_stage_local_model_on_reviewed_saved_pages",
        frozen_candidate_sha256=freeze_sha,
        actual_case_file_sha256=identity["cases_sha256"],
        citation_policy_version=GROUNDED_POLICY_VERSION,
        source_view_version=GROUNDED_VIEW_VERSION,
        code_tree_sha256=identity["code"]["tree_sha256"],
        python=identity["python"],
        endpoint_profile_sha256=identity["endpoint_profile_sha256"],
        source_views=identity["inputs"],
        actual_grounding_catalogs=catalogs,
        business_output={
            "valid_json_object_cases": counts["raw_json_objects"],
            "complete_schema_valid_cases": counts["schema_valid_records"],
            "exact_record_cases": counts["exact_records"],
            "null_controls": counts["null_controls"],
            "correct_null_controls": counts["correct_null_controls"],
            "raw_facts": business_score["raw_facts"],
        },
        grounding={
            key: counts["grounding_" + key]
            for key in (
                "populated_leaves",
                "catalogued_leaves",
                "valid_ids",
                "source_valid_leaves",
                "invalid_or_missing_ids",
                "abstentions",
            )
        },
        reviewed_anchor_agreement=counts["reviewed_anchor_agreement"],
        input_gold_coverage=dict(coverage),
        wrong_accepted_facts=receipt["accepted_facts"]["wrong_or_unlabeled"],
        actual_model_calls=counts["model_calls"],
        model_calls_unknown_cases=counts["model_calls_unknown_cases"],
        maximum_reserved_model_calls=counts["maximum_reserved_model_calls"],
        stage_usage={
            name: {
                **{key: value[key] for key in ("calls", "input_tokens", "output_tokens", "failed_calls")},
                "latency": evaluate._latencies(stage_latencies[name]),
            }
            for name, value in stage_usage.items()
        },
        maximum_model_calls_per_case=2,
        model_call_definition="Client-observed local inference submissions; interrupted requests without a completed checkpoint have unknown actual call counts and remain bounded by their two-call reservation",
        repair_attempts=0,
        paid_api_calls=0,
        training_performed=False,
        model_weight_load_verification="external_server_supervisor_required",
    )
    receipt["limitations"] = [
        "This compares our two-stage extractor on saved reviewed pages; it does not test live retrieval or Firecrawl native extraction and establishes no superiority claim.",
        "Reviewed-anchor agreement is a proxy: a different source-valid anchor may also be semantically correct.",
        "Literal source validity does not establish arbitrary custom field meaning; inspected errors and held-out labels remain necessary.",
        "Model identity is supplied by the operator; the serving supervisor must independently verify loaded weights.",
        "No cases or gold facts are removed for window omissions, preflight failures, inference errors, interruption or truncation.",
        "Business-only metrics use unmodified parseable stage-one JSON even when grounding fails; accepted metrics require the complete application gate.",
    ]
    return receipt


async def run_evaluation(
    cases, *, identity, freeze, freeze_sha, checkpoint, base_url="http://127.0.0.1:8003/v1", client=None
):
    verify_freeze(freeze, freeze_sha, identity)
    require(evaluate._hash(cases) == identity["cases_hash"], "FROZEN_CASES_CHANGED")
    require(code_identity() == identity["code"], "FROZEN_CODE_CHANGED")
    require(python_identity() == identity["python"], "FROZEN_PYTHON_CHANGED")
    base_url = validate_local_endpoint(base_url)
    require(
        evaluate._hash({"base_url": base_url}) == identity["endpoint_profile_sha256"],
        "FROZEN_ENDPOINT_CHANGED",
    )
    checkpoint = Path(checkpoint)
    with checkpoint_lock(checkpoint):
        rows = []
        if checkpoint.exists():
            require(not stat.S_IMODE(checkpoint.stat().st_mode) & 0o077, "CHECKPOINT_NOT_PRIVATE")
            saved = evaluate._parse(_read(checkpoint).decode())
            require(
                isinstance(saved, dict)
                and saved.get("format") == CHECKPOINT_FORMAT
                and saved.get("identity") == freeze_sha,
                "CHECKPOINT_PROFILE_CHANGED",
            )
            rows = saved.get("outputs")
            evaluate.score_outputs(cases, rows, model=identity["model"], inference=identity["inference"])
            in_flight = saved.get("in_flight")
            require(
                in_flight is None
                or in_flight in identity["case_order"]
                and in_flight not in {row["case_id"] for row in rows},
                "INVALID_CHECKPOINT_IN_FLIGHT",
            )
            if in_flight is not None:
                rows.append(
                    {
                        "case_id": in_flight,
                        "error_code": "INFERENCE_INTERRUPTED",
                        "maximum_model_calls_reserved": 2,
                    }
                )
                _checkpoint(checkpoint, freeze_sha, rows)
        settings = {
            "base_url": base_url,
            "max_tokens": identity["inference"]["max_tokens"],
            "timeout": identity["timeout_seconds_per_stage"],
            "max_source_chars": identity["inference"]["max_source_chars"],
        }
        offline = GroundedLocalExtractor(identity["model"]["id"], client=object(), **settings)
        indexed = {case["case_id"]: case for case in cases}
        for row in rows:
            request, spec, snapshot, _ = evaluate._context(indexed[row["case_id"]])
            scoped = offline.for_request(request, spec)
            plan, source = _plan(scoped, snapshot)
            require(plan == identity["inputs"][row["case_id"]], "FROZEN_INPUT_CHANGED")
            if plan.get("error_code"):
                row["error_code"] = plan["error_code"]
            else:
                _regate(row, scoped, snapshot, source)
        if rows:
            _checkpoint(checkpoint, freeze_sha, rows)
        completed = {row["case_id"] for row in rows}
        # All freeze/profile/source checks above precede creation of a live client.
        extractor = GroundedLocalExtractor(identity["model"]["id"], client=client, **settings)
        deadline = time.monotonic() + identity["wall_seconds"]
        try:
            for case in cases:
                case_id = case["case_id"]
                if case_id in completed:
                    continue
                row = {"case_id": case_id, "model_calls": 0, "maximum_model_calls_reserved": 0}
                request, spec, snapshot, _ = evaluate._context(case)
                scoped = extractor.for_request(request, spec)
                plan, source = _plan(scoped, snapshot)
                require(plan == identity["inputs"][case_id], "FROZEN_INPUT_CHANGED")
                begin = time.monotonic()
                remaining = deadline - begin
                if remaining <= 0:
                    row["error_code"] = "BENCHMARK_DEADLINE"
                elif plan.get("error_code"):
                    row["error_code"] = plan["error_code"]
                else:
                    row["maximum_model_calls_reserved"] = 2
                    _checkpoint(checkpoint, freeze_sha, rows, case_id)
                    latencies = _time_stages(scoped)
                    try:
                        async with asyncio.timeout(remaining):
                            result = await scoped.extract(snapshot)
                        row.update(candidate=result.payload, finish_reason="stop")
                    except TimeoutError:
                        row["error_code"] = "BENCHMARK_DEADLINE"
                    except ScraperError as exc:
                        row["error_code"] = (
                            exc.error.code if _CODE.fullmatch(exc.error.code) else "LOCAL_EVALUATION_FAILED"
                        )
                    except Exception:  # noqa: BLE001 -- never save third-party exception strings
                        row["error_code"] = "LOCAL_EVALUATION_FAILED"
                    except (BenchmarkDeadline, KeyboardInterrupt, asyncio.CancelledError):
                        row["error_code"] = "INFERENCE_INTERRUPTED"
                        _capture(row, scoped, latencies)
                        row["elapsed_ms"] = (time.monotonic() - begin) * 1000
                        rows.append(row)
                        _checkpoint(checkpoint, freeze_sha, rows)
                        raise
                    _capture(row, scoped, latencies)
                row["elapsed_ms"] = (time.monotonic() - begin) * 1000
                rows.append(row)
                _checkpoint(checkpoint, freeze_sha, rows)
        finally:
            await extractor.close()
    return rows, score_evaluation(cases, rows, identity, freeze_sha)


def _execute(args):
    cases, cases_sha = _cases(args.cases)
    identity = freeze_identity(
        cases,
        cases_sha256=cases_sha,
        model={"id": args.model, "revision": args.revision, "weights_sha256": args.weights_sha256},
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
        max_source_chars=args.max_source_chars,
        wall_seconds=args.wall_seconds,
    )
    if args.freeze_candidate:
        sha = write_freeze(args.freeze_candidate, identity)
        print(evaluate._canonical({"freeze_sha256": sha, "case_count": len(cases), "model_calls": 0}))
        return 0
    require(args.output_dir is not None, "OUTPUT_DIRECTORY_REQUIRED")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    require(
        not output.is_symlink() and not stat.S_IMODE(output.stat().st_mode) & 0o077,
        "OUTPUT_DIRECTORY_NOT_PRIVATE",
    )
    checkpoint, metrics = output / "raw-checkpoint.json", output / "metrics.json"
    selected = [args.cases.resolve(), args.freeze.resolve(), checkpoint.resolve(), metrics.resolve()]
    require(len(set(selected)) == len(selected), "OVERLAPPING_PATHS")
    verify_freeze(args.freeze, args.freeze_sha, identity)
    _, receipt = asyncio.run(
        run_evaluation(
            cases,
            identity=identity,
            freeze=args.freeze,
            freeze_sha=args.freeze_sha,
            checkpoint=checkpoint,
            base_url=args.base_url,
        )
    )
    atomic_json(metrics, receipt)
    print(
        evaluate._canonical(
            {
                "case_count": receipt["case_count"],
                "accepted_cases": receipt["accepted_cases"],
                "actual_model_calls": receipt["actual_model_calls"],
            }
        )
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--max-source-chars", type=int, default=8000)
    parser.add_argument("--wall-seconds", type=float, default=1800)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-candidate", type=Path)
    mode.add_argument("--freeze", type=Path)
    parser.add_argument("--freeze-sha")
    parser.add_argument("--output-dir", "--outputdir", dest="output_dir", type=Path)
    args = parser.parse_args(argv)
    try:
        require(
            math.isfinite(args.wall_seconds) and 0 < args.wall_seconds <= MAX_WALL_SECONDS,
            "INVALID_WALL_LIMIT",
        )
        os.umask(0o077)
        with hard_deadline(args.wall_seconds):
            return _execute(args)
    except BenchmarkDeadline:
        print("BENCHMARK_DEADLINE: Frozen evaluation exceeded its wall limit", file=sys.stderr)
    except (OSError, ValueError, TypeError, KeyError, ValidationError, ScraperError):
        print("INVALID_EVALUATION: Invalid frozen input or unavailable private state", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
