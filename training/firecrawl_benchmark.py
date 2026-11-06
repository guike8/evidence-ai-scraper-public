"""Frozen, resumable local-model baselines over saved Firecrawl sources.

This measures our extraction contract on Firecrawl-prepared content. It does
not run Firecrawl's native JSON extraction pipeline, retrieve URLs or train.
Raw candidates stay in a private checkpoint; public receipts contain counts,
trusted field names and hashes. Every case remains in the denominator.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import signal
import stat
import sys
import time
from pathlib import Path

from pydantic import ValidationError

import ai_scraper
from ai_scraper.contracts import ScraperError
from ai_scraper.extraction.local import validate_local_endpoint
from ai_scraper.storage import atomic_json
from ai_scraper.validate import _pointer

from . import evaluate
from .firecrawl_views import FirecrawlViewExtractor

FORMAT = "firecrawl-model-baseline-v1"
FREEZE_FORMAT = "firecrawl-model-baseline-freeze-v1"
CHECKPOINT_FORMAT = "firecrawl-model-baseline-checkpoint-v1"
MAX_WALL_SECONDS = 1200
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_CODE = re.compile(r"[A-Z][A-Z0-9_]{0,79}\Z")


class BenchmarkError(ValueError):
    """Stable identifiers, never third-party paths or error messages."""


class BenchmarkDeadline(BaseException):
    """The CLI's hard deadline must not be swallowed by transport handlers."""


def require(condition, code):
    if not condition:
        raise BenchmarkError(code)


def _sha(body):
    return hashlib.sha256(body).hexdigest()


def _read(path: Path, limit=evaluate.MAX_FILE_BYTES):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and 0 < before.st_size <= limit, "INVALID_BOUNDED_FILE")
        body = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        current = path.lstat()
    identity = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )
    require(
        len(body) == before.st_size and identity(before) == identity(after) == identity(current),
        "INPUT_CHANGED",
    )
    return body


def _cases(path):
    body = _read(Path(path))
    cases = []
    for line in body.splitlines():
        if not line.strip():
            continue
        require(len(line) <= evaluate.MAX_ROW_BYTES and len(cases) < evaluate.MAX_ROWS, "CASE_LIMIT")
        row = evaluate._parse(line.decode("utf-8"))
        require(isinstance(row, dict), "INVALID_CASE")
        cases.append(row)
    evaluate.validate_cases(cases)
    return cases, _sha(body)


def code_identity():
    """Bind the actual application and training Python source, without paths."""
    files = {}
    for prefix, directory, recursive in (
        ("ai_scraper", Path(ai_scraper.__file__).parent, True),
        ("training", Path(__file__).parent, False),
    ):
        selected = directory.rglob("*.py") if recursive else directory.glob("*.py")
        for path in sorted(selected):
            require(len(files) < 5000, "CODE_LIMIT")
            files[prefix + "/" + path.relative_to(directory).as_posix()] = _sha(_read(path, 10_000_000))
    return {"tree_sha256": evaluate._hash(files), "files": files}


def _settings(model, *, base_url, source_policy, max_tokens, timeout, max_source_chars, wall_seconds):
    base_url = validate_local_endpoint(base_url)
    require(source_policy in {"full_record", "markdown"}, "INVALID_SOURCE_POLICY")
    require(
        type(timeout) in {int, float} and math.isfinite(timeout) and 0 < timeout <= 300,
        "INVALID_TIMEOUT",
    )
    require(
        type(wall_seconds) in {int, float}
        and math.isfinite(wall_seconds)
        and 0 < wall_seconds <= MAX_WALL_SECONDS,
        "INVALID_WALL_LIMIT",
    )
    require(
        isinstance(model, dict)
        and isinstance(model.get("weights_sha256"), str)
        and bool(_SHA.fullmatch(model["weights_sha256"])),
        "INVALID_MODEL_SHA",
    )
    inference = {
        "temperature": 0,
        "max_tokens": max_tokens,
        "enable_thinking": False,
        "response_format": "json_schema",
        "repair_attempts": 0,
        "max_source_chars": max_source_chars,
    }
    model, inference = evaluate._model_metadata(model, inference)
    require(type(max_source_chars) is int and 0 < max_source_chars <= 1_000_000, "INVALID_SOURCE_LIMIT")
    return model, inference, base_url


def _input_plan(scoped, snapshot):
    """Do not retain source text in a freeze or public receipt."""
    try:
        content = scoped._source_content(snapshot)
    except ScraperError as error:
        require(bool(_CODE.fullmatch(error.error.code)), "INVALID_ERROR_CODE")
        descriptor = getattr(scoped, "last_source_view", None) or {}
        return {
            "error_code": error.error.code,
            "schema_fingerprint": scoped.schema_fingerprint,
            "source_hash": snapshot.source_hash,
            "system_instruction_sha256": _sha(scoped.system_instruction.encode()),
            "source_descriptor": {
                key: descriptor[key]
                for key in (
                    "selection_version",
                    "selection_hash",
                    "view_hash",
                    "scope",
                    "coverage",
                    "original_chars",
                    "view_chars",
                    "included_section_ids",
                    "excluded_section_ids",
                )
                if key in descriptor
            },
        }, None
    payload = evaluate._parse(content)
    sections = payload.get("sections")
    ld_json = payload.get("ld_json")
    require(isinstance(sections, list) and isinstance(ld_json, list), "INVALID_SOURCE_VIEW")
    ids = [section.get("id") for section in sections if isinstance(section, dict)]
    require(
        len(ids) == len(sections)
        and all(isinstance(value, str) for value in ids)
        and len(ids) == len(set(ids)),
        "INVALID_SOURCE_VIEW",
    )
    original = [section.id for section in snapshot.sections]
    require(set(ids) <= set(original), "INVALID_SOURCE_VIEW")
    descriptor = getattr(scoped, "last_source_view", None) or {}
    plan = {
        "source_content_sha256": _sha(content.encode()),
        "source_chars": len(content),
        "schema_fingerprint": scoped.schema_fingerprint,
        "system_instruction_sha256": _sha(scoped.system_instruction.encode()),
        "source_hash": snapshot.source_hash,
        "included_section_ids": ids,
        "excluded_section_ids": [value for value in original if value not in set(ids)],
        "upstream_excluded_section_ids": list(snapshot.excluded_sections),
        "input_ld_json_sha256": evaluate._hash(ld_json),
        "input_ld_json_empty": not ld_json,
        "source_descriptor": {
            key: descriptor[key]
            for key in (
                "selection_version",
                "selection_hash",
                "view_hash",
                "scope",
                "coverage",
                "original_chars",
                "view_chars",
                "json_ld_excluded",
            )
            if key in descriptor
        },
    }
    return plan, payload


def freeze_identity(
    cases,
    *,
    cases_sha256,
    model,
    base_url,
    source_policy,
    max_tokens=4096,
    timeout=45,
    max_source_chars=48000,
    wall_seconds=1200,
):
    """Offline freeze identity; the supplied dummy client makes no HTTP client."""
    evaluate.validate_cases(cases)
    require(bool(_SHA.fullmatch(cases_sha256)), "INVALID_CASES_SHA")
    model, inference, base_url = _settings(
        model,
        base_url=base_url,
        source_policy=source_policy,
        max_tokens=max_tokens,
        timeout=timeout,
        max_source_chars=max_source_chars,
        wall_seconds=wall_seconds,
    )
    extractor = FirecrawlViewExtractor(
        model["id"],
        base_url=base_url,
        source_policy=source_policy,
        max_tokens=max_tokens,
        timeout=timeout,
        max_source_chars=max_source_chars,
        client=object(),
    )
    inputs = {}
    for case in cases:
        request, spec, snapshot, _ = evaluate._context(case)
        require(
            snapshot.retrieval_method in {"firecrawl_self_hosted", "firecrawl_preparation_replay"},
            "FIRECRAWL_SOURCE_REQUIRED",
        )
        inputs[case["case_id"]] = _input_plan(extractor.for_request(request, spec), snapshot)[0]
    return {
        "format": FREEZE_FORMAT,
        "cases_sha256": cases_sha256,
        "cases_hash": evaluate._hash(cases),
        "case_count": len(cases),
        "case_order": [case["case_id"] for case in cases],
        "source_retrieval_methods": sorted({case["snapshot"]["retrieval_method"] for case in cases}),
        "source_preparation_kinds": {
            case["case_id"]: evaluate._hash(case["snapshot"]["raw_record"].get("source_preparation"))
            for case in cases
        },
        "model": model,
        "inference": inference,
        "timeout_seconds": timeout,
        "wall_seconds": wall_seconds,
        "source_policy": source_policy,
        "endpoint_profile_sha256": evaluate._hash({"base_url": base_url}),
        "evidence_gate": "actual-input-locators-v1",
        "source_bound_evidence": False,
        "source_quote_evidence": False,
        "code": code_identity(),
        "inputs": inputs,
    }


def verify_freeze(path, expected_sha, actual):
    require(isinstance(expected_sha, str) and bool(_SHA.fullmatch(expected_sha)), "INVALID_FREEZE_SHA")
    body = _read(Path(path), 10_000_000)
    require(_sha(body) == expected_sha, "FREEZE_CHANGED")
    require(evaluate._parse(body.decode()) == actual, "FROZEN_PROFILE_CHANGED")
    return expected_sha


def write_freeze(path, identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    body = (json.dumps(identity, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb"
    ) as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    return _sha(body)


def evidence_outside_input(candidate, source):
    """Occurrence in excluded raw content cannot pass this model-input gate."""
    if not isinstance(candidate, dict) or not isinstance(candidate.get("evidence"), list):
        return False  # Rich schema validation handles malformed evidence.
    included = {section["id"] for section in source["sections"]}
    for item in candidate["evidence"]:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("section_id"), str) and item["section_id"] not in included:
            return True
        if isinstance(item.get("json_pointer"), str):
            try:
                _pointer(source["ld_json"], item["json_pointer"])
            except (ValueError, KeyError, IndexError, TypeError):
                return True
    return False


@contextlib.contextmanager
def checkpoint_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    info = path.parent.stat()
    require(
        info.st_uid == os.getuid() and not stat.S_IMODE(info.st_mode) & 0o077,
        "CHECKPOINT_DIRECTORY_NOT_PRIVATE",
    )
    descriptor = os.open(path.with_name(path.name + ".lock"), os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+") as stream:
        require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), "INVALID_CHECKPOINT_LOCK")
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise BenchmarkError("BENCHMARK_BUSY") from None
        yield


def _checkpoint(path, freeze_sha, rows, in_flight=None):
    require(not Path(path).is_symlink(), "INVALID_CHECKPOINT")
    atomic_json(
        Path(path),
        {"format": CHECKPOINT_FORMAT, "identity": freeze_sha, "outputs": rows, "in_flight": in_flight},
    )


def _raw_completion(envelope):
    """Keep failed adapter completions private for independent raw scoring."""
    try:
        content = envelope["choices"][0]["message"]["content"]
        if isinstance(content, str) and len(content.encode()) <= evaluate.MAX_ROW_BYTES:
            return content
    except (KeyError, IndexError, TypeError):
        pass
    return None


def _capture_completion(original, captured):
    async def completion(**kwargs):
        envelope = await original(**kwargs)
        captured["envelope"] = envelope
        return envelope

    return completion


def _regate_saved(cases, rows, extractor, identity):
    """A resumed checkpoint cannot bypass the same actual-input gate."""
    by_id = {case["case_id"]: case for case in cases}
    for row in rows:
        request, spec, snapshot, _ = evaluate._context(by_id[row["case_id"]])
        plan, source = _input_plan(extractor.for_request(request, spec), snapshot)
        require(plan == identity["inputs"][row["case_id"]], "FROZEN_INPUT_CHANGED")
        if plan.get("error_code"):
            row["error_code"] = plan["error_code"]
            continue
        candidate = row.get("candidate")
        if candidate is None and "raw_output" in row:
            try:
                candidate = evaluate._parse(row["raw_output"])
            except (ValueError, TypeError, RecursionError):
                pass
        if evidence_outside_input(candidate, source):
            row["error_code"] = row.get("error_code") or "EVIDENCE_OUTSIDE_INPUT"


async def run_baseline(
    cases, *, identity, freeze, freeze_sha, checkpoint, base_url="http://127.0.0.1:8003/v1", client=None
):
    """Verify freeze first; checkpoint completed and interrupted requests exactly once."""
    verify_freeze(freeze, freeze_sha, identity)
    require(evaluate._hash(cases) == identity["cases_hash"], "FROZEN_CASES_CHANGED")
    require(code_identity() == identity["code"], "FROZEN_CODE_CHANGED")
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
                or (
                    in_flight in identity["case_order"] and in_flight not in {row["case_id"] for row in rows}
                ),
                "INVALID_CHECKPOINT_IN_FLIGHT",
            )
            if in_flight is not None:
                rows.append({"case_id": in_flight, "error_code": "INFERENCE_INTERRUPTED"})
                _checkpoint(checkpoint, freeze_sha, rows)
        inference = identity["inference"]
        offline = FirecrawlViewExtractor(
            identity["model"]["id"],
            base_url=base_url,
            source_policy=identity["source_policy"],
            max_tokens=inference["max_tokens"],
            timeout=identity["timeout_seconds"],
            max_source_chars=inference["max_source_chars"],
            client=object(),
        )
        _regate_saved(cases, rows, offline, identity)
        completed = {row["case_id"] for row in rows}
        extractor = FirecrawlViewExtractor(
            identity["model"]["id"],
            base_url=base_url,
            source_policy=identity["source_policy"],
            max_tokens=inference["max_tokens"],
            timeout=identity["timeout_seconds"],
            max_source_chars=inference["max_source_chars"],
            client=client,
        )
        deadline = time.monotonic() + identity["wall_seconds"]
        try:
            for case in cases:
                case_id = case["case_id"]
                if case_id in completed:
                    continue
                row = {"case_id": case_id}
                request, spec, snapshot, _ = evaluate._context(case)
                scoped = extractor.for_request(request, spec)
                plan, source = _input_plan(scoped, snapshot)
                require(plan == identity["inputs"][case_id], "FROZEN_INPUT_CHANGED")
                begin = time.monotonic()
                remaining = deadline - begin
                if remaining <= 0:
                    row["error_code"] = "BENCHMARK_DEADLINE"
                elif plan.get("error_code"):
                    row["error_code"] = plan["error_code"]
                else:
                    _checkpoint(checkpoint, freeze_sha, rows, case_id)
                    captured = {}
                    scoped._local_client.chat_completion = _capture_completion(
                        scoped._local_client.chat_completion, captured
                    )
                    try:
                        async with asyncio.timeout(remaining):
                            response = await scoped.extract(snapshot)
                        row.update(candidate=response.payload, finish_reason="stop")
                        if evidence_outside_input(response.payload, source):
                            row["error_code"] = "EVIDENCE_OUTSIDE_INPUT"
                    except TimeoutError:
                        row["error_code"] = "BENCHMARK_DEADLINE"
                    except ScraperError as error:
                        row["error_code"] = (
                            error.error.code
                            if _CODE.fullmatch(error.error.code)
                            else "LOCAL_EVALUATION_FAILED"
                        )
                    except Exception:  # noqa: BLE001 - no third-party exception content.
                        row["error_code"] = "LOCAL_EVALUATION_FAILED"
                    except (BenchmarkDeadline, KeyboardInterrupt, asyncio.CancelledError):
                        row["error_code"] = "INFERENCE_INTERRUPTED"
                        rows.append(row)
                        _checkpoint(checkpoint, freeze_sha, rows)
                        raise
                    if "candidate" not in row:
                        raw = _raw_completion(captured.get("envelope"))
                        if raw is not None:
                            row["raw_output"] = raw
                row["elapsed_ms"] = (time.monotonic() - begin) * 1000
                for key in ("input_tokens", "output_tokens"):
                    value = getattr(scoped, "last_usage", {}).get(key)
                    row[key] = value if type(value) is int and value >= 0 else None
                rows.append(row)
                _checkpoint(checkpoint, freeze_sha, rows)
        finally:
            await extractor.close()
    receipt = evaluate.score_outputs(cases, rows, model=identity["model"], inference=inference)
    receipt.update(
        benchmark_format=FORMAT,
        evaluation_mode="our_local_extraction_on_saved_firecrawl_sources",
        frozen_candidate_sha256=freeze_sha,
        actual_case_file_sha256=identity["cases_sha256"],
        source_policy=identity["source_policy"],
        source_retrieval_methods=identity["source_retrieval_methods"],
        source_preparation_metadata_sha256=identity["source_preparation_kinds"],
        runtime_track=(
            "firecrawl_preparation_component_replay"
            if identity["source_retrieval_methods"] == ["firecrawl_preparation_replay"]
            else "saved_self_hosted_or_mixed_firecrawl_sources"
        ),
        source_views=identity["inputs"],
        code_tree_sha256=identity["code"]["tree_sha256"],
        endpoint_profile_sha256=identity["endpoint_profile_sha256"],
        timeout_seconds=identity["timeout_seconds"],
        wall_seconds=identity["wall_seconds"],
        source_bound_evidence=False,
        source_quote_evidence=False,
        training_performed=False,
        paid_api_calls=0,
        evidence_gate=identity["evidence_gate"],
        input_binding_failure_cases=sum(row.get("error_code") == "EVIDENCE_OUTSIDE_INPUT" for row in rows),
        model_weight_load_verification="external_server_supervisor_required",
    )
    receipt["limitations"] = [
        "This is our extraction contract on Firecrawl-prepared saved sources, not Firecrawl native JSON extraction or a superiority comparison.",
        "The firecrawl_preparation_replay method measures only the actual HTML-to-Markdown preparation component on previously saved HTML; it does not establish Firecrawl API retrieval, browser or asynchronous job behavior.",
        "Source-policy changes are input ablations; keep model, schema and decoding controls fixed when attributing improvements.",
        "Actual input locator membership is an extra accepted-output gate; raw candidates remain available to the independent scorer.",
        "Raw source-support diagnostics consult the complete saved snapshot; accepted output additionally requires references present in the actual model input.",
        "Model identity is supplied by the operator; the serving supervisor must independently verify loaded weights.",
        "No cases are removed for source, schema, inference, timeout or evidence failures.",
        *receipt["limitations"],
    ]
    return rows, receipt


@contextlib.contextmanager
def hard_deadline(seconds):
    def expired(*_):
        raise BenchmarkDeadline()

    previous = signal.signal(signal.SIGALRM, expired)
    prior_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *prior_timer)
        signal.signal(signal.SIGALRM, previous)


def _execute(args):
    cases, cases_sha = _cases(args.cases)
    identity = freeze_identity(
        cases,
        cases_sha256=cases_sha,
        model={"id": args.model, "revision": args.revision, "weights_sha256": args.weights_sha256},
        base_url=args.base_url,
        source_policy=args.source_policy,
        max_tokens=args.max_tokens,
        max_source_chars=args.max_source_chars,
        timeout=args.timeout,
        wall_seconds=args.wall_seconds,
    )
    if args.freeze_candidate:
        digest = write_freeze(args.freeze_candidate, identity)
        print(json.dumps({"freeze_sha256": digest, "case_count": len(cases), "model_calls": 0}))
    else:
        require(args.checkpoint is not None and args.output is not None, "CHECKPOINT_AND_OUTPUT_REQUIRED")
        selected = [
            args.cases.resolve(),
            args.freeze.resolve(),
            args.checkpoint.resolve(),
            args.output.resolve(),
        ]
        require(len(set(selected)) == len(selected), "OVERLAPPING_PATHS")
        verify_freeze(args.freeze, args.freeze_sha, identity)
        _, receipt = asyncio.run(
            run_baseline(
                cases,
                identity=identity,
                freeze=args.freeze,
                freeze_sha=args.freeze_sha,
                checkpoint=args.checkpoint,
                base_url=args.base_url,
            )
        )
        atomic_json(args.output, receipt)
        print(json.dumps({"case_count": receipt["case_count"], "accepted_cases": receipt["accepted_cases"]}))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--weights-sha256", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8003/v1")
    parser.add_argument("--source-policy", choices=("full_record", "markdown"), required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-source-chars", type=int, default=48000)
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--wall-seconds", type=float, default=1200)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--freeze-candidate", type=Path)
    mode.add_argument("--freeze", type=Path)
    parser.add_argument("--freeze-sha")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        require(
            math.isfinite(args.wall_seconds) and 0 < args.wall_seconds <= MAX_WALL_SECONDS,
            "INVALID_WALL_LIMIT",
        )
        with hard_deadline(args.wall_seconds):
            return _execute(args)
    except BenchmarkDeadline:
        print("BENCHMARK_DEADLINE: Frozen benchmark exceeded its wall limit", file=sys.stderr)
    except (OSError, ValueError, TypeError, KeyError, ValidationError, ScraperError):
        print("INVALID_BENCHMARK: Invalid frozen input or unavailable private state", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
