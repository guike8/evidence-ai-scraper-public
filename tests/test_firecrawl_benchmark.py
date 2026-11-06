"""Freeze before inference, preserve every failure and bind actual input refs."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os

import httpx
import pytest

from ai_scraper.extraction.generic import populated_fields
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest
from ai_scraper.storage import atomic_json
from training import evaluate
from training import firecrawl_benchmark as bench

MODEL = {"id": "example/local-model", "revision": "test-revision", "weights_sha256": "a" * 64}
ENDPOINT = "http://127.0.0.1:8003/v1"


def case(index=0, method="firecrawl_self_hosted"):
    request = ExtractionRequest(
        url=f"https://public.test/record-{index}",
        schema_id="custom_v1",
        fields=[
            {"name": "doc_title", "required": True, "critical": True},
            {"name": "count", "type": "integer", "required": True},
            {"name": "missing_contact", "required": False},
        ],
    )
    snapshot = normalize_record(
        {
            "final_url": request.url,
            "page_html": "<h1>Ledger entry</h1><p>Count: 17</p><p>HTML only</p>",
            "markdown": "# Ledger entry\n\nCount: 17\n",
            "ld_json": [{"name": "Ledger entry", "count": 17}],
        },
        request,
        retrieval_method=method,
    )
    sections = [section for section in snapshot.sections if section.representation == "markdown"]
    candidate = {
        "data": {"doc_title": "Ledger entry", "count": 17, "missing_contact": None},
        "evidence": [
            {
                "field_path": "doc_title",
                "section_id": sections[0].id,
                "excerpt": "Ledger entry",
                "json_pointer": None,
            },
            {"field_path": "count", "section_id": sections[1].id, "excerpt": "17", "json_pointer": None},
        ],
        "warnings": [],
    }
    spec = get_schema(request)
    value = {
        "case_id": f"firecrawl-dev-{index}",
        "schema_id": spec.schema_id,
        "schema_fingerprint": spec.fingerprint,
        "request": request.model_dump(mode="json"),
        "snapshot": snapshot.model_dump(mode="json"),
        "source_hash": snapshot.source_hash,
        "expected_candidate": candidate,
        "expected_fields": populated_fields(candidate["data"]),
        "expected_status": "success",
    }
    evaluate.validate_cases([value])
    return value


def frozen(tmp_path, cases, policy="markdown", **options):
    identity = bench.freeze_identity(
        cases,
        cases_sha256="b" * 64,
        model=MODEL,
        base_url=ENDPOINT,
        source_policy=policy,
        **options,
    )
    freeze = tmp_path / "freeze.json"
    sha = bench.write_freeze(freeze, identity)
    return identity, freeze, sha


def response(candidate):
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": json.dumps(candidate)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 50},
        },
    )


async def run(tmp_path, cases, *, identity, freeze, sha, handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await bench.run_baseline(
            cases,
            identity=identity,
            freeze=freeze,
            freeze_sha=sha,
            checkpoint=tmp_path / "raw.json",
            base_url=ENDPOINT,
            client=client,
        )


def test_freeze_is_offline_and_binds_actual_inputs_without_source_text(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: pytest.fail("HTTP client created during freeze"))
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    assert bench.verify_freeze(freeze, sha, identity) == sha
    assert identity["source_policy"] == "markdown"
    plan = identity["inputs"][cases[0]["case_id"]]
    assert plan["input_ld_json_empty"]
    assert plan["excluded_section_ids"]
    assert identity["code"]["files"]["training/firecrawl_views.py"]
    text = freeze.read_text()
    assert "Ledger entry" not in text and str(tmp_path) not in text and ENDPOINT not in text
    with pytest.raises(FileExistsError):
        bench.write_freeze(freeze, identity)


@pytest.mark.parametrize(
    "change", ["source_policy", "model", "endpoint_profile_sha256", "code", "cases_sha256"]
)
@pytest.mark.asyncio
async def test_frozen_profile_mismatch_rejected_before_model(tmp_path, change):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    changed = copy.deepcopy(identity)
    changed[change] = "changed"
    with pytest.raises(bench.BenchmarkError, match="FROZEN_PROFILE_CHANGED"):
        await run(
            tmp_path,
            cases,
            identity=changed,
            freeze=freeze,
            sha=sha,
            handler=lambda _: pytest.fail("model called"),
        )


@pytest.mark.asyncio
async def test_freeze_checksum_mismatch_rejected_before_checkpoint(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    freeze.write_text(freeze.read_text() + " ")
    with pytest.raises(bench.BenchmarkError, match="FREEZE_CHANGED"):
        await run(
            tmp_path,
            cases,
            identity=identity,
            freeze=freeze,
            sha=sha,
            handler=lambda _: pytest.fail("model called"),
        )
    assert not (tmp_path / "raw.json").exists()


@pytest.mark.asyncio
async def test_completed_cases_resume_without_duplicate_requests(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return response(cases[0]["expected_candidate"])

    rows, receipt = await run(tmp_path, cases, identity=identity, freeze=freeze, sha=sha, handler=handler)
    assert len(requests) == 1 and receipt["accepted_cases"] == 1
    assert rows[0]["candidate"] == cases[0]["expected_candidate"]
    assert requests[0]["tool_choice"] == "none"
    assert requests[0]["chat_template_kwargs"] == {"enable_thinking": False}
    _, resumed = await run(
        tmp_path,
        cases,
        identity=identity,
        freeze=freeze,
        sha=sha,
        handler=lambda _: pytest.fail("duplicate model call"),
    )
    assert resumed["accepted_cases"] == 1
    assert resumed["source_quote_evidence"] is False and resumed["source_bound_evidence"] is False
    assert resumed["evaluation_mode"] == "our_local_extraction_on_saved_firecrawl_sources"
    assert not os.stat(tmp_path / "raw.json").st_mode & 0o077


@pytest.mark.asyncio
async def test_interrupted_inflight_resume_becomes_failure_without_resubmission(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    atomic_json(
        tmp_path / "raw.json",
        {"format": bench.CHECKPOINT_FORMAT, "identity": sha, "outputs": [], "in_flight": cases[0]["case_id"]},
    )
    rows, receipt = await run(
        tmp_path,
        cases,
        identity=identity,
        freeze=freeze,
        sha=sha,
        handler=lambda _: pytest.fail("ambiguous inference repeated"),
    )
    assert rows == [{"case_id": cases[0]["case_id"], "error_code": "INFERENCE_INTERRUPTED"}]
    assert receipt["case_count"] == 1 and receipt["accepted_cases"] == 0
    assert receipt["raw_facts"]["present"] == 2


@pytest.mark.parametrize("reference", ["excluded_html", "json_ld"])
@pytest.mark.asyncio
async def test_excluded_input_reference_rejects_acceptance_but_keeps_raw_facts(tmp_path, reference):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    candidate = copy.deepcopy(cases[0]["expected_candidate"])
    anchor = candidate["evidence"][0]
    if reference == "excluded_html":
        anchor["section_id"] = next(
            s["id"] for s in cases[0]["snapshot"]["sections"] if s["representation"] == "html_text_block"
        )
    else:
        anchor.update(section_id=None, json_pointer="/0/name")
    rows, receipt = await run(
        tmp_path, cases, identity=identity, freeze=freeze, sha=sha, handler=lambda _: response(candidate)
    )
    assert rows[0]["error_code"] == "EVIDENCE_OUTSIDE_INPUT"
    assert rows[0]["candidate"] == candidate
    assert receipt["raw_facts"]["correct"] == 2 and receipt["accepted_facts"]["correct"] == 0


@pytest.mark.asyncio
async def test_full_record_policy_uses_same_actual_input_membership_gate(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases, "full_record")
    candidate = copy.deepcopy(cases[0]["expected_candidate"])
    candidate["evidence"][0]["section_id"] = "s9999"
    rows, receipt = await run(
        tmp_path, cases, identity=identity, freeze=freeze, sha=sha, handler=lambda _: response(candidate)
    )
    assert rows[0]["error_code"] == "EVIDENCE_OUTSIDE_INPUT"
    assert receipt["raw_facts"]["correct"] == 2


@pytest.mark.asyncio
async def test_oversized_input_is_frozen_and_counted_without_model_call(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases, max_source_chars=10)
    assert identity["inputs"][cases[0]["case_id"]]["error_code"] == "CONTENT_TOO_LARGE"
    assert identity["inputs"][cases[0]["case_id"]]["source_descriptor"]["view_chars"] > 10
    rows, receipt = await run(
        tmp_path,
        cases,
        identity=identity,
        freeze=freeze,
        sha=sha,
        handler=lambda _: pytest.fail("oversized source reached model"),
    )
    assert rows[0]["error_code"] == "CONTENT_TOO_LARGE"
    assert receipt["case_count"] == 1 and receipt["raw_facts"]["present"] == 2


@pytest.mark.asyncio
async def test_checkpoint_from_another_profile_is_rejected(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    atomic_json(
        tmp_path / "raw.json",
        {"format": bench.CHECKPOINT_FORMAT, "identity": "c" * 64, "outputs": [], "in_flight": None},
    )
    with pytest.raises(bench.BenchmarkError, match="CHECKPOINT_PROFILE_CHANGED"):
        await run(
            tmp_path,
            cases,
            identity=identity,
            freeze=freeze,
            sha=sha,
            handler=lambda _: pytest.fail("model called"),
        )


@pytest.mark.asyncio
async def test_duplicate_checkpoint_outcomes_are_rejected(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    row = {"case_id": cases[0]["case_id"], "error_code": "INFERENCE_TIMEOUT"}
    atomic_json(
        tmp_path / "raw.json",
        {"format": bench.CHECKPOINT_FORMAT, "identity": sha, "outputs": [row, row], "in_flight": None},
    )
    with pytest.raises(ValueError, match="duplicate"):
        await run(
            tmp_path,
            cases,
            identity=identity,
            freeze=freeze,
            sha=sha,
            handler=lambda _: pytest.fail("model called"),
        )


def test_private_checkpoint_lock_excludes_competing_runner(tmp_path):
    checkpoint = tmp_path / "raw.json"
    with (
        bench.checkpoint_lock(checkpoint),
        pytest.raises(bench.BenchmarkError, match="BENCHMARK_BUSY"),
        bench.checkpoint_lock(checkpoint),
    ):
        pytest.fail("second runner acquired the lock")


def test_symlink_and_fifo_inputs_rejected_without_blocking(tmp_path):
    regular = tmp_path / "regular"
    regular.write_text("{}")
    symlink = tmp_path / "symlink"
    symlink.symlink_to(regular)
    with pytest.raises(OSError):
        bench._read(symlink)
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(bench.BenchmarkError, match="INVALID_BOUNDED_FILE"):
        bench._read(fifo)


def test_cli_freeze_has_no_model_calls_and_run_requires_its_checksum(tmp_path, capsys):
    cases = [case()]
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(evaluate._canonical(value) for value in cases) + "\n")
    freeze = tmp_path / "freeze.json"
    common = [
        str(path),
        "--model",
        MODEL["id"],
        "--revision",
        MODEL["revision"],
        "--weights-sha256",
        MODEL["weights_sha256"],
        "--source-policy",
        "markdown",
    ]
    assert bench.main([*common, "--freeze-candidate", str(freeze)]) == 0
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["model_calls"] == 0
    assert metadata["freeze_sha256"] == hashlib.sha256(freeze.read_bytes()).hexdigest()
    assert (
        bench.main(
            [
                *common,
                "--freeze",
                str(freeze),
                "--checkpoint",
                str(tmp_path / "raw.json"),
                "--output",
                str(tmp_path / "receipt.json"),
            ]
        )
        == 1
    )
    assert not (tmp_path / "raw.json").exists()


@pytest.mark.asyncio
async def test_actual_case_changes_cannot_be_hidden_by_supplied_identity(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    changed = copy.deepcopy(cases)
    changed[0]["expected_status"] = "partial"
    with pytest.raises(bench.BenchmarkError, match="FROZEN_CASES_CHANGED"):
        await run(
            tmp_path,
            changed,
            identity=identity,
            freeze=freeze,
            sha=sha,
            handler=lambda _: pytest.fail("model called"),
        )


@pytest.mark.asyncio
async def test_resume_reapplies_input_reference_gate(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    candidate = copy.deepcopy(cases[0]["expected_candidate"])
    candidate["evidence"][0]["section_id"] = next(
        s["id"] for s in cases[0]["snapshot"]["sections"] if s["representation"] == "html_text_block"
    )
    atomic_json(
        tmp_path / "raw.json",
        {
            "format": bench.CHECKPOINT_FORMAT,
            "identity": sha,
            "outputs": [{"case_id": cases[0]["case_id"], "candidate": candidate}],
            "in_flight": None,
        },
    )
    rows, receipt = await run(
        tmp_path,
        cases,
        identity=identity,
        freeze=freeze,
        sha=sha,
        handler=lambda _: pytest.fail("completed case resubmitted"),
    )
    assert rows[0]["error_code"] == "EVIDENCE_OUTSIDE_INPUT"
    assert receipt["raw_facts"]["correct"] == 2 and receipt["accepted_cases"] == 0


@pytest.mark.asyncio
async def test_truncated_completion_remains_available_for_raw_fact_scoring(tmp_path):
    cases = [case()]
    identity, freeze, sha = frozen(tmp_path, cases)
    candidate = copy.deepcopy(cases[0]["expected_candidate"])

    def handler(_):
        value = response(candidate).json()
        value["choices"][0]["finish_reason"] = "length"
        return httpx.Response(200, json=value)

    rows, receipt = await run(tmp_path, cases, identity=identity, freeze=freeze, sha=sha, handler=handler)
    assert "raw_output" in rows[0] and rows[0]["error_code"] == "TRUNCATED_OUTPUT"
    assert receipt["raw_facts"]["correct"] == 2 and receipt["valid_schema_cases"] == 1
    assert receipt["accepted_cases"] == 0


@pytest.mark.asyncio
async def test_expired_wall_limit_retains_remaining_cases_without_extra_calls(tmp_path):
    cases = [case(0), case(1)]
    identity, freeze, sha = frozen(tmp_path, cases, wall_seconds=0.2)
    calls = []

    async def handler(request):
        calls.append(request)
        await asyncio.sleep(0.5)
        return response(cases[0]["expected_candidate"])

    rows, receipt = await run(tmp_path, cases, identity=identity, freeze=freeze, sha=sha, handler=handler)
    assert len(calls) == 1 and len(rows) == 2
    assert all(row["error_code"] == "BENCHMARK_DEADLINE" for row in rows)
    assert receipt["case_count"] == 2 and receipt["raw_facts"]["present"] == 4


def test_invalid_deadline_and_nonloopback_endpoint_fail_before_client(tmp_path):
    for change in (
        {"wall_seconds": 1201},
        {"wall_seconds": float("nan")},
        {"base_url": "https://api.example.org/v1"},
    ):
        options = {
            "cases_sha256": "b" * 64,
            "model": MODEL,
            "source_policy": "markdown",
            "base_url": ENDPOINT,
        }
        options.update(change)
        with pytest.raises(ValueError):
            bench.freeze_identity([case()], **options)


@pytest.mark.asyncio
async def test_preparation_component_replay_is_explicitly_labeled(tmp_path):
    cases = [case(method="firecrawl_preparation_replay")]
    identity, freeze, sha = frozen(tmp_path, cases)
    _, receipt = await run(
        tmp_path,
        cases,
        identity=identity,
        freeze=freeze,
        sha=sha,
        handler=lambda _: response(cases[0]["expected_candidate"]),
    )
    assert receipt["runtime_track"] == "firecrawl_preparation_component_replay"
    assert receipt["source_retrieval_methods"] == ["firecrawl_preparation_replay"]
    assert any("does not establish Firecrawl API retrieval" in value for value in receipt["limitations"])
