"""Authored frozen two-stage evaluation fixtures; no network or model weights."""

from __future__ import annotations

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
from training import realpage_evaluate as bench

MODEL = {"id": "example/local-model", "revision": "authored-fixture", "weights_sha256": "a" * 64}
ENDPOINT = "http://127.0.0.1:8003/v1"


def case(index=0, *, oversized_count=False):
    request = ExtractionRequest(
        url=f"https://public.test/record-{index}",
        schema_id="custom_v1",
        instruction="Extract the ledger name and its exact entry count.",
        fields=[
            {"name": "name", "description": "Ledger's displayed name", "critical": True},
            {"name": "count", "type": "integer", "description": "Number of ledger entries"},
            {"name": "missing_contact", "description": "Public fax number", "required": False},
        ],
    )
    count_text = ("Context " + "unrelated " * 400 + "Count: 17") if oversized_count else "Count: 17"
    snapshot = normalize_record(
        {
            "final_url": request.url,
            "page_html": "<h1>Ledger entry</h1><p>Count: 17</p>",
            "markdown": "# Ledger entry\n\n" + count_text,
            "source_preparation": {"kind": "firecrawl_preparation_component_replay"},
        },
        request,
        retrieval_method="firecrawl_preparation_replay",
    )
    sections = [item for item in snapshot.sections if item.representation == "markdown"]
    evidence = [
        {"field_path": "name", "section_id": sections[0].id, "excerpt": "Ledger entry", "json_pointer": None},
        {"field_path": "count", "section_id": sections[1].id, "excerpt": "17", "json_pointer": None},
    ]
    candidate = {
        "data": {"name": "Ledger entry", "count": 17, "missing_contact": None},
        "evidence": evidence,
        "warnings": [],
    }
    spec = get_schema(request)
    result = {
        "case_id": f"realpage-fixture-{index}",
        "schema_id": spec.schema_id,
        "schema_fingerprint": spec.fingerprint,
        "request": request.model_dump(mode="json"),
        "snapshot": snapshot.model_dump(mode="json"),
        "source_hash": snapshot.source_hash,
        "expected_candidate": candidate,
        "expected_fields": populated_fields(candidate["data"]),
        "expected_status": "success",
        "reviewed_evidence": [
            {**item, "semantic_context": "Authored fixture labels the ledger identity and entry count."}
            for item in evidence
        ],
    }
    evaluate.validate_cases([result])
    return result


def frozen(tmp_path, cases, **options):
    identity = bench.freeze_identity(
        cases,
        cases_sha256="b" * 64,
        model=MODEL,
        base_url=ENDPOINT,
        **options,
    )
    path = tmp_path / "freeze.json"
    checksum = bench.write_freeze(path, identity)
    return identity, path, checksum


def response(payload, *, finish="stop", prompt=100, output=20):
    return httpx.Response(
        200,
        json={
            "choices": [{"finish_reason": finish, "message": {"content": json.dumps(payload)}}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": output},
        },
    )


def successful_handler(calls, data=None):
    def respond(http_request):
        body = json.loads(http_request.content)
        calls.append(body)
        if body["messages"][1]["content"].startswith("UNTRUSTED_PAGE_MARKDOWN\n"):
            return response(data or {"name": "Ledger entry", "count": 17, "missing_contact": None})
        context = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        return response(
            {
                "selections": {
                    path: items[0]["id"] if items else None for path, items in context["candidates"].items()
                }
            }
        )

    return respond


async def run(tmp_path, cases, identity, freeze, sha, handler):
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        return await bench.run_evaluation(
            cases,
            identity=identity,
            freeze=freeze,
            freeze_sha=sha,
            checkpoint=tmp_path / "raw.json",
            base_url=ENDPOINT,
            client=client,
        )


def test_freeze_is_offline_and_contains_only_input_hashes_and_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: pytest.fail("HTTP client created during freeze"))
    cases = [case()]
    identity, path, sha = frozen(tmp_path, cases)
    assert bench.verify_freeze(path, sha, identity) == sha
    assert identity["case_order"] == [cases[0]["case_id"]]
    assert identity["cases_sha256"] == "b" * 64
    assert "training/realpage_evaluate.py" in identity["code"]["files"]
    body = path.read_text()
    assert "Ledger entry" not in body and "Authored fixture labels" not in body
    assert str(tmp_path) not in body and ENDPOINT not in body
    assert identity["inputs"][cases[0]["case_id"]]


def test_large_full_saved_case_is_read_without_widening_the_model_view(tmp_path):
    original = case()
    enlarged = copy.deepcopy(original)
    request = ExtractionRequest.model_validate(enlarged["request"])
    record = copy.deepcopy(enlarged["snapshot"]["raw_record"])
    # Saved vendor HTML may retain a large template/comment while the supplied
    # Markdown and its original evidence sections remain exactly unchanged.
    record["page_html"] += "<!--" + "x" * 2_050_000 + "-->"
    snapshot = normalize_record(record, request, retrieval_method="firecrawl_preparation_replay")
    enlarged["snapshot"] = snapshot.model_dump(mode="json")
    enlarged["source_hash"] = snapshot.source_hash
    body = (json.dumps(enlarged, separators=(",", ":")) + "\n").encode()
    assert evaluate.MAX_ROW_BYTES < len(body) < bench.MAX_CASE_ROW_BYTES
    path = tmp_path / "large.cases.jsonl"
    path.write_bytes(body)
    loaded, sha = bench._cases(path)
    assert loaded == [enlarged] and sha == hashlib.sha256(body).hexdigest()
    small = bench.freeze_identity([original], cases_sha256="b" * 64, model=MODEL, base_url=ENDPOINT)
    large = bench.freeze_identity(loaded, cases_sha256=sha, model=MODEL, base_url=ENDPOINT)
    first, second = small["inputs"][original["case_id"]], large["inputs"][original["case_id"]]
    assert second["source_descriptor"]["view_chars"] <= 8000
    assert second["business_messages_sha256"] == first["business_messages_sha256"]
    assert second["business_schema_sha256"] == first["business_schema_sha256"]


def test_large_case_row_limit_still_fails_before_unbounded_json_parsing(tmp_path):
    path = tmp_path / "oversized.cases.jsonl"
    path.write_bytes(b'{"padding":"' + b"x" * bench.MAX_CASE_ROW_BYTES + b'"}\n')
    with pytest.raises(ValueError, match="CASE_LIMIT"):
        bench._cases(path)


@pytest.mark.parametrize("body", [b'{"case_id":"one","case_id":"two"}\n', b'{"broken":\n', b"[]\n"])
def test_case_reader_rejects_duplicate_keys_malformed_json_and_nonobjects(tmp_path, body):
    path = tmp_path / "malformed.cases.jsonl"
    path.write_bytes(body)
    with pytest.raises(ValueError):
        bench._cases(path)


def test_case_reader_refuses_symlinks(tmp_path):
    target = tmp_path / "real.cases.jsonl"
    target.write_text(json.dumps(case()) + "\n")
    link = tmp_path / "linked.cases.jsonl"
    link.symlink_to(target)
    with pytest.raises(OSError):
        bench._cases(link)


async def test_success_uses_two_actual_stage_requests_and_receipt_contains_no_raw_passages(tmp_path):
    cases = [case()]
    identity, path, sha = frozen(tmp_path, cases)
    calls = []
    rows, receipt = await run(tmp_path, cases, identity, path, sha, successful_handler(calls))
    assert len(calls) == 2
    assert receipt["case_count"] == receipt["accepted_cases"] == 1
    assert receipt["raw_facts"]["present"] == receipt["raw_facts"]["correct"] == 2
    assert receipt["accepted_facts"]["wrong_or_unlabeled"] == 0
    assert rows[0]["candidate"]["data"] == cases[0]["expected_candidate"]["data"]
    assert rows[0]["input_tokens"] == 200 and rows[0]["output_tokens"] == 40
    assert rows[0]["model_calls"] == 2
    assert receipt["business_output"]["exact_record_cases"] == 1
    assert receipt["business_output"]["complete_schema_valid_cases"] == 1
    assert receipt["business_output"]["correct_null_controls"] == 1
    assert receipt["grounding"]["source_valid_leaves"] == receipt["reviewed_anchor_agreement"] == 2
    assert receipt["actual_model_calls"] == 2 and receipt["model_calls_unknown_cases"] == 0
    assert receipt["maximum_reserved_model_calls"] == 2
    for name in ("business", "grounding"):
        assert receipt["stage_usage"][name]["calls"] == 1
        assert receipt["stage_usage"][name]["latency"]["p50_ms"] >= 0
    assert calls[0]["response_format"]["json_schema"]["schema"]["properties"].keys() == {
        "name",
        "count",
        "missing_contact",
    }
    assert calls[1]["response_format"]["json_schema"]["schema"]["properties"].keys() == {"selections"}
    public = json.dumps(receipt)
    assert "Ledger entry" not in public and "Count: 17" not in public and str(tmp_path) not in public
    assert not os.stat(tmp_path / "raw.json").st_mode & 0o077


async def test_failed_grounding_retains_correct_raw_business_facts_without_acceptance(tmp_path):
    cases = [case()]
    identity, path, sha = frozen(tmp_path, cases)
    calls = []

    def reject(http_request):
        calls.append(json.loads(http_request.content))
        if len(calls) == 1:
            return response(cases[0]["expected_candidate"]["data"])
        return response({"selections": {"name": "c9999", "count": "c9998"}})

    rows, receipt = await run(tmp_path, cases, identity, path, sha, reject)
    assert len(calls) == rows[0]["model_calls"] == 2
    assert rows[0]["error_code"] == "GROUNDING_REJECTED"
    assert rows[0]["candidate"] == {
        "data": cases[0]["expected_candidate"]["data"],
        "evidence": [],
        "warnings": [],
    }
    assert receipt["raw_facts"]["correct"] == 2
    assert receipt["accepted_cases"] == receipt["accepted_facts"]["correct"] == 0
    assert receipt["business_output"]["raw_facts"]["correct"] == 2
    assert receipt["grounding"]["invalid_or_missing_ids"] == 2


async def test_completed_checkpoint_resume_skips_both_model_calls(tmp_path):
    cases = [case()]
    identity, path, sha = frozen(tmp_path, cases)
    calls = []
    await run(tmp_path, cases, identity, path, sha, successful_handler(calls))
    rows, receipt = await run(
        tmp_path, cases, identity, path, sha, lambda _: pytest.fail("Completed case was resubmitted")
    )
    assert len(calls) == 2 and len(rows) == 1 and receipt["accepted_cases"] == 1


async def test_ambiguous_inflight_resume_records_failure_without_resubmission(tmp_path):
    cases = [case()]
    identity, path, sha = frozen(tmp_path, cases)
    atomic_json(
        tmp_path / "raw.json",
        {"format": bench.CHECKPOINT_FORMAT, "identity": sha, "outputs": [], "in_flight": cases[0]["case_id"]},
    )
    rows, receipt = await run(
        tmp_path, cases, identity, path, sha, lambda _: pytest.fail("Ambiguous case was resubmitted")
    )
    assert len(rows) == 1 and rows[0]["error_code"] == "INFERENCE_INTERRUPTED"
    assert receipt["case_count"] == 1 and receipt["accepted_cases"] == 0
    assert receipt["raw_facts"]["present"] == 2
    assert receipt["actual_model_calls"] == 0 and receipt["model_calls_unknown_cases"] == 1
    assert receipt["maximum_reserved_model_calls"] == 2


@pytest.mark.parametrize("change", ["identity", "endpoint", "checksum"])
async def test_changed_profile_or_endpoint_rejected_before_client_creation(tmp_path, monkeypatch, change):
    cases = [case()]
    identity, path, sha = frozen(tmp_path, cases)
    endpoint = ENDPOINT
    if change == "identity":
        identity = copy.deepcopy(identity)
        identity["model"]["revision"] = "different-model"
    elif change == "endpoint":
        endpoint = "http://127.0.0.1:8004/v1"
    else:
        path.write_text(path.read_text() + " ")
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **_: pytest.fail("Client created before freeze verification")
    )
    with pytest.raises(ValueError):
        await bench.run_evaluation(
            cases,
            identity=identity,
            freeze=path,
            freeze_sha=sha,
            checkpoint=tmp_path / "raw.json",
            base_url=endpoint,
        )
    assert not (tmp_path / "raw.json").exists()


@pytest.mark.parametrize("stage", ["business", "grounding"])
async def test_truncation_preserves_actual_stage_calls_tokens_and_raw_parseable_business(tmp_path, stage):
    cases = [case()]
    identity, path, sha = frozen(tmp_path, cases)
    calls = []

    def truncated(http_request):
        calls.append(json.loads(http_request.content))
        if len(calls) == 1:
            return response(
                cases[0]["expected_candidate"]["data"], finish="length" if stage == "business" else "stop"
            )
        return response({"selections": {}}, finish="length")

    rows, receipt = await run(tmp_path, cases, identity, path, sha, truncated)
    count = 1 if stage == "business" else 2
    assert len(calls) == rows[0]["model_calls"] == count
    assert rows[0]["error_code"] == "TRUNCATED_OUTPUT"
    assert rows[0]["input_tokens"] == 100 * count
    assert rows[0]["output_tokens"] == 20 * count
    assert receipt["raw_facts"]["correct"] == 2 and receipt["accepted_cases"] == 0


async def test_source_window_omission_keeps_full_gold_fact_denominator(tmp_path):
    cases = [case(oversized_count=True)]
    identity, path, sha = frozen(tmp_path, cases, max_source_chars=1024)
    calls = []
    rows, receipt = await run(
        tmp_path,
        cases,
        identity,
        path,
        sha,
        successful_handler(calls, {"name": "Ledger entry", "count": None, "missing_contact": None}),
    )
    assert len(calls) == 2 and len(rows) == receipt["case_count"] == 1
    assert receipt["raw_facts"]["present"] == 2 and receipt["raw_facts"]["correct"] == 1
    assert "Count: 17" not in calls[0]["messages"][1]["content"]
    assert receipt["input_gold_coverage"] == {"positive_fields": 2, "reviewed_fields_visible": 1}


async def test_changed_completed_candidate_is_regated_and_durably_rejected_without_resubmission(tmp_path):
    cases = [case()]
    identity, path, sha = frozen(tmp_path, cases)
    calls = []
    await run(tmp_path, cases, identity, path, sha, successful_handler(calls))
    saved = evaluate._parse((tmp_path / "raw.json").read_text())
    saved["outputs"][0]["candidate"]["data"]["count"] = 999
    atomic_json(tmp_path / "raw.json", saved)
    rows, receipt = await run(
        tmp_path, cases, identity, path, sha, lambda _: pytest.fail("Changed case was resubmitted")
    )
    assert rows[0]["error_code"] == "CHECKPOINT_GROUNDING_CHANGED"
    assert receipt["accepted_cases"] == 0
    durable = evaluate._parse((tmp_path / "raw.json").read_text())
    assert durable["outputs"][0]["error_code"] == "CHECKPOINT_GROUNDING_CHANGED"


async def test_literal_but_wrong_business_fact_is_measured_as_wrong_accepted_not_hidden(tmp_path):
    item = case()
    req = ExtractionRequest.model_validate(item["request"])
    record = copy.deepcopy(item["snapshot"]["raw_record"])
    record["markdown"] += "\n\nAdvertisement for a different ledger: 999 entries."
    snap = normalize_record(record, req, retrieval_method="firecrawl_preparation_replay")
    item["snapshot"] = snap.model_dump(mode="json")
    item["source_hash"] = snap.source_hash
    evaluate.validate_cases([item])
    identity, path, sha = frozen(tmp_path, [item])
    calls = []
    _, receipt = await run(
        tmp_path,
        [item],
        identity,
        path,
        sha,
        successful_handler(calls, {"name": "Ledger entry", "count": 999, "missing_contact": None}),
    )
    assert receipt["accepted_cases"] == 1
    assert receipt["wrong_accepted_facts"] == 1
    assert receipt["business_output"]["raw_facts"]["wrong_or_unlabeled"] == 1
    assert receipt["reviewed_anchor_agreement"] == 1
