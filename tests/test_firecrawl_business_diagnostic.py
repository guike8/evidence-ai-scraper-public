"""Self-contained, networkless tests of the published business-only diagnostic."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from ai_scraper.extraction.generic import populated_fields
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest
from benchmarks import firecrawl_business_diagnostic as diagnostic

MODEL = {"id": "example/local-model", "revision": "authored-test", "weights_sha256": "a" * 64}
URL = "http://127.0.0.1:8004/v1"
HISTORICAL_SCRIPT_SHA256 = "439c1aa26c1a7a7cbcc54bbad0c0b3ad92d9491cdd8b0bca95c840b7f39b204b"


def authored_case(index):
    """Facts and gold are authored before rendering synthetic page content."""
    facts = {
        "doc_title": f"Ledger entry {index}",
        "display_status": "Published",
        "count": 17 + index,
        "support_telephone": None,
    }
    request = ExtractionRequest(
        url=f"https://public.test/ledger/{index}",
        schema_id="custom_v1",
        instruction=(
            "Extract only the requested fields. Use short exact supporting excerpts for populated values. "
            "For integer/number fields, quote just the numeric literal as evidence. Do not guess."
        ),
        fields=[
            {"name": "doc_title", "description": "Copy the visible main heading exactly.", "critical": True},
            {"name": "display_status", "description": "Copy the status label exactly."},
            {
                "name": "count",
                "type": "integer",
                "description": "Explicit integer after Count, and quote only the numeric literal as evidence.",
            },
            {
                "name": "support_telephone",
                "required": False,
                "description": "A literal telephone number, or null if absent.",
            },
        ],
    )
    markdown = f"# {facts['doc_title']}\n\nStatus: {facts['display_status']}\n\nCount: {facts['count']}\n"
    html = (
        f"<h1>{facts['doc_title']}</h1><p>Status: {facts['display_status']}</p><p>Count: {facts['count']}</p>"
    )
    snapshot = normalize_record(
        {
            "final_url": request.url,
            "page_html": html,
            "markdown": markdown,
            "source_preparation": {"kind": "authored_test_component_replay"},
        },
        request,
        retrieval_method="firecrawl_preparation_replay",
    )
    sections = [section for section in snapshot.sections if section.representation == "markdown"]
    evidence = []
    for field in ("doc_title", "display_status", "count"):
        literal = str(facts[field])
        section = next(section for section in sections if literal in section.text)
        evidence.append(
            {"field_path": field, "section_id": section.id, "excerpt": literal, "json_pointer": None}
        )
    spec = get_schema(request)
    return {
        "case_id": f"business-diagnostic-authored-{index}",
        "schema_id": spec.schema_id,
        "schema_fingerprint": spec.fingerprint,
        "request": request.model_dump(mode="json"),
        "snapshot": snapshot.model_dump(mode="json"),
        "source_hash": snapshot.source_hash,
        "expected_candidate": {"data": facts, "evidence": evidence, "warnings": []},
        "expected_fields": populated_fields(facts),
        "expected_status": "success",
    }


def frozen_cases(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    cases_path.write_text("".join(json.dumps(authored_case(index)) + "\n" for index in range(2)))
    cases, cases_sha = diagnostic.bounded._cases(cases_path)
    frozen = diagnostic.identity(cases, cases_sha, MODEL, URL)
    freeze = tmp_path / "freeze.json"
    checksum = diagnostic.bounded.write_freeze(freeze, frozen)
    return cases, frozen, freeze, checksum


def test_historical_script_bytes_are_preserved():
    assert hashlib.sha256(Path(diagnostic.__file__).read_bytes()).hexdigest() == HISTORICAL_SCRIPT_SHA256


def test_freeze_is_offline_and_binds_authored_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: pytest.fail("HTTP client created during freeze"))
    cases, frozen, freeze, checksum = frozen_cases(tmp_path)
    assert diagnostic.bounded.verify_freeze(freeze, checksum, frozen) == checksum
    assert frozen["case_count"] == 2
    assert frozen["source_policy"] == "raw_unnumbered_markdown_only"
    assert frozen["diagnostic_script_sha256"] == HISTORICAL_SCRIPT_SHA256
    for case in cases:
        planned = frozen["inputs"][case["case_id"]]
        raw_markdown = case["snapshot"]["raw_record"]["markdown"]
        assert planned["markdown_sha256"] == hashlib.sha256(raw_markdown.encode()).hexdigest()
        assert planned["source_retrieval_method"] == "firecrawl_preparation_replay"
    text = freeze.read_text()
    assert "Ledger entry" not in text and str(tmp_path) not in text and URL not in text
    with pytest.raises(FileExistsError):
        diagnostic.bounded.write_freeze(freeze, frozen)


def test_correct_business_values_have_no_contract_acceptance_and_resume_without_calls(tmp_path):
    cases, frozen, freeze, checksum = frozen_cases(tmp_path)
    calls = []

    def respond(request):
        body = json.loads(request.content)
        case = cases[len(calls)]
        calls.append(body)
        assert body["tool_choice"] == "none"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["temperature"] == 0 and body["max_tokens"] == 4096
        assert len(body["messages"]) == 2
        assert body["messages"][1]["content"] == case["snapshot"]["raw_record"]["markdown"]
        assert "evidence" not in body["structured_outputs"]["json_schema"]["properties"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps(case["expected_candidate"]["data"])},
                    }
                ]
            },
        )

    async def execute():
        out = tmp_path / "run"
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            receipt = await diagnostic.run(cases, frozen, freeze, checksum, out, URL, client)
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _: pytest.fail("duplicate model call"))
        ) as client:
            again = await diagnostic.run(cases, frozen, freeze, checksum, out, URL, client)
        assert again["raw_facts"] == receipt["raw_facts"]
        return receipt

    receipt = asyncio.run(execute())
    assert len(calls) == 2
    assert receipt["exact_business_records"] == receipt["schema_valid_business_records"] == 2
    assert receipt["typed_factual"]["correct"] == receipt["typed_factual"]["present"] == 6
    assert receipt["typed_factual"]["predicted"] == 6
    assert receipt["absent_controls"] == {"present": 2, "correct_nulls": 2}
    assert receipt["accepted_cases"] == receipt["accepted_contract_records"] == 0
    assert receipt["accepted_facts"]["predicted"] == 0


@pytest.mark.parametrize("change", ["endpoint", "freeze"])
def test_mismatch_rejected_before_http_client_or_checkpoint(tmp_path, monkeypatch, change):
    cases, frozen, freeze, checksum = frozen_cases(tmp_path)
    monkeypatch.setattr(httpx, "AsyncClient", lambda **_: pytest.fail("HTTP client created"))
    endpoint = URL
    error = "FROZEN_ENDPOINT_CHANGED"
    if change == "endpoint":
        endpoint = "http://127.0.0.1:8005/v1"
    else:
        freeze.write_text(freeze.read_text() + " ")
        error = "FREEZE_CHANGED"
    with pytest.raises(ValueError, match=error):
        asyncio.run(diagnostic.run(cases, frozen, freeze, checksum, tmp_path / "run", endpoint))
    assert not (tmp_path / "run").exists()


def test_prompt_strips_citation_clauses_preserving_business_request():
    case = authored_case(0)
    _, (_, _, prompt, _) = diagnostic.plan(case, 48000)
    assert "evidence" not in prompt
    assert "supporting excerpts" not in prompt
    assert "Explicit integer after Count" in prompt
    assert "Copy the status label exactly" in prompt
