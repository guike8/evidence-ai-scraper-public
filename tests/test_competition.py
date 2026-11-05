"""Comparison scores retain failures and distinguish labels from quote support."""

from __future__ import annotations

import copy
import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from ai_scraper import competition
from ai_scraper.competition import (
    MANIFEST_FORMAT,
    RECEIPT_FORMAT,
    _hash,
    case_context,
    main,
    run_manifest,
    score_receipt,
    validate_manifest,
)


def manifest(*, schema="page_v1", fields=None, labels=None, html=None):
    request = {"schema_id": schema}
    if fields:
        request["fields"] = fields
    return {
        "format": MANIFEST_FORMAT,
        "corpus_id": "reviewed-test-v1",
        "labels_status": "reviewed",
        "labels_complete": True,
        "provenance": "Authored literal synthetic test content and reviewed complete leaf labels.",
        "cases": [
            {
                "id": "one",
                "url": "https://public.example.test/page",
                "request": request,
                "record": {"page_html": html or "<main><h1>Alpha</h1><p>Literal text.</p></main>"},
                "expected_fields": labels
                if labels is not None
                else {"title": "Alpha", "text.0": "Alpha", "text.1": "Literal text.", "headings.0": "Alpha"},
                "expected_status": "success",
            }
        ],
    }


def receipt(corpus, data, *, status="success", evidence=None, elapsed=5.0):
    _, _, snapshot, request_hash = case_context(corpus["cases"][0])
    return {
        "format": RECEIPT_FORMAT,
        "corpus_id": corpus["corpus_id"],
        "manifest_hash": _hash(corpus),
        "metadata": {
            "candidate": "independent-test-collector",
            "version": "test-revision",
            "mode": "recorded_snapshot",
            "runtime": "test Python",
            "hardware": {"system": "test-system", "machine": "test-machine"},
            "model": None,
            "provider": None,
            "cache_state": "disabled",
            "track": "identical_snapshot",
            "deadline_seconds": 1,
            "network_calls": None,
        },
        "results": [
            {
                "case_id": "one",
                "source_hash": snapshot.source_hash,
                "request_hash": request_hash,
                "status": status,
                "data": data,
                "evidence": evidence,
                "wall_latency_ms": elapsed,
            }
        ],
    }


def test_non_product_nested_array_leaves_are_scored_without_product_defaults():
    corpus = manifest()
    data = {"title": "Alpha", "text": ["Alpha", "Literal text."], "headings": ["Alpha"]}
    report = score_receipt(corpus, receipt(corpus, data))
    assert report["all"]["present"] == report["all"]["correct"] == 4
    assert report["required"]["present"] == 2  # page_v1 requires text, including each text leaf
    assert report["optional"]["present"] == 2
    assert set(report["by_schema"]) == {"page_v1"}
    assert report["evidence"]["coverage"] == 0  # correct labels do not invent verified quotes
    assert report["candidate"]["network_calls"] is None


@pytest.mark.parametrize("status", ["failed", "pending"])
def test_failed_and_pending_keep_every_present_gold_leaf_in_recall(status):
    corpus = manifest()
    report = score_receipt(corpus, receipt(corpus, None, status=status))
    assert report["all"]["present"] == report["all"]["missed"] == 4
    assert report["all"]["recall"] == 0
    assert report["all"]["precision"] is None
    assert report["statuses"] == {status: 1}


def test_omitted_outcomes_are_explicit_misses_instead_of_disappearing():
    corpus = manifest()
    observed = receipt(corpus, None, status="failed")
    observed["results"] = []
    report = score_receipt(corpus, observed)
    assert report["missing_receipts"] == ["one"]
    assert report["statuses"] == {"failed": 1}
    assert report["all"]["missed"] == 4
    assert report["expected_status_matches"] == 0


def test_zero_false_and_nested_list_values_remain_present_typed_facts():
    corpus = manifest(
        schema="custom_v1",
        fields=[
            {"name": "count", "type": "integer"},
            {"name": "active", "type": "boolean"},
            {"name": "tags", "type": "string_list", "required": False},
        ],
        labels={"count": 0, "active": False, "tags.0": "Alpha"},
        html="<main><p>Count: 0</p><p>Active: false</p><p>Alpha</p></main>",
    )
    _, _, snapshot, _ = case_context(corpus["cases"][0])
    evidence = []
    for path, text, excerpt in (
        ("count", "Count: 0", "0"),
        ("active", "Active: false", "false"),
        ("tags.0", "Alpha", "Alpha"),
    ):
        evidence.append(
            {
                "field_path": path,
                "section_id": next(section.id for section in snapshot.sections if section.text == text),
                "json_pointer": None,
                "excerpt": excerpt,
            }
        )
    report = score_receipt(
        corpus, receipt(corpus, {"count": 0, "active": False, "tags": ["Alpha"]}, evidence=evidence)
    )
    assert report["all"]["correct"] == 3
    assert report["required"]["correct"] == 2
    assert report["optional"]["correct"] == 1
    assert report["evidence"]["verified_populated_fields"] == 3
    assert report["evidence"]["coverage"] == 1


@pytest.mark.parametrize("predicted", [False, 0.0, "0"])
def test_strict_comparison_does_not_coerce_booleans_numbers_or_strings(predicted):
    corpus = manifest(schema="custom_v1", fields=[{"name": "count", "type": "integer"}], labels={"count": 0})
    report = score_receipt(corpus, receipt(corpus, {"count": predicted}))
    assert report["all"]["correct"] == 0
    assert report["all"]["false_positive"] == report["all"]["missed"] == 1


def test_extra_populated_and_explicitly_absent_values_are_label_mismatches():
    corpus = manifest(labels={"title": "Alpha", "author": None})
    report = score_receipt(
        corpus, receipt(corpus, {"title": "Alpha", "author": "Invented", "unknown": "Extra"})
    )
    assert report["all"]["present"] == 1
    assert report["all"]["predicted"] == 3
    assert report["unsupported_populated_facts"] == 2
    assert report["cases"][0]["label_mismatches"] == ["author", "unknown"]


def test_exact_quote_support_does_not_establish_task_correctness():
    corpus = manifest(
        schema="custom_v1",
        fields=[{"name": "answer"}],
        labels={"answer": "Alpha"},
        html="<main><p>Alpha</p><p>Beta</p></main>",
    )
    _, _, snapshot, _ = case_context(corpus["cases"][0])
    beta_id = next(section.id for section in snapshot.sections if section.text == "Beta")
    evidence = [{"field_path": "answer", "section_id": beta_id, "json_pointer": None, "excerpt": "Beta"}]
    report = score_receipt(corpus, receipt(corpus, {"answer": "Beta"}, evidence=evidence))
    assert report["evidence"]["coverage"] == 1
    assert report["all"]["precision"] == report["all"]["recall"] == 0
    assert report["unsupported_populated_facts"] == 1


def test_correct_data_with_wrong_excerpt_fails_the_independent_evidence_gate():
    corpus = manifest(schema="custom_v1", fields=[{"name": "answer"}], labels={"answer": "Alpha"})
    evidence = [
        {"field_path": "answer", "section_id": "s0001", "json_pointer": None, "excerpt": "Absent quote"}
    ]
    report = score_receipt(corpus, receipt(corpus, {"answer": "Alpha"}, evidence=evidence))
    assert report["all"]["correct"] == 1
    assert report["evidence"]["coverage"] == 0


def test_late_completions_keep_actual_latency_and_count_as_misses():
    corpus = manifest()
    observed = receipt(corpus, {"title": "Alpha"}, elapsed=1001)
    observed["results"][0]["component_latency_ms"] = {"extract": 800}
    report = score_receipt(corpus, observed)
    assert report["statuses"] == {"failed": 1}
    assert report["all"]["missed"] == 4
    assert report["wall_latency"]["p95_ms"] == 1001
    assert report["component_latency"]["extract"]["p50_ms"] == 800
    assert report["cases"][0]["deadline_exceeded"] is True


@pytest.mark.parametrize("change", ["unknown", "duplicate", "source", "request", "manifest"])
def test_receipts_cannot_silently_compare_different_or_duplicate_inputs(change):
    corpus = manifest()
    observed = receipt(corpus, {})
    if change == "unknown":
        observed["results"][0]["case_id"] = "unknown"
    elif change == "duplicate":
        observed["results"].append(copy.deepcopy(observed["results"][0]))
    elif change in {"source", "request"}:
        observed["results"][0][change + "_hash"] = "different"
    else:
        observed["manifest_hash"] = "different"
    with pytest.raises(ValueError):
        score_receipt(corpus, observed)


@pytest.mark.parametrize(
    "key",
    [
        "candidate",
        "version",
        "model",
        "provider",
        "hardware",
        "cache_state",
        "deadline_seconds",
        "network_calls",
        "runtime",
        "mode",
    ],
)
def test_required_metadata_cannot_be_omitted(key):
    corpus = manifest()
    observed = receipt(corpus, {})
    observed["metadata"].pop(key)
    with pytest.raises(ValueError):
        score_receipt(corpus, observed)


def test_unknown_timing_is_not_replaced_with_zero():
    corpus = manifest()
    report = score_receipt(corpus, receipt(corpus, {}, elapsed=None))
    assert report["wall_latency"] == {"samples": 0, "p50_ms": None, "p95_ms": None}
    assert report["unknown_completion_timing"] == ["one"]
    assert report["usage"]["input_tokens"] == {"measured_cases": 0, "total": None}
    assert report["retrieval_coverage"]["unknown"] == 1


def test_usage_is_aggregated_only_when_it_was_measured():
    corpus = manifest()
    observed = receipt(corpus, {})
    observed["results"][0]["usage"] = {"input_tokens": 0, "output_tokens": 3}
    report = score_receipt(corpus, observed)
    assert report["usage"]["input_tokens"] == {"measured_cases": 1, "total": 0}
    assert report["usage"]["output_tokens"] == {"measured_cases": 1, "total": 3}
    assert report["usage"]["inference_attempts"] == {"measured_cases": 0, "total": None}


def test_prompt_changes_can_compete_but_different_user_goals_cannot(monkeypatch):
    corpus = manifest(schema="custom_v1", fields=[{"name": "answer"}], labels={"answer": "Alpha"})
    case = corpus["cases"][0]
    initial = case_context(case)[3]
    original = competition.get_schema

    def another_prompt(request):
        return replace(original(request), fingerprint="another-prompt", prompt_version="another-version")

    monkeypatch.setattr(competition, "get_schema", another_prompt)
    assert case_context(case)[3] == initial
    different = copy.deepcopy(case)
    different["request"]["instruction"] = "A different extraction goal"
    assert case_context(different)[3] != initial


@pytest.mark.parametrize(
    "labels", [{"title.0": "Wrong shape"}, {"text.word": "Wrong path"}, {"title": False}]
)
def test_reviewed_label_paths_and_scalar_types_still_need_to_match_the_contract(labels):
    with pytest.raises(ValueError):
        validate_manifest(manifest(labels=labels))


def test_wrong_failure_payload_is_not_accepted_as_business_data():
    corpus = manifest()
    with pytest.raises(ValueError):
        score_receipt(corpus, receipt(corpus, {"title": "Alpha"}, status="failed"))


@pytest.mark.parametrize("key,value", [("labels_status", "unreviewed"), ("labels_complete", False)])
def test_incomplete_or_unreviewed_labels_cannot_support_fact_precision(key, value):
    corpus = manifest()
    corpus[key] = value
    with pytest.raises(ValueError):
        validate_manifest(corpus)


@pytest.mark.asyncio
async def test_real_pipeline_fixture_run_is_credentialless_and_network_free(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Fixture comparison must not connect to a network")

    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setenv("BRIGHTDATA_API_KEY", "ignored-test-secret")
    monkeypatch.setenv("HF_TOKEN", "ignored-test-secret")
    corpus = manifest()
    observed = await run_manifest(corpus)
    assert observed["metadata"]["mode"] == "deterministic_offline"
    assert observed["metadata"]["network_calls"] == 0
    assert observed["metadata"]["model"] is None
    assert observed["metadata"]["cache_state"] == "cold"
    assert observed["results"][0]["status"] == "success"
    report = score_receipt(corpus, observed)
    assert report["all"]["correct"] == 4
    assert report["evidence"]["coverage"] == 1


@pytest.mark.asyncio
async def test_reviewed_fixture_corpus_spans_universal_schemas_and_retains_semantic_misses():
    path = Path(__file__).parents[1] / "benchmarks/fixtures.json"
    corpus = json.loads(path.read_text())
    assert len(validate_manifest(corpus)) >= 12
    observed = await run_manifest(corpus)
    report = score_receipt(corpus, observed)
    assert set(report["by_schema"]) >= {
        "page_v1",
        "article_v1",
        "organization_v1",
        "person_v1",
        "job_v1",
        "event_v1",
        "listing_v1",
        "custom_v1",
        "product_v1",
    }
    assert report["unsupported_populated_facts"] == 0
    custom = next(case for case in report["cases"] if case["case_id"] == "custom-zero-false")
    assert custom["present"] == 2 and custom["correct"] == 0
    assert report["all"]["recall"] < 1


def test_cli_rejects_invalid_json_without_echoing_input(tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_text('{"private-marker"')
    output = tmp_path / "output.json"
    assert main(["run", str(path), "--output", str(output)]) == 1
    assert not output.exists()
    assert "private-marker" not in capsys.readouterr().err
