"""Reviewed synthetic labels exercise facts, failures, and evidence relationships."""

import copy
import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from ai_scraper.extraction.deterministic import deterministic_extract, parse_price
from ai_scraper.normalize import detect_content_problem, normalize_record
from ai_scraper.schemas import REQUIRED_FIELDS, ExtractionRequest, ProductV1
from ai_scraper.validate import populated_fields, validate_candidate

CORPUS = json.loads((Path(__file__).parent / "fixtures/products.json").read_text())
CASES = CORPUS["cases"]


def run_case(case):
    request = ExtractionRequest(url=case["url"], locale=case.get("locale"))
    snapshot = normalize_record(case["record"], request)
    problem = detect_content_problem(snapshot)
    if problem:
        return snapshot, None, None, problem
    candidate = deterministic_extract(snapshot, request.locale)
    assessment = validate_candidate(candidate, snapshot, request.locale)
    return snapshot, candidate, assessment, None


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_labeled_fixture(case):
    snapshot, _candidate, result, problem = run_case(case)
    assert problem == case.get("content_problem")
    if problem:
        assert case["status"] == "failed"
        return
    assert result.status == case["status"], result.errors
    expected = ProductV1.model_validate(case["expected"])
    if result.data is not None:
        assert result.data == expected
        assert set(result.evidence) == set(populated_fields(result.data))
        for evidence in result.evidence.values():
            if evidence.section_id:
                assert evidence.excerpt in next(
                    s.text for s in snapshot.sections if s.id == evidence.section_id
                )
    else:
        assert result.status == "failed"
        assert expected.name is None


def evaluate_corpus():
    """Failed outcomes contribute zero predictions and misses for all present truth fields."""
    counts = defaultdict(lambda: {"correct": 0, "predicted": 0, "present": 0})
    unsupported = 0
    retrieved = 0
    statuses = defaultdict(int)
    for case in CASES:
        _, _, result, problem = run_case(case)
        actual = result.data.model_dump() if result and result.data else {}
        expected = ProductV1.model_validate(case["expected"]).model_dump()
        statuses["failed" if problem else result.status] += 1
        retrieved += not bool(problem)
        if result and result.data:
            truth_leaves = populated_fields(expected)
            for path, value in populated_fields(result.data).items():
                unsupported += truth_leaves.get(path) != value
        for field in REQUIRED_FIELDS:
            for dimension in ("all", f"field:{field}", f"domain:{urlsplit(case['url']).hostname}"):
                item = counts[dimension]
                item["present"] += expected[field] is not None
                item["predicted"] += actual.get(field) is not None
                item["correct"] += expected[field] is not None and actual.get(field) == expected[field]
    for item in counts.values():
        item["precision"] = item["correct"] / item["predicted"] if item["predicted"] else None
        item["recall"] = item["correct"] / item["present"] if item["present"] else None
    return {
        "corpus_version": CORPUS["corpus_version"],
        "fixture_count": len(CASES),
        "retrieval_coverage": {"usable": retrieved, "total": len(CASES)},
        "statuses": dict(statuses),
        "unsupported_populated_facts": unsupported,
        "metrics": dict(counts),
    }


def test_release_corpus_gate():
    report = evaluate_corpus()
    assert len(CASES) >= 20
    assert report["unsupported_populated_facts"] == 0
    assert report["metrics"]["all"]["precision"] >= 0.95
    assert report["metrics"]["all"]["recall"] >= 0.95


def test_nonexistent_and_unrelated_evidence_rejected():
    snapshot, candidate, _, _ = run_case(CASES[0])
    payload = candidate.model_dump()
    payload["data"]["name"] = "Fabricated Product"
    result = validate_candidate(payload, snapshot)
    assert result.status == "failed"
    assert any("evidence:name" in error for error in result.errors)
    # A scalar that occurs in JSON-LD under sku cannot establish the product name.
    payload = candidate.model_dump()
    payload["data"]["name"] = "MH09"
    for item in payload["evidence"]:
        if item["field_path"] == "name":
            item.update(section_id=None, json_pointer="/0/sku", excerpt="MH09")
    assert validate_candidate(payload, snapshot).status == "failed"


def test_provenance_cannot_be_model_generated():
    snapshot, candidate, _, _ = run_case(CASES[0])
    payload = candidate.model_dump()
    payload["provenance"] = {"provider": "pretend"}
    result = validate_candidate(payload, snapshot)
    assert result.status == "failed"
    assert any("Extra inputs" in error for error in result.errors)


def test_strict_types_and_business_rules():
    from pydantic import ValidationError

    for value in (10.0, 10, True, "NaN", "-1", "1e10", "1,234"):
        with pytest.raises(ValidationError):
            ProductV1(price=value)
    for value in ("$", "usd", "ZZZ"):
        with pytest.raises(ValidationError):
            ProductV1(currency=value)
    with pytest.raises(ValidationError):
        ProductV1(images=["javascript:alert(1)"])
    with pytest.raises(ValidationError):
        ProductV1(name=123)


def test_source_conflict_cannot_be_overridden_by_model():
    case = next(c for c in CASES if c["id"] == "price-conflict")
    snapshot, candidate, _, _ = run_case(case)
    payload = candidate.model_dump()
    payload["data"]["price"] = "99.00"
    payload["warnings"] = []
    payload["evidence"].append(
        {"field_path": "price", "section_id": None, "json_pointer": "/0/offers/price", "excerpt": "99.00"}
    )
    result = validate_candidate(payload, snapshot)
    assert result.status == "failed"
    assert any(error.startswith("conflict:price") for error in result.errors)


def test_price_locale_is_explicit_and_dollar_currency_not_guessed():
    assert parse_price("USD 1,499.00") is None
    assert parse_price("USD 1,499.00", "en_US") == "1499.00"
    assert parse_price("1.299,50 EUR", "de_DE") == "1299.50"
    assert parse_price("10 - 20 USD") is None
    case = next(c for c in CASES if c["id"] == "dollar-only")
    _, _, result, _ = run_case(case)
    assert result.data.currency is None
    assert result.status == "partial"


def test_snapshot_hash_stable_raw_record_preserved_and_exclusion_not_success():
    case = CASES[0]
    request = ExtractionRequest(url=case["url"])
    original = copy.deepcopy(case["record"])
    snapshot = normalize_record(original, request)
    assert original == snapshot.raw_record
    assert normalize_record(original, request).source_hash == snapshot.source_hash
    snapshot.excluded_sections = ["s9999"]
    result = validate_candidate(deterministic_extract(snapshot), snapshot)
    assert result.status == "partial"
    assert any("excluded" in warning for warning in result.warnings)


def test_injection_does_not_change_deterministic_contract():
    case = next(c for c in CASES if c["id"] == "injection")
    _, _, result, _ = run_case(case)
    assert result.data.name == "Safe Charger"
    assert result.data.price == "18.00"
    assert set(result.data.model_dump()) == set(ProductV1.model_fields)


def test_evidence_from_different_product_cannot_replace_target_identity():
    case = next(c for c in CASES if c["id"] == "multiple-matched")
    snapshot, candidate, _, _ = run_case(case)
    payload = candidate.model_dump()
    payload["data"]["name"] = "Other Book"
    for item in payload["evidence"]:
        if item["field_path"] == "name":
            item.update(section_id=None, json_pointer="/0/name", excerpt="Other Book")
    result = validate_candidate(payload, snapshot)
    assert result.status == "failed"
    assert any(error.startswith("identity:name") for error in result.errors)
