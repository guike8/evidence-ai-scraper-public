"""Check citation targets and immutable replay, not model accuracy."""

import json
from pathlib import Path

import pytest

from ai_scraper.extraction.huggingface import HuggingFaceExtractor, request_instruction
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest, SourceSnapshot
from training.citation_dataset import SCHEMAS, build_case, generate_dataset
from training.dataset import check_disjoint
from training.evaluate import validate_cases


@pytest.mark.parametrize("schema", SCHEMAS)
def test_authored_citation_modes_pass_independent_validation_and_replay(schema):
    cases = []
    for split in ("train", "valid"):
        for n in range(10):
            row, case = build_case(split, schema, n)
            cases.append(case)
            assert json.loads(row["messages"][-1]["content"]) == case["expected_candidate"]
            source = json.loads(row["messages"][1]["content"].split("\n", 1)[1])
            ids = {s["id"] for s in source["sections"]}
            references = case["expected_candidate"]["evidence"]
            assert all(e["section_id"] in ids for e in references if e["section_id"])
            assert all(not e["field_path"].startswith("data.") for e in references)
            assert set(case["expected_fields"]) == {e["field_path"] for e in references}
    validate_cases(cases)


def test_train_shuffle_preserves_ids_text_and_gold_while_valid_is_exact_production():
    row, case = build_case("train", "page_v1", 0)
    request = ExtractionRequest.model_validate(case["request"])
    snapshot = SourceSnapshot.model_validate_json(json.dumps(case["snapshot"]))
    spec = get_schema(request)
    extractor = HuggingFaceExtractor("offline", "offline", client=object()).for_request(request, spec)
    production = json.loads(extractor._source_content(snapshot))
    training = json.loads(row["messages"][1]["content"].split("\n", 1)[1])
    assert training["sections"] != production["sections"]
    assert sorted(training["sections"], key=lambda s: s["id"]) == sorted(
        production["sections"], key=lambda s: s["id"]
    )
    assert row["messages"][0]["content"] == request_instruction(request, spec)
    row, case = build_case("valid", "page_v1", 0)
    request = ExtractionRequest.model_validate(case["request"])
    snapshot = SourceSnapshot.model_validate_json(json.dumps(case["snapshot"]))
    extractor = HuggingFaceExtractor("offline", "offline", client=object()).for_request(
        request, get_schema(request)
    )
    assert row["messages"][1]["content"] == "UNTRUSTED_SOURCE_JSON\n" + extractor._source_content(snapshot)


def test_no_ld_majority_has_only_section_targets_and_no_excluded_ids():
    cases = [build_case("train", schema, n)[1] for schema in SCHEMAS for n in range(10)]
    without_ld = [c for c in cases if c["citation_curriculum"]["without_json_ld"]]
    assert len(without_ld) / len(cases) >= 0.6
    for case in without_ld:
        assert case["snapshot"]["ld_json"] == []
        assert all(e["json_pointer"] is None for e in case["expected_candidate"]["evidence"])
        excluded = set((case["source_view"] or {}).get("excluded_section_ids", []))
        assert not excluded.intersection(e["section_id"] for e in case["expected_candidate"]["evidence"])


def test_nested_scalar_zero_false_and_relative_attribute_targets():
    _, custom = build_case("train", "custom_v1", 0)
    refs = {e["field_path"]: e for e in custom["expected_candidate"]["evidence"]}
    assert custom["expected_candidate"]["data"]["seat_count"] == 0
    assert custom["expected_candidate"]["data"]["accepting"] is False
    assert refs["seat_count"]["excerpt"] == "0"
    assert refs["accepting"]["excerpt"] == "false"
    _, page = build_case("train", "page_v1", 0)
    refs = {e["field_path"]: e for e in page["expected_candidate"]["evidence"]}
    assert refs["links.0.url"]["excerpt"].startswith("/details/")
    assert refs["tables.0.rows.0.1"]["excerpt"] == "0"


def test_role_negatives_and_positive_short_description_are_explicit():
    _, positive = build_case("train", "product_v1", 4)
    assert positive["expected_candidate"]["data"]["short_description"].startswith("Compact reference")
    assert positive["expected_candidate"]["data"]["long_description"] is not None
    _, absent = build_case("train", "product_v1", 1)
    assert absent["expected_candidate"]["data"]["short_description"] is None
    _, event = build_case("train", "event_v1", 2)
    assert event["expected_candidate"]["data"]["start_date"] is None
    assert event["expected_candidate"]["data"]["end_date"] is not None
    _, job = build_case("train", "job_v1", 6)
    data = job["expected_candidate"]["data"]
    assert data["title"] is None
    assert "Collections Assistant" not in data["description"]


def test_split_groups_and_templates_remain_disjoint():
    cases = {s: [build_case(s, schema, 0)[1] for schema in SCHEMAS] for s in ("train", "valid", "test")}
    assert all(check_disjoint(cases).values())


def test_generation_is_reproducible_private_and_refuses_mismatched_outputs(tmp_path):
    kwargs = {"train_per_schema": 2, "valid_per_schema": 2, "test_per_schema": 2}
    first, second = tmp_path / "one", tmp_path / "two"
    manifest = generate_dataset(first, **kwargs)
    assert manifest == generate_dataset(second, **kwargs)
    assert manifest == generate_dataset(first, **kwargs)
    for name in manifest["files"]:
        assert (first / name).read_bytes() == (second / name).read_bytes()
        assert (first / name).stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="configuration differs"):
        generate_dataset(first, seed=1, **kwargs)
    checkout = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside the public source checkout"):
        generate_dataset(checkout / "training" / "bad-output", **kwargs)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unrelated").write_text("keep")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        generate_dataset(occupied, **kwargs)
