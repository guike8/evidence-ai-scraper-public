"""Verify authored labels, representation crossing and immutable source binding."""

import json
from collections import Counter
from pathlib import Path

import pytest

from ai_scraper.extraction.huggingface import HuggingFaceExtractor, request_instruction
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest, SourceSnapshot
from ai_scraper.validate import _pointer
from training.citation_dataset_v3 import MODES, REPRESENTATIONS, SCHEMAS, build_case, generate_dataset
from training.dataset import check_disjoint
from training.evaluate import validate_cases


@pytest.mark.parametrize("schema", SCHEMAS)
def test_all_modes_cross_all_representations_with_source_verified_gold(schema):
    cases = []
    counts = Counter()
    for split in ("train", "valid"):
        for number in range(24):
            row, case = build_case(split, schema, number)
            cases.append(case)
            counts[split, case["mode"], case["citation_curriculum"]["representation"]] += 1
            assert json.loads(row["messages"][-1]["content"]) == case["expected_candidate"]
            source = json.loads(row["messages"][1]["content"].split("\n", 1)[1])
            sections = {section["id"]: section["text"] for section in source["sections"]}
            evidence = case["expected_candidate"]["evidence"]
            assert set(case["expected_fields"]) == {item["field_path"] for item in evidence}
            for item in evidence:
                assert not item["field_path"].startswith("data.")
                if item["section_id"] is not None:
                    assert item["json_pointer"] is None
                    assert item["excerpt"] in sections[item["section_id"]]
                else:
                    scalar = _pointer(source["ld_json"], item["json_pointer"])
                    assert item["excerpt"] == (
                        scalar if isinstance(scalar, str) else json.dumps(scalar, separators=(",", ":"))
                    )
    for split in ("train", "valid"):
        for representation in REPRESENTATIONS:
            assert counts[split, "complete", representation] == 4
            for mode in MODES[1:]:
                assert counts[split, mode, representation] == 1
    validate_cases(cases)


def test_positive_jsonld_has_regular_gold_and_plaintext_has_no_pointers():
    cases = [build_case("train", schema, number)[1] for schema in SCHEMAS for number in range(24)]
    positive_ld = [
        case
        for case in cases
        if case["mode"] == "complete" and case["citation_curriculum"]["representation"] == "jsonld"
    ]
    assert len(positive_ld) == len(SCHEMAS) * 4
    assert all(case["expected_status"] == "success" for case in positive_ld)
    assert all(case["snapshot"]["ld_json"] for case in positive_ld)
    assert all(case["expected_candidate"]["evidence"] for case in positive_ld)
    assert all(
        all(item["json_pointer"] for item in case["expected_candidate"]["evidence"]) for case in positive_ld
    )
    for case in cases:
        if case["citation_curriculum"]["representation"] == "plaintext":
            assert not case["snapshot"]["ld_json"]
            assert all(item["json_pointer"] is None for item in case["expected_candidate"]["evidence"])


def test_custom_schema_varies_typed_names_and_preserves_zero_false():
    cases = [build_case("train", "custom_v1", number)[1] for number in range(120)]
    assert len({case["schema_fingerprint"] for case in cases}) >= 15
    field_names = {tuple(field["name"] for field in case["request"]["fields"]) for case in cases}
    assert len(field_names) >= 15
    positive_even = next(
        case
        for case in cases
        if case["mode"] == "complete"
        and case["expected_candidate"]["data"][case["request"]["fields"][1]["name"]] == 0
    )
    data = positive_even["expected_candidate"]["data"]
    assert data[positive_even["request"]["fields"][3]["name"]] is False
    validate_cases(cases)


def test_layout_and_section_offsets_are_not_fixed_by_semantic_mode():
    cases = [build_case("train", "article_v1", number)[1] for number in range(120)]
    for mode in MODES:
        matching = [case for case in cases if case["mode"] == mode]
        assert len({case["template_id"] for case in matching}) >= 4
        assert len({case["citation_curriculum"]["source_noise_count"] for case in matching}) >= 4
    assert len({case["template_id"] for case in cases}) == 8


def test_fresh_twelve_per_schema_contains_every_mode_and_balanced_representation():
    cases = [build_case("test", schema, number)[1] for schema in SCHEMAS for number in range(12)]
    for schema in SCHEMAS:
        assert {case["mode"] for case in cases if case["schema_id"] == schema} == set(MODES)
    for mode in MODES:
        counts = Counter(
            case["citation_curriculum"]["representation"] for case in cases if case["mode"] == mode
        )
        assert len(set(counts.values())) == 1


def test_validation_source_is_exact_production_and_train_shuffle_preserves_ids():
    saw_shuffle = saw_normal = False
    for split in ("train", "valid"):
        for number in range(24):
            row, case = build_case(split, "page_v1", number)
            request = ExtractionRequest.model_validate(case["request"])
            spec = get_schema(request)
            snapshot = SourceSnapshot.model_validate_json(json.dumps(case["snapshot"]))
            extractor = HuggingFaceExtractor("offline", "offline", client=object()).for_request(request, spec)
            production = json.loads(extractor._source_content(snapshot))
            actual = json.loads(row["messages"][1]["content"].split("\n", 1)[1])
            assert row["messages"][0]["content"] == request_instruction(request, spec)
            assert actual["ld_json"] == production["ld_json"]
            assert sorted(actual["sections"], key=lambda section: section["id"]) == sorted(
                production["sections"], key=lambda section: section["id"]
            )
            if split == "valid" or not case["citation_curriculum"]["train_section_presentation_shuffled"]:
                assert actual == production
                saw_normal = saw_normal or split == "train"
            else:
                saw_shuffle = True
    assert saw_shuffle and saw_normal


def test_splits_have_disjoint_domains_entities_templates_and_source_hashes():
    cases = {
        split: [build_case(split, schema, number)[1] for schema in SCHEMAS for number in range(15)]
        for split in ("train", "valid", "test")
    }
    assert all(check_disjoint(cases).values())
    custom_names = {
        split: {
            field["name"]
            for case in rows
            if case["schema_id"] == "custom_v1"
            for field in case["request"]["fields"]
        }
        for split, rows in cases.items()
    }
    assert not custom_names["train"] & custom_names["valid"]
    assert not custom_names["train"] & custom_names["test"]
    assert not custom_names["valid"] & custom_names["test"]


def test_dataset_reproducible_private_and_manifest_cross_tabulation(tmp_path):
    kwargs = {"train_per_schema": 24, "valid_per_schema": 20, "test_per_schema": 12}
    one, two = tmp_path / "one", tmp_path / "two"
    manifest = generate_dataset(one, **kwargs)
    assert manifest == generate_dataset(two, **kwargs)
    assert manifest == generate_dataset(one, **kwargs)
    assert manifest["counts"] == {"train": 216, "valid": 180, "test": 108}
    for schema in SCHEMAS:
        assert len(manifest["representation_mode_counts"]["valid"][schema]) == 15
    assert set(manifest["citation_form_counts"]["train"]) == {"section_id", "json_pointer"}
    for name in manifest["files"]:
        assert (one / name).read_bytes() == (two / name).read_bytes()
        assert (one / name).stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="configuration differs"):
        generate_dataset(one, seed=1, **kwargs)
    checkout = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="outside the public source checkout"):
        generate_dataset(checkout / "training" / "bad-v3-output", **kwargs)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "unrelated").write_text("keep")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        generate_dataset(occupied, **kwargs)
