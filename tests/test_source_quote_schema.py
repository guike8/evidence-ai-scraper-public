"""Literal quote/location constraints leave business validation independent."""

from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from ai_scraper.extraction.huggingface import compile_provider_schema
from ai_scraper.extraction.source_quote_schema import (
    MAX_SOURCE_TEXT_BYTES,
    SourceQuoteSchemaError,
    bind_source_quote_evidence,
)
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest

FAMILIES = (
    "product_v1",
    "page_v1",
    "article_v1",
    "organization_v1",
    "person_v1",
    "job_v1",
    "event_v1",
    "listing_v1",
    "custom_v1",
)


def compiled(family="product_v1"):
    request = ExtractionRequest(
        url="https://quote.test/",
        schema_id=family,
        fields=[{"name": "count", "type": "integer"}] if family == "custom_v1" else [],
    )
    spec = get_schema(request)
    return compile_provider_schema(spec.candidate_model.model_json_schema()), spec


def source():
    return {
        "sections": [{"id": "s0042", "text": "Public Folio"}, {"id": "s0007", "text": "Count: 37"}],
        "ld_json": [{"name": "Structured Folio", "count": 37, "active": True}],
        "excluded_sections": ["s0999"],
    }


def item(quote="Public Folio", section="s0042", pointer=None, path="name"):
    return {"field_path": path, "excerpt": quote, "section_id": section, "json_pointer": pointer}


def payload(spec, evidence):
    return {"data": spec.data_model().model_dump(mode="json"), "evidence": evidence, "warnings": []}


@pytest.mark.parametrize("family", FAMILIES)
def test_all_families_bind_quotes_and_locators_without_mutating_business_contract(family):
    schema, spec = compiled(family)
    before = copy.deepcopy(schema)
    included = source()
    included_before = copy.deepcopy(included)
    bound = bind_source_quote_evidence(schema, included)
    assert schema == before and included == included_before
    Draft202012Validator.check_schema(bound.schema)
    validator = Draft202012Validator(bound.schema)
    assert validator.is_valid(payload(spec, [item()]))
    assert validator.is_valid(payload(spec, [item("Structured Folio", None, "/0/name")]))
    assert validator.is_valid(payload(spec, [item("37", "s0007", path="count")]))
    assert validator.is_valid(payload(spec, [item("true", None, "/0/active", path="active")]))
    for bad in (
        item(section="s0007"),
        item(section="s0999"),
        item("Invented quote"),
        item("Structured Folio", None, "/0/count"),
        item(pointer="/0/name"),
        item(section=None),
    ):
        assert not validator.is_valid(payload(spec, [bad]))
    restored = copy.deepcopy(bound.schema)
    restored["$defs"]["EvidenceCandidate"] = before["$defs"]["EvidenceCandidate"]
    assert restored == before
    for branch in bound.schema["$defs"]["EvidenceCandidate"]["anyOf"]:
        assert list(branch["properties"]) == ["field_path", "excerpt", "section_id", "json_pointer"]
        assert branch["required"] == list(branch["properties"])
        assert branch["additionalProperties"] is False


def test_literal_strings_unicode_escaped_pointers_finite_numbers_and_booleans():
    schema, spec = compiled()
    included = {
        "sections": [{"id": "section-雪", "text": 'Exact "quoted" café\nsecond line'}],
        "ld_json": [{"a/b~c": ["雪", False, 0, -2.5, 1e-5, None]}],
    }
    bound = bind_source_quote_evidence(schema, included)
    validator = Draft202012Validator(bound.schema)
    assert validator.is_valid(payload(spec, [item('Exact "quoted" café\nsecond line', "section-雪")]))
    for index, quote in enumerate(("雪", "false", "0", "-2.5", "1e-05")):
        assert validator.is_valid(payload(spec, [item(quote, None, f"/0/a~1b~0c/{index}")]))
    assert not validator.is_valid(payload(spec, [item("null", None, "/0/a~1b~0c/5")]))
    assert bound.metadata()["json_scalar_quote_count"] == 5


def test_same_quote_at_distinct_locations_preserves_only_observed_pairs():
    schema, spec = compiled()
    included = {
        "sections": [{"id": "a", "text": "Same"}, {"id": "b", "text": "Same"}],
        "ld_json": [{"name": "Same", "other": "Different"}],
    }
    bound = bind_source_quote_evidence(schema, included)
    validator = Draft202012Validator(bound.schema)
    for evidence in (item("Same", "a"), item("Same", "b"), item("Same", None, "/0/name")):
        assert validator.is_valid(payload(spec, [evidence]))
    assert not validator.is_valid(payload(spec, [item("Same", None, "/0/other")]))
    assert bound.metadata()["quote_pair_count"] == bound.metadata()["branch_count"] == 4


@pytest.mark.parametrize(
    ("text", "suffix"),
    [
        ("Count: 37", "37"),
        ("Open: true", "true"),
        ("Ready: false", "false"),
        ("Amount: -2.50e+3", "-2.50e+3"),
        ("Year: 2031", "2031"),
        ("Count: 0037", None),
        ("Count: 37 items", None),
        ("Count: 37.0 units", None),
        ("Open: True", None),
        ("Amount: 1e9999", None),
        ("No colon 37", None),
        ("Count: 37\nOther text", None),
        ("URL: https://quote.test/37", None),
    ],
)
def test_only_exact_json_numeric_or_boolean_colon_suffixes_are_admitted(text, suffix):
    schema, spec = compiled()
    bound = bind_source_quote_evidence(schema, {"sections": [{"id": "a", "text": text}], "ld_json": []})
    validator = Draft202012Validator(bound.schema)
    assert validator.is_valid(payload(spec, [item(text, "a")]))
    if suffix is not None:
        assert validator.is_valid(payload(spec, [item(suffix, "a")]))
        assert bound.metadata()["scalar_suffix_quote_count"] == 1
    else:
        assert bound.metadata()["scalar_suffix_quote_count"] == 0


def test_long_and_empty_literals_are_reported_without_clipping():
    schema, spec = compiled()
    included = {
        "sections": [{"id": "long", "text": "x" * 513}, {"id": "empty", "text": ""}],
        "ld_json": [{"long": "y" * 513, "empty": "", "name": "Usable"}],
    }
    bound = bind_source_quote_evidence(schema, included)
    metadata = bound.metadata()
    assert metadata["coverage"] == "partial"
    assert metadata["excluded_long_sections"] == metadata["excluded_long_scalars"] == 1
    assert metadata["excluded_empty_literals"] == 2
    assert metadata["branch_count"] == 1
    validator = Draft202012Validator(bound.schema)
    assert validator.is_valid(payload(spec, [item("Usable", None, "/0/name")]))
    assert not validator.is_valid(payload(spec, [item("x" * 512, "long")]))
    assert not validator.is_valid(payload(spec, [item("y" * 512, None, "/0/long")]))


@pytest.mark.parametrize("prefix", ["href: ", "src: "])
@pytest.mark.parametrize(
    "literal", ["/relative/path", "https://quote.test/a", "mailto:public@quote.test", "tel:+15550000"]
)
def test_known_url_attribute_values_are_literal_quotes_at_the_same_location(prefix, literal):
    schema, spec = compiled()
    included = {"sections": [{"id": "attribute", "text": prefix + literal}], "ld_json": []}
    bound = bind_source_quote_evidence(schema, included)
    validator = Draft202012Validator(bound.schema)
    assert validator.is_valid(payload(spec, [item(literal, "attribute")]))
    assert not validator.is_valid(payload(spec, [item(literal, "other")]))
    assert bound.metadata()["attribute_suffix_quote_count"] == 1


def test_arbitrary_string_label_suffixes_are_not_quote_candidates():
    schema, spec = compiled()
    bound = bind_source_quote_evidence(
        schema, {"sections": [{"id": "a", "text": "name: Folio"}], "ld_json": []}
    )
    validator = Draft202012Validator(bound.schema)
    assert not validator.is_valid(payload(spec, [item("Folio", "a")]))
    assert bound.metadata()["attribute_suffix_quote_count"] == 0


def test_literal_quote_binding_does_not_certify_business_field_support():
    from ai_scraper.normalize import normalize_record
    from ai_scraper.validate import validate_candidate

    schema, spec = compiled()
    request = ExtractionRequest(url="https://quote.test/", schema_id="product_v1")
    snapshot = normalize_record({"page_html": "<html><body><h1>Alpha</h1><p>Beta</p></body></html>"}, request)
    included = {
        "sections": [{"id": s.id, "text": s.text} for s in snapshot.sections],
        "ld_json": snapshot.ld_json,
    }
    bound = bind_source_quote_evidence(schema, included)
    wrong_section = next(section.id for section in snapshot.sections if section.text == "Beta")
    candidate = payload(spec, [item("Beta", wrong_section)])
    candidate["data"]["name"] = "Alpha"
    assert Draft202012Validator(bound.schema).is_valid(candidate)
    assessment = validate_candidate(candidate, snapshot)
    assert assessment.status == "failed"
    assert any(error.startswith("evidence:name:") for error in assessment.errors)


def test_empty_catalog_permits_empty_evidence_only_and_retains_array_constraints():
    schema, spec = compiled()
    schema["properties"]["evidence"]["maxItems"] = 10
    bound = bind_source_quote_evidence(schema, {"sections": [], "ld_json": []})
    Draft202012Validator.check_schema(bound.schema)
    validator = Draft202012Validator(bound.schema)
    assert validator.is_valid(payload(spec, []))
    assert not validator.is_valid(payload(spec, [item()]))
    assert bound.metadata()["coverage"] == "empty"
    schema["properties"]["evidence"]["minItems"] = 1
    with pytest.raises(SourceQuoteSchemaError, match="Required evidence"):
        bind_source_quote_evidence(schema, {"sections": [], "ld_json": []})


def test_original_scalar_and_object_constraints_are_preserved_and_intersected():
    schema, spec = compiled()
    definition = schema["$defs"]["EvidenceCandidate"]
    definition["description"] = "Trusted annotation"
    definition["properties"]["field_path"]["enum"] = ["name"]
    definition["properties"]["excerpt"]["maxLength"] = 12
    definition["properties"]["section_id"]["anyOf"][0]["enum"] = ["s0042"]
    definition["properties"]["json_pointer"]["anyOf"][0]["pattern"] = r"^/0/name$"
    bound = bind_source_quote_evidence(schema, source())
    assert bound.metadata()["coverage"] == "partial"
    assert bound.metadata()["excluded_by_schema"] > 0
    validator = Draft202012Validator(bound.schema)
    assert validator.is_valid(payload(spec, [item()]))
    assert not validator.is_valid(payload(spec, [item(path="other")]))
    for branch in bound.schema["$defs"]["EvidenceCandidate"]["anyOf"]:
        assert branch["description"] == "Trusted annotation"
        assert branch["properties"]["excerpt"]["maxLength"] == 12
        assert branch["properties"]["field_path"]["enum"] == ["name"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda schema: schema["properties"].pop("evidence"),
        lambda schema: schema["properties"]["evidence"].update(items={"$ref": "#/$defs/Other"}),
        lambda schema: schema["$defs"]["EvidenceCandidate"].update(additionalProperties=True),
        lambda schema: schema["$defs"]["EvidenceCandidate"].update(required=["field_path"]),
        lambda schema: schema["$defs"]["EvidenceCandidate"]["properties"].pop("excerpt"),
    ],
)
def test_malformed_canonical_contract_is_rejected(mutate):
    schema, _ = compiled()
    mutate(schema)
    with pytest.raises(SourceQuoteSchemaError):
        bind_source_quote_evidence(schema, source())


@pytest.mark.parametrize(
    "included",
    [
        {"sections": [], "ld_json": [{"bad": float("nan")}]},
        {"sections": [{"id": "a", "text": "One"}, {"id": "a", "text": "Two"}], "ld_json": []},
        {"sections": "not an array", "ld_json": []},
        {"sections": [], "ld_json": [object()]},
        {"sections": [{"id": "a", "text": "\ud800"}], "ld_json": []},
    ],
)
def test_malformed_source_is_rejected_without_echoing_content(included):
    schema, _ = compiled()
    with pytest.raises(SourceQuoteSchemaError) as error:
        bind_source_quote_evidence(schema, included)
    assert "One" not in str(error.value) and "nan" not in str(error.value)


def test_branch_schema_source_node_and_depth_budgets_fail_instead_of_truncating():
    schema, _ = compiled()
    with pytest.raises(SourceQuoteSchemaError, match="branch budget"):
        bind_source_quote_evidence(schema, source(), max_branches=1)
    with pytest.raises(SourceQuoteSchemaError, match="schema byte budget"):
        bind_source_quote_evidence(schema, source(), max_schema_bytes=50)
    with pytest.raises(SourceQuoteSchemaError, match="source text byte budget"):
        bind_source_quote_evidence(
            schema, {"sections": [{"id": "a", "text": "x" * (MAX_SOURCE_TEXT_BYTES + 1)}], "ld_json": []}
        )
    too_deep = "value"
    for _ in range(65):
        too_deep = [too_deep]
    for included in (
        {"sections": [], "ld_json": [too_deep]},
        {"sections": [], "ld_json": [[None] * 32768]},
    ):
        with pytest.raises(SourceQuoteSchemaError, match="catalog"):
            bind_source_quote_evidence(schema, included)


@pytest.mark.parametrize("limit", [{"max_branches": True}, {"max_quote_chars": 513}, {"max_schema_bytes": 0}])
def test_caller_cannot_expand_hard_limits(limit):
    schema, _ = compiled()
    with pytest.raises(SourceQuoteSchemaError, match="Invalid"):
        bind_source_quote_evidence(schema, source(), **limit)


def test_receipt_is_deterministic_private_and_caller_owned():
    schema, _ = compiled()
    first = bind_source_quote_evidence(schema, source())
    second = bind_source_quote_evidence(schema, source())
    assert first.schema == second.schema
    assert first.metadata() == second.metadata()
    receipt = first.metadata()
    assert "Folio" not in json.dumps(receipt)
    receipt["coverage"] = "changed"
    assert first.metadata()["coverage"] == "complete"
