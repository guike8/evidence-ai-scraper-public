"""CPU-only transport semantics for exclusive saved evidence references."""

from __future__ import annotations

import copy
from datetime import UTC, datetime

import pytest
from jsonschema import Draft202012Validator

from ai_scraper.extraction.evidence_schema import constrain_evidence_references
from ai_scraper.extraction.huggingface import compile_provider_schema
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest, SourceSection, SourceSnapshot
from ai_scraper.validate import validate_candidate

SCHEMAS = (
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


@pytest.mark.parametrize("schema_id", SCHEMAS)
def test_local_adapter_uses_exclusive_references_for_every_request_schema(schema_id):
    request = ExtractionRequest(
        url="https://example.test/record",
        schema_id=schema_id,
        fields=[{"name": "active", "type": "boolean"}] if schema_id == "custom_v1" else [],
    )
    spec = get_schema(request)
    adapter = LocalExtractor("local-test").for_request(request, spec)
    evidence_schema = adapter.schema["$defs"]["EvidenceCandidate"]
    assert len(evidence_schema["anyOf"]) == 2
    validator = Draft202012Validator(adapter.schema)
    assert validator.is_valid(payload(spec, evidence()))
    assert not validator.is_valid(payload(spec, evidence("s0001", "/0/name")))


def compiled(schema_id="product_v1"):
    request = ExtractionRequest(
        url="https://example.test/record",
        schema_id=schema_id,
        fields=[{"name": "active", "type": "boolean"}] if schema_id == "custom_v1" else [],
    )
    spec = get_schema(request)
    return compile_provider_schema(spec.candidate_model.model_json_schema()), spec


def evidence(section_id="s0001", json_pointer=None):
    return {
        "field_path": "name",
        "section_id": section_id,
        "excerpt": "Public Folio",
        "json_pointer": json_pointer,
    }


def payload(spec, item):
    return {"data": spec.data_model().model_dump(mode="json"), "evidence": [item], "warnings": []}


@pytest.mark.parametrize("schema_id", SCHEMAS)
def test_all_registered_provider_schemas_constrain_only_canonical_evidence(schema_id):
    original, spec = compiled(schema_id)
    before = copy.deepcopy(original)
    constrained = constrain_evidence_references(original)
    assert original == before
    assert constrained is not original
    assert constrain_evidence_references(constrained) == constrained
    Draft202012Validator.check_schema(constrained)
    validator = Draft202012Validator(constrained)
    for item in (evidence(), evidence(None, "/0/name")):
        assert validator.is_valid(payload(spec, item))
    for item in (evidence("s0001", "/0/name"), evidence(None, None)):
        assert not validator.is_valid(payload(spec, item))
        assert Draft202012Validator(original).is_valid(payload(spec, item))
    changed = copy.deepcopy(constrained)
    changed["$defs"]["EvidenceCandidate"] = before["$defs"]["EvidenceCandidate"]
    assert changed == before
    # Recompilation exercises recursive provider keyword validation. No oneOf,
    # allOf, conditionals, dependent schemas, or external refs were introduced.
    assert compile_provider_schema(constrained) == constrained
    expected_order = ["field_path", "section_id", "excerpt", "json_pointer"]
    for branch in constrained["$defs"]["EvidenceCandidate"]["anyOf"]:
        assert list(branch["properties"]) == expected_order
        assert branch["required"] == expected_order
        assert branch["additionalProperties"] is False


@pytest.mark.parametrize(
    "mutator",
    [
        lambda item: item.update(extra="forbidden"),
        lambda item: item.update(field_path=7),
        lambda item: item.update(excerpt=False),
        lambda item: item.update(section_id=17),
        lambda item: item.update(json_pointer=[]),
        lambda item: item.pop("json_pointer"),
        lambda item: item.pop("section_id"),
    ],
)
def test_wrong_types_missing_keys_and_extra_properties_are_rejected(mutator):
    original, spec = compiled()
    item = evidence()
    mutator(item)
    assert not Draft202012Validator(constrain_evidence_references(original)).is_valid(payload(spec, item))


def test_preserves_other_business_constraints_and_all_property_orders():
    original, spec = compiled()
    business = original["$defs"]["ProductV1"]["properties"]
    business["currency"]["anyOf"][0]["enum"] = ["USD", "EUR"]
    business["price"]["anyOf"][0]["pattern"] = r"^\d+\.\d{2}$"
    definition = original["$defs"]["EvidenceCandidate"]
    definition["description"] = "Retain exact saved citations."
    definition["properties"]["field_path"]["minLength"] = 1
    definition["properties"]["excerpt"]["maxLength"] = 80
    definition["properties"]["section_id"]["description"] = "Saved section identifier."
    definition["properties"]["section_id"]["anyOf"][0].update(pattern=r"^s\d{4}$", minLength=5)
    definition["properties"]["json_pointer"]["anyOf"][0]["pattern"] = r"^/0/"
    constrained = constrain_evidence_references(original)
    assert constrained["$defs"]["ProductV1"] == original["$defs"]["ProductV1"]
    assert list(constrained["$defs"]["ProductV1"]["properties"]) == list(business)
    text, structured = constrained["$defs"]["EvidenceCandidate"]["anyOf"]
    assert text["description"] == definition["description"] == structured["description"]
    assert text["properties"]["field_path"] == definition["properties"]["field_path"]
    assert text["properties"]["excerpt"] == definition["properties"]["excerpt"]
    assert text["properties"]["section_id"] == {
        "type": "string",
        "pattern": r"^s\d{4}$",
        "minLength": 5,
        "description": "Saved section identifier.",
    }
    assert structured["properties"]["json_pointer"] == {"type": "string", "pattern": r"^/0/"}
    validator = Draft202012Validator(constrained)
    good = payload(spec, evidence())
    assert validator.is_valid(good)
    bad = copy.deepcopy(good)
    bad["evidence"][0]["section_id"] = "unknown"
    assert not validator.is_valid(bad)
    bad = copy.deepcopy(good)
    bad["data"]["currency"] = "GBP"
    assert not validator.is_valid(bad)
    bad = copy.deepcopy(good)
    bad["data"]["price"] = "1.234"
    assert not validator.is_valid(bad)


def test_unrelated_names_shapes_and_unreferenced_definitions_are_untouched():
    original, _ = compiled()
    altered = copy.deepcopy(original)
    altered["$defs"]["Evidence"] = altered["$defs"].pop("EvidenceCandidate")
    altered["properties"]["evidence"]["items"]["$ref"] = "#/$defs/Evidence"
    assert constrain_evidence_references(altered) == altered
    altered = copy.deepcopy(original)
    altered["$defs"]["EvidenceCandidate"]["properties"]["unrelated"] = {"type": "string"}
    assert constrain_evidence_references(altered) == altered
    altered = copy.deepcopy(original)
    altered["properties"]["evidence"]["items"] = {"type": "string"}
    assert constrain_evidence_references(altered) == altered
    assert constrain_evidence_references({"type": "object", "properties": {}}) == {
        "type": "object",
        "properties": {},
    }


def test_malformed_canonical_shape_fails_without_weakening_constraints():
    original, _ = compiled()
    original["$defs"]["EvidenceCandidate"]["properties"]["section_id"] = {"type": "string"}
    with pytest.raises(ValueError, match="nullable strings"):
        constrain_evidence_references(original)
    original, _ = compiled()
    original["$defs"]["EvidenceCandidate"]["additionalProperties"] = True
    with pytest.raises(ValueError, match="fixed four-property"):
        constrain_evidence_references(original)
    original, _ = compiled()
    reference = original["$defs"]["EvidenceCandidate"]["properties"]["section_id"]
    reference["enum"] = ["s0001", "s0002", None]
    reference["anyOf"][0]["enum"] = ["s0001"]
    with pytest.raises(ValueError, match="Conflicting evidence reference constraints"):
        constrain_evidence_references(original)


def test_existing_both_locator_candidate_newly_rejected_by_transport_local_validation_unchanged():
    original, spec = compiled()
    snapshot = SourceSnapshot(
        requested_url="https://example.test/record",
        final_url="https://example.test/record",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        raw_record={},
        source_hash="a" * 64,
        sections=[SourceSection(id="s0001", text="Public Folio")],
        ld_json=[{"@type": "Product", "name": "Public Folio"}],
    )
    candidate = payload(spec, evidence("s0001", "/0/name"))
    candidate["data"]["name"] = "Public Folio"
    before = validate_candidate(candidate, snapshot)
    assert before.status == "failed"
    assert any(error.startswith("evidence:name:") for error in before.errors)
    assert Draft202012Validator(original).is_valid(candidate)
    assert not Draft202012Validator(constrain_evidence_references(original)).is_valid(candidate)
    assert validate_candidate(candidate, snapshot) == before
    # A valid reference is still checked for occurrence by the unchanged local
    # validator: transport validity alone cannot make invented facts acceptable.
    candidate["evidence"][0]["json_pointer"] = None
    assert Draft202012Validator(constrain_evidence_references(original)).is_valid(candidate)
    assert validate_candidate(candidate, snapshot).status == "partial"
    candidate["evidence"][0]["excerpt"] = "Invented description"
    assert Draft202012Validator(constrain_evidence_references(original)).is_valid(candidate)
    assert validate_candidate(candidate, snapshot).status == "failed"
