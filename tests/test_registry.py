from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from ai_scraper.contracts import ScraperError
from ai_scraper.registry import get_schema, list_schemas
from ai_scraper.schemas import CustomField, ExtractionRequest


def request(schema_id="page_v1", **kwargs):
    return ExtractionRequest(url="https://example.com", schema_id=schema_id, **kwargs)


def test_registered_schemas_and_aliases_are_trusted_and_immutable():
    specs = list_schemas()
    assert {item["schema_id"] for item in specs} == {
        "page_v1",
        "article_v1",
        "organization_v1",
        "person_v1",
        "job_v1",
        "event_v1",
        "listing_v1",
        "product_v1",
        "custom_v1",
    }
    assert get_schema(request("company")).schema_id == "organization_v1"
    assert get_schema(request("auto")).schema_id == "page_v1"
    spec = get_schema(request())
    with pytest.raises(FrozenInstanceError):
        spec.schema_id = "untrusted"
    with pytest.raises(ScraperError, match="Unknown registered schema"):
        get_schema(request("untrusted"))


def test_custom_contract_has_independent_types_completeness_and_cache_identity():
    fields = [
        CustomField(name="capacity", type="integer", critical=True),
        CustomField(name="tags", type="string_list", required=False),
    ]
    first = get_schema(request("custom_v1", fields=fields))
    assert first.required_fields == ("capacity",)
    assert first.critical_fields == ("capacity",)
    assert first.data_model.model_validate({"capacity": 12}).model_dump() == {
        "capacity": 12,
        "tags": [],
    }
    for value in ("12", True, 12.5):
        with pytest.raises(ValidationError):
            first.data_model.model_validate({"capacity": value})
    second = get_schema(request("custom_v1", fields=[CustomField(name="capacity", type="string")]))
    assert second.fingerprint != first.fingerprint
    assert get_schema(request("custom_v1", fields=fields)).fingerprint == first.fingerprint
    with pytest.raises(ValidationError):
        first.data_model.model_validate({"capacity": 12, "other": "unrequested"})


def test_custom_contract_rejects_unsafe_ambiguous_or_nonfinite_definitions():
    for name in ("model_dump", "__class__", "provenance", "status", "a.b"):
        with pytest.raises(ValidationError):
            CustomField(name=name)
    with pytest.raises(ScraperError):
        get_schema(request("custom_v1"))
    with pytest.raises(ScraperError):
        get_schema(request("custom_v1", fields=[CustomField(name="x"), CustomField(name="x")]))
    with pytest.raises(ScraperError):
        get_schema(request(fields=[CustomField(name="x")]))
    number = get_schema(request("custom_v1", fields=[CustomField(name="x", type="number")]))
    for value in (float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            number.data_model.model_validate({"x": value})


def test_general_result_preserves_generic_data_and_typed_product():
    from ai_scraper.schemas import ExtractionResult, ProductV1, Provenance

    result = ExtractionResult(
        job_id="x",
        status="success",
        data={"name": "Generic"},
        provenance=Provenance(requested_url="https://example.com", schema_version="custom_v1"),
    )
    assert type(result.data) is dict
    assert result.data == {"name": "Generic"}
    restored = ExtractionResult.model_validate_json(result.model_dump_json())
    assert restored.data == {"name": "Generic"}
    product = result.model_copy(update={"data": ProductV1(name="Product")})
    assert isinstance(product.data, ProductV1)
