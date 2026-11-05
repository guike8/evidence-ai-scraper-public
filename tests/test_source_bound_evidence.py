"""Actual source catalogs narrow grammar without changing business validators."""

from __future__ import annotations

import asyncio
import copy
import json
from datetime import UTC, datetime

import httpx
import pytest
from jsonschema import Draft202012Validator

from ai_scraper.config import Settings
from ai_scraper.contracts import RetrievalOutcome, ScraperError
from ai_scraper.extraction import evidence_schema
from ai_scraper.extraction.evidence_schema import (
    SOURCE_BOUND_EVIDENCE_VERSION,
    EvidenceCatalogLimitError,
    bind_source_evidence_references,
    source_evidence_catalog,
)
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.normalize import normalize_record
from ai_scraper.pipeline import Pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest, ProductV1, SourceSection, SourceSnapshot
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
URL = "https://citation.test/record"


def source(section_ids=("s0007", "s0019"), ld_json=None):
    return SourceSnapshot(
        requested_url=URL,
        final_url=URL,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        raw_record={},
        sections=[SourceSection(id=item, text="Public Folio") for item in section_ids],
        ld_json=ld_json if ld_json is not None else [{"name": "Public Folio"}],
        source_hash="a" * 64,
    )


def adapter_for(schema_id="product_v1", **kwargs):
    request = ExtractionRequest(
        url=URL,
        schema_id=schema_id,
        fields=[{"name": "active", "type": "boolean"}] if schema_id == "custom_v1" else [],
    )
    spec = get_schema(request)
    adapter = LocalExtractor("local-model", client=object(), source_bound_evidence=True, **kwargs)
    return adapter.for_request(request, spec), spec


def candidate(spec, section_id="s0007", pointer=None, excerpt="Public Folio"):
    return {
        "data": spec.data_model().model_dump(mode="json"),
        "evidence": [
            {
                "field_path": "name",
                "section_id": section_id,
                "json_pointer": pointer,
                "excerpt": excerpt,
            }
        ],
        "warnings": [],
    }


@pytest.mark.parametrize("schema_id", SCHEMAS)
def test_every_local_schema_limits_locators_to_actual_included_source(schema_id):
    adapter, spec = adapter_for(schema_id)
    template = copy.deepcopy(adapter.schema)
    snapshot = source()
    sent = json.loads(adapter._source_content(snapshot))
    actual_ids = [section["id"] for section in sent["sections"]]
    Draft202012Validator.check_schema(adapter.schema)
    validator = Draft202012Validator(adapter.schema)
    assert validator.is_valid(candidate(spec, actual_ids[0]))
    assert validator.is_valid(candidate(spec, None, "/0/name"))
    assert not validator.is_valid(candidate(spec, "s1"))
    assert not validator.is_valid(candidate(spec, None, "/0/invented"))
    assert not validator.is_valid(candidate(spec, actual_ids[0], "/0/name"))
    assert not validator.is_valid(candidate(spec, None, None))
    assert adapter._evidence_schema_template == template
    assert adapter.last_evidence_catalog["policy_version"] == SOURCE_BOUND_EVIDENCE_VERSION
    assert adapter.last_evidence_catalog["section_id_count"] == len(actual_ids)


@pytest.mark.parametrize(
    "section_ids,ld_json,form",
    [
        (("s0100",), [], "text"),
        ((), [{"name": "Public Folio"}], "structured"),
        ((), [], "empty"),
        ((), [{"missing": None}], "empty"),
    ],
)
def test_absent_representations_cannot_be_fabricated(section_ids, ld_json, form):
    adapter, spec = adapter_for()
    adapter._source_content(source(section_ids, ld_json))
    validator = Draft202012Validator(adapter.schema)
    assert validator.is_valid({"data": spec.data_model().model_dump(), "evidence": [], "warnings": []})
    assert validator.is_valid(candidate(spec, "s0100")) is (form == "text")
    assert validator.is_valid(candidate(spec, None, "/0/name")) is (form == "structured")
    if form == "empty":
        assert adapter.schema["properties"]["evidence"]["maxItems"] == 0


def test_catalog_uses_scalar_leaves_and_rfc6901_escaping_without_reinterpreting_keys():
    catalog = source_evidence_catalog(
        {
            "sections": [{"id": "actual-é", "text": "Facts"}],
            "ld_json": [{"a/b": {"~key": ["value", 0, False, 1.5, None]}, "": "blank key"}],
        }
    )
    assert catalog.section_ids == ("actual-é",)
    assert catalog.json_pointers == (
        "/0/a~1b/~0key/0",
        "/0/a~1b/~0key/1",
        "/0/a~1b/~0key/2",
        "/0/a~1b/~0key/3",
        "/0/",
    )
    assert "/0/a~1b/~0key/4" not in catalog.json_pointers
    assert (
        catalog.metadata()
        == source_evidence_catalog(
            {
                "sections": [{"id": "actual-é", "text": "Other wording"}],
                "ld_json": [{"a/b": {"~key": ["other", 0, False, 1.5, None]}, "": "blank key"}],
            }
        ).metadata()
    )


@pytest.mark.parametrize(
    "change",
    [
        {"MAX_CATALOG_LOCATORS": 1},
        {"MAX_CATALOG_BYTES": 2},
        {"MAX_CATALOG_NODES": 2},
        {"MAX_CATALOG_DEPTH": 1},
        {"MAX_LOCATOR_LENGTH": 2},
    ],
)
def test_every_catalog_budget_fails_closed_instead_of_truncating(monkeypatch, change):
    for name, value in change.items():
        monkeypatch.setattr(evidence_schema, name, value)
    with pytest.raises(EvidenceCatalogLimitError):
        source_evidence_catalog(
            {
                "sections": [{"id": "s0001", "text": "Facts"}],
                "ld_json": [{"nested": {"value": 17}}],
            }
        )


@pytest.mark.parametrize(
    "source_payload",
    [
        {"sections": [], "ld_json": {}},
        {"sections": [{"id": "s1", "text": "A"}, {"id": "s1", "text": "B"}], "ld_json": []},
        {"sections": [{"id": "", "text": "A"}], "ld_json": []},
        {"sections": [], "ld_json": [{"number": float("inf")}]},
    ],
)
def test_malformed_source_catalogs_raise_without_echoing_source(source_payload):
    with pytest.raises((TypeError, ValueError)):
        source_evidence_catalog(source_payload)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "properties": {}},
        {"type": "object", "properties": None},
        {"type": "object", "properties": {"evidence": None}},
    ],
)
def test_unknown_evidence_shapes_fail_explicitly_without_changing_existing_helper(schema):
    catalog = source_evidence_catalog({"sections": [], "ld_json": []})
    assert evidence_schema.constrain_evidence_references(schema) == schema
    with pytest.raises((TypeError, ValueError), match="canonical evidence array"):
        bind_source_evidence_references(schema, catalog)


def test_catalog_does_not_weaken_existing_enums_or_business_constraints():
    adapter, spec = adapter_for()
    original = copy.deepcopy(adapter.schema)
    original["$defs"]["EvidenceCandidate"]["anyOf"][0]["properties"]["section_id"]["enum"] = ["s0019"]
    original["$defs"]["ProductV1"]["properties"]["currency"]["anyOf"][0]["enum"] = ["USD"]
    before = copy.deepcopy(original)
    catalog = source_evidence_catalog(
        {
            "sections": [{"id": item, "text": "Public Folio"} for item in ("s0007", "s0019")],
            "ld_json": [],
        }
    )
    result = bind_source_evidence_references(original, catalog)
    assert original == before
    validator = Draft202012Validator(result)
    assert validator.is_valid(candidate(spec, "s0019"))
    assert not validator.is_valid(candidate(spec, "s0007"))
    populated = candidate(spec, "s0019")
    populated["data"]["currency"] = "EUR"
    assert not validator.is_valid(populated)


def test_included_view_catalog_excludes_deduplicated_and_out_of_scope_sections():
    navigation = "".join(f"<p>Navigation destination number {index}.</p>" for index in range(2100))
    request = ExtractionRequest(url=URL, schema_id="page_v1")
    snapshot = normalize_record(
        {
            "page_html": f"<title>Research brief</title><nav>{navigation}</nav>"
            "<main><h1>Research brief</h1><p>Volunteer count: 37.</p></main>"
        },
        request,
    )
    adapter, spec = adapter_for("page_v1")
    sent = json.loads(adapter._source_content(snapshot))
    assert adapter.last_source_view["scope"] == "main"
    assert adapter.last_source_view["out_of_scope_section_ids"]
    included = {section["id"] for section in sent["sections"]}
    assert included == set(adapter.last_source_view["included_section_ids"])
    enum = adapter.schema["$defs"]["EvidenceCandidate"]["anyOf"][0]["properties"]["section_id"]["enum"]
    assert set(enum) == included
    excluded = adapter.last_source_view["out_of_scope_section_ids"][0]
    assert not Draft202012Validator(adapter.schema).is_valid(candidate(spec, excluded))


async def test_catalog_exhaustion_is_detected_before_product_model_call(monkeypatch):
    monkeypatch.setattr(evidence_schema, "MAX_CATALOG_LOCATORS", 1)
    calls = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: calls.append(request))
    ) as client:
        extractor = LocalExtractor("local-model", client=client, source_bound_evidence=True)
        for action in (extractor.prepare_source, extractor._source_content):
            with pytest.raises(ScraperError) as error:
                action(source())
            assert error.value.error.code == "CONTENT_TOO_LARGE"
        with pytest.raises(ScraperError) as error:
            await extractor.extract(source())
        assert error.value.error.code == "CONTENT_TOO_LARGE"
    assert not calls


def test_failed_source_preparation_does_not_retain_previous_source_locators(monkeypatch):
    adapter, _ = adapter_for()
    template = copy.deepcopy(adapter.schema)
    adapter._source_content(source(("previous-source",), []))
    assert adapter.schema != template
    monkeypatch.setattr(evidence_schema, "MAX_CATALOG_LOCATORS", 0)
    with pytest.raises(ScraperError):
        adapter.prepare_source(source(("new-source",), []))
    assert adapter.schema == template
    assert adapter.last_evidence_catalog is None


async def test_sequential_snapshots_rebuild_grammar_and_preserve_payloads():
    calls = []
    returned = {"data": {"name": "Public Folio"}, "evidence": [], "warnings": []}

    def respond(request):
        calls.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(returned)}}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        extractor = LocalExtractor("local-model", client=client, source_bound_evidence=True)
        template = copy.deepcopy(extractor.schema)
        for item in (source(("actual-first",), []), source(("actual-second",), [{"other": 17}])):
            assert (await extractor.extract(item)).payload == returned
        assert extractor._evidence_schema_template == template
    first = calls[0]["response_format"]["json_schema"]["schema"]
    second = calls[1]["response_format"]["json_schema"]["schema"]
    first_branches = first["$defs"]["EvidenceCandidate"]["anyOf"]
    second_branches = second["$defs"]["EvidenceCandidate"]["anyOf"]
    assert len(first_branches) == 1
    assert first_branches[0]["properties"]["section_id"]["enum"] == ["actual-first"]
    assert second_branches[0]["properties"]["section_id"]["enum"] == ["actual-second"]
    assert second_branches[1]["properties"]["json_pointer"]["enum"] == ["/0/other"]
    assert calls[0]["structured_outputs"] == {"json_schema": first}
    assert calls[1]["structured_outputs"] == {"json_schema": second}


async def test_concurrent_request_views_never_share_bound_source_grammar():
    calls = []

    async def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        base = LocalExtractor("local-model", client=client, source_bound_evidence=True)
        template = copy.deepcopy(base.schema)
        scoped = []
        for schema_id in ("article_v1", "person_v1"):
            request = ExtractionRequest(url=URL, schema_id=schema_id)
            scoped.append(base.for_request(request, get_schema(request)))
        await asyncio.gather(
            *[
                adapter.extract(source((f"actual-{index}",), [{f"property-{index}": "value"}]))
                for index, adapter in enumerate(scoped)
            ]
        )
        assert base.schema == template and base.last_evidence_catalog is None
    for index, body in enumerate(calls):
        branches = body["response_format"]["json_schema"]["schema"]["$defs"]["EvidenceCandidate"]["anyOf"]
        assert branches[0]["properties"]["section_id"]["enum"] == [f"actual-{index}"]
        assert branches[1]["properties"]["json_pointer"]["enum"] == [f"/0/property-{index}"]
    assert scoped[0]._evidence_schema_template is not scoped[1]._evidence_schema_template


def test_bound_locators_still_require_verbatim_quotes_and_correct_field_support():
    adapter, spec = adapter_for()
    snapshot = source(("s0007",), [{"@type": "Product", "name": "Public Folio", "description": "Other text"}])
    adapter._source_content(snapshot)
    payload = candidate(spec, None, "/0/description", "Other text")
    payload["data"]["name"] = "Public Folio"
    assert Draft202012Validator(adapter.schema).is_valid(payload)
    assert validate_candidate(payload, snapshot).status == "failed"
    payload["evidence"][0].update(section_id="s0007", json_pointer=None, excerpt="Invented excerpt")
    assert Draft202012Validator(adapter.schema).is_valid(payload)
    assert validate_candidate(payload, snapshot).status == "failed"


async def test_citation_policy_separates_product_cache_and_keeps_resumes_idempotent(tmp_path):
    record = {"markdown": "Product: Public Folio\nPrice: USD 20.00", "final_url": URL}
    calls = []

    class Retriever:
        count = 0

        async def retrieve(self, request, job, checkpoint, wait_seconds):
            self.count += 1
            checkpoint({"state": "fetched", "record": record, "retrieval_method": "fixture_replay"})
            return RetrievalOutcome(status="ready", record=record, retrieval_method="fixture_replay")

    def respond(request):
        body = json.loads(request.content)
        calls.append(body)
        sent = json.loads(body["messages"][1]["content"].split("\n", 1)[1])
        data = {"name": "Public Folio", "price": "20.00", "currency": "USD"}
        references = []
        for name, value in data.items():
            section = next(section for section in sent["sections"] if value in section["text"])
            references.append(
                {
                    "field_path": name,
                    "section_id": section["id"],
                    "excerpt": section["text"],
                    "json_pointer": None,
                }
            )
        payload = {"data": ProductV1(**data).model_dump(), "warnings": [], "evidence": references}
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}],
            },
        )

    settings = Settings(state_dir=tmp_path / "state", retrieval_backend="direct")
    retriever = Retriever()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        old = Pipeline(
            settings,
            retriever=retriever,
            policy=URLPolicy(["citation.test"], resolve_dns=False),
            live=False,
            inference=LocalExtractor("same-model", client=client),
        )
        result = await old.extract_url(
            ExtractionRequest(url=URL, instruction="Extract the exact product facts.")
        )
        assert result.status == "success" and len(calls) == 1
        await old.close()
        new = Pipeline(
            settings,
            retriever=retriever,
            policy=URLPolicy(["citation.test"], resolve_dns=False),
            live=False,
            inference=LocalExtractor("same-model", client=client, source_bound_evidence=True),
        )
        changed = await new.resume(result.job_id)
        assert changed.status == "success" and len(calls) == 2
        counts = new.store.get(result.job_id)["inference_counts"]
        assert len(counts) == 2 and sorted(counts.values()) == [1, 1]
        resumed = await new.resume(result.job_id)
        assert resumed.status == "success" and resumed.usage.cache_hit and len(calls) == 2
        assert retriever.count == 1
        await new.close()


async def test_dense_product_catalog_fails_before_any_reservation_and_stays_resumable(tmp_path, monkeypatch):
    record = {
        "final_url": URL,
        "ld_json": [{"@type": "Product", "name": "Public Folio", "values": [0] * 4097}],
    }
    request = ExtractionRequest(url=URL)
    snapshot = normalize_record(record, request)
    legacy = LocalExtractor("fixture-model", client=object())
    assert len(legacy._source_content(snapshot)) < 48000
    model_calls, inference_reservations, spend_reservations = [], [], []

    class Retriever:
        calls = 0

        async def retrieve(self, request, job, checkpoint, wait_seconds):
            self.calls += 1
            checkpoint({"state": "fetched", "record": record, "retrieval_method": "direct_http"})
            return RetrievalOutcome(status="ready", record=record, retrieval_method="direct_http")

    def reject_model_call(http_request):
        model_calls.append(http_request)
        raise AssertionError("The complete product catalog must be checked before inference")

    retriever = Retriever()
    async with httpx.AsyncClient(transport=httpx.MockTransport(reject_model_call)) as client:
        pipeline = Pipeline(
            Settings(
                state_dir=tmp_path / "state",
                retrieval_backend="direct",
                inference_backend="local",
                max_spend_usd=1.0,
                hf_input_million_usd=1.0,
                hf_output_million_usd=1.0,
            ),
            retriever=retriever,
            inference=LocalExtractor("fixture-model", client=client, source_bound_evidence=True),
            policy=URLPolicy(["citation.test"], resolve_dns=False),
            live=True,
        )
        monkeypatch.setattr(pipeline, "_reserve_inference", lambda *args: inference_reservations.append(args))
        monkeypatch.setattr(pipeline.store, "reserve", lambda *args: spend_reservations.append(args))
        first = await pipeline.extract_url(request)
        second = await pipeline.resume(first.job_id)
        for result in (first, second):
            assert result.status == "failed" and result.error.code == "CONTENT_TOO_LARGE"
            assert result.usage.inference_attempts == 0
            job = pipeline.store.get(result.job_id)
            assert "inference_counts" not in job
            assert "inference_attempts" not in job
            assert result.provenance.provider is None
        assert retriever.calls == 1
        assert not model_calls and not inference_reservations and not spend_reservations
        assert pipeline.store.reserved_spend() == 0.0
        await pipeline.close()


async def test_legacy_product_policy_keeps_preparation_timing_unchanged(tmp_path):
    record = {"final_url": URL, "ld_json": [{"@type": "Product", "name": "Public Folio"}]}
    model_calls = []

    class Retriever:
        async def retrieve(self, request, job, checkpoint, wait_seconds):
            checkpoint({"state": "fetched", "record": record, "retrieval_method": "fixture_replay"})
            return RetrievalOutcome(status="ready", record=record, retrieval_method="fixture_replay")

    class LegacyExtractor(LocalExtractor):
        def prepare_source(self, snapshot):
            raise AssertionError("Legacy product policy must retain its original preparation timing")

    def respond(http_request):
        model_calls.append(http_request)
        payload = {
            "data": ProductV1(name="Public Folio").model_dump(),
            "evidence": [
                {
                    "field_path": "name",
                    "section_id": None,
                    "excerpt": "Public Folio",
                    "json_pointer": "/0/name",
                }
            ],
            "warnings": [],
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}],
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        pipeline = Pipeline(
            Settings(state_dir=tmp_path / "state", retrieval_backend="direct"),
            retriever=Retriever(),
            inference=LegacyExtractor("fixture-model", client=client),
            policy=URLPolicy(["citation.test"], resolve_dns=False),
            live=False,
        )
        result = await pipeline.extract_url(ExtractionRequest(url=URL))
        assert result.status == "partial" and len(model_calls) == 1
        await pipeline.close()
