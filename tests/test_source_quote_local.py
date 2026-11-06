"""Opt-in literal citation grammar through the real local HTTP adapter."""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from ai_scraper.cli import _settings, parser
from ai_scraper.config import Settings
from ai_scraper.contracts import RetrievalOutcome, ScraperError
from ai_scraper.extraction import local
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.extraction.source_quote_schema import (
    SOURCE_QUOTE_SCHEMA_VERSION,
    SourceQuoteSchemaError,
    bind_source_quote_evidence,
)
from ai_scraper.pipeline import Pipeline, create_pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest, ProductV1, SourceSection, SourceSnapshot

URL = "https://citation.test/record"
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


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(
        os,
        "environ",
        {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("LOCAL_", "HF_", "SCRAPER_", "BRIGHTDATA_", "FIRECRAWL_"))
        },
    )


def source(section="s0007", text="Public Folio"):
    return SourceSnapshot(
        requested_url=URL,
        final_url=URL,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        raw_record={},
        sections=[SourceSection(id=section, text=text)],
        ld_json=[{"name": text}],
        source_hash="a" * 64,
    )


def request_for(schema_id):
    return ExtractionRequest(
        url=URL,
        schema_id=schema_id,
        fields=[{"name": "active", "type": "boolean"}] if schema_id == "custom_v1" else [],
    )


def response(payload):
    return httpx.Response(
        200, json={"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}]}
    )


@pytest.mark.parametrize("schema_id", SCHEMAS)
async def test_real_local_payload_matches_the_experimental_grammar_and_exact_source(schema_id):
    calls = []
    returned = {"data": {}, "evidence": [], "warnings": []}

    def respond(request):
        calls.append(json.loads(request.content))
        return response(returned)

    request = request_for(schema_id)
    spec = get_schema(request)
    snapshot = source()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        base = LocalExtractor("local-model", client=client, source_quote_evidence=True)
        scoped = base.for_request(request, spec)
        template = copy.deepcopy(scoped.schema)
        legacy = LocalExtractor("local-model", client=client).for_request(request, spec)
        exact = legacy._source_content(snapshot)
        bound = bind_source_quote_evidence(template, json.loads(exact))
        assert (await scoped.extract(snapshot)).payload == returned
        assert calls[0]["messages"][1]["content"] == "UNTRUSTED_SOURCE_JSON\n" + exact
        actual = calls[0]["response_format"]["json_schema"]["schema"]
        assert actual == bound.schema == scoped.schema
        assert calls[0]["structured_outputs"] == {"json_schema": actual}
        assert scoped.last_evidence_catalog == bound.metadata()
        assert scoped.citation_policy_version == SOURCE_QUOTE_SCHEMA_VERSION
        assert scoped._evidence_schema_template == template
        assert base.last_evidence_catalog is None


def test_quote_mode_enforces_pairing_and_business_paths_without_changing_business_schema():
    adapter = LocalExtractor("local-model", client=object(), source_quote_evidence=True)
    before = copy.deepcopy(adapter.schema)
    adapter.prepare_source(source())
    assert adapter.schema["$defs"]["ProductV1"] == before["$defs"]["ProductV1"]
    payload = {
        "data": ProductV1(name="Public Folio").model_dump(),
        "warnings": [],
        "evidence": [
            {"field_path": "name", "excerpt": "Public Folio", "section_id": "s0007", "json_pointer": None}
        ],
    }
    validator = Draft202012Validator(adapter.schema)
    assert validator.is_valid(payload)
    for patch in (
        {"field_path": "data.name"},
        {"section_id": "invented"},
        {"excerpt": "Invented quotation"},
        {"json_pointer": "/0/name"},
    ):
        changed = copy.deepcopy(payload)
        changed["evidence"][0].update(patch)
        assert not validator.is_valid(changed)


async def test_each_source_rebuilds_quote_grammar_and_cloned_requests_are_independent():
    calls = []

    async def respond(request):
        calls.append(json.loads(request.content))
        await asyncio.sleep(0)
        return response({})

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        base = LocalExtractor("local-model", client=client, source_quote_evidence=True)
        template = copy.deepcopy(base.schema)
        first = base.for_request(request_for("article_v1"), get_schema(request_for("article_v1")))
        second = base.for_request(request_for("person_v1"), get_schema(request_for("person_v1")))
        await asyncio.gather(
            first.extract(source("first", "First text")), second.extract(source("second", "Second text"))
        )
        assert base.schema == template and base.last_evidence_catalog is None
        await first.extract(source("third", "Third text"))
    for call, expected in zip(calls, ("First text", "Second text", "Third text"), strict=True):
        branches = call["response_format"]["json_schema"]["schema"]["$defs"]["EvidenceCandidate"]["anyOf"]
        assert {branch["properties"]["excerpt"]["const"] for branch in branches} == {expected}


def test_failed_catalog_preparation_restores_template_and_redacts_source(monkeypatch):
    adapter = LocalExtractor("local-model", client=object(), source_quote_evidence=True)
    template = copy.deepcopy(adapter.schema)
    adapter.prepare_source(source())

    def fail(*args):
        raise SourceQuoteSchemaError("private source text /private/example")

    monkeypatch.setattr(local, "bind_source_quote_evidence", fail)
    with pytest.raises(ScraperError) as error:
        adapter.prepare_source(source("other"))
    assert error.value.error.code == "CITATION_CATALOG_UNAVAILABLE"
    assert "private" not in error.value.error.message and "source text" not in error.value.error.message
    assert adapter.schema == template and adapter.last_evidence_catalog is None


@pytest.mark.parametrize("bad", [1, "true", None])
def test_quote_constructor_requires_a_strict_boolean(bad):
    with pytest.raises(ValueError, match="must be a boolean"):
        LocalExtractor("local-model", client=object(), source_quote_evidence=bad)


def test_modes_are_exclusive_and_defaults_remain_unbound():
    with pytest.raises(ValueError, match="mutually exclusive"):
        LocalExtractor("local-model", client=object(), source_bound_evidence=True, source_quote_evidence=True)
    with pytest.raises(ValidationError, match="mutually exclusive"):
        Settings(local_source_bound_evidence=True, local_source_quote_evidence=True)
    with pytest.raises(ValidationError):
        Settings(local_source_quote_evidence=1)
    original = LocalExtractor("local-model", client=object())
    assert original.citation_policy_version is None and not original.source_quote_evidence
    assert Settings().local_source_quote_evidence is False


def test_cli_positive_flags_override_other_environment_mode_and_negative_flags_work(monkeypatch):
    monkeypatch.setenv("LOCAL_SOURCE_BOUND_EVIDENCE", "true")
    args = parser().parse_args(["extract", URL, "--no-source-bound-evidence", "--source-quote-evidence"])
    settings = _settings(args)
    assert settings.local_source_quote_evidence and not settings.local_source_bound_evidence
    monkeypatch.delenv("LOCAL_SOURCE_BOUND_EVIDENCE")
    monkeypatch.setenv("LOCAL_SOURCE_QUOTE_EVIDENCE", "true")
    assert Settings.from_env().local_source_quote_evidence
    settings = _settings(parser().parse_args(["extract", URL, "--source-bound-evidence"]))
    assert settings.local_source_bound_evidence and not settings.local_source_quote_evidence
    settings = _settings(parser().parse_args(["extract", URL, "--no-source-quote-evidence"]))
    assert not settings.local_source_quote_evidence
    with pytest.raises(ScraperError, match="mutually exclusive"):
        _settings(parser().parse_args(["extract", URL, "--source-bound-evidence", "--source-quote-evidence"]))


async def test_factory_and_deployment_allow_explicit_quote_setting(tmp_path, monkeypatch):
    settings = Settings(
        state_dir=tmp_path / "state",
        inference_backend="local",
        local_model="local-model",
        local_source_quote_evidence=True,
    )
    pipeline = create_pipeline(settings)
    assert pipeline.inference.source_quote_evidence
    assert pipeline.inference.citation_policy_version == SOURCE_QUOTE_SCHEMA_VERSION
    await pipeline.close()
    spec = importlib.util.spec_from_file_location(
        "quote_deployment", Path(__file__).parents[1] / "deploy/launch.py"
    )
    deployment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deployment)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"LOCAL_SOURCE_QUOTE_EVIDENCE": "true"}))
    config.chmod(0o600)
    environment = deployment.load_environment(tmp_path, config, {})
    monkeypatch.setattr(os, "environ", environment)
    assert Settings.from_env().local_source_quote_evidence


async def test_quote_policy_has_a_separate_inference_cache_and_reuses_retrieval(tmp_path):
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
        evidence = []
        for path, value in data.items():
            section = next(item for item in sent["sections"] if value in item["text"])
            evidence.append(
                {
                    "field_path": path,
                    "section_id": section["id"],
                    "excerpt": section["text"],
                    "json_pointer": None,
                }
            )
        return response({"data": ProductV1(**data).model_dump(), "evidence": evidence, "warnings": []})

    retriever = Retriever()
    settings = Settings(state_dir=tmp_path / "state", retrieval_backend="direct")
    request = ExtractionRequest(
        url=URL, schema_id="product_v1", instruction="Extract the exact product facts."
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        job_id = None
        for flags in ({}, {"source_bound_evidence": True}, {"source_quote_evidence": True}):
            pipeline = Pipeline(
                settings,
                retriever=retriever,
                policy=URLPolicy(["citation.test"], resolve_dns=False),
                live=False,
                inference=LocalExtractor("same-model", client=client, **flags),
            )
            result = await pipeline.extract_url(request) if job_id is None else await pipeline.resume(job_id)
            assert result.status == "success"
            job_id = result.job_id
            await pipeline.close()
        assert len(calls) == 3 and retriever.count == 1
        assert len(pipeline.store.get(job_id)["inference_counts"]) == 3
        repeated = await pipeline.resume(job_id)
        assert repeated.status == "success" and repeated.usage.cache_hit and len(calls) == 3


async def test_quote_catalog_failure_occurs_before_inference_budget_reservation(tmp_path, monkeypatch):
    record = {"final_url": URL, "ld_json": [{"@type": "Product", "name": "Public Folio"}]}
    calls, reservations = [], []

    class Retriever:
        count = 0

        async def retrieve(self, request, job, checkpoint, wait_seconds):
            self.count += 1
            checkpoint({"state": "fetched", "record": record, "retrieval_method": "direct_http"})
            return RetrievalOutcome(status="ready", record=record, retrieval_method="direct_http")

    def reject(request):
        calls.append(True)
        raise AssertionError("No model call allowed")

    def catalog_error(*args):
        raise SourceQuoteSchemaError("catalog cannot fit")

    monkeypatch.setattr(local, "bind_source_quote_evidence", catalog_error)
    retriever = Retriever()
    async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
        pipeline = Pipeline(
            Settings(state_dir=tmp_path / "state", retrieval_backend="direct"),
            retriever=retriever,
            inference=LocalExtractor("local-model", client=client, source_quote_evidence=True),
            policy=URLPolicy(["citation.test"], resolve_dns=False),
            live=False,
        )
        monkeypatch.setattr(pipeline, "_reserve_inference", lambda *args: reservations.append(args))
        result = await pipeline.extract_url(ExtractionRequest(url=URL, schema_id="product_v1"))
        assert result.status == "failed" and result.error.code == "CITATION_CATALOG_UNAVAILABLE"
        assert result.usage.inference_attempts == 0 and not calls and not reservations
        await pipeline.close()
