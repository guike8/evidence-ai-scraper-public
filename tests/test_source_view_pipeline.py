from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
import pytest

from ai_scraper.config import Settings
from ai_scraper.contracts import RetrievalOutcome, ScraperError
from ai_scraper.extraction.generic import deterministic_extract
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.jobs import JobStore
from ai_scraper.normalize import normalize_record
from ai_scraper.pipeline import Pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.registry import get_schema
from ai_scraper.schemas import CustomField, ExtractionRequest

URL = "https://context.test/brief"


class FixtureRetriever:
    def __init__(self, record):
        self.record = record
        self.calls = 0

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        self.calls += 1
        checkpoint({"state": "fetched", "record": self.record, "retrieval_method": "fixture_replay"})
        return RetrievalOutcome(status="ready", record=self.record, retrieval_method="fixture_replay")


def noisy_record():
    navigation = "".join(f"<p>Navigation destination number {index}.</p>" for index in range(2100))
    return {
        "page_html": f"<title>Research brief</title><nav>{navigation}</nav>"
        "<main><h1>Research brief</h1><p>Volunteer count: 37.</p></main>"
    }


async def test_source_view_is_saved_before_rpc_and_recreated_on_cached_resume(tmp_path):
    record = noisy_record()
    request = ExtractionRequest(url=URL, schema_id="page_v1", instruction="Return the main brief.")
    snapshot = normalize_record(record, request)
    expected = deterministic_extract(snapshot, get_schema(request)).model_dump(mode="json")
    state = tmp_path / "state"
    calls = []

    def respond(http_request):
        descriptors = list((state / "source-views").glob("*.json"))
        assert len(descriptors) == 1
        descriptor = json.loads(descriptors[0].read_text())
        assert descriptor["coverage"] == "schema_scope"
        assert descriptor["scope"] == "main"
        assert len(descriptor["out_of_scope_section_ids"]) >= 2100
        payload = json.loads(http_request.content)
        sent = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
        assert {section["id"] for section in sent["sections"]} == set(descriptor["included_section_ids"])
        assert descriptor["view_chars"] < descriptor["original_chars"]
        assert descriptor["source_hash"] == snapshot.source_hash
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(expected)}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50},
            },
        )

    settings = Settings(state_dir=state, retrieval_backend="direct", inference_backend="local")
    retriever = FixtureRetriever(record)
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        pipeline = Pipeline(
            settings,
            retriever=retriever,
            inference=LocalExtractor("fixture-model", client=client),
            policy=URLPolicy(["context.test"], resolve_dns=False),
            live=False,
        )
        first = await pipeline.extract_url(request)
        assert first.status == "success"
        assert first.provenance.inference_source.coverage == "schema_scope"
        descriptor_path = first.provenance.inference_source.descriptor_path
        assert pipeline.store.get(first.job_id)["inference_sources"]
        source_before = json.loads(Path(first.provenance.snapshot_path).read_text())
        second = await pipeline.resume(first.job_id)
        assert second.status == "success" and second.usage.cache_hit
        assert len(calls) == 1 and retriever.calls == 1
        assert second.provenance.inference_source.descriptor_path == descriptor_path
        assert source_before == json.loads(Path(second.provenance.snapshot_path).read_text())
        await pipeline.close()


async def test_oversized_custom_view_is_recorded_without_spending_model_attempt(tmp_path):
    record = noisy_record()
    request = ExtractionRequest(
        url=URL, schema_id="custom_v1", fields=[CustomField(name="volunteers", type="integer")]
    )
    model_calls = []

    def reject(http_request):
        model_calls.append(http_request)
        raise AssertionError("An oversized full-source custom task must not call a model")

    async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
        pipeline = Pipeline(
            Settings(state_dir=tmp_path / "state", retrieval_backend="direct"),
            retriever=FixtureRetriever(record),
            inference=LocalExtractor("fixture-model", client=client),
            policy=URLPolicy(["context.test"], resolve_dns=False),
            live=False,
        )
        result = await pipeline.extract_url(request)
        assert result.status == "failed" and result.error.code == "CONTENT_TOO_LARGE"
        assert not model_calls and result.usage.inference_attempts == 0
        assert result.provenance.provider is None
        assert result.provenance.inference_source.coverage == "full"
        assert result.provenance.inference_source.excluded_sections == 1  # duplicate title only
        assert "inference_counts" not in pipeline.store.get(result.job_id)
        await pipeline.close()


def test_source_view_storage_is_private_immutable_bounded_and_pruned(tmp_path):
    store = JobStore(tmp_path / "state")
    request = ExtractionRequest(url=URL, schema_id="page_v1")
    job = store.get_or_create(request)
    descriptor = {"source_hash": "a" * 64, "included_section_ids": ["s0001"]}
    path = store.save_source_view(job["id"], descriptor)
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    assert store.save_source_view(job["id"], descriptor) == path
    path.write_text("{}")
    with pytest.raises(ScraperError, match="Saved source-view descriptor changed"):
        store.save_source_view(job["id"], descriptor)
    with pytest.raises(ScraperError) as error:
        store.save_source_view("../outside", descriptor)
    assert error.value.error.code == "INVALID_REQUEST"
    bounded = JobStore(store.root, max_snapshot_bytes=10)
    with pytest.raises(ScraperError) as error:
        bounded.save_source_view(job["id"], descriptor)
    assert error.value.error.code == "CONTENT_TOO_LARGE"
    store.update(job["id"], state="complete")
    with store.connect() as db:
        db.execute("UPDATE jobs SET updated=? WHERE id=?", (time.time() - 60 * 86400, job["id"]))
    assert store.prune(30) == 1
    assert not path.exists()


async def test_malformed_optional_adapter_descriptor_fails_before_rpc(tmp_path):
    class BrokenAdapter:
        provider = "local"
        model = "fixture"
        source_view_version = "third-party-v1"
        last_source_view = None

        def prepare_source(self, snapshot):
            self.last_source_view = {"source_hash": snapshot.source_hash}

        async def extract(self, snapshot, feedback=None):
            raise AssertionError("Malformed metadata must fail before an inference call")

    pipeline = Pipeline(
        Settings(state_dir=tmp_path / "state", retrieval_backend="direct"),
        retriever=FixtureRetriever({"page_html": "<main><h1>Note</h1><p>Public text.</p></main>"}),
        inference=BrokenAdapter(),
        policy=URLPolicy(["context.test"], resolve_dns=False),
        live=False,
    )
    result = await pipeline.extract_url(
        ExtractionRequest(url=URL, schema_id="page_v1", instruction="Read the main text.")
    )
    assert result.status == "failed" and result.error.code == "SOURCE_VIEW_INVALID"
    assert result.usage.inference_attempts == 0 and result.data is None
