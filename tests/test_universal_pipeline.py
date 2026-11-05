from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from ai_scraper.config import Settings
from ai_scraper.contracts import InferenceResponse, RetrievalOutcome
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.pipeline import Pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.brightdata import BrightDataRetriever
from ai_scraper.schemas import CustomField, ExtractionRequest, ProductV1, SourceSnapshot

HOST = "universal.test"
URL = f"https://{HOST}/page"


def settings(tmp_path, **extra):
    return Settings(state_dir=tmp_path / "state", retrieval_backend="direct", allowed_domains=["*"], **extra)


def policy():
    return URLPolicy(["*"], resolve_dns=False)


class SavedRecordRetriever:
    retrieval_method = "direct_http"

    def __init__(self, record):
        self.record = record
        self.calls = 0

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        self.calls += 1
        checkpoint(
            {
                "state": "fetched",
                "record": self.record,
                "retrieval_method": self.retrieval_method,
                "attempts": 1,
            }
        )
        return RetrievalOutcome(
            status="ready", record=self.record, retrieval_method=self.retrieval_method, attempts=1
        )


@pytest.mark.parametrize(
    ("schema_id", "record", "field", "expected"),
    [
        (
            "page_v1",
            {
                "page_html": "<html><title>General Reference</title><h1>General Reference</h1><p>A literal public explanation.</p></html>"
            },
            "title",
            "General Reference",
        ),
        (
            "article_v1",
            {
                "page_html": "<article><h1>Research Notes</h1><p>The complete reviewed article body.</p></article>",
                "ld_json": [
                    {
                        "@type": "Article",
                        "headline": "Research Notes",
                        "articleBody": "The complete reviewed article body.",
                    }
                ],
            },
            "title",
            "Research Notes",
        ),
        (
            "organization_v1",
            {
                "page_html": "<h1>Harbor Library</h1><p>A public research organization.</p>",
                "ld_json": [
                    {
                        "@type": "Organization",
                        "name": "Harbor Library",
                        "description": "A public research organization.",
                    }
                ],
            },
            "name",
            "Harbor Library",
        ),
    ],
)
async def test_nonproduct_presets_use_same_durable_pipeline_without_product_fields(
    tmp_path, schema_id, record, field, expected
):
    record = {**record, "input": {"url": URL}, "final_url": URL}
    retriever = SavedRecordRetriever(record)
    pipeline = Pipeline(settings(tmp_path), retriever=retriever, policy=policy())
    result = await pipeline.extract_url(ExtractionRequest(url=URL, schema_id=schema_id))
    assert result.status == "success", result.model_dump(mode="json")
    assert isinstance(result.data, dict)
    assert result.data[field] == expected
    assert not {"price", "currency", "sku"}.intersection(result.data)
    assert result.provenance.schema_version == schema_id
    assert result.provenance.retrieval_method == "direct_http"
    assert result.evidence[field].field_path == field
    assert result.usage.cost_estimate_usd == pipeline.store.reserved_spend() == 0.0
    assert retriever.calls == 1
    source_path = Path(result.provenance.snapshot_path)
    saved_bytes = source_path.read_bytes()
    source = SourceSnapshot.model_validate_json(saved_bytes)
    assert source.raw_record == record
    resumed = await pipeline.resume(result.job_id)
    assert resumed.status == "success" and resumed.usage.cache_hit
    assert source_path.read_bytes() == saved_bytes
    assert retriever.calls == 1
    assert (
        json.loads(Path(pipeline.store.get(result.job_id)["result_path"]).read_text())["data"][field]
        == expected
    )
    await pipeline.close()


class ScopedInference:
    provider = "local"
    model = "Generic-Test-MLX"

    def __init__(self, values, *, bad_evidence=False):
        self.values = values
        self.bad_evidence = bad_evidence
        self.calls = []
        self.scopes = []

    def for_request(self, request, spec):
        owner = self
        owner.scopes.append((request, spec))

        class RequestInference:
            provider = owner.provider
            model = owner.model

            async def extract(self, snapshot, feedback=None):
                owner.calls.append((spec.fingerprint, request.instruction, snapshot.source_hash, feedback))
                choices = owner.values(request) if callable(owner.values) else owner.values
                data = spec.data_model.model_validate(choices).model_dump()
                evidence = []
                for field, value in data.items():
                    if value is None:
                        continue
                    values = enumerate(value) if isinstance(value, list) else [(None, value)]
                    for index, item in values:
                        excerpt = json.dumps(item) if type(item) in {bool, int, float} else item
                        section = next(s for s in snapshot.sections if excerpt in s.text)
                        evidence.append(
                            {
                                "field_path": field if index is None else f"{field}.{index}",
                                "section_id": section.id,
                                "excerpt": "invented quotation" if owner.bad_evidence else excerpt,
                                "json_pointer": None,
                            }
                        )
                return InferenceResponse(
                    payload={"data": data, "evidence": evidence, "warnings": []},
                    usage={"input_tokens": 12, "output_tokens": 10},
                )

        return RequestInference()

    async def extract(self, snapshot, feedback=None):
        raise AssertionError("Pipeline must create an isolated for_request view before inference")


def custom_request(fields, instruction=None):
    return ExtractionRequest(url=URL, schema_id="custom_v1", fields=fields, instruction=instruction)


async def test_custom_types_schema_definitions_and_goals_share_source_but_not_extraction_cache(tmp_path):
    record = {
        "final_url": URL,
        "markdown": "First metric: 12\nSecond metric: 18\nFlag: true\nTopic words: local public",
    }
    retriever = SavedRecordRetriever(record)
    inference = ScopedInference(
        lambda request: (
            {"amount": 18 if request.instruction == "Use the second metric" else 12}
            if request.fields[0].name == "amount"
            else {"flag": True, "topics": ["local", "public"]}
        )
    )
    pipeline = Pipeline(settings(tmp_path), retriever=retriever, inference=inference, policy=policy())
    numeric_fields = [CustomField(name="amount", type="integer", critical=True)]
    first_request = custom_request(numeric_fields, "Use the first metric")
    second_request = custom_request(numeric_fields, "Use the second metric")
    third_request = custom_request(
        [
            CustomField(name="flag", type="boolean", critical=True),
            CustomField(name="topics", type="string_list"),
        ]
    )
    first = await pipeline.extract_url(first_request)
    second = await pipeline.extract_url(second_request)
    third = await pipeline.extract_url(third_request)
    assert first.status == second.status == third.status == "success"
    assert first.data == {"amount": 12}
    assert second.data == {"amount": 18}
    assert third.data == {"flag": True, "topics": ["local", "public"]}
    assert len({first.job_id, second.job_id, third.job_id}) == 3
    assert first.provenance.source_hash == second.provenance.source_hash == third.provenance.source_hash
    assert retriever.calls == 1
    assert len(inference.calls) == 3
    assert len({call[0] for call in inference.calls}) == 2
    assert {call[1] for call in inference.calls} == {None, "Use the first metric", "Use the second metric"}
    repeated = await pipeline.extract_url(first_request)
    assert repeated.status == "success" and repeated.job_id == first.job_id and repeated.usage.cache_hit
    assert repeated.data == {"amount": 12}
    assert len(inference.calls) == 3 and retriever.calls == 1
    assert pipeline.store.reserved_spend() == 0.0
    await pipeline.close()


async def test_custom_missing_critical_fact_returns_null_failure_without_inventing_value(tmp_path):
    inference = ScopedInference({"missing_fact": None})
    retriever = SavedRecordRetriever(
        {"final_url": URL, "markdown": "A page that does not contain the requested detail."}
    )
    pipeline = Pipeline(settings(tmp_path), retriever=retriever, inference=inference, policy=policy())
    result = await pipeline.extract_url(
        custom_request([CustomField(name="missing_fact", type="string", critical=True)])
    )
    assert result.status == "failed" and result.data is None
    assert "missing_fact" in result.missing_fields
    assert result.error.code == "INVALID_EXTRACTION"
    assert len(inference.calls) == 2 and retriever.calls == 1
    await pipeline.close()


async def test_same_custom_field_name_and_goal_with_different_types_cannot_reuse_cached_candidate(tmp_path):
    record = {"final_url": URL, "markdown": "The displayed count is 12."}
    retriever = SavedRecordRetriever(record)
    inference = ScopedInference(
        lambda request: {"amount": 12 if request.fields[0].type == "integer" else "12"}
    )
    pipeline = Pipeline(settings(tmp_path), retriever=retriever, inference=inference, policy=policy())
    number_request = custom_request([CustomField(name="amount", type="integer", critical=True)])
    text_request = custom_request([CustomField(name="amount", type="string", critical=True)])
    numeric = await pipeline.extract_url(number_request)
    textual = await pipeline.extract_url(text_request)
    assert numeric.status == textual.status == "success"
    assert numeric.data == {"amount": 12}
    assert textual.data == {"amount": "12"}
    assert numeric.provenance.source_hash == textual.provenance.source_hash
    assert numeric.job_id != textual.job_id
    assert retriever.calls == 1 and len(inference.calls) == 2
    assert inference.calls[0][0] != inference.calls[1][0]
    assert inference.calls[0][1] is inference.calls[1][1] is None
    await pipeline.close()


async def test_custom_unsupported_quote_stays_failed_after_one_repair_and_restart(tmp_path):
    record = {"final_url": URL, "markdown": "A source says the metric is 12."}
    inference = ScopedInference({"amount": 12}, bad_evidence=True)
    retriever = SavedRecordRetriever(record)
    pipeline = Pipeline(settings(tmp_path), retriever=retriever, inference=inference, policy=policy())
    request = custom_request([CustomField(name="amount", type="integer", critical=True)])
    result = await pipeline.extract_url(request)
    assert result.status == "failed" and result.data is None
    assert len(inference.calls) == result.usage.inference_attempts == 2
    assert len({call[2] for call in inference.calls}) == 1
    assert "evidence:amount" in inference.calls[1][3]
    await pipeline.close()
    restarted = Pipeline(settings(tmp_path), retriever=retriever, inference=inference, policy=policy())
    resumed = await restarted.resume(result.job_id)
    assert resumed.status == "failed" and resumed.data is None
    assert len(inference.calls) == 2 and retriever.calls == 1
    await restarted.close()


async def test_actual_local_adapter_is_scoped_to_custom_contract_not_product_schema(tmp_path):
    calls = []

    def complete(http_request):
        body = json.loads(http_request.content)
        calls.append(body)
        schema = body["response_format"]["json_schema"]["schema"]
        reference = schema["properties"]["data"]["$ref"].split("/")[-1]
        assert set(schema["$defs"][reference]["properties"]) == {"founder"}
        assert "ProductV1" not in schema["$defs"]
        assert body["tool_choice"] == "none"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": json.dumps(
                                {
                                    "data": {"founder": "Ada Example"},
                                    "evidence": [
                                        {
                                            "field_path": "founder",
                                            "section_id": "s0001",
                                            "excerpt": "Ada Example",
                                            "json_pointer": None,
                                        }
                                    ],
                                    "warnings": [],
                                }
                            )
                        },
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 20},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(complete)) as client:
        inference = LocalExtractor("Local-General-MLX", client=client)
        record = {"final_url": URL, "markdown": "Founded by Ada Example."}
        pipeline = Pipeline(
            settings(tmp_path), retriever=SavedRecordRetriever(record), inference=inference, policy=policy()
        )
        request = custom_request(
            [CustomField(name="founder", type="string", critical=True)], "Who founded this project?"
        )
        result = await pipeline.extract_url(request)
        assert result.status == "success", result.model_dump(mode="json")
        assert result.data == {"founder": "Ada Example"}
        assert result.provenance.provider == "local" and result.provenance.model == "Local-General-MLX"
        assert result.usage.input_tokens == 50 and result.usage.cost_estimate_usd == 0.0
        assert "Who founded this project?" in calls[0]["messages"][0]["content"]
        assert "ProductV1" in inference.schema["$defs"]  # parent remains unchanged
        await pipeline.close()


async def test_waiting_different_schema_survives_restart_and_never_duplicates_vendor_post(tmp_path):
    clock = [0.0]
    ready = [False]
    methods = []
    record = {
        "input": {"url": URL},
        "final_url": URL,
        "page_html": "<article><h1>Shared Article</h1><p>A single saved source for multiple extraction tasks.</p></article>",
        "ld_json": [
            {
                "@type": "Article",
                "headline": "Shared Article",
                "articleBody": "A single saved source for multiple extraction tasks.",
            }
        ],
    }

    def vendor(http_request):
        methods.append(http_request.method)
        clock[0] += 1.0
        if http_request.method == "POST":
            return httpx.Response(202, json={"snapshot_id": "durable-shared-snapshot"})
        if "/progress/" in http_request.url.path:
            return httpx.Response(200, json={"status": "ready" if ready[0] else "running"})
        return httpx.Response(200, json=[record])

    config = Settings(state_dir=tmp_path / "state", allowed_domains=["*"], wait_seconds=0.25)
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor)) as client:
        adapter = BrightDataRetriever("fixture-key", client=client, clock=lambda: clock[0])
        pipeline = Pipeline(config, retriever=adapter, policy=policy(), live=False)
        parent = await pipeline.extract_url(ExtractionRequest(url=URL, schema_id="page_v1"))
        child = await pipeline.extract_url(ExtractionRequest(url=URL, schema_id="article_v1"))
        assert parent.status == child.status == "pending"
        assert child.job_id != parent.job_id
        child_state = pipeline.store.get(child.job_id)
        assert child_state["retrieval_parent_id"] == parent.job_id
        assert child_state["request"]["schema_id"] == "article_v1"
        assert methods.count("POST") == 1
        await pipeline.close()
    ready[0] = True
    async with httpx.AsyncClient(transport=httpx.MockTransport(vendor)) as client:
        adapter = BrightDataRetriever("fixture-key", client=client, clock=lambda: clock[0])
        restarted = Pipeline(config, retriever=adapter, policy=policy(), live=False)
        # Permit progress and snapshot download to fit into this one resumed
        # operation, while keeping the original submission already persisted.
        adapter.clock = lambda: 0.0
        completed = await restarted.resume(child.job_id)
        assert completed.status == "success", completed.model_dump(mode="json")
        assert completed.provenance.schema_version == "article_v1"
        assert completed.provenance.snapshot_id == "durable-shared-snapshot"
        assert completed.data["title"] == "Shared Article"
        assert methods.count("POST") == 1
        assert restarted.store.get(child.job_id)["record"] == record
        await restarted.close()


async def test_old_product_job_without_new_request_fields_resumes_from_immutable_source(tmp_path):
    record = {
        "final_url": URL,
        "ld_json": [
            {
                "@type": "Product",
                "name": "Legacy Hoodie",
                "offers": {"price": "12.00", "priceCurrency": "USD"},
            }
        ],
    }
    retriever = SavedRecordRetriever(record)
    pipeline = Pipeline(settings(tmp_path), retriever=retriever, policy=policy())
    first = await pipeline.extract_url(ExtractionRequest(url=URL, schema_id="product_v1"))
    assert first.status == "success" and isinstance(first.data, ProductV1)
    source_path = Path(first.provenance.snapshot_path)
    original_bytes = source_path.read_bytes()
    job = pipeline.store.get(first.job_id)
    legacy_request = {
        key: value for key, value in job["request"].items() if key not in {"fields", "instruction"}
    }
    pipeline.store.update(first.job_id, request=legacy_request)
    await pipeline.close()
    resumed_pipeline = Pipeline(settings(tmp_path), retriever=retriever, policy=policy())
    resumed = await resumed_pipeline.resume(first.job_id)
    assert resumed.status == "success" and isinstance(resumed.data, ProductV1)
    assert resumed.data.name == "Legacy Hoodie" and resumed.data.price == "12.00"
    assert resumed.provenance.schema_version == "product_v1"
    assert resumed.usage.cache_hit and retriever.calls == 1
    assert source_path.read_bytes() == original_bytes
    await resumed_pipeline.close()
