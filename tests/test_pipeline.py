from __future__ import annotations

import json
import socket
from pathlib import Path

import httpx
import pytest

from ai_scraper.config import Settings
from ai_scraper.contracts import InferenceResponse, RetrievalOutcome, ScraperError
from ai_scraper.pipeline import Pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.brightdata import BrightDataRetriever
from ai_scraper.schemas import ExtractionRequest, ProductV1, SourceSnapshot
from ai_scraper.validate import validate_candidate

URL = "https://shop.test/product/semantic-hoodie"
REQUEST = ExtractionRequest(url=URL)
SEMANTIC_RECORD = {
    "input": {"url": URL},
    "final_url": URL,
    "markdown": "# Semantic Hoodie\nUSD 42.00\nA warm, comfortable hoodie with a full zipper and pockets for everyday outdoor use.",
}
LD_RECORD = {
    "input": {"url": URL},
    "final_url": URL,
    "ld_json": [
        {
            "@type": "Product",
            "url": URL,
            "name": "Semantic Hoodie",
            "offers": {"@type": "Offer", "price": "42.00", "priceCurrency": "USD"},
        }
    ],
}


def settings(tmp_path, **kwargs):
    return Settings(state_dir=tmp_path / "state", allowed_domains=["shop.test"], **kwargs)


def policy():
    return URLPolicy(["shop.test"], resolve_dns=False)


class FixtureRetriever:
    def __init__(self, record=None):
        self.record = record or SEMANTIC_RECORD
        self.calls = 0

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        self.calls += 1
        checkpoint({"state": "fetched", "record": self.record, "attempts": 1})
        return RetrievalOutcome(status="ready", record=self.record, attempts=1)


class FixtureInference:
    provider = "fixture-provider"
    model = "fixture-model-a"

    def __init__(self, bad_calls=0, model=None):
        self.calls = []
        self.bad_calls = bad_calls
        if model:
            self.model = model

    async def extract(self, snapshot, feedback=None):
        self.calls.append((snapshot.source_hash, feedback))
        fields = {"name": "Semantic Hoodie", "price": "42.00", "currency": "USD"}
        evidence = []
        for field_path, value in fields.items():
            section = next(s for s in snapshot.sections if value in s.text)
            evidence.append(
                {
                    "field_path": field_path,
                    "section_id": section.id,
                    "json_pointer": None,
                    "excerpt": section.text,
                }
            )
        if len(self.calls) <= self.bad_calls:
            evidence[0]["excerpt"] = "A hallucinated source quote"
        return InferenceResponse(
            payload={"data": ProductV1(**fields).model_dump(), "evidence": evidence, "warnings": []},
            usage={"input_tokens": 80, "output_tokens": 60},
        )


async def test_end_to_end_source_to_validated_json_and_application_provenance(tmp_path):
    fetcher = FixtureRetriever()
    inference = FixtureInference()
    pipeline = Pipeline(
        settings(tmp_path), retriever=fetcher, inference=inference, policy=policy(), live=False
    )
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "success"
    assert result.data.name == "Semantic Hoodie"
    assert result.data.price == "42.00"
    assert result.provenance.requested_url == result.provenance.final_url == URL
    assert result.provenance.provider == inference.provider
    assert result.provenance.model == inference.model
    saved = SourceSnapshot.model_validate_json(Path(result.provenance.snapshot_path).read_text())
    assert saved.source_hash == result.provenance.source_hash
    assert saved.raw_record == SEMANTIC_RECORD
    assert {"name", "price", "currency"} <= result.evidence.keys()
    exported = Path(pipeline.store.get(result.job_id)["result_path"])
    assert json.loads(exported.read_text())["status"] == "success"
    assert result.usage.inference_attempts == len(inference.calls) == 1
    assert result.usage.input_tokens == 80
    assert result.usage.output_tokens == 60
    assessment = validate_candidate(
        {
            "data": result.data.model_dump(),
            "evidence": [item.model_dump() for item in result.evidence.values()],
        },
        saved,
    )
    assert assessment.status == "success"
    await pipeline.close()


async def test_json_ld_fast_path_never_calls_inference(tmp_path):
    inference = FixtureInference()
    pipeline = Pipeline(
        settings(tmp_path),
        retriever=FixtureRetriever(LD_RECORD),
        inference=inference,
        policy=policy(),
        live=False,
    )
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "success"
    assert not inference.calls
    assert result.usage.inference_attempts == 0
    assert all(item.json_pointer for item in result.evidence.values())


async def test_one_repair_reuses_saved_snapshot_and_never_refetches(tmp_path):
    fetcher = FixtureRetriever()
    inference = FixtureInference(bad_calls=1)
    pipeline = Pipeline(
        settings(tmp_path), retriever=fetcher, inference=inference, policy=policy(), live=False
    )
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "success"
    assert fetcher.calls == 1
    assert len(inference.calls) == result.usage.inference_attempts == 2
    assert len({item[0] for item in inference.calls}) == 1
    assert inference.calls[0][1] is None
    assert "evidence:name" in inference.calls[1][1]


async def test_unsupported_evidence_fails_after_exactly_one_repair_and_resume_stays_bounded(tmp_path):
    fetcher = FixtureRetriever()
    inference = FixtureInference(bad_calls=99)
    pipeline = Pipeline(
        settings(tmp_path), retriever=fetcher, inference=inference, policy=policy(), live=False
    )
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "failed"
    assert result.data is None
    assert result.error.code == "INVALID_EXTRACTION"
    assert fetcher.calls == 1
    assert len(inference.calls) == 2
    resumed = await pipeline.resume(result.job_id)
    assert resumed.status == "failed"
    assert resumed.data is None
    assert fetcher.calls == 1
    assert len(inference.calls) == 2


async def test_same_snapshot_and_model_cache_avoids_both_paid_steps(tmp_path):
    fetcher = FixtureRetriever()
    inference = FixtureInference()
    pipeline = Pipeline(
        settings(tmp_path), retriever=fetcher, inference=inference, policy=policy(), live=False
    )
    first = await pipeline.extract_url(REQUEST)
    second = await pipeline.extract_url(REQUEST)
    assert first.status == second.status == "success"
    assert first.job_id == second.job_id
    assert second.usage.cache_hit
    assert fetcher.calls == len(inference.calls) == 1


async def test_changed_model_reuses_retrieval_but_runs_new_extraction(tmp_path):
    fetcher = FixtureRetriever()
    first_model = FixtureInference()
    first_pipeline = Pipeline(
        settings(tmp_path), retriever=fetcher, inference=first_model, policy=policy(), live=False
    )
    first = await first_pipeline.extract_url(REQUEST)
    second_model = FixtureInference(model="fixture-model-b")
    second_pipeline = Pipeline(
        settings(tmp_path), retriever=fetcher, inference=second_model, policy=policy(), live=False
    )
    second = await second_pipeline.extract_url(REQUEST)
    assert first.status == second.status == "success"
    assert first.job_id == second.job_id
    assert second.provenance.model == "fixture-model-b"
    assert fetcher.calls == 1
    assert len(first_model.calls) == len(second_model.calls) == 1


async def test_real_adapter_pending_survives_pipeline_restart_without_post(tmp_path):
    clock = [0.0]
    post_calls = []

    def submit(request):
        post_calls.append(request)
        clock[0] = 2.0
        return httpx.Response(202, json={"snapshot_id": "vendor-existing"})

    first_fetcher = BrightDataRetriever(
        "fixture-key", client=httpx.AsyncClient(transport=httpx.MockTransport(submit)), clock=lambda: clock[0]
    )
    config = settings(tmp_path, wait_seconds=1.0)
    first_pipeline = Pipeline(config, retriever=first_fetcher, policy=policy(), live=False)
    pending = await first_pipeline.extract_url(REQUEST)
    assert pending.status == "pending"
    assert pending.provenance.snapshot_id == "vendor-existing"
    await first_pipeline.close()
    get_calls = []

    def complete(request):
        get_calls.append(request)
        assert request.method == "GET"
        if "/progress/" in request.url.path:
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(200, json=[LD_RECORD])

    second_fetcher = BrightDataRetriever(
        "fixture-key", client=httpx.AsyncClient(transport=httpx.MockTransport(complete))
    )
    second_pipeline = Pipeline(config, retriever=second_fetcher, policy=policy(), live=False)
    completed = await second_pipeline.resume(pending.job_id)
    assert completed.status == "success"
    assert completed.provenance.snapshot_id == "vendor-existing"
    assert len(post_calls) == 1
    assert len(get_calls) == 2


async def test_ambiguous_timeout_requires_explicit_snapshot_reconciliation(tmp_path):
    post_calls = []

    def timeout(request):
        post_calls.append(request)
        raise httpx.ReadTimeout("submission outcome unknown", request=request)

    fetcher = BrightDataRetriever(
        "fixture-key", client=httpx.AsyncClient(transport=httpx.MockTransport(timeout))
    )
    pipeline = Pipeline(settings(tmp_path), retriever=fetcher, policy=policy(), live=False)
    pending = await pipeline.extract_url(REQUEST)
    assert pending.status == "pending"
    assert pending.error.code == "RECONCILIATION_REQUIRED"
    assert (await pipeline.resume(pending.job_id)).status == "pending"
    assert len(post_calls) == 1

    def complete(request):
        assert request.method == "GET"
        if "/progress/" in request.url.path:
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(200, json=[LD_RECORD])

    resumed = Pipeline(
        settings(tmp_path),
        policy=policy(),
        live=False,
        retriever=BrightDataRetriever(
            "fixture-key", client=httpx.AsyncClient(transport=httpx.MockTransport(complete))
        ),
    )
    result = await resumed.resume(pending.job_id, snapshot_id="reconciled-vendor-job")
    assert result.status == "success"
    assert result.provenance.snapshot_id == "reconciled-vendor-job"
    assert len(post_calls) == 1


async def test_fetched_record_recovers_without_another_vendor_call(tmp_path):
    def unexpected(request):
        pytest.fail("Durable fetched record should be reused")

    fetcher = BrightDataRetriever(
        "fixture-key", client=httpx.AsyncClient(transport=httpx.MockTransport(unexpected))
    )
    pipeline = Pipeline(settings(tmp_path), retriever=fetcher, policy=policy(), live=False)
    job = pipeline.store.get_or_create(REQUEST, {"method": "brightdata_fetch"})
    pipeline.store.update(job["id"], state="fetched", record=LD_RECORD, attempts=1)
    restarted = Pipeline(settings(tmp_path), retriever=fetcher, policy=policy(), live=False)
    result = await restarted.resume(job["id"])
    assert result.status == "success"
    assert Path(result.provenance.snapshot_path).exists()


@pytest.mark.parametrize("tamper", ["raw_record", "sections", "ld_json", "final_url"])
async def test_snapshot_tampering_is_rejected_without_new_work(tmp_path, tamper):
    fetcher = FixtureRetriever(LD_RECORD)
    inference = FixtureInference()
    pipeline = Pipeline(
        settings(tmp_path), retriever=fetcher, inference=inference, policy=policy(), live=False
    )
    first = await pipeline.extract_url(REQUEST)
    path = Path(first.provenance.snapshot_path)
    payload = json.loads(path.read_text())
    if tamper == "raw_record":
        payload["raw_record"]["ld_json"][0]["offers"]["price"] = "999.00"
    elif tamper == "sections":
        payload["sections"] = [{"id": "s0001", "text": "Forged text", "representation": "markdown"}]
    elif tamper == "ld_json":
        payload["ld_json"][0]["offers"]["priceCurrency"] = "EUR"
    else:
        payload["final_url"] = "https://shop.test/product/different-product"
    path.write_text(json.dumps(payload))
    result = await pipeline.resume(first.job_id)
    assert result.status == "failed"
    assert result.error.code == "SNAPSHOT_CORRUPT"
    assert fetcher.calls == 1
    assert not inference.calls


@pytest.mark.parametrize(
    "config,code",
    [
        ({}, "PRICING_UNCONFIGURED"),
        ({"retrieval_record_usd": 0.01}, "BUDGET_UNCONFIGURED"),
        ({"retrieval_record_usd": 0.01, "max_spend_usd": 0.0}, "BUDGET_EXHAUSTED"),
    ],
)
async def test_missing_or_exhausted_live_budget_stops_before_vendor_call(tmp_path, config, code):
    fetcher = FixtureRetriever(LD_RECORD)
    pipeline = Pipeline(settings(tmp_path, **config), retriever=fetcher, policy=policy(), live=True)
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "failed"
    assert result.error.code == code
    assert fetcher.calls == 0
    assert pipeline.store.reserved_spend() == 0


async def test_budget_configuration_can_be_fixed_and_job_resumed(tmp_path):
    fetcher = FixtureRetriever(LD_RECORD)
    pipeline = Pipeline(
        settings(tmp_path, retrieval_record_usd=0.01), retriever=fetcher, policy=policy(), live=True
    )
    failed = await pipeline.extract_url(REQUEST)
    assert failed.error.code == "BUDGET_UNCONFIGURED"
    configured = Pipeline(
        settings(tmp_path, retrieval_record_usd=0.01, max_spend_usd=0.02),
        retriever=fetcher,
        policy=policy(),
        live=True,
    )
    result = await configured.resume(failed.job_id)
    assert result.status == "success"
    assert fetcher.calls == 1
    assert configured.store.reserved_spend() == pytest.approx(0.01)


async def test_unsupported_schema_and_redirect_outside_allowlist_stop_extraction(tmp_path):
    inference = FixtureInference()
    fetcher = FixtureRetriever()
    pipeline = Pipeline(
        settings(tmp_path), retriever=fetcher, inference=inference, policy=policy(), live=False
    )
    unsupported = await pipeline.extract_url(ExtractionRequest(url=URL, schema_id="untrusted_schema"))
    assert unsupported.error.code == "UNSUPPORTED_SCHEMA"
    assert fetcher.calls == 0
    fetcher.record = {**SEMANTIC_RECORD, "final_url": "http://127.0.0.1/private"}
    redirected = await pipeline.extract_url(REQUEST)
    assert redirected.status == "failed"
    assert redirected.error.code == "EGRESS_DENIED"
    assert not inference.calls


async def test_active_host_claim_returns_pending_without_new_work(tmp_path):
    fetcher = FixtureRetriever(LD_RECORD)
    pipeline = Pipeline(settings(tmp_path), retriever=fetcher, policy=policy(), live=False)
    occupied = pipeline.store.get_or_create(ExtractionRequest(url="https://shop.test/product/occupied"))
    assert pipeline.store.acquire(occupied["id"], "other-process")
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "pending"
    assert "concurrency" in result.warnings[0]
    assert fetcher.calls == 0


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/",
        "http://127.0.0.1/",
        "http://10.1.2.3/",
        "http://[::1]/",
        "https://shop.test:8443/",
        "https://unknown.test/",
        "https://shop.test.local/",
    ],
)
async def test_url_policy_rejects_nonpublic_or_unlisted_targets(url):
    configured = URLPolicy(
        ["shop.test", "localhost", "127.0.0.1", "10.1.2.3", "::1", "shop.test.local"], resolve_dns=False
    )
    with pytest.raises(ScraperError) as caught:
        await configured.check(url)
    assert caught.value.error.code == "EGRESS_DENIED"


async def test_dns_resolution_rejects_mixed_public_and_private_answers(monkeypatch):
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **kw: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("8.8.8.8", 443)),
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        ],
    )
    with pytest.raises(ScraperError) as caught:
        await URLPolicy(["shop.test"]).check(URL)
    assert caught.value.error.code == "EGRESS_DENIED"


async def test_reported_intermediate_redirect_is_also_checked():
    with pytest.raises(ScraperError) as caught:
        await policy().check_record(
            {"final_url": URL, "redirect_chain": [{"url": "http://127.0.0.1/private"}, URL]}, URL
        )
    assert caught.value.error.code == "EGRESS_DENIED"


async def test_hf_budget_can_be_configured_after_fetch_without_refetch(tmp_path):
    fetcher = FixtureRetriever()
    inference = FixtureInference()
    initial = Pipeline(
        settings(tmp_path, retrieval_record_usd=0.01, max_spend_usd=1.0),
        retriever=fetcher,
        inference=inference,
        policy=policy(),
        live=True,
    )
    failed = await initial.extract_url(REQUEST)
    assert failed.status == "failed"
    assert failed.error.code == "PRICING_UNCONFIGURED"
    assert fetcher.calls == 1
    assert not inference.calls
    configured = Pipeline(
        settings(
            tmp_path,
            retrieval_record_usd=0.01,
            max_spend_usd=1.0,
            hf_input_million_usd=1.0,
            hf_output_million_usd=1.0,
        ),
        retriever=fetcher,
        inference=inference,
        policy=policy(),
        live=True,
    )
    result = await configured.resume(failed.job_id)
    assert result.status == "success"
    assert fetcher.calls == len(inference.calls) == 1
    assert result.usage.input_tokens == 80
    assert result.usage.output_tokens == 60


async def test_changed_model_gets_a_separate_paid_reservation(tmp_path):
    fetcher = FixtureRetriever()
    first_model = FixtureInference()
    config = settings(
        tmp_path,
        retrieval_record_usd=0.01,
        max_spend_usd=0.30,
        hf_input_million_usd=1.0,
        hf_output_million_usd=1.0,
    )
    first_pipeline = Pipeline(config, retriever=fetcher, inference=first_model, policy=policy(), live=True)
    first = await first_pipeline.extract_url(REQUEST)
    assert first.status == "success"
    second_model = FixtureInference(model="fixture-model-b")
    second_pipeline = Pipeline(config, retriever=fetcher, inference=second_model, policy=policy(), live=True)
    second = await second_pipeline.extract_url(REQUEST)
    assert second.status == "failed"
    assert second.error.code == "BUDGET_EXHAUSTED"
    assert not second_model.calls
    assert fetcher.calls == 1


async def test_locale_change_reuses_fresh_retrieval_but_revalidates_extraction(tmp_path):
    fetcher = FixtureRetriever()
    inference = FixtureInference()
    pipeline = Pipeline(
        settings(tmp_path), retriever=fetcher, inference=inference, policy=policy(), live=False
    )
    first = await pipeline.extract_url(REQUEST)
    second = await pipeline.extract_url(ExtractionRequest(url=URL, locale="en_US"))
    assert first.status == second.status == "success"
    assert fetcher.calls == 1
    assert len(inference.calls) == 2
    assert first.provenance.source_hash == second.provenance.source_hash


async def test_semantic_conflict_cannot_replace_trusted_product_identity(tmp_path):
    record = {
        "input": {"url": URL},
        "final_url": URL,
        "page_html": "<h1 class='product_title'>Semantic Hoodie</h1><p>Related accessory: Other Item</p><p>USD 42.00</p>",
    }

    class GapInference(FixtureInference):
        async def extract(self, snapshot, feedback=None):
            self.calls.append((snapshot.source_hash, feedback))
            evidence = []
            fields = {"name": "Other Item", "price": "42.00", "currency": "USD"}
            for field_path, value in fields.items():
                section = next(s for s in snapshot.sections if value in s.text)
                evidence.append(
                    {
                        "field_path": field_path,
                        "section_id": section.id,
                        "json_pointer": None,
                        "excerpt": section.text,
                    }
                )
            return InferenceResponse(payload={"data": ProductV1(**fields).model_dump(), "evidence": evidence})

    inference = GapInference()
    pipeline = Pipeline(
        settings(tmp_path),
        retriever=FixtureRetriever(record),
        inference=inference,
        policy=policy(),
        live=False,
    )
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "failed"
    assert result.error.code == "INVALID_EXTRACTION"
    assert result.data is None
    assert len(inference.calls) == 2


async def test_semantic_nulls_preserve_existing_deterministic_fields(tmp_path):
    record = {
        "input": {"url": URL},
        "final_url": URL,
        "page_html": "<h1 class='product_title'>Semantic Hoodie</h1><p>Current selling price: USD 42.00</p>",
    }

    class GapInference(FixtureInference):
        async def extract(self, snapshot, feedback=None):
            self.calls.append((snapshot.source_hash, feedback))
            section = next(s for s in snapshot.sections if "USD 42.00" in s.text)
            return InferenceResponse(
                payload={
                    "data": ProductV1(price="42.00", currency="USD").model_dump(),
                    "evidence": [
                        {"field_path": field, "section_id": section.id, "excerpt": section.text}
                        for field in ("price", "currency")
                    ],
                }
            )

    inference = GapInference()
    pipeline = Pipeline(
        settings(tmp_path),
        retriever=FixtureRetriever(record),
        inference=inference,
        policy=policy(),
        live=False,
    )
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "success"
    assert result.data.name == "Semantic Hoodie"
    assert result.evidence["name"].excerpt == "Semantic Hoodie"


async def test_saved_redirect_chain_is_rechecked_after_policy_changes(tmp_path):
    record = {**LD_RECORD, "redirect_chain": ["https://previously-allowed.test/intermediate", URL]}
    fetcher = FixtureRetriever(record)
    initial_policy = URLPolicy(["shop.test", "previously-allowed.test"], resolve_dns=False)
    initial = Pipeline(settings(tmp_path), retriever=fetcher, policy=initial_policy, live=False)
    first = await initial.extract_url(REQUEST)
    assert first.status == "success"
    tightened = Pipeline(settings(tmp_path), retriever=fetcher, policy=policy(), live=False)
    resumed = await tightened.resume(first.job_id)
    assert resumed.status == "failed"
    assert resumed.error.code == "EGRESS_DENIED"
    assert fetcher.calls == 1


async def test_render_record_recovers_after_crash_before_snapshot_save(tmp_path, monkeypatch):
    shell = {
        "input": {"url": URL},
        "final_url": URL,
        "page_html": '<div id="root"></div><script src="bundle.js"></script>',
    }
    fetcher = FixtureRetriever(shell)

    class Renderer:
        def __init__(self):
            self.calls = 0

        async def render(self, request):
            self.calls += 1
            return RetrievalOutcome(
                status="ready", record=LD_RECORD, retrieval_method="brightdata_web_unlocker", attempts=1
            )

    renderer = Renderer()
    initial = Pipeline(settings(tmp_path), retriever=fetcher, renderer=renderer, policy=policy(), live=False)
    save_snapshot = initial.store.save_snapshot
    saves = []

    def crash_before_rendered_snapshot(job_id, snapshot):
        saves.append(snapshot.retrieval_method)
        if snapshot.retrieval_method == "brightdata_web_unlocker":
            raise SystemExit("simulated process exit after renderer result checkpoint")
        return save_snapshot(job_id, snapshot)

    monkeypatch.setattr(initial.store, "save_snapshot", crash_before_rendered_snapshot)
    with pytest.raises(SystemExit):
        await initial.extract_url(REQUEST)
    job = initial.store.get_or_create(REQUEST, {"method": "brightdata_fetch"})
    assert job["render_record"] == LD_RECORD
    assert job["render_attempted"]
    restarted = Pipeline(
        settings(tmp_path), retriever=fetcher, renderer=renderer, policy=policy(), live=False
    )
    result = await restarted.resume(job["id"])
    assert result.status == "success"
    assert result.provenance.retrieval_method == "brightdata_web_unlocker"
    assert fetcher.calls == renderer.calls == 1
