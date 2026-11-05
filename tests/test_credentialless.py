from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest

from ai_scraper.cli import _ReplayRetriever, main
from ai_scraper.config import Settings
from ai_scraper.contracts import InferenceResponse
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.normalize import normalize_record
from ai_scraper.pipeline import Pipeline, create_pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.direct import DirectHTTPRetriever
from ai_scraper.schemas import ExtractionRequest, ProductV1, SourceSnapshot
from ai_scraper.validate import validate_candidate

URL = "https://studio-shop.test/products/hoodie"
TEXT = "# Studio Hoodie\nPrice: USD 49.95\nA cotton hoodie for everyday use."


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    # CLI and the deployment wrapper deliberately update their process config.
    # Isolate that mutation, without relying on credentials in the test runner.
    clean = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("HF_", "LOCAL_", "BRIGHTDATA_", "FIRECRAWL_", "SCRAPER_"))
    }
    monkeypatch.setattr(os, "environ", clean)


class FakeLocalInference:
    provider = "local"
    model = "Studio-MLX"

    def __init__(self, *, bad_calls=0):
        self.calls = []
        self.bad_calls = bad_calls
        self.closed = False

    async def extract(self, snapshot, feedback=None):
        self.calls.append((snapshot.source_hash, feedback))
        fields = {"name": "Studio Hoodie", "price": "49.95", "currency": "USD"}
        evidence = []
        for field, value in fields.items():
            section = next(section for section in snapshot.sections if value in section.text)
            evidence.append(
                {"field_path": field, "section_id": section.id, "excerpt": section.text, "json_pointer": None}
            )
        if len(self.calls) <= self.bad_calls:
            evidence[0]["excerpt"] = "Unsupported invented quotation"
        return InferenceResponse(
            payload={"data": ProductV1(**fields).model_dump(), "evidence": evidence, "warnings": []},
            usage={"input_tokens": 100, "output_tokens": 80},
        )

    async def close(self):
        self.closed = True


def local_settings(tmp_path, **overrides):
    return Settings(
        state_dir=tmp_path / "state",
        retrieval_backend="direct",
        inference_backend="local",
        local_model="Studio-MLX",
        allowed_domains=["studio-shop.test"],
        **overrides,
    )


def product_record(name: str, price: str) -> dict:
    return {
        "final_url": URL,
        "page_html": f'<h1 class="product_title">{name}</h1><p class="price">USD {price}</p>',
    }


@pytest.mark.parametrize("legacy_collision", [False, True])
async def test_live_direct_fetch_never_reuses_a_replay_snapshot(tmp_path, legacy_collision):
    settings = local_settings(tmp_path)
    policy = URLPolicy(settings.allowed_domains, resolve_dns=False)
    request = ExtractionRequest(url=URL)
    fixture = product_record("Authored Fixture Hoodie", "5.00")
    offline = Pipeline(settings, retriever=_ReplayRetriever(fixture), policy=policy, live=False)
    if legacy_collision:
        # Reproduce the old bug: a fixture record persisted under the live
        # direct-http request key. Its recorded method still identifies it.
        polluted = offline.store.get_or_create(request, {"method": "direct_http"})
        snapshot = normalize_record(fixture, request, retrieval_method="fixture_replay")
        offline.store.update(
            polluted["id"], record=fixture, retrieval_method="fixture_replay", state="fetched"
        )
        offline.store.save_snapshot(polluted["id"], snapshot)
        offline.store.update(polluted["id"], state="complete")
        fixture_job_id = polluted["id"]
    else:
        replayed = await offline.extract_url(request)
        assert replayed.status == "success" and replayed.data.price == "5.00"
        fixture_job_id = replayed.job_id
    await offline.close()
    calls = []

    def real_page(http_request):
        calls.append(http_request)
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            text=product_record("Actual Live Hoodie", "21.00")["page_html"],
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(real_page)) as client:
        live = Pipeline(settings, retriever=DirectHTTPRetriever(policy, client=client), policy=policy)
        result = await live.extract_url(request)
        assert result.status == "success"
        assert result.job_id != fixture_job_id
        assert result.data.name == "Actual Live Hoodie" and result.data.price == "21.00"
        assert result.provenance.retrieval_method == "direct_http"
        assert len(calls) == result.usage.retrieval_attempts == 1
        assert live.store.reserved_spend() == result.usage.cost_estimate_usd == 0.0
        saved = SourceSnapshot.model_validate_json(Path(result.provenance.snapshot_path).read_text())
        assert "Actual Live Hoodie" in saved.raw_record["page_html"]
        assert "Authored Fixture Hoodie" not in saved.raw_record["page_html"]
        await live.close()


async def test_replay_cache_is_content_specific_and_reuses_identical_records(tmp_path):
    settings = local_settings(tmp_path)
    policy = URLPolicy(settings.allowed_domains, resolve_dns=False)
    request = ExtractionRequest(url=URL)

    class CountedReplay(_ReplayRetriever):
        def __init__(self, record):
            super().__init__(record)
            self.calls = 0

        async def retrieve(self, *args, **kwargs):
            self.calls += 1
            return await super().retrieve(*args, **kwargs)

    first_retriever = CountedReplay(product_record("First Fixture", "10.00"))
    first_pipeline = Pipeline(settings, retriever=first_retriever, policy=policy, live=False)
    first = await first_pipeline.extract_url(request)
    second_retriever = CountedReplay(product_record("Updated Fixture", "20.00"))
    second_pipeline = Pipeline(settings, retriever=second_retriever, policy=policy, live=False)
    second = await second_pipeline.extract_url(request)
    assert first.status == second.status == "success"
    assert first.job_id != second.job_id
    assert first.data.price == "10.00" and second.data.price == "20.00"
    same_record = dict(reversed(list(product_record("First Fixture", "10.00").items())))
    third_retriever = CountedReplay(same_record)
    third_pipeline = Pipeline(settings, retriever=third_retriever, policy=policy, live=False)
    third = await third_pipeline.extract_url(request)
    assert third.status == "success" and third.job_id == first.job_id
    assert third.data.name == "First Fixture" and third.usage.cache_hit
    assert first_retriever.calls == second_retriever.calls == 1
    assert third_retriever.calls == 0
    for pipeline in (first_pipeline, second_pipeline, third_pipeline):
        await pipeline.close()


def test_unknown_pending_vendor_method_keeps_existing_job_identity(tmp_path):
    pipeline = Pipeline(local_settings(tmp_path), live=False)
    request = ExtractionRequest(url=URL)
    job = pipeline.store.get_or_create(request, {"method": "brightdata_fetch"})
    pipeline.store.update(job["id"], state="submitting")
    resumed = pipeline.store.get_or_create(request, {"method": "brightdata_fetch"})
    assert resumed["id"] == job["id"]
    assert resumed["state"] == "submitting"


@pytest.mark.parametrize("legacy_prices", [False, True])
async def test_direct_local_end_to_end_without_keys_or_billable_budget(tmp_path, legacy_prices):
    calls = []

    def page(request):
        calls.append(request)
        assert request.method == "GET" and str(request.url) == URL
        assert "Authorization" not in request.headers
        return httpx.Response(200, headers={"Content-Type": "text/markdown"}, text=TEXT)

    overrides = (
        {
            "max_spend_usd": 0.0,
            "retrieval_record_usd": 999.0,
            "hf_input_million_usd": 999.0,
            "hf_output_million_usd": 999.0,
        }
        if legacy_prices
        else {}
    )
    settings = local_settings(tmp_path, **overrides)
    assert settings.brightdata_api_key is settings.hf_token is settings.firecrawl_api_key is None
    policy = URLPolicy(settings.allowed_domains, resolve_dns=False)
    inference = FakeLocalInference()
    async with httpx.AsyncClient(transport=httpx.MockTransport(page)) as client:
        retriever = DirectHTTPRetriever(policy, client=client)
        pipeline = Pipeline(settings, retriever=retriever, inference=inference, policy=policy)
        result = await pipeline.extract_url(ExtractionRequest(url=URL))
        assert result.status == "success"
        assert result.provenance.retrieval_method == "direct_http"
        assert result.provenance.provider == "local"
        assert result.provenance.model == "Studio-MLX"
        assert result.data.name == "Studio Hoodie"
        assert len(calls) == len(inference.calls) == result.usage.inference_attempts == 1
        assert result.usage.cost_estimate_usd == pipeline.store.reserved_spend() == 0.0
        saved = SourceSnapshot.model_validate_json(Path(result.provenance.snapshot_path).read_text())
        assert saved.raw_record["markdown"] == TEXT
        assert (
            validate_candidate(
                {
                    "data": result.data.model_dump(),
                    "evidence": [e.model_dump() for e in result.evidence.values()],
                },
                saved,
            ).status
            == "success"
        )
        exported = json.loads(Path(pipeline.store.get(result.job_id)["result_path"]).read_text())
        assert exported["status"] == "success" and exported["usage"]["cost_estimate_usd"] == 0.0
        assert (await pipeline.resume(result.job_id)).status == "success"
        assert len(calls) == len(inference.calls) == 1
        await pipeline.close()
    assert inference.closed


async def test_local_repair_remains_bounded_without_a_money_cap(tmp_path):
    policy = URLPolicy(["studio-shop.test"], resolve_dns=False)
    inference = FakeLocalInference(bad_calls=99)
    calls = []

    def page(request):
        calls.append(request)
        return httpx.Response(200, headers={"Content-Type": "text/plain"}, text=TEXT)

    async with httpx.AsyncClient(transport=httpx.MockTransport(page)) as client:
        pipeline = Pipeline(
            local_settings(tmp_path),
            retriever=DirectHTTPRetriever(policy, client=client),
            inference=inference,
            policy=policy,
        )
        result = await pipeline.extract_url(ExtractionRequest(url=URL))
        assert result.status == "failed" and result.data is None
        assert result.error.code == "INVALID_EXTRACTION"
        assert len(calls) == 1 and len(inference.calls) == 2
        assert len({call[0] for call in inference.calls}) == 1
        assert inference.calls[1][1] is not None
        assert pipeline.store.reserved_spend() == 0.0
        resumed = await pipeline.resume(result.job_id)
        assert resumed.status == "failed"
        assert len(calls) == 1 and len(inference.calls) == 2
        await pipeline.close()


async def test_factory_builds_real_direct_and_loopback_adapters_without_keys(tmp_path):
    pipeline = create_pipeline(local_settings(tmp_path))
    assert isinstance(pipeline.retriever, DirectHTTPRetriever)
    assert isinstance(pipeline.inference, LocalExtractor)
    assert pipeline.inference.provider == "local"
    assert pipeline.inference.base_url == "http://127.0.0.1:8000/v1"
    await pipeline.close()
    assert pipeline.retriever.client.is_closed
    assert pipeline.inference._local_client.client.is_closed


def test_cli_local_probe_requires_no_cloud_credentials_prices_or_cap(tmp_path, monkeypatch, capsys):
    created = []

    class ProbePipeline:
        def __init__(self):
            self.inference = self
            self.closed = False

        async def probe(self):
            return {
                "passed": True,
                "provider": "local",
                "model": "Studio-MLX",
                "usage": {"input_tokens": 10, "output_tokens": 20},
            }

        async def close(self):
            self.closed = True

    def factory(settings):
        assert settings.hf_token is settings.brightdata_api_key is settings.max_spend_usd is None
        instance = ProbePipeline()
        created.append(instance)
        return instance

    monkeypatch.setenv("SCRAPER_INFERENCE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_MODEL", "Studio-MLX")
    monkeypatch.setenv("SCRAPER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr("ai_scraper.pipeline.create_pipeline", factory)
    assert main(["probe"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["passed"] is True
    assert result["usage"]["cost_estimate_usd"] == 0.0
    assert len(created) == 1 and created[0].closed


def test_cli_replay_can_use_explicit_local_inference_without_external_retrieval(
    tmp_path, monkeypatch, capsys
):
    inference = FakeLocalInference()
    closed = []

    class PlaceholderRetriever:
        async def close(self):
            closed.append(True)

    class Configured:
        retriever = PlaceholderRetriever()

    configured = Configured()
    configured.inference = inference
    monkeypatch.setattr("ai_scraper.pipeline.create_pipeline", lambda settings: configured)
    monkeypatch.setenv("SCRAPER_INFERENCE_BACKEND", "local")
    monkeypatch.setenv("SCRAPER_RETRIEVAL_BACKEND", "direct")
    monkeypatch.setenv("LOCAL_MODEL", "Studio-MLX")
    monkeypatch.setenv("SCRAPER_ALLOWED_DOMAINS", "studio-shop.test")
    path = tmp_path / "record.json"
    path.write_text(json.dumps({"markdown": TEXT, "final_url": URL}))
    assert (
        main(
            [
                "--state-dir",
                str(tmp_path / "state"),
                "replay",
                str(path),
                "--url",
                URL,
                "--schema",
                "product_v1",
                "--local-inference",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "success" and result["provenance"]["provider"] == "local"
    assert result["usage"]["cost_estimate_usd"] == 0.0
    assert len(inference.calls) == 1 and inference.closed and closed == [True]


def test_deployment_status_accepts_direct_local_with_no_api_settings(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location(
        "credentialless_launch", Path(__file__).parents[1] / "deploy" / "launch.py"
    )
    deployment = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deployment)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "SCRAPER_RETRIEVAL_BACKEND": "direct",
                "SCRAPER_INFERENCE_BACKEND": "local",
                "LOCAL_MODEL": "Studio-MLX",
                "LOCAL_BASE_URL": "http://127.0.0.1:8000/v1",
                "SCRAPER_ALLOWED_DOMAINS": "studio-shop.test",
            }
        )
    )
    config.chmod(0o600)
    assert deployment.main(["--root", str(tmp_path), "status"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["live_retrieval_configured"] is report["live_inference_configured"] is True
    assert report["missing_for_retrieval"] == report["missing_for_inference"] == []
    assert report["live_verified"] is False
    assert report["configured"]["HF_TOKEN"] is report["configured"]["BRIGHTDATA_API_KEY"] is False
    assert report["configured"]["SCRAPER_MAX_SPEND_USD"] is False
