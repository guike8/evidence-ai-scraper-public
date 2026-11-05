import json
from typing import ClassVar

import httpx
import pytest
from pydantic import SecretStr

from ai_scraper.benchmark import _bind_evidence, benchmark_manifest
from ai_scraper.config import Settings
from ai_scraper.contracts import InferenceResponse, RetrievalOutcome, ScraperError
from ai_scraper.extraction.deterministic import deterministic_extract
from ai_scraper.normalize import normalize_record
from ai_scraper.policy import URLPolicy
from ai_scraper.schemas import ExtractionRequest

URL = "https://shop.test/hoodie"
PRODUCT = {
    "@type": "Product",
    "name": "Bench Hoodie",
    "url": URL,
    "offers": {"price": "19.95", "priceCurrency": "USD"},
}
HTML = (
    '<h1 class="product_title">Bench Hoodie</h1><p class="price">USD 19.95</p>'
    + '<script type="application/ld+json">'
    + json.dumps(PRODUCT)
    + "</script>"
)
RECORD = {"input": {"url": URL}, "final_url": URL, "page_html": HTML}


def setup(tmp_path, **overrides):
    manifest = {
        "pages": [
            {
                "url": URL,
                "expected": {
                    "name": "Bench Hoodie",
                    "price": "19.95",
                    "price_text": "USD 19.95",
                    "currency": "USD",
                },
            }
        ],
        "candidates": [
            {"provider": "cerebras", "model": "candidate-a"},
            {"provider": "cerebras", "model": "candidate-b"},
        ],
    }
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(manifest))
    settings = Settings(
        state_dir=tmp_path / "state",
        allowed_domains=["shop.test"],
        brightdata_api_key=SecretStr("bdsecret"),
        hf_token=SecretStr("hfsecret"),
        firecrawl_api_key=SecretStr("fcsecret"),
        max_spend_usd=10.0,
        retrieval_record_usd=0.0015,
        hf_input_million_usd=1.0,
        hf_output_million_usd=1.0,
        firecrawl_record_usd=0.01,
    )
    return path, settings.model_copy(update=overrides)


class FakeRetriever:
    calls = 0

    def __init__(self, token):
        assert token == "bdsecret"

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        type(self).calls += 1
        checkpoint({"state": "fetched", "record": RECORD, "attempts": 1})
        return RetrievalOutcome(status="ready", record=RECORD, attempts=1)

    async def close(self):
        pass


class FakeExtractor:
    instances: ClassVar[list] = []

    def __init__(self, provider, model, token, **kwargs):
        self.provider, self.model = provider, model
        self.snapshots = []
        self.last_usage = {}
        self.probes = 0
        type(self).instances.append(self)

    async def probe(self):
        self.probes += 1
        return {
            "passed": True,
            "provider": self.provider,
            "model": self.model,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        }

    async def extract(self, snapshot, feedback=None):
        self.snapshots.append(snapshot)
        self.last_usage = {"input_tokens": 10, "output_tokens": 4}
        return InferenceResponse(deterministic_extract(snapshot).model_dump(), dict(self.last_usage))

    async def close(self):
        pass


def firecrawl_response(request):
    body = json.loads(request.content)
    assert body["url"] == URL
    assert body["formats"][2]["type"] == "json"
    assert body["formats"][2]["schema"]["properties"]["data"]
    snapshot = normalize_record(RECORD, ExtractionRequest(url=URL))
    return httpx.Response(
        200,
        json={
            "success": True,
            "data": {
                "rawHtml": HTML,
                "markdown": "Bench Hoodie\nUSD 19.95",
                "metadata": {"url": URL},
                "json": deterministic_extract(snapshot).model_dump(),
            },
        },
    )


async def test_missing_configuration_does_not_read_manifest_or_call_network(tmp_path):
    report = await benchmark_manifest(tmp_path / "does-not-exist.json", Settings())
    assert report["status"] == "not_run"
    assert report["network_calls"] == 0
    assert not report["live_verified"]
    assert "BRIGHTDATA_API_KEY" in report["missing_configuration"]
    assert "FIRECRAWL_RECORD_USD" in report["missing_configuration"]


async def test_two_probes_share_one_immutable_fetch_and_compare_firecrawl(tmp_path):
    path, settings = setup(tmp_path)
    FakeRetriever.calls = 0
    FakeExtractor.instances = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(firecrawl_response)) as client:
        report = await benchmark_manifest(
            path,
            settings,
            retriever_factory=FakeRetriever,
            extractor_factory=FakeExtractor,
            firecrawl_client=client,
            policy=URLPolicy(["shop.test"], resolve_dns=False),
        )
    assert report["status"] == "complete"
    assert report["live_verified"]
    assert FakeRetriever.calls == 1
    assert len(FakeExtractor.instances) == 2
    a, b = FakeExtractor.instances
    assert a.probes == b.probes == 1
    assert a.snapshots[0].source_hash == b.snapshots[0].source_hash
    assert a.snapshots[0].raw_record == b.snapshots[0].raw_record
    assert len(report["configurations"]) == 3
    for config in report["configurations"]:
        assert config["evaluation"]["all"]["true_positive"] == 3
        assert config["evaluation"]["gates"]["passed"]
        assert config["cost_estimate_usd"] > 0
        assert config["correct_accepted_records"] == 1
        assert config["cost_per_correct_accepted_record_usd"] == config["cost_estimate_usd"]
    assert report["selected_hf"]["model"] in {"candidate-a", "candidate-b"}
    assert report["reserved_spend_usd"] <= settings.max_spend_usd
    assert list((settings.state_dir / "benchmarks").glob("*.json"))
    assert all(secret not in json.dumps(report) for secret in ("bdsecret", "hfsecret", "fcsecret"))


async def test_budget_zero_prevents_all_vendor_actions_and_fabricated_measurements(tmp_path):
    path, settings = setup(tmp_path, max_spend_usd=0.0)
    FakeRetriever.calls = 0
    FakeExtractor.instances = []

    def unexpected(request):
        raise AssertionError("No HTTP call should happen after failed budget reservation")

    async with httpx.AsyncClient(transport=httpx.MockTransport(unexpected)) as client:
        report = await benchmark_manifest(
            path,
            settings,
            retriever_factory=FakeRetriever,
            extractor_factory=FakeExtractor,
            firecrawl_client=client,
            policy=URLPolicy(["shop.test"], resolve_dns=False),
        )
    assert FakeRetriever.calls == 0
    assert all(instance.probes == 0 for instance in FakeExtractor.instances)
    assert not report["live_verified"]
    assert report["selected_hf"] is None
    assert all(config["cost_estimate_usd"] == 0 for config in report["configurations"])
    for config in report["configurations"]:
        assert config["evaluation"]["all"]["present"] == 3
        assert config["evaluation"]["all"]["recall"] == 0


async def test_probe_failure_stays_in_all_ground_truth_denominators(tmp_path):
    path, settings = setup(tmp_path)

    class FailingProbe(FakeExtractor):
        async def probe(self):
            raise ScraperError("CAPABILITY_PROBE_FAILED", "Unsupported schema")

    async with httpx.AsyncClient(transport=httpx.MockTransport(firecrawl_response)) as client:
        report = await benchmark_manifest(
            path,
            settings,
            retriever_factory=FakeRetriever,
            extractor_factory=FailingProbe,
            firecrawl_client=client,
            policy=URLPolicy(["shop.test"], resolve_dns=False),
        )
    for config in report["configurations"][:2]:
        assert config["evaluation"]["all"]["present"] == 3
        assert config["evaluation"]["all"]["true_positive"] == 0
        assert config["results"][0]["error"]["code"] == "CAPABILITY_PROBE_FAILED"
    assert report["selected_hf"] is None
    assert not report["live_verified"]


async def test_repair_is_bounded_and_does_not_refetch(tmp_path):
    path, settings = setup(tmp_path)
    FakeRetriever.calls = 0

    class NeedsRepair(FakeExtractor):
        async def extract(self, snapshot, feedback=None):
            if not self.snapshots:
                self.snapshots.append(snapshot)
                return InferenceResponse(
                    {"data": {"name": "Unsupported"}, "evidence": [], "warnings": []},
                    {"input_tokens": 10, "output_tokens": 4},
                )
            assert feedback and "SAME saved source" in feedback
            return await super().extract(snapshot, feedback)

    NeedsRepair.instances = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(firecrawl_response)) as client:
        report = await benchmark_manifest(
            path,
            settings,
            retriever_factory=FakeRetriever,
            extractor_factory=NeedsRepair,
            firecrawl_client=client,
            policy=URLPolicy(["shop.test"], resolve_dns=False),
        )
    assert FakeRetriever.calls == 1
    for config in report["configurations"][:2]:
        assert config["results"][0]["usage"]["inference_attempts"] == 2
        assert config["results"][0]["usage"]["input_tokens"] == 20
        assert config["evaluation"]["gates"]["passed"]


async def test_invalid_candidate_prices_rejected_before_billable_actions(tmp_path):
    path, settings = setup(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["candidates"][0]["input_million_usd"] = -1
    path.write_text(json.dumps(manifest))
    FakeRetriever.calls = 0
    with pytest.raises(ValueError, match="finite and nonnegative"):
        await benchmark_manifest(
            path,
            settings,
            retriever_factory=FakeRetriever,
            policy=URLPolicy(["shop.test"], resolve_dns=False),
        )
    assert FakeRetriever.calls == 0


def test_firecrawl_unbound_quotes_do_not_accept_fabricated_references():
    snapshot = normalize_record(RECORD, ExtractionRequest(url=URL))
    payload = deterministic_extract(snapshot).model_dump()
    for evidence in payload["evidence"]:
        if evidence["field_path"] == "name":
            evidence.update(section_id="invented", json_pointer=None, excerpt="Absent Product")
    bound = _bind_evidence(payload, snapshot, None)
    name = next(item for item in bound["evidence"] if item["field_path"] == "name")
    # The application can replace an unusable quote only with independently
    # supported deterministic evidence for the same field value.
    assert name["section_id"] != "invented"


async def test_unreviewed_template_never_makes_vendor_calls(tmp_path):
    path, settings = setup(tmp_path)
    manifest = json.loads(path.read_text())
    manifest["labels_status"] = "review_required"
    path.write_text(json.dumps(manifest))
    FakeRetriever.calls = 0
    report = await benchmark_manifest(path, settings, retriever_factory=FakeRetriever)
    assert report["status"] == "not_run"
    assert report["network_calls"] == 0
    assert "set labels_status=reviewed" in report["reason"]
    assert FakeRetriever.calls == 0


async def test_direct_blank_secret_is_missing_configuration(tmp_path):
    path, settings = setup(tmp_path, hf_token=SecretStr(""))
    report = await benchmark_manifest(path, settings)
    assert report["status"] == "not_run"
    assert "HF_TOKEN" in report["missing_configuration"]
    assert report["network_calls"] == 0


@pytest.mark.parametrize("declared", [False, True], ids=["streamed", "content-length"])
async def test_oversized_firecrawl_response_is_terminal_content_error(tmp_path, declared):
    path, settings = setup(tmp_path, max_response_bytes=1024)

    class OversizedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"x" * 600
            yield b"x" * 600

    def oversized_response(request):
        if declared:
            return httpx.Response(200, headers={"Content-Length": "2048"}, content=b"{}")
        return httpx.Response(200, stream=OversizedStream())

    async with httpx.AsyncClient(transport=httpx.MockTransport(oversized_response)) as client:
        report = await benchmark_manifest(
            path,
            settings,
            retriever_factory=FakeRetriever,
            extractor_factory=FakeExtractor,
            firecrawl_client=client,
            policy=URLPolicy(["shop.test"], resolve_dns=False),
        )
    firecrawl = report["configurations"][-1]
    result = firecrawl["results"][0]
    assert result["status"] == "failed"
    assert result["data"] is None
    assert result["error"]["code"] == "CONTENT_TOO_LARGE"
    assert firecrawl["evaluation"]["all"]["true_positive"] == 0
    assert firecrawl["evaluation"]["all"]["present"] == 3
    assert not report["live_verified"]


async def test_default_brightdata_adapter_receives_configured_body_limit(tmp_path, monkeypatch):
    import ai_scraper.benchmark as benchmark_module

    path, settings = setup(tmp_path, max_response_bytes=4096)
    observed_limits = []

    def default_retriever(token, *, max_response_bytes):
        observed_limits.append(max_response_bytes)
        return FakeRetriever(token)

    monkeypatch.setattr(benchmark_module, "BrightDataRetriever", default_retriever)
    async with httpx.AsyncClient(transport=httpx.MockTransport(firecrawl_response)) as client:
        await benchmark_manifest(
            path,
            settings,
            extractor_factory=FakeExtractor,
            firecrawl_client=client,
            policy=URLPolicy(["shop.test"], resolve_dns=False),
        )
    assert observed_limits == [4096]
