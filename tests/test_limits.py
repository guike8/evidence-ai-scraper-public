from __future__ import annotations

import json

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from ai_scraper.cli import _load_record, main
from ai_scraper.config import Settings
from ai_scraper.contracts import RetrievalOutcome, ScraperError
from ai_scraper.jobs import JobStore
from ai_scraper.normalize import normalize_record
from ai_scraper.pipeline import Pipeline, create_pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.brightdata import BrightDataRetriever
from ai_scraper.schemas import ExtractionRequest

URL = "https://limits.test/product"
REQUEST = ExtractionRequest(url=URL)
SHELL = {"input": {"url": URL}, "page_html": '<div id="root"></div><script src="/app.js"></script>'}


def test_raw_record_byte_limit_prevents_normalization():
    with pytest.raises(ScraperError) as error:
        normalize_record({"markdown": "界" * 20}, REQUEST, max_record_bytes=50)
    assert error.value.error.code == "CONTENT_TOO_LARGE"


def test_normalized_representation_expansion_fails_without_truncating():
    record = {"page_html": '<h1 class="product_title">' + "x" * 80 + "</h1>"}
    # A heading is retained as both a block and a text line. The raw record fits;
    # its cumulative evidence text crosses the separately bounded source limit.
    assert len(json.dumps(record).encode()) < 1000
    with pytest.raises(ScraperError) as error:
        normalize_record(record, REQUEST, max_record_bytes=1000, max_normalized_bytes=100)
    assert error.value.error.code == "CONTENT_TOO_LARGE"
    complete = normalize_record(record, REQUEST, max_record_bytes=1000, max_normalized_bytes=1000)
    assert len(complete.sections) == 2
    assert all(section.text == "x" * 80 for section in complete.sections)
    assert complete.excluded_sections == []


def test_snapshot_storage_cap_prevents_writing_oversized_source(tmp_path):
    store = JobStore(tmp_path / "state", max_snapshot_bytes=1024)
    job = store.get_or_create(REQUEST)
    snapshot = normalize_record({"markdown": "x" * 1500}, REQUEST)
    with pytest.raises(ScraperError) as error:
        store.save_snapshot(job["id"], snapshot)
    assert error.value.error.code == "CONTENT_TOO_LARGE"
    assert not list((store.root / "snapshots").glob("*.json"))
    assert "snapshot_path" not in store.get(job["id"])


@pytest.mark.asyncio
async def test_resume_rejects_oversized_snapshot_before_parsing(tmp_path):
    settings = Settings(
        state_dir=tmp_path / "state", allowed_domains=["limits.test"], max_snapshot_bytes=1024
    )
    pipeline = Pipeline(settings, policy=URLPolicy(["limits.test"], resolve_dns=False), live=False)
    job = pipeline.store.get_or_create(REQUEST)
    path = pipeline.store.root / "oversized.json"
    path.write_text("This is deliberately invalid JSON." * 100)
    pipeline.store.update(job["id"], snapshot_path=str(path), state="complete")
    result = await pipeline.resume(job["id"])
    assert result.status == "failed"
    # Parsing this invalid document would instead produce LOCAL_STATE_ERROR.
    assert result.error.code == "CONTENT_TOO_LARGE"
    await pipeline.close()


class ShellRetriever:
    def __init__(self):
        self.calls = 0

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        self.calls += 1
        checkpoint({"record": SHELL, "state": "fetched", "attempts": 1})
        return RetrievalOutcome(status="ready", record=SHELL, attempts=1)

    async def close(self):
        pass


class UnobservableRenderer:
    supports_observed_redirects = False

    def __init__(self):
        self.calls = 0

    async def render(self, request):
        self.calls += 1
        raise AssertionError("Unobservable rendering must be rejected before a paid request")

    async def close(self):
        pass


@pytest.mark.asyncio
async def test_live_unobservable_renderer_is_rejected_before_call_or_reservation(tmp_path):
    settings = Settings(
        state_dir=tmp_path / "state",
        allowed_domains=["limits.test"],
        max_spend_usd=1.0,
        retrieval_record_usd=0.01,
        render_record_usd=0.25,
    )
    renderer = UnobservableRenderer()
    retriever = ShellRetriever()
    pipeline = Pipeline(
        settings,
        retriever=retriever,
        renderer=renderer,
        policy=URLPolicy(["limits.test"], resolve_dns=False),
        live=True,
    )
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "failed"
    assert result.error.code == "RENDER_EGRESS_UNVERIFIABLE"
    assert retriever.calls == 1
    assert renderer.calls == 0
    assert pipeline.store.reserved_spend() == 0.01
    await pipeline.close()


@pytest.mark.asyncio
async def test_configured_unlocker_zone_does_not_enable_unverifiable_rendering(tmp_path):
    settings = Settings(
        state_dir=tmp_path / "state",
        allowed_domains=["limits.test"],
        brightdata_api_key=SecretStr("construction-only-no-network"),
        unlocker_zone="test_zone",
        max_spend_usd=1.0,
        retrieval_record_usd=0.01,
        render_record_usd=0.25,
    )
    pipeline = create_pipeline(settings)
    assert pipeline.renderer is None
    await pipeline.retriever.close()
    retriever = ShellRetriever()
    pipeline.retriever = retriever
    pipeline.policy = URLPolicy(["limits.test"], resolve_dns=False)
    result = await pipeline.extract_url(REQUEST)
    assert result.status == "failed"
    assert result.error.code == "RENDER_EGRESS_UNVERIFIABLE"
    assert retriever.calls == 1
    assert pipeline.store.reserved_spend() == 0.01
    await pipeline.close()


def test_cli_replay_uses_configured_file_size_limit_before_json_parse(monkeypatch, tmp_path, capsys):
    record = tmp_path / "large.json"
    record.write_text("Not JSON." * 300)
    with pytest.raises(ScraperError) as error:
        _load_record(record, URL, max_bytes=1024)
    assert error.value.error.code == "CONTENT_TOO_LARGE"
    monkeypatch.setenv("SCRAPER_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("SCRAPER_MAX_RESPONSE_BYTES", "1024")
    assert main(["replay", str(record), "--url", URL]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["error"]["code"] == "CONTENT_TOO_LARGE"


def test_response_limit_config_has_minimum_and_environment_names(monkeypatch):
    with pytest.raises(ValidationError):
        Settings(max_response_bytes=1023)
    monkeypatch.setenv("SCRAPER_MAX_RESPONSE_BYTES", "2048")
    monkeypatch.setenv("SCRAPER_MAX_SNAPSHOT_BYTES", "4096")
    settings = Settings.from_env()
    assert settings.max_response_bytes == 2048
    assert settings.max_snapshot_bytes == 4096


@pytest.mark.parametrize("live", [True, False])
@pytest.mark.parametrize("observed", [None, False])
@pytest.mark.asyncio
async def test_old_durable_raw_render_recovery_requires_observed_destination(
    tmp_path, monkeypatch, live, observed
):
    record = {
        "input": {"url": URL},
        "final_url": URL,
        "page_html": '<h1 class="product_title">Recovered Hoodie</h1><p class="price">USD 35.00</p>',
    }
    if observed is not None:
        record["observed_final_url"] = observed
    retriever, renderer = ShellRetriever(), UnobservableRenderer()
    settings = Settings(state_dir=tmp_path / "state", allowed_domains=["limits.test"])
    pipeline = Pipeline(
        settings,
        retriever=retriever,
        renderer=renderer,
        policy=URLPolicy(["limits.test"], resolve_dns=False),
        live=live,
    )
    job = pipeline.store.get_or_create(REQUEST, {"method": "brightdata_fetch"})
    pipeline.store.update(
        job["id"], state="fetched", render_record=record, render_method="brightdata_web_unlocker"
    )
    if live:

        def must_not_extract(*args, **kwargs):
            raise AssertionError("Legacy unobserved raw rendering must fail before extraction")

        monkeypatch.setattr("ai_scraper.pipeline.deterministic_extract", must_not_extract)
    result = await pipeline.resume(job["id"])
    if live:
        assert result.status == "failed"
        assert result.error.code == "RENDER_EGRESS_UNVERIFIABLE"
        assert not list((pipeline.store.root / "snapshots").glob("*.json"))
    else:
        assert result.status == "success"
        assert result.data.name == "Recovered Hoodie"
    assert retriever.calls == renderer.calls == 0
    assert pipeline.store.reserved_spend() == 0
    await pipeline.close()


@pytest.mark.parametrize("live", [True, False])
@pytest.mark.parametrize("observed", [None, False])
@pytest.mark.asyncio
async def test_old_raw_render_snapshot_only_resume_requires_observed_destination(
    tmp_path, monkeypatch, live, observed
):
    record = {
        "input": {"url": URL},
        "final_url": URL,
        "page_html": '<h1 class="product_title">Saved Hoodie</h1><p class="price">USD 35.00</p>',
    }
    if observed is not None:
        record["observed_final_url"] = observed
    settings = Settings(state_dir=tmp_path / "state", allowed_domains=["limits.test"])
    retriever, renderer = ShellRetriever(), UnobservableRenderer()
    pipeline = Pipeline(
        settings,
        retriever=retriever,
        renderer=renderer,
        policy=URLPolicy(["limits.test"], resolve_dns=False),
        live=live,
    )
    job = pipeline.store.get_or_create(REQUEST, {"method": "brightdata_fetch"})
    snapshot = normalize_record(record, REQUEST, retrieval_method="brightdata_web_unlocker")
    pipeline.store.save_snapshot(job["id"], snapshot)
    pipeline.store.update(
        job["id"],
        state="complete",
        retrieval_method="brightdata_web_unlocker",
        snapshot_retrieved_at=snapshot.retrieved_at.isoformat(),
    )
    if live:

        def must_not_extract(*args, **kwargs):
            raise AssertionError("Saved unobserved raw rendering must fail before extraction")

        monkeypatch.setattr("ai_scraper.pipeline.deterministic_extract", must_not_extract)
    result = await pipeline.resume(job["id"])
    if live:
        assert result.status == "failed"
        assert result.error.code == "RENDER_EGRESS_UNVERIFIABLE"
    else:
        assert result.status == "success"
        assert result.data.name == "Saved Hoodie"
    assert retriever.calls == renderer.calls == 0
    assert pipeline.store.reserved_spend() == 0
    await pipeline.close()


@pytest.mark.asyncio
async def test_pending_fetch_reuses_job_when_only_render_zone_configuration_changes(tmp_path):
    now = {"seconds": 0.0}
    requests = []

    def handler(request):
        requests.append((request.method, request.url.path))
        now["seconds"] += 1.0
        if request.method == "POST":
            return httpx.Response(202, json={"snapshot_id": "persisted-fetch"})
        return httpx.Response(200, json={"status": "running"})

    settings = Settings(
        state_dir=tmp_path / "state",
        allowed_domains=["limits.test"],
        wait_seconds=0.1,
        max_spend_usd=1.0,
        retrieval_record_usd=0.01,
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        first = Pipeline(
            settings,
            retriever=BrightDataRetriever("test-key", client=client, clock=lambda: now["seconds"]),
            policy=URLPolicy(["limits.test"], resolve_dns=False),
            live=True,
        )
        result = await first.extract_url(REQUEST)
        assert result.status == "pending"
        await first.close()
        restarted = Pipeline(
            settings.model_copy(update={"unlocker_zone": "changed_render_zone"}),
            retriever=BrightDataRetriever("test-key", client=client, clock=lambda: now["seconds"]),
            policy=URLPolicy(["limits.test"], resolve_dns=False),
            live=True,
        )
        resumed = await restarted.extract_url(REQUEST)
        assert resumed.status == "pending"
        assert resumed.job_id == result.job_id
        assert sum(method == "POST" for method, _ in requests) == 1
        assert restarted.store.reserved_spend() == 0.01
        await restarted.close()
