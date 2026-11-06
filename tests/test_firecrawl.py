from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import httpx
import pytest
from pydantic import ValidationError

from ai_scraper.config import Settings
from ai_scraper.contracts import ScraperError
from ai_scraper.crawl import Crawler, CrawlRequest
from ai_scraper.pipeline import Pipeline, create_pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.firecrawl import (
    FIRECRAWL_POLICY_VERSION,
    SERVICE_WARNING,
    FirecrawlRetriever,
    validate_firecrawl_endpoint,
)
from ai_scraper.schemas import BrowserAction, ExtractionRequest, SourceSnapshot

URL = "https://firecrawl.example.test/page"
ID = "019a52f7-8d5a-7000-8000-123456789abc"
RUNTIME = "e" * 40
ENDPOINT = "http://127.0.0.1:3002"
REQUEST = ExtractionRequest(url=URL, schema_id="page_v1")
RAW = "<html><head><title>Authored page</title></head><body><main><h1>Authored page</h1><p>Literal source paragraph.</p></main></body></html>"


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, delay):
        self.sleeps.append(delay)
        self.now += delay


def document(**updates):
    value = {
        "rawHtml": RAW,
        "html": "<main><h1>Authored page</h1><p>Literal source paragraph.</p></main>",
        "markdown": "# Authored page\n\nLiteral source paragraph.",
        "metadata": {
            "sourceURL": URL,
            "url": URL,
            "statusCode": 200,
            "title": "Authored page",
            "authorization": "never retain this metadata key",
        },
    }
    value.update(updates)
    return value


def completed(data=None, **updates):
    value = {
        "success": True,
        "status": "completed",
        "completed": 1,
        "total": 1,
        "data": [document()] if data is None else data,
    }
    value.update(updates)
    return value


def acknowledgment():
    return {"success": True, "id": ID, "url": "https://untrusted.invalid/do-not-follow", "invalidURLs": []}


def adapter(handler, clock=None, **overrides):
    clock = clock or Clock()
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        headers={
            "Authorization": "Bearer legacy-secret",
            "Cookie": "private=cookie",
            "Proxy-Authorization": "private",
        },
        follow_redirects=True,
    )
    return FirecrawlRetriever(
        URLPolicy(["firecrawl.example.test"], resolve_dns=False),
        base_url=ENDPOINT,
        runtime_id=RUNTIME,
        trust_service_egress=True,
        client=client,
        clock=clock,
        wall_clock=clock,
        sleep=clock.sleep,
        **overrides,
    )


def checkpoint(job, path=None):
    def save(patch):
        job.update(patch)
        if path:
            path.write_text(json.dumps(job))

    return save


def settings(tmp_path, **updates):
    return Settings(
        state_dir=tmp_path / "state",
        retrieval_backend="firecrawl",
        firecrawl_base_url=ENDPOINT,
        firecrawl_runtime_id=RUNTIME,
        firecrawl_trust_service_egress=True,
        allowed_domains=["firecrawl.example.test"],
        **updates,
    )


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://api.firecrawl.dev",
        "http://192.168.1.4:3002",
        "http://127.0.0.1",
        "http://127.0.0.1:80",
        "http://localhost:3002",
        "http://127.0.0.1:3002/v2",
        "http://user:pass@127.0.0.1:3002",
        "http://127.0.0.1:3002/?",
        "http://127.0.0.1:3002/#",
        "file:///tmp/api",
        "http://[::1%25lo0]:3002",
    ],
)
def test_endpoint_is_explicit_literal_loopback_without_redirect_or_credentials(endpoint):
    with pytest.raises(ValueError):
        validate_firecrawl_endpoint(endpoint)


def test_ipv6_loopback_endpoint_and_explicit_configuration(tmp_path):
    assert validate_firecrawl_endpoint("http://[::1]:3002/") == "http://[::1]:3002"
    assert settings(tmp_path).retrieval_backend == "firecrawl"
    for updates in (
        {"firecrawl_trust_service_egress": False},
        {"firecrawl_runtime_id": "main"},
        {"firecrawl_base_url": None},
    ):
        values = settings(tmp_path).model_dump()
        values.update(updates)
        with pytest.raises(ValidationError):
            Settings(**values)


async def test_async_one_url_batch_has_source_only_options_and_no_credentials():
    calls = []
    job = {"state": "new"}

    def handle(request):
        calls.append(request)
        assert request.url.host == "127.0.0.1"
        assert all(name not in request.headers for name in ("Authorization", "Cookie", "Proxy-Authorization"))
        if request.method == "POST":
            assert job["state"] == "submitting" and job["firecrawl_binding"]
            assert json.loads(request.content) == {
                "urls": [URL],
                "formats": ["rawHtml", "html", "markdown"],
                "parsers": [],
                "onlyMainContent": False,
                "maxAge": 0,
                "storeInCache": False,
                "skipTlsVerification": False,
                "ignoreInvalidURLs": False,
                "maxConcurrency": 1,
            }
            return httpx.Response(200, json=acknowledgment())
        assert request.url.path == f"/v2/batch/scrape/{ID}"
        return httpx.Response(200, json=completed())

    result = await adapter(handle).retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.status == "ready" and result.snapshot_id == ID
    assert result.warnings == [SERVICE_WARNING]
    assert result.record["page_html"] == RAW
    assert result.record["firecrawl_html"] != RAW
    assert "authorization" not in result.record["firecrawl_metadata"]
    assert "observed_final_url" not in result.record
    assert result.record["firecrawl_service"]["native_connection_pinning"] is False
    assert result.record["firecrawl_service"]["skip_tls_verification"] is False
    assert result.record["firecrawl_service"]["parsers"] == []
    assert job["state"] == "fetched" and [r.method for r in calls] == ["POST", "GET"]


@pytest.mark.parametrize("option, expected", [("skip_tls_verification", False), ("parsers", [])])
async def test_explicit_service_choices_are_bound_to_cache_and_saved_source(option, expected):
    from ai_scraper.jobs import digest

    def handle(request):
        return httpx.Response(200, json=acknowledgment() if request.method == "POST" else completed())

    instance = adapter(handle)
    assert instance.cache_options[option] == expected
    implicit_options = dict(instance.cache_options)
    implicit_options.pop(option)
    assert digest(implicit_options) != instance.binding
    job = {"state": "new"}
    assert (await instance.retrieve(REQUEST, job, checkpoint(job), 10)).status == "ready"
    saved_without_policy = copy.deepcopy(job)
    saved_without_policy["record"]["firecrawl_service"].pop(option)
    with pytest.raises(ScraperError, match="different Firecrawl configuration"):
        instance.validate_job_identity(saved_without_policy)
    await instance.client.aclose()


async def test_pending_poll_partial_data_and_restart_never_repeat_post(tmp_path):
    clock = Clock()
    calls = []
    path = tmp_path / "job.json"
    job = {"state": "new"}

    def first(request):
        calls.append(request.method)
        return httpx.Response(200, json=acknowledgment())

    def persist(patch):
        checkpoint(job, path)(patch)
        if patch.get("snapshot_id"):
            clock.now += 2

    assert (await adapter(first, clock).retrieve(REQUEST, job, persist, 1)).status == "pending"
    restored = json.loads(path.read_text())
    poll = iter(
        [
            {
                "success": True,
                "status": "scraping",
                "completed": 0,
                "total": 0,
                "data": [],
                "next": "http://127.0.0.1:22/forbidden",
            },
            {
                "success": True,
                "status": "scraping",
                "completed": 1,
                "total": 1,
                "data": [document()],
                "next": "https://evil.invalid/not-followed",
            },
            completed(),
        ]
    )

    def resume(request):
        calls.append(request.method)
        assert request.url.path == f"/v2/batch/scrape/{ID}"
        return httpx.Response(200, json=next(poll))

    result = await adapter(resume, clock).retrieve(REQUEST, restored, checkpoint(restored, path), 10)
    assert result.status == "ready" and calls.count("POST") == 1
    assert clock.sleeps == [1, 1]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, json={"success": True}),
        httpx.Response(202, json={"success": True, "id": "bad/id"}),
        httpx.Response(200, content=b'{"success":true,"id":'),
        httpx.Response(500),
        httpx.Response(408),
        httpx.Response(200, json={**acknowledgment(), "invalidURLs": [URL]}),
    ],
)
async def test_invalid_or_lost_acknowledgment_requires_reconciliation_without_resubmission(response):
    calls = []
    job = {"state": "new"}

    def handle(request):
        calls.append(request)
        return response

    instance = adapter(handle)
    first = await instance.retrieve(REQUEST, job, checkpoint(job), 10)
    assert first.status == "pending" and first.error.code == "RECONCILIATION_REQUIRED"
    assert job["state"] == "reconciliation_required"
    second = await instance.retrieve(REQUEST, job, checkpoint(job), 10)
    assert second.status == "pending" and len(calls) == 1


async def test_post_timeout_and_callback_failure_cannot_duplicate_work():
    calls = []
    job = {"state": "new"}

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("private-header-token", request=request)

    instance = adapter(handle)
    result = await instance.retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.error.code == "RECONCILIATION_REQUIRED"
    assert "private-header-token" not in result.error.model_dump_json()
    await instance.retrieve(REQUEST, job, checkpoint(job), 10)
    assert len(calls) == 1

    def fail(_):
        raise OSError("private-state-error")

    with pytest.raises(OSError):
        await instance.retrieve(REQUEST, {"state": "new"}, fail, 10)
    assert len(calls) == 1


async def test_acknowledgment_checkpoint_failure_leaves_submitting_state_and_no_repost(tmp_path):
    calls = []
    durable = {"state": "new"}
    active = copy.deepcopy(durable)

    def handle(request):
        calls.append(request.method)
        return httpx.Response(200, json=acknowledgment())

    def save(patch):
        if patch.get("snapshot_id"):
            raise OSError("disk-full")
        durable.update(patch)

    instance = adapter(handle)
    with pytest.raises(OSError):
        await instance.retrieve(REQUEST, active, save, 10)
    assert durable["state"] == "submitting" and "snapshot_id" not in durable
    result = await instance.retrieve(REQUEST, durable, checkpoint(durable), 10)
    assert result.error.code == "RECONCILIATION_REQUIRED" and calls == ["POST"]


async def test_429_cooldown_persists_until_resume_then_retries_only_rejection():
    clock = Clock()
    job = {"state": "new"}
    calls = []

    def handle(request):
        calls.append(request.method)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "20"})
        return httpx.Response(200, json=acknowledgment() if request.method == "POST" else completed())

    instance = adapter(handle, clock)
    assert (await instance.retrieve(REQUEST, job, checkpoint(job), 5)).status == "pending"
    assert calls == ["POST"] and job["state"] == "new" and job["firecrawl_retry_not_before"] == 1020
    assert (await instance.retrieve(REQUEST, job, checkpoint(job), 5)).status == "pending"
    assert calls == ["POST"]
    clock.now += 20
    assert (await instance.retrieve(REQUEST, job, checkpoint(job), 5)).status == "ready"
    assert calls == ["POST", "POST", "GET"]


async def test_poll_transport_failure_remains_resumable_without_post():
    calls = []
    clock = Clock()
    job = {"state": "new"}

    def handle(request):
        calls.append(request.method)
        if request.method == "POST":
            return httpx.Response(200, json=acknowledgment())
        raise httpx.ReadTimeout("private", request=request)

    instance = adapter(handle, clock, max_attempts=2)
    assert (await instance.retrieve(REQUEST, job, checkpoint(job), 10)).status == "pending"
    assert job["snapshot_id"] == ID
    instance.client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=completed()))
    )
    assert (await instance.retrieve(REQUEST, job, checkpoint(job), 10)).status == "ready"
    assert calls.count("POST") == 1


@pytest.mark.parametrize(
    "change,code",
    [
        ({"url": None}, "FETCH_DESTINATION_UNVERIFIABLE"),
        ({"sourceURL": "https://other.test/"}, "FETCH_RECORD_MAPPING"),
        ({"url": "http://127.0.0.1/private"}, "EGRESS_DENIED"),
        ({"url": "https://outside.test/"}, "EGRESS_DENIED"),
        ({"statusCode": 404}, "FETCH_RECORD_ERROR"),
        ({"statusCode": True}, "FETCH_RECORD_ERROR"),
        ({"error": "private error text"}, "FETCH_RECORD_ERROR"),
    ],
)
async def test_invalid_or_disallowed_destination_and_per_page_errors_are_terminal(change, code):
    doc = document()
    doc["metadata"].update(change)
    calls = []
    job = {"state": "new"}

    def handle(request):
        calls.append(request.method)
        return httpx.Response(200, json=acknowledgment() if request.method == "POST" else completed([doc]))

    instance = adapter(handle)
    result = await instance.retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.status == "failed" and result.error.code == code
    assert "private error text" not in result.error.model_dump_json()
    await instance.retrieve(REQUEST, job, checkpoint(job), 10)
    assert calls == ["POST", "GET"]


async def test_optional_redirect_chain_is_checked():
    doc = document(redirect_chain=[URL, "https://127.0.0.1/private"])

    def handle(request):
        return httpx.Response(200, json=acknowledgment() if request.method == "POST" else completed([doc]))

    job = {"state": "new"}
    result = await adapter(handle).retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.error.code == "EGRESS_DENIED"


async def test_completed_empty_batch_fetches_bounded_error_records_without_logging_messages():
    calls = []
    job = {"state": "new"}

    def handle(request):
        calls.append(request.url.path)
        if request.method == "POST":
            return httpx.Response(200, json=acknowledgment())
        if request.url.path.endswith("/errors"):
            return httpx.Response(
                200,
                json={
                    "errors": [
                        {"id": ID, "url": URL, "code": "SCRAPE_FAILED", "error": "private vendor details"}
                    ],
                    "robotsBlocked": [],
                },
            )
        return httpx.Response(200, json=completed([], completed=0, total=0))

    result = await adapter(handle).retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.status == "failed" and result.error.code == "FETCH_RECORD_ERROR"
    assert job["firecrawl_error_counts"]["matching_codes"] == ["SCRAPE_FAILED"]
    assert "private vendor details" not in json.dumps(job)
    assert calls[-1].endswith(f"/{ID}/errors")


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_terminal_job_failure_has_no_automatic_retry(status):
    job = {"state": "new"}

    def handle(request):
        if request.method == "POST":
            body = acknowledgment()
        elif request.url.path.endswith("/errors"):
            body = {"errors": [], "robotsBlocked": []}
        else:
            body = {"success": False, "status": status}
        return httpx.Response(200, json=body)

    result = await adapter(handle).retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.status == "failed" and result.error.code == "FETCH_JOB_FAILED"


@pytest.mark.parametrize(
    "target_request",
    [
        ExtractionRequest(url=URL, country="US"),
        ExtractionRequest(url=URL, browser_actions=[BrowserAction(kind="click", selector=".next")]),
        ExtractionRequest(url="https://127.0.0.1/private"),
    ],
)
async def test_unsupported_inputs_are_rejected_before_any_post(target_request):
    def handle(_):
        pytest.fail("No network request is allowed")

    with pytest.raises(ScraperError):
        await adapter(handle).retrieve(target_request, {"state": "new"}, lambda _: None, 10)


async def test_zero_wait_and_invalid_reconciled_id_never_submit():
    def handle(_):
        pytest.fail("No network request is allowed")

    instance = adapter(handle)
    job = {"state": "new"}
    assert (await instance.retrieve(REQUEST, job, checkpoint(job), 0)).status == "pending"
    job.update(state="pending", snapshot_id="../../forbidden")
    with pytest.raises(ScraperError, match="invalid batch"):
        await instance.retrieve(REQUEST, job, checkpoint(job), 10)


async def test_configuration_identity_refuses_wrong_runtime_on_resumption():
    job = {"state": "new"}
    instance = adapter(lambda _: httpx.Response(200, json=acknowledgment()))
    await instance.retrieve(REQUEST, job, checkpoint(job), 0)
    changed = adapter(lambda _: pytest.fail("No network"))
    changed.binding = "different-runtime-binding"
    with pytest.raises(ScraperError, match="original Firecrawl configuration"):
        await changed.retrieve(REQUEST, job, checkpoint(job), 10)


async def test_pipeline_end_to_end_saves_raw_source_and_requires_no_vendor_budget(tmp_path):
    calls = []

    def handle(request):
        calls.append(request.method)
        return httpx.Response(200, json=acknowledgment() if request.method == "POST" else completed())

    instance = Pipeline(
        settings(tmp_path),
        retriever=adapter(handle),
        policy=URLPolicy(["firecrawl.example.test"], resolve_dns=False),
    )
    result = await instance.extract_url(REQUEST)
    assert result.status in {"success", "partial"} and result.data["title"] == "Authored page"
    assert result.provenance.retrieval_method == "firecrawl_self_hosted"
    assert result.usage.cost_estimate_usd == instance.store.reserved_spend() == 0
    assert "Self-hosted Firecrawl" in result.usage.pricing_basis
    saved = SourceSnapshot.model_validate_json(Path(result.provenance.snapshot_path).read_text())
    assert saved.raw_record["page_html"] == RAW and saved.raw_record["firecrawl_html"] != RAW
    assert SERVICE_WARNING in result.warnings
    assert (await instance.resume(result.job_id)).status in {"success", "partial"}
    assert calls == ["POST", "GET"]


async def test_factory_and_cache_identity_are_explicit_and_do_not_expose_legacy_key(tmp_path):
    config = settings(tmp_path, firecrawl_api_key="legacy-private-key")
    instance = create_pipeline(config)
    assert isinstance(instance.retriever, FirecrawlRetriever)
    assert "legacy-private-key" in instance.secrets
    one = instance.store.get_or_create(
        REQUEST, {**instance.retriever.cache_options, "method": instance.retriever.retrieval_method}
    )
    other = FirecrawlRetriever(
        instance.policy, base_url=ENDPOINT, runtime_id="f" * 40, trust_service_egress=True
    )
    two = instance.store.get_or_create(REQUEST, {**other.cache_options, "method": other.retrieval_method})
    assert one["id"] != two["id"] and "legacy-private-key" not in json.dumps(instance.retriever.cache_options)
    await other.close()
    await instance.close()


def test_environment_and_deploy_status_do_not_require_keys_or_prices(tmp_path, monkeypatch):
    env = {
        "SCRAPER_RETRIEVAL_BACKEND": "firecrawl",
        "FIRECRAWL_BASE_URL": ENDPOINT,
        "FIRECRAWL_RUNTIME_ID": RUNTIME,
        "FIRECRAWL_TRUST_SERVICE_EGRESS": "true",
        "SCRAPER_ALLOWED_DOMAINS": "firecrawl.example.test",
    }
    monkeypatch.setattr(os, "environ", env)
    assert Settings.from_env().firecrawl_trust_service_egress is True
    spec = importlib.util.spec_from_file_location(
        "deployment_firecrawl", Path(__file__).parents[1] / "deploy/launch.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(env))
    path.chmod(0o600)
    report = module.status(module.load_environment(tmp_path, path, {}))
    assert report["live_retrieval_configured"] and not report["live_verified"]
    assert not report["configured"]["FIRECRAWL_API_KEY"]
    env["FIRECRAWL_TRUST_SERVICE_EGRESS"] = "invalid"
    with pytest.raises(ValueError):
        Settings.from_env()


async def test_crawl_configuration_binds_service_profile_without_changing_other_backends(tmp_path):
    config = settings(tmp_path)
    crawler = Crawler(config, policy=URLPolicy(["firecrawl.example.test"], resolve_dns=False))
    request = CrawlRequest(seeds=[URL], wait_seconds=0)
    result = await crawler.start(request)
    bound = crawler._configuration()
    assert bound["firecrawl_retrieval"]["runtime_id"] == RUNTIME
    assert bound["firecrawl_retrieval"]["firecrawl_policy_version"] == FIRECRAWL_POLICY_VERSION
    config2 = config.model_copy(update={"firecrawl_base_url": "http://127.0.0.1:3003"})
    changed = Crawler(config2, policy=URLPolicy(["firecrawl.example.test"], resolve_dns=False))
    with pytest.raises(ScraperError, match="original extraction configuration"):
        await changed.resume(result.crawl_id)
    legacy = Crawler(config.model_copy(update={"retrieval_backend": "direct"}))
    assert "firecrawl_retrieval" not in legacy._configuration()


@pytest.mark.parametrize("phase", ["POST", "GET"])
async def test_body_limit_preserves_unknown_submission_and_bounds_completed_source(phase):
    calls = []
    job = {"state": "new"}

    def handle(request):
        calls.append(request.method)
        if request.method == "POST" and phase == "GET":
            return httpx.Response(200, json=acknowledgment())
        return httpx.Response(200, headers={"Content-Length": "10000"}, content=b"x" * 10000)

    result = await adapter(handle, max_response_bytes=1024).retrieve(REQUEST, job, checkpoint(job), 10)
    assert len(calls) == (1 if phase == "POST" else 2)
    assert result.status == ("pending" if phase == "POST" else "failed")
    assert result.error.code == ("RECONCILIATION_REQUIRED" if phase == "POST" else "CONTENT_TOO_LARGE")


@pytest.mark.parametrize("phase", ["POST", "GET"])
async def test_api_redirect_is_never_followed_even_with_redirecting_injected_client(phase):
    calls = []
    job = {"state": "new"}

    def handle(request):
        calls.append(request.method)
        assert request.url.host == "127.0.0.1"
        if request.method == "POST" and phase == "GET":
            return httpx.Response(200, json=acknowledgment())
        return httpx.Response(302, headers={"Location": "https://evil.invalid/steal"})

    result = await adapter(handle).retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.status == "failed" and result.error.code == "FETCH_HTTP_ERROR"
    assert len(calls) == (1 if phase == "POST" else 2)


@pytest.mark.parametrize(
    "body,code",
    [
        (completed([document(), document()]), "FETCH_INVALID_RESPONSE"),
        (completed(next="https://evil.invalid/page2"), "FETCH_RECORD_MAPPING"),
        (completed(completed=True), "FETCH_INVALID_RESPONSE"),
        (
            {"success": True, "status": "unknown", "completed": 0, "total": 0, "data": []},
            "FETCH_INVALID_RESPONSE",
        ),
        (
            completed([{"metadata": {"sourceURL": URL, "url": URL, "statusCode": 200}}]),
            "FETCH_INVALID_RESPONSE",
        ),
    ],
)
async def test_malformed_poll_envelope_does_not_yield_a_success_or_new_submission(body, code):
    calls = []
    job = {"state": "new"}

    def handle(request):
        calls.append(request.method)
        return httpx.Response(200, json=acknowledgment() if request.method == "POST" else body)

    instance = adapter(handle)
    result = await instance.retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.status == "failed" and result.error.code == code
    await instance.retrieve(REQUEST, job, checkpoint(job), 10)
    assert calls == ["POST", "GET"]


async def test_pipeline_restart_retains_batch_id_and_reuses_retrieval_across_schemas(tmp_path):
    clock = Clock()
    calls = []

    def handle(request):
        calls.append(request.method)
        return httpx.Response(200, json=acknowledgment() if request.method == "POST" else completed())

    first = Pipeline(
        settings(tmp_path),
        retriever=adapter(handle, clock),
        policy=URLPolicy(["firecrawl.example.test"], resolve_dns=False),
    )
    original_update = first.store.update

    def update(job_id, **patch):
        saved = original_update(job_id, **patch)
        if patch.get("snapshot_id"):
            clock.now += 1000
        return saved

    first.store.update = update
    pending = await first.extract_url(REQUEST)
    assert pending.status == "pending"
    reopened = Pipeline(
        settings(tmp_path),
        retriever=adapter(handle, clock),
        policy=URLPolicy(["firecrawl.example.test"], resolve_dns=False),
    )
    request2 = REQUEST.model_copy(update={"schema_id": "article_v1"})
    child = await reopened.prepare_job(request2)
    assert reopened.store.get(child)["retrieval_parent_id"] == pending.job_id
    result = await reopened.resume(child)
    assert result.status in {"success", "partial"} and calls.count("POST") == 1
    assert result.provenance.snapshot_id == ID
    assert reopened.store.get(child)["record"]["firecrawl_service"]["runtime_id"] == RUNTIME


async def test_completed_pipeline_resume_rejects_endpoint_drift_before_using_saved_source(tmp_path):
    def handle(request):
        return httpx.Response(200, json=acknowledgment() if request.method == "POST" else completed())

    instance = Pipeline(
        settings(tmp_path),
        retriever=adapter(handle),
        policy=URLPolicy(["firecrawl.example.test"], resolve_dns=False),
    )
    result = await instance.extract_url(REQUEST)
    changed = adapter(lambda _: pytest.fail("No HTTP is allowed"))
    changed.cache_options["endpoint_sha256"] = "f" * 64
    from ai_scraper.jobs import digest

    changed.binding = digest(changed.cache_options)
    resumed = Pipeline(
        settings(tmp_path), retriever=changed, policy=URLPolicy(["firecrawl.example.test"], resolve_dns=False)
    )
    rejected = await resumed.resume(result.job_id)
    assert rejected.error.code == "FIRECRAWL_CONFIG_CHANGED"


async def test_invalid_reconciled_uuid_is_rejected_before_state_is_mutated(tmp_path):
    instance = Pipeline(
        settings(tmp_path),
        retriever=adapter(lambda _: pytest.fail("No HTTP")),
        policy=URLPolicy(["firecrawl.example.test"], resolve_dns=False),
    )
    job = await instance.prepare_job(REQUEST)
    instance.store.update(job, state="reconciliation_required")
    with pytest.raises(ScraperError, match="invalid batch"):
        await instance.resume(job, snapshot_id="not-a-uuid")
    assert instance.store.get(job).get("snapshot_id") is None
