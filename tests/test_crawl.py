from __future__ import annotations

import asyncio
import json
from collections import Counter

import httpx
import pytest
from pydantic import ValidationError

from ai_scraper.config import Settings
from ai_scraper.contracts import ScraperError
from ai_scraper.crawl import (
    Crawler,
    CrawlRequest,
    CrawlStore,
    canonical_url,
    discover_links,
    robots_rules,
)
from ai_scraper.jobs import JobStore
from ai_scraper.pipeline import Pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.brightdata import BrightDataRetriever
from ai_scraper.retrieval.direct import DirectHTTPRetriever
from ai_scraper.schemas import AppError, ExtractionResult, Provenance

ROOT = "https://crawl.example.test/"


class Clock:
    def __init__(self):
        self.now = 100000.0

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.now += seconds
        await asyncio.sleep(0)


class FakePipeline:
    def __init__(self, settings, pages, events, *, pending=None, fail=None, blocker=None):
        self.settings, self.pages, self.events = settings, pages, events
        self.store = JobStore(settings.state_dir)
        self.pending, self.fail, self.blocker = pending or set(), fail or {}, blocker

    async def prepare_job(self, request):
        self.events.append(("prepare", request.url))
        return self.store.get_or_create(request, {"method": "fixture_http"})["id"]

    async def resume(self, job_id):
        job = self.store.get(job_id)
        url = job["request"]["url"]
        self.events.append(("resume", url))
        if self.blocker is not None:
            await self.blocker(url)
        if job["state"] == "new":
            self.events.append(("submit", url))
            if url in self.pending:
                self.store.update(job_id, state="pending", snapshot_id="vendor-fixture-123")
                return ExtractionResult(
                    job_id=job_id, status="pending", provenance=Provenance(requested_url=url)
                )
        if job["state"] == "pending" and url in self.pending:
            self.pending.remove(url)
        record = self.pages.get(url, {"page_html": "<main><p>Literal fixture text.</p></main>"})
        record = {"final_url": url, **record}
        status = "failed" if url in self.fail else "success"
        error = (
            AppError(code=self.fail[url], message="Fixture extraction failed") if url in self.fail else None
        )
        result = ExtractionResult(
            job_id=job_id,
            status=status,
            data={"text": ["Literal fixture text."]} if not error else None,
            provenance=Provenance(requested_url=url, final_url=record["final_url"]),
            error=error,
        )
        self.store.update(
            job_id,
            state="complete" if not error else "failed",
            record=record,
            retrieval_method="fixture_http",
        )
        self.store.save_result(result)
        return result

    async def close(self):
        self.events.append(("close", ""))


def setup(tmp_path, *, pages=None, handler=None, pending=None, fail=None, blocker=None, allowed=None):
    clock, events, scoped = Clock(), [], []
    settings = Settings(
        state_dir=tmp_path / "state",
        retrieval_backend="direct",
        allowed_domains=["*"] if allowed is None else allowed,
    )
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler or (lambda request: httpx.Response(404, request=request)))
    )

    def factory(value):
        scoped.append(value.allowed_domains)
        return FakePipeline(value, pages or {}, events, pending=pending, fail=fail, blocker=blocker)

    crawler = Crawler(
        settings,
        pipeline_factory=factory,
        robots_client=client,
        policy=URLPolicy(settings.allowed_domains, resolve_dns=False),
        clock=clock,
        sleep=clock.sleep,
    )
    return crawler, clock, events, client, scoped


def test_canonical_identity_preserves_query_semantics():
    assert canonical_url("HTTPS://Crawl.Example.Test:443/x?a=1&a=2#heading") == ROOT + "x?a=1&a=2"
    assert canonical_url(ROOT + "?a=1&a=2") != canonical_url(ROOT + "?a=2&a=1")
    assert canonical_url(ROOT + "a/../b") == ROOT + "a/../b"
    with pytest.raises(ValueError):
        canonical_url("https://user:password@example.com/")


@pytest.mark.parametrize(
    "change",
    [
        {"max_pages": 0},
        {"max_depth": 11},
        {"max_frontier": 0},
        {"scope_domains": ["*"]},
        {"seeds": [ROOT], "scope_domains": ["other.test"]},
    ],
)
def test_crawl_bounds_and_exact_scope_are_strict(change):
    with pytest.raises(ValidationError):
        CrawlRequest(**{"seeds": [ROOT], **change})


def test_discovery_resolves_observed_url_and_safe_base_with_limits():
    html = '<base href="/docs/"><a href="a#one">A</a><a href="a#two">A</a><a href="?x=1&x=2">Q</a>'
    html += '<a href="https://outside.example/">Outside</a><a href="javascript:alert(1)">No</a>'
    links, warnings = discover_links(
        {"page_html": html}, ROOT + "redirected", {"crawl.example.test"}, limit=10, max_chars=10000
    )
    assert links == [ROOT + "docs/a", ROOT + "docs/?x=1&x=2"]
    assert warnings == []
    links, warnings = discover_links(
        {"page_html": '<base href="https://outside.example/"><a href="b">B</a>'},
        ROOT + "nested/page",
        {"crawl.example.test"},
        limit=1,
        max_chars=10000,
    )
    assert links == [ROOT + "nested/b"]
    assert warnings == ["out_of_scope_base_ignored"]


def test_robots_longest_allow_wildcards_specific_agents_and_delays():
    text = "User-agent: *\nDisallow: /private*\nAllow: /private/public$\nCrawl-delay: 2\n"
    assert robots_rules(text, "EvidenceAIScraper/0.5", ROOT + "private/public") == (True, 2.0)
    assert robots_rules(text, "EvidenceAIScraper/0.5", ROOT + "private/public/child") == (False, 2.0)
    text += "\nUser-agent: EvidenceAIScraper\nDisallow: /specific\n"
    assert robots_rules(text, "EvidenceAIScraper/0.5", ROOT + "private/a") == (True, 0.0)
    assert robots_rules(text, "EvidenceAIScraper/0.5", ROOT + "specific") == (False, 0.0)


async def test_bfs_cycles_fragments_query_order_and_scoped_transport(tmp_path):
    pages = {
        ROOT: {
            "page_html": '<a href="/a#x">A</a><a href="/a#y">A</a><a href="/?x=1&x=2">Q1</a>'
            '<a href="/?x=2&x=1">Q2</a><a href="https://other.test/x">Outside</a>'
        },
        ROOT + "a": {"page_html": '<a href="/">Cycle</a><a href="/deep">Deep</a>'},
    }
    crawler, _, events, client, scoped = setup(tmp_path, pages=pages)
    result = await crawler.start(CrawlRequest(seeds=[ROOT], max_pages=10, max_depth=1, delay_seconds=0))
    assert result.status == "success"
    assert result.counts["accepted"] == 4
    assert [url for event, url in events if event == "submit"] == [
        ROOT,
        ROOT + "a",
        ROOT + "?x=1&x=2",
        ROOT + "?x=2&x=1",
    ]
    assert scoped == [["crawl.example.test"]]
    assert all(entry.job_id and entry.result_path for entry in result.entries)
    assert all(entry.depth <= 1 for entry in result.entries)
    await client.aclose()


async def test_robots_denial_occurs_after_job_binding_before_retrieval(tmp_path):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, text="User-agent: *\nDisallow: /\n", request=request)

    crawler, _, events, client, _ = setup(tmp_path, handler=handler)
    result = await crawler.start(CrawlRequest(seeds=[ROOT], delay_seconds=0))
    assert result.status == "failed"
    assert result.counts["skipped"] == 1
    assert result.entries[0].error.code == "ROBOTS_DENIED"
    assert result.entries[0].job_id is not None
    assert not any(event == "resume" for event, _ in events)
    assert seen[0].headers["user-agent"] == crawler.settings.user_agent
    await client.aclose()


async def test_unavailable_robots_remain_pending_and_retry_without_duplicate_job(tmp_path):
    calls = 0

    def handler(request):
        nonlocal calls
        calls += 1
        return (
            httpx.Response(503, headers={"Retry-After": "5"}, request=request)
            if calls == 1
            else httpx.Response(404, request=request)
        )

    crawler, clock, events, client, _ = setup(tmp_path, handler=handler)
    first = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=2, delay_seconds=0))
    assert first.status == "pending"
    job_id = first.entries[0].job_id
    assert not any(event == "submit" for event, _ in events)
    clock.now += 5
    second = await crawler.resume(first.crawl_id, wait_seconds=5)
    assert second.status == "success"
    assert second.entries[0].job_id == job_id
    assert calls == 2
    assert Counter(event for event, _ in events)["prepare"] == 1
    await client.aclose()


async def test_vendor_pending_resume_uses_bound_job_and_never_resubmits(tmp_path):
    pending = {ROOT}
    crawler, _, events, client, _ = setup(tmp_path, pending=pending)
    first = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=0, delay_seconds=0))
    assert first.status == "pending" and first.entries[0].job_id is None
    result = await crawler.resume(first.crawl_id, wait_seconds=5)
    assert result.status == "success"
    assert Counter(event for event, _ in events)["submit"] == 1
    assert Counter(event for event, _ in events)["resume"] == 2
    assert Counter(event for event, _ in events)["prepare"] == 1
    await client.aclose()


@pytest.mark.parametrize("crash_at", ["bind", "resume"])
async def test_crash_before_and_after_binding_recovers_without_new_work(tmp_path, monkeypatch, crash_at):
    crawler, _, events, client, _ = setup(tmp_path)
    first = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=0, delay_seconds=0))
    if crash_at == "bind":
        original = crawler.store.bind_job

        def crash(*args, **kwargs):
            raise RuntimeError("simulated process death before frontier bind")

        monkeypatch.setattr(crawler.store, "bind_job", crash)
    else:
        original_factory = crawler.pipeline_factory

        def factory(settings):
            instance = original_factory(settings)

            async def crash(job_id):
                raise RuntimeError("simulated process death after frontier bind")

            instance.resume = crash
            return instance

        monkeypatch.setattr(crawler, "pipeline_factory", factory)
    with pytest.raises(RuntimeError):
        await crawler.resume(first.crawl_id, wait_seconds=5)
    if crash_at == "bind":
        monkeypatch.setattr(crawler.store, "bind_job", original)
    else:
        monkeypatch.setattr(crawler, "pipeline_factory", original_factory)
    result = await crawler.resume(first.crawl_id, wait_seconds=5)
    assert result.status == "success"
    with crawler.jobs.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1
    assert Counter(event for event, _ in events)["submit"] == 1
    await client.aclose()


async def test_discovery_survives_semantic_failure_but_skips_challenges(tmp_path):
    pages = {
        ROOT: {"page_html": '<a href="/a">A</a>'},
        ROOT + "a": {"page_html": '<a href="/blocked">Blocked</a>'},
    }
    crawler, _, events, client, _ = setup(
        tmp_path, pages=pages, fail={ROOT: "INVALID_EXTRACTION", ROOT + "a": "CONTENT_CHALLENGE"}
    )
    result = await crawler.start(CrawlRequest(seeds=[ROOT], delay_seconds=0))
    assert result.counts["failed"] == 2
    assert [url for event, url in events if event == "submit"] == [ROOT, ROOT + "a"]
    await client.aclose()


async def test_page_frontier_and_discovery_limits_are_explicit(tmp_path):
    pages = {ROOT: {"page_html": "".join(f'<a href="/{n}">Next</a>' for n in range(10))}}
    crawler, _, events, client, _ = setup(tmp_path, pages=pages)
    result = await crawler.start(
        CrawlRequest(seeds=[ROOT], max_pages=2, max_frontier=3, max_links_per_page=4, delay_seconds=0)
    )
    assert result.status == "partial"
    assert result.counts["discovered"] == 3
    assert result.counts["processed"] == 2
    assert result.counts["limited"] == 1
    assert set(result.warnings) == {"frontier_limit_reached", "link_limit_reached", "page_limit_reached"}
    assert Counter(event for event, _ in events)["submit"] == 2
    assert (await crawler.resume(result.crawl_id)).counts == result.counts
    await client.aclose()


async def test_host_pacing_and_robots_delay_are_persisted(tmp_path):
    handler = lambda request: httpx.Response(200, text="User-agent: *\nCrawl-delay: 3\n", request=request)
    crawler, clock, events, client, _ = setup(
        tmp_path, pages={ROOT: {"page_html": '<a href="/a">A</a>'}}, handler=handler
    )
    first = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=1, delay_seconds=0))
    assert first.status == "pending" and first.counts["accepted"] == 1
    assert Counter(event for event, _ in events)["submit"] == 1
    second = await crawler.resume(first.crawl_id, wait_seconds=5)
    assert second.status == "success" and clock() >= 100003
    await client.aclose()


async def test_two_controllers_share_one_crawl_lease(tmp_path):
    entered, release = asyncio.Event(), asyncio.Event()

    async def blocker(url):
        entered.set()
        await release.wait()

    crawler, _, events, client, _ = setup(tmp_path, blocker=blocker)
    first = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=0, delay_seconds=0))
    running = asyncio.create_task(crawler.resume(first.crawl_id, wait_seconds=5))
    await entered.wait()
    concurrent = await crawler.resume(first.crawl_id, wait_seconds=5)
    assert concurrent.status == "pending"
    assert Counter(event for event, _ in events)["resume"] == 1
    release.set()
    assert (await running).status == "success"
    await client.aclose()


async def test_explicit_global_scope_cannot_be_broadened_and_private_seed_never_runs(tmp_path):
    crawler, _, events, client, _ = setup(tmp_path, allowed=["allowed.example.test"])
    with pytest.raises(ScraperError, match="allowlist"):
        await crawler.start(CrawlRequest(seeds=[ROOT]))
    with pytest.raises(ScraperError):
        await crawler.start(CrawlRequest(seeds=["http://127.0.0.1/"]))
    assert events == []
    await client.aclose()


async def test_observed_off_scope_destination_is_rejected_even_at_depth_zero(tmp_path):
    crawler, _, _, client, _ = setup(
        tmp_path, pages={ROOT: {"final_url": "https://outside.example/", "page_html": "<p>Other</p>"}}
    )
    result = await crawler.start(CrawlRequest(seeds=[ROOT], max_depth=0, delay_seconds=0))
    assert result.status == "failed"
    assert result.entries[0].error.code == "EGRESS_DENIED"
    await client.aclose()


async def test_configuration_change_rejected_without_credentials_in_saved_configuration(tmp_path):
    crawler, _, events, client, _ = setup(tmp_path)
    first = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=0))
    run = crawler.store.get(first.crawl_id)
    assert not {"hf_token", "brightdata_api_key", "local_api_key"} & run["configuration"].keys()
    crawler.settings = crawler.settings.model_copy(update={"max_output_tokens": 128})
    with pytest.raises(ScraperError, match="configuration"):
        await crawler.resume(first.crawl_id)
    assert events == []
    await client.aclose()


def test_invalid_and_unknown_crawl_ids_are_bounded(tmp_path):
    store = CrawlStore(JobStore(tmp_path))
    for value in ("../private", "a" * 32):
        with pytest.raises(ScraperError, match="crawl ID"):
            store.get(value)


async def test_robots_redirect_cannot_leave_scope_and_credentials_are_stripped(tmp_path):
    requests = []

    def handler(request):
        requests.append(request)
        assert "authorization" not in request.headers and "cookie" not in request.headers
        return httpx.Response(302, headers={"Location": "http://127.0.0.1/private"}, request=request)

    crawler, _, events, client, _ = setup(tmp_path, handler=handler)
    client.headers.update({"Authorization": "Bearer fixture-only", "Cookie": "fixture=yes"})
    result = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=1))
    assert result.status == "pending"
    assert len(requests) == 1 and not any(event == "submit" for event, _ in events)
    await client.aclose()


async def test_empty_allowlist_can_resume_saved_seed_scope_safely(tmp_path):
    crawler, _, _, client, _ = setup(tmp_path, allowed=[])
    first = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=0))
    result = await crawler.resume(first.crawl_id, wait_seconds=5)
    assert result.status == "success"
    await client.aclose()


async def test_expired_controller_lease_recovers_bound_active_entry(tmp_path):
    crawler, clock, events, client, _ = setup(tmp_path)
    first = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=0, delay_seconds=0))
    assert crawler.store.acquire(first.crawl_id, "dead-controller", 1)
    entry, _ = crawler.store.pick(first.crawl_id, "dead-controller", 1)
    job_id = crawler.jobs.get_or_create(
        CrawlRequest(seeds=[ROOT]).extraction(ROOT), {"method": "fixture_http"}
    )["id"]
    crawler.store.bind_job(first.crawl_id, "dead-controller", entry["url"], job_id)
    clock.now += 2
    recovered = await crawler.resume(first.crawl_id, wait_seconds=5)
    assert recovered.status == "success" and recovered.entries[0].job_id == job_id
    assert Counter(event for event, _ in events)["prepare"] == 0
    assert Counter(event for event, _ in events)["submit"] == 1
    await client.aclose()


async def test_crash_after_atomic_page_completion_does_not_lose_children(tmp_path, monkeypatch):
    crawler, _, events, client, _ = setup(tmp_path, pages={ROOT: {"page_html": '<a href="/child">Next</a>'}})
    first = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=0, delay_seconds=0))
    finish = crawler.store.finish

    def crash_after_commit(*args, **kwargs):
        finish(*args, **kwargs)
        raise RuntimeError("simulated process death after atomic completion")

    monkeypatch.setattr(crawler.store, "finish", crash_after_commit)
    with pytest.raises(RuntimeError):
        await crawler.resume(first.crawl_id, wait_seconds=5)
    monkeypatch.setattr(crawler.store, "finish", finish)
    recovered = await crawler.resume(first.crawl_id, wait_seconds=5)
    assert recovered.status == "success" and recovered.counts["accepted"] == 2
    assert [url for event, url in events if event == "submit"] == [ROOT, ROOT + "child"]
    await client.aclose()


@pytest.mark.parametrize(
    "response",
    [
        {"content": b"x" * 65537},
        {"content": b"\xff"},
        {"text": "User-agent: *\nCrawl-delay: nan\n"},
        {"text": "User-agent: *\nDisallow: /", "headers": {"Content-Encoding": "br"}},
    ],
)
async def test_invalid_or_unbounded_robots_fail_closed(tmp_path, response):
    crawler, _, events, client, _ = setup(
        tmp_path, handler=lambda request: httpx.Response(200, request=request, **response)
    )
    result = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=1, delay_seconds=0))
    assert result.status == "pending" and result.entries[0].error.code == "ROBOTS_UNAVAILABLE"
    assert not any(event == "resume" for event, _ in events)
    await client.aclose()


async def test_actual_pipeline_vendor_202_restart_never_repeats_post(tmp_path):
    crawler, clock, _, robots_client, _ = setup(tmp_path)
    crawler.settings = crawler.settings.model_copy(
        update={"retrieval_backend": "brightdata", "wait_seconds": 1.0}
    )
    calls, clients = [], []
    ready = False

    def vendor(request):
        calls.append(request.method)
        if request.method == "POST":
            clock.now += 2
            return httpx.Response(202, json={"snapshot_id": "existing-crawl-snapshot"})
        assert ready
        if "/progress/" in request.url.path:
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(
            200,
            json=[
                {
                    "input": {"url": ROOT},
                    "final_url": ROOT,
                    "page_html": "<main><h1>Saved source</h1><p>Literal fixture page for extraction.</p></main>",
                }
            ],
        )

    def factory(settings):
        client = httpx.AsyncClient(transport=httpx.MockTransport(vendor))
        clients.append(client)
        adapter = BrightDataRetriever("fixture-only", client=client, clock=clock, sleep=clock.sleep)
        return Pipeline(
            settings,
            retriever=adapter,
            policy=URLPolicy(settings.allowed_domains, resolve_dns=False),
            live=False,
        )

    crawler.pipeline_factory = factory
    first = await crawler.start(CrawlRequest(seeds=[ROOT], max_depth=0, wait_seconds=1, delay_seconds=0))
    assert first.status == "pending" and calls == ["POST"]
    job_id = first.entries[0].job_id
    assert crawler.jobs.get(job_id)["snapshot_id"] == "existing-crawl-snapshot"
    ready = True
    result = await crawler.resume(first.crawl_id, wait_seconds=5)
    assert result.status == "success", result.model_dump(mode="json")
    assert result.entries[0].job_id == job_id and calls == ["POST", "GET", "GET"]
    assert crawler.jobs.get(job_id)["snapshot_path"]
    for client in clients:
        await client.aclose()
    await robots_client.aclose()


async def test_actual_scoped_direct_pipeline_blocks_redirect_before_offhost_request(tmp_path):
    crawler, _, _, robots_client, _ = setup(tmp_path)
    requests, clients = [], []

    def target(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"Location": "https://outside.example/private"}, request=request)

    def factory(settings):
        policy = URLPolicy(settings.allowed_domains, resolve_dns=False)
        client = httpx.AsyncClient(transport=httpx.MockTransport(target))
        clients.append(client)
        return Pipeline(
            settings, retriever=DirectHTTPRetriever(policy, client=client), policy=policy, live=False
        )

    crawler.pipeline_factory = factory
    result = await crawler.start(CrawlRequest(seeds=[ROOT], delay_seconds=0))
    assert result.status == "failed" and result.entries[0].error.code == "EGRESS_DENIED"
    assert requests == [ROOT]
    for client in clients:
        await client.aclose()
    await robots_client.aclose()


def test_discovery_does_not_read_past_requested_character_bound():
    html = '<a href="/inside">Inside</a>' + " " * 20 + '<a href="/outside">Outside</a>'
    links, warnings = discover_links(
        {"page_html": html}, ROOT, {"crawl.example.test"}, limit=10, max_chars=48
    )
    assert links == [ROOT + "inside"]
    assert warnings == ["discovery_size_limit_reached"]


async def test_saved_state_is_parseable_without_opening_network_clients(tmp_path):
    crawler, _, _, client, _ = setup(tmp_path)
    result = await crawler.start(CrawlRequest(seeds=[ROOT], wait_seconds=0))
    status = Crawler(crawler.settings, policy=URLPolicy(["*"], resolve_dns=False)).status(result.crawl_id)
    assert status.status == "pending"
    assert json.loads(status.model_dump_json())["entries"][0]["url"] == ROOT
    await client.aclose()
