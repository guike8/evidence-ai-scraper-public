from __future__ import annotations

import asyncio
import copy
import json
import os

import httpx
import pytest

from ai_scraper.contracts import RetrievalOutcome, ScraperError
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.browser import (
    BrowserRetriever,
    DirectThenBrowserRetriever,
    _Bridge,
    _DenyProxy,
)
from ai_scraper.schemas import BrowserAction, ExtractionRequest

URL = "https://site.test/start"
HTML = "<html><head><title>Rendered page</title></head><body><h1>Rendered page</h1><p>Useful source content.</p></body></html>"


def policy(*hosts):
    return URLPolicy(list(hosts or ["site.test"]), resolve_dns=False)


def checkpoint(job):
    return lambda patch: job.update(copy.deepcopy(patch))


def record(url=URL):
    return {
        "input": {"url": URL},
        "final_url": url,
        "observed_final_url": True,
        "page_html": HTML,
        "rendered": True,
        "browser": {"blocked_requests": {}},
    }


class FakeRequest:
    def __init__(self, url=URL, method="GET", *, navigation=True, frame=None, redirected_from=None):
        self.url, self.method = url, method
        self.headers = {
            "accept": "text/html",
            "authorization": "secret",
            "cookie": "secret",
            "x-secret": "secret",
        }
        self.navigation = navigation
        self.frame = frame
        self.redirected_from = redirected_from

    def is_navigation_request(self):
        return self.navigation


class FakeRoute:
    def __init__(self, request):
        self.request = request
        self.fulfilled = None
        self.aborted = None

    async def fulfill(self, **kwargs):
        self.fulfilled = kwargs

    async def abort(self, reason):
        self.aborted = reason


def bridge(retriever):
    value = _Bridge(retriever, __import__("time").monotonic() + 5)
    value.page = type("Page", (), {"main_frame": object()})()
    return value


async def test_bridge_strips_credentials_and_stages_checked_navigation_redirect():
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.headers["accept-encoding"] == "identity"
        assert "authorization" not in request.headers
        assert "cookie" not in request.headers
        assert "x-secret" not in request.headers
        return httpx.Response(
            302, headers={"Location": "/actual", "Set-Cookie": "secret", "Content-Type": "text/html"}
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle), auth=("name", "password"), cookies={"private": "secret"}
    )
    retriever = BrowserRetriever(policy(), client=client)
    state = bridge(retriever)
    route = FakeRoute(FakeRequest(frame=state.page.main_frame))
    await state.route(route)
    assert state.error is None
    assert route.fulfilled["status"] == 200
    assert "location" not in route.fulfilled["headers"]
    assert state.pending_navigation == "https://site.test/actual"
    assert "set-cookie" not in route.fulfilled["headers"]
    assert state.redirect_chain == [{"url": "https://site.test/actual", "http_status": 302}]
    assert len(calls) == 1  # The owning controller follows the checked destination.
    assert not client.cookies
    await client.aclose()


@pytest.mark.parametrize(
    "url", ["http://127.0.0.1/secret", "http://10.0.0.1/", "https://other.test/", "file:///etc/passwd"]
)
async def test_every_resource_is_checked_before_native_connection(url):
    calls = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: calls.append(req) or httpx.Response(200))
    )
    state = bridge(BrowserRetriever(policy(), client=client))
    route = FakeRoute(FakeRequest(url, navigation=False))
    await state.route(route)
    assert not calls
    assert state.error.error.code in {"EGRESS_DENIED", "INVALID_REQUEST"}
    assert route.aborted == "blockedbyclient"
    await client.aclose()


async def test_explicit_resource_host_does_not_expand_navigation_scope():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"Content-Type": "application/javascript"}, text="true")
        )
    )
    state = bridge(BrowserRetriever(policy(), client=client, resource_domains=["cdn.test"]))
    asset = FakeRoute(FakeRequest("https://cdn.test/app.js", navigation=False))
    await state.route(asset)
    assert asset.fulfilled["body"] == b"true"
    navigation = FakeRoute(FakeRequest("https://cdn.test/", frame=state.page.main_frame))
    await state.route(navigation)
    assert navigation.aborted
    assert state.error.error.code == "EGRESS_DENIED"
    await client.aclose()


async def test_private_redirect_is_rejected_before_browser_receives_location():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(302, headers={"Location": "http://127.0.0.1/private"})
        )
    )
    state = bridge(BrowserRetriever(policy(), client=client))
    route = FakeRoute(FakeRequest(frame=state.page.main_frame))
    await state.route(route)
    assert route.fulfilled is None
    assert state.error.error.code == "EGRESS_DENIED"
    await client.aclose()


@pytest.mark.parametrize("navigation", [False, True])
async def test_post_is_never_forwarded(navigation):
    calls = []
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda req: calls.append(req) or httpx.Response(200))
    )
    state = bridge(BrowserRetriever(policy(), client=client))
    route = FakeRoute(FakeRequest(method="POST", navigation=navigation))
    await state.route(route)
    assert calls == [] and route.aborted
    assert state.blocked["unsafe_method"] == 1
    assert bool(state.error) == navigation
    await client.aclose()


async def test_websocket_is_closed_without_connecting():
    class WebSocket:
        closed = False

        async def close(self, **kwargs):
            self.closed = kwargs

    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail("No HTTP call expected")))
    state = bridge(BrowserRetriever(policy(), client=client))
    ws = WebSocket()
    await state.websocket(ws)
    assert ws.closed["code"] == 1008
    assert state.blocked["websocket"] == 1
    await client.aclose()


async def test_resource_request_and_total_byte_limits_are_enforced():
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"x" * 1000))
    )
    state = bridge(BrowserRetriever(policy(), client=client, max_requests=2, max_total_bytes=1500))
    first = FakeRoute(FakeRequest(navigation=False))
    second = FakeRoute(FakeRequest(navigation=False))
    await state.route(first)
    await state.route(second)
    assert first.fulfilled and not second.fulfilled
    assert state.bytes_received == 1000
    assert state.error.error.code == "CONTENT_TOO_LARGE"
    state = bridge(BrowserRetriever(policy(), client=client, max_requests=1))
    await state.route(FakeRoute(FakeRequest(navigation=False)))
    extra = FakeRoute(FakeRequest(navigation=False))
    await state.route(extra)
    assert extra.aborted and state.error.error.code == "BROWSER_REQUEST_LIMIT"
    await client.aclose()


async def test_unexpected_compression_rejected_without_consuming_body():
    consumed = []

    class Body(httpx.AsyncByteStream):
        async def __aiter__(self):
            consumed.append(True)
            yield b"never decompressed"

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=Body())
        )
    )
    state = bridge(BrowserRetriever(policy(), client=client))
    route = FakeRoute(FakeRequest(navigation=False))
    await state.route(route)
    assert not consumed and not route.fulfilled
    assert state.error.error.code == "FETCH_INVALID_RESPONSE"
    await client.aclose()


async def test_attempt_is_checkpointed_before_launch_and_saved_record_survives_restart():
    job = {"state": "new"}
    calls = []

    class Browser(BrowserRetriever):
        async def _render(self, request, budget):
            assert job["attempts"] == 1 and job["browser_contract"]
            calls.append(request.url)
            return record()

    browser = Browser(policy())
    result = await browser.retrieve(ExtractionRequest(url=URL), job, checkpoint(job), 5)
    assert result.status == "ready" and job["record"] == result.record
    reloaded = json.loads(json.dumps(job))
    second = await browser.retrieve(ExtractionRequest(url=URL), reloaded, checkpoint(reloaded), 5)
    assert second.record == result.record and calls == [URL]
    changed = ExtractionRequest(url=URL, browser_actions=[BrowserAction(kind="click", selector="button")])
    mismatch = await browser.retrieve(changed, reloaded, checkpoint(reloaded), 5)
    assert mismatch.error.code == "BROWSER_STATE_MISMATCH"
    await browser.close()


async def test_crashed_or_timed_out_browser_retries_are_bounded():
    seen = []

    class Browser(BrowserRetriever):
        async def _render(self, request, budget):
            seen.append(budget)
            raise ScraperError("BROWSER_TIMEOUT", "timeout", retryable=True)

    browser = Browser(policy())
    job = {"state": "new"}
    first = await browser.retrieve(ExtractionRequest(url=URL), job, checkpoint(job), 1)
    assert first.status == "pending" and job["attempts"] == 1
    second = await browser.retrieve(ExtractionRequest(url=URL), job, checkpoint(job), 1)
    assert second.status == "failed" and job["attempts"] == 2
    assert (await browser.retrieve(ExtractionRequest(url=URL), job, checkpoint(job), 1)).status == "failed"
    assert len(seen) == 2
    await browser.close()


async def test_no_time_budget_performs_no_launch_and_country_is_not_misrepresented():
    class Browser(BrowserRetriever):
        async def _render(self, *args):
            pytest.fail("No browser launch expected")

    browser = Browser(policy())
    job = {"state": "new"}
    assert (await browser.retrieve(ExtractionRequest(url=URL), job, checkpoint(job), 0)).status == "pending"
    assert "attempts" not in job
    result = await browser.retrieve(ExtractionRequest(url=URL, country="US"), job, checkpoint(job), 2)
    assert result.error.code == "BROWSER_COUNTRY_UNSUPPORTED"
    await browser.close()


class FakeLocator:
    def __init__(self, page, selector):
        self.page, self.selector = page, selector

    async def click(self, **kwargs):
        self.page.actions.append(("click", self.selector))

    async def wait_for(self, **kwargs):
        self.page.actions.append(("wait_for", self.selector))


class FakePage:
    def __init__(self, context):
        self.context = context
        self.main_frame = object()
        self.url = URL
        self.actions = []
        self.mouse = self

    def on(self, *args):
        pass

    async def goto(self, url, **kwargs):
        self.url = url
        route = FakeRoute(FakeRequest(url, frame=self.main_frame))
        await self.context.handler(route)
        if route.aborted:
            raise ValueError("URL contents deliberately not echoed")

    def locator(self, selector):
        return FakeLocator(self, selector)

    async def wheel(self, x, y):
        self.actions.append(("scroll", y))

    async def wait_for_load_state(self, *args, **kwargs):
        pass

    async def content(self):
        return HTML

    async def title(self):
        return "Rendered page"


class FakeContext:
    def __init__(self):
        self.page = FakePage(self)
        self.closed = False

    def set_default_timeout(self, value):
        self.timeout = value

    async def route(self, pattern, handler):
        self.handler = handler

    async def route_web_socket(self, pattern, handler):
        self.ws_handler = handler

    async def new_page(self):
        return self.page

    async def close(self):
        self.closed = True


class FakeBrowser:
    version = "fixture-browser"

    def __init__(self):
        self.context = FakeContext()
        self.closed = False

    async def new_context(self, **kwargs):
        self.context_options = kwargs
        return self.context

    async def close(self):
        self.closed = True

    def is_connected(self):
        return not self.closed


class FakeManager:
    def __init__(self, browser):
        self.browser = browser
        self.chromium = self
        self.stopped = False

    async def start(self):
        return self

    async def launch(self, **kwargs):
        self.launch_options = kwargs
        return self.browser

    async def stop(self):
        self.stopped = True


async def test_ephemeral_sandboxed_browser_executes_only_typed_css_actions_and_cleans_up(monkeypatch):
    async def proxy_start(self):
        return "http://127.0.0.1:9"

    monkeypatch.setattr(_DenyProxy, "start", proxy_start)
    browser = FakeBrowser()
    manager = FakeManager(browser)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, headers={"Content-Type": "text/html"}, text=HTML)
        )
    )
    adapter = BrowserRetriever(policy(), client=client, playwright_factory=lambda: manager)
    request = ExtractionRequest(
        url=URL,
        browser_actions=[
            BrowserAction(kind="click", selector="button.next"),
            BrowserAction(kind="wait_for", selector=".loaded"),
            BrowserAction(kind="scroll", pixels=800),
        ],
    )
    result = await adapter.retrieve(request, {"state": "new"}, lambda _: None, 5)
    assert result.status == "ready" and result.record["page_html"] == HTML
    assert manager.launch_options["chromium_sandbox"] is True
    assert "--disable-quic" in manager.launch_options["args"]
    assert "--force-webrtc-ip-handling-policy=disable_non_proxied_udp" in manager.launch_options["args"]
    assert manager.launch_options["proxy"] == {"server": "http://127.0.0.1:9"}
    assert browser.context_options == {
        "service_workers": "block",
        "accept_downloads": False,
        "ignore_https_errors": False,
        "permissions": [],
        "user_agent": "EvidenceAIScraper/0.5",
    }
    assert browser.context.page.actions == [
        ("click", "css=button.next"),
        ("wait_for", "css=.loaded"),
        ("scroll", 800),
    ]
    assert browser.context.closed and browser.closed and manager.stopped
    await client.aclose()


async def test_global_deadline_covers_browser_startup_and_cleanup_is_attempted(monkeypatch):
    async def proxy_start(self):
        return "http://127.0.0.1:9"

    monkeypatch.setattr(_DenyProxy, "start", proxy_start)
    browser = FakeBrowser()
    manager = FakeManager(browser)

    async def slow_launch(**kwargs):
        await asyncio.sleep(5)

    manager.launch = slow_launch
    adapter = BrowserRetriever(policy(), timeout=0.03, playwright_factory=lambda: manager)
    result = await adapter.retrieve(ExtractionRequest(url=URL), {"state": "new"}, lambda _: None, 1)
    assert result.status == "pending" and result.error.code == "BROWSER_TIMEOUT"
    assert manager.stopped
    await adapter.close()


async def test_missing_chromium_is_actionable_without_logging_paths_or_retrying(monkeypatch):
    async def proxy_start(self):
        return "http://127.0.0.1:9"

    monkeypatch.setattr(_DenyProxy, "start", proxy_start)
    manager = FakeManager(FakeBrowser())

    async def missing_launch(**kwargs):
        raise RuntimeError("Executable doesn't exist at /private/example/browser")

    manager.launch = missing_launch
    adapter = BrowserRetriever(policy(), playwright_factory=lambda: manager)
    job = {"state": "new"}
    result = await adapter.retrieve(ExtractionRequest(url=URL), job, checkpoint(job), 5)
    assert result.status == "failed" and result.error.code == "BROWSER_BINARY_MISSING"
    assert "/private/" not in result.error.message and not result.error.retryable
    assert manager.stopped and job["attempts"] == 1
    await adapter.close()


class FakeDirect:
    max_response_bytes = 5_000_000

    def __init__(self, html):
        self.html = html
        self.calls = 0

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        if not job.get("record"):
            self.calls += 1
            checkpoint(
                {
                    "record": {"page_html": self.html, "final_url": request.url},
                    "state": "fetched",
                    "attempts": 1,
                }
            )
        return RetrievalOutcome(
            status="ready", record=job["record"], retrieval_method="direct_http", attempts=job["attempts"]
        )

    async def close(self):
        pass


async def test_auto_uses_separate_stage_records_and_resumes_browser_without_refetching_shell():
    direct = FakeDirect('<html><div id="root"></div><script src="app.js"></script></html>')
    calls = []

    class Browser(BrowserRetriever):
        async def _render(self, request, budget):
            calls.append(request.url)
            if len(calls) == 1:
                raise ScraperError("BROWSER_TIMEOUT", "timeout", retryable=True)
            return record()

    browser = Browser(policy())
    adapter = DirectThenBrowserRetriever(direct, browser)
    job = {"state": "new"}
    first = await adapter.retrieve(ExtractionRequest(url=URL), job, checkpoint(job), 5)
    assert first.status == "pending" and "record" not in job
    assert 'id="root"' in job["direct_stage"]["record"]["page_html"]
    assert "record" not in job["browser_stage"]
    second = await adapter.retrieve(ExtractionRequest(url=URL), job, checkpoint(job), 5)
    assert second.status == "ready" and "Useful source" in second.record["page_html"]
    assert direct.calls == 1 and len(calls) == 2
    assert job["record"] == second.record
    assert second.retrieval_method == "direct_then_browser"
    await adapter.close()


@pytest.mark.parametrize("html", [HTML, "<html><p>Verify you are human</p></html>"])
async def test_auto_does_not_render_healthy_pages_or_solve_challenges(html):
    class Browser(BrowserRetriever):
        async def _render(self, *args):
            pytest.fail("Browser must not run")

    browser = Browser(policy())
    adapter = DirectThenBrowserRetriever(FakeDirect(html), browser)
    result = await adapter.retrieve(ExtractionRequest(url=URL), {"state": "new"}, lambda _: None, 5)
    assert result.status == "ready" and result.record["rendered"] is False
    await adapter.close()


async def test_explicit_action_plan_forces_browser_and_does_not_fetch_static_page_first():
    direct = FakeDirect(HTML)

    class Browser(BrowserRetriever):
        async def _render(self, request, budget):
            return record()

    browser = Browser(policy())
    adapter = DirectThenBrowserRetriever(direct, browser)
    request = ExtractionRequest(url=URL, browser_actions=[BrowserAction(kind="scroll")])
    result = await adapter.retrieve(request, {"state": "new"}, lambda _: None, 5)
    assert result.status == "ready" and direct.calls == 0
    await adapter.close()


@pytest.mark.skipif(
    os.environ.get("RUN_BROWSER_TESTS") != "1", reason="Optional installed Chromium integration"
)
async def test_real_chromium_executes_js_and_get_fetch_then_trusted_click_without_direct_network():
    html = """<html><title>Live JS</title><body><main id="root"></main>
    <script src="/app.js"></script><button id="more" onclick="document.querySelector('#root').append(' clicked')">More</button></body></html>"""
    calls = []

    def handle(request):
        calls.append(str(request.url))
        path = request.url.path
        if path == "/app.js":
            return httpx.Response(
                200,
                headers={"Content-Type": "application/javascript"},
                text="fetch('/data').then(r=>r.text()).then(t=>document.querySelector('#root').textContent=t)",
            )
        if path == "/data":
            return httpx.Response(200, headers={"Content-Type": "text/plain"}, text="Rendered by JavaScript")
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text=html)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    adapter = BrowserRetriever(policy(), client=client, timeout=15)
    request = ExtractionRequest(
        url=URL,
        browser_actions=[
            BrowserAction(kind="wait_for", selector="#root:not(:empty)"),
            BrowserAction(kind="click", selector="#more"),
        ],
    )
    result = await adapter.retrieve(request, {"state": "new"}, lambda _: None, 20)
    assert result.status == "ready", result.error
    assert "Rendered by JavaScript clicked" in result.record["page_html"]
    assert URL + "/app.js" not in calls
    assert "https://site.test/app.js" in calls and "https://site.test/data" in calls
    assert result.record["final_url"] == URL
    await client.aclose()
