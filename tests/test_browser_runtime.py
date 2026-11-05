"""Optional real Chromium checks with entirely intercepted fixture HTTP."""

from __future__ import annotations

import os

import httpx
import pytest

from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.browser import BrowserRetriever
from ai_scraper.schemas import BrowserAction, ExtractionRequest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_BROWSER_TESTS") != "1", reason="Optional installed Chromium integration"
)

URL = "https://runtime.test/start"


async def test_real_browser_blocks_post_and_websocket_and_reports_incomplete_source():
    calls = []
    html = """<html><title>Read-only browser</title><body><main id="done"></main>
    <script>
    const post = fetch('/submit', {method: 'POST', body: 'ignored'}).catch(() => {});
    const socket = new Promise(resolve => {
        const ws = new WebSocket('wss://runtime.test/socket');
        ws.onclose = resolve;
        ws.onerror = resolve;
    });
    Promise.all([post, socket]).then(() => document.querySelector('#done').textContent = 'complete');
    </script></body></html>"""

    def handle(request):
        calls.append((request.method, str(request.url)))
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text=html)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = BrowserRetriever(URLPolicy(["runtime.test"], resolve_dns=False), client=client, timeout=15)
        result = await adapter.retrieve(
            ExtractionRequest(
                url=URL,
                browser_actions=[BrowserAction(kind="wait_for", selector="#done:not(:empty)")],
            ),
            {"state": "new"},
            lambda _: None,
            20,
        )
    assert result.status == "ready", result.error
    assert calls == [("GET", URL)]
    assert result.record["browser"]["blocked_requests"] == {"unsafe_method": 1, "websocket": 1}
    assert "complete" in result.record["page_html"]
    assert any(warning.startswith("incomplete_source:") for warning in result.warnings)


async def test_real_browser_private_fetch_fails_before_native_transport():
    calls = []
    html = """<html><title>Private fetch refused</title><body><p>Public source.</p>
    <script>fetch('https://127.0.0.1/private').catch(() => {});</script></body></html>"""

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text=html)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = BrowserRetriever(URLPolicy(["*"], resolve_dns=False), client=client, timeout=15)
        result = await adapter.retrieve(ExtractionRequest(url=URL), {"state": "new"}, lambda _: None, 20)
    assert calls == [URL]
    assert result.status == "failed" and result.error.code == "EGRESS_DENIED"


async def test_real_browser_observes_redirect_and_resolves_javascript_fetch_from_final_url():
    calls = []
    html = """<html><title>Redirect destination</title><body><main id="data"></main>
    <script>fetch('data').then(r => r.text()).then(t => document.querySelector('#data').textContent = t);</script>
    </body></html>"""

    def handle(request):
        calls.append(str(request.url))
        assert "cookie" not in request.headers
        if request.url.path == "/start":
            return httpx.Response(
                302, headers={"Location": "/section/landing", "Set-Cookie": "private=ignored"}
            )
        if request.url.path == "/section/data":
            return httpx.Response(200, headers={"Content-Type": "text/plain"}, text="Final URL data")
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text=html)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = BrowserRetriever(URLPolicy(["runtime.test"], resolve_dns=False), client=client, timeout=15)
        result = await adapter.retrieve(
            ExtractionRequest(
                url=URL,
                browser_actions=[BrowserAction(kind="wait_for", selector="#data:not(:empty)")],
            ),
            {"state": "new"},
            lambda _: None,
            20,
        )
    assert result.status == "ready", result.error
    assert calls == [URL, "https://runtime.test/section/landing", "https://runtime.test/section/data"]
    assert result.record["final_url"] == "https://runtime.test/section/landing"
    assert result.record["observed_final_url"] is True
    assert result.record["redirect_chain"] == [
        {"url": "https://runtime.test/section/landing", "http_status": 302}
    ]
    assert "Final URL data" in result.record["page_html"]


async def test_real_browser_css_click_follows_validated_http_redirect():
    calls = []

    def handle(request):
        calls.append(str(request.url))
        if request.url.path == "/start":
            html = '<html><title>Start</title><body><a id="next" href="/redirect">Next</a></body></html>'
        elif request.url.path == "/redirect":
            return httpx.Response(302, headers={"Location": "/destination"})
        else:
            html = '<html><title>Destination</title><body><main id="landed">Clicked destination</main></body></html>'
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text=html)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = BrowserRetriever(URLPolicy(["runtime.test"], resolve_dns=False), client=client, timeout=15)
        result = await adapter.retrieve(
            ExtractionRequest(
                url=URL,
                browser_actions=[
                    BrowserAction(kind="click", selector="#next"),
                    BrowserAction(kind="wait_for", selector="#landed"),
                ],
            ),
            {"state": "new"},
            lambda _: None,
            20,
        )
    assert result.status == "ready", result.error
    assert calls == [URL, "https://runtime.test/redirect", "https://runtime.test/destination"]
    assert result.record["final_url"] == "https://runtime.test/destination"
    assert result.record["redirect_chain"] == [
        {"url": "https://runtime.test/destination", "http_status": 302}
    ]
    assert "Clicked destination" in result.record["page_html"]


async def test_real_browser_resource_redirect_fails_without_network_fallback():
    calls = []

    def handle(request):
        calls.append(str(request.url))
        if request.url.path == "/start":
            return httpx.Response(
                200,
                headers={"Content-Type": "text/html"},
                text='<html><title>Resource redirect</title><body><script src="/redirect.js"></script></body></html>',
            )
        if request.url.path == "/redirect.js":
            return httpx.Response(302, headers={"Location": "/actual.js"})
        pytest.fail("Redirected resource must not be forwarded or silently approximated")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        adapter = BrowserRetriever(URLPolicy(["runtime.test"], resolve_dns=False), client=client, timeout=15)
        result = await adapter.retrieve(ExtractionRequest(url=URL), {"state": "new"}, lambda _: None, 20)
    assert calls == [URL, "https://runtime.test/redirect.js"]
    assert result.status == "failed" and result.error.code == "BROWSER_RESOURCE_REDIRECT_UNSUPPORTED"
