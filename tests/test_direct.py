from __future__ import annotations

import json
import socket

import httpx
import pytest

from ai_scraper.contracts import ScraperError
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.direct import DirectHTTPRetriever, PublicHTTPTransport, PublicNetworkBackend
from ai_scraper.schemas import ExtractionRequest

URL = "https://shop.test/product/hoodie"
REQUEST = ExtractionRequest(url=URL)
HTML = "<html><h1>Hoodie</h1><p>USD 42.00</p></html>"


def policy(*hosts):
    return URLPolicy(list(hosts or ("shop.test",)), resolve_dns=False)


def save(job, path=None):
    def checkpoint(patch):
        job.update(patch)
        if path:
            path.write_text(json.dumps(job))

    return checkpoint


def page(html=HTML):
    return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, text=html)


class Clock:
    now = 0.0

    def __call__(self):
        return self.now

    async def sleep(self, delay):
        self.now += delay


async def test_direct_200_saves_html_and_observed_final_url_without_credentials():
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        assert request.headers["Accept-Encoding"] == "identity"
        assert "Authorization" not in request.headers
        assert "Cookie" not in request.headers
        return page()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    retriever = DirectHTTPRetriever(policy(), client=client)
    job = {"state": "new"}
    result = await retriever.retrieve(REQUEST, job, save(job), 5)
    assert result.status == "ready"
    assert result.retrieval_method == "direct_http"
    assert result.snapshot_id is None
    assert result.record["page_html"] == HTML
    assert result.record["final_url"] == URL
    assert result.record["observed_final_url"] is True
    assert result.record["input"] == {"url": URL}
    assert job["state"] == "fetched"
    assert job["record"] == result.record
    assert job["attempts"] == 1
    assert (await retriever.retrieve(REQUEST, job, save(job), 5)).record == result.record
    assert len(calls) == 1
    await client.aclose()


async def test_each_redirect_is_checked_and_cookie_or_auth_state_never_sent():
    calls = []
    target = "https://other.test/products/hoodie"

    def handle(request):
        calls.append(str(request.url))
        assert "Cookie" not in request.headers
        assert "Authorization" not in request.headers
        if str(request.url) == URL:
            return httpx.Response(302, headers={"Location": target, "Set-Cookie": "session=private; Path=/"})
        return page()

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle),
        headers={"Authorization": "Bearer private"},
        cookies={"session": "private"},
        auth=httpx.BasicAuth("private", "private"),
    )
    retriever = DirectHTTPRetriever(policy("shop.test", "other.test"), client=client)
    job = {"state": "new"}
    result = await retriever.retrieve(REQUEST, job, save(job), 5)
    assert result.status == "ready"
    assert calls == [URL, target]
    assert result.record["final_url"] == target
    assert result.record["redirect_chain"] == [{"url": target, "http_status": 302}]
    assert not client.cookies
    await client.aclose()


@pytest.mark.parametrize(
    "target", ["http://127.0.0.1/private", "http://10.0.0.1/", "https://other.test/", "file:///etc/passwd"]
)
async def test_forbidden_redirect_is_rejected_before_connection(target):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": target})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    job = {"state": "new"}
    result = await DirectHTTPRetriever(policy(), client=client).retrieve(REQUEST, job, save(job), 5)
    assert result.status == "failed"
    assert result.error.code in {"EGRESS_DENIED", "INVALID_REQUEST"}
    assert len(calls) == 1
    await client.aclose()


async def test_relative_redirect_and_loop_limits():
    calls = []

    def handle(request):
        calls.append(str(request.url))
        return httpx.Response(302, headers={"Location": "/second" if len(calls) == 1 else "/third"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    job = {"state": "new"}
    result = await DirectHTTPRetriever(policy(), client=client, max_redirects=1).retrieve(
        REQUEST, job, save(job), 5
    )
    assert result.status == "failed"
    assert result.error.code == "FETCH_REDIRECT_ERROR"
    assert calls == [URL, "https://shop.test/second"]
    await client.aclose()


async def test_retry_after_is_durable_and_resume_does_not_request_before_cooldown(tmp_path, monkeypatch):
    clock = Clock()
    wall = [1000.0]
    monkeypatch.setattr("ai_scraper.retrieval.direct.time.time", lambda: wall[0])
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "30"}) if len(calls) == 1 else page()

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    retriever = DirectHTTPRetriever(policy(), client=client, clock=clock, sleep=clock.sleep)
    job = {"state": "new"}
    path = tmp_path / "job.json"
    pending = await retriever.retrieve(REQUEST, job, save(job, path), 5)
    assert pending.status == "pending"
    assert len(calls) == 1
    assert job["direct_retry_not_before"] == 1030.0
    restored = json.loads(path.read_text())
    restarted = DirectHTTPRetriever(policy(), client=client, clock=clock, sleep=clock.sleep)
    assert (await restarted.retrieve(REQUEST, restored, save(restored), 5)).status == "pending"
    assert len(calls) == 1
    wall[0] = 1031.0
    assert (await restarted.retrieve(REQUEST, restored, save(restored), 5)).status == "ready"
    assert len(calls) == 2
    await client.aclose()


async def test_safe_transport_retries_are_bounded_and_errors_do_not_include_raw_secrets():
    clock = Clock()
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ConnectError("private-secret from transport", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    job = {"state": "new"}
    result = await DirectHTTPRetriever(policy(), client=client, clock=clock, sleep=clock.sleep).retrieve(
        REQUEST, job, save(job), 50
    )
    assert result.status == "pending"
    assert result.error.code == "FETCH_RETRY_EXHAUSTED"
    assert result.error.retryable
    assert len(calls) == 3
    assert "private-secret" not in result.error.model_dump_json()
    assert job["direct_retry_not_before"] > 0
    await client.aclose()


@pytest.mark.parametrize("content_type", ["application/pdf", "image/jpeg", "application/json"])
async def test_binary_or_nonpage_response_is_rejected(content_type):
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, headers={"Content-Type": content_type}, content=b"not a page")
        )
    )
    job = {"state": "new"}
    result = await DirectHTTPRetriever(policy(), client=client).retrieve(REQUEST, job, save(job), 5)
    assert result.status == "failed"
    assert result.error.code == "FETCH_CONTENT_TYPE"
    await client.aclose()


async def test_direct_response_is_bounded_before_normalization():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: page()))
    job = {"state": "new"}
    result = await DirectHTTPRetriever(policy(), client=client, max_response_bytes=10).retrieve(
        REQUEST, job, save(job), 5
    )
    assert result.status == "failed"
    assert result.error.code == "CONTENT_TOO_LARGE"
    assert "record" not in job
    await client.aclose()


async def test_direct_public_policy_rejects_initial_target_before_any_request():
    def unexpected(request):
        pytest.fail("Disallowed target must not be requested")

    client = httpx.AsyncClient(transport=httpx.MockTransport(unexpected))
    job = {"state": "new"}
    result = await DirectHTTPRetriever(policy(), client=client).retrieve(
        ExtractionRequest(url="http://127.0.0.1/"), job, save(job), 5
    )
    assert result.status == "failed"
    assert result.error.code == "EGRESS_DENIED"
    await client.aclose()


class FakeNetworkStream:
    def __init__(self):
        self.response = f"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: {len(HTML)}\r\n\r\n{HTML}".encode()
        self.writes = []
        self.server_names = []

    async def read(self, max_bytes, timeout=None):
        data, self.response = self.response, b""
        return data

    async def write(self, data, timeout=None):
        self.writes.append(bytes(data))

    async def start_tls(self, ssl_context, server_hostname=None, timeout=None):
        assert ssl_context.check_hostname
        assert ssl_context.verify_mode != 0
        self.server_names.append(server_hostname)
        return self

    async def aclose(self):
        pass

    def get_extra_info(self, info):
        return False if info == "is_readable" else None


class FakeNetworkBackend:
    def __init__(self):
        self.connections = []
        self.stream = FakeNetworkStream()

    async def connect_tcp(self, host, port, **kwargs):
        self.connections.append((host, port))
        return self.stream

    async def sleep(self, seconds):
        pass


def answer(address):
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))]


async def test_native_transport_pins_public_literal_but_preserves_tls_sni_and_host(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: answer("8.8.8.8"))
    inner = FakeNetworkBackend()
    guarded = PublicNetworkBackend(policy(), backend=inner)
    client = httpx.AsyncClient(transport=PublicHTTPTransport(policy(), backend=guarded), trust_env=False)
    retriever = DirectHTTPRetriever(policy(), client=client)
    job = {"state": "new"}
    result = await retriever.retrieve(REQUEST, job, save(job), 5)
    assert result.status == "ready"
    assert inner.connections == [("8.8.8.8", 443)]
    assert inner.stream.server_names == ["shop.test"]
    assert b"Host: shop.test" in b"".join(inner.stream.writes)
    await client.aclose()


async def test_dns_rebinding_to_private_address_never_reaches_socket(monkeypatch):
    answers = iter([answer("8.8.8.8"), answer("127.0.0.1")])
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: next(answers))
    checked_policy = URLPolicy(["shop.test"])
    await checked_policy.check(URL)  # Initial policy check sees a public address.
    inner = FakeNetworkBackend()
    backend = PublicNetworkBackend(checked_policy, backend=inner)
    with pytest.raises(ScraperError) as caught:
        await backend.connect_tcp("shop.test", 443, timeout=1)
    assert caught.value.error.code == "EGRESS_DENIED"
    assert not inner.connections


async def test_mixed_public_and_private_dns_answers_are_all_rejected(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: answer("8.8.8.8") + answer("10.0.0.1"))
    inner = FakeNetworkBackend()
    with pytest.raises(ScraperError) as caught:
        await PublicNetworkBackend(policy(), backend=inner).connect_tcp("shop.test", 443, timeout=1)
    assert caught.value.error.code == "EGRESS_DENIED"
    assert not inner.connections
