from __future__ import annotations

import socket

import pytest

from ai_scraper.contracts import ScraperError
from ai_scraper.policy import URLPolicy
from ai_scraper.retrieval.direct import PublicNetworkBackend


@pytest.mark.parametrize(
    "url", ["http://93.184.216.34/", "https://93.184.216.34/path", "https://public.test/"]
)
async def test_wildcard_selects_any_public_host_without_relaxing_standard_ports(url):
    await URLPolicy(["*"], resolve_dns=False).check(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://10.1.2.3/",
        "http://172.16.10.2/",
        "http://192.168.1.10/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
        "http://[fd00::1]/",
        "http://[fe80::1]/",
        "http://[::ffff:127.0.0.1]/",
        "https://localhost/",
        "https://studio.local/",
        "https://service.internal/",
        "https://other.localhost/",
        "http://user:password@93.184.216.34/",
        "http://@93.184.216.34/",
        "ftp://93.184.216.34/",
        "file:///etc/passwd",
        "https://93.184.216.34:8443/",
    ],
)
async def test_wildcard_never_allows_private_local_credentialed_or_non_http_targets(url):
    with pytest.raises(ScraperError) as caught:
        await URLPolicy(["*"], resolve_dns=False).check(url)
    assert caught.value.error.code in {"EGRESS_DENIED", "INVALID_REQUEST"}


def _answers(*addresses):
    return [
        (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, 443)) for address in addresses
    ]


async def test_wildcard_dns_requires_every_resolved_address_to_be_public(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: _answers("93.184.216.34", "10.1.2.3"))
    with pytest.raises(ScraperError) as caught:
        await URLPolicy(["*"]).check("https://otherwise-public.example/")
    assert caught.value.error.code == "EGRESS_DENIED"
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: _answers("93.184.216.34"))
    await URLPolicy(["*"]).check("https://otherwise-public.example/")


@pytest.mark.parametrize(
    "record",
    [
        {"final_url": "http://127.0.0.1/admin"},
        {"final_url": "https://93.184.216.34/", "redirect_chain": ["http://169.254.169.254/"]},
        {"final_url": "https://93.184.216.34/", "redirects": [{"url": "http://10.0.0.1/"}]},
    ],
)
async def test_wildcard_rechecks_every_reported_redirect(record):
    with pytest.raises(ScraperError) as caught:
        await URLPolicy(["*"], resolve_dns=False).check_record(record, "https://93.184.216.34/")
    assert caught.value.error.code == "EGRESS_DENIED"


async def test_wildcard_native_transport_still_blocks_rebinding_before_socket(monkeypatch):
    class SocketBackend:
        def __init__(self):
            self.calls = []

        async def connect_tcp(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            raise AssertionError("A nonpublic address must never reach the socket backend")

    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: _answers("93.184.216.34", "127.0.0.1"))
    backend = SocketBackend()
    native = PublicNetworkBackend(URLPolicy(["*"], resolve_dns=False), backend=backend)
    with pytest.raises(ScraperError) as caught:
        await native.connect_tcp("universal.example", 443, timeout=1)
    assert caught.value.error.code == "EGRESS_DENIED"
    assert backend.calls == []


async def test_explicit_allowlist_keeps_prior_behavior_when_wildcard_is_absent():
    restricted = URLPolicy(["one.example"], resolve_dns=False)
    await restricted.check("https://one.example/")
    with pytest.raises(ScraperError) as caught:
        await restricted.check("https://two.example/")
    assert caught.value.error.code == "EGRESS_DENIED"
