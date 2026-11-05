"""Universal public-web mode retains destination and redirect restrictions."""

import socket

import pytest

from ai_scraper.contracts import ScraperError
from ai_scraper.policy import URLPolicy


def dns(*addresses):
    return [
        (socket.AF_INET6 if ":" in address else socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 443))
        for address in addresses
    ]


async def test_public_web_mode_allows_distinct_public_domains_and_ips(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: dns("93.184.216.34"))
    policy = URLPolicy(["*"])
    for url in ("https://example.com/article", "http://another.example/jobs", "https://8.8.8.8"):
        await policy.check(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://[::1]/",
        "http://192.168.1.87/",
        "http://10.0.0.1/",
        "http://169.254.169.254/",
        "http://[fc00::1]/",
        "http://localhost/",
        "http://host.local/",
        "http://example.com:8000/",
        "http://user:pass@example.com/",
    ],
)
async def test_public_web_mode_never_authorizes_local_or_credentialed_destinations(url):
    with pytest.raises(ScraperError):
        await URLPolicy(["*"]).check(url)


@pytest.mark.parametrize("addresses", [("127.0.0.1",), ("93.184.216.34", "192.168.1.87"), ("::1",)])
async def test_public_web_mode_rejects_nonpublic_and_mixed_dns_answers(monkeypatch, addresses):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: dns(*addresses))
    with pytest.raises(ScraperError) as caught:
        await URLPolicy(["*"]).check("https://public-looking.example/")
    assert caught.value.error.code == "EGRESS_DENIED"


async def test_public_web_mode_checks_every_observed_redirect(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: dns("93.184.216.34"))
    with pytest.raises(ScraperError) as caught:
        await URLPolicy(["*"]).check_record(
            {"final_url": "https://example.com/", "redirect_chain": ["http://10.0.0.1/private"]},
            "https://example.com/",
        )
    assert caught.value.error.code == "EGRESS_DENIED"


async def test_exact_domain_mode_remains_available():
    with pytest.raises(ScraperError) as caught:
        await URLPolicy(["allowed.example"], resolve_dns=False).check("https://other.example/")
    assert caught.value.error.code == "EGRESS_DENIED"
