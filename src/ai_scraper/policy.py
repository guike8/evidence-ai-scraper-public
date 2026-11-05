import asyncio
import ipaddress
import socket
from urllib.parse import urlsplit

from .contracts import ScraperError
from .schemas import http_url


class URLPolicy:
    """Public hosts from an exact allowlist, or explicit ``*`` public-web mode.

    The wildcard changes host selection only: DNS, IP, port and redirect checks
    still apply. DirectHTTPRetriever additionally pins the verified destination
    for each connection to prevent DNS rebinding between check and retrieval.
    """

    def __init__(self, allowed_domains: list[str], *, resolve_dns: bool = True):
        self.allowed = {d.lower().rstrip(".") for d in allowed_domains}
        self.public_web = "*" in self.allowed
        self.resolve_dns = resolve_dns

    async def check(self, url: str) -> None:
        try:
            http_url(url)
            parsed = urlsplit(url)
            host = (parsed.hostname or "").lower().rstrip(".")
            port = parsed.port
        except ValueError as exc:
            raise ScraperError("INVALID_REQUEST", "Invalid target URL") from exc
        if (not self.public_web and host not in self.allowed) or port not in {None, 80, 443}:
            raise ScraperError("EGRESS_DENIED", "Target host or port is outside the configured allowlist")
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise ScraperError("EGRESS_DENIED", "Local hosts are forbidden")
        try:
            address = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            address = None
        if address is not None:
            if not address.is_global:
                raise ScraperError("EGRESS_DENIED", "Nonpublic target address")
        elif self.resolve_dns:
            try:
                results = await asyncio.wait_for(
                    asyncio.to_thread(socket.getaddrinfo, host, port or 443, type=socket.SOCK_STREAM), 10
                )
            except (OSError, TimeoutError) as exc:
                raise ScraperError("EGRESS_DNS_FAILED", "Could not verify public target DNS") from exc
            if not results or any(not ipaddress.ip_address(x[4][0]).is_global for x in results):
                raise ScraperError("EGRESS_DENIED", "Target resolves to a nonpublic address")

    async def check_record(self, record: dict, requested_url: str) -> None:
        await self.check(str(record.get("final_url") or record.get("url") or requested_url))
        for key in ("redirects", "redirect_chain"):
            chain = record.get(key) or []
            for hop in chain:
                url = hop.get("url") if isinstance(hop, dict) else hop
                if isinstance(url, str):
                    await self.check(url)
