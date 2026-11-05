"""Credential-free retrieval of explicitly allowed public, noninteractive pages.

Native connections resolve and validate addresses at connection time, then pass
an IP literal to the socket backend. HTTPcore retains the original origin for
TLS SNI and certificate verification. No proxy environment, browser state,
authentication, cookies, or automatic redirects are used.
"""

from __future__ import annotations

import asyncio
import ipaddress
import random
import socket
import ssl
import time
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urljoin

import httpcore
import httpx
from httpcore._backends.auto import AutoBackend

from ..contracts import RetrievalOutcome, ScraperError
from ..policy import URLPolicy
from ..schemas import AppError, ExtractionRequest
from .brightdata import MAX_RESPONSE_BYTES, _retry_after, bounded_request

REDIRECTS = {301, 302, 303, 307, 308}
TRANSIENT = {408, 425, 429, 500, 502, 503, 504}


class PublicNetworkBackend(httpcore.AsyncNetworkBackend):
    """Connect only to vetted IP literals, retaining the TLS origin hostname.

    Uses the HTTPcore backend interface locked by uv.lock. The inner backend
    receives an address literal, preventing a second hostname resolution after
    the public-address check (DNS rebinding).
    """

    def __init__(self, policy: URLPolicy, *, backend=None):
        self.policy = policy
        self.backend = backend or AutoBackend()

    async def connect_tcp(self, host, port, timeout=None, local_address=None, socket_options=None):
        bracketed = f"[{host}]" if ":" in host else host
        await self.policy.check(f"{'https' if port == 443 else 'http'}://{bracketed}:{port}/")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            try:
                answers = await asyncio.wait_for(
                    asyncio.to_thread(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM),
                    min(timeout or 10.0, 10.0),
                )
            except (OSError, TimeoutError):
                raise ScraperError(
                    "EGRESS_DNS_FAILED", "Could not verify target connection addresses"
                ) from None
            addresses = list(dict.fromkeys(item[4][0] for item in answers))
        else:
            addresses = [str(address)]
        if not addresses:
            raise ScraperError("EGRESS_DNS_FAILED", "Target has no public connection address")
        try:
            public = all("%" not in item and ipaddress.ip_address(item).is_global for item in addresses)
        except ValueError:
            public = False
        if not public:
            raise ScraperError("EGRESS_DENIED", "Connection DNS resolved to a nonpublic address")
        deadline = time.monotonic() + min(timeout or 10.0, 10.0)
        for address in addresses[:16]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                return await self.backend.connect_tcp(
                    address, port, timeout=remaining, local_address=None, socket_options=socket_options
                )
            except (httpcore.ConnectError, httpcore.ConnectTimeout, OSError):
                continue
        raise httpcore.ConnectError("Public target connection failed")

    async def connect_unix_socket(self, *args, **kwargs):
        raise ScraperError("EGRESS_DENIED", "Unix socket retrieval is forbidden")

    async def sleep(self, seconds):
        await self.backend.sleep(seconds)


class PublicHTTPTransport(httpx.AsyncHTTPTransport):
    """HTTPX transport with a public-address-only HTTPcore network backend.

    HTTPX 0.28 exposes its connection pool through a private slot; the pinned
    HTTPX/HTTPcore combination and the TLS-origin test cover this integration.
    """

    def __init__(self, policy: URLPolicy, *, backend=None):
        super().__init__(trust_env=False, retries=0)
        self._pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(),
            max_connections=4,
            max_keepalive_connections=0,
            http1=True,
            http2=False,
            retries=0,
            network_backend=backend or PublicNetworkBackend(policy),
        )


async def _strip_credentials(request: httpx.Request) -> None:
    for key in ("Authorization", "Proxy-Authorization", "Cookie"):
        request.headers.pop(key, None)


class DirectHTTPRetriever:
    retrieval_method = "direct_http"

    def __init__(
        self,
        policy: URLPolicy,
        *,
        client: httpx.AsyncClient | None = None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
        max_redirects: int = 5,
        timeout: float = 75.0,
        max_attempts: int = 3,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be a positive integer")
        if not 0 <= max_redirects <= 20 or not 1 <= max_attempts <= 3 or not 0 < timeout <= 300:
            raise ValueError("Redirects, attempts, or timeout exceed supported bounds")
        self.policy = policy
        self.client = client or httpx.AsyncClient(
            transport=PublicHTTPTransport(policy),
            trust_env=False,
            follow_redirects=False,
            event_hooks={"request": [_strip_credentials]},
        )
        self._owns_client = client is None
        if client is not None:
            # Custom clients are intended for fixture transports. Strip inherited
            # authentication/cookie headers even if a fixture client has defaults.
            self.client.event_hooks["request"].append(_strip_credentials)
        self.max_response_bytes = max_response_bytes
        self.max_redirects = max_redirects
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.sleep = sleep
        self.clock = clock

    async def close(self):
        if self._owns_client:
            await self.client.aclose()

    def _checkpoint(self, job, checkpoint, **patch):
        checkpoint(patch)
        job.update(patch)

    def _outcome(self, job, status, *, record=None, error=None):
        return RetrievalOutcome(
            status=status,
            record=record,
            error=error,
            snapshot_id=None,
            retrieval_method=self.retrieval_method,
            attempts=int(job.get("attempts", 0)),
        )

    def _fail(self, job, checkpoint, error):
        self._checkpoint(job, checkpoint, state="failed", error=error.model_dump(mode="json"))
        return self._outcome(job, "failed", error=error)

    async def retrieve(self, request: ExtractionRequest, job: dict[str, Any], checkpoint, wait_seconds):
        if job.get("record") is not None:
            return self._outcome(job, "ready", record=job["record"])
        if job.get("state") == "failed":
            error = AppError.model_validate(
                job.get("error") or {"code": "FETCH_FAILED", "message": "Direct retrieval failed"}
            )
            return self._outcome(job, "failed", error=error)
        deadline = self.clock() + max(0.0, wait_seconds)
        current = job.get("direct_url") or request.url
        chain = list(job.get("redirect_chain") or [])
        try:
            await self.policy.check(request.url)
            for hop in chain:
                await self.policy.check(hop["url"])
            while self.clock() < deadline:
                await self.policy.check(current)
                response = None
                for attempt in range(self.max_attempts):
                    remaining = deadline - self.clock()
                    cooldown = max(0.0, float(job.get("direct_retry_not_before") or 0) - time.time())
                    if remaining <= 0 or cooldown >= remaining:
                        return self._outcome(job, "pending")
                    if cooldown:
                        await self.sleep(cooldown)
                    self._checkpoint(
                        job,
                        checkpoint,
                        state="pending",
                        direct_url=current,
                        redirect_chain=chain,
                        direct_retry_not_before=None,
                        snapshot_id=None,
                        retrieval_method=self.retrieval_method,
                        attempts=int(job.get("attempts", 0)) + 1,
                    )
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        return self._outcome(job, "pending")
                    try:
                        self.client.cookies.clear()
                        response = await bounded_request(
                            self.client,
                            "GET",
                            current,
                            max_response_bytes=self.max_response_bytes,
                            headers={
                                "User-Agent": "EvidenceAIScraper/0.1 (public product extraction)",
                                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,text/markdown;q=0.8",
                            },
                            timeout=httpx.Timeout(min(self.timeout, remaining), connect=min(10.0, remaining)),
                        )
                    except httpx.TransportError:
                        response = None
                        error = AppError(
                            code="FETCH_RETRY_EXHAUSTED",
                            message="Direct HTTP transport retries exhausted",
                            retryable=True,
                        )
                        delay = min(2**attempt + random.random(), 10.0)
                    else:
                        if response.status_code not in TRANSIENT:
                            break
                        error = AppError(
                            code="FETCH_RATE_LIMITED"
                            if response.status_code == 429
                            else "FETCH_RETRY_EXHAUSTED",
                            message="Direct HTTP page returned a transient failure",
                            retryable=True,
                            vendor_details={"http_status": response.status_code},
                        )
                        delay = _retry_after(response, min(2**attempt + random.random(), 10.0))
                    finally:
                        self.client.cookies.clear()
                    self._checkpoint(job, checkpoint, direct_retry_not_before=time.time() + delay)
                    if attempt + 1 == self.max_attempts:
                        return self._outcome(job, "pending", error=error)
                if response is None:
                    return self._outcome(job, "pending")
                if response.status_code in REDIRECTS:
                    location = response.headers.get("Location")
                    if not location or len(chain) >= self.max_redirects:
                        return self._fail(
                            job,
                            checkpoint,
                            AppError(
                                code="FETCH_REDIRECT_ERROR",
                                message="Missing redirect location or redirect limit exceeded",
                            ),
                        )
                    target = urljoin(str(response.url), location)
                    await self.policy.check(target)
                    if target == current or any(hop["url"] == target for hop in chain):
                        return self._fail(
                            job,
                            checkpoint,
                            AppError(code="FETCH_REDIRECT_ERROR", message="Redirect loop detected"),
                        )
                    chain.append({"url": target, "http_status": response.status_code})
                    current = target
                    self._checkpoint(job, checkpoint, direct_url=current, redirect_chain=chain)
                    continue
                if response.status_code != 200:
                    return self._fail(
                        job,
                        checkpoint,
                        AppError(
                            code="FETCH_HTTP_ERROR",
                            message="Direct HTTP page request failed",
                            vendor_details={"http_status": response.status_code},
                        ),
                    )
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type not in {"text/html", "application/xhtml+xml", "text/plain", "text/markdown"}:
                    return self._fail(
                        job,
                        checkpoint,
                        AppError(
                            code="FETCH_CONTENT_TYPE",
                            message="Direct retrieval accepts HTML or plain text pages only",
                        ),
                    )
                record = {
                    "input": {"url": request.url},
                    "final_url": str(response.url),
                    "observed_final_url": True,
                    "redirect_chain": chain,
                    "content_type": content_type,
                }
                record[
                    "page_html" if content_type in {"text/html", "application/xhtml+xml"} else "markdown"
                ] = response.text
                self._checkpoint(
                    job,
                    checkpoint,
                    state="fetched",
                    record=record,
                    error=None,
                    direct_retry_not_before=None,
                    retrieval_method=self.retrieval_method,
                )
                return self._outcome(job, "ready", record=record)
            return self._outcome(job, "pending")
        except ScraperError as exc:
            return self._fail(job, checkpoint, exc.error)
        except (ValueError, TypeError, KeyError):
            return self._fail(
                job,
                checkpoint,
                AppError(code="FETCH_INVALID_RESPONSE", message="Invalid direct retrieval state or response"),
            )
