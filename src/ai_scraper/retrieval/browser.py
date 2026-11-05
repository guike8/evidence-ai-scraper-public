"""Optional read-only Chromium retrieval with public, pinned native HTTP I/O.

Browser HTTP requests are fulfilled by PublicHTTPTransport. An owned deny proxy
and disabled alternate transports provide additional defence; these settings
are not a claim of OS-level network isolation. No personal browser profile,
cookies, authentication, arbitrary scripts or model-authored actions are used.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin

import httpx

from ..contracts import RetrievalOutcome, ScraperError
from ..normalize import detect_content_problem, normalize_record
from ..policy import URLPolicy
from ..schemas import AppError, ExtractionRequest
from .brightdata import bounded_request
from .direct import REDIRECTS, DirectHTTPRetriever, PublicHTTPTransport, _strip_credentials

BROWSER_POLICY_VERSION = "public-browser-v2"
_SAFE_RESPONSE_HEADERS = frozenset(
    {
        "content-type",
        "location",
        "cache-control",
        "etag",
        "last-modified",
        "vary",
        "content-security-policy",
        "content-security-policy-report-only",
        "x-content-type-options",
        "access-control-allow-origin",
        "access-control-allow-methods",
        "access-control-allow-headers",
        "access-control-allow-credentials",
        "access-control-expose-headers",
        "cross-origin-resource-policy",
        "cross-origin-opener-policy",
        "cross-origin-embedder-policy",
        "referrer-policy",
    }
)
_BROWSER_ARGS = (
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--disable-extensions",
    "--disable-quic",
    "--disable-gpu",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-domain-reliability",
    "--disable-client-side-phishing-detection",
    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
    "--proxy-bypass-list=<-loopback>",
    "--js-flags=--max-old-space-size=256",
)


def _version() -> str | None:
    try:
        return importlib.metadata.version("playwright")
    except importlib.metadata.PackageNotFoundError:
        return None


class _DenyProxy:
    def __init__(self):
        self.server = None
        self.writers: set[asyncio.StreamWriter] = set()
        self.denials = 0

    async def _deny(self, reader, writer):
        self.denials += 1
        self.writers.add(writer)
        try:
            writer.write(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await asyncio.wait_for(writer.drain(), 1)
        except (OSError, TimeoutError):
            pass
        finally:
            writer.close()
            self.writers.discard(writer)

    async def start(self):
        self.server = await asyncio.start_server(self._deny, "127.0.0.1", 0, limit=8192)
        return f"http://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def close(self):
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        for writer in tuple(self.writers):
            writer.close()
        self.writers.clear()


@dataclass
class _Bridge:
    retriever: Any
    deadline: float
    page: Any = None
    requests: int = 0
    bytes_received: int = 0
    error: ScraperError | None = None
    blocked: Counter = field(default_factory=Counter)
    redirect_chain: list[dict] = field(default_factory=list)
    last_main_status: int | None = None
    pending_navigation: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def remaining(self) -> float:
        return max(0.001, self.deadline - time.monotonic())

    def fail(self, error: ScraperError):
        if self.error is None:
            self.error = error

    async def websocket(self, route):
        self.blocked["websocket"] += 1
        await route.close(code=1008, reason="WebSockets are disabled for extraction")

    async def route(self, route):
        request = route.request
        try:
            if self.error is not None:
                await route.abort("blockedbyclient")
                return
            async with self.lock:
                self.requests += 1
                if self.requests > self.retriever.max_requests:
                    raise ScraperError("BROWSER_REQUEST_LIMIT", "Browser request budget exhausted")
            method = request.method.upper()
            if method not in {"GET", "HEAD"}:
                self.blocked["unsafe_method"] += 1
                if request.is_navigation_request():
                    raise ScraperError("BROWSER_METHOD_DENIED", "Browser navigation must be read-only")
                await route.abort("blockedbyclient")
                return
            navigation = request.is_navigation_request()
            policy = self.retriever.policy if navigation else self.retriever.resource_policy
            await policy.check(request.url)
            redirects, ancestor = 0, request.redirected_from
            while ancestor is not None:
                redirects += 1
                if redirects > 5:
                    raise ScraperError("FETCH_REDIRECT_ERROR", "Browser redirect limit exceeded")
                ancestor = ancestor.redirected_from
            headers = {"User-Agent": self.retriever.user_agent}
            incoming = request.headers
            for name in ("accept", "accept-language", "origin"):
                if name in incoming:
                    headers[name] = incoming[name]
            async with self.lock:
                remaining_bytes = self.retriever.max_total_bytes - self.bytes_received
                if remaining_bytes <= 0:
                    raise ScraperError("CONTENT_TOO_LARGE", "Browser total byte budget exhausted")
                # Serial native requests prevent concurrent responses from
                # reserving the same remaining total-byte allowance.
                self.retriever.client.cookies.clear()
                try:
                    response = await bounded_request(
                        self.retriever.client,
                        method,
                        request.url,
                        max_response_bytes=min(self.retriever.max_response_bytes, remaining_bytes),
                        headers=headers,
                        auth=None,
                        timeout=httpx.Timeout(self.remaining(), connect=min(10.0, self.remaining())),
                    )
                finally:
                    self.retriever.client.cookies.clear()
                self.bytes_received += len(response.content)
            if response.status_code in REDIRECTS:
                location = response.headers.get("location")
                if not location:
                    raise ScraperError("FETCH_REDIRECT_ERROR", "Browser redirect has no destination")
                target = urljoin(request.url, location)
                await policy.check(target)
                if navigation and request.frame == self.page.main_frame:
                    if len(self.redirect_chain) >= 5:
                        raise ScraperError("FETCH_REDIRECT_ERROR", "Browser redirect limit exceeded")
                    self.redirect_chain.append({"url": target, "http_status": response.status_code})
                    self.pending_navigation = target
                    # Playwright routes only the first URL in an HTTP redirect
                    # chain. Do not hand Chromium a Location it could follow
                    # outside interception. Complete a temporary empty document
                    # and let the owning controller explicitly navigate to the
                    # checked destination. This document is never exported.
                    await route.fulfill(
                        status=200,
                        headers={"content-type": "text/html", "cache-control": "no-store"},
                        body=b"<!doctype html><title>Following public redirect</title>",
                    )
                    return
                raise ScraperError(
                    "BROWSER_RESOURCE_REDIRECT_UNSUPPORTED",
                    "Redirecting browser subresources require an explicit final resource URL",
                )
            if navigation and request.frame == self.page.main_frame:
                self.last_main_status = response.status_code
                content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                if response.status_code not in REDIRECTS and content_type not in {
                    "text/html",
                    "application/xhtml+xml",
                    "text/plain",
                    "text/markdown",
                }:
                    raise ScraperError(
                        "FETCH_CONTENT_TYPE", "Browser extraction requires an HTML or text document"
                    )
            safe_headers = {k: v for k, v in response.headers.items() if k.lower() in _SAFE_RESPONSE_HEADERS}
            await route.fulfill(status=response.status_code, headers=safe_headers, body=response.content)
        except ScraperError as exc:
            self.fail(exc)
            await route.abort("blockedbyclient")
        except (httpx.TransportError, OSError, TimeoutError):
            self.fail(
                ScraperError("BROWSER_FETCH_FAILED", "Browser resource retrieval failed", retryable=True)
            )
            await route.abort("failed")
        except Exception:  # noqa: BLE001 -- never echo page, URL or driver errors
            self.fail(ScraperError("BROWSER_FETCH_FAILED", "Browser resource could not be processed"))
            await route.abort("failed")


class BrowserRetriever:
    retrieval_method = "browser_playwright"
    supports_browser_actions = True

    def __init__(
        self,
        policy: URLPolicy,
        *,
        client: httpx.AsyncClient | None = None,
        max_response_bytes: int = 5_000_000,
        max_total_bytes: int = 10_000_000,
        max_requests: int = 80,
        timeout: float = 45,
        max_attempts: int = 2,
        user_agent: str = "EvidenceAIScraper/0.5",
        resource_domains: list[str] | None = None,
        executable_path: str | None = None,
        playwright_factory=None,
    ):
        for name, value, minimum, maximum in (
            ("max_response_bytes", max_response_bytes, 1024, 100_000_000),
            ("max_total_bytes", max_total_bytes, 1024, 100_000_000),
            ("max_requests", max_requests, 1, 500),
            ("max_attempts", max_attempts, 1, 3),
        ):
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"Invalid browser limit: {name}")
        if type(timeout) not in {float, int} or not math.isfinite(timeout) or not 0 < timeout <= 120:
            raise ValueError("Browser timeout must be finite and between zero and 120 seconds")
        if (
            not isinstance(user_agent, str)
            or not user_agent
            or len(user_agent) > 200
            or any(not 32 <= ord(c) <= 126 for c in user_agent)
        ):
            raise ValueError("Invalid browser user agent")
        self.policy = policy
        self.resource_policy = URLPolicy(
            list(policy.allowed) + list(resource_domains or []), resolve_dns=policy.resolve_dns
        )
        self.client = client or httpx.AsyncClient(
            transport=PublicHTTPTransport(self.resource_policy),
            trust_env=False,
            follow_redirects=False,
            event_hooks={"request": [_strip_credentials]},
        )
        if client is not None:
            client.event_hooks["request"].append(_strip_credentials)
        self._owns_client = client is None
        self.max_response_bytes, self.max_total_bytes = max_response_bytes, max_total_bytes
        self.max_requests, self.timeout, self.max_attempts = max_requests, float(timeout), max_attempts
        self.user_agent, self.executable_path = user_agent, executable_path
        self.playwright_factory = playwright_factory

    def cache_options_for_request(self, request: ExtractionRequest) -> dict:
        return {
            "policy_version": BROWSER_POLICY_VERSION,
            "engine": "chromium",
            "playwright": _version(),
            "max_response_bytes": self.max_response_bytes,
            "max_total_bytes": self.max_total_bytes,
            "max_requests": self.max_requests,
            "timeout": self.timeout,
            "max_attempts": self.max_attempts,
            "user_agent": self.user_agent,
            "resource_domains": sorted(self.resource_policy.allowed),
            "executable_identity": hashlib.sha256(self.executable_path.encode()).hexdigest()
            if self.executable_path
            else None,
            "actions": [action.model_dump(mode="json") for action in request.browser_actions],
        }

    async def close(self):
        if self._owns_client:
            await self.client.aclose()

    @staticmethod
    def _checkpoint(job, checkpoint, **patch):
        checkpoint(patch)
        job.update(patch)

    def _outcome(self, job, status, *, record=None, error=None, warnings=None):
        return RetrievalOutcome(
            status=status,
            record=record,
            error=error,
            warnings=warnings or [],
            retrieval_method=self.retrieval_method,
            attempts=int(job.get("attempts", 0)),
        )

    async def _render(self, request: ExtractionRequest, budget: float) -> dict:
        if self.playwright_factory is None:
            try:
                from playwright.async_api import async_playwright
            except ImportError:
                raise ScraperError(
                    "BROWSER_DEPENDENCY_MISSING", "Install the browser extra and its Chromium binary"
                ) from None
            factory = async_playwright
        else:
            factory = self.playwright_factory
        manager = browser = context = builder = None
        proxy = _DenyProxy()
        bridge = _Bridge(self, time.monotonic() + budget)
        try:
            async with asyncio.timeout(budget):
                proxy_url = await proxy.start()
                builder = factory()
                manager = await builder.start()
                options = {
                    "headless": True,
                    "chromium_sandbox": True,
                    "args": list(_BROWSER_ARGS),
                    "proxy": {"server": proxy_url},
                    "timeout": max(1, int(bridge.remaining() * 1000)),
                }
                if self.executable_path:
                    options["executable_path"] = self.executable_path
                browser = await manager.chromium.launch(**options)
                context = await browser.new_context(
                    service_workers="block",
                    accept_downloads=False,
                    ignore_https_errors=False,
                    permissions=[],
                    user_agent=self.user_agent,
                )
                context.set_default_timeout(max(1, int(bridge.remaining() * 1000)))
                await context.route("**/*", bridge.route)
                await context.route_web_socket("**/*", bridge.websocket)
                page = await context.new_page()
                bridge.page = page

                async def dismiss_dialog(dialog):
                    bridge.blocked["dialog"] += 1
                    await dialog.dismiss()

                async def close_popup(popup):
                    bridge.fail(
                        ScraperError("BROWSER_POPUP_DENIED", "Browser extraction supports one page only")
                    )
                    await popup.close()

                async def cancel_download(download):
                    bridge.fail(ScraperError("BROWSER_DOWNLOAD_DENIED", "Browser downloads are disabled"))
                    await download.cancel()

                page.on("dialog", dismiss_dialog)
                page.on("popup", close_popup)
                page.on("download", cancel_download)

                async def follow_navigation_redirects():
                    while bridge.pending_navigation is not None:
                        target, bridge.pending_navigation = bridge.pending_navigation, None
                        await self.policy.check(target)
                        await page.goto(
                            target,
                            wait_until="networkidle",
                            timeout=max(1, int(bridge.remaining() * 1000)),
                        )
                        if bridge.error:
                            raise bridge.error

                await page.goto(
                    request.url, wait_until="networkidle", timeout=max(1, int(bridge.remaining() * 1000))
                )
                if bridge.error:
                    raise bridge.error
                await follow_navigation_redirects()
                for action in request.browser_actions:
                    timeout_ms = max(1, min(action.timeout_ms, int(bridge.remaining() * 1000)))
                    if action.kind == "click":
                        await page.locator("css=" + action.selector).click(timeout=timeout_ms)
                    elif action.kind == "wait_for":
                        await page.locator("css=" + action.selector).wait_for(
                            state="visible", timeout=timeout_ms
                        )
                    else:
                        await page.mouse.wheel(0, action.pixels)
                    await page.wait_for_load_state(
                        "networkidle", timeout=max(1, int(bridge.remaining() * 1000))
                    )
                    if bridge.error:
                        raise bridge.error
                    await follow_navigation_redirects()
                await self.policy.check(page.url)
                if bridge.last_main_status is not None and not 200 <= bridge.last_main_status < 300:
                    raise ScraperError(
                        "FETCH_HTTP_ERROR",
                        "Rendered page returned an unsuccessful HTTP status",
                        vendor_details={"http_status": bridge.last_main_status},
                    )
                html = await page.content()
                if len(html.encode()) > self.max_response_bytes:
                    raise ScraperError("CONTENT_TOO_LARGE", "Rendered DOM exceeds its byte limit")
                title = await page.title()
                return {
                    "input": {"url": request.url},
                    "final_url": page.url,
                    "observed_final_url": True,
                    "redirect_chain": bridge.redirect_chain,
                    "page_html": html,
                    "page_title": title,
                    "content_type": "text/html",
                    "rendered": True,
                    "browser": {
                        "policy_version": BROWSER_POLICY_VERSION,
                        "engine": "chromium",
                        "engine_version": browser.version,
                        "requests": bridge.requests,
                        "resource_bytes": bridge.bytes_received,
                        "blocked_requests": dict(bridge.blocked),
                        "fallback_proxy_denials": proxy.denials,
                        "actions": [action.model_dump(mode="json") for action in request.browser_actions],
                        "wait_state": "networkidle",
                    },
                }
        except TimeoutError:
            raise ScraperError(
                "BROWSER_TIMEOUT", "Browser retrieval exceeded its finite deadline", retryable=True
            ) from None
        except ScraperError:
            raise
        except Exception as exc:  # noqa: BLE001 -- driver exceptions may contain source/URLs
            if bridge.error:
                raise bridge.error from None
            if browser is None and "Executable doesn't exist" in str(exc):
                raise ScraperError(
                    "BROWSER_BINARY_MISSING", "Install the Chromium binary matching the browser extra"
                ) from None
            timed_out = type(exc).__name__ == "TimeoutError"
            raise ScraperError(
                "BROWSER_TIMEOUT" if timed_out else "BROWSER_FAILED",
                "Browser operation timed out" if timed_out else "Browser could not render the page",
                retryable=True,
            ) from None
        finally:
            # Each operation owns only these ephemeral browser objects. Never
            # terminate a shared browser or signal a process by a stored PID.
            cleanup_errors = set()
            for name, target in (("context", context), ("browser", browser), ("manager", manager)):
                if target is not None:
                    operation = target.stop if target is manager else target.close
                    try:
                        await asyncio.wait_for(operation(), 3)
                    except Exception:  # noqa: BLE001 -- cleanup stays bounded and redacted
                        cleanup_errors.add(name)
            if manager is None and builder is not None and hasattr(builder, "__aexit__"):
                try:
                    await asyncio.wait_for(builder.__aexit__(None, None, None), 3)
                except Exception:  # noqa: BLE001 -- no driver exception strings
                    cleanup_errors.add("manager")
            await proxy.close()
            if "manager" in cleanup_errors or (browser is not None and browser.is_connected()):
                raise ScraperError("BROWSER_CLEANUP_FAILED", "Could not verify ephemeral browser cleanup")

    async def retrieve(self, request: ExtractionRequest, job: dict, checkpoint, wait_seconds):
        options = self.cache_options_for_request(request)
        identity = hashlib.sha256(
            json.dumps({"url": request.url, "options": options}, sort_keys=True).encode()
        ).hexdigest()
        existing = job.get("browser_contract")
        if existing is not None and existing != identity:
            return self._outcome(
                job,
                "failed",
                error=AppError(
                    code="BROWSER_STATE_MISMATCH",
                    message="Saved browser job belongs to a different action plan",
                ),
            )
        if job.get("record") is not None:
            return self._outcome(job, "ready", record=job["record"])
        if job.get("state") == "failed":
            return self._outcome(
                job,
                "failed",
                error=AppError.model_validate(
                    job.get("error") or {"code": "BROWSER_FAILED", "message": "Browser retrieval failed"}
                ),
            )
        if wait_seconds <= 0:
            return self._outcome(job, "pending")
        try:
            await self.policy.check(request.url)
            if request.country:
                raise ScraperError(
                    "BROWSER_COUNTRY_UNSUPPORTED",
                    "Local browser retrieval cannot guarantee a requested country",
                )
            attempts = int(job.get("attempts", 0))
            if attempts >= self.max_attempts:
                raise ScraperError("BROWSER_RETRY_EXHAUSTED", "Browser retrieval attempt limit reached")
            self._checkpoint(
                job,
                checkpoint,
                state="pending",
                browser_contract=identity,
                attempts=attempts + 1,
                retrieval_method=self.retrieval_method,
                error=None,
            )
            record = await self._render(request, min(float(wait_seconds), self.timeout))
            await self.policy.check_record(record, request.url)
            warnings = []
            observations = record.get("browser", {})
            if observations.get("blocked_requests"):
                warnings.append("incomplete_source: browser blocked unsupported network activity")
            self._checkpoint(
                job,
                checkpoint,
                record=record,
                state="fetched",
                error=None,
                retrieval_method=self.retrieval_method,
            )
            return self._outcome(job, "ready", record=record, warnings=warnings)
        except ScraperError as exc:
            pending = exc.error.retryable and int(job.get("attempts", 0)) < self.max_attempts
            self._checkpoint(
                job,
                checkpoint,
                state="pending" if pending else "failed",
                error=exc.error.model_dump(mode="json"),
            )
            return self._outcome(job, "pending" if pending else "failed", error=exc.error)


class DirectThenBrowserRetriever:
    """Use direct HTML first, rendering only script shells or explicit actions."""

    retrieval_method = "direct_then_browser"
    supports_browser_actions = True

    def __init__(self, direct: DirectHTTPRetriever, browser: BrowserRetriever):
        self.direct, self.browser = direct, browser

    def cache_options_for_request(self, request):
        return {
            "direct_then_browser_version": BROWSER_POLICY_VERSION,
            "browser": self.browser.cache_options_for_request(request),
        }

    async def close(self):
        await self.direct.close()
        await self.browser.close()

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        if job.get("record") is not None:
            return RetrievalOutcome(
                status="ready",
                record=job["record"],
                retrieval_method=self.retrieval_method,
                attempts=int(job.get("attempts", 0)),
            )
        started = time.monotonic()

        def stage(name):
            value = dict(job.get(name) or {"state": "new", "attempts": 0})

            def persist(patch):
                value.update(patch)
                checkpoint({name: dict(value)})
                job[name] = dict(value)

            return value, persist

        direct_record = None
        if not request.browser_actions:
            direct_job, save_direct = stage("direct_stage")
            result = await self.direct.retrieve(request, direct_job, save_direct, wait_seconds)
            if result.status != "ready":
                patch = {"state": result.status, "attempts": result.attempts}
                if result.error:
                    patch["error"] = result.error.model_dump(mode="json")
                checkpoint(patch)
                job.update(patch)
                return RetrievalOutcome(
                    status=result.status,
                    error=result.error,
                    warnings=result.warnings,
                    retrieval_method=self.retrieval_method,
                    attempts=result.attempts,
                )
            direct_record = result.record
            snapshot = normalize_record(
                direct_record,
                request,
                retrieval_method="direct_http",
                max_record_bytes=self.direct.max_response_bytes,
            )
            problem = detect_content_problem(snapshot)
            html = str(direct_record.get("page_html") or "")
            use_browser = problem == "CONTENT_JS_SHELL" or (
                problem == "CONTENT_EMPTY" and "<script" in html.lower()
            )
            if not use_browser:
                record = {**direct_record, "rendered": False}
                patch = {
                    "record": record,
                    "state": "fetched",
                    "retrieval_method": self.retrieval_method,
                    "attempts": int(direct_job.get("attempts", 0)),
                }
                checkpoint(patch)
                job.update(patch)
                return RetrievalOutcome(
                    status="ready",
                    record=record,
                    retrieval_method=self.retrieval_method,
                    attempts=patch["attempts"],
                )
        remaining = max(0, wait_seconds - (time.monotonic() - started))
        browser_job, save_browser = stage("browser_stage")
        rendered = await self.browser.retrieve(request, browser_job, save_browser, remaining)
        attempts = int((job.get("direct_stage") or {}).get("attempts", 0)) + int(
            browser_job.get("attempts", 0)
        )
        patch = {
            "attempts": attempts,
            "retrieval_method": self.retrieval_method,
            "state": "fetched" if rendered.status == "ready" else rendered.status,
        }
        if rendered.status == "ready":
            patch["record"] = rendered.record
        if rendered.error:
            patch["error"] = rendered.error.model_dump(mode="json")
        checkpoint(patch)
        job.update(patch)
        return RetrievalOutcome(
            status=rendered.status,
            record=rendered.record,
            error=rendered.error,
            warnings=rendered.warnings,
            retrieval_method=self.retrieval_method,
            attempts=attempts,
        )
