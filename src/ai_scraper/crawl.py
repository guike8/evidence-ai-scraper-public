"""Bounded, durable public-site discovery over the extraction pipeline.

The frontier stores an extraction job ID before retrieval begins. Restarting a
crawl resumes that job, including pending vendor snapshots and reconciliation
states, instead of submitting a second retrieval. Page content supplies links,
never instructions, browser actions, configuration or credentials.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
import time
import uuid
from collections.abc import Callable
from email.utils import parsedate_to_datetime
from html.parser import HTMLParser
from typing import Literal
from urllib.parse import quote, urljoin, urlsplit, urlunsplit

import httpx
from pydantic import Field, field_validator, model_validator

from .config import Settings
from .contracts import ScraperError
from .extraction.source_quote_schema import SOURCE_QUOTE_SCHEMA_VERSION
from .jobs import JobStore, digest
from .policy import URLPolicy
from .retrieval.direct import REDIRECTS, PublicHTTPTransport, _strip_credentials
from .schemas import (
    DEFAULT_SCHEMA_ID,
    AppError,
    BrowserAction,
    CustomField,
    ExtractionRequest,
    StrictModel,
    http_url,
)

CRAWL_VERSION = "bounded-crawl-v1"
MAX_URL_CHARS = 4096
MAX_ROBOTS_BYTES = 65536
MAX_DISCOVERY_TAGS = 50000
_ID = re.compile(r"[a-f0-9]{32}\Z")
_TERMINAL = {"done", "failed", "skipped", "limited"}
_CONTENT_FAILURES = {"CONTENT_CHALLENGE", "CONTENT_JS_SHELL", "CONTENT_EMPTY"}


def canonical_url(value: str) -> str:
    """Normalize identity without guessing query semantics or decoding paths."""
    if not isinstance(value, str) or len(value) > MAX_URL_CHARS:
        raise ValueError("Crawl URLs exceed their supported length")
    http_url(value)
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower().rstrip(".")
    host.encode("ascii")  # use explicit punycode for international hostnames
    port = parsed.port
    authority = f"[{host}]" if ":" in host else host
    if port is not None and (parsed.scheme, port) not in {("http", 80), ("https", 443)}:
        authority += f":{port}"
    return urlunsplit((parsed.scheme.lower(), authority, parsed.path or "/", parsed.query, ""))


def _host(url: str) -> str:
    return (urlsplit(url).hostname or "").lower().rstrip(".")


class CrawlRequest(StrictModel):
    seeds: list[str] = Field(min_length=1, max_length=10)
    schema_id: str = DEFAULT_SCHEMA_ID
    fields: list[CustomField] = Field(default_factory=list, max_length=32)
    instruction: str | None = Field(default=None, max_length=2000)
    browser_actions: list[BrowserAction] = Field(default_factory=list, max_length=10)
    country: str | None = None
    locale: Literal["en_US", "en_GB", "de_DE", "fr_FR"] | None = None
    max_pages: int = Field(default=20, ge=1, le=1000)
    max_depth: int = Field(default=2, ge=0, le=10)
    max_frontier: int = Field(default=1000, ge=1, le=10000)
    max_links_per_page: int = Field(default=500, ge=1, le=5000)
    delay_seconds: float = Field(default=1.0, ge=0, le=60)
    wait_seconds: float = Field(default=180.0, ge=0, le=1800)
    scope_domains: list[str] = Field(default_factory=list, max_length=50)

    @field_validator("seeds")
    @classmethod
    def valid_seeds(cls, values):
        return list(dict.fromkeys(canonical_url(value) for value in values))

    @field_validator("scope_domains")
    @classmethod
    def exact_hosts(cls, values):
        result = []
        for value in values:
            host = value.lower().rstrip(".")
            if not host or len(host) > 253 or any(char in host for char in "/?#@*\\"):
                raise ValueError("Crawl scope requires exact hostnames")
            bracketed = f"[{host}]" if ":" in host else host
            if _host(canonical_url(f"https://{bracketed}/")) != host:
                raise ValueError("Invalid crawl scope hostname")
            result.append(host)
        return list(dict.fromkeys(result))

    @model_validator(mode="after")
    def validate_contract(self):
        if self.max_frontier < len(self.seeds):
            raise ValueError("Frontier limit must include every seed")
        scope = set(self.scope_domains or [_host(url) for url in self.seeds])
        if any(_host(url) not in scope for url in self.seeds):
            raise ValueError("Every seed must be inside the exact crawl scope")
        # Reuse country, field, URL and schema-independent request validation.
        self.extraction(self.seeds[0])
        return self

    def extraction(self, url: str) -> ExtractionRequest:
        return ExtractionRequest(
            url=url,
            schema_id=self.schema_id,
            fields=self.fields,
            instruction=self.instruction,
            browser_actions=self.browser_actions,
            country=self.country,
            locale=self.locale,
        )


class CrawlEntry(StrictModel):
    url: str
    depth: int
    parent_url: str | None = None
    state: str
    job_id: str | None = None
    extraction_status: str | None = None
    result_path: str | None = None
    error: AppError | None = None


class CrawlResult(StrictModel):
    crawl_id: str
    status: Literal["pending", "success", "partial", "failed"]
    counts: dict[str, int] = Field(default_factory=dict)
    entries: list[CrawlEntry] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    error: AppError | None = None


class CrawlStore:
    """Crawl and extraction state share a database and retention boundary."""

    def __init__(self, jobs: JobStore, *, clock=time.time):
        self.jobs, self.clock = jobs, clock
        with jobs.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS crawl_runs(
                    id TEXT PRIMARY KEY, contract_key TEXT, created REAL, updated REAL, body TEXT);
                CREATE INDEX IF NOT EXISTS crawl_contracts ON crawl_runs(contract_key);
                CREATE TABLE IF NOT EXISTS crawl_entries(
                    crawl_id TEXT, url TEXT, seq INTEGER, depth INTEGER, parent TEXT, state TEXT,
                    job_id TEXT, result TEXT, error TEXT, next_at REAL,
                    PRIMARY KEY(crawl_id,url));
                CREATE INDEX IF NOT EXISTS crawl_frontier ON crawl_entries(crawl_id,depth,seq);
                CREATE TABLE IF NOT EXISTS crawl_leases(crawl_id TEXT PRIMARY KEY, owner TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS crawl_hosts(host TEXT PRIMARY KEY, next_at REAL);
                CREATE TABLE IF NOT EXISTS crawl_robots(origin TEXT PRIMARY KEY, expires REAL, body TEXT);
            """)

    def _run(self, db, crawl_id):
        if not _ID.fullmatch(crawl_id):
            raise ScraperError("CRAWL_NOT_FOUND", "Unknown crawl ID")
        row = db.execute("SELECT body FROM crawl_runs WHERE id=?", (crawl_id,)).fetchone()
        if row is None:
            raise ScraperError("CRAWL_NOT_FOUND", "Unknown crawl ID")
        return json.loads(row[0])

    def get(self, crawl_id):
        with self.jobs.connect() as db:
            return self._run(db, crawl_id)

    def _save(self, db, crawl_id, body):
        db.execute(
            "UPDATE crawl_runs SET body=?,updated=? WHERE id=?",
            (json.dumps(body), self.clock(), crawl_id),
        )

    def create(self, request, scope, configuration):
        key = digest({"request": request.model_dump(mode="json"), "configuration": configuration})
        now = self.clock()
        with self.jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for crawl_id, serialized in db.execute(
                "SELECT id,body FROM crawl_runs WHERE contract_key=? ORDER BY created DESC", (key,)
            ):
                if json.loads(serialized)["status"] == "pending":
                    return crawl_id
            crawl_id = uuid.uuid4().hex
            body = {
                "request": request.model_dump(mode="json"),
                "scope": scope,
                "configuration": configuration,
                "status": "pending",
                "started": 0,
                "warnings": [],
                "error": None,
            }
            db.execute(
                "INSERT INTO crawl_runs VALUES(?,?,?,?,?)", (crawl_id, key, now, now, json.dumps(body))
            )
            for seq, url in enumerate(request.seeds):
                db.execute(
                    "INSERT INTO crawl_entries VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (crawl_id, url, seq, 0, None, "queued", None, None, None, 0),
                )
        return crawl_id

    def acquire(self, crawl_id, owner, ttl):
        now = self.clock()
        with self.jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._run(db, crawl_id)
            lease = db.execute(
                "SELECT owner,expires FROM crawl_leases WHERE crawl_id=?", (crawl_id,)
            ).fetchone()
            if lease is not None and lease[1] > now and lease[0] != owner:
                return False
            db.execute("INSERT OR REPLACE INTO crawl_leases VALUES(?,?,?)", (crawl_id, owner, now + ttl))
            # Only a newly owning controller can recover interrupted frontier work.
            db.execute(
                "UPDATE crawl_entries SET state=CASE WHEN job_id IS NULL THEN 'queued' ELSE 'pending' END "
                "WHERE crawl_id=? AND state='active'",
                (crawl_id,),
            )
        return True

    def _owned(self, db, crawl_id, owner):
        row = db.execute("SELECT owner,expires FROM crawl_leases WHERE crawl_id=?", (crawl_id,)).fetchone()
        if row is None or row[0] != owner or row[1] <= self.clock():
            raise ScraperError("CRAWL_BUSY", "Another controller owns this crawl")

    def release(self, crawl_id, owner):
        with self.jobs.connect() as db:
            db.execute("DELETE FROM crawl_leases WHERE crawl_id=? AND owner=?", (crawl_id, owner))

    def pick(self, crawl_id, owner, ttl):
        now = self.clock()
        with self.jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owned(db, crawl_id, owner)
            run = self._run(db, crawl_id)
            if run["started"] >= run["request"]["max_pages"]:
                changed = db.execute(
                    "UPDATE crawl_entries SET state='limited' WHERE crawl_id=? AND job_id IS NULL "
                    "AND state='queued'",
                    (crawl_id,),
                ).rowcount
                if changed:
                    run["warnings"] = list(dict.fromkeys(run["warnings"] + ["page_limit_reached"]))
                    self._save(db, crawl_id, run)
            row = db.execute(
                "SELECT MIN(depth) FROM crawl_entries WHERE crawl_id=? AND state IN ('queued','pending')",
                (crawl_id,),
            ).fetchone()
            if row[0] is None:
                return None, None
            rows = db.execute(
                "SELECT url,depth,parent,job_id,next_at FROM crawl_entries WHERE crawl_id=? "
                "AND depth=? AND state IN ('queued','pending') ORDER BY seq",
                (crawl_id, row[0]),
            ).fetchall()
            ready = next((item for item in rows if item[4] <= now), None)
            if ready is None:
                return None, min(item[4] for item in rows)
            db.execute("UPDATE crawl_leases SET expires=? WHERE crawl_id=?", (now + ttl, crawl_id))
            db.execute(
                "UPDATE crawl_entries SET state='active' WHERE crawl_id=? AND url=?", (crawl_id, ready[0])
            )
            return dict(zip(("url", "depth", "parent", "job_id", "next_at"), ready, strict=True)), None

    def bind_job(self, crawl_id, owner, url, job_id):
        with self.jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owned(db, crawl_id, owner)
            old = db.execute(
                "SELECT job_id FROM crawl_entries WHERE crawl_id=? AND url=?",
                (crawl_id, url),
            ).fetchone()
            if old is None or (old[0] is not None and old[0] != job_id):
                raise ScraperError("CRAWL_STATE_ERROR", "Cannot replace a bound extraction job")
            db.execute(
                "UPDATE crawl_entries SET job_id=? WHERE crawl_id=? AND url=?", (job_id, crawl_id, url)
            )
            if old[0] is None:
                run = self._run(db, crawl_id)
                run["started"] += 1
                self._save(db, crawl_id, run)

    def defer(self, crawl_id, owner, url, until, error=None):
        with self.jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owned(db, crawl_id, owner)
            db.execute(
                "UPDATE crawl_entries SET state='pending',next_at=?,error=? WHERE crawl_id=? AND url=?",
                (until, json.dumps(error.model_dump(mode="json")) if error else None, crawl_id, url),
            )

    def pace(self, host, delay):
        now = self.clock()
        with self.jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT next_at FROM crawl_hosts WHERE host=?", (host,)).fetchone()
            if row is not None and row[0] > now:
                return row[0]
            db.execute("INSERT OR REPLACE INTO crawl_hosts VALUES(?,?)", (host, now + delay))
        return now

    def finish(
        self, crawl_id, owner, entry, *, result=None, error=None, skipped=False, links=(), warnings=()
    ):
        """Page completion and child insertion form one transaction."""
        with self.jobs.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            self._owned(db, crawl_id, owner)
            run = self._run(db, crawl_id)
            state = (
                "skipped"
                if skipped
                else "failed"
                if error or (result and result.status == "failed")
                else "done"
            )
            summary = None
            if result is not None:
                job = self.jobs.get(result.job_id)
                summary = {"extraction_status": result.status, "result_path": job.get("result_path")}
                error = result.error
            db.execute(
                "UPDATE crawl_entries SET state=?,result=?,error=?,next_at=0 WHERE crawl_id=? AND url=?",
                (
                    state,
                    json.dumps(summary) if summary else None,
                    json.dumps(error.model_dump(mode="json")) if error else None,
                    crawl_id,
                    entry["url"],
                ),
            )
            count, last = db.execute(
                "SELECT COUNT(*),COALESCE(MAX(seq),-1) FROM crawl_entries WHERE crawl_id=?",
                (crawl_id,),
            ).fetchone()
            for link in links:
                if db.execute(
                    "SELECT 1 FROM crawl_entries WHERE crawl_id=? AND url=?", (crawl_id, link)
                ).fetchone():
                    continue
                if count >= run["request"]["max_frontier"]:
                    warnings = (*warnings, "frontier_limit_reached")
                    break
                last += 1
                count += 1
                state = "limited" if run["started"] >= run["request"]["max_pages"] else "queued"
                if state == "limited":
                    warnings = (*warnings, "page_limit_reached")
                db.execute(
                    "INSERT INTO crawl_entries VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (crawl_id, link, last, entry["depth"] + 1, entry["url"], state, None, None, None, 0),
                )
            run["warnings"] = list(dict.fromkeys(run["warnings"] + list(warnings)))[:64]
            self._save(db, crawl_id, run)

    def summarize(self, crawl_id, *, finalize=False):
        with self.jobs.connect() as db:
            if finalize:
                db.execute("BEGIN IMMEDIATE")
            run = self._run(db, crawl_id)
            rows = db.execute(
                "SELECT url,depth,parent,state,job_id,result,error FROM crawl_entries WHERE crawl_id=? "
                "ORDER BY depth,seq",
                (crawl_id,),
            ).fetchall()
            entries = []
            counts = {
                key: 0
                for key in (
                    "discovered",
                    "processed",
                    "accepted",
                    "partial",
                    "failed",
                    "skipped",
                    "pending",
                    "queued",
                    "limited",
                )
            }
            for url, depth, parent, state, job_id, result, error in rows:
                summary = json.loads(result) if result else {}
                entries.append(
                    CrawlEntry(
                        url=url,
                        depth=depth,
                        parent_url=parent,
                        state=state,
                        job_id=job_id,
                        error=AppError.model_validate_json(error) if error else None,
                        **summary,
                    )
                )
                counts["discovered"] += 1
                if state in {"done", "failed", "skipped"}:
                    counts["processed"] += 1
                if state in counts:
                    counts[state] += 1
                if state == "active":
                    counts["pending"] += 1
                if summary.get("extraction_status") in {"success", "partial"}:
                    counts["accepted"] += 1
                if summary.get("extraction_status") == "partial":
                    counts["partial"] += 1
            pending = any(item.state not in _TERMINAL for item in entries)
            status = (
                "pending"
                if pending
                else "failed"
                if not counts["accepted"]
                else (
                    "partial"
                    if counts["failed"] or counts["partial"] or counts["skipped"] or run["warnings"]
                    else "success"
                )
            )
            error = AppError.model_validate(run["error"]) if run.get("error") else None
            if not pending and not counts["accepted"] and error is None:
                error = AppError(
                    code="CRAWL_NO_ACCEPTED_RESULTS", message="No page produced accepted extraction data"
                )
            if finalize:
                run["status"] = status
                run["error"] = error.model_dump(mode="json") if error else None
                self._save(db, crawl_id, run)
        return CrawlResult(
            crawl_id=crawl_id,
            status=status,
            counts=counts,
            entries=entries,
            warnings=run["warnings"],
            error=error,
        )


class _Links(HTMLParser):
    def __init__(self, limit):
        super().__init__(convert_charrefs=True)
        self.limit, self.links, self.base = limit, [], None
        self.omitted = self.tags = 0

    def handle_starttag(self, tag, attrs):
        self.tags += 1
        attributes = dict(attrs)
        if tag.lower() == "base" and self.base is None:
            self.base = attributes.get("href")
        if tag.lower() in {"a", "area"} and isinstance(attributes.get("href"), str):
            if len(self.links) < self.limit:
                self.links.append(attributes["href"])
            else:
                self.omitted += 1


def discover_links(record, final_url, scope, *, limit, max_chars):
    html = record.get("page_html")
    if not isinstance(html, str):
        return [], ["discovery_html_unavailable"]
    parser, warnings = _Links(limit), []
    for index in range(0, min(len(html), max_chars), 4096):
        parser.feed(html[index : min(index + 4096, max_chars)])
        if parser.tags >= MAX_DISCOVERY_TAGS:
            warnings.append("discovery_tag_limit_reached")
            break
    if len(html) > max_chars:
        warnings.append("discovery_size_limit_reached")
    if parser.omitted:
        warnings.append("link_limit_reached")
    base = final_url
    if parser.base:
        try:
            proposed = canonical_url(urljoin(final_url, parser.base))
            if _host(proposed) in scope and urlsplit(proposed).port in {None, 80, 443}:
                base = proposed
            else:
                warnings.append("out_of_scope_base_ignored")
        except (ValueError, UnicodeError):
            warnings.append("invalid_base_ignored")
    found = []
    for href in parser.links:
        try:
            url = canonical_url(urljoin(base, href))
            if _host(url) in scope and urlsplit(url).port in {None, 80, 443}:
                found.append(url)
        except (ValueError, UnicodeError):
            continue
    return list(dict.fromkeys(found)), warnings


def _robots_path(value):
    value = quote(value, safe="/%?=&!$'()*+,;:@-._~")
    return re.sub(
        r"%([0-9a-fA-F]{2})",
        lambda m: (
            chr(int(m[1], 16))
            if chr(int(m[1], 16)) in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~"
            else "%" + m[1].upper()
        ),
        value,
    )


def _matches(pattern, path):
    """Wildcard matching with bounded literal searches, not backtracking regex."""
    ended = pattern.endswith("$")
    if ended:
        pattern = pattern[:-1]
    parts = pattern.split("*")
    if not path.startswith(parts[0]):
        return False
    position = len(parts[0])
    if len(parts) == 1:
        return path == pattern if ended else True
    for part in parts[1:-1]:
        index = path.find(part, position)
        if index < 0:
            return False
        position = index + len(part)
    last = parts[-1]
    return (
        (path.endswith(last) and len(path) - len(last) >= position)
        if ended
        else path.find(last, position) >= 0
    )


def robots_rules(text, user_agent, url):
    """Select user-agent groups and longest matching allow/disallow rules."""
    groups, agents, rules, delay, has_directive = [], [], [], 0.0, False
    for line in text.splitlines():
        if not line.strip() and agents:
            groups.append((agents, rules, delay))
            agents, rules, delay, has_directive = [], [], 0.0, False
            continue
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = (item.strip() for item in line.split(":", 1))
        key = key.lower()
        if key == "user-agent":
            if has_directive:
                groups.append((agents, rules, delay))
                agents, rules, delay, has_directive = [], [], 0.0, False
            agents.append(value.lower())
        elif agents and key in {"allow", "disallow"}:
            has_directive = True
            if not value:
                continue
            if len(value) > 2048 or len(rules) >= 1024:
                raise ValueError("Robots rule limits exceeded")
            rules.append((key == "allow", _robots_path(value)))
        elif agents and key == "crawl-delay":
            has_directive = True
            delay = float(value)
            if not math.isfinite(delay) or not 0 <= delay <= 86400:
                raise ValueError("Unsupported robots delay")
    groups.append((agents, rules, delay))
    ua = user_agent.lower()
    scored = [
        (
            max(
                (
                    len(agent) if agent != "*" else 0
                    for agent in group[0]
                    if agent == "*" or (agent and agent in ua)
                ),
                default=-1,
            ),
            group,
        )
        for group in groups
    ]
    best = max((item[0] for item in scored), default=-1)
    path = _robots_path(urlsplit(url).path or "/")
    if urlsplit(url).query:
        path += "?" + _robots_path(urlsplit(url).query)
    applicable, pause = [], 0.0
    for score, (_, selected_rules, selected_delay) in scored:
        if score == best and score >= 0:
            pause = max(pause, selected_delay)
            applicable.extend(
                (len(rule.replace("*", "").rstrip("$").encode()), allow)
                for allow, rule in selected_rules
                if _matches(rule, path)
            )
    return max(applicable, default=(0, True))[1], pause


class RobotsGate:
    def __init__(self, store, policy, user_agent, *, client=None, timeout=10):
        self.store, self.policy, self.user_agent, self.timeout = store, policy, user_agent, timeout
        self.client = client or httpx.AsyncClient(
            transport=PublicHTTPTransport(policy),
            trust_env=False,
            follow_redirects=False,
            event_hooks={"request": [_strip_credentials]},
        )
        self.owns_client = client is None
        if client is not None and _strip_credentials not in client.event_hooks["request"]:
            client.event_hooks["request"].append(_strip_credentials)

    async def close(self):
        if self.owns_client:
            await self.client.aclose()

    async def check(self, url, remaining):
        parsed = urlsplit(url)
        origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        cache_key = origin + "#" + hashlib.sha256(self.user_agent.encode()).hexdigest()
        now = self.store.clock()
        with self.store.jobs.connect() as db:
            row = db.execute("SELECT expires,body FROM crawl_robots WHERE origin=?", (cache_key,)).fetchone()
        if row is not None and row[0] > now:
            body, expires = json.loads(row[1]), row[0]
        else:
            body, expires = await self._fetch(origin + "/robots.txt", min(self.timeout, remaining))
            with self.store.jobs.connect() as db:
                db.execute(
                    "INSERT OR REPLACE INTO crawl_robots VALUES(?,?,?)",
                    (cache_key, expires, json.dumps(body)),
                )
        if body["kind"] == "unavailable":
            return None, 0.0, expires
        if body["kind"] == "deny":
            return False, 0.0, expires
        try:
            allowed, delay = robots_rules(body.get("text", ""), self.user_agent, url)
        except (ValueError, OverflowError):
            return None, 0.0, now + 300
        return allowed, delay, expires

    async def _fetch(self, url, timeout):
        now, current = self.store.clock(), url
        try:
            async with asyncio.timeout(timeout):
                for _ in range(6):
                    await self.policy.check(current)
                    async with self.client.stream(
                        "GET",
                        current,
                        headers={"User-Agent": self.user_agent, "Accept-Encoding": "identity"},
                        timeout=timeout,
                        follow_redirects=False,
                    ) as response:
                        status = response.status_code
                        if status in REDIRECTS:
                            location = response.headers.get("location")
                            if not location:
                                raise ValueError("Missing robots redirect")
                            current = canonical_url(urljoin(current, location))
                            continue
                        if status in {404, 410}:
                            return {"kind": "rules", "text": ""}, now + 3600
                        if status in {401, 403}:
                            return {"kind": "deny"}, now + 3600
                        if status != 200:
                            retry = response.headers.get("retry-after", "")
                            try:
                                delay = float(retry)
                            except ValueError:
                                try:
                                    delay = parsedate_to_datetime(retry).timestamp() - now
                                except (ValueError, TypeError, OverflowError):
                                    delay = 30.0
                            if not math.isfinite(delay):
                                delay = 300.0
                            return {"kind": "unavailable"}, now + min(3600, max(1, delay))
                        if response.headers.get("content-encoding", "identity").lower() not in {
                            "",
                            "identity",
                        }:
                            raise ValueError("Unsupported robots encoding")
                        chunks, size = [], 0
                        if response.is_stream_consumed:
                            chunks = [response.content]
                            size = len(response.content)
                        else:
                            async for chunk in response.aiter_raw():
                                size += len(chunk)
                                if size > MAX_ROBOTS_BYTES:
                                    raise ValueError("Robots response exceeded its byte limit")
                                chunks.append(chunk)
                        if size > MAX_ROBOTS_BYTES:
                            raise ValueError("Robots response exceeded its byte limit")
                        text = b"".join(chunks).decode("utf-8-sig")
                        return {"kind": "rules", "text": text}, now + 3600
        except (httpx.HTTPError, ScraperError, ValueError, TimeoutError, UnicodeError):
            pass
        return {"kind": "unavailable"}, now + 30


class Crawler:
    def __init__(
        self,
        settings: Settings,
        *,
        pipeline_factory: Callable | None = None,
        robots_client=None,
        policy=None,
        clock=time.time,
        sleep=asyncio.sleep,
    ):
        self.settings, self.pipeline_factory = settings, pipeline_factory
        self.policy = policy or URLPolicy(settings.allowed_domains)
        self.robots_client, self.clock, self.sleep = robots_client, clock, sleep
        self.jobs = JobStore(settings.state_dir, max_snapshot_bytes=settings.max_snapshot_bytes)
        self.store = CrawlStore(self.jobs, clock=clock)

    async def close(self):
        # Each invocation owns/closes its scoped network clients. Injected
        # fixture clients remain caller-owned.
        return None

    def _configuration(self):
        value = {
            "version": CRAWL_VERSION,
            "retrieval_backend": self.settings.retrieval_backend,
            "inference_backend": self.settings.inference_backend,
            "hf_provider": self.settings.hf_provider,
            "hf_model": self.settings.hf_model,
            "local_model": self.settings.local_model,
            "source_bound_evidence": self.settings.local_source_bound_evidence,
            "max_source_chars": self.settings.max_source_chars,
            "max_output_tokens": self.settings.max_output_tokens,
            "user_agent": getattr(self.settings, "user_agent", "EvidenceAIScraper/0.5"),
        }
        for name in (
            "browser_timeout_seconds",
            "browser_max_requests",
            "browser_max_total_bytes",
            "browser_max_attempts",
            "browser_resource_domains",
        ):
            if hasattr(self.settings, name):
                value[name] = getattr(self.settings, name)
        executable = getattr(self.settings, "browser_executable_path", None)
        if self.settings.local_source_quote_evidence:
            value["source_quote_evidence"] = True
            value["source_quote_policy_version"] = SOURCE_QUOTE_SCHEMA_VERSION
        value["browser_executable_path_hash"] = (
            hashlib.sha256(executable.encode()).hexdigest() if executable else None
        )
        if self.settings.retrieval_backend == "firecrawl":
            from .retrieval.firecrawl import firecrawl_cache_options

            value["firecrawl_retrieval"] = firecrawl_cache_options(
                self.settings.firecrawl_base_url,
                self.settings.firecrawl_runtime_id,
                self.settings.firecrawl_trust_service_egress,
            )
        return value

    async def _scope(self, request):
        scope = request.scope_domains or list(dict.fromkeys(_host(url) for url in request.seeds))
        global_policy = (
            self.policy if self.policy.allowed else URLPolicy(scope, resolve_dns=self.policy.resolve_dns)
        )
        for domain in scope:
            authority = f"[{domain}]" if ":" in domain else domain
            await global_policy.check(f"https://{authority}/")
        for url in request.seeds:
            await global_policy.check(url)
        return scope

    async def start(self, request: CrawlRequest) -> CrawlResult:
        request = CrawlRequest.model_validate(request)
        from .registry import get_schema

        get_schema(request.extraction(request.seeds[0]))
        scope = await self._scope(request)
        crawl_id = self.store.create(request, scope, self._configuration())
        return await self.resume(crawl_id)

    def status(self, crawl_id: str) -> CrawlResult:
        return self.store.summarize(crawl_id)

    async def resume(self, crawl_id: str, *, wait_seconds: float | None = None) -> CrawlResult:
        run = self.store.get(crawl_id)
        request = CrawlRequest.model_validate(run["request"])
        if run["configuration"] != self._configuration():
            raise ScraperError(
                "CRAWL_CONFIG_CHANGED", "Resume requires the original extraction configuration"
            )
        scope = await self._scope(request)
        if scope != run["scope"]:
            raise ScraperError("CRAWL_CONFIG_CHANGED", "Cannot broaden a saved crawl scope")
        budget = request.wait_seconds if wait_seconds is None else wait_seconds
        if type(budget) not in {int, float} or not math.isfinite(budget) or not 0 <= budget <= 1800:
            raise ScraperError(
                "INVALID_REQUEST", "Crawl waiting budget must be between zero and 1800 seconds"
            )
        if run["status"] != "pending" or budget == 0:
            return self.status(crawl_id)
        owner, ttl = uuid.uuid4().hex, budget + 30
        if not self.store.acquire(crawl_id, owner, ttl):
            return self.status(crawl_id)
        scoped = self.settings.model_copy(update={"allowed_domains": scope})
        factory = self.pipeline_factory
        if factory is None:
            from .pipeline import create_pipeline

            factory = create_pipeline
        pipeline = gate = None
        deadline = self.clock() + budget
        try:
            pipeline = factory(scoped)
            gate = RobotsGate(
                self.store,
                URLPolicy(scope, resolve_dns=self.policy.resolve_dns),
                getattr(self.settings, "user_agent", "EvidenceAIScraper/0.5"),
                client=self.robots_client,
            )
            while self.clock() < deadline:
                entry, until = self.store.pick(crawl_id, owner, ttl)
                if entry is None:
                    if until is None:
                        break
                    pause = min(max(0, until - self.clock()), max(0, deadline - self.clock()))
                    if pause <= 0:
                        break
                    await self.sleep(pause)
                    continue
                try:
                    if entry["job_id"] is None:
                        job_id = await pipeline.prepare_job(request.extraction(entry["url"]))
                        self.store.bind_job(crawl_id, owner, entry["url"], job_id)
                        entry["job_id"] = job_id
                    allowed, robots_delay, retry_at = await gate.check(entry["url"], deadline - self.clock())
                    if allowed is None:
                        self.store.defer(
                            crawl_id,
                            owner,
                            entry["url"],
                            retry_at,
                            AppError(
                                code="ROBOTS_UNAVAILABLE",
                                message="Robots policy is temporarily unavailable",
                                retryable=True,
                            ),
                        )
                        continue
                    if not allowed:
                        self.store.finish(
                            crawl_id,
                            owner,
                            entry,
                            skipped=True,
                            error=AppError(code="ROBOTS_DENIED", message="Robots policy disallows this URL"),
                        )
                        continue
                    until = self.store.pace(_host(entry["url"]), max(request.delay_seconds, robots_delay))
                    if until > self.clock():
                        self.store.defer(crawl_id, owner, entry["url"], until)
                        continue
                    remaining = deadline - self.clock()
                    if remaining <= 0:
                        self.store.defer(crawl_id, owner, entry["url"], self.clock())
                        break
                    result = await asyncio.wait_for(pipeline.resume(entry["job_id"]), timeout=remaining)
                    if result.status == "pending":
                        self.store.defer(
                            crawl_id,
                            owner,
                            entry["url"],
                            self.clock() + max(1, request.delay_seconds),
                            result.error,
                        )
                        continue
                    links, warnings = [], []
                    job = pipeline.store.get(entry["job_id"])
                    record = job.get("render_record") or job.get("record")
                    observed = str(
                        (record or {}).get("final_url") or result.provenance.final_url or entry["url"]
                    )
                    final_url = canonical_url(observed)
                    if _host(final_url) not in scope:
                        raise ScraperError("EGRESS_DENIED", "Observed final URL left the crawl scope")
                    if (
                        entry["depth"] < request.max_depth
                        and (result.error is None or result.error.code not in _CONTENT_FAILURES)
                        and isinstance(record, dict)
                    ):
                        links, warnings = discover_links(
                            record,
                            final_url,
                            scope,
                            limit=request.max_links_per_page,
                            max_chars=self.settings.max_response_bytes,
                        )
                    self.store.finish(crawl_id, owner, entry, result=result, links=links, warnings=warnings)
                except TimeoutError:
                    self.store.defer(
                        crawl_id,
                        owner,
                        entry["url"],
                        self.clock(),
                        AppError(
                            code="CRAWL_WAIT_EXPIRED", message="Crawl waiting budget expired", retryable=True
                        ),
                    )
                    break
                except ScraperError as exc:
                    self.store.finish(crawl_id, owner, entry, error=exc.error)
        finally:
            if gate is not None:
                await gate.close()
            if pipeline is not None:
                await pipeline.close()
            self.store.release(crawl_id, owner)
        return self.store.summarize(crawl_id, finalize=True)
