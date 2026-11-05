"""SQLite transactions serialize claims, cache creation and spending reservations."""

import hashlib
import json
import math
import re
import sqlite3
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit

from .contracts import ScraperError
from .schemas import ExtractionRequest
from .storage import atomic_json, private_dir


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def extraction_contract(request: ExtractionRequest) -> dict:
    """Freshness controls retrieval reuse; it does not change the output task."""
    return request.model_dump(mode="json", exclude={"freshness_seconds"})


class JobStore:
    def __init__(self, root: Path, *, max_snapshot_bytes: int = 20_000_000):
        self.root = root.resolve()
        self.max_snapshot_bytes = max_snapshot_bytes
        private_dir(self.root)
        self.path = self.root / "jobs.sqlite3"
        with self.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, request_key TEXT, created REAL, updated REAL, body TEXT);
                CREATE INDEX IF NOT EXISTS request_keys ON jobs(request_key,created);
                CREATE TABLE IF NOT EXISTS leases(job_id TEXT PRIMARY KEY, host TEXT, owner TEXT, expires REAL);
                CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY, body TEXT);
                CREATE TABLE IF NOT EXISTS spend(id TEXT PRIMARY KEY, amount REAL NOT NULL, basis TEXT NOT NULL);
            """)
        self.path.chmod(0o600)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.execute("PRAGMA busy_timeout=10000")
        return db

    def get(self, job_id: str) -> dict:
        with self.connect() as db:
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise ScraperError("JOB_NOT_FOUND", "Unknown local job ID")
        return json.loads(row[0])

    def get_or_create(self, request: ExtractionRequest, retrieval_options: dict | None = None) -> dict:
        key = digest({"url": request.url, "country": request.country, "options": retrieval_options or {}})
        now = time.time()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute(
                "SELECT created, body FROM jobs WHERE request_key=? ORDER BY created DESC", (key,)
            ).fetchall()
            eligible = []
            for created, body in rows:
                job = json.loads(body)
                # Legacy replay jobs could have been keyed under a live backend.
                # Reject known mode mismatches while retaining the original key
                # shape and unknown pending submissions to avoid duplicate jobs.
                requested_method = (retrieval_options or {}).get("method")
                existing_method = job.get("retrieval_method")
                if requested_method and existing_method and requested_method != existing_method:
                    continue
                if (
                    job["state"] not in {"complete", "failed"}
                    or now - job.get("retrieval_created", created) < request.freshness_seconds
                ):
                    eligible.append((created, job))
            # Look across all shared tasks before creating another dependent
            # request; the newest row may belong to a different output schema.
            for created, job in eligible:
                if extraction_contract(
                    ExtractionRequest.model_validate(job["request"])
                ) == extraction_contract(request):
                    return job
            for created, job in eligible:
                # A different output schema/locale shares retrieval, with a
                # separate durable extraction request and extraction cache.
                # Keep a distinct durable extraction request while exactly
                # one root job owns the existing vendor submission. A child
                # waits on that root instead of losing its schema or POSTing
                # the same URL again after a restart.
                if job["state"] in {"new", "submitting", "pending", "reconciliation_required", "waiting"}:
                    job_id = uuid.uuid4().hex
                    shared = {
                        "id": job_id,
                        "state": "waiting",
                        "request": request.model_dump(mode="json"),
                        "attempts": 0,
                        "retrieval_parent_id": job.get("retrieval_parent_id", job["id"]),
                        "retrieval_created": job.get("retrieval_created", created),
                    }
                    db.execute(
                        "INSERT INTO jobs VALUES(?,?,?,?,?)", (job_id, key, now, now, json.dumps(shared))
                    )
                    return shared
                if job.get("record"):
                    job_id = uuid.uuid4().hex
                    shared = {
                        k: job[k]
                        for k in (
                            "record",
                            "snapshot_id",
                            "retrieval_method",
                            "snapshot_retrieved_at",
                            "render_record",
                            "render_method",
                            "render_attempted",
                            "render_retrieved_at",
                        )
                        if k in job
                    }
                    shared.update(
                        id=job_id,
                        state="fetched",
                        request=request.model_dump(mode="json"),
                        attempts=0,
                        retrieval_created=job.get("retrieval_created", created),
                    )
                    db.execute(
                        "INSERT INTO jobs VALUES(?,?,?,?,?)", (job_id, key, now, now, json.dumps(shared))
                    )
                    return shared
            job_id = uuid.uuid4().hex
            job = {
                "id": job_id,
                "state": "new",
                "request": request.model_dump(mode="json"),
                "attempts": 0,
                "retrieval_created": now,
            }
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?)", (job_id, key, now, now, json.dumps(job)))
        return job

    def adopt_retrieval(self, job_id: str, parent: dict) -> dict:
        """Copy immutable retrieval facts, never the parent's extraction state."""
        shared = {
            key: parent[key]
            for key in (
                "record",
                "snapshot_id",
                "retrieval_method",
                "snapshot_retrieved_at",
                "render_record",
                "render_method",
                "render_attempted",
                "render_retrieved_at",
            )
            if key in parent
        }
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise ScraperError("JOB_NOT_FOUND", "Unknown local job ID")
            job = json.loads(row[0])
            # A concurrent resume may already have adopted, rendered or finished
            # this child. Never replace its newer provenance with a stale copy
            # loaded before that caller acquired the child's execution lease.
            if job.get("record") or job["state"] != "waiting":
                return job
            job.update(state="fetched", **shared)
            db.execute("UPDATE jobs SET body=?, updated=? WHERE id=?", (json.dumps(job), time.time(), job_id))
        return job

    def update(self, job_id: str, **patch) -> dict:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                raise ScraperError("JOB_NOT_FOUND", "Unknown local job ID")
            job = json.loads(row[0])
            job.update(patch)
            db.execute("UPDATE jobs SET body=?, updated=? WHERE id=?", (json.dumps(job), time.time(), job_id))
        return job

    def acquire(self, job_id: str, owner: str, ttl: float = 900) -> bool:
        host = urlsplit(self.get(job_id)["request"]["url"]).hostname
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM leases WHERE expires < ?", (time.time(),))
            if db.execute("SELECT 1 FROM leases WHERE job_id=?", (job_id,)).fetchone():
                return False
            if db.execute("SELECT COUNT(*) FROM leases").fetchone()[0] >= 4:
                return False
            if db.execute("SELECT 1 FROM leases WHERE host=?", (host,)).fetchone():
                return False
            db.execute("INSERT INTO leases VALUES(?,?,?,?)", (job_id, host, owner, time.time() + ttl))
        return True

    def renew(self, job_id: str, owner: str, ttl: float = 900):
        with self.connect() as db:
            db.execute(
                "UPDATE leases SET expires=? WHERE job_id=? AND owner=?", (time.time() + ttl, job_id, owner)
            )

    def release(self, job_id: str, owner: str):
        with self.connect() as db:
            db.execute("DELETE FROM leases WHERE job_id=? AND owner=?", (job_id, owner))

    def reserve(self, reservation_id: str, amount: float, limit: float | None, basis: str):
        if (
            not math.isfinite(amount)
            or amount < 0
            or (limit is not None and (not math.isfinite(limit) or limit < 0))
        ):
            raise ScraperError("INVALID_BUDGET", "Spending values must be finite and nonnegative")
        if limit is None:
            raise ScraperError("BUDGET_UNCONFIGURED", "Configure a spending cap before live vendor calls")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM spend WHERE id=?", (reservation_id,)).fetchone():
                return
            used = db.execute("SELECT COALESCE(SUM(amount),0) FROM spend").fetchone()[0]
            if used + amount > limit + 1e-12:
                raise ScraperError("BUDGET_EXHAUSTED", "Live spending reservation exceeds configured cap")
            db.execute("INSERT INTO spend VALUES(?,?,?)", (reservation_id, amount, basis))

    def reserved_spend(self) -> float:
        with self.connect() as db:
            return db.execute("SELECT COALESCE(SUM(amount),0) FROM spend").fetchone()[0]

    def cache_get(self, key: str):
        with self.connect() as db:
            row = db.execute("SELECT body FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def cache_put(self, key: str, value: dict):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO cache VALUES(?,?)", (key, json.dumps(value)))

    def save_snapshot(self, job_id: str, snapshot) -> Path:
        path = self.root / "snapshots" / f"{job_id}-{snapshot.source_hash}.json"
        payload = snapshot.model_dump(mode="json")
        if len(json.dumps(payload, ensure_ascii=False, indent=2).encode()) + 1 > self.max_snapshot_bytes:
            raise ScraperError("CONTENT_TOO_LARGE", "Saved snapshot exceeds its configured byte limit")
        if path.exists():
            if path.stat().st_size > self.max_snapshot_bytes:
                raise ScraperError("SNAPSHOT_CORRUPT", "Existing saved source exceeds its byte limit")
            if json.loads(path.read_text()) != payload:
                raise ScraperError("SNAPSHOT_CONFLICT", "Refusing to overwrite immutable saved source")
        else:
            atomic_json(path, payload)
        self.update(job_id, snapshot_path=str(path))
        return path

    def save_result(self, result):
        path = self.root / "results" / f"{result.job_id}.json"
        atomic_json(path, result.model_dump(mode="json"))
        self.update(result.job_id, result_path=str(path))

    def save_source_view(self, job_id: str, descriptor: dict) -> Path:
        """Persist an immutable application-generated inference selection plan."""
        if not re.fullmatch(r"[a-f0-9]{32}", job_id):
            raise ScraperError("INVALID_REQUEST", "Invalid local source-view job ID")
        self.get(job_id)
        payload = json.dumps(descriptor, ensure_ascii=False, indent=2, allow_nan=False)
        if len(payload.encode()) + 1 > self.max_snapshot_bytes:
            raise ScraperError("CONTENT_TOO_LARGE", "Source-view descriptor exceeds its byte limit")
        directory = self.root / "source-views"
        private_dir(directory)
        path = directory / f"{job_id}-{digest(descriptor)}.json"
        if path.exists():
            if path.stat().st_size > self.max_snapshot_bytes or json.loads(path.read_text()) != descriptor:
                raise ScraperError("SOURCE_VIEW_CORRUPT", "Saved source-view descriptor changed")
        else:
            atomic_json(path, descriptor)
        return path

    def prune(self, retention_days: int) -> int:
        cutoff = time.time() - retention_days * 86400
        count = 0
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            rows = db.execute("SELECT id, body FROM jobs WHERE updated<?", (cutoff,)).fetchall()
            # A waiting extraction must still be able to resume its original
            # retrieval even when that root job is older than retention.
            protected_parents = {
                job["retrieval_parent_id"]
                for (body,) in db.execute("SELECT body FROM jobs").fetchall()
                if (job := json.loads(body)).get("retrieval_parent_id")
                and job["state"] not in {"complete", "failed"}
            }
            for job_id, body in rows:
                job = json.loads(body)
                if job_id in protected_parents:
                    continue
                if job["state"] not in {"complete", "failed"}:
                    continue
                if db.execute(
                    "SELECT 1 FROM leases WHERE job_id=? AND expires>?", (job_id, time.time())
                ).fetchone():
                    continue
                for key in ("snapshot_path", "result_path"):
                    if job.get(key):
                        path = Path(job[key]).resolve()
                        if path.is_relative_to(self.root):
                            path.unlink(missing_ok=True)
                for path in (self.root / "snapshots").glob(f"{job_id}-*.json"):
                    path.unlink(missing_ok=True)
                for path in (self.root / "source-views").glob(f"{job_id}-*.json"):
                    path.unlink(missing_ok=True)
                db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
                count += 1
            # Cached business data is source-derived and shares the retention boundary.
            if count:
                db.execute("DELETE FROM cache")
        return count
