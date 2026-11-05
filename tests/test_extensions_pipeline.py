from __future__ import annotations

import json
import time

import pytest

from ai_scraper.config import Settings
from ai_scraper.contracts import RetrievalOutcome, ScraperError
from ai_scraper.jobs import JobStore
from ai_scraper.pipeline import Pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.schemas import BrowserAction, ExtractionRequest

URL = "https://crawl.example.test/start"
HTML = "<main><h1>Public handbook</h1><p>A literal public paragraph.</p></main>"


class PendingRetriever:
    retrieval_method = "brightdata_fetch"

    def __init__(self):
        self.submissions = self.polls = 0

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        if not job.get("snapshot_id"):
            self.submissions += 1
            checkpoint({"snapshot_id": "existing-vendor-job", "state": "pending", "attempts": 1})
            return RetrievalOutcome("pending", snapshot_id="existing-vendor-job", attempts=1)
        self.polls += 1
        return RetrievalOutcome(
            "ready",
            record={"page_html": HTML, "final_url": URL},
            snapshot_id="existing-vendor-job",
            attempts=1,
        )


def pipeline(tmp_path, retriever, backend="brightdata"):
    return Pipeline(
        Settings(state_dir=tmp_path, retrieval_backend=backend),
        retriever=retriever,
        policy=URLPolicy(["crawl.example.test"], resolve_dns=False),
        live=False,
    )


async def test_preparation_is_durable_and_does_not_submit_before_frontier_binding(tmp_path):
    retriever = PendingRetriever()
    first = pipeline(tmp_path, retriever)
    request = ExtractionRequest(url=URL, schema_id="page_v1")
    job_id = await first.prepare_job(request)
    assert retriever.submissions == retriever.polls == 0
    assert first.store.get(job_id)["state"] == "new"
    assert (await first.resume(job_id)).status == "pending"
    reopened = pipeline(tmp_path, retriever)
    assert await reopened.prepare_job(request.model_copy(update={"freshness_seconds": 0})) == job_id
    result = await reopened.resume(job_id)
    assert result.status in {"success", "partial"}
    assert result.job_id == job_id
    assert (retriever.submissions, retriever.polls) == (1, 1)


async def test_actions_fail_before_job_creation_for_nonbrowser_adapter(tmp_path):
    retriever = PendingRetriever()
    instance = pipeline(tmp_path, retriever)
    request = ExtractionRequest(url=URL, browser_actions=[BrowserAction(kind="click", selector=".next")])
    result = await instance.extract_url(request)
    assert result.error.code == "INVALID_REQUEST"
    assert retriever.submissions == 0
    with instance.store.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 0


def test_empty_browser_options_preserve_legacy_request_serialization():
    assert "browser_actions" not in ExtractionRequest(url=URL).model_dump(mode="json")
    assert "browser_actions" not in ExtractionRequest(url=URL, browser_actions=[]).model_dump(mode="json")


async def test_action_plans_change_retrieval_identity_before_rendering(tmp_path):
    class Browser(PendingRetriever):
        retrieval_method = "browser_playwright"
        supports_browser_actions = True

        def cache_options_for_request(self, request):
            return {"actions": [item.model_dump(mode="json") for item in request.browser_actions]}

    instance = pipeline(tmp_path, Browser(), backend="browser")
    first = ExtractionRequest(url=URL, browser_actions=[BrowserAction(kind="click", selector=".first")])
    second = first.model_copy(update={"browser_actions": [BrowserAction(kind="click", selector=".second")]})
    assert await instance.prepare_job(first) != await instance.prepare_job(second)
    assert instance.retriever.submissions == 0


async def test_incomplete_browser_warning_survives_saved_source_replay(tmp_path):
    class Browser:
        retrieval_method = "browser_playwright"

        def __init__(self):
            self.calls = 0

        async def retrieve(self, request, job, checkpoint, wait_seconds):
            self.calls += 1
            return RetrievalOutcome(
                "ready",
                record={"page_html": HTML, "final_url": URL, "observed_final_url": True},
                retrieval_method=self.retrieval_method,
                attempts=1,
                warnings=["incomplete_source: a non-read-only request was blocked"],
            )

    browser = Browser()
    first = pipeline(tmp_path, browser, backend="browser")
    request = ExtractionRequest(url=URL, schema_id="page_v1")
    result = await first.extract_url(request)
    assert result.status == "partial"
    reopened = pipeline(tmp_path, browser, backend="browser")
    replayed = await reopened.extract_url(request)
    assert replayed.status == "partial"
    assert any(w.startswith("incomplete_source:") for w in replayed.warnings)
    assert browser.calls == 1


def test_prune_retains_crawl_jobs_and_retrieval_ancestors_until_crawl_expires(tmp_path):
    store = JobStore(tmp_path)
    parent = store.get_or_create(ExtractionRequest(url=URL))
    store.update(parent["id"], state="pending", snapshot_id="paid-job")
    child = store.get_or_create(ExtractionRequest(url=URL, schema_id="page_v1"))
    assert child["retrieval_parent_id"] == parent["id"]
    store.update(parent["id"], state="complete")
    store.update(child["id"], state="complete")
    old = time.time() - 40 * 86400
    with store.connect() as db:
        db.executescript("""
            CREATE TABLE crawl_runs(id TEXT PRIMARY KEY, created REAL, updated REAL, body TEXT);
            CREATE TABLE crawl_entries(crawl_id TEXT, job_id TEXT);
            CREATE TABLE crawl_leases(crawl_id TEXT, owner TEXT, expires REAL);
        """)
        db.execute("UPDATE jobs SET updated=?", (old,))
        db.execute(
            "INSERT INTO crawl_runs VALUES(?,?,?,?)", ("crawl", old, old, json.dumps({"status": "pending"}))
        )
        db.execute("INSERT INTO crawl_entries VALUES(?,?)", ("crawl", child["id"]))
    assert store.prune(30) == 0
    assert store.get(parent["id"])["snapshot_id"] == "paid-job"
    with store.connect() as db:
        db.execute("UPDATE crawl_runs SET body=?", (json.dumps({"status": "success"}),))
        db.execute("INSERT INTO crawl_leases VALUES(?,?,?)", ("crawl", "active", time.time() + 60))
    assert store.prune(30) == 0
    with store.connect() as db:
        db.execute("DELETE FROM crawl_leases")
    assert store.prune(30) == 2
    for job in (parent, child):
        with pytest.raises(ScraperError):
            store.get(job["id"])
