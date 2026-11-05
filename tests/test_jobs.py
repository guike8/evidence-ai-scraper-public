from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from ai_scraper.contracts import ScraperError
from ai_scraper.jobs import JobStore
from ai_scraper.normalize import normalize_record
from ai_scraper.schemas import ExtractionRequest, ExtractionResult, ProductV1, Provenance


def request(path="one", host="shop.test", **kwargs):
    return ExtractionRequest(url=f"https://{host}/product/{path}", **kwargs)


def test_create_is_atomic_across_independent_store_instances(tmp_path):
    root = tmp_path / "state"
    JobStore(root)

    def create(_):
        return JobStore(root).get_or_create(request())["id"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(create, range(16)))
    assert len(set(ids)) == 1
    assert JobStore(root).get(ids[0])["request"]["url"] == request().url


def test_pending_work_survives_restart_and_request_freshness_changes(tmp_path):
    store = JobStore(tmp_path)
    job = store.get_or_create(request())
    store.update(job["id"], state="pending", snapshot_id="vendor-existing")
    reopened = JobStore(tmp_path)
    reused = reopened.get_or_create(request(freshness_seconds=0))
    assert reused["id"] == job["id"]
    assert reused["snapshot_id"] == "vendor-existing"
    assert reused["state"] == "pending"


def test_completed_freshness_cache_and_expiry(tmp_path):
    store = JobStore(tmp_path)
    req = request(freshness_seconds=10)
    first = store.get_or_create(req)
    store.update(first["id"], state="complete")
    assert store.get_or_create(req)["id"] == first["id"]
    store.update(first["id"], retrieval_created=time.time() - 20)
    with store.connect() as db:
        db.execute("UPDATE jobs SET created=? WHERE id=?", (time.time() - 20, first["id"]))
    assert store.get_or_create(req)["id"] != first["id"]


def test_global_limit_is_shared_and_atomic_across_connections(tmp_path):
    store = JobStore(tmp_path)
    jobs = [store.get_or_create(request(host=f"shop{i}.test")) for i in range(8)]

    def acquire(job):
        return JobStore(tmp_path).acquire(job["id"], f"worker-{job['id']}")

    with ThreadPoolExecutor(max_workers=8) as pool:
        acquired = list(pool.map(acquire, jobs))
    assert sum(acquired) == 4


def test_one_job_per_host_and_release_requires_correct_owner(tmp_path):
    store = JobStore(tmp_path)
    first = store.get_or_create(request("first"))
    second = store.get_or_create(request("second"))
    assert store.acquire(first["id"], "owner-a")
    assert not store.acquire(second["id"], "owner-b")
    store.release(first["id"], "wrong-owner")
    assert not store.acquire(second["id"], "owner-b")
    store.release(first["id"], "owner-a")
    assert store.acquire(second["id"], "owner-b")


def test_expired_leases_are_recoverable_after_process_exit(tmp_path):
    store = JobStore(tmp_path)
    job = store.get_or_create(request())
    assert store.acquire(job["id"], "crashed", ttl=10)
    with store.connect() as db:
        db.execute("UPDATE leases SET expires=? WHERE job_id=?", (time.time() - 1, job["id"]))
    assert JobStore(tmp_path).acquire(job["id"], "restarted")


def test_live_budget_reservations_are_atomic_and_idempotent(tmp_path):
    store = JobStore(tmp_path)

    def reserve(i):
        try:
            JobStore(tmp_path).reserve(f"paid-{i}", 0.4, 1.0, "fixture-unit-price")
            return True
        except ScraperError as exc:
            assert exc.error.code == "BUDGET_EXHAUSTED"
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        admitted = list(pool.map(reserve, range(8)))
    assert sum(admitted) == 2
    assert store.reserved_spend() == pytest.approx(0.8)
    accepted_id = f"paid-{admitted.index(True)}"
    store.reserve(accepted_id, 0.4, 1.0, "same-operation-after-restart")
    assert JobStore(tmp_path).reserved_spend() == pytest.approx(0.8)


def test_live_budget_is_required_before_any_reservation(tmp_path):
    store = JobStore(tmp_path)
    with pytest.raises(ScraperError) as caught:
        store.reserve("unconfigured", 0.01, None, "test")
    assert caught.value.error.code == "BUDGET_UNCONFIGURED"
    assert store.reserved_spend() == 0


def test_snapshot_is_private_immutable_and_round_trips(tmp_path):
    store = JobStore(tmp_path)
    req = request()
    job = store.get_or_create(req)
    snapshot = normalize_record({"input": {"url": req.url}, "markdown": "Hoodie\nUSD 42.00"}, req)
    path = store.save_snapshot(job["id"], snapshot)
    assert json.loads(path.read_text())["raw_record"]["markdown"] == "Hoodie\nUSD 42.00"
    assert store.save_snapshot(job["id"], snapshot) == path
    assert os.stat(path).st_mode & 0o777 == 0o600
    assert os.stat(path.parent).st_mode & 0o777 == 0o700
    assert os.stat(store.path).st_mode & 0o777 == 0o600
    changed = snapshot.model_copy(deep=True)
    changed.raw_record["markdown"] = "Changed content with the same claimed hash"
    with pytest.raises(ScraperError) as caught:
        store.save_snapshot(job["id"], changed)
    assert caught.value.error.code == "SNAPSHOT_CONFLICT"
    assert json.loads(path.read_text())["raw_record"]["markdown"] == "Hoodie\nUSD 42.00"


def test_cached_extraction_survives_restart(tmp_path):
    store = JobStore(tmp_path)
    payload = {"data": {"name": "Hoodie"}, "evidence": [], "warnings": []}
    store.cache_put("hash-schema-prompt-provider-model", payload)
    assert JobStore(tmp_path).cache_get("hash-schema-prompt-provider-model") == payload
    assert JobStore(tmp_path).cache_get("different-model") is None


def test_retention_removes_only_old_terminal_work_and_cache(tmp_path):
    store = JobStore(tmp_path)
    req = request()
    job = store.get_or_create(req)
    snapshot = normalize_record({"markdown": "Hoodie"}, req)
    snapshot_path = store.save_snapshot(job["id"], snapshot)
    result = ExtractionResult(
        job_id=job["id"],
        status="partial",
        data=ProductV1(name="Hoodie"),
        provenance=Provenance(requested_url=req.url),
    )
    store.save_result(result)
    store.update(job["id"], state="complete")
    result_path = store.get(job["id"])["result_path"]
    pending = store.get_or_create(request("pending"))
    store.update(pending["id"], state="pending", snapshot_id="still-running")
    store.cache_put("old-derived-data", {"name": "Hoodie"})
    with store.connect() as db:
        db.execute("UPDATE jobs SET updated=?", (time.time() - 40 * 86400,))
    assert store.prune(30) == 1
    assert not snapshot_path.exists()
    assert not os.path.exists(result_path)
    assert store.get(pending["id"])["snapshot_id"] == "still-running"
    assert store.cache_get("old-derived-data") is None
    with pytest.raises(ScraperError) as caught:
        store.get(job["id"])
    assert caught.value.error.code == "JOB_NOT_FOUND"


def test_redaction_preserves_usage_token_counts_and_removes_credentials():
    from ai_scraper.storage import redact

    payload = {
        "usage": {"input_tokens": 123, "output_tokens": 45},
        "token": "private",
        "hf_token": "private",
        "authorization": "Bearer private",
        "details": "https://mcp.example/?token=private&x=1",
    }
    cleaned = redact(payload, ("private",))
    assert cleaned["usage"] == {"input_tokens": 123, "output_tokens": 45}
    assert cleaned["token"] == cleaned["hf_token"] == cleaned["authorization"] == "[REDACTED]"
    assert "private" not in json.dumps(cleaned)


@pytest.mark.parametrize(
    "amount,limit", [(-1, 1), (float("nan"), 1), (float("inf"), 1), (0.1, float("nan")), (0.1, -1)]
)
def test_nonfinite_or_negative_spend_is_rejected(tmp_path, amount, limit):
    store = JobStore(tmp_path)
    with pytest.raises(ScraperError) as caught:
        store.reserve("bad-amount", amount, limit, "test")
    assert caught.value.error.code == "INVALID_BUDGET"
    assert store.reserved_spend() == 0
