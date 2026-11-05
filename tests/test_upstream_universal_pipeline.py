"""End-to-end contract isolation and shared pending retrieval recovery."""

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ai_scraper.config import Settings
from ai_scraper.contracts import InferenceResponse, RetrievalOutcome
from ai_scraper.jobs import JobStore
from ai_scraper.pipeline import Pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.schemas import CustomField, ExtractionRequest

URL = "https://general.test/research"
RECORD = {
    "input": {"url": URL},
    "final_url": URL,
    "page_html": "<html><head><title>Research notes</title></head><body><h1>Research notes</h1><p>Answer: 42</p><p>Reviewed public evidence supports this experimental answer.</p></body></html>",
}


class Retrieval:
    retrieval_method = "direct_http"

    def __init__(self):
        self.calls = 0

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        self.calls += 1
        checkpoint({"state": "fetched", "record": RECORD, "attempts": 1})
        return RetrievalOutcome(
            status="ready", record=RECORD, attempts=1, retrieval_method=self.retrieval_method
        )


class PendingRetrieval:
    retrieval_method = "brightdata_fetch"

    def __init__(self):
        self.submissions = 0
        self.polls = 0
        self.ready = False

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        if not job.get("snapshot_id"):
            self.submissions += 1
            checkpoint({"state": "pending", "snapshot_id": "vendor-shared", "attempts": 1})
        else:
            self.polls += 1
        if not self.ready:
            return RetrievalOutcome(status="pending", snapshot_id="vendor-shared", attempts=1)
        checkpoint({"state": "fetched", "record": RECORD})
        return RetrievalOutcome(status="ready", record=RECORD, snapshot_id="vendor-shared", attempts=1)


class TaskInference:
    provider = "local"
    model = "fixture-local"

    def __init__(self):
        self.calls = []

    def for_schema(self, spec, instruction=None):
        parent = self

        class Scoped:
            async def extract(self, snapshot, feedback=None):
                parent.calls.append((spec.fingerprint, instruction, snapshot.source_hash))
                section = next(item for item in snapshot.sections if "42" in item.text)
                return InferenceResponse(
                    payload={
                        "data": {"answer": "42"},
                        "evidence": [{"field_path": "answer", "section_id": section.id, "excerpt": "42"}],
                        "warnings": [],
                    },
                    usage={"input_tokens": 10, "output_tokens": 20},
                )

        return Scoped()


def pipeline(tmp_path, retrieval=None, inference=None):
    return Pipeline(
        Settings(state_dir=tmp_path, allowed_domains=["*"], retrieval_backend="direct"),
        retriever=retrieval or Retrieval(),
        inference=inference,
        policy=URLPolicy(["*"], resolve_dns=False),
        live=False,
    )


async def test_general_page_runs_without_product_fields_or_inference(tmp_path):
    scraper = pipeline(tmp_path)
    result = await scraper.extract_url(ExtractionRequest(url=URL, schema_id="page_v1"))
    assert result.status == "success"
    assert result.data["title"] == "Research notes"
    assert "Answer: 42" in result.data["text"]
    assert result.provenance.schema_version == "page_v1"
    assert result.provenance.provider is None
    assert result.usage.inference_attempts == 0
    assert Path(result.provenance.snapshot_path).is_file()


async def test_custom_contract_and_instruction_isolate_extraction_cache_but_share_source(tmp_path):
    retrieval, inference = Retrieval(), TaskInference()
    scraper = pipeline(tmp_path, retrieval, inference)
    first = ExtractionRequest(
        url=URL,
        schema_id="custom_v1",
        fields=[CustomField(name="answer", description="The displayed answer")],
    )
    result = await scraper.extract_url(first)
    assert result.status == "success"
    assert result.data == {"answer": "42"}
    repeated = await scraper.extract_url(first)
    assert repeated.status == "success"
    assert len(inference.calls) == 1
    second = first.model_copy(update={"instruction": "Extract the answer printed in the reviewed notes."})
    changed = await scraper.extract_url(second)
    assert changed.status == "success"
    assert len(inference.calls) == 2
    third = first.model_copy(
        update={"fields": [CustomField(name="answer", description="Answer to the research question")]}
    )
    assert (await scraper.extract_url(third)).status == "success"
    assert len(inference.calls) == 3
    assert inference.calls[0][0] != inference.calls[2][0]
    assert retrieval.calls == 1


async def test_changed_task_waits_durably_on_same_vendor_job_across_restart(tmp_path):
    retrieval = PendingRetrieval()
    scraper = pipeline(tmp_path, retrieval)
    original = await scraper.extract_url(ExtractionRequest(url=URL))
    assert original.status == "pending"
    general_request = ExtractionRequest(url=URL, schema_id="page_v1")
    general = await scraper.extract_url(general_request)
    assert general.status == "pending"
    assert general.job_id != original.job_id
    child = scraper.store.get(general.job_id)
    assert child["retrieval_parent_id"] == original.job_id
    assert child["request"]["schema_id"] == "page_v1"
    assert (
        scraper.store.get_or_create(ExtractionRequest(url=URL), {"method": "brightdata_fetch"})["id"]
        == original.job_id
    )
    assert (
        JobStore(tmp_path).get_or_create(general_request, {"method": "brightdata_fetch"})["id"]
        == general.job_id
    )
    retrieval.ready = True
    restarted = pipeline(tmp_path, retrieval)
    completed = await restarted.resume(general.job_id)
    assert completed.status == "success"
    assert completed.provenance.schema_version == "page_v1"
    assert completed.provenance.snapshot_id == "vendor-shared"
    assert completed.data["title"] == "Research notes"
    assert retrieval.submissions == 1
    assert restarted.store.get(original.job_id)["state"] == "fetched"


async def test_unknown_contract_is_rejected_before_retrieval(tmp_path):
    retrieval = Retrieval()
    result = await pipeline(tmp_path, retrieval).extract_url(
        ExtractionRequest(url=URL, schema_id="execute_code")
    )
    assert result.status == "failed"
    assert result.error.code == "UNSUPPORTED_SCHEMA"
    assert retrieval.calls == 0


def test_concurrent_task_registration_keeps_one_child_and_one_retrieval_owner(tmp_path):
    store = JobStore(tmp_path)
    parent = store.get_or_create(ExtractionRequest(url=URL))
    store.update(parent["id"], state="pending", snapshot_id="one-vendor-job")
    request = ExtractionRequest(url=URL, schema_id="page_v1")
    with ThreadPoolExecutor(max_workers=8) as pool:
        children = list(pool.map(lambda _: JobStore(tmp_path).get_or_create(request), range(16)))
    assert len({child["id"] for child in children}) == 1
    assert {child["retrieval_parent_id"] for child in children} == {parent["id"]}


def test_retention_protects_finished_parent_needed_by_waiting_task(tmp_path):
    store = JobStore(tmp_path)
    parent = store.get_or_create(ExtractionRequest(url=URL))
    store.update(parent["id"], state="pending", snapshot_id="still-needed")
    child = store.get_or_create(ExtractionRequest(url=URL, schema_id="page_v1"))
    store.update(parent["id"], state="complete", record=RECORD)
    with store.connect() as db:
        db.execute("UPDATE jobs SET updated=? WHERE id=?", (time.time() - 100 * 86400, parent["id"]))
    assert store.prune(30) == 0
    assert store.get(child["id"])["retrieval_parent_id"] == parent["id"]
    assert store.get(parent["id"])["record"] == RECORD


async def test_execution_reloads_submission_identity_after_acquiring_lease(tmp_path, monkeypatch):
    retrieval = PendingRetrieval()
    retrieval.ready = True
    scraper = pipeline(tmp_path, retrieval)
    stale = scraper.store.get_or_create(ExtractionRequest(url=URL, schema_id="page_v1"))
    acquire = scraper.store.acquire

    def concurrent_submission(job_id, owner, ttl):
        # Another process submitted after this caller read the 'new' body,
        # before this caller obtained exclusive ownership.
        scraper.store.update(job_id, state="pending", snapshot_id="vendor-shared", attempts=1)
        return acquire(job_id, owner, ttl)

    monkeypatch.setattr(scraper.store, "acquire", concurrent_submission)
    result = await scraper._run(stale)
    assert result.status == "success"
    assert retrieval.submissions == 0
    assert retrieval.polls == 1


def test_repeated_adoption_cannot_overwrite_progressed_child_provenance(tmp_path):
    store = JobStore(tmp_path)
    parent = store.get_or_create(ExtractionRequest(url=URL))
    store.update(parent["id"], state="pending", snapshot_id="vendor-shared")
    child = store.get_or_create(ExtractionRequest(url=URL, schema_id="page_v1"))
    parent = store.update(
        parent["id"], state="fetched", record=RECORD, snapshot_retrieved_at="2026-11-05T10:00:00+00:00"
    )
    store.adopt_retrieval(child["id"], parent)
    progressed = store.update(
        child["id"],
        state="complete",
        snapshot_retrieved_at="2026-11-05T10:01:00+00:00",
        render_record={"html": "newer rendered source"},
    )
    assert store.adopt_retrieval(child["id"], parent) == progressed
