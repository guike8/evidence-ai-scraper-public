from __future__ import annotations

import json

import httpx
import pytest

from ai_scraper.retrieval.brightdata import BrightDataRetriever, WebUnlockerRenderer
from ai_scraper.schemas import ExtractionRequest

URL = "https://shop.example/product/hoodie"
REQUEST = ExtractionRequest(url=URL, country="US")
RECORD = {"input": {"url": URL}, "markdown": "# Hoodie\nUSD 42.00", "final_url": URL}


class Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def adapter(handler, **kwargs):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return BrightDataRetriever("private-token", client=client, **kwargs)


def checkpoint(job, path=None):
    def write(changes):
        job.update(changes)
        if path:
            path.write_text(json.dumps(job))

    return write


async def test_200_saves_record_before_return_and_requests_all_representations():
    calls = []

    def handle(request):
        calls.append(request)
        assert request.url.path == "/datasets/v3/scrape"
        assert request.url.params["include_errors"] == "true"
        assert request.url.params["dataset_id"] == "gd_m6gjtfmeh43we6cqc"
        assert json.loads(request.content) == [{"url": URL, "country": "US"}]
        assert request.headers["Authorization"] == "Bearer private-token"
        return httpx.Response(200, json=[RECORD])

    job = {"state": "new"}
    result = await adapter(handle).retrieve(REQUEST, job, checkpoint(job), 5)
    assert result.status == "ready"
    assert result.record == RECORD == job["record"]
    assert job["state"] == "fetched"
    assert len(calls) == result.attempts == 1


async def test_202_polling_download_and_restart_do_not_resubmit(tmp_path):
    calls = []
    clock = Clock()

    def submit(request):
        calls.append(request.method)
        return httpx.Response(202, json={"snapshot_id": "s-123"})

    path = tmp_path / "job.json"
    job = {"state": "new"}
    first = adapter(submit, clock=clock, sleep=clock.sleep)

    # Submission consumes the whole deadline, leaving a durable pending job.
    def save(changes):
        checkpoint(job, path)(changes)
        if changes.get("snapshot_id"):
            clock.now = 2.0

    pending = await first.retrieve(REQUEST, job, save, 1)
    assert pending.status == "pending"
    assert json.loads(path.read_text())["snapshot_id"] == "s-123"
    restarted = json.loads(path.read_text())
    statuses = iter(["starting", "running", "ready"])

    def resume(request):
        calls.append(request.method)
        assert request.method == "GET"
        if "/progress/" in request.url.path:
            return httpx.Response(200, json={"status": next(statuses)})
        assert request.url.path.endswith("/s-123")
        assert request.url.params["format"] == "json"
        return httpx.Response(200, json=[RECORD])

    result = await adapter(resume, clock=clock, sleep=clock.sleep).retrieve(
        REQUEST, restarted, checkpoint(restarted, path), 10
    )
    assert result.status == "ready"
    assert calls.count("POST") == 1
    assert clock.sleeps == [1, 2]
    assert result.snapshot_id == "s-123"


@pytest.mark.parametrize("state", ["submitting", "pending", "reconciliation_required"])
async def test_ambiguous_submission_never_resubmits(state):
    def handle(request):
        pytest.fail("No network request is safe without reconciliation")

    job = {"state": state}
    result = await adapter(handle).retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.status == "pending"
    assert result.error.code == "RECONCILIATION_REQUIRED"


async def test_transport_timeout_on_post_requires_reconciliation():
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("Authorization: private-token", request=request)

    job = {"state": "new"}
    client = adapter(handle)
    result = await client.retrieve(REQUEST, job, checkpoint(job), 5)
    assert result.status == "pending"
    assert job["state"] == "reconciliation_required"
    assert "private-token" not in result.error.model_dump_json()
    await client.retrieve(REQUEST, job, checkpoint(job), 5)
    assert len(calls) == 1


async def test_submission_retry_only_after_explicit_429_and_retry_after():
    calls = []
    clock = Clock()

    def handle(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, json={"error": "rate limit"})
        return httpx.Response(200, json=[RECORD])

    job = {"state": "new"}
    result = await adapter(handle, clock=clock, sleep=clock.sleep).retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.status == "ready"
    assert clock.sleeps == [2]
    assert result.attempts == 2


async def test_retry_after_beyond_deadline_is_persisted_without_sleep():
    clock = Clock()
    job = {"state": "new"}
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(429, headers={"Retry-After": "100"}, json={"error": "rate limit"})

    client = adapter(handle, clock=clock, sleep=clock.sleep)
    result = await client.retrieve(REQUEST, job, checkpoint(job), 10)
    assert result.status == "pending"
    assert job["state"] == "new"
    assert job["retry_not_before"] > 0
    await client.retrieve(REQUEST, job, checkpoint(job), 5)
    assert len(calls) == 1
    assert not clock.sleeps


@pytest.mark.parametrize("status", ["failed", "canceled"])
async def test_terminal_vendor_job_is_saved_and_not_retried(status):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, json={"status": status, "error": "Bad page", "token": "private-token"})

    job = {"state": "pending", "snapshot_id": "s-terminal"}
    client = adapter(handle)
    result = await client.retrieve(REQUEST, job, checkpoint(job), 5)
    assert result.status == "failed"
    assert result.error.code == "FETCH_JOB_FAILED"
    assert "private-token" not in result.error.model_dump_json()
    await client.retrieve(REQUEST, job, checkpoint(job), 5)
    assert len(calls) == 1


async def test_200_record_error_is_failure_even_when_http_success():
    error_record = {
        "input": {"url": URL},
        "error_code": "page_not_found",
        "error": "404",
        "token": "private-token",
    }
    job = {"state": "new"}
    result = await adapter(lambda r: httpx.Response(200, json=[error_record])).retrieve(
        REQUEST, job, checkpoint(job), 5
    )
    assert result.status == "failed"
    assert result.error.code == "FETCH_RECORD_ERROR"
    assert not result.error.retryable
    assert "private-token" not in result.error.model_dump_json()


async def test_per_record_rate_limit_is_classified_without_resubmission():
    record = {"input": {"url": URL}, "error_code": "rate_limit", "error": "retry later"}
    job = {"state": "new"}
    result = await adapter(lambda r: httpx.Response(200, json=[record])).retrieve(
        REQUEST, job, checkpoint(job), 5
    )
    assert result.error.code == "FETCH_RATE_LIMITED"
    assert result.error.retryable
    assert job["state"] == "failed"


async def test_records_are_matched_by_input_url_and_every_error_inspected():
    unrelated_error = {"input": {"url": "https://other.example/"}, "error_code": "404"}
    unrelated_ok = {"input": {"url": "https://third.example/"}, "markdown": "wrong page"}
    job = {"state": "new"}
    result = await adapter(
        lambda r: httpx.Response(200, json=[unrelated_error, unrelated_ok, RECORD])
    ).retrieve(REQUEST, job, checkpoint(job), 5)
    assert result.record == RECORD
    assert "1 error record" in result.warnings[0]


@pytest.mark.parametrize("records", [[{"markdown": "no identity"}], [RECORD, RECORD], []])
async def test_missing_or_duplicate_input_mapping_fails(records):
    job = {"state": "new"}
    result = await adapter(lambda r: httpx.Response(200, json=records)).retrieve(
        REQUEST, job, checkpoint(job), 5
    )
    assert result.status == "failed"
    assert result.error.code == "FETCH_RECORD_MAPPING"


async def test_poll_get_retries_transport_and_rate_limit_without_post():
    calls = []
    clock = Clock()

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        if len(calls) == 1:
            raise httpx.ConnectError("temporary", request=request)
        if len(calls) == 2:
            return httpx.Response(429, headers={"Retry-After": "3"})
        if "/progress/" in request.url.path:
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(200, json=[RECORD])

    job = {"state": "pending", "snapshot_id": "s-safe"}
    result = await adapter(handle, clock=clock, sleep=clock.sleep).retrieve(REQUEST, job, checkpoint(job), 20)
    assert result.status == "ready"
    assert clock.sleeps[1] == 3
    assert result.attempts == len(calls) == 4


async def test_poll_retries_exhausted_remains_resumable():
    job = {"state": "pending", "snapshot_id": "s-safe"}
    clock = Clock()
    result = await adapter(lambda r: httpx.Response(503), clock=clock, sleep=clock.sleep).retrieve(
        REQUEST, job, checkpoint(job), 50
    )
    assert result.status == "pending"
    assert result.error.code == "FETCH_RETRY_EXHAUSTED"
    assert job["state"] == "pending"
    assert result.attempts == 3


async def test_checkpoint_failure_prevents_paid_submission():
    def handle(request):
        pytest.fail("No POST until durable pre-submit checkpoint")

    def fail(changes):
        raise OSError("disk full")

    with pytest.raises(OSError):
        await adapter(handle).retrieve(REQUEST, {"state": "new"}, fail, 5)


async def test_batch_trigger_is_durable_and_uses_distinct_inputs():
    calls = []
    second = ExtractionRequest(url="https://shop.example/second")

    def handle(request):
        calls.append(request)
        assert request.url.path == "/datasets/v3/trigger"
        assert json.loads(request.content) == [{"url": URL, "country": "US"}, {"url": second.url}]
        return httpx.Response(200, json={"snapshot_id": "batch-id"})

    job = {"state": "new"}
    client = adapter(handle)
    result = await client.trigger_batch([REQUEST, second], job, checkpoint(job))
    assert result.status == "pending"
    assert job["snapshot_id"] == "batch-id"
    await client.trigger_batch([REQUEST, second], job, checkpoint(job))
    assert len(calls) == 1


async def test_bounded_renderer_uses_explicit_zone_and_render_flag():
    calls = []

    def handle(request):
        calls.append(request)
        assert request.url.path == "/request"
        assert json.loads(request.content) == {
            "zone": "existing-zone",
            "url": URL,
            "format": "raw",
            "data_format": "markdown",
            "render": "true",
            "country": "US",
        }
        return httpx.Response(200, text="# Rendered hoodie\nUSD 42.00")

    renderer = WebUnlockerRenderer(
        "private-token", "existing-zone", client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    result = await renderer.render(REQUEST)
    assert result.status == "ready"
    assert result.retrieval_method == "brightdata_web_unlocker"
    assert renderer.supports_observed_redirects is False
    assert result.record["observed_final_url"] is False
    assert "final_url" not in result.record
    assert "Observed final URL is unavailable" in result.warnings[0]
    assert result.attempts == len(calls) == 1


async def test_render_timeout_has_no_hidden_retry():
    calls = []

    def handle(request):
        calls.append(request)
        raise httpx.ReadTimeout("private-token", request=request)

    renderer = WebUnlockerRenderer(
        "private-token", "existing-zone", client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    )
    result = await renderer.render(REQUEST)
    assert result.status == "failed"
    assert result.error.code == "RENDER_OUTCOME_UNKNOWN"
    assert len(calls) == 1
    assert "private-token" not in result.error.model_dump_json()


async def test_http_502_submission_is_ambiguous_and_retains_redacted_vendor_details():
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(
            502, json={"message": "private-token backend accepted then failed", "api_key": "private-token"}
        )

    job = {"state": "new"}
    client = adapter(handle)
    result = await client.retrieve(REQUEST, job, checkpoint(job), 5)
    assert result.status == "pending"
    assert result.error.code == "RECONCILIATION_REQUIRED"
    assert result.error.vendor_details["http_status"] == 502
    assert "backend accepted then failed" in result.error.model_dump_json()
    assert "private-token" not in result.error.model_dump_json()
    await client.retrieve(REQUEST, job, checkpoint(job), 5)
    assert len(calls) == 1


async def test_202_without_snapshot_id_needs_reconciliation():
    job = {"state": "new"}
    result = await adapter(lambda r: httpx.Response(202, json={"message": "queued"})).retrieve(
        REQUEST, job, checkpoint(job), 5
    )
    assert result.status == "pending"
    assert result.error.code == "RECONCILIATION_REQUIRED"
    assert result.error.vendor_details["response"]["message"] == "queued"


async def test_401_is_terminal_and_retains_error_without_secret():
    job = {"state": "new"}
    result = await adapter(
        lambda r: httpx.Response(401, json={"error": "invalid key private-token"})
    ).retrieve(REQUEST, job, checkpoint(job), 5)
    assert result.status == "failed"
    assert result.error.code == "FETCH_HTTP_ERROR"
    assert result.error.vendor_details["http_status"] == 401
    assert "private-token" not in result.error.model_dump_json()


@pytest.mark.parametrize("max_attempts", [1, 3])
async def test_get_retry_after_cooldown_survives_deadline_and_restart(tmp_path, monkeypatch, max_attempts):
    clock = Clock()
    wall = [1000.0]
    monkeypatch.setattr("ai_scraper.retrieval.brightdata.time.time", lambda: wall[0])
    calls = []

    def handle(request):
        calls.append(request)
        assert request.method == "GET"
        if len(calls) == 1:
            return httpx.Response(429, headers={"Retry-After": "30"}, json={"error": "rate limit"})
        if "/progress/" in request.url.path:
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(200, json=[RECORD])

    path = tmp_path / "pending-job.json"
    job = {"state": "pending", "snapshot_id": "s-cooldown"}
    first = adapter(handle, clock=clock, sleep=clock.sleep, max_attempts=max_attempts)
    pending = await first.retrieve(REQUEST, job, checkpoint(job, path), 5)
    assert pending.status == "pending"
    assert len(calls) == 1
    assert json.loads(path.read_text())["poll_retry_not_before"] == 1030.0
    if max_attempts == 1:
        assert pending.error.code == "FETCH_RETRY_EXHAUSTED"
    restarted_job = json.loads(path.read_text())
    restarted = adapter(handle, clock=clock, sleep=clock.sleep)
    still_pending = await restarted.retrieve(REQUEST, restarted_job, checkpoint(restarted_job, path), 5)
    assert still_pending.status == "pending"
    assert len(calls) == 1
    assert not clock.sleeps
    assert restarted_job["snapshot_id"] == "s-cooldown"
    wall[0] = 1031.0
    completed = await restarted.retrieve(REQUEST, restarted_job, checkpoint(restarted_job, path), 5)
    assert completed.status == "ready"
    assert len(calls) == 3
    assert restarted_job["poll_retry_not_before"] is None
    assert restarted_job["state"] == "fetched"


async def test_existing_poll_cooldown_waits_within_deadline_before_get(monkeypatch):
    clock = Clock()
    wall = [1000.0]
    monkeypatch.setattr("ai_scraper.retrieval.brightdata.time.time", lambda: wall[0])
    events = []

    async def sleep(seconds):
        events.append(("sleep", seconds))
        clock.now += seconds
        wall[0] += seconds

    def handle(request):
        events.append(("GET", request.url.path))
        assert wall[0] >= 1002.0
        if "/progress/" in request.url.path:
            return httpx.Response(200, json={"status": "ready"})
        return httpx.Response(200, json=[RECORD])

    job = {"state": "pending", "snapshot_id": "s-cooldown", "poll_retry_not_before": 1002.0}
    result = await adapter(handle, clock=clock, sleep=sleep).retrieve(REQUEST, job, checkpoint(job), 5)
    assert result.status == "ready"
    assert events[0] == ("sleep", 2.0)
    assert len(events) == 3
    assert job["poll_retry_not_before"] is None


class CountedBody(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.reads = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk

    async def aclose(self):
        self.closed = True


@pytest.mark.parametrize("operation", ["submit", "poll", "render"])
@pytest.mark.parametrize("declared", [True, False])
async def test_response_byte_cap_stops_headers_or_chunked_body_before_parsing(operation, declared):
    stream = CountedBody([b"[" * 16, b"x" * 17, b"]" * 100_000])
    calls = []

    def handle(request):
        calls.append(request)
        assert request.headers["Accept-Encoding"] == "identity"
        return httpx.Response(
            202 if operation == "submit" else 200,
            headers={"Content-Length": "100033"} if declared else {},
            stream=stream,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle))
    if operation == "render":
        renderer = WebUnlockerRenderer("private-token", "existing-zone", client=client, max_response_bytes=32)
        result = await renderer.render(REQUEST)
    else:
        fetcher = BrightDataRetriever("private-token", client=client, max_response_bytes=32)
        job = {"state": "new"} if operation == "submit" else {"state": "pending", "snapshot_id": "s-cap"}
        result = await fetcher.retrieve(REQUEST, job, checkpoint(job), 5)
        assert job["state"] == "failed"
        resumed = await fetcher.retrieve(REQUEST, job, checkpoint(job), 5)
        assert resumed.error.code == "CONTENT_TOO_LARGE"
    assert result.status == "failed"
    assert result.error.code == "CONTENT_TOO_LARGE"
    assert len(calls) == 1
    assert stream.reads == (0 if declared else 2)
    assert stream.closed
    await client.aclose()


async def test_valid_stream_at_exact_byte_cap_is_accepted():
    body = json.dumps([RECORD]).encode()
    stream = CountedBody([body[:20], body[20:]])
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream)))
    fetcher = BrightDataRetriever("private-token", client=client, max_response_bytes=len(body))
    job = {"state": "new"}
    result = await fetcher.retrieve(REQUEST, job, checkpoint(job), 5)
    assert result.status == "ready"
    assert result.record == RECORD
    assert stream.reads == 2
    assert stream.closed
    await client.aclose()


async def test_unexpected_compression_is_rejected_before_decompression():
    stream = CountedBody([b"untrusted compressed bytes"])
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda r: httpx.Response(200, headers={"Content-Encoding": "gzip"}, stream=stream)
        )
    )
    fetcher = BrightDataRetriever("private-token", client=client)
    job = {"state": "new"}
    result = await fetcher.retrieve(REQUEST, job, checkpoint(job), 5)
    assert result.status == "failed"
    assert result.error.code == "FETCH_INVALID_RESPONSE"
    assert stream.reads == 0
    assert stream.closed
    await client.aclose()


async def test_streaming_helper_total_timeout_closes_slow_response():
    import asyncio

    from ai_scraper.retrieval.brightdata import bounded_request

    class SlowBody(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self):
            await asyncio.sleep(0.1)
            yield b"slow bytes"

        async def aclose(self):
            self.closed = True

    stream = SlowBody()
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream)))
    with pytest.raises(httpx.ReadTimeout):
        await bounded_request(
            client,
            "GET",
            "https://api.brightdata.com/poll",
            max_response_bytes=100,
            timeout=httpx.Timeout(0.01),
        )
    assert stream.closed
    await client.aclose()
