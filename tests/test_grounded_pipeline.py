"""Two-stage call budgets and provenance are owned by the application."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from ai_scraper.cli import _settings, parser
from ai_scraper.config import Settings
from ai_scraper.contracts import RetrievalOutcome
from ai_scraper.extraction.grounded import GROUNDED_POLICY_VERSION, GroundedLocalExtractor
from ai_scraper.pipeline import Pipeline, create_pipeline
from ai_scraper.policy import URLPolicy
from ai_scraper.schemas import ExtractionRequest

URL = "https://source.test/record"


def request():
    return ExtractionRequest(
        url=URL,
        schema_id="custom_v1",
        instruction="Extract the main record title.",
        fields=[
            {
                "name": "record_title",
                "type": "string",
                "description": "The exact main record title",
                "critical": True,
            }
        ],
    )


class Retriever:
    calls = 0

    async def retrieve(self, request, job, checkpoint, wait_seconds):
        self.calls += 1
        record = {
            "final_url": URL,
            "markdown": "# Field Notes\nA public reference document.",
            "page_html": "<h1>Field Notes</h1><p>A public reference document.</p>",
        }
        checkpoint({"state": "fetched", "record": record})
        return RetrievalOutcome(status="ready", record=record, retrieval_method="fixture_replay")


async def run(tmp_path, *, bad_grounding_calls=0):
    calls = []

    async def handler(http_request):
        body = json.loads(http_request.content)
        calls.append(body)
        if "selections" in body["response_format"]["json_schema"]["schema"]["properties"]:
            catalog = json.loads(body["messages"][-1]["content"].split("\n", 1)[1])["candidates"]
            seen = sum(
                "selections" in c["response_format"]["json_schema"]["schema"]["properties"] for c in calls
            )
            payload = {
                "selections": {
                    "record_title": None if seen <= bad_grounding_calls else catalog["record_title"][0]["id"]
                }
            }
        else:
            payload = {"record_title": "Field Notes"}
        return httpx.Response(
            200,
            json={
                "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 10},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    inference = GroundedLocalExtractor("test-grounded", client=client)
    fetcher = Retriever()
    settings = Settings(
        state_dir=tmp_path / "state",
        allowed_domains=["source.test"],
        retrieval_backend="direct",
        inference_backend="local",
        local_model="test-grounded",
        local_two_stage_grounding=True,
    )
    pipeline = Pipeline(
        settings,
        retriever=fetcher,
        inference=inference,
        policy=URLPolicy(["source.test"], resolve_dns=False),
        live=False,
    )
    result = await pipeline.extract_url(request())
    return pipeline, client, fetcher, calls, result


async def test_two_calls_have_one_attempt_saved_snapshot_and_application_provenance(tmp_path):
    pipeline, client, fetcher, calls, result = await run(tmp_path)
    try:
        assert result.status == "partial"  # Window excludes other saved representations.
        assert result.data == {"record_title": "Field Notes"}
        assert result.usage.inference_attempts == 1 and result.usage.model_calls == 2
        assert result.usage.input_tokens == 40 and result.usage.output_tokens == 20
        assert result.provenance.prompt_version == GROUNDED_POLICY_VERSION
        assert result.provenance.inference_source.coverage == "incomplete"
        assert result.evidence["record_title"].excerpt.startswith("# Field Notes")
        assert fetcher.calls == 1 and len(calls) == 2
        assert list(pipeline.store.get(result.job_id)["inference_call_reservations"].values()) == [2]
    finally:
        await pipeline.close()
        await client.aclose()


async def test_one_grounding_repair_uses_four_calls_and_never_refetches(tmp_path):
    pipeline, client, fetcher, calls, result = await run(tmp_path, bad_grounding_calls=1)
    try:
        assert result.status == "partial"
        assert result.usage.inference_attempts == 2 and result.usage.model_calls == 4
        assert fetcher.calls == 1 and len(calls) == 4
        assert list(pipeline.store.get(result.job_id)["inference_call_reservations"].values()) == [4]
        assert "Local validation feedback" in calls[2]["messages"][0]["content"]
    finally:
        await pipeline.close()
        await client.aclose()


async def test_rejected_grounding_stops_after_four_calls_and_resume_does_not_repeat(tmp_path):
    pipeline, client, fetcher, calls, result = await run(tmp_path, bad_grounding_calls=10)
    try:
        assert result.status == "failed" and result.error.code == "GROUNDING_REJECTED"
        assert result.usage.model_calls == 4 and len(calls) == 4
        resumed = await pipeline.resume(result.job_id)
        assert resumed.status == "failed" and len(calls) == 4 and fetcher.calls == 1
    finally:
        await pipeline.close()
        await client.aclose()


@pytest.mark.parametrize("other", ["local_source_bound_evidence", "local_source_quote_evidence"])
def test_two_stage_setting_cannot_mix_citation_grammars(other):
    with pytest.raises(ValidationError, match="mutually exclusive"):
        Settings(local_two_stage_grounding=True, **{other: True})


def test_cli_explicit_policy_overrides_env_without_silent_mixing(monkeypatch):
    monkeypatch.setenv("LOCAL_TWO_STAGE_GROUNDING", "true")
    assert (
        _settings(
            parser().parse_args(["replay", "example.json", "--url", URL, "--source-bound-evidence"])
        ).local_two_stage_grounding
        is False
    )
    monkeypatch.setenv("LOCAL_TWO_STAGE_GROUNDING", "false")
    monkeypatch.setenv("LOCAL_SOURCE_BOUND_EVIDENCE", "true")
    settings = _settings(
        parser().parse_args(["replay", "example.json", "--url", URL, "--two-stage-grounding"])
    )
    assert settings.local_two_stage_grounding is True and settings.local_source_bound_evidence is False


async def test_factory_uses_opt_in_adapter_without_fetching(tmp_path):
    pipeline = create_pipeline(
        Settings(
            state_dir=tmp_path / "state",
            retrieval_backend="direct",
            inference_backend="local",
            local_model="selected",
            local_two_stage_grounding=True,
        )
    )
    try:
        assert isinstance(pipeline.inference, GroundedLocalExtractor)
    finally:
        await pipeline.close()
