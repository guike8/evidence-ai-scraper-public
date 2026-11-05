from __future__ import annotations

import json

import pytest

from ai_scraper.cli import build_request, main, parser
from ai_scraper.config import Settings
from ai_scraper.schemas import ExtractionResult, Provenance


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch, tmp_path):
    for key in (
        "BRIGHTDATA_API_KEY",
        "HF_TOKEN",
        "HF_PROVIDER",
        "HF_MODEL",
        "SCRAPER_ALLOWED_DOMAINS",
        "SCRAPER_MAX_SPEND_USD",
        "BRIGHTDATA_RECORD_USD",
        "HF_INPUT_MILLION_USD",
        "HF_OUTPUT_MILLION_USD",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SCRAPER_STATE_DIR", str(tmp_path / "state"))


def test_schema_discovery_needs_no_vendor_configuration(capsys):
    assert main(["schema", "--schema", "product_v1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema_id"] == "product_v1"
    assert result["required_fields"] == ["name", "price", "currency"]
    assert result["critical_fields"] == ["name"]
    assert result["business_schema"]["additionalProperties"] is False
    assert main(["schema", "--schema", "product_v1", "--provider-compatible"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"data", "evidence", "warnings"}


@pytest.mark.parametrize(
    ("status", "exit_code"), [("success", 0), ("partial", 2), ("pending", 3), ("failed", 1)]
)
def test_cli_status_exit_codes_and_pipeline_cleanup(monkeypatch, capsys, status, exit_code):
    from ai_scraper import pipeline

    closed = []

    class FakePipeline:
        async def extract_url(self, request):
            return ExtractionResult(
                job_id="abc", status=status, provenance=Provenance(requested_url=request.url)
            )

        async def close(self):
            closed.append(True)

    monkeypatch.setattr(pipeline, "create_pipeline", lambda _: FakePipeline())
    assert main(["extract", "https://example.com/product"]) == exit_code
    assert json.loads(capsys.readouterr().out)["status"] == status
    assert closed == [True]


def test_offline_replay_persists_snapshot_and_exports_json_atomically(tmp_path, capsys):
    record = tmp_path / "record.json"
    record.write_text(
        json.dumps(
            {
                "input": {"url": "https://apparel.test/hoodie"},
                "page_html": '<h1 class="product_title">Abominable Hoodie</h1><p class="price">69.00 USD</p>',
            }
        )
    )
    export = tmp_path / "result.json"
    state = tmp_path / "replay-state"
    assert (
        main(
            [
                "replay",
                str(record),
                "--schema",
                "product_v1",
                "--schema",
                "product_v1",
                "--url",
                "https://apparel.test/hoodie",
                "--state-dir",
                str(state),
                "--output",
                str(export),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(export.read_text())
    assert payload["data"]["name"] == "Abominable Hoodie"
    assert payload["data"]["price"] == "69.00"
    assert payload["provenance"]["retrieval_method"] == "fixture_replay"
    assert payload["usage"]["retrieval_attempts"] == 0
    assert payload["usage"]["inference_attempts"] == 0
    assert len(list((state / "snapshots").glob("*.json"))) == 1
    assert export.stat().st_mode & 0o777 == 0o600
    assert not list(tmp_path.glob(".tmp-*"))


def test_bad_input_and_config_errors_do_not_echo_secrets(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SCRAPER_MAX_SPEND_USD", "secret-invalid-number")
    assert main(["extract", "https://example.com"]) == 1
    captured = capsys.readouterr()
    assert "secret-invalid-number" not in captured.out + captured.err
    monkeypatch.delenv("SCRAPER_MAX_SPEND_USD")
    broken = tmp_path / "record.json"
    broken.write_text('{"not valid JSON secret"')
    assert main(["replay", str(broken), "--url", "https://example.com"]) == 1
    captured = capsys.readouterr()
    assert "not valid JSON secret" not in captured.out + captured.err
    assert "Traceback" not in captured.err


def test_invalid_url_is_structured_failure_without_fetch(capsys):
    assert main(["extract", "file:///etc/passwd"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_probe_without_credentials_or_budget_makes_no_call(capsys, monkeypatch):
    assert main(["probe"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "MISSING_CREDENTIALS"
    monkeypatch.setenv("HF_TOKEN", "secret-value")
    monkeypatch.setenv("HF_PROVIDER", "cerebras")
    monkeypatch.setenv("HF_MODEL", "Qwen/Qwen3.8-27B")
    assert main(["probe"]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["error"]["code"] == "BUDGET_UNCONFIGURED"
    assert "secret-value" not in captured.out + captured.err


def test_resume_passes_vendor_reconciliation_and_always_closes(monkeypatch, capsys):
    from ai_scraper import pipeline

    seen = []

    class FakePipeline:
        async def resume(self, job_id, snapshot_id=None):
            seen.append((job_id, snapshot_id))
            return ExtractionResult(
                job_id=job_id, status="pending", provenance=Provenance(requested_url="https://example.com")
            )

        async def close(self):
            seen.append("closed")

    monkeypatch.setattr(pipeline, "create_pipeline", lambda _: FakePipeline())
    assert main(["resume", "abc", "--snapshot-id", "vendor123"]) == 3
    assert seen == [("abc", "vendor123"), "closed"]
    capsys.readouterr()


def test_state_directory_can_precede_or_follow_subcommand(tmp_path):
    before = parser().parse_args(["--state-dir", str(tmp_path), "schema"])
    after = parser().parse_args(["schema", "--state-dir", str(tmp_path)])
    assert before.state_dir == after.state_dir == tmp_path


def test_probe_reserves_before_live_call_and_records_measured_usage(monkeypatch, tmp_path, capsys):
    from pydantic import SecretStr

    from ai_scraper import cli
    from ai_scraper.jobs import JobStore

    settings = Settings(
        state_dir=tmp_path / "state",
        hf_token=SecretStr("secret"),
        hf_provider="cerebras",
        hf_model="Qwen/Qwen3.8-27B",
        max_spend_usd=1.0,
        hf_input_million_usd=0.5,
        hf_output_million_usd=1.0,
    )
    seen = []

    class FakeInference:
        def __init__(self, *args, **kwargs):
            self.response_format = {"type": "json_schema"}

        async def probe(self):
            assert JobStore(settings.state_dir).reserved_spend() > 0
            seen.append("called")
            return {"passed": True, "usage": {"input_tokens": 100, "output_tokens": 10, "latency_ms": 8.0}}

        async def close(self):
            seen.append("closed")

    monkeypatch.setattr(cli, "_settings", lambda _: settings)
    monkeypatch.setattr(cli, "HuggingFaceExtractor", FakeInference)
    assert main(["probe"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["usage"]["cost_estimate_usd"] == 0.00006
    assert payload["reserved_usd"] > payload["usage"]["cost_estimate_usd"]
    assert seen == ["called", "closed"]


def test_evaluation_failed_gate_is_nonzero_and_live_benchmark_skip_is_explicit(monkeypatch, capsys, tmp_path):
    from ai_scraper import evaluation

    monkeypatch.setattr(evaluation, "evaluate_manifest", lambda _: {"gates": {"passed": False}})
    assert main(["evaluate", str(tmp_path / "manifest.json")]) == 1
    assert json.loads(capsys.readouterr().out)["gates"]["passed"] is False
    assert main(["benchmark", str(tmp_path / "manifest.json")]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "not_run"
    assert payload["missing_configuration"]


def test_universal_default_and_schema_discovery_need_no_configuration(capsys):
    assert main(["schema"]) == 0
    schema = json.loads(capsys.readouterr().out)
    assert schema["schema_id"] == "page_v1"
    assert schema["required_fields"] == ["text"]
    assert main(["schemas"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["default_schema_id"] == "page_v1"
    assert {
        "page_v1",
        "article_v1",
        "person_v1",
        "organization_v1",
        "job_v1",
        "event_v1",
        "listing_v1",
        "product_v1",
        "custom_v1",
    } == {item["schema_id"] for item in listing["schemas"]}
    assert main(["schema", "article", "--provider-compatible"]) == 0
    contract = json.loads(capsys.readouterr().out)
    assert contract["additionalProperties"] is False
    assert "ArticleV1" in contract["$defs"]


def test_build_request_handles_typed_shorthand_goal_only_and_explicit_presets():
    from ai_scraper.contracts import ScraperError

    req = build_request("https://example.com")
    assert req.schema_id == "page_v1" and not req.fields
    req = build_request("https://example.com", fields="headline,year:integer,valid:boolean,tags:string_list")
    assert req.schema_id == "custom_v1"
    assert [(field.name, field.type) for field in req.fields] == [
        ("headline", "string"),
        ("year", "integer"),
        ("valid", "boolean"),
        ("tags", "string_list"),
    ]
    goal = "Extract the visible accessibility features and opening times."
    req = build_request("https://example.com", schema_id="auto", instruction=goal)
    assert req.schema_id == "custom_v1" and req.instruction == goal
    assert req.fields[0].name == "matches" and req.fields[0].type == "string_list"
    assert req.fields[0].description == goal
    req = build_request("https://example.com", schema_id="article", instruction="Focus on the author")
    assert req.schema_id == "article_v1" and req.instruction == "Focus on the author"
    with pytest.raises(ScraperError):
        build_request("https://example.com", schema_id="product_v1", fields="title")


def test_custom_schema_shorthand_and_descriptor_file(capsys, tmp_path):
    assert main(["schema", "--fields", "headline,year:integer"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_id"] == "custom_v1"
    assert set(payload["business_schema"]["properties"]) == {"headline", "year"}
    path = tmp_path / "fields.json"
    path.write_text(
        json.dumps(
            [
                {"name": "topics", "type": "string_list", "required": False},
                {"name": "count", "type": "integer", "critical": True},
            ]
        )
    )
    assert main(["schema", "custom_v1", "--fields-file", str(path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["required_fields"] == payload["critical_fields"] == ["count"]
    assert main(["schema", "--schema", "article_v1"]) == 0
    assert json.loads(capsys.readouterr().out)["schema_id"] == "article_v1"


@pytest.mark.parametrize("fields", ["", "name,,year", "amount:executable", "name,name", "model_dump"])
def test_invalid_custom_shorthand_is_structured_failure(fields, capsys):
    assert main(["schema", "--fields", fields]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed" and payload["error"]["code"] == "INVALID_REQUEST"


def test_custom_field_file_shape_size_and_conflicting_inputs_are_bounded(tmp_path, capsys):
    path = tmp_path / "fields.json"
    path.write_text(json.dumps({"name": "unexpected"}))
    assert main(["schema", "--fields-file", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "INVALID_REQUEST"
    path.write_text(" " * 65_537)
    assert main(["schema", "--fields-file", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "INVALID_REQUEST"
    assert main(["schema", "--fields", "title", "--fields-file", str(path)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "INVALID_REQUEST"
    assert main(["schema", "article_v1", "--schema", "product_v1"]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "INVALID_REQUEST"


def test_cli_forwards_typed_custom_request_and_ask_goal_without_dropping_fields(monkeypatch, capsys):
    from ai_scraper import pipeline

    seen = []

    class FakePipeline:
        async def extract_url(self, request):
            seen.append(request)
            return ExtractionResult(
                job_id="custom",
                status="partial",
                data={"matches": []},
                provenance=Provenance(requested_url=request.url, schema_version=request.schema_id),
            )

        async def close(self):
            seen.append("closed")

    monkeypatch.setattr(pipeline, "create_pipeline", lambda _: FakePipeline())
    assert (
        main(
            ["extract", "https://example.com", "--fields", "year:integer", "--ask", "Only the displayed year"]
        )
        == 2
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["provenance"]["schema_version"] == "custom_v1"
    assert seen[0].fields[0].name == "year" and seen[0].fields[0].type == "integer"
    assert seen[0].instruction == "Only the displayed year" and seen[1] == "closed"
    assert main(["extract", "https://example.com", "--ask", "Which dates are listed?"]) == 2
    capsys.readouterr()
    assert seen[2].fields[0].name == "matches" and seen[2].instruction == "Which dates are listed?"
