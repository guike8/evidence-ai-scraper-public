"""Inference budgets survive environment/deployment loading before client creation."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ai_scraper.config import Settings
from ai_scraper.extraction.grounded import GroundedLocalExtractor
from ai_scraper.extraction.grounded_view import build_grounded_view
from ai_scraper.normalize import normalize_record
from ai_scraper.pipeline import create_pipeline
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest

SPEC = importlib.util.spec_from_file_location(
    "inference_budget_deployment", Path(__file__).parents[1] / "deploy" / "launch.py"
)
deployment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deployment)

BUDGETS = {"SCRAPER_MAX_SOURCE_CHARS": "8000", "SCRAPER_MAX_OUTPUT_TOKENS": "768"}


def test_environment_loads_exact_measured_budgets_without_changing_defaults(monkeypatch):
    monkeypatch.setattr("os.environ", {})
    defaults = Settings.from_env()
    assert defaults.max_source_chars == 48000
    assert defaults.max_output_tokens == 4096
    for key, value in BUDGETS.items():
        monkeypatch.setenv(key, value)
    settings = Settings.from_env()
    assert settings.max_source_chars == 8000
    assert settings.max_output_tokens == 768
    assert type(settings.max_source_chars) is type(settings.max_output_tokens) is int


@pytest.mark.parametrize("key", BUDGETS)
@pytest.mark.parametrize("value", ["-1", "0", "not-an-integer", "1.5"])
def test_invalid_budget_fails_before_factory_constructs_clients(monkeypatch, key, value):
    monkeypatch.setattr("os.environ", {key: value})

    def unexpected_client(*args, **kwargs):
        pytest.fail("A client was constructed before settings validation")

    monkeypatch.setattr("httpx.AsyncClient", unexpected_client)
    with pytest.raises((ValueError, ValidationError)):
        create_pipeline()


def test_deployment_accepts_budgets_and_rejects_invalid_values_before_cli(tmp_path, monkeypatch, capsys):
    config = tmp_path / "config.json"
    config.write_text(json.dumps(BUDGETS))
    config.chmod(0o600)
    environment = deployment.load_environment(tmp_path, config, {})
    assert all(environment[key] == value for key, value in BUDGETS.items())
    monkeypatch.setattr("os.environ", environment)
    assert Settings.from_env().max_output_tokens == 768
    assert Settings.from_env().max_source_chars == 8000

    def unexpected_cli(*args, **kwargs):
        pytest.fail("Invalid deployment settings reached the CLI")

    monkeypatch.setattr("ai_scraper.cli.main", unexpected_cli)
    for key in BUDGETS:
        for invalid in ("-1", "not-an-integer"):
            config.write_text(json.dumps({**BUDGETS, key: invalid}))
            assert deployment.main(["--root", str(tmp_path), "cli", "schema"]) == 1
            captured = capsys.readouterr()
            assert captured.out == ""
            assert captured.err == (
                "DEPLOY_CONFIG_INVALID: Invalid or unavailable private deployment configuration\n"
            )


@pytest.mark.parametrize("source_chars", [8000, 12000])
async def test_factory_passes_budgets_and_grounded_view_keeps_shared_8000_limit(
    tmp_path, monkeypatch, source_chars
):
    monkeypatch.setattr(
        "os.environ",
        {
            **BUDGETS,
            "SCRAPER_MAX_SOURCE_CHARS": str(source_chars),
            "SCRAPER_STATE_DIR": str(tmp_path / "state"),
            "SCRAPER_RETRIEVAL_BACKEND": "direct",
            "SCRAPER_ALLOWED_DOMAINS": "source.test",
            "SCRAPER_INFERENCE_BACKEND": "local",
            "LOCAL_MODEL": "fixture-grounded-model",
            "LOCAL_TWO_STAGE_GROUNDING": "true",
        },
    )
    pipeline = create_pipeline()
    try:
        assert isinstance(pipeline.inference, GroundedLocalExtractor)
        assert pipeline.inference.max_tokens == 768
        assert pipeline.inference.max_source_chars == source_chars
        request = ExtractionRequest(
            url="https://source.test/record",
            schema_id="custom_v1",
            instruction="Find the project codename.",
            fields=[{"name": "codename", "type": "string", "description": "The project codename"}],
        )
        spec = get_schema(request)
        markdown = "# Project\n\nProject codename: Amber\n\n" + "\n\n".join(
            f"Unrelated archive entry {index}: " + "historical details " * 8 for index in range(150)
        )
        snapshot = normalize_record(
            {"markdown": markdown, "final_url": request.url}, request, retrieval_method="fixture_replay"
        )
        scoped = pipeline.inference.for_request(request, spec)
        actual = scoped._source_content(snapshot)
        expected = build_grounded_view(snapshot, request, spec, 8000)
        assert actual == expected.content
        assert scoped.last_source_view == expected.as_metadata()
        assert len(actual) <= 8000
    finally:
        await pipeline.close()
