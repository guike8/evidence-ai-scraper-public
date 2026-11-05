from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "deployment_launch", Path(__file__).parents[1] / "deploy" / "launch.py"
)
deployment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deployment)


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch):
    monkeypatch.setattr(deployment.os, "environ", dict(deployment.os.environ))


def config_file(tmp_path, values):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values))
    path.chmod(0o600)
    return path


def test_private_config_loads_without_shell_evaluation_or_unrelated_env_mutation(tmp_path):
    malicious = "$(touch /tmp/do-not-execute); `echo secret`"
    path = config_file(tmp_path, {"HF_TOKEN": malicious, "HF_PROVIDER": "cerebras"})
    inherited = {
        "PATH": "/custom/bin",
        "UNRELATED": "keep",
        "HF_TOKEN": "old",
        "SCRAPER_STATE_DIR": "relative",
    }
    environment = deployment.load_environment(tmp_path, path, inherited)
    assert environment["HF_TOKEN"] == malicious
    assert environment["PATH"] == "/custom/bin"
    assert environment["UNRELATED"] == "keep"
    assert environment["SCRAPER_STATE_DIR"] == str(tmp_path / "state")
    assert inherited["HF_TOKEN"] == "old"
    report = json.dumps(deployment.status(environment))
    assert malicious not in report
    assert '"HF_TOKEN": true' in report


@pytest.mark.parametrize("mode", [0o644, 0o640, 0o666])
def test_group_or_public_config_permissions_rejected(tmp_path, mode):
    path = config_file(tmp_path, {"HF_TOKEN": "secret-not-logged"})
    path.chmod(mode)
    with pytest.raises(deployment.ConfigurationError, match="DEPLOY_CONFIG_PERMISSIONS"):
        deployment.load_environment(tmp_path, path, {})


@pytest.mark.parametrize(
    "values", [[], {"PYTHONPATH": "/tmp/injected"}, {"HF_TOKEN": 123}, {"HF_TOKEN": "token\nheader"}]
)
def test_invalid_config_is_secret_safe(tmp_path, values, capsys):
    config_file(tmp_path, values)
    assert deployment.main(["--root", str(tmp_path), "status"]) == 1
    captured = capsys.readouterr()
    assert "DEPLOY_CONFIG_INVALID" in captured.err
    assert "token" not in captured.err
    assert "injected" not in captured.err
    assert captured.out == ""


def test_malformed_json_never_appears_in_error(tmp_path, capsys):
    path = config_file(tmp_path, {})
    path.write_text('{"HF_TOKEN":"secret-malformed"')
    assert deployment.main(["--root", str(tmp_path), "status"]) == 1
    assert "secret-malformed" not in capsys.readouterr().err


@pytest.mark.parametrize("price", ["nan", "inf", "-1", "not-a-price"])
def test_invalid_spending_configuration_rejected_without_value(tmp_path, price):
    path = config_file(tmp_path, {"SCRAPER_MAX_SPEND_USD": price})
    with pytest.raises(deployment.ConfigurationError, match="^DEPLOY_PRICING_INVALID$"):
        deployment.load_environment(tmp_path, path, {})


def test_empty_configuration_reports_missing_without_vendor_calls(tmp_path, capsys):
    config_file(tmp_path, {})
    assert deployment.main(["--root", str(tmp_path), "status"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["live_verified"] is False
    assert report["live_retrieval_configured"] is False
    assert "BRIGHTDATA_API_KEY" in report["missing_for_retrieval"]


def test_cli_arguments_pass_through_existing_main(tmp_path, monkeypatch):
    config_file(tmp_path, {"HF_PROVIDER": "fixture"})
    calls = []
    monkeypatch.setattr("ai_scraper.cli.main", lambda args: calls.append(args) or 2)
    monkeypatch.setattr(deployment.os, "environ", {"PATH": "/preserved"})
    assert deployment.main(["--root", str(tmp_path), "cli", "extract", "https://example.test/product"]) == 2
    assert calls == [["extract", "https://example.test/product"]]
    assert deployment.os.environ["PATH"] == "/preserved"
    assert deployment.os.environ["HF_PROVIDER"] == "fixture"


def test_startup_failure_does_not_print_secret(tmp_path, monkeypatch, capsys):
    config_file(tmp_path, {"HF_TOKEN": "private-secret"})

    def fail(args):
        raise RuntimeError("private-secret")

    monkeypatch.setattr("ai_scraper.cli.main", fail)
    monkeypatch.setattr(deployment.os, "environ", {})
    assert deployment.main(["--root", str(tmp_path), "cli", "schema"]) == 1
    assert capsys.readouterr().err == "DEPLOY_START_FAILED: Application startup failed\n"
