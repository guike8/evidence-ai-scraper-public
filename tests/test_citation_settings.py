import pytest

from ai_scraper.cli import _settings, parser
from ai_scraper.config import Settings


def test_cli_citation_policy_can_explicitly_override_environment(monkeypatch):
    monkeypatch.setenv("LOCAL_SOURCE_BOUND_EVIDENCE", "true")
    args = parser().parse_args(["probe", "--no-source-bound-evidence"])
    assert _settings(args).local_source_bound_evidence is False
    monkeypatch.setenv("LOCAL_SOURCE_BOUND_EVIDENCE", "false")
    args = parser().parse_args(["probe", "--source-bound-evidence"])
    assert _settings(args).local_source_bound_evidence is True


def test_citation_policy_does_not_treat_false_as_truthy(monkeypatch):
    monkeypatch.setenv("LOCAL_SOURCE_BOUND_EVIDENCE", "false")
    assert Settings.from_env().local_source_bound_evidence is False


def test_invalid_citation_policy_setting_fails_explicitly(monkeypatch):
    monkeypatch.setenv("LOCAL_SOURCE_BOUND_EVIDENCE", "maybe")
    with pytest.raises(ValueError):
        Settings.from_env()
