"""The new target mode remains explicit and retains the actual token mask gates."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from training import train


class Tokenizer:
    def apply_chat_template(self, messages, *, enable_thinking, return_dict, add_generation_prompt=False):
        assert enable_thinking is False
        assert return_dict is False
        return [1, 2, 3, 4, 5, 6] if messages[-1]["role"] == "assistant" else [1, 2, 3]


def row(task="business"):
    return {
        "messages": [
            {"role": "user", "content": "Untrusted source"},
            {"role": "assistant", "content": json.dumps({"title": "Field Notes"})},
        ],
        "grounded": {"task": task},
    }


def validator(monkeypatch, function):
    monkeypatch.setitem(
        sys.modules, "training.realpage_dataset", SimpleNamespace(validate_training_row=function)
    )


def test_opt_in_delegates_schema_and_prompt_validation_for_each_row(monkeypatch):
    checked = []

    def validate(value):
        checked.append(value)
        return value["grounded"]["task"]

    validator(monkeypatch, validate)
    rows = [row("business"), row("grounding")]
    result = train.preflight_rows(Tokenizer(), rows, 6, target_format="realpage_grounded_v1")
    assert checked == rows
    assert result["target_format"] == "realpage_grounded_v1"
    assert result["task_counts"] == {"business": 1, "grounding": 1}
    assert result["supervised_tokens"] == {"min": 3, "p50": 3.0, "p95": 3, "max": 3}
    assert result["truncated_rows"] == 0


def test_no_opt_in_cannot_bypass_old_envelope_with_grounded_metadata():
    value = row()
    value["messages"][-1]["content"] = json.dumps({"data": {}, "evidence": [], "warnings": []})
    with pytest.raises(ValueError, match="target keys"):
        train.preflight_rows(Tokenizer(), [value], 6)


def test_opt_in_cannot_ignore_a_failed_schema_or_prompt_validator(monkeypatch):
    def reject(_):
        raise ValueError("Target does not match the production business schema")

    validator(monkeypatch, reject)
    with pytest.raises(ValueError, match="production business schema"):
        train.preflight_rows(Tokenizer(), [row()], 6, target_format="realpage_grounded_v1")


@pytest.mark.parametrize(
    "processor,max_length,expected",
    [
        (lambda _: ([1, 7, 3, 4, 5, 6], 3), 6, "masked-prefix"),
        (lambda _: ([1, 2, 3, 4, 5, 6], 2), 6, "masked-prefix"),
        (None, 5, "overlength"),
    ],
)
def test_new_targets_still_reject_prefix_drift_prompt_loss_and_truncation(
    monkeypatch, processor, max_length, expected
):
    validator(monkeypatch, lambda _: "business")
    with pytest.raises(ValueError, match=expected):
        train.preflight_rows(
            Tokenizer(), [row()], max_length, processor, target_format="realpage_grounded_v1"
        )


def test_unsupported_format_fails_before_importing_the_curriculum(monkeypatch):
    monkeypatch.delitem(sys.modules, "training.realpage_dataset", raising=False)
    with pytest.raises(ValueError, match="target format"):
        train.preflight_rows(Tokenizer(), [row()], 6, target_format="freeform")
    assert "training.realpage_dataset" not in sys.modules


def test_unrecognized_curriculum_task_cannot_be_counted_as_valid(monkeypatch):
    validator(monkeypatch, lambda _: "provenance")
    with pytest.raises(ValueError, match="Unexpected real-page task"):
        train.preflight_rows(Tokenizer(), [row()], 6, target_format="realpage_grounded_v1")


def test_manifest_opt_in_mismatch_is_rejected_before_optional_model_imports(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"target_format": "citation_envelope_v1"}))
    with pytest.raises(ValueError, match="manifest"):
        train.execute(SimpleNamespace(target_format="realpage_grounded_v1", data=str(tmp_path)))


def test_realpage_manifest_requires_explicit_opt_in_before_optional_model_imports(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"target_format": "realpage_grounded_v1"}))
    with pytest.raises(ValueError, match="manifest"):
        train.execute(SimpleNamespace(data=str(tmp_path)))


def test_opt_in_workflow_hashes_bind_curriculum_and_production_grounding(tmp_path, monkeypatch):
    directory = tmp_path / "training"
    directory.mkdir()
    default_names = ("__init__.py", "train.py", "dataset.py", "job.py", "requirements.lock")
    for name in (*default_names, "realpage_dataset.py"):
        (directory / name).write_text(name)
    production = tmp_path / "src" / "ai_scraper" / "extraction" / "grounded.py"
    production.parent.mkdir(parents=True)
    production.write_text("original production prompt")
    (production.parent / "grounded_view.py").write_text("original production selector")
    monkeypatch.setattr(train, "__file__", str(directory / "train.py"))
    historical = train.workflow_hashes()
    assert set(historical) == set(default_names)
    current = train.workflow_hashes("realpage_grounded_v1")
    assert current.keys() == historical.keys() | {
        "realpage_dataset.py",
        "src/ai_scraper/extraction/grounded.py",
        "src/ai_scraper/extraction/grounded_view.py",
    }
    production.write_text("changed production prompt")
    assert train.workflow_hashes("realpage_grounded_v1") != current
    assert train.workflow_hashes() == historical
