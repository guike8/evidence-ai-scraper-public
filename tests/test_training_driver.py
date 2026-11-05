"""Token boundaries and bounded runs can be verified without loading MLX."""

import hashlib
import json

import pytest

from training import train
from training.train import preflight_rows, run_location_allowed, validate_config


class Tokenizer:
    def apply_chat_template(self, messages, *, enable_thinking, return_dict, add_generation_prompt=False):
        assert enable_thinking is False
        assert return_dict is False
        tokens = [1, 2, 3]
        if messages[-1]["role"] == "assistant":
            tokens += [4, 5, 6]
        return tokens


def row():
    return {
        "messages": [
            {"role": "user", "content": "Source"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {"data": {"active": False, "count": 0}, "evidence": [], "warnings": []}
                ),
            },
        ]
    }


def test_mask_leaves_all_answer_tokens_supervised():
    report = preflight_rows(Tokenizer(), [row()], 6)
    assert report["supervised_tokens"]["min"] == 3
    assert report["prompt_tokens"]["max"] == 3


def test_rejects_trainer_prefix_drift_even_with_same_length():
    with pytest.raises(ValueError, match="masked-prefix"):
        preflight_rows(Tokenizer(), [row()], 6, lambda _: ([1, 7, 3, 4, 5, 6], 3))


def test_rejects_mask_that_includes_prompt_in_loss():
    with pytest.raises(ValueError, match="masked-prefix"):
        preflight_rows(Tokenizer(), [row()], 6, lambda _: ([1, 2, 3, 4, 5, 6], 2))


def test_rejects_silent_truncation():
    with pytest.raises(ValueError, match="overlength"):
        preflight_rows(Tokenizer(), [row()], 5)


def test_provenance_is_never_a_model_training_target():
    value = row()
    target = json.loads(value["messages"][-1]["content"])
    target["provenance"] = {"model": "forged"}
    value["messages"][-1]["content"] = json.dumps(target)
    with pytest.raises(ValueError, match="target keys"):
        preflight_rows(Tokenizer(), [value], 6)


def config(**changes):
    return {
        "iters": 200,
        "grad_accumulation_steps": 4,
        "max_seq_length": 4096,
        "num_layers": 8,
        "batch_size": 1,
        "save_every": 40,
        "learning_rate": 1e-5,
        **changes,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"iters": 201},
        {"iters": 4000},
        {"learning_rate": float("nan")},
        {"max_seq_length": 256},
        {"batch_size": True},
    ],
)
def test_invalid_training_budgets_fail_before_model_load(changes):
    with pytest.raises(ValueError):
        validate_config(config(**changes))


def test_complete_accumulation_groups():
    validate_config(config())


def test_run_receipts_cannot_mutate_the_base_or_dataset(tmp_path):
    base, data = tmp_path / "base", tmp_path / "data"
    assert not run_location_allowed(base, data, base)
    assert not run_location_allowed(base, data, data / "run")
    assert run_location_allowed(base, data, tmp_path / "run")


@pytest.fixture
def pinned_package(tmp_path, monkeypatch):
    package = tmp_path / "mlx_lm"
    package.mkdir()
    digest = hashlib.sha256()
    for index in range(204):
        path = package / f"m{index:03d}.py"
        path.write_text(f"VALUE = {index}\n")
        digest.update(path.name.encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest())
    monkeypatch.setattr(train, "MLX_LM_SOURCE_SHA256", digest.hexdigest())

    class Distribution:
        direct = "{}"  # Offline wheels have no VCS direct_url metadata.

        def read_text(self, name):
            return self.direct

        def locate_file(self, name):
            return package

    distribution = Distribution()
    monkeypatch.setattr(train.importlib.metadata, "distribution", lambda _: distribution)
    return package, distribution


def test_offline_wheel_requires_exact_pinned_source_identity(pinned_package):
    train.verify_training_runtime()
    package, _ = pinned_package
    (package / "m003.py").write_text("VALUE = 'changed'\n")
    with pytest.raises(ValueError, match="sources differ"):
        train.verify_training_runtime()


def test_false_commit_metadata_cannot_claim_the_pinned_revision(pinned_package):
    _, distribution = pinned_package
    distribution.direct = json.dumps({"vcs_info": {"commit_id": "0" * 40}})
    with pytest.raises(ValueError, match="pinned training commit"):
        train.verify_training_runtime()
