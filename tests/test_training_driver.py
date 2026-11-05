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


@pytest.fixture
def initial_adapter(tmp_path):
    path = tmp_path / "prior-run" / "adapter" / "adapters.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bounded test weights")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (path.parent / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "num_layers": 8, "lora_parameters": train.LORA_PARAMETERS})
    )
    (path.parent.parent / "train.json").write_text(
        json.dumps(
            {
                "adapter_sha256": digest,
                "base": {
                    "repo_id": train.BASE_ID,
                    "revision": train.BASE_REVISION,
                    "files": {"model.safetensors": train.BASE_WEIGHT_SHA256},
                },
            }
        )
    )
    return path, digest


def test_warm_start_requires_exact_weights_config_and_pinned_base_receipt(initial_adapter):
    path, digest = initial_adapter
    identity = train.verify_initial_adapter(path, digest)
    assert identity["adapter_sha256"] == digest and not identity["optimizer_state_loaded"]
    assert str(path.parent) not in json.dumps(identity)
    config_path = path.parent / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["lora_parameters"]["rank"] = 16
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="configuration"):
        train.verify_initial_adapter(path, digest)


def test_initial_adapter_rejects_changed_weights_or_base(initial_adapter):
    path, digest = initial_adapter
    with pytest.raises(ValueError, match="checksum"):
        train.verify_initial_adapter(path, "0" * 64)
    receipt_path = path.parent.parent / "train.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["base"]["files"]["model.safetensors"] = "0" * 64
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="different base"):
        train.verify_initial_adapter(path, digest)


@pytest.mark.parametrize(
    "path,digest", [(None, "a" * 64), ("adapter.safetensors", None), ("file", "invalid")]
)
def test_initial_adapter_is_an_explicit_checksum_pair(path, digest):
    with pytest.raises(ValueError, match="explicit SHA256"):
        train.verify_initial_adapter(path, digest)


def test_initial_adapter_requires_bounded_metadata_and_a_training_receipt(initial_adapter):
    path, digest = initial_adapter
    receipt_path = path.parent.parent / "train.json"
    receipt_path.unlink()
    with pytest.raises(ValueError, match="training receipt"):
        train.verify_initial_adapter(path, digest)
    (path.parent / "adapter_config.json").write_bytes(b" " * 65537)
    with pytest.raises(ValueError, match="byte bound"):
        train.verify_initial_adapter(path, digest)


def test_private_initialization_snapshot_never_tracks_later_source_mutations(initial_adapter, tmp_path):
    path, digest = initial_adapter
    run = tmp_path / "new-run"
    run.mkdir()
    saved = train.snapshot_initial_adapter(path, run, digest)
    assert saved.read_bytes() == path.read_bytes() and saved.stat().st_mode & 0o777 == 0o400
    path.write_bytes(b"changed source")
    assert hashlib.sha256(saved.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="checksum"):
        train.verify_initial_adapter(path, digest)
    saved.chmod(0o600)
    saved.write_bytes(b"changed saved weights")
    with pytest.raises(ValueError, match="between stages"):
        train.snapshot_initial_adapter(path, run, digest)


def test_initialization_metadata_cannot_change_between_preflight_and_sanity(initial_adapter, tmp_path):
    path, digest = initial_adapter
    run = tmp_path / "new-run"
    run.mkdir()
    identity = train.verify_initial_adapter(path, digest)
    train.freeze_initialization_identity(run, identity)
    config_path = path.parent / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["note"] = "Compatible values, different metadata"
    config_path.write_text(json.dumps(config))
    changed = train.verify_initial_adapter(path, digest)
    with pytest.raises(ValueError, match="metadata changed between stages"):
        train.freeze_initialization_identity(run, changed)


class ScalarResult:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class AdapterArray:
    def __init__(self, value=1, shape=(2, 8), dtype="float32"):
        self.value, self.shape, self.dtype = value, shape, dtype
        self.size = 16


class AdapterModel:
    def __init__(self):
        self.parameters = {"layer.self_attn.q_proj.lora_a": AdapterArray(value=0)}

    def trainable_parameters(self):
        return self.parameters

    def load_weights(self, pairs, strict):
        assert strict is False
        self.parameters.update(pairs)


class AdapterMX:
    def __init__(self, arrays):
        self.arrays = arrays

    def load(self, path):
        return self.arrays

    def isfinite(self, array):
        return ScalarResult(array.value != float("inf"))

    def all(self, result):
        return result

    def eval(self, values):
        pass

    def array_equal(self, actual, expected):
        return ScalarResult(actual.value == expected.value and actual.dtype == expected.dtype)


@pytest.mark.parametrize("problem", ["keys", "shape", "dtype", "nonfinite"])
def test_tensor_initialization_checks_actual_wrapped_model_before_loading(tmp_path, problem):
    model = AdapterModel()
    key = next(iter(model.parameters))
    array = AdapterArray()
    if problem == "shape":
        array.shape = (8, 2)
    elif problem == "dtype":
        array.dtype = "float16"
    elif problem == "nonfinite":
        array.value = float("inf")
    arrays = {"unexpected.lora_a" if problem == "keys" else key: array}
    with pytest.raises(ValueError, match="tensor keys, shapes or finite"):
        train.load_initial_adapter(
            model, tmp_path / "weights", AdapterMX(arrays), lambda value: value.items()
        )
    assert model.parameters[key].value == 0


def test_tensor_initialization_proves_exact_reloaded_values(tmp_path):
    model = AdapterModel()
    key = next(iter(model.parameters))
    result = train.load_initial_adapter(
        model, tmp_path / "weights", AdapterMX({key: AdapterArray(7)}), lambda value: value.items()
    )
    assert result == {"tensors": 1, "parameters": 16, "exact": True}
    assert model.parameters[key].value == 7
