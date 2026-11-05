"""Exercise generation orchestration without loading or running a real model."""

import copy
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from training import infer
from training.dataset import build_case


class Tokenizer:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((copy.deepcopy(messages), kwargs))
        return [10, 20, 30]


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    base, data, run = (tmp_path / name for name in ("base", "data", "run"))
    base.mkdir()
    data.mkdir()
    (base / "model.safetensors").write_bytes(b"fake-base")
    options = infer.parser().parse_args(
        ["--model", str(base), "--data", str(data), "--run-dir", str(run), "--max-tokens", "128"]
    )
    cases = []
    for split in ("valid", "test", "train"):
        rows, split_cases = zip(*(build_case(split, "custom_v1", index) for index in range(2)), strict=True)
        (data / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        (data / f"{split}.cases.jsonl").write_text("\n".join(json.dumps(case) for case in split_cases) + "\n")
        if split == "valid":
            cases = list(split_cases)
    monkeypatch.setattr(
        infer,
        "verify_base",
        lambda path: {
            "repo_id": infer.BASE_ID,
            "revision": infer.BASE_REVISION,
            "files": {"model.safetensors": infer.file_hash(path / "model.safetensors")},
        },
    )
    return options, cases


def _runtime(cases, calls, *, failure=False):
    tokenizer = Tokenizer()

    def generate(prefix, max_tokens, check_deadline):
        check_deadline()
        calls.append((prefix, max_tokens))
        if failure:
            raise RuntimeError("A private input must not appear in any output log")
        text = json.dumps(cases[len(calls) - 1]["expected_candidate"])
        yield SimpleNamespace(text=text, generation_tokens=41, finish_reason="stop")

    return lambda _: (tokenizer, generate, {"test_double": True}), tokenizer


def test_prompt_prefix_matches_production_and_has_no_assistant_target(experiment):
    options, cases = experiment
    rows = infer.read_jsonl(Path(options.data) / "valid.jsonl")
    tokenizer = Tokenizer()
    assert infer.render_prefix(tokenizer, cases[0], rows[0], 4096) == [10, 20, 30]
    messages, kwargs = tokenizer.calls[0]
    assert messages == rows[0]["messages"][:-1]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert kwargs == {"add_generation_prompt": True, "enable_thinking": False, "return_dict": False}
    changed = copy.deepcopy(rows[0])
    changed["messages"][0]["content"] += "changed"
    with pytest.raises(ValueError):
        infer.render_prefix(tokenizer, cases[0], changed, 4096)


def test_prefix_rejects_overlength_instead_of_truncating(experiment):
    _, cases = experiment
    row = {"messages": infer.production_messages(cases[0]) + [{"role": "assistant"}]}
    with pytest.raises(ValueError):
        infer.render_prefix(Tokenizer(), cases[0], row, 2)


def test_direct_generation_and_resume_preserve_identity_without_duplicate_work(experiment):
    options, cases = experiment
    calls = []
    loader, tokenizer = _runtime(cases, calls)
    receipt = infer.execute(options, runtime_loader=loader)
    assert receipt["accepted_cases"] == 2
    assert receipt["run_status"] == "completed"
    assert receipt["inference"]["response_format"] == "none"
    assert receipt["grammar_verification"].startswith("not_performed")
    assert receipt["runtime_controls"] == {"test_double": True}
    assert len(calls) == 2
    assert all(max_tokens == options.max_tokens for _, max_tokens in calls)
    checkpoint = Path(options.run_dir) / "outputs.json"
    assert checkpoint.stat().st_mode & 0o777 == 0o600
    resumed = infer.execute(
        options, runtime_loader=lambda _: pytest.fail("Completed cases must not load a model")
    )
    assert resumed["prompt_hashes_hash"] == receipt["prompt_hashes_hash"]
    assert len(calls) == 2
    options.max_tokens += 1
    with pytest.raises(ValueError, match="Checkpoint"):
        infer.execute(options, runtime_loader=loader)
    assert len(tokenizer.calls) == 2


def test_test_split_stays_unopened_until_matching_candidate_is_frozen(experiment, monkeypatch):
    options, cases = experiment
    options.split = "test"
    original = infer.read_jsonl
    reads = []
    monkeypatch.setattr(infer, "read_jsonl", lambda path: (reads.append(path), original(path))[1])
    with pytest.raises(ValueError, match="Freeze"):
        infer.prepare(options)
    assert reads == []
    options.split = "valid"
    options.freeze_candidate = str(Path(options.run_dir).parent / "frozen.json")
    loader, _ = _runtime(cases, [])
    infer.execute(options, runtime_loader=loader)
    frozen = json.loads(Path(options.freeze_candidate).read_text())
    assert frozen["test_results_used"] is False
    options.frozen_candidate = options.freeze_candidate
    options.freeze_candidate = None
    options.split = "test"
    assert infer.prepare(options)["selection"]["split"] == "test"
    reads.clear()
    options.max_tokens += 1
    with pytest.raises(ValueError, match="Frozen"):
        infer.prepare(options)
    assert reads == []


def test_errors_are_sanitized_and_all_cases_remain_in_denominator(experiment):
    options, cases = experiment
    loader, _ = _runtime(cases, [], failure=True)
    receipt = infer.execute(options, runtime_loader=loader)
    assert receipt["case_count"] == 2
    assert receipt["accepted_cases"] == 0
    assert receipt["raw_facts"]["present"] == sum(len(case["expected_fields"]) for case in cases)
    assert receipt["diagnostics"]["errors"] == {"LOCAL_GENERATION_FAILED": 2}
    assert "private input" not in (Path(options.run_dir) / "outputs.json").read_text()


def test_truncation_and_memory_failures_are_distinct(experiment):
    options, _ = experiment
    counter = []

    def generate(*_):
        counter.append(1)
        if len(counter) == 2:
            raise MemoryError("Do not log this allocation error")
        yield SimpleNamespace(text="{}", generation_tokens=options.max_tokens, finish_reason="length")

    receipt = infer.execute(options, runtime_loader=lambda _: (Tokenizer(), generate, {}))
    assert receipt["diagnostics"]["token_limit_cases"] == 1
    assert receipt["diagnostics"]["errors"] == {"INVALID_SCHEMA": 1, "LOCAL_MEMORY_LIMIT": 1}
    assert receipt["tokens"]["output_tokens"] == {"total": options.max_tokens, "measured_cases": 1}


def test_expired_wall_budget_records_every_unattempted_case(experiment):
    options, _ = experiment
    options.deadline_monotonic = 1
    receipt = infer.execute(options, runtime_loader=lambda _: (Tokenizer(), None, {}), clock=lambda: 2)
    assert receipt["run_status"] == "timed_out"
    assert receipt["case_count"] == 2
    assert receipt["missing_outputs"] == 0
    assert receipt["diagnostics"]["errors"] == {"RUN_WALL_TIMEOUT": 2}


def test_diagnostic_subsets_are_explicit(experiment):
    options, _ = experiment
    options.split = "train"
    with pytest.raises(ValueError, match="explicit limit"):
        infer.prepare(options)
    options.max_cases = 1
    receipt = infer.execute(options, runtime_loader=lambda _: (Tokenizer(), lambda *_: iter(()), {}))
    assert receipt["selection"]["is_subset"] is True
    assert receipt["selection"]["split_case_count"] == 2
    assert receipt["case_count"] == 1


def test_candidate_artifact_mutation_invalidates_resume_identity(experiment):
    options, cases = experiment
    options.arm = "lora"
    artifact = Path(options.run_dir).parent / "adapter"
    artifact.mkdir()
    (artifact / "adapters.safetensors").write_bytes(b"adapter-one")
    (artifact / "adapter_config.json").write_text("{}")
    options.artifact = str(artifact)
    loader, _ = _runtime(cases, [])
    infer.execute(options, runtime_loader=loader)
    (artifact / "adapters.safetensors").write_bytes(b"adapter-two")
    with pytest.raises(ValueError, match="Checkpoint"):
        infer.execute(options, runtime_loader=loader)


def test_outputs_never_write_into_base_or_candidate_artifacts(experiment):
    options, _ = experiment
    options.freeze_candidate = str(Path(options.model) / "freeze.json")
    with pytest.raises(ValueError, match="freeze output"):
        infer.prepare(options)
    options.freeze_candidate = None
    options.run_dir = str(Path(options.model) / "run")
    with pytest.raises(ValueError, match="inference output"):
        infer.supervise(options, [])
    assert not Path(options.run_dir).exists()


def test_hard_deadline_supervisor_only_stops_its_created_child(experiment, monkeypatch):
    options, _ = experiment
    observed = {}
    forbidden = (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "OPENAI_API_KEY",
        "BRIGHTDATA_API_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "SSH_AUTH_SOCK",
        "SCRAPER_TEST_SECRET",
    )
    for key in forbidden:
        monkeypatch.setenv(key, "credential-isolation-sentinel")
    monkeypatch.setenv("LANG", "C")

    class Child:
        pid = 123456789

        def wait(self, timeout):
            raise subprocess.TimeoutExpired("test-child", timeout)

    def spawn(command, **kwargs):
        observed.update(command=command, kwargs=kwargs)
        return Child()

    stopped = []
    monkeypatch.setattr(infer.subprocess, "Popen", spawn)
    monkeypatch.setattr(infer, "_stop_owned_group", lambda child: stopped.append(child))
    receipt = infer.supervise(options, ["--model", options.model])
    assert receipt["run_status"] == "timed_out"
    assert receipt["diagnostics"]["errors"] == {"RUN_WALL_TIMEOUT": 2}
    assert len(stopped) == 1
    assert observed["kwargs"]["start_new_session"] is True
    assert observed["kwargs"]["env"]["HF_HUB_OFFLINE"] == "1"
    assert observed["kwargs"]["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert observed["kwargs"]["env"]["LANG"] == "C"
    assert not (set(forbidden) & observed["kwargs"]["env"].keys())
    assert "credential-isolation-sentinel" not in observed["kwargs"]["env"].values()
    assert observed["kwargs"]["stdout"] == subprocess.DEVNULL
    assert "--deadline-monotonic" in observed["command"]


def test_real_hard_deadline_terminates_only_the_new_owned_process(experiment, monkeypatch):
    options, _ = experiment
    options.wall_seconds = 1
    real_popen = subprocess.Popen
    children = []

    def spawn(_command, **kwargs):
        child = real_popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(infer.subprocess, "Popen", spawn)
    receipt = infer.supervise(options, [])
    assert receipt["run_status"] == "timed_out"
    assert children[0].poll() is not None
    assert children[0].returncode < 0


def test_runtime_loader_uses_local_only_base_and_explicit_adapter(experiment, monkeypatch):
    options, _ = experiment
    options.arm = "lora"
    options.artifact = str(Path(options.run_dir).parent / "adapter")
    observations = []
    fake_mlx = ModuleType("mlx")
    fake_core = ModuleType("mlx.core")
    fake_core.set_memory_limit = lambda value: observations.append(("memory", value))
    fake_core.set_cache_limit = lambda _: None
    fake_core.random = SimpleNamespace(seed=lambda _: None)
    fake_lm = ModuleType("mlx_lm")
    model = SimpleNamespace(
        freeze=lambda: observations.append("frozen"), eval=lambda: observations.append("eval")
    )

    def load(*args, **kwargs):
        observations.append((args, kwargs))
        return model, Tokenizer()

    fake_lm.load = load
    fake_lm.stream_generate = lambda *args, **kwargs: None
    fake_sample = ModuleType("mlx_lm.sample_utils")
    fake_sample.make_sampler = lambda **kwargs: observations.append(("sampler", kwargs))
    for key, value in (
        ("mlx", fake_mlx),
        ("mlx.core", fake_core),
        ("mlx_lm", fake_lm),
        ("mlx_lm.sample_utils", fake_sample),
    ):
        monkeypatch.setitem(sys.modules, key, value)
    monkeypatch.setattr(infer, "verify_training_runtime", lambda: observations.append("pinned-runtime"))
    monkeypatch.setattr(infer, "verify_loaded_adapter", lambda model, path: {"exact_tensor_reload": True})
    infer.load_runtime(options)
    assert observations[0] == "pinned-runtime"
    assert ("memory", 8 * 1024**3) in observations
    load = next(item for item in observations if isinstance(item, tuple) and isinstance(item[0], tuple))
    assert load[0] == (str(Path(options.model).resolve()),)
    assert load[1]["trust_remote_code"] is False
    assert load[1]["adapter_path"] == str(Path(options.artifact).resolve())
    assert ("sampler", {"temp": 0}) in observations
    assert "frozen" in observations and "eval" in observations


def test_adapter_reload_requires_every_saved_tensor_shape_dtype_and_value():
    key = "model.layers.0.self_attn.q_proj.lora_a"

    def tensor(values=(1, 2), shape=(1, 2), dtype="float32"):
        return SimpleNamespace(values=values, shape=shape, dtype=dtype)

    saved = {key: tensor()}
    equal = lambda left, right: left.values == right.values
    assert infer.check_adapter_tensors(saved, {key: tensor()}, equal) == 1
    for actual in (
        {},
        {key: tensor(), key + "_extra": tensor()},
        {key: tensor(shape=(2, 1))},
        {key: tensor(dtype="float16")},
        {key: tensor(values=(1, 3))},
    ):
        with pytest.raises(ValueError):
            infer.check_adapter_tensors(saved, actual, equal)


def test_unverified_cleanup_preserves_completed_model_results_and_checkpoints(experiment, monkeypatch):
    options, cases = experiment
    loader, _ = _runtime(cases, [])
    completed = infer.execute(options, runtime_loader=loader)
    checkpoint = Path(options.run_dir) / "outputs.json"
    checkpoint_before = checkpoint.read_bytes()

    class Child:
        def wait(self, timeout):
            return 0

    monkeypatch.setattr(infer.subprocess, "Popen", lambda *args, **kwargs: Child())

    def refuse_cleanup(_):
        raise PermissionError("private cleanup diagnostic must not be logged")

    monkeypatch.setattr(infer, "_stop_owned_group", refuse_cleanup)
    receipt = infer.supervise(options, [])
    assert receipt["run_status"] == "failed"
    assert receipt["pre_cleanup_run_status"] == "completed"
    assert receipt["cleanup_confirmed"] is False
    assert receipt["cleanup_exception_type"] == "PermissionError"
    assert receipt["raw_facts"] == completed["raw_facts"]
    assert receipt["accepted_cases"] == completed["accepted_cases"] == 2
    assert checkpoint.read_bytes() == checkpoint_before
    assert "private cleanup diagnostic" not in (Path(options.run_dir) / "receipt.json").read_text()


def test_timeout_cleanup_uncertainty_preserves_partial_results_and_all_denominators(experiment, monkeypatch):
    options, cases = experiment
    prepared = infer.prepare(options)
    first = {"case_id": cases[0]["case_id"], "candidate": cases[0]["expected_candidate"]}
    infer._save(Path(options.run_dir) / "outputs.json", prepared, [first])

    class Child:
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("test-only-child", timeout)

    monkeypatch.setattr(infer.subprocess, "Popen", lambda *args, **kwargs: Child())

    def refuse_cleanup(_):
        raise RuntimeError("unverified reused process group")

    monkeypatch.setattr(infer, "_stop_owned_group", refuse_cleanup)
    receipt = infer.supervise(options, [])
    assert receipt["run_status"] == "failed"
    assert receipt["pre_cleanup_run_status"] == "timed_out"
    assert receipt["cleanup_confirmed"] is False
    assert receipt["cleanup_exception_type"] == "RuntimeError"
    assert receipt["case_count"] == 2
    assert receipt["raw_facts"]["present"] == sum(len(case["expected_fields"]) for case in cases)
    assert receipt["accepted_cases"] == 1
    rows = json.loads((Path(options.run_dir) / "outputs.json").read_text())["outputs"]
    assert rows[0] == first
    assert rows[1]["error_code"] == "RUN_WALL_TIMEOUT"
