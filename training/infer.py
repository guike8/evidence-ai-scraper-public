"""Isolated, offline MLX initial-completion comparisons for base, LoRA and WSP.

Direct generation is unconstrained. It does not establish production JSON grammar
support; use training.evaluate local for the separate loopback server check.
The CLI supervises its own child with a hard overall deadline, including loading.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from ai_scraper.extraction.huggingface import HuggingFaceExtractor, request_instruction
from ai_scraper.storage import atomic_json
from training.evaluate import (
    MAX_FILE_BYTES,
    _context,
    _hash,
    _parse,
    read_jsonl,
    score_outputs,
    validate_cases,
)
from training.job import _stop_owned_group, offline_environment
from training.train import BASE_ID, BASE_REVISION, file_hash, verify_base, verify_training_runtime

FREEZE_FORMAT = "scraper-candidate-freeze-v1"


def _paths(options):
    model, data, output = (Path(getattr(options, name)).resolve() for name in ("model", "data", "run_dir"))
    protected = [model, data] + ([Path(options.artifact).resolve()] if options.artifact else [])
    if any(output == item or output in item.parents or item in output.parents for item in protected):
        raise ValueError("Keep inference output separate from model, dataset and artifacts")
    if options.freeze_candidate:
        freeze = Path(options.freeze_candidate).resolve()
        if any(freeze == item or item in freeze.parents for item in protected):
            raise ValueError("Keep freeze output outside model, dataset and artifacts")
    return model, data, output


def production_messages(case: dict) -> list[dict]:
    """Render only the application's trusted instructions and saved source."""
    request, spec, snapshot, _ = _context(case)
    adapter = HuggingFaceExtractor("offline", "offline", client=object()).for_request(request, spec)
    return [
        {"role": "system", "content": request_instruction(request, spec)},
        {"role": "user", "content": "UNTRUSTED_SOURCE_JSON\n" + adapter._source_content(snapshot)},
    ]


def render_prefix(tokenizer, case: dict, row: dict, max_prompt_tokens: int) -> list[int]:
    expected = production_messages(case)
    messages = row.get("messages", [])[:-1]
    marker = "UNTRUSTED_SOURCE_JSON\n"
    if (
        len(messages) != 2
        or messages[0] != expected[0]
        or messages[1].get("role") != "user"
        or not messages[1].get("content", "").startswith(marker)
        or row["messages"][-1].get("role") != "assistant"
        or _hash(_parse(messages[1]["content"][len(marker) :]))
        != _hash(_parse(expected[1]["content"][len(marker) :]))
    ):
        raise ValueError("Inference inputs differ from production training messages")
    # Sidecar JSON canonicalizes object keys. Preserve the original message's
    # byte-level serialization after independently verifying every source fact.
    prefix = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, enable_thinking=False, return_dict=False
    )
    if not isinstance(prefix, list) or not prefix or any(type(token) is not int for token in prefix):
        raise ValueError("Tokenizer did not return a nonempty token prefix")
    if len(prefix) > max_prompt_tokens:
        raise ValueError("Input exceeds the explicit prompt-token bound")
    return prefix


def _read_object(path: Path) -> dict:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Experiment state exceeds its byte bound")
    value = _parse(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("Experiment state must be an object")
    return value


def artifact_files(options) -> dict:
    if options.arm == "base":
        if options.artifact:
            raise ValueError("A base run does not take an edit or adapter")
        return {}
    if not options.artifact or not Path(options.artifact).is_dir():
        raise ValueError("LoRA and WSP runs require local artifact directories")
    base, artifact = Path(options.model).resolve(), Path(options.artifact).resolve()
    if base == artifact or base in artifact.parents:
        raise ValueError("Keep candidate artifacts outside the base directory")
    if options.arm == "lora":
        names = ("adapter_config.json", "adapters.safetensors")
    else:
        from training.weight_space import read_bundle

        read_bundle(artifact)  # Strict metadata, arrays, budget, checksum and no-pickle checks.
        names = tuple(path.name for path in sorted(artifact.iterdir()) if path.is_file())
    if not names or any(not (artifact / name).is_file() for name in names):
        raise ValueError("Candidate artifact is incomplete")
    return {name: file_hash(artifact / name) for name in names}


def prepare(options) -> dict:
    """Verify candidate identity before opening any requested held-out test rows."""
    for name, low, high in (
        ("max_tokens", 1, 8192),
        ("max_prompt_tokens", 512, 8192),
        ("wall_seconds", 1, 21600),
        ("memory_gib", 4, 8),
    ):
        value = getattr(options, name)
        if type(value) is not int or not low <= value <= high:
            raise ValueError("Invalid bounded inference option")
    if options.arm not in {"base", "lora", "wsp"} or options.split not in {"train", "valid", "test"}:
        raise ValueError("Unknown arm or dataset split")
    if options.max_cases is not None and (
        type(options.max_cases) is not int or not 1 <= options.max_cases <= 1000
    ):
        raise ValueError("Probe case limit must be explicit and bounded")
    if options.split == "train" and (options.max_cases is None or options.max_cases > 50):
        raise ValueError("Training diagnostics require an explicit limit of at most 50 cases")
    if options.freeze_candidate and options.split != "valid":
        raise ValueError("Freeze candidates using a validation run, before opening test cases")
    model_dir, data_dir, _ = _paths(options)
    base = verify_base(model_dir)
    artifacts = artifact_files(options)
    inference = {
        "temperature": 0,
        "max_tokens": options.max_tokens,
        "enable_thinking": False,
        "response_format": "none",
        "repair_attempts": 0,
        "max_source_chars": 48000,
    }
    candidate = {"base": base, "arm": options.arm, "artifacts": artifacts}
    candidate_hash = _hash(candidate)
    inference_hash = _hash({**inference, "max_prompt_tokens": options.max_prompt_tokens})
    if options.split == "test":
        if not options.frozen_candidate:
            raise ValueError("Freeze the candidate and inference settings before opening test cases")
        frozen = _read_object(Path(options.frozen_candidate))
        if (
            frozen.get("format") != FREEZE_FORMAT
            or frozen.get("candidate_hash") != candidate_hash
            or (frozen.get("inference_hash") != inference_hash)
        ):
            raise ValueError("Frozen candidate does not match these exact weights and inference settings")
    cases = read_jsonl(data_dir / f"{options.split}.cases.jsonl")
    rows = read_jsonl(data_dir / f"{options.split}.jsonl")
    if len(cases) != len(rows) or any(case.get("split") != options.split for case in cases):
        raise ValueError("Dataset rows and declared split do not match")
    corpus_count = len(cases)
    cases, rows = cases[: options.max_cases], rows[: options.max_cases]
    validate_cases(cases)
    model = {"id": f"local-scraper/{options.arm}", "revision": candidate_hash[:40]}
    selection = {
        "split": options.split,
        "split_case_count": corpus_count,
        "selected_case_count": len(cases),
        "is_subset": len(cases) != corpus_count,
        "selected_case_ids_hash": _hash([case["case_id"] for case in cases]),
        "selection": "file_order_first_n" if len(cases) != corpus_count else "entire_split",
    }
    identity = _hash(
        {
            "candidate": candidate_hash,
            "inference": inference_hash,
            "cases": cases,
            "rows": rows,
            "selection": selection,
        }
    )
    return {
        "base": base,
        "artifacts": artifacts,
        "candidate_hash": candidate_hash,
        "inference_hash": inference_hash,
        "model": model,
        "inference": inference,
        "cases": cases,
        "rows": rows,
        "selection": selection,
        "identity": identity,
    }


def _checkpoint(path: Path, experiment: dict) -> list[dict]:
    if not path.exists():
        return []
    saved = _read_object(path)
    if saved.get("identity") != experiment["identity"]:
        raise ValueError("Checkpoint belongs to different weights, inputs or inference settings")
    outputs = saved.get("outputs")
    score_outputs(experiment["cases"], outputs, model=experiment["model"], inference=experiment["inference"])
    return outputs


def _save(path: Path, experiment: dict, outputs: list[dict]) -> None:
    atomic_json(path, {"identity": experiment["identity"], "outputs": outputs})


def _report(options, experiment: dict, outputs: list[dict], status: str, elapsed: float) -> dict:
    receipt = score_outputs(
        experiment["cases"], outputs, model=experiment["model"], inference=experiment["inference"]
    )
    receipt.update(
        run_status=status,
        candidate_hash=experiment["candidate_hash"],
        inference_hash=experiment["inference_hash"],
        selection=experiment["selection"],
        arm=options.arm,
        base_model=BASE_ID,
        base_revision=BASE_REVISION,
        elapsed_seconds=elapsed,
        memory_limit_bytes=options.memory_gib * 1024**3,
        memory_limit_scope="MLX allocator; excludes Python/tokenizer/OS memory",
        deadline_seconds=options.wall_seconds,
        deadline_enforcement="CLI child process hard deadline; in-worker cooperative token/prefill checks",
        grammar_verification="not_performed: direct MLX generation is unconstrained",
    )
    receipt["prompt_hashes_hash"] = _hash({row["case_id"]: row.get("prompt_hash") for row in outputs})
    receipt["cache_state"] = "Fresh KV cache per case; loaded model weights shared across selected cases"
    receipt["mlx_generation_peak_memory_bytes"] = max(
        (row["peak_memory_bytes"] for row in outputs if row.get("peak_memory_bytes") is not None),
        default=None,
    )
    runtime_path = Path(options.run_dir) / "runtime.json"
    if runtime_path.exists():
        runtime = _read_object(runtime_path)
        if runtime.get("identity") == experiment["identity"]:
            receipt["runtime_controls"] = runtime["controls"]
    atomic_json(Path(options.run_dir) / "receipt.json", receipt)
    return receipt


def _finish_pending(options, experiment: dict, outputs: list[dict], code: str) -> None:
    completed = {row["case_id"] for row in outputs}
    outputs.extend(
        {
            "case_id": case["case_id"],
            "error_code": code,
            "elapsed_ms": None,
            "input_tokens": None,
            "output_tokens": None,
        }
        for case in experiment["cases"]
        if case["case_id"] not in completed
    )
    _save(Path(options.run_dir) / "outputs.json", experiment, outputs)


def _freeze(path: Path, experiment: dict, receipt: dict) -> None:
    if path.exists():
        prior = _read_object(path)
        if (
            prior.get("format") != FREEZE_FORMAT
            or prior.get("candidate_hash") != experiment["candidate_hash"]
            or (prior.get("inference_hash") != experiment["inference_hash"])
        ):
            raise ValueError("Existing freeze belongs to another candidate")
        return
    atomic_json(
        path,
        {
            "format": FREEZE_FORMAT,
            "created_at": datetime.now(UTC).isoformat(),
            "candidate_hash": experiment["candidate_hash"],
            "inference_hash": experiment["inference_hash"],
            "validation_receipt_hash": _hash(receipt),
            "test_results_used": False,
        },
    )


def check_adapter_tensors(saved: dict, actual: dict, array_equal) -> int:
    """Reject silent partial adapter loads before scoring a candidate."""
    if not saved or set(saved) != set(actual) or any(not key.endswith(("lora_a", "lora_b")) for key in saved):
        raise ValueError("Loaded adapter keys differ from the saved artifact")
    if any(
        value.shape != actual[key].shape
        or value.dtype != actual[key].dtype
        or not array_equal(actual[key], value)
        for key, value in saved.items()
    ):
        raise ValueError("Loaded adapter tensors differ from the saved artifact")
    return len(saved)


def verify_loaded_adapter(model, path: Path) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    saved = mx.load(str(path / "adapters.safetensors"))
    actual = {
        key: value for key, value in tree_flatten(model.parameters()) if key.endswith(("lora_a", "lora_b"))
    }
    count = check_adapter_tensors(saved, actual, lambda left, right: mx.array_equal(left, right).item())
    return {"exact_tensor_reload": True, "tensor_count": count}


def load_runtime(options):
    """No network identifiers reach MLX load; remote code and auth are disabled."""
    verify_training_runtime()
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    mx.set_memory_limit(options.memory_gib * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    mx.random.seed(42)
    model, tokenizer = load(
        str(Path(options.model).resolve()),
        trust_remote_code=False,
        tokenizer_config={"trust_remote_code": False},
        adapter_path=str(Path(options.artifact).resolve()) if options.arm == "lora" else None,
    )
    if options.arm == "wsp":
        from training.weight_space import apply_bundle

        controls = apply_bundle(
            model,
            Path(options.artifact),
            base_dir=Path(options.model),
            base_model=BASE_ID,
            revision=BASE_REVISION,
        )
    else:
        controls = {"training": False, "base_requantized": False}
        if options.arm == "lora":
            controls["adapter_reload"] = verify_loaded_adapter(model, Path(options.artifact))
    model.freeze()
    model.eval()
    sampler = make_sampler(temp=0)

    def generate(prefix, max_tokens, check_deadline):
        return stream_generate(
            model,
            tokenizer,
            prefix,
            max_tokens=max_tokens,
            sampler=sampler,
            prompt_progress_callback=lambda *_: check_deadline(),
        )

    return tokenizer, generate, controls


def _generate_one(case, row, tokenizer, generate, options, deadline, clock) -> dict:
    result = {"case_id": case["case_id"]}
    started, parts = clock(), []

    def check_deadline():
        if clock() >= deadline:
            raise TimeoutError

    stream = None
    try:
        check_deadline()
        prefix = render_prefix(tokenizer, case, row, options.max_prompt_tokens)
        result["input_tokens"] = len(prefix)
        result["prompt_hash"] = _hash(prefix)
        stream = generate(prefix, options.max_tokens, check_deadline)
        finished = False
        for response in stream:
            parts.append(response.text)
            result["output_tokens"] = response.generation_tokens
            peak = getattr(response, "peak_memory", None)
            if type(peak) in {int, float} and math.isfinite(peak) and peak >= 0:
                result["peak_memory_bytes"] = round(peak * 1_000_000_000)
            if response.finish_reason is not None:
                result["finish_reason"] = response.finish_reason
                finished = True
            check_deadline()
            if response.generation_tokens >= options.max_tokens:
                result.setdefault("finish_reason", "length")
                finished = True
                break
        if not finished:
            result["error_code"] = "INCOMPLETE_GENERATION"
    except TimeoutError:
        result["error_code"] = "RUN_WALL_TIMEOUT"
    except Exception as exc:  # noqa: BLE001 -- never log model inputs or third-party exception text
        result["error_code"] = (
            "LOCAL_MEMORY_LIMIT" if isinstance(exc, MemoryError) else "LOCAL_GENERATION_FAILED"
        )
    finally:
        if stream is not None and hasattr(stream, "close"):
            try:
                stream.close()
            except Exception:  # noqa: BLE001 -- preserve sanitized generation failure and partial output
                result.setdefault("error_code", "GENERATOR_CLOSE_FAILED")
    result.update(raw_output="".join(parts), elapsed_ms=max(0, (clock() - started) * 1000))
    return result


def execute(options, *, runtime_loader=None, clock=time.monotonic) -> dict:
    """Worker body; fake runtime injection permits testing without model inference."""
    started = clock()
    deadline = options.deadline_monotonic or started + options.wall_seconds
    experiment = prepare(options)
    path = Path(options.run_dir) / "outputs.json"
    outputs = _checkpoint(path, experiment)
    _save(path, experiment, outputs)
    completed = {row["case_id"] for row in outputs}
    if len(completed) < len(experiment["cases"]):
        tokenizer, generate, controls = (runtime_loader or load_runtime)(options)
        atomic_json(
            Path(options.run_dir) / "runtime.json", {"identity": experiment["identity"], "controls": controls}
        )
        for case, row in zip(experiment["cases"], experiment["rows"], strict=True):
            if case["case_id"] in completed:
                continue
            if clock() >= deadline:
                _finish_pending(options, experiment, outputs, "RUN_WALL_TIMEOUT")
                break
            outputs.append(_generate_one(case, row, tokenizer, generate, options, deadline, clock))
            _save(path, experiment, outputs)
    if (
        verify_base(Path(options.model)) != experiment["base"]
        or artifact_files(options) != experiment["artifacts"]
    ):
        raise ValueError("Candidate or base files changed during inference")
    status = (
        "timed_out" if any(row.get("error_code") == "RUN_WALL_TIMEOUT" for row in outputs) else "completed"
    )
    receipt = _report(options, experiment, outputs, status, clock() - started)
    if options.freeze_candidate and status == "completed":
        _freeze(Path(options.freeze_candidate), experiment, receipt)
    return receipt


def supervise(options, argv: list[str]) -> dict:
    """Hard-limit only the newly created child session; never signal stored PIDs."""
    started = time.monotonic()
    _, _, directory = _paths(options)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_fd = os.open(directory / "inference.lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        environment = offline_environment()
        deadline = started + options.wall_seconds
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "training.infer",
                *argv,
                "--worker",
                "--deadline-monotonic",
                str(deadline),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            pass_fds=(lock_fd,),
            env=environment,
        )
        code, status = "INFERENCE_PROCESS_FAILED", "failed"
        try:
            if child.wait(timeout=max(0.01, deadline - time.monotonic())) == 0:
                return _read_object(directory / "receipt.json")
        except subprocess.TimeoutExpired:
            code, status = "RUN_WALL_TIMEOUT", "timed_out"
        finally:
            _stop_owned_group(child)
        experiment = prepare(options)
        outputs = _checkpoint(directory / "outputs.json", experiment)
        _finish_pending(options, experiment, outputs, code)
        return _report(options, experiment, outputs, status, time.monotonic() - started)
    finally:
        os.close(lock_fd)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "data", "run-dir"):
        result.add_argument("--" + name, required=True, help="Local private directory")
    result.add_argument("--arm", choices=("base", "lora", "wsp"), default="base")
    result.add_argument("--artifact", help="Local adapter or WSP bundle; never a repository identifier")
    result.add_argument("--split", choices=("train", "valid", "test"), default="valid")
    result.add_argument(
        "--max-cases", type=int, help="Declared first-N subset; required for train diagnostics"
    )
    result.add_argument("--max-tokens", type=int, default=4096)
    result.add_argument("--max-prompt-tokens", type=int, default=4096)
    result.add_argument("--wall-seconds", type=int, default=1800)
    result.add_argument("--memory-gib", type=int, choices=range(4, 9), default=8)
    result.add_argument(
        "--freeze-candidate", help="Write frozen configuration after a completed validation run"
    )
    result.add_argument(
        "--frozen-candidate", help="Required exact candidate/config binding before test access"
    )
    result.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    result.add_argument("--deadline-monotonic", type=float, help=argparse.SUPPRESS)
    return result


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    options = parser().parse_args(argv)
    os.umask(0o077)
    try:
        receipt = execute(options) if options.worker else supervise(options, argv)
        print(
            json.dumps(
                {
                    "status": receipt["run_status"],
                    "case_count": receipt["case_count"],
                    "accepted_cases": receipt["accepted_cases"],
                }
            )
        )
        return 0 if receipt["run_status"] == "completed" else 1
    except Exception as exc:  # noqa: BLE001 -- paths, model data and credentials stay out of logs
        print(json.dumps({"status": "failed", "exception_type": type(exc).__name__}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
