"""Isolated MLX QLoRA driver. The scraper runtime never imports this module.

Use a pinned, separately installed MLX environment. Stages deliberately require
local model files and leave the serving environment and original weights alone.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

BASE_ID = "mlx-community/Qwen3-1.7B-4bit"
BASE_REVISION = "3b1b1768f8f8cf8351c712464f906e86c2b8269e"
BASE_WEIGHT_SHA256 = "0e86d9677e519323849eac1bc272caae88567a481ff188c431f70be543d9995f"
BASE_WEIGHT_BYTES = 968080210
MLX_LM_COMMIT = "736ab0d8c1d0edc79c3d798044fe6904e2fa23be"
# Canonical SHA256 of the 204 .py files in the verified pinned VCS install.
# Allows an offline-built wheel to prove code identity without VCS direct_url.
MLX_LM_SOURCE_SHA256 = "ac34053e7d3f7f4825790feba96b90fa81ab4dd2069fafd971438c13d239cb99"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Atomic private run receipts, including on callback updates."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def verify_base(model_dir: Path) -> dict:
    if not model_dir.is_dir():
        raise ValueError("Training requires a local model directory")
    weight = model_dir / "model.safetensors"
    if weight.stat().st_size != BASE_WEIGHT_BYTES or file_hash(weight) != BASE_WEIGHT_SHA256:
        raise ValueError("Base weight size/checksum differs from the pinned training base")
    config = json.loads((model_dir / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("num_hidden_layers") != 28:
        raise ValueError("Unexpected base architecture")
    # Include tokenizer/config hashes so a changed template cannot masquerade as
    # the same experiment. The weight digest alone does not identify a tokenizer.
    return {
        "repo_id": BASE_ID,
        "revision": BASE_REVISION,
        "files": {
            p.name: file_hash(p)
            for p in sorted(model_dir.iterdir())
            if p.is_file() and p.suffix in {".json", ".safetensors"}
        },
    }


def verify_training_runtime() -> None:
    distribution = importlib.metadata.distribution("mlx-lm")
    direct = json.loads(distribution.read_text("direct_url.json") or "{}")
    vcs_commit = direct.get("vcs_info", {}).get("commit_id")
    if vcs_commit is not None and vcs_commit != MLX_LM_COMMIT:
        raise ValueError("MLX-LM installation does not match the pinned training commit")
    package = Path(distribution.locate_file("mlx_lm"))
    digest = hashlib.sha256()
    files = sorted(package.rglob("*.py"))
    for path in files:
        digest.update(
            path.relative_to(package).as_posix().encode() + b"\0" + hashlib.sha256(path.read_bytes()).digest()
        )
    if len(files) != 204 or digest.hexdigest() != MLX_LM_SOURCE_SHA256:
        raise ValueError("Installed MLX-LM Python sources differ from the pinned reference")


def validate_config(config: dict) -> None:
    for key, minimum, maximum in (
        ("iters", 4, 2000),
        ("grad_accumulation_steps", 1, 16),
        ("max_seq_length", 512, 8192),
        ("num_layers", 1, 28),
        ("batch_size", 1, 4),
        ("save_every", 1, 1000),
    ):
        value = config.get(key)
        if type(value) is not int or not minimum <= value <= maximum:
            raise ValueError(f"Invalid bounded training option: {key}")
    if config["iters"] % config["grad_accumulation_steps"]:
        raise ValueError("Iterations must include only complete gradient-accumulation groups")
    rate = config["learning_rate"]
    if not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 < rate <= 0.001:
        raise ValueError("Learning rate must be finite and bounded")


def run_location_allowed(model_dir: Path, data_dir: Path, run_dir: Path) -> bool:
    return not any(
        run_dir.resolve().is_relative_to(parent.resolve())
        for parent in (model_dir, data_dir, Path(__file__).parents[1])
    )


def _summary(values: list[int]) -> dict:
    values = sorted(values)
    return {
        "min": values[0],
        "p50": statistics.median(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "max": values[-1],
    }


def preflight_rows(tokenizer, rows: list[dict], max_seq_length: int, processor=None) -> dict:
    """Reject masked-prefix drift and truncation before any gradient operation.

    ``processor`` can be ChatDataset.process to verify the installed trainer's
    output against an independently rendered no-thinking generation prefix.
    """
    if not rows:
        raise ValueError("Empty training split")
    lengths, prompts, completions = [], [], []
    for index, row in enumerate(rows):
        messages = row.get("messages", [])
        if not messages or messages[-1].get("role") != "assistant" or "tools" in row:
            raise ValueError(f"Invalid no-tools chat row {index}")
        target = json.loads(messages[-1]["content"])
        if not isinstance(target, dict) or set(target) != {"data", "evidence", "warnings"}:
            raise ValueError(f"Unexpected training target keys in row {index}")
        prefix = tokenizer.apply_chat_template(
            messages[:-1], add_generation_prompt=True, enable_thinking=False, return_dict=False
        )
        full = tokenizer.apply_chat_template(messages, enable_thinking=False, return_dict=False)
        if processor is not None:
            actual, offset = processor(row)
        else:
            actual, offset = full, len(prefix)
        if actual != full or actual[:offset] != prefix or offset != len(prefix):
            raise ValueError(f"Training/generation masked-prefix mismatch in row {index}")
        if not 0 < offset < len(actual) <= max_seq_length:
            raise ValueError(f"Empty supervision or overlength row {index}: {len(actual)} tokens")
        lengths.append(len(actual))
        prompts.append(offset)
        completions.append(len(actual) - offset)
    return {
        "rows": len(rows),
        "full_tokens": _summary(lengths),
        "prompt_tokens": _summary(prompts),
        "supervised_tokens": _summary(completions),
        "enable_thinking": False,
        "mask_prompt": True,
        "truncated_rows": 0,
    }


def make_config(model_dir: Path, data_dir: Path, run_dir: Path, options) -> dict:
    from mlx_lm.lora import CONFIG_DEFAULTS

    config = dict(CONFIG_DEFAULTS)
    config.update(
        model=str(model_dir),
        data=str(data_dir),
        adapter_path=str(run_dir / "adapter"),
        train=True,
        test=False,
        fine_tune_type="lora",
        optimizer="adam",
        seed=42,
        num_layers=8,
        batch_size=1,
        iters=options.iters,
        grad_accumulation_steps=4,
        learning_rate=1e-5,
        max_seq_length=options.max_seq_length,
        mask_prompt=True,
        grad_checkpoint=True,
        val_batches=-1,
        steps_per_report=8,
        steps_per_eval=40,
        save_every=40,
        clear_cache_threshold=0,
        report_to=None,
        lora_parameters={
            "rank": 8,
            "scale": 20.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
        },
    )
    validate_config(config)
    return config


def execute(options) -> dict:
    import mlx.core as mx
    import mlx.optimizers as optim
    import numpy as np
    from mlx import nn
    from mlx.utils import tree_flatten
    from mlx_lm import load
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import default_loss, iterate_batches
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model_dir, data_dir, run_dir = map(Path, (options.model, options.data, options.run_dir))
    if not run_location_allowed(model_dir, data_dir, run_dir):
        raise ValueError("Keep private run outputs outside the model, dataset and source directories")
    verify_training_runtime()
    if not data_dir.is_dir() or any(
        not (data_dir / f"{split}.jsonl").is_file() for split in ("train", "valid", "test")
    ):
        raise ValueError("Training requires all three local dataset splits")
    base = verify_base(model_dir)
    config = make_config(model_dir, data_dir, run_dir, options)
    if (run_dir / "adapter").exists() and options.stage == "train":
        raise ValueError("Existing training run: choose a new run directory; checkpoints are weights-only")
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    # These limits affect this isolated MLX process only, not machine services.
    mx.set_memory_limit(options.memory_gib * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    np.random.seed(config["seed"])
    mx.random.seed(config["seed"])
    started = time.monotonic()
    model, tokenizer = load(str(model_dir), tokenizer_config={"trust_remote_code": False})
    tokenizer.apply_chat_template = functools.partial(tokenizer.apply_chat_template, enable_thinking=False)
    args = SimpleNamespace(**config)
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    splits = {}
    # Test token boundaries are checked without computing any test loss/score.
    for name, dataset in zip(("train", "valid", "test"), (train_set, valid_set, test_set), strict=True):
        rows = [json.loads(line) for line in (data_dir / f"{name}.jsonl").read_text().splitlines()]
        splits[name] = preflight_rows(tokenizer, rows, options.max_seq_length, dataset.process)
    public_config = {k: v for k, v in config.items() if k not in {"model", "data", "adapter_path"}}
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt = {
        "format": "mlx-scraper-training-v1",
        "stage": options.stage,
        "status": "preflight_passed",
        "base": base,
        "mlx_lm_commit": MLX_LM_COMMIT,
        "mlx_lm_source_sha256": MLX_LM_SOURCE_SHA256,
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("mlx", "mlx-lm", "transformers", "tokenizers", "numpy")
        },
        "hardware": {"system": platform.system(), "machine": platform.machine()},
        "config": public_config,
        "tokenization": splits,
        "data_hashes": {f"{name}.jsonl": file_hash(data_dir / f"{name}.jsonl") for name in splits},
        "dataset_fingerprint": manifest["dataset_fingerprint"],
        "data_manifest_sha256": file_hash(manifest_path),
        "workflow_hashes": {
            name: file_hash(Path(__file__).parent / name)
            for name in ("__init__.py", "train.py", "dataset.py", "job.py", "requirements.lock")
        },
        "training_performed": False,
        "test_evaluation_performed": False,
    }
    write_json(run_dir / "preflight.json", receipt)
    if options.stage == "preflight":
        return receipt

    if options.stage == "sanity":
        model.freeze()
        linear_to_lora_layers(model, config["num_layers"], config["lora_parameters"])
        before = dict(tree_flatten(model.trainable_parameters()))
        if not before or any(not key.endswith(("lora_a", "lora_b")) for key in before):
            raise RuntimeError("Unexpected unfrozen parameters")
        trainable_count = sum(p.size for p in before.values())
        batch = next(iterate_batches(CacheDataset(train_set), 1, options.max_seq_length, seed=42))
        (loss, supervised), grads = nn.value_and_grad(model, default_loss)(model, *batch)
        flat_grads = dict(tree_flatten(grads))
        mx.eval(loss, supervised, grads)
        magnitude = sum(float(mx.sum(mx.abs(g)).item()) for g in flat_grads.values())
        if (
            not math.isfinite(loss.item())
            or supervised.item() <= 0
            or not math.isfinite(magnitude)
            or magnitude <= 0
            or any(not mx.all(mx.isfinite(g)).item() for g in flat_grads.values())
        ):
            raise RuntimeError("Nonfinite or empty training signal")
        path = run_dir / "sanity-adapter"
        path.mkdir(exist_ok=False)
        mx.save_safetensors(str(path / "before.safetensors"), before)
        optim.Adam(learning_rate=config["learning_rate"]).update(model, grads)
        mx.eval(model.parameters())
        mx.save_safetensors(
            str(path / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters()))
        )
        write_json(path / "adapter_config.json", config)
        changed = file_hash(path / "before.safetensors") != file_hash(path / "adapters.safetensors")
        if not changed:
            raise RuntimeError("Sanity optimizer update changed no adapter weights")
        del model, grads, before
        mx.clear_cache()
        reloaded, _ = load(str(model_dir), adapter_path=str(path))
        expected_arrays = mx.load(str(path / "adapters.safetensors"))
        actual_arrays = {
            key: value
            for key, value in tree_flatten(reloaded.parameters())
            if key.endswith(("lora_a", "lora_b"))
        }
        if set(actual_arrays) != set(expected_arrays) or any(
            actual_arrays[key].shape != value.shape or not mx.array_equal(actual_arrays[key], value).item()
            for key, value in expected_arrays.items()
        ):
            raise RuntimeError("Adapter reload did not preserve exact saved tensors")
        reloaded.freeze()
        reload_loss, _ = default_loss(reloaded, *batch)
        mx.eval(reload_loss)
        if not math.isfinite(reload_loss.item()):
            raise RuntimeError("Reloaded adapter produced nonfinite loss")
        receipt.update(
            status="sanity_passed",
            sanity={
                "loss_before": loss.item(),
                "loss_after_reload": reload_loss.item(),
                "supervised_tokens": supervised.item(),
                "gradient_l1": magnitude,
                "adapter_changed": changed,
                "trainable_parameters": trainable_count,
            },
            training_performed=True,
            training_scope="one disposable optimizer update",
        )
    else:
        # A successful disposable update/reload is required for this exact input.
        sanity_path = run_dir / "sanity.json"
        if not sanity_path.exists():
            raise ValueError("Run --stage sanity in this run directory before training")
        sanity = json.loads(sanity_path.read_text())
        if (
            sanity.get("status") != "sanity_passed"
            or sanity["base"] != base
            or sanity["data_hashes"] != receipt["data_hashes"]
            or sanity["config"] != public_config
            or sanity["workflow_hashes"] != receipt["workflow_hashes"]
        ):
            raise ValueError("Sanity receipt does not match this base/dataset")

        class Callback:
            def on_train_loss_report(self, info):
                if not math.isfinite(info["train_loss"]):
                    raise RuntimeError("Nonfinite training loss")
                write_json(run_dir / "progress.json", {"stage": "training", **info})

            def on_val_loss_report(self, info):
                if not math.isfinite(info["val_loss"]):
                    raise RuntimeError("Nonfinite validation loss")
                with (run_dir / "validation.jsonl").open("a") as stream:
                    stream.write(json.dumps(info, allow_nan=False) + "\n")

        train_model(args, model, train_set, valid_set, training_callback=Callback())
        weight = run_dir / "adapter" / "adapters.safetensors"
        arrays = mx.load(str(weight))
        if not arrays or any(not mx.all(mx.isfinite(value)).item() for value in arrays.values()):
            raise RuntimeError("Invalid saved adapter")
        receipt.update(
            status="training_completed",
            training_performed=True,
            adapter_sha256=file_hash(weight),
            microbatches=config["iters"],
            optimizer_updates=config["iters"] // config["grad_accumulation_steps"],
        )
    # Verify no in-place base mutation before declaring either operation done.
    if verify_base(model_dir) != base:
        raise RuntimeError("Base changed during experiment")
    receipt.update(elapsed_seconds=time.monotonic() - started, peak_memory_bytes=mx.get_peak_memory())
    write_json(run_dir / f"{options.stage}.json", receipt)
    return receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local pinned base directory")
    parser.add_argument("--data", required=True, help="Private generated JSONL directory")
    parser.add_argument("--run-dir", required=True, help="Private receipts/checkpoints directory")
    parser.add_argument("--stage", choices=("preflight", "sanity", "train"), default="preflight")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--memory-gib", type=int, choices=range(4, 17), default=8)
    options = parser.parse_args(argv)
    os.umask(0o077)
    try:
        receipt = execute(options)
    except Exception as error:  # noqa: BLE001 -- top-level boundary sanitizes library exceptions
        # Do not echo tokenizer messages, source text, paths or environment.
        if run_location_allowed(Path(options.model), Path(options.data), Path(options.run_dir)):
            try:
                write_json(
                    Path(options.run_dir) / "failure.json",
                    {"stage": options.stage, "status": "failed", "exception_type": type(error).__name__},
                )
            except OSError:
                pass  # The sanitized stdout result remains available if storage fails.
        print(json.dumps({"status": "failed", "exception_type": type(error).__name__}))
        return 1
    print(json.dumps({"status": receipt["status"], "training_performed": receipt["training_performed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
