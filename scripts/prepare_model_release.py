"""Prepare a model-only, experimental MLX adapter bundle; never uploads it.

Run with ``python -m scripts.prepare_model_release --help`` from the checkout.
All inputs remain immutable. Only verified tensors and allowlisted metadata
reach the new release directory. This is an artifact check, not an accuracy
certification or a guarantee that every possible private fact is recognizable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import struct
from pathlib import Path
from typing import Any

from scripts.audit_release import content_rules
from training.train import BASE_ID, BASE_REVISION, BASE_WEIGHT_SHA256, MLX_LM_COMMIT, MLX_LM_SOURCE_SHA256

FORMAT = "experimental-mlx-adapter-release-v1"
MAX_JSON_BYTES = 4_000_000
MAX_WEIGHT_BYTES = 128 * 1024**2
MAX_HEADER_BYTES = 256 * 1024
ROOT = Path(__file__).resolve().parents[1]
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(
    r"[0-9]{1,4}\.[0-9]{1,4}(?:\.[0-9]{1,4})?(?:(?:a|b|rc)[0-9]{1,8})?"
    r"(?:\.post[0-9]{1,8})?(?:\.dev[0-9]{1,8})?(?:\+g[0-9a-f]{7,40})?\Z"
)
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_PYTHON = re.compile(r"3\.12\.[0-9]{1,3}\Z")
_MODEL_ID = re.compile(
    re.escape(BASE_ID) + r"|local-scraper/(?:base|lora)|scraper-qwen3-(?:base|lora)(?:-[0-9a-f]{8})?\Z"
)
_BASE_FILES = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "model.safetensors",
        "model.safetensors.index.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    }
)
_WORKFLOW_FILES = frozenset({"__init__.py", "train.py", "dataset.py", "job.py", "requirements.lock"})
_RUNTIME_PACKAGES = frozenset({"mlx", "mlx-lm", "transformers", "tokenizers", "numpy"})
_SCHEMAS = frozenset(
    {
        "page_v1",
        "article_v1",
        "organization_v1",
        "person_v1",
        "job_v1",
        "event_v1",
        "listing_v1",
        "product_v1",
        "custom_v1",
    }
)
_ERROR_CODES = frozenset(
    {
        "CAPABILITY_PROBE_FAILED",
        "CONTENT_TOO_LARGE",
        "INFERENCE_FAILED",
        "INFERENCE_TIMEOUT",
        "INVALID_EXTRACTION",
        "LOCAL_INFERENCE_HTTP_ERROR",
        "LOCAL_SCHEMA_UNENFORCED",
        "MISSING_CREDENTIALS",
        "TRUNCATED_OUTPUT",
        "INVALID_JSON",
        "INVALID_SCHEMA",
        "MISSING_OUTPUT",
        "RUN_WALL_TIMEOUT",
        "INCOMPLETE_GENERATION",
        "LOCAL_MEMORY_LIMIT",
        "LOCAL_GENERATION_FAILED",
        "GENERATOR_CLOSE_FAILED",
        "INFERENCE_PROCESS_FAILED",
        "LOCAL_EVALUATION_FAILED",
    }
)
_TENSOR = re.compile(r"model\.layers\.([0-9]+)\.self_attn\.([qv])_proj\.lora_([ab])\Z")
LOADER_KEYS = ("fine_tune_type", "num_layers", "lora_parameters")
SCHEMA_KEYS = ("schema_variant", "schema_guard_sha256", "provider_schemas_hash")
ALLOWLIST = frozenset(
    {
        "adapters.safetensors",
        "adapter_config.json",
        "LICENSE",
        "NOTICE",
        "README.md",
        "metrics.json",
        "training.json",
        "release.json",
        "SHA256SUMS",
    }
)


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unique(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate JSON keys are forbidden")
        result[key] = value
    return result


def _constant(_):
    raise ValueError("Nonfinite JSON is forbidden")


def _parse(value: str | bytes) -> Any:
    return json.loads(value, object_pairs_hook=_unique, parse_constant=_constant)


def _regular(path: Path, bound: int) -> None:
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= bound:
        raise ValueError("Input must be a bounded regular file without a symlink")


def read_object(path: Path) -> dict:
    _regular(path, MAX_JSON_BYTES)
    value = _parse(path.read_bytes())
    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object")
    return value


def minimal_config(config: dict) -> dict:
    """These are precisely the settings consumed by MLX load_adapters."""
    result = {key: config[key] for key in LOADER_KEYS}
    params = result["lora_parameters"]
    if result["fine_tune_type"] != "lora" or type(result["num_layers"]) is not int:
        raise ValueError("Only the expected LoRA adapter format is supported")
    if not 1 <= result["num_layers"] <= 28 or not isinstance(params, dict):
        raise ValueError("Invalid adapter layers or parameters")
    if set(params) != {"rank", "scale", "dropout", "keys"}:
        raise ValueError("Unexpected adapter parameters")
    if type(params["rank"]) is not int or not 1 <= params["rank"] <= 64:
        raise ValueError("Invalid adapter rank")
    for name, low, high in (("scale", 0, 256), ("dropout", 0, 0.5)):
        value = params[name]
        if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("Invalid finite adapter scalar")
    if params["scale"] == 0 or params["dropout"] != 0:
        raise ValueError("This release workflow expects nonzero scale and inference-safe zero dropout")
    if not isinstance(params["keys"], list) or sorted(params["keys"]) != [
        "self_attn.q_proj",
        "self_attn.v_proj",
    ]:
        raise ValueError("Only the trained q/v projection keys are supported")
    return result


def verify_tensors(path: Path, config: dict) -> dict:
    """Validate the actual Qwen3 tensor names, shapes, ranges and finite data.

    No NumPy, MLX, pickle or model execution is needed. MLX's metadata null is
    supported; arbitrary metadata is rejected rather than copied into weights.
    """
    _regular(path, MAX_WEIGHT_BYTES)
    rank, count = config["lora_parameters"]["rank"], config["num_layers"]
    expected = {}
    for layer in range(28 - count, 28):
        for projection, out_dim in (("q", 2048), ("v", 1024)):
            prefix = f"model.layers.{layer}.self_attn.{projection}_proj."
            expected[prefix + "lora_a"] = [2048, rank]
            expected[prefix + "lora_b"] = [rank, out_dim]
    with path.open("rb") as stream:
        size = stream.read(8)
        if len(size) != 8:
            raise ValueError("Truncated tensor header")
        header_size = struct.unpack("<Q", size)[0]
        if not 2 <= header_size <= MAX_HEADER_BYTES or 8 + header_size >= path.stat().st_size:
            raise ValueError("Invalid bounded tensor header")
        header = _parse(stream.read(header_size))
        payload = stream.read()
    if not isinstance(header, dict) or header.pop("__metadata__", None) is not None:
        raise ValueError("Unexpected tensor metadata")
    if set(header) != set(expected):
        raise ValueError("Adapter tensor names do not match the declared layer/rank configuration")
    intervals, parameters = [], 0
    for name, item in header.items():
        if (
            not _TENSOR.fullmatch(name)
            or not isinstance(item, dict)
            or set(item) != {"dtype", "shape", "data_offsets"}
        ):
            raise ValueError("Unexpected tensor structure")
        if item["shape"] != expected[name] or any(type(n) is not int for n in item["shape"]):
            raise ValueError("Adapter tensor shape mismatch")
        dtype = item["dtype"]
        if dtype not in {"F32", "F16", "BF16"}:
            raise ValueError("Unsupported adapter dtype")
        offsets = item["data_offsets"]
        if not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets):
            raise ValueError("Invalid tensor offsets")
        start, end = offsets
        elements = math.prod(item["shape"])
        width = 4 if dtype == "F32" else 2
        if not 0 <= start < end <= len(payload) or end - start != elements * width:
            raise ValueError("Truncated or invalid tensor payload")
        section = memoryview(payload)[start:end]
        if dtype == "BF16":
            finite = all((value >> 7) & 255 != 255 for (value,) in struct.iter_unpack("<H", section))
        else:
            finite = all(
                math.isfinite(value)
                for (value,) in struct.iter_unpack("<f" if dtype == "F32" else "<e", section)
            )
        if not finite:
            raise ValueError("Nonfinite adapter tensor")
        parameters += elements
        intervals.append((start, end))
    previous = 0
    for start, end in sorted(intervals):
        if start != previous:
            raise ValueError("Overlapping or unused tensor bytes")
        previous = end
    if previous != len(payload):
        raise ValueError("Unexpected trailing tensor bytes")
    return {"tensor_count": len(expected), "parameters": parameters, "all_finite": True}


def _safe_json(value: dict, name: str) -> bytes:
    encoded = (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode()
    if len(encoded) > MAX_JSON_BYTES or content_rules(encoded, name):
        raise ValueError("Release metadata contains a credential, private path or invalid size")
    return encoded


def _object(value: Any, required, optional=()) -> dict:
    if (
        not isinstance(value, dict)
        or set(value) - set(required) - set(optional)
        or set(required) - set(value)
    ):
        raise ValueError("Invalid or incomplete public metadata schema")
    return value


def _integer(value: Any, low: int = 0, high: int = 1_000_000) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ValueError("Public count must be a bounded integer")
    return value


def _number(value: Any, low: float = 0, high: float = 1_000_000_000) -> int | float:
    if type(value) not in {int, float} or not math.isfinite(value) or not low <= value <= high:
        raise ValueError("Public measurement must be a finite bounded number")
    return value


def _pattern(value: Any, pattern) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError("Public identifier must match its approved value schema")
    return value


def _hashes(value: Any, names) -> dict:
    _object(value, names)
    for digest in value.values():
        _pattern(digest, _SHA)
    return dict(value)


def _fact_pair(value: Any) -> None:
    _object(value, ("raw", "accepted"))
    for item in value.values():
        _fact_metrics(item)


def _group(value: Any) -> None:
    _object(value, ("case_count", "raw_facts", "accepted_facts", "required_facts", "optional_facts"))
    _integer(value["case_count"], 1, 10_000)
    for key in ("raw_facts", "accepted_facts"):
        _fact_metrics(value[key])
    for key in ("required_facts", "optional_facts"):
        _fact_pair(value[key])


def _metric_details(value: dict, count: int) -> None:
    evidence = _object(
        value["source_evidence"],
        (
            "populated_fields",
            "source_supported_fields",
            "correct_with_source_support",
            "authored_reference_matches",
        ),
    )
    for item in evidence.values():
        _integer(item)
    if (
        evidence["populated_fields"] != value["raw_facts"]["predicted"]
        or not evidence["authored_reference_matches"]
        <= evidence["correct_with_source_support"]
        <= evidence["source_supported_fields"]
        <= evidence["populated_fields"]
        or evidence["correct_with_source_support"] > value["raw_facts"]["correct"]
    ):
        raise ValueError("Inconsistent source evidence counts")
    absent = _object(value["abstention"], ("labeled_absent_paths", "correct_abstentions", "accuracy"))
    total, correct = (_integer(absent[key]) for key in ("labeled_absent_paths", "correct_abstentions"))
    _ratio(absent["accuracy"], correct / total if total else None)
    if correct > total:
        raise ValueError("Inconsistent abstention counts")
    diagnostics = _object(
        value["diagnostics"], ("token_limit_cases", "repetition_heuristic_cases", "statuses", "errors")
    )
    for key in ("token_limit_cases", "repetition_heuristic_cases"):
        _integer(diagnostics[key], 0, count)
    statuses = _object(diagnostics["statuses"], (), ("pending", "success", "partial", "failed"))
    if sum(_integer(number, 0, count) for number in statuses.values()) != count:
        raise ValueError("Incomplete outcome status counts")
    errors = _object(diagnostics["errors"], (), _ERROR_CODES)
    if sum(_integer(number, 0, count) for number in errors.values()) > count:
        raise ValueError("Inconsistent error counts")
    latency = _object(value["latency"], ("measured_cases", "p50_ms", "p95_ms", "unknown_cases"))
    measured = _integer(latency["measured_cases"], 0, count)
    if measured + _integer(latency["unknown_cases"], 0, count) != count:
        raise ValueError("Incomplete latency coverage")
    for key in ("p50_ms", "p95_ms"):
        if measured:
            _number(latency[key], 0, 86_400_000)
        elif latency[key] is not None:
            raise ValueError("Unmeasured latency must be null")
    if measured and latency["p50_ms"] > latency["p95_ms"]:
        raise ValueError("Invalid latency percentiles")
    _object(value["tokens"], ("input_tokens", "output_tokens"))
    for item in value["tokens"].values():
        _object(item, ("total", "measured_cases"))
        if _integer(item["measured_cases"], 0, count):
            _integer(item["total"], 0, 1_000_000_000)
        elif item["total"] is not None:
            raise ValueError("Unmeasured token total must be null")
    groups = _object(value["by_schema"], (), _SCHEMAS)
    for item in groups.values():
        _group(item)
    if sum(item["case_count"] for item in groups.values()) != count:
        raise ValueError("Incomplete per-schema case coverage")


def _ratio(value: Any, expected: float | None) -> None:
    if expected is None:
        if value is not None:
            raise ValueError("Unmeasured ratio must be null")
    elif _number(value, 0, 1) != expected:
        raise ValueError("Inconsistent measured ratio")


def _metrics_summary(value: dict) -> dict:
    required = (
        "format",
        "model",
        "inference",
        "cases_hash",
        "outputs_hash",
        "case_count",
        "missing_outputs",
        "valid_json_cases",
        "valid_schema_cases",
        "accepted_cases",
        "expected_status_matches",
        "raw_facts",
        "accepted_facts",
        "required_facts",
        "optional_facts",
        "source_evidence",
        "abstention",
        "diagnostics",
        "latency",
        "tokens",
        "by_schema",
        "evaluation_mode",
    )
    # Ignore fields outside this explicit public projection; every retained
    # value is then validated, including nested keys and scalar types.
    if not isinstance(value, dict) or set(required) - set(value):
        raise ValueError("Missing required measured extraction fields")
    result = {key: value[key] for key in required}
    result.update(_schema_metadata(value))
    if result["format"] != "scraper-model-evaluation-v1" or result["evaluation_mode"] != (
        "initial_model_completion_on_identical_saved_snapshot"
    ):
        raise ValueError("Unsupported measured extraction protocol")
    for key in ("cases_hash", "outputs_hash"):
        _pattern(result[key], _SHA)
    count = _integer(result["case_count"], 1, 10_000)
    for key in (
        "missing_outputs",
        "valid_json_cases",
        "valid_schema_cases",
        "accepted_cases",
        "expected_status_matches",
    ):
        _integer(result[key], 0, count)
    if not result["accepted_cases"] <= result["valid_schema_cases"] <= result["valid_json_cases"] <= count:
        raise ValueError("Inconsistent accepted/schema/JSON counts")
    for key in ("raw_facts", "accepted_facts"):
        _fact_metrics(result[key])
    for key in ("required_facts", "optional_facts"):
        _fact_pair(result[key])
    model = _object(result["model"], ("id", "revision"), ("weights_sha256",))
    _pattern(model["id"], _MODEL_ID)
    _pattern(model["revision"], _COMMIT)
    if "weights_sha256" in model:
        _pattern(model["weights_sha256"], _SHA)
    inference = _object(
        result["inference"],
        ("temperature", "max_tokens", "enable_thinking", "response_format", "repair_attempts"),
        ("seed", "max_source_chars"),
    )
    if (
        inference["enable_thinking"] is not False
        or type(inference["repair_attempts"]) is not int
        or (inference["repair_attempts"] != 0 or inference["response_format"] not in {"none", "json_schema"})
    ):
        raise ValueError("Invalid comparison inference settings")
    _number(inference["temperature"], 0, 2)
    _integer(inference["max_tokens"], 1, 32768)
    for key in ("seed", "max_source_chars"):
        if key in inference:
            _integer(inference[key], 0, 1_000_000_000)
    _metric_details(result, count)
    _safe_json(result, "metrics.json")  # Secondary defense after the typed projection.
    return result


def _fact_metrics(value: dict) -> None:
    counts = ("present", "predicted", "correct", "wrong_or_unlabeled", "missed")
    if not isinstance(value, dict) or set(value) != {*counts, "precision", "recall", "f1"}:
        raise ValueError("Invalid fact metrics")
    if any(type(value[key]) is not int or not 0 <= value[key] <= 1_000_000 for key in counts):
        raise ValueError("Invalid fact counts")
    present, predicted, correct = (value[key] for key in counts[:3])
    if (
        correct > min(present, predicted)
        or value["wrong_or_unlabeled"] != predicted - correct
        or (value["missed"] != present - correct)
    ):
        raise ValueError("Inconsistent fact counts")
    expected = {
        "precision": correct / predicted if predicted else None,
        "recall": correct / present if present else None,
        "f1": 2 * correct / (present + predicted) if present + predicted else None,
    }
    for key, number in expected.items():
        _ratio(value[key], number)


def _bind_candidate(metrics: dict, training: dict, artifacts: dict) -> None:
    digest = artifacts["adapters.safetensors"]
    candidate = _hash({"base": training["base"], "arm": "lora", "artifacts": artifacts})
    if not (
        (
            metrics.get("adapter_sha256") == digest
            and metrics.get("adapter_config_sha256") == artifacts["adapter_config.json"]
        )
        or metrics.get("artifact_files") == artifacts
        or metrics.get("candidate_hash") == candidate
    ):
        raise ValueError("Candidate metrics do not bind to this evaluated adapter")


def _schema_metadata(value: dict) -> dict:
    """Preserve explicit grammar identity without inventing it for old receipts."""
    result = {key: value[key] for key in SCHEMA_KEYS if key in value}
    for key, item in result.items():
        if key == "schema_variant":
            if not isinstance(item, str) or item not in {"original", "exclusive_evidence_v1"}:
                raise ValueError("Invalid public schema variant identifier")
        elif item is not None and (not isinstance(item, str) or not _SHA.fullmatch(item)):
            raise ValueError("Invalid schema grammar fingerprint")
    return result


def paired_metrics(base: dict, candidate: dict, training: dict, artifacts: dict) -> dict:
    _bind_candidate(candidate, training, artifacts)
    if any((key in base) != (key in candidate) or base.get(key) != candidate.get(key) for key in SCHEMA_KEYS):
        raise ValueError("Paired metrics require identical schema variant and grammar fingerprints")
    if (
        base.get("cases_hash") != candidate.get("cases_hash")
        or base.get("case_count") != candidate.get("case_count")
        or base.get("inference") != candidate.get("inference")
        or base.get("evaluation_mode") != candidate.get("evaluation_mode")
    ):
        raise ValueError("Paired metrics require identical cases and inference settings")
    return {"base": _metrics_summary(base), "adapter": _metrics_summary(candidate)}


def _training_summary(training: dict, config: dict) -> dict:
    required = (
        "format",
        "status",
        "base",
        "mlx_lm_commit",
        "mlx_lm_source_sha256",
        "python",
        "versions",
        "config",
        "data_hashes",
        "dataset_fingerprint",
        "data_manifest_sha256",
        "workflow_hashes",
        "training_performed",
        "test_evaluation_performed",
        "adapter_sha256",
        "microbatches",
        "optimizer_updates",
        "elapsed_seconds",
        "peak_memory_bytes",
    )
    if not isinstance(training, dict) or set(required) - set(training):
        raise ValueError("Missing required training provenance fields")
    result = {key: training[key] for key in required}
    if (
        result["format"] != "mlx-scraper-training-v1"
        or result["status"] != "training_completed"
        or (
            training.get("stage") != "train"
            or result["training_performed"] is not True
            or result["test_evaluation_performed"] is not False
        )
    ):
        raise ValueError("A completed real training receipt is required")
    base = _object(result["base"], ("repo_id", "revision", "files"))
    if (
        base["repo_id"] != BASE_ID
        or base["revision"] != BASE_REVISION
        or (
            not isinstance(base["files"], dict)
            or base["files"].get("model.safetensors") != BASE_WEIGHT_SHA256
        )
    ):
        raise ValueError("Training base does not match the pinned model")
    result["base"] = {
        "repo_id": BASE_ID,
        "revision": BASE_REVISION,
        "files": _hashes(base["files"], _BASE_FILES),
    }
    result["data_hashes"] = _hashes(result["data_hashes"], ("train.jsonl", "valid.jsonl", "test.jsonl"))
    result["workflow_hashes"] = _hashes(result["workflow_hashes"], _WORKFLOW_FILES)
    for key in ("dataset_fingerprint", "data_manifest_sha256", "adapter_sha256", "mlx_lm_source_sha256"):
        _pattern(result[key], _SHA)
    _pattern(result["mlx_lm_commit"], _COMMIT)
    if result["mlx_lm_commit"] != MLX_LM_COMMIT or result["mlx_lm_source_sha256"] != MLX_LM_SOURCE_SHA256:
        raise ValueError("Training runtime differs from the pinned source")
    _pattern(result["python"], _PYTHON)
    versions = _object(result["versions"], _RUNTIME_PACKAGES)
    for version in versions.values():
        _pattern(version, _VERSION)
    if minimal_config(training["config"]) != config:
        raise ValueError("Training receipt and saved loader configuration disagree")
    scalars = (
        "seed",
        "optimizer",
        "batch_size",
        "iters",
        "grad_accumulation_steps",
        "learning_rate",
        "max_seq_length",
        "mask_prompt",
        "grad_checkpoint",
    )
    if {*scalars, *LOADER_KEYS} - set(training["config"]):
        raise ValueError("Missing required training configuration fields")
    cfg = {key: training["config"][key] for key in (*scalars, *LOADER_KEYS)}
    if cfg["optimizer"] not in {"adam", "adamw", "sgd", "adafactor", "muon"}:
        raise ValueError("Invalid recorded optimizer")
    for key, low, high in (
        ("seed", 0, 2**32 - 1),
        ("batch_size", 1, 4),
        ("iters", 4, 2000),
        ("grad_accumulation_steps", 1, 16),
        ("max_seq_length", 512, 8192),
    ):
        _integer(cfg[key], low, high)
    if (
        _number(cfg["learning_rate"], 0, 0.001) == 0
        or cfg["mask_prompt"] is not True
        or (type(cfg["grad_checkpoint"]) is not bool)
    ):
        raise ValueError("Invalid learning rate or training mask/checkpoint flags")
    result["config"] = cfg
    microbatches = _integer(result["microbatches"], 4, 2000)
    updates = _integer(result["optimizer_updates"], 1, 2000)
    if (
        microbatches != cfg["iters"]
        or microbatches % cfg["grad_accumulation_steps"]
        or (updates != microbatches // cfg["grad_accumulation_steps"])
    ):
        raise ValueError("Training counters do not match the recorded configuration")
    if _number(result["elapsed_seconds"], 0, 21600) == 0:
        raise ValueError("Training elapsed time must be positive")
    _integer(result["peak_memory_bytes"], 1, 64 * 1024**3)
    tokenization = _object(training["tokenization"], ("train", "valid", "test"))
    counts = {}
    for split in ("train", "valid", "test"):
        if not isinstance(tokenization[split], dict):
            raise TypeError("Invalid tokenization split metadata")
        counts[split] = _integer(tokenization[split]["rows"], 1, 100_000)
    result["data_provenance"] = {
        "kind": "synthetic_authored_templates",
        "counts": counts,
        "human_reviewed_real_web_data": False,
        "training_targets": ["data", "evidence", "warnings"],
        "provenance_assigned_by": "application_code",
    }
    _safe_json(result, "training.json")  # Secondary scan; typed projection is the primary sanitizer.
    return result


def _percent(value) -> str:
    return "unavailable" if value is None else f"{100 * value:.2f}%"


def model_card(training: dict, metrics: dict, release: dict) -> str:
    rows = []
    for label, result in metrics["validation"].items():
        required = result["required_facts"]["accepted"]
        raw = result["required_facts"]["raw"]
        rows.append(
            f"| {label} | {result['case_count']} | {_percent(raw['precision'])} | "
            f"{_percent(raw['recall'])} | {_percent(required['precision'])} | "
            f"{_percent(required['recall'])} | {result['accepted_cases']} |"
        )
    counts = training["data_provenance"]["counts"]
    schema = release["schema_controls"]["validation"]
    schema_variant = schema.get("schema_variant", "unrecorded_legacy")
    schema_guard = schema.get("schema_guard_sha256", "not_recorded")
    provider_schemas = schema.get("provider_schemas_hash", "not_recorded")
    return f"""---
library_name: mlx
license: apache-2.0
pipeline_tag: text-generation
base_model: {BASE_ID}
base_model_relation: adapter
tags:
- mlx
- lora
- information-extraction
- experimental
---

# Experimental evidence scraper adapter

This is a research QLoRA checkpoint for schema-directed webpage extraction.
Production readiness is **not established**. Synthetic validation results do
not establish real-web accuracy, universal coverage, or superiority over other
scrapers. JSON validity and source occurrence alone do not establish factual
accuracy. Application code must validate schema, completeness and exact source
evidence; it assigns status and provenance.

Base: `{BASE_ID}` at `{BASE_REVISION}`. The original quantized base is not
included. Adapter SHA256: `{release["adapter_sha256"]}`.
Dataset fingerprint: `{training["dataset_fingerprint"]}`.
Training used {training.get("microbatches", "unreported")} microbatches and
{training.get("optimizer_updates", "unreported")} optimizer updates.
The synthetic curriculum has {counts["train"]} training, {counts["valid"]} validation and {counts["test"]} held-out
cases across eight presets and custom fields. It uses disjoint DOM/lexical
families, entities, domains and source hashes. Shared simple task patterns
remain. Labels are authored and automatically validated, not independently
reviewed real-web data. Private snapshots and existing public benchmark rows
are excluded. Only data/evidence/warnings are supervised.

## Measured validation

| Artifact | Cases | Raw required precision | Raw required recall | Accepted required precision | Accepted required recall | Accepted records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

Both artifacts use identical cases and inference settings. Inference mode:
`{metrics["validation"]["adapter"]["inference"].get("response_format")}`.
Validation grammar variant: `{schema_variant}`.
Schema guard SHA256: `{schema_guard}`.
Provider schemas hash: `{provider_schemas}`.
Recorded grammar variants and fingerprints must match within each pair;
grammar changes are separate from weight changes. Legacy receipts without
these fields do not establish grammar-byte identity. `release.json` records
these controls separately for validation and any supplied test comparison.
See `metrics.json` for fact/evidence counts, abstention, truncation, latency,
per-schema results and exact case/output hashes. Failed, missing and timed-out
outcomes remain in recall. Accepted records must pass application validation;
raw facts include rejected completions. Test status:
**{release["test_status"]}**. Reviewed real-web evaluation has not been supplied.

## Usage and limits

This is MLX adapter format, not a PEFT or GGUF artifact. Install the pinned
MLX-LM version recorded in `training.json`, obtain the pinned base and load:

```python
from huggingface_hub import snapshot_download
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

base = snapshot_download("{BASE_ID}", revision="{BASE_REVISION}", token=False)
model, tokenizer = load(base, adapter_path="LOCAL_ADAPTER_DIRECTORY")
model.freeze()
model.eval()
prompt = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, tokenize=False
)
text = generate(model, tokenizer, prompt=prompt,
                max_tokens={metrics["validation"]["adapter"]["inference"].get("max_tokens")}, sampler=make_sampler(temp=0))
```

Supply the scraper's trusted extraction instruction and saved numbered source
sections in `messages`. Preserve the schema/evidence/no-tools contract and
disable thinking. Download the exact base revision above rather than silently
using a newer revision; the short example assumes it is already pinned in the
local cache or replaced with its local directory. Direct generation is
unconstrained; test a compatible schema grammar separately. No serving model
is switched by downloading this checkpoint. Fused/requantized artifacts need
their own evaluation. The release config contains exactly the three settings
consumed by MLX's adapter loader; its tensors and loader settings are identical
to the evaluated adapter. Ignored private run paths are removed.

## License and attribution

Model weights derive from Qwen3-1.7B by Qwen, converted by the MLX community,
and retain Apache-2.0 terms. See `LICENSE` and `NOTICE`. The separately published
scraper source and authored synthetic generator are MIT; this does not
relicense the model weights. This checkpoint is a new modification, not an
endorsement by Qwen or the MLX community. No third-party weight-space edit
artifact is incorporated into this gradient-trained adapter.
"""


def prepare_release(
    *,
    adapter_dir: Path,
    training_receipt: Path,
    base_metrics: Path,
    candidate_metrics: Path,
    upstream_license: Path,
    output: Path,
    base_test_metrics: Path | None = None,
    candidate_test_metrics: Path | None = None,
) -> dict:
    adapter_dir, output = Path(adapter_dir), Path(output)
    if adapter_dir.is_symlink() or not adapter_dir.is_dir():
        raise ValueError("Adapter must be a local regular directory")
    if output.exists() or output.is_symlink() or output.resolve().is_relative_to(ROOT):
        raise ValueError("Choose a new release directory outside the public checkout")
    protected = [adapter_dir.resolve(), Path(training_receipt).resolve().parent]
    if any(
        output.resolve().is_relative_to(item) or item.is_relative_to(output.resolve()) for item in protected
    ):
        raise ValueError("Release directory must be separate from original adapter and run data")
    config_path, weights = adapter_dir / "adapter_config.json", adapter_dir / "adapters.safetensors"
    original_config = read_object(config_path)
    config = minimal_config(original_config)
    training = read_object(Path(training_receipt))
    public_training = _training_summary(training, config)
    expected_digest = training.get("adapter_sha256")
    if not isinstance(expected_digest, str) or not _SHA.fullmatch(expected_digest):
        raise ValueError("Training receipt needs an adapter checksum")
    tensors = verify_tensors(weights, config)
    artifacts = {"adapter_config.json": file_hash(config_path), "adapters.safetensors": file_hash(weights)}
    if artifacts["adapters.safetensors"] != expected_digest:
        raise ValueError("Adapter does not match its completed training receipt")
    metrics = {
        "validation": paired_metrics(
            read_object(base_metrics), read_object(candidate_metrics), training, artifacts
        )
    }
    if bool(base_test_metrics) != bool(candidate_test_metrics):
        raise ValueError("Held-out test results require both baseline and candidate")
    if base_test_metrics:
        metrics["test"] = paired_metrics(
            read_object(base_test_metrics), read_object(candidate_test_metrics), training, artifacts
        )
        if metrics["test"]["base"]["cases_hash"] == metrics["validation"]["base"]["cases_hash"]:
            raise ValueError("Held-out test and validation cases must differ")
    _regular(upstream_license, 50_000)
    license_bytes = upstream_license.read_bytes()
    if b"Apache License" not in license_bytes or b"Version 2.0" not in license_bytes:
        raise ValueError("Expected the original upstream Apache-2.0 license")
    release = {
        "format": FORMAT,
        "artifact_type": "mlx_lora_adapter",
        "experimental": True,
        "production_readiness": "not_established",
        "upload_performed": False,
        "test_status": "evaluated" if "test" in metrics else "test_results_unverified",
        "base_model": BASE_ID,
        "base_revision": BASE_REVISION,
        "adapter_sha256": expected_digest,
        "original_config_sha256": artifacts["adapter_config.json"],
        "public_config_sha256": hashlib.sha256(_safe_json(config, "adapter_config.json")).hexdigest(),
        "loader_settings_unchanged": True,
        "tensor_verification": tensors,
        "reviewed_real_web_evaluation": False,
        "schema_controls": {split: _schema_metadata(pair["adapter"]) for split, pair in metrics.items()},
    }
    for split, pair in metrics.items():
        candidate = pair["adapter"]
        required = candidate["required_facts"]["accepted"]
        release[f"{split}_gates"] = {
            "required_precision_and_recall_at_least_95_percent": all(
                required[key] is not None and required[key] >= 0.95 for key in ("precision", "recall")
            ),
            "zero_unsupported_or_wrong_populated_facts": (
                candidate["raw_facts"]["wrong_or_unlabeled"] == 0
                and candidate["source_evidence"]["source_supported_fields"]
                == candidate["source_evidence"]["populated_fields"]
            ),
        }
    files = {
        "adapter_config.json": _safe_json(config, "adapter_config.json"),
        "training.json": _safe_json(public_training, "training.json"),
        "metrics.json": _safe_json(metrics, "metrics.json"),
        "release.json": _safe_json(release, "release.json"),
        "LICENSE": license_bytes,
        "NOTICE": (
            b"This adapter modifies Qwen3-1.7B by Qwen, using the MLX community 4-bit conversion.\n"
            b"Original model weights retain Apache-2.0 terms.\n"
            b"The new QLoRA modification uses authored synthetic extraction training examples.\n"
            b"Scraper source code is separately licensed MIT.\n"
        ),
        "README.md": model_card(public_training, metrics, release).encode(),
    }
    for name, data in files.items():
        if content_rules(data, name):
            raise ValueError("Release text contains a credential or private path")
    input_hashes = {path: file_hash(path) for path in (config_path, weights, upstream_license)}
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output.mkdir(mode=0o700)  # Exclusive reservation: never replace an existing directory.
    try:
        for name, data in files.items():
            with (output / name).open("xb") as stream:
                stream.write(data)
            (output / name).chmod(0o600)
        shutil.copyfile(weights, output / "adapters.safetensors")
        (output / "adapters.safetensors").chmod(0o600)
        if file_hash(output / "adapters.safetensors") != expected_digest or any(
            file_hash(path) != digest for path, digest in input_hashes.items()
        ):
            raise ValueError("An immutable input changed during release preparation")
        hashes = {path.name: file_hash(path) for path in sorted(output.iterdir())}
        (output / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()))
        (output / "SHA256SUMS").chmod(0o600)
        if {path.name for path in output.iterdir()} != ALLOWLIST:
            raise ValueError("Unexpected release artifact")
    except BaseException:
        shutil.rmtree(output)
        raise
    return release


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "adapter-dir",
        "training-receipt",
        "base-metrics",
        "candidate-metrics",
        "upstream-license",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--base-test-metrics", type=Path)
    parser.add_argument("--candidate-test-metrics", type=Path)
    options = parser.parse_args(argv)
    try:
        result = prepare_release(**vars(options))
    except (ValueError, TypeError, KeyError, OSError, struct.error) as error:
        print(json.dumps({"status": "failed", "exception_type": type(error).__name__}))
        return 1
    print(
        json.dumps(
            {
                "status": "prepared",
                "adapter_sha256": result["adapter_sha256"],
                "upload_performed": False,
                "production_readiness": "not_established",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
