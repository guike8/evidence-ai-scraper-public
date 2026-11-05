"""Actual bytes, metadata binding and immutable public model bundle checks."""

from __future__ import annotations

import copy
import json
import struct

import pytest

from scripts import prepare_model_release as release


def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def tensor_file(path):
    header, offset = {"__metadata__": None}, 0
    for projection, output in (("q", 2048), ("v", 1024)):
        for suffix, shape in (("a", [2048, 1]), ("b", [1, output])):
            size = shape[0] * shape[1] * 4
            header[f"model.layers.27.self_attn.{projection}_proj.lora_{suffix}"] = {
                "dtype": "F32",
                "shape": shape,
                "data_offsets": [offset, offset + size],
            }
            offset += size
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + struct.pack("<f", 0.125) * (offset // 4))


def facts(present=10, predicted=10, correct=8):
    return {
        "present": present,
        "predicted": predicted,
        "correct": correct,
        "wrong_or_unlabeled": predicted - correct,
        "missed": present - correct,
        "precision": correct / predicted if predicted else None,
        "recall": correct / present if present else None,
        "f1": 2 * correct / (present + predicted) if present + predicted else None,
    }


def metrics():
    group = {
        "case_count": 2,
        "raw_facts": facts(),
        "accepted_facts": facts(10, 5, 4),
        "required_facts": {"raw": facts(), "accepted": facts(10, 5, 4)},
        "optional_facts": {"raw": facts(0, 0, 0), "accepted": facts(0, 0, 0)},
    }
    return {
        "format": "scraper-model-evaluation-v1",
        "cases_hash": "c" * 64,
        "outputs_hash": "d" * 64,
        "case_count": 2,
        "missing_outputs": 0,
        "expected_status_matches": 1,
        "valid_json_cases": 2,
        "valid_schema_cases": 2,
        "accepted_cases": 1,
        "model": {"id": "local-scraper/lora", "revision": "a" * 40},
        "inference": {
            "temperature": 0,
            "max_tokens": 4096,
            "enable_thinking": False,
            "response_format": "none",
            "repair_attempts": 0,
        },
        "evaluation_mode": "initial_model_completion_on_identical_saved_snapshot",
        "raw_facts": facts(),
        "accepted_facts": facts(10, 5, 4),
        "required_facts": {"raw": facts(), "accepted": facts(10, 5, 4)},
        "optional_facts": {"raw": facts(0, 0, 0), "accepted": facts(0, 0, 0)},
        "source_evidence": {
            "populated_fields": 10,
            "source_supported_fields": 8,
            "correct_with_source_support": 8,
            "authored_reference_matches": 7,
        },
        "latency": {"measured_cases": 2, "p50_ms": 10.0, "p95_ms": 20.0, "unknown_cases": 0},
        "abstention": {"labeled_absent_paths": 2, "correct_abstentions": 1, "accuracy": 0.5},
        "diagnostics": {
            "token_limit_cases": 0,
            "repetition_heuristic_cases": 0,
            "statuses": {"success": 1, "failed": 1},
            "errors": {},
        },
        "tokens": {
            "input_tokens": {"total": 200, "measured_cases": 2},
            "output_tokens": {"total": 100, "measured_cases": 2},
        },
        "by_schema": {"custom_v1": group},
    }


@pytest.fixture
def inputs(tmp_path):
    original = tmp_path / "run"
    adapter = original / "adapter"
    adapter.mkdir(parents=True)
    config = {
        "fine_tune_type": "lora",
        "num_layers": 1,
        "lora_parameters": {
            "rank": 1,
            "scale": 20.0,
            "dropout": 0.0,
            "keys": ["self_attn.q_proj", "self_attn.v_proj"],
        },
        "model": "/private/example/base",
        "data": "/private/example/data",
        "adapter_path": "/private/example/run",
        "report_to": None,
    }
    write(adapter / "adapter_config.json", config)
    tensor_file(adapter / "adapters.safetensors")
    training = {
        "format": "mlx-scraper-training-v1",
        "status": "training_completed",
        "stage": "train",
        "training_performed": True,
        "test_evaluation_performed": False,
        "adapter_sha256": release.file_hash(adapter / "adapters.safetensors"),
        "base": {
            "repo_id": release.BASE_ID,
            "revision": release.BASE_REVISION,
            "files": {
                **dict.fromkeys(release._BASE_FILES, "b" * 64),
                "model.safetensors": release.BASE_WEIGHT_SHA256,
            },
        },
        "config": {
            **copy.deepcopy(config),
            "optimizer": "adam",
            "seed": 42,
            "iters": 200,
            "batch_size": 1,
            "grad_accumulation_steps": 4,
            "learning_rate": 1e-5,
            "max_seq_length": 4096,
            "mask_prompt": True,
            "grad_checkpoint": True,
        },
        "dataset_fingerprint": "e" * 64,
        "versions": {
            "mlx-lm": "0.32.3.dev17+g736ab0d8c",
            "mlx": "0.32.4",
            "transformers": "5.22.1",
            "tokenizers": "0.23.3",
            "numpy": "2.5.4",
        },
        "mlx_lm_commit": release.MLX_LM_COMMIT,
        "mlx_lm_source_sha256": release.MLX_LM_SOURCE_SHA256,
        "python": "3.12.16",
        "data_hashes": dict.fromkeys(("train.jsonl", "valid.jsonl", "test.jsonl"), "b" * 64),
        "data_manifest_sha256": "b" * 64,
        "workflow_hashes": dict.fromkeys(release._WORKFLOW_FILES, "b" * 64),
        "elapsed_seconds": 80.0,
        "peak_memory_bytes": 3_000_000_000,
        "tokenization": {
            split: {"rows": rows} for split, rows in (("train", 720), ("valid", 72), ("test", 108))
        },
        "microbatches": 200,
        "optimizer_updates": 50,
    }
    write(original / "train.json", training)
    baseline, candidate = metrics(), metrics()
    candidate["adapter_sha256"] = training["adapter_sha256"]
    candidate["adapter_config_sha256"] = release.file_hash(adapter / "adapter_config.json")
    write(tmp_path / "base-metrics.json", baseline)
    write(tmp_path / "candidate-metrics.json", candidate)
    license_path = tmp_path / "LICENSE"
    license_path.write_text("Apache License\nVersion 2.0, January 2004\nFixture license only.\n")
    return {
        "adapter_dir": adapter,
        "training_receipt": original / "train.json",
        "base_metrics": tmp_path / "base-metrics.json",
        "candidate_metrics": tmp_path / "candidate-metrics.json",
        "upstream_license": license_path,
        "output": tmp_path / "release",
    }


def change(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    write(path, value)


def test_exact_model_only_bundle_preserves_source_and_loader_semantics(inputs):
    originals = {p: release.file_hash(p) for p in inputs["adapter_dir"].iterdir()}
    result = release.prepare_release(**inputs)
    output = inputs["output"]
    assert {p.name for p in output.iterdir()} == release.ALLOWLIST
    assert result["experimental"] and result["production_readiness"] == "not_established"
    assert result["test_status"] == "test_results_unverified"
    assert result["tensor_verification"] == {"tensor_count": 4, "parameters": 7168, "all_finite": True}
    public_config = json.loads((output / "adapter_config.json").read_text())
    assert public_config == release.minimal_config(
        json.loads((inputs["adapter_dir"] / "adapter_config.json").read_text())
    )
    assert set(public_config) == set(release.LOADER_KEYS)
    assert all(release.file_hash(p) == digest for p, digest in originals.items())
    assert release.file_hash(output / "adapters.safetensors") == result["adapter_sha256"]
    assert (output / "LICENSE").read_bytes() == inputs["upstream_license"].read_bytes()
    assert "Apache-2.0" in (output / "NOTICE").read_text()
    assert "base_model_relation: adapter" in (output / "README.md").read_text()
    assert "Private snapshots" in (output / "README.md").read_text()
    for path in output.iterdir():
        if path.suffix != ".safetensors":
            assert "/private/example" not in path.read_text()
    checksums = dict(line.split("  ", 1)[::-1] for line in (output / "SHA256SUMS").read_text().splitlines())
    assert len(checksums) == len(release.ALLOWLIST) - 1
    assert all(release.file_hash(output / name) == digest for name, digest in checksums.items())
    assert result["validation_gates"]["required_precision_and_recall_at_least_95_percent"] is False


def test_ignored_private_config_tokens_never_enter_bundle(inputs):
    secret = "hf_" + "x" * 32
    for path in (inputs["adapter_dir"] / "adapter_config.json", inputs["training_receipt"]):
        change(path, lambda d: d.update(HF_TOKEN=secret, ignored_path="/private/example/hidden"))
    change(
        inputs["candidate_metrics"],
        lambda d: d.update(
            adapter_config_sha256=release.file_hash(inputs["adapter_dir"] / "adapter_config.json")
        ),
    )
    release.prepare_release(**inputs)
    assert all(secret.encode() not in p.read_bytes() for p in inputs["output"].iterdir())


def test_weights_must_match_completed_training_receipt(inputs):
    change(inputs["training_receipt"], lambda d: d.update(adapter_sha256="0" * 64))
    with pytest.raises(ValueError, match="completed training receipt"):
        release.prepare_release(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize("status", ["sanity_passed", "preflight_passed", "failed"])
def test_a_disposable_or_untrained_model_cannot_be_published(inputs, status):
    change(inputs["training_receipt"], lambda d: d.update(status=status))
    with pytest.raises(ValueError, match="completed real training"):
        release.prepare_release(**inputs)


def test_missing_weights_rejected(inputs):
    (inputs["adapter_dir"] / "adapters.safetensors").unlink()
    with pytest.raises(ValueError, match="regular file"):
        release.prepare_release(**inputs)


def test_truncated_tensor_payload_rejected_before_release_creation(inputs):
    path = inputs["adapter_dir"] / "adapters.safetensors"
    path.write_bytes(path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="payload"):
        release.prepare_release(**inputs)
    assert not inputs["output"].exists()


def test_nonfinite_weights_rejected_without_numpy(inputs):
    path = inputs["adapter_dir"] / "adapters.safetensors"
    value = bytearray(path.read_bytes())
    start = 8 + struct.unpack("<Q", value[:8])[0]
    value[start : start + 4] = struct.pack("<f", float("inf"))
    path.write_bytes(value)
    with pytest.raises(ValueError, match="Nonfinite"):
        release.prepare_release(**inputs)


@pytest.mark.parametrize("field,value", [("cases_hash", "f" * 64), ("case_count", 3)])
def test_paired_corpus_identity_required(inputs, field, value):
    change(inputs["candidate_metrics"], lambda d: d.update({field: value}))
    with pytest.raises(ValueError, match="identical cases"):
        release.prepare_release(**inputs)


def test_paired_inference_settings_required(inputs):
    change(inputs["candidate_metrics"], lambda d: d["inference"].update(max_tokens=512))
    with pytest.raises(ValueError, match="identical cases"):
        release.prepare_release(**inputs)


@pytest.mark.parametrize(
    "field,base_value,candidate_value",
    [
        ("schema_variant", "original", "exclusive_evidence_v1"),
        ("schema_guard_sha256", "a" * 64, "b" * 64),
        ("provider_schemas_hash", "a" * 64, "b" * 64),
    ],
)
def test_grammar_changes_cannot_be_attributed_to_weights(inputs, field, base_value, candidate_value):
    change(inputs["base_metrics"], lambda d: d.update({field: base_value}))
    change(inputs["candidate_metrics"], lambda d: d.update({field: candidate_value}))
    with pytest.raises(ValueError, match="identical schema variant"):
        release.prepare_release(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_variant", "original"),
        ("schema_guard_sha256", None),
        ("provider_schemas_hash", "a" * 64),
    ],
)
@pytest.mark.parametrize("side", ["base_metrics", "candidate_metrics"])
def test_one_sided_grammar_metadata_rejected_even_when_value_is_null(inputs, field, value, side):
    change(inputs[side], lambda d: d.update({field: value}))
    with pytest.raises(ValueError, match="identical schema variant"):
        release.prepare_release(**inputs)


def test_identical_grammar_controls_are_preserved_in_metrics_card_and_release_state(inputs):
    schema = {
        "schema_variant": "exclusive_evidence_v1",
        "schema_guard_sha256": "a" * 64,
        "provider_schemas_hash": "b" * 64,
    }
    for side in ("base_metrics", "candidate_metrics"):
        change(inputs[side], lambda d: d.update(schema))
    result = release.prepare_release(**inputs)
    assert result["schema_controls"] == {"validation": schema}
    saved = json.loads((inputs["output"] / "metrics.json").read_text())["validation"]
    assert all({key: pair[key] for key in schema} == schema for pair in saved.values())
    state = json.loads((inputs["output"] / "release.json").read_text())
    assert state["schema_controls"] == result["schema_controls"]
    card = (inputs["output"] / "README.md").read_text()
    assert all(str(value) in card for value in schema.values())


def test_legacy_metrics_remain_explicitly_unrecorded(inputs):
    result = release.prepare_release(**inputs)
    assert result["schema_controls"] == {"validation": {}}
    assert "unrecorded_legacy" in (inputs["output"] / "README.md").read_text()


def test_original_schema_can_explicitly_declare_no_guard(inputs):
    schema = {"schema_variant": "original", "schema_guard_sha256": None, "provider_schemas_hash": "c" * 64}
    for side in ("base_metrics", "candidate_metrics"):
        change(inputs[side], lambda d: d.update(schema))
    assert release.prepare_release(**inputs)["schema_controls"]["validation"] == schema


def test_candidate_metrics_cannot_describe_some_other_adapter(inputs):
    change(inputs["candidate_metrics"], lambda d: d.update(adapter_sha256="f" * 64))
    with pytest.raises(ValueError, match="bind"):
        release.prepare_release(**inputs)


def test_weights_hash_alone_cannot_claim_an_evaluated_loader_configuration(inputs):
    change(inputs["candidate_metrics"], lambda d: d.pop("adapter_config_sha256"))
    with pytest.raises(ValueError, match="bind"):
        release.prepare_release(**inputs)


def test_native_candidate_hash_also_binds_original_config_and_weights(inputs):
    training = json.loads(inputs["training_receipt"].read_text())
    artifacts = {p.name: release.file_hash(p) for p in inputs["adapter_dir"].iterdir()}
    identity = release._hash({"base": training["base"], "arm": "lora", "artifacts": artifacts})
    change(
        inputs["candidate_metrics"], lambda d: (d.pop("adapter_sha256"), d.update(candidate_hash=identity))
    )
    release.prepare_release(**inputs)


def test_release_refuses_secrets_in_retained_metrics(inputs):
    change(inputs["candidate_metrics"], lambda d: d["model"].update(id="hf_" + "a" * 32))
    with pytest.raises(ValueError, match="identifier"):
        release.prepare_release(**inputs)


def test_arbitrary_training_text_cannot_become_public_scalar_metadata(inputs):
    change(inputs["training_receipt"], lambda d: d["config"].update(optimizer="/private/example/customer"))
    with pytest.raises(ValueError, match="optimizer"):
        release.prepare_release(**inputs)


def test_arbitrary_paths_cannot_hide_inside_statistics(inputs):
    change(
        inputs["candidate_metrics"], lambda d: d.update(diagnostics={"message": "/private/example/source"})
    )
    with pytest.raises(ValueError, match="schema"):
        release.prepare_release(**inputs)


def test_base_hash_labels_cannot_publish_private_paths(inputs):
    change(
        inputs["training_receipt"], lambda d: d["base"]["files"].update({"/private/example/base": "a" * 64})
    )
    with pytest.raises(ValueError, match="schema"):
        release.prepare_release(**inputs)


def test_existing_directory_is_never_overwritten(inputs):
    inputs["output"].mkdir()
    marker = inputs["output"] / "marker"
    marker.write_text("preserve")
    with pytest.raises(ValueError, match="new release"):
        release.prepare_release(**inputs)
    assert marker.read_text() == "preserve"


def test_output_may_not_escape_into_source_adapter_through_symlink(inputs, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(inputs["adapter_dir"], target_is_directory=True)
    inputs["output"] = alias / "release"
    with pytest.raises(ValueError, match="separate"):
        release.prepare_release(**inputs)


def test_symlinked_weights_are_not_followed(inputs, tmp_path):
    source = inputs["adapter_dir"] / "adapters.safetensors"
    target = tmp_path / "borrowed.safetensors"
    source.rename(target)
    source.symlink_to(target)
    with pytest.raises(ValueError, match="regular file"):
        release.prepare_release(**inputs)


def test_test_result_needs_paired_distinct_corpus(inputs):
    with pytest.raises(ValueError, match="both baseline"):
        release.prepare_release(**inputs, base_test_metrics=inputs["base_metrics"])
    with pytest.raises(ValueError, match="must differ"):
        release.prepare_release(
            **inputs,
            base_test_metrics=inputs["base_metrics"],
            candidate_test_metrics=inputs["candidate_metrics"],
        )


def test_missing_upstream_model_license_is_not_replaced_by_mit(inputs):
    inputs["upstream_license"].write_text("MIT License")
    with pytest.raises(ValueError, match="Apache"):
        release.prepare_release(**inputs)


def assign(value, path, replacement):
    keys = path.split(".")
    for key in keys[:-1]:
        value = value[key]
    value[keys[-1]] = replacement


@pytest.mark.parametrize(
    "field",
    [
        "mlx_lm_commit",
        "mlx_lm_source_sha256",
        "python",
        "data_manifest_sha256",
        "adapter_sha256",
        "dataset_fingerprint",
        "microbatches",
        "optimizer_updates",
        "elapsed_seconds",
        "peak_memory_bytes",
        "training_performed",
        "test_evaluation_performed",
        "config.seed",
        "config.learning_rate",
        "config.mask_prompt",
        "config.grad_checkpoint",
        "versions.mlx",
        "versions.tokenizers",
    ],
)
def test_unpatterned_private_fact_rejected_in_retained_training_fields(inputs, field):
    private_fact = "unpublishedCustomerRoadmapAlpha"
    assert release.content_rules(private_fact.encode()) == set()
    change(inputs["training_receipt"], lambda d: assign(d, field, private_fact))
    with pytest.raises((ValueError, TypeError)):
        release.prepare_release(**inputs)
    assert not inputs["output"].exists()


@pytest.mark.parametrize(
    "field",
    [
        "outputs_hash",
        "cases_hash",
        "missing_outputs",
        "expected_status_matches",
        "model.id",
        "model.revision",
        "inference.seed",
        "inference.max_source_chars",
        "latency.p50_ms",
        "tokens.output_tokens.total",
        "abstention.accuracy",
        "source_evidence.authored_reference_matches",
    ],
)
def test_unpatterned_private_fact_rejected_in_retained_metrics_fields(inputs, field):
    value = json.loads(inputs["candidate_metrics"].read_text())
    private_fact = "unpublishedCustomerRoadmapAlpha"
    assert release.content_rules(private_fact.encode()) == set()
    assign(value, field, private_fact)
    with pytest.raises((ValueError, TypeError)):
        release._metrics_summary(value)


@pytest.mark.parametrize(
    "field,replacement",
    [
        (
            "model",
            {"id": "local-scraper/lora", "revision": "a" * 40, "message": "unpublishedCustomerRoadmapAlpha"},
        ),
        ("diagnostics.errors", {"UNPUBLISHED_CUSTOMER_ROADMAP_ALPHA": 1}),
        ("diagnostics.statuses", {"unpublishedCustomerRoadmapAlpha": 2}),
        ("by_schema", {"unpublishedCustomerRoadmapAlpha": 2}),
        ("source_evidence", {"unpublishedCustomerRoadmapAlpha": 2}),
        ("required_facts", {"raw": facts(), "accepted": facts(), "unpublishedCustomerRoadmapAlpha": 2}),
        ("inference.seed", {"unpublishedCustomerRoadmapAlpha": 2}),
    ],
)
def test_nested_metric_objects_have_exact_approved_keys_and_types(inputs, field, replacement):
    value = json.loads(inputs["candidate_metrics"].read_text())
    assign(value, field, replacement)
    with pytest.raises((ValueError, TypeError)):
        release._metrics_summary(value)


@pytest.mark.parametrize(
    "field", ["microbatches", "elapsed_seconds", "mlx_lm_commit", "data_manifest_sha256"]
)
def test_missing_required_training_provenance_rejected(inputs, field):
    change(inputs["training_receipt"], lambda d: d.pop(field))
    with pytest.raises(ValueError, match="Missing required"):
        release.prepare_release(**inputs)


@pytest.mark.parametrize("field", ["seed", "learning_rate", "mask_prompt", "grad_checkpoint"])
def test_missing_required_training_scalar_rejected(inputs, field):
    change(inputs["training_receipt"], lambda d: d["config"].pop(field))
    with pytest.raises(ValueError, match="Missing required"):
        release.prepare_release(**inputs)


@pytest.mark.parametrize("field", ["outputs_hash", "missing_outputs", "expected_status_matches"])
def test_missing_required_measurement_rejected(inputs, field):
    value = json.loads(inputs["candidate_metrics"].read_text())
    value.pop(field)
    with pytest.raises(ValueError, match="Missing required"):
        release._metrics_summary(value)


def test_version_field_cannot_be_an_arbitrary_private_slug_or_local_suffix(inputs):
    for version in ("privateCustomerRoadmap", "0.32.4+privateCustomerRoadmap"):
        change(inputs["training_receipt"], lambda d, version=version: d["versions"].update(mlx=version))
        with pytest.raises(ValueError, match="approved value schema"):
            release.prepare_release(**inputs)


def test_additional_private_metadata_cannot_hide_in_runtime_package_keys(inputs):
    change(
        inputs["training_receipt"], lambda d: d["versions"].update(unpublishedCustomerRoadmapAlpha="1.0.0")
    )
    with pytest.raises(ValueError, match="schema"):
        release.prepare_release(**inputs)


@pytest.mark.parametrize(
    "path", ["raw_facts.precision", "required_facts.raw.recall", "latency.measured_cases"]
)
def test_boolean_is_not_a_numeric_metric(inputs, path):
    value = json.loads(inputs["candidate_metrics"].read_text())
    assign(value, path, True)
    with pytest.raises(ValueError):
        release._metrics_summary(value)
