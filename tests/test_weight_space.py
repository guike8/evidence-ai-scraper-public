"""Mathematical and artifact controls for optional direct weight-space edits."""

from __future__ import annotations

import copy
import json
import subprocess
import sys

import pytest

from training.weight_space import (
    file_hash,
    rank_one_edit,
    read_bundle,
    validate_manifest,
    verify_base_files,
    write_bundle,
)


def test_import_does_not_initialize_optional_model_runtime():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import training.weight_space; "
                "assert 'mlx.core' not in sys.modules; assert 'numpy' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.fixture
def np():
    return pytest.importorskip("numpy")


@pytest.fixture
def bundle_args(tmp_path, np):
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"synthetic test weights, not a released model")
    return {
        "base_dir": base,
        "base_model": "example/synthetic-matrix",
        "revision": "a" * 40,
        "base_files": {"model.safetensors": file_hash(base / "model.safetensors")},
        "diagnosis": {
            "observed_failure": "A synthetic repeated output appeared in the calibration cases.",
            "rationale": "A synthetic unit direction tests artifact accounting only.",
            "split": "valid",
            "calibration_source_hashes": ["b" * 64],
            "control_source_hashes": ["c" * 64],
        },
        "edits": [
            {
                "module_path": "model.layers.2.mlp.down_proj",
                "weight": np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]),
                "direction": np.array([1.0, 0.0, 0.0]),
                "columns": [1, 3],
                "alpha": 0.01,
                "packed_tensor_hashes": {"weight": "d" * 64},
            }
        ],
    }


def test_arbitrary_projection_matches_explicit_matrix_and_keeps_unselected_columns(np):
    rng = np.random.default_rng(29)
    weight = rng.normal(size=(7, 11))
    original = weight.copy()
    direction = rng.normal(size=7)
    direction /= np.linalg.norm(direction)
    value = rank_one_edit(weight, direction, [2, 7], 0.04)
    inputs = rng.normal(size=(3, 11))
    additive = inputs @ weight.T - 0.04 * np.outer(inputs @ value["coefficients"], direction)
    np.testing.assert_allclose(additive, inputs @ (weight + value["delta"]).T, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(weight, original)
    np.testing.assert_array_equal(value["delta"][:, [0, 1, 3, 4, 5, 6, 8, 9, 10]], 0)
    assert value["changed_parameter_count"] == 14
    assert 0 < value["relative_frobenius_norm"] <= 0.05


def test_zero_dose_is_exact_identity(np):
    weight = np.arange(1.0, 13.0).reshape(3, 4)
    value = rank_one_edit(weight, [1.0, 0.0, 0.0], [0, 1], 0.0)
    np.testing.assert_array_equal(value["delta"], 0)
    np.testing.assert_array_equal(weight + value["delta"], weight)
    assert value["changed_parameter_count"] == 0
    assert value["relative_frobenius_norm"] == 0


@pytest.mark.parametrize(
    "bad",
    [
        {"direction": [2.0, 0.0, 0.0]},
        {"direction": [float("nan"), 0.0, 0.0]},
        {"direction": [1.0, 0.0]},
        {"columns": [1, 1]},
        {"columns": [True]},
        {"columns": [-1]},
        {"columns": [4]},
        {"columns": []},
        {"alpha": float("inf")},
        {"alpha": -0.01},
        {"alpha": 10.0},
        {"max_relative_norm": float("nan")},
    ],
)
def test_malformed_or_unbounded_edit_is_rejected(np, bad):
    args = {
        "weight": np.arange(1.0, 13.0).reshape(3, 4),
        "direction": [1.0, 0.0, 0.0],
        "columns": [1],
        "alpha": 0.01,
    }
    args.update(bad)
    with pytest.raises(ValueError):
        rank_one_edit(**args)


def test_bundle_roundtrip_is_private_exact_and_does_not_touch_base(tmp_path, bundle_args):
    output = tmp_path / "edit"
    before = file_hash(bundle_args["base_dir"] / "model.safetensors")
    manifest = write_bundle(output, **bundle_args)
    observed, arrays = read_bundle(output)
    assert manifest == observed
    assert set(arrays) == {"direction_000", "coefficients_000"}
    assert manifest["training"] is False
    assert manifest["base_requantized"] is False
    assert manifest["edits"][0]["changed_parameter_count"] == 2
    assert file_hash(bundle_args["base_dir"] / "model.safetensors") == before
    assert (output.stat().st_mode & 0o077) == 0
    assert ((output / "patch.npz").stat().st_mode & 0o077) == 0
    assert ((output / "manifest.json").stat().st_mode & 0o077) == 0
    with pytest.raises(FileExistsError):
        write_bundle(output, **bundle_args)
    with pytest.raises(ValueError, match="outside"):
        write_bundle(bundle_args["base_dir"] / "new-edit", **bundle_args)


def test_wrong_identity_and_corrupted_base_are_rejected(tmp_path, bundle_args):
    manifest = write_bundle(tmp_path / "edit", **bundle_args)
    args = {key: bundle_args[key] for key in ("base_dir", "base_model", "revision")}
    verify_base_files(manifest, **args)
    with pytest.raises(ValueError, match="revision"):
        verify_base_files(manifest, **{**args, "revision": "f" * 40})
    with pytest.raises(ValueError, match="repository"):
        verify_base_files(manifest, **{**args, "base_model": "other/model"})
    (bundle_args["base_dir"] / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_base_files(manifest, **args)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda m: m.update(extra="surprise"),
        lambda m: m.update(training=True),
        lambda m: m.update(base_requantized=True),
        lambda m: m["base_files"].update({"../model.safetensors": "a" * 64}),
        lambda m: m["diagnosis"].update(split="test"),
        lambda m: m["diagnosis"].update(rationale="Inspect /private/example/model for details"),
        lambda m: m["diagnosis"].update(control_source_hashes=["b" * 64]),
        lambda m: m["edits"][0].update(relative_frobenius_norm=0.2),
        lambda m: m["edits"][0].update(packed_tensor_hashes={"weight": "d" * 64, "scales": "e" * 64}),
        lambda m: m["edits"].append(copy.deepcopy(m["edits"][0])),
    ],
)
def test_manifest_rejects_untrusted_fields_paths_and_test_leakage(tmp_path, bundle_args, mutation):
    manifest = write_bundle(tmp_path / "edit", **bundle_args)
    mutation(manifest)
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_corrupted_npz_and_duplicate_json_keys_are_rejected(tmp_path, bundle_args):
    output = tmp_path / "edit"
    write_bundle(output, **bundle_args)
    with (output / "patch.npz").open("ab") as stream:
        stream.write(b"extra")
    with pytest.raises(ValueError, match="checksum"):
        read_bundle(output)
    (output / "manifest.json").write_text('{"format":"a","format":"b"}')
    with pytest.raises(ValueError, match="Duplicate"):
        read_bundle(output)


def test_nonfinite_or_out_of_scope_arrays_are_rejected_even_with_updated_file_hash(tmp_path, bundle_args, np):
    output = tmp_path / "edit"
    manifest = write_bundle(output, **bundle_args)
    np.savez(
        output / "patch.npz", direction_000=np.array([float("nan"), 0.0, 0.0]), coefficients_000=np.zeros(4)
    )
    manifest["arrays_sha256"] = file_hash(output / "patch.npz")
    (output / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="finite"):
        read_bundle(output)
