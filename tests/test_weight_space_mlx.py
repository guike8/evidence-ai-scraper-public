"""Opt-in Apple GPU arithmetic controls; never loads a language model."""

from __future__ import annotations

import os

import pytest

from training.weight_space import apply_bundle, file_hash, packed_tensor_hashes, write_bundle

pytestmark = pytest.mark.skipif(os.environ.get("RUN_MLX_TESTS") != "1", reason="Optional MLX GPU checks")


@pytest.mark.parametrize("alpha", [0.0, 0.02])
@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_frozen_quantized_patch_preserves_base_and_matches_additive_formula(tmp_path, alpha, dtype_name):
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")
    np = pytest.importorskip("numpy")
    from mlx.utils import tree_flatten

    dtype = getattr(mx, dtype_name)

    class ToyModel(nn.Module):
        def __init__(self):
            super().__init__()
            linear = nn.Linear(64, 8, bias=False)
            linear.weight = linear.weight.astype(dtype)
            self.projection = nn.QuantizedLinear.from_linear(linear, group_size=64, bits=4)

        def __call__(self, x):
            return self.projection(x)

    mx.random.seed(31)
    model = ToyModel()
    original = model.projection
    before = packed_tensor_hashes(original)
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    mx.save_safetensors(str(base_dir / "model.safetensors"), dict(tree_flatten(model.parameters())))
    weight = np.asarray(
        mx.dequantize(
            original.weight,
            original.scales,
            original.biases,
            group_size=original.group_size,
            bits=original.bits,
            mode=original.mode,
        ).astype(mx.float32)
    )
    direction = np.arange(1.0, 9.0)
    direction /= np.linalg.norm(direction)
    bundle = tmp_path / "wsp"
    write_bundle(
        bundle,
        base_dir=base_dir,
        base_model="example/quantized-toy",
        revision="a" * 40,
        base_files={"model.safetensors": file_hash(base_dir / "model.safetensors")},
        diagnosis={
            "observed_failure": "Synthetic arithmetic control only; no language-model failure is claimed.",
            "rationale": "Check a diagnosed-projection container preserves quantization and matrix semantics.",
            "split": "valid",
            "calibration_source_hashes": ["b" * 64],
            "control_source_hashes": ["c" * 64],
        },
        edits=[
            {
                "module_path": "projection",
                "weight": weight,
                "direction": direction,
                "columns": [3, 5, 17],
                "alpha": alpha,
                "packed_tensor_hashes": before,
            }
        ],
    )
    inputs = mx.concatenate((mx.eye(64), mx.random.normal((4, 64)))).astype(dtype)
    base_result = model(inputs)
    base_output = np.asarray(base_result.astype(mx.float32))
    base_bits = np.asarray(base_result.view(mx.uint8))
    result = apply_bundle(
        model,
        bundle,
        base_dir=base_dir,
        base_model="example/quantized-toy",
        revision="a" * 40,
    )
    assert result["base_tensors_unchanged"]
    assert result["runtime_patches"]["projection"]["dtype"] == str(dtype)
    assert packed_tensor_hashes(model.projection.linear) == before
    assert tree_flatten(model.trainable_parameters()) == []
    assert model.projection.lora_a.dtype == model.projection.lora_b.dtype == dtype
    c = np.zeros(64)
    c[[3, 5, 17]] = direction @ weight[:, [3, 5, 17]]
    expected = base_output - alpha * np.outer(np.asarray(inputs.astype(mx.float32)) @ c, direction)
    result_array = model(inputs)
    observed = np.asarray(result_array.astype(mx.float32))
    assert result_array.dtype == base_result.dtype == dtype
    tolerance = {"rtol": 0.01, "atol": 0.003} if dtype_name == "bfloat16" else {"rtol": 2e-5, "atol": 2e-6}
    np.testing.assert_allclose(observed, expected, **tolerance)
    if alpha == 0:
        np.testing.assert_array_equal(np.asarray(result_array.view(mx.uint8)), base_bits)
    else:
        assert np.max(np.abs(observed - base_output)) > 0
    with pytest.raises(ValueError, match="untouched base"):
        apply_bundle(model, bundle, base_dir=base_dir, base_model="example/quantized-toy", revision="a" * 40)
