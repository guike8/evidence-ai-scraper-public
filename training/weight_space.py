"""Auditable, non-gradient rank-one edits on a diagnosed model projection.

This implements an arithmetic container, not a reproduction of the article's
results. No layer, direction, dose or failure is inferred automatically. Select
an edit on calibration data first, freeze it, and evaluate unseen cases later.
NumPy and MLX are optional training dependencies, imported only when needed.

Method reference: https://rikiyat.github.io/blog/weight-space-programming/
This code is independently implemented; no published edit directions are bundled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import zipfile
from pathlib import Path
from typing import Any

FORMAT = "evidence-wsp-rank-one-v1"
FORMULA = "W_new = W - alpha * outer(unit_direction, mask * (unit_direction @ W))"
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_MODULE = re.compile(r"(?:[A-Za-z][A-Za-z0-9_]*|[0-9]+)(?:\.(?:[A-Za-z][A-Za-z0-9_]*|[0-9]+))*\Z")
_MANIFEST_KEYS = {
    "format",
    "formula",
    "base_model",
    "revision",
    "base_files",
    "diagnosis",
    "max_relative_norm",
    "edits",
    "arrays_sha256",
    "training",
    "base_requantized",
}
_EDIT_KEYS = {
    "module_path",
    "weight_shape",
    "weight_sha256",
    "packed_tensor_hashes",
    "columns",
    "alpha",
    "direction_key",
    "coefficients_key",
    "direction_sha256",
    "coefficients_sha256",
    "relative_frobenius_norm",
    "changed_parameter_count",
}
_DIAGNOSIS_KEYS = {
    "observed_failure",
    "rationale",
    "split",
    "calibration_source_hashes",
    "control_source_hashes",
}


def _numpy():
    import numpy as np

    return np


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_hash(value, *, logical_dtype: str | None = None) -> str:
    """Hash shape, dtype and C-order bytes; no values are printed."""
    np = _numpy()
    array = np.asarray(value)
    header = _json({"shape": list(array.shape), "dtype": logical_dtype or str(array.dtype)})
    return hashlib.sha256(header.encode() + np.ascontiguousarray(array).tobytes()).hexdigest()


def _sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA.fullmatch(value) is not None


def _finite(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _safe_text(value: Any) -> bool:
    # Do not embed host paths, credentials or URLs into a distributable manifest.
    return (
        isinstance(value, str)
        and 10 <= len(value) <= 2000
        and not re.search(r"(?:^|\s)(?:/[^\s]+|[A-Za-z]:\\)", value)
        and not re.search(r"https?://|hf_[A-Za-z0-9]{8,}|Bearer\s+", value)
    )


def _safe_filename(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", value) is not None


def _validate_diagnosis(diagnosis: Any) -> None:
    if not isinstance(diagnosis, dict) or set(diagnosis) != _DIAGNOSIS_KEYS:
        raise ValueError("Diagnosis must contain exactly the documented calibration metadata")
    if diagnosis["split"] not in {"train", "valid"}:
        raise ValueError("Edits cannot be selected using held-out test cases")
    if not all(_safe_text(diagnosis[k]) for k in ("observed_failure", "rationale")):
        raise ValueError("Diagnosis needs a bounded explanation without host paths or credentials")
    for key in ("calibration_source_hashes", "control_source_hashes"):
        hashes = diagnosis[key]
        if not isinstance(hashes, list) or not hashes or not all(_sha(h) for h in hashes):
            raise ValueError("Diagnosis requires nonempty calibration and control source hashes")
        if len(set(hashes)) != len(hashes):
            raise ValueError("Diagnostic source hashes must be unique")
    if set(diagnosis["calibration_source_hashes"]) & set(diagnosis["control_source_hashes"]):
        raise ValueError("Calibration and independent control sources must be disjoint")


def rank_one_edit(weight, direction, columns, alpha: float, *, max_relative_norm: float = 0.05):
    """Return unit u, sparse c and measured delta, without mutating W.

    The direction must already be unit length, rather than silently normalizing
    an operator's malformed input. The Frobenius bound is per target matrix.
    """
    np = _numpy()
    weight = np.asarray(weight, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    if weight.ndim != 2 or not all(weight.shape) or direction.shape != (weight.shape[0],):
        raise ValueError("Expected W=(out,in) and u=(out,)")
    if not np.isfinite(weight).all() or not np.isfinite(direction).all():
        raise ValueError("Weights and direction must be finite")
    if not np.isclose(np.linalg.norm(direction), 1.0, rtol=0, atol=1e-5):
        raise ValueError("Direction must have unit norm")
    if not _finite(alpha) or alpha < 0 or not _finite(max_relative_norm) or not 0 < max_relative_norm <= 1:
        raise ValueError("Dose and norm bound must be finite and nonnegative")
    columns = list(columns)
    if not columns or any(type(c) is not int or c < 0 or c >= weight.shape[1] for c in columns):
        raise ValueError("Columns must be explicit, nonempty, in-range integer indices")
    if len(set(columns)) != len(columns):
        raise ValueError("Column indices must be unique")
    coefficients = np.zeros(weight.shape[1], dtype=np.float64)
    coefficients[columns] = direction @ weight[:, columns]
    delta = -alpha * np.outer(direction, coefficients)
    norm = float(np.linalg.norm(weight))
    if (
        not np.isfinite(coefficients).all()
        or not np.isfinite(delta).all()
        or not math.isfinite(norm)
        or norm == 0
    ):
        raise ValueError("Projection and weight norm must be finite and nonzero")
    relative = float(np.linalg.norm(delta) / norm)
    if not math.isfinite(relative) or relative > max_relative_norm + 1e-12:
        raise ValueError("Edit exceeds the declared relative Frobenius norm bound")
    return {
        "direction": direction,
        "coefficients": coefficients,
        "delta": delta,
        "relative_frobenius_norm": relative,
        "changed_parameter_count": int(np.count_nonzero(delta)),
    }


def validate_manifest(manifest: dict) -> None:
    """Validate bounded public metadata without importing NumPy or MLX."""
    if not isinstance(manifest, dict) or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("Unexpected WSP manifest fields")
    if manifest["format"] != FORMAT or manifest["formula"] != FORMULA:
        raise ValueError("Unsupported WSP format or formula")
    if manifest["training"] is not False or manifest["base_requantized"] is not False:
        raise ValueError("WSP bundles must declare non-gradient editing without requantization")
    if not isinstance(manifest["base_model"], str) or not re.fullmatch(
        r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", manifest["base_model"]
    ):
        raise ValueError("An explicit model repository is required")
    if not isinstance(manifest["revision"], str) or not _REVISION.fullmatch(manifest["revision"]):
        raise ValueError("An exact 40-character base revision is required")
    files = manifest["base_files"]
    if (
        not isinstance(files, dict)
        or not files
        or not all(_safe_filename(k) and _sha(v) for k, v in files.items())
        or not any(k.endswith(".safetensors") for k in files)
    ):
        raise ValueError("Safe base filenames and SHA256s, including weights, are required")
    if not _sha(manifest["arrays_sha256"]):
        raise ValueError("Missing arrays checksum")
    bound = manifest["max_relative_norm"]
    if not _finite(bound) or not 0 < bound <= 1:
        raise ValueError("Invalid Frobenius norm bound")
    _validate_diagnosis(manifest["diagnosis"])
    edits = manifest["edits"]
    if not isinstance(edits, list) or not 1 <= len(edits) <= 32:
        raise ValueError("A bounded explicit set of target modules is required")
    modules, array_keys = set(), set()
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != _EDIT_KEYS:
            raise ValueError("Unexpected edit fields")
        module = edit["module_path"]
        if not isinstance(module, str) or not _MODULE.fullmatch(module) or len(module) > 200:
            raise ValueError("Invalid module path")
        if module in modules:
            raise ValueError("A module cannot be edited twice in one bundle")
        modules.add(module)
        shape = edit["weight_shape"]
        if not isinstance(shape, list) or len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape):
            raise ValueError("Invalid target matrix shape")
        if (
            not _sha(edit["weight_sha256"])
            or not _sha(edit["direction_sha256"])
            or not _sha(edit["coefficients_sha256"])
        ):
            raise ValueError("Missing target and patch tensor checksums")
        packed = edit["packed_tensor_hashes"]
        if (
            not isinstance(packed, dict)
            or "weight" not in packed
            or not all(k in {"weight", "scales", "biases", "bias"} and _sha(v) for k, v in packed.items())
        ):
            raise ValueError("Packed target tensor checksums are required")
        if ("scales" in packed) != ("biases" in packed):
            raise ValueError("Quantized targets require both scales and biases")
        cols = edit["columns"]
        if (
            not isinstance(cols, list)
            or not cols
            or any(type(c) is not int or c < 0 or c >= shape[1] for c in cols)
            or len(cols) != len(set(cols))
        ):
            raise ValueError("Invalid edit columns")
        alpha, norm = edit["alpha"], edit["relative_frobenius_norm"]
        if not _finite(alpha) or alpha < 0 or not _finite(norm) or not 0 <= norm <= bound + 1e-12:
            raise ValueError("Invalid dose or measured norm")
        changed = edit["changed_parameter_count"]
        if type(changed) is not int or not 0 <= changed <= shape[0] * len(cols):
            raise ValueError("Invalid changed-parameter count")
        for key in ("direction_key", "coefficients_key"):
            name = edit[key]
            if not isinstance(name, str) or not re.fullmatch(r"(?:direction|coefficients)_[0-9]{3}", name):
                raise ValueError("Invalid patch array name")
            if name in array_keys:
                raise ValueError("Patch arrays cannot be aliased")
            array_keys.add(name)


def write_bundle(
    output_dir: Path,
    *,
    base_dir: Path,
    base_model: str,
    revision: str,
    base_files: dict[str, str],
    diagnosis: dict,
    edits: list[dict],
    max_relative_norm: float = 0.05,
) -> dict:
    """Write a new private bundle; never overwrite a model or existing output."""
    np = _numpy()
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("WSP output must be a new directory outside the base model")
    base_dir = Path(base_dir).resolve()
    if output_dir.resolve() == base_dir or base_dir in output_dir.resolve().parents:
        raise ValueError("Keep WSP bundles outside the base model directory")
    manifest = {
        "format": FORMAT,
        "formula": FORMULA,
        "base_model": base_model,
        "revision": revision,
        "base_files": dict(base_files),
        "diagnosis": diagnosis,
        "max_relative_norm": max_relative_norm,
        "edits": [],
        "arrays_sha256": "0" * 64,
        "training": False,
        "base_requantized": False,
    }
    arrays = {}
    for index, edit in enumerate(edits):
        values = rank_one_edit(
            edit["weight"],
            edit["direction"],
            edit["columns"],
            edit["alpha"],
            max_relative_norm=max_relative_norm,
        )
        ukey, ckey = f"direction_{index:03d}", f"coefficients_{index:03d}"
        arrays[ukey], arrays[ckey] = values["direction"], values["coefficients"]
        manifest["edits"].append(
            {
                "module_path": edit["module_path"],
                "weight_shape": list(np.asarray(edit["weight"]).shape),
                "weight_sha256": tensor_hash(np.asarray(edit["weight"], dtype=np.float64)),
                "packed_tensor_hashes": dict(edit["packed_tensor_hashes"]),
                "columns": list(edit["columns"]),
                "alpha": edit["alpha"],
                "direction_key": ukey,
                "coefficients_key": ckey,
                "direction_sha256": tensor_hash(arrays[ukey]),
                "coefficients_sha256": tensor_hash(arrays[ckey]),
                "relative_frobenius_norm": values["relative_frobenius_norm"],
                "changed_parameter_count": values["changed_parameter_count"],
            }
        )
    validate_manifest(manifest)
    verify_base_files(manifest, base_dir, base_model=base_model, revision=revision)
    output_dir.mkdir(parents=True, mode=0o700)
    with (output_dir / "patch.npz").open("xb") as stream:
        os.fchmod(stream.fileno(), 0o600)
        np.savez(stream, **arrays)
    manifest["arrays_sha256"] = file_hash(output_dir / "patch.npz")
    with (output_dir / "manifest.json").open("x") as stream:
        os.fchmod(stream.fileno(), 0o600)
        stream.write(_json(manifest))
    return manifest


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate manifest JSON key")
        result[key] = value
    return result


def read_bundle(bundle_dir: Path) -> tuple[dict, dict]:
    np = _numpy()
    bundle_dir = Path(bundle_dir)
    if (bundle_dir / "manifest.json").stat().st_size > 256_000:
        raise ValueError("Oversized WSP manifest")
    manifest = json.loads((bundle_dir / "manifest.json").read_text(), object_pairs_hook=_unique_pairs)
    validate_manifest(manifest)
    arrays_path = bundle_dir / "patch.npz"
    if arrays_path.stat().st_size > 64_000_000 or file_hash(arrays_path) != manifest["arrays_sha256"]:
        raise ValueError("Patch array checksum or size mismatch")
    expected = {edit[key] for edit in manifest["edits"] for key in ("direction_key", "coefficients_key")}
    with zipfile.ZipFile(arrays_path) as archive:
        entries = archive.infolist()
        if (
            len(entries) != len(expected)
            or {entry.filename for entry in entries} != {key + ".npy" for key in expected}
            or any(entry.file_size > 64_000_000 for entry in entries)
            or sum(entry.file_size for entry in entries) > 64_000_000
        ):
            raise ValueError("Unexpected or oversized uncompressed patch arrays")
    with np.load(arrays_path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("Unexpected patch arrays")
        arrays = {key: archive[key] for key in archive.files}
    for edit in manifest["edits"]:
        u, c = arrays[edit["direction_key"]], arrays[edit["coefficients_key"]]
        if (
            u.dtype != np.float64
            or c.dtype != np.float64
            or u.shape != (edit["weight_shape"][0],)
            or (c.shape != (edit["weight_shape"][1],))
        ):
            raise ValueError("Patch tensor shape or dtype mismatch")
        if (
            not np.isfinite(u).all()
            or not np.isfinite(c).all()
            or not np.isclose(np.linalg.norm(u), 1.0, rtol=0, atol=1e-5)
        ):
            raise ValueError("Patch tensors must be finite with a unit direction")
        if tensor_hash(u) != edit["direction_sha256"] or tensor_hash(c) != edit["coefficients_sha256"]:
            raise ValueError("Patch tensor checksum mismatch")
        unselected = np.ones(c.size, dtype=bool)
        unselected[edit["columns"]] = False
        if np.any(c[unselected] != 0):
            raise ValueError("Coefficients modify undeclared columns")
    return manifest, arrays


def verify_base_files(manifest: dict, base_dir: Path, *, base_model: str, revision: str) -> None:
    validate_manifest(manifest)
    if base_model != manifest["base_model"] or revision != manifest["revision"]:
        raise ValueError("Base repository or revision mismatch")
    base_dir = Path(base_dir).resolve()
    for name, expected in manifest["base_files"].items():
        path = base_dir / name
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError("Base file checksum mismatch")


def _mlx_tensor_hash(tensor):
    import mlx.core as mx

    original_dtype = str(tensor.dtype)
    if tensor.dtype == mx.bfloat16:
        tensor = tensor.astype(mx.float32)
    return tensor_hash(_numpy().asarray(tensor), logical_dtype=original_dtype)


def packed_tensor_hashes(module) -> dict[str, str]:
    """Hashes include quantization metadata tensors without dequantizing."""
    return {
        name: _mlx_tensor_hash(getattr(module, name))
        for name in ("weight", "scales", "biases", "bias")
        if name in module
    }


def apply_bundle(model, bundle_dir: Path, *, base_dir: Path, base_model: str, revision: str) -> dict:
    """Install frozen additive wrappers, keeping all original tensors intact.

    Call with a freshly loaded base, before generation. Applying an edit twice,
    layering it over trained adapters, or using a different base is rejected.
    """
    manifest, arrays = read_bundle(bundle_dir)
    verify_base_files(manifest, base_dir, base_model=base_model, revision=revision)
    base_dir, bundle_dir = Path(base_dir).resolve(), Path(bundle_dir).resolve()
    if bundle_dir == base_dir or base_dir in bundle_dir.parents:
        raise ValueError("Keep WSP bundles outside the base model directory")
    import mlx.core as mx
    from mlx import nn
    from mlx.utils import tree_flatten, tree_unflatten
    from mlx_lm.tuner.lora import LoRALinear

    np = _numpy()
    modules = dict(model.named_modules())
    if any(isinstance(module, LoRALinear) for module in modules.values()):
        raise ValueError("WSP requires an untouched base without existing additive adapters")
    before = {name: _mlx_tensor_hash(value) for name, value in tree_flatten(model.parameters())}
    replacements, target_paths, runtime_patches = [], [], {}
    for edit in manifest["edits"]:
        path = edit["module_path"]
        original = modules.get(path)
        if not isinstance(original, (nn.Linear, nn.QuantizedLinear)):
            raise TypeError("Target is missing or is not a supported linear projection")
        if packed_tensor_hashes(original) != edit["packed_tensor_hashes"]:
            raise ValueError("Loaded target packed tensor hashes differ from the diagnosed base")
        if isinstance(original, nn.QuantizedLinear):
            patch_dtype = original.scales.dtype
            weight = mx.dequantize(
                original.weight,
                original.scales,
                original.biases,
                group_size=original.group_size,
                bits=original.bits,
                mode=original.mode,
            )
        else:
            patch_dtype = original.weight.dtype
            weight = original.weight
        if patch_dtype not in {mx.float16, mx.bfloat16, mx.float32}:
            raise TypeError("Target projection requires a supported floating-point computation dtype")
        weight = np.asarray(weight.astype(mx.float32), dtype=np.float64)
        if list(weight.shape) != edit["weight_shape"] or tensor_hash(weight) != edit["weight_sha256"]:
            raise ValueError("Dequantized target checksum or shape mismatch")
        values = rank_one_edit(
            weight,
            arrays[edit["direction_key"]],
            edit["columns"],
            edit["alpha"],
            max_relative_norm=manifest["max_relative_norm"],
        )
        if (
            not np.array_equal(values["coefficients"], arrays[edit["coefficients_key"]])
            or (values["changed_parameter_count"] != edit["changed_parameter_count"])
            or not math.isclose(
                values["relative_frobenius_norm"], edit["relative_frobenius_norm"], abs_tol=1e-12
            )
        ):
            raise ValueError("Stored edit does not match recomputed bounded projection")
        wrapper = LoRALinear.from_base(original, r=1, scale=1.0, dropout=0.0)
        # from_base initializes float32 adapters even for a BF16 projection.
        # Keep the patch in the original projection's floating-point dtype;
        # LoRALinear additionally casts the additive term back to x.dtype.
        wrapper.lora_a = mx.array(values["coefficients"].astype(np.float32), dtype=patch_dtype)[:, None]
        wrapper.lora_b = mx.array(
            (-edit["alpha"] * values["direction"]).astype(np.float32), dtype=patch_dtype
        )[None, :]
        if not bool(mx.all(mx.isfinite(wrapper.lora_a)).item()) or not bool(
            mx.all(mx.isfinite(wrapper.lora_b)).item()
        ):
            raise ValueError("Patch overflows the original projection's computation dtype")
        rounded_a = np.asarray(wrapper.lora_a.astype(mx.float32), dtype=np.float64)
        rounded_b = np.asarray(wrapper.lora_b.astype(mx.float32), dtype=np.float64)
        rounded_relative_norm = float(np.linalg.norm(rounded_b.T @ rounded_a.T) / np.linalg.norm(weight))
        if rounded_relative_norm > manifest["max_relative_norm"] + 1e-12:
            raise ValueError("Rounded patch exceeds the declared relative Frobenius norm bound")
        runtime_patches[path] = {
            "dtype": str(patch_dtype),
            "rounded_relative_frobenius_norm": rounded_relative_norm,
        }
        wrapper.freeze()
        replacements.append((path, wrapper))
        target_paths.append(path)
    model.update_modules(tree_unflatten(replacements))
    model.freeze()
    model.eval()
    after = dict(tree_flatten(model.parameters()))
    # The wrapper inserts '.linear' into each selected original tensor path.
    for name, digest in before.items():
        mapped = name
        for path in target_paths:
            if name.startswith(path + "."):
                mapped = path + ".linear" + name[len(path) :]
                break
        if mapped not in after or _mlx_tensor_hash(after[mapped]) != digest:
            raise RuntimeError("An original base tensor changed while installing the patch")
    if tree_flatten(model.trainable_parameters()):
        raise RuntimeError("WSP inference must have no trainable parameters")
    return {
        "format": FORMAT,
        "bundle_sha256": file_hash(bundle_dir / "manifest.json"),
        "modules": target_paths,
        "runtime_patches": runtime_patches,
        "base_tensor_count": len(before),
        "base_tensors_unchanged": True,
        "training": False,
        "base_requantized": False,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle_dir", type=Path, help="Inspect an existing operator-authored WSP bundle")
    args = parser.parse_args(argv)
    manifest, _ = read_bundle(args.bundle_dir)
    print(_json(manifest), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
