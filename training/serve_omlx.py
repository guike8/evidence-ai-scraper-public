"""Experimental foreground oMLX serving for the verified unfused MLX adapter.

This uses unsupported, process-local oMLX hooks. It does not patch installed
files, train/fuse/requantize weights, change OS settings, or manage other
services. Heavy serving dependencies are imported only after explicit opt-in,
local artifact checks, a private state lock and exact backend-version checks.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import functools
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
import re
import socket
import stat
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from .job import offline_environment
from .train import BASE_ID, BASE_REVISION, BASE_WEIGHT_SHA256, LORA_PARAMETERS, verify_base

TESTED_BACKENDS = {
    "omlx": "0.7.3rc1",
    "mlx": "0.32.4",
    "mlx-lm": "0.32.3.dev8+g0d4d3d04f",
    "xgrammar": "0.2.8",
}
PACKAGE_DIRECTORIES = {"omlx": "omlx", "mlx": "mlx", "mlx-lm": "mlx_lm", "xgrammar": "xgrammar"}
TESTED_PACKAGE_IDENTITIES = {
    "omlx": {
        "file_count": 641,
        "total_bytes": 105358682,
        "sha256": "4a2955e933c7f5efe26c8282bd0747a4969477d032d4627729005b6dc0408709",
    },
    "mlx": {
        "file_count": 452,
        "total_bytes": 243821514,
        "sha256": "fbea43bc9098d1c93c44b7d6ad3eda22e8fd2e13684c6a6cb0c8078c57bf9ce2",
    },
    "mlx-lm": {
        "file_count": 209,
        "total_bytes": 2036132,
        "sha256": "5d099ec02f044540cb32b8721bec4b0f0fced01168fa17592bc1508882a9f7f4",
    },
    "xgrammar": {
        "file_count": 40,
        "total_bytes": 4059583,
        "sha256": "06be1e91fd5b9260e5c60c6f0c88d67def919d75bded0cab81bfe37f4c2b86e5",
    },
}
MLX_LM_SERVING_URL = "https://github.com/ml-explore/mlx-lm"
MLX_LM_SERVING_COMMIT = "0d4d3d04f8ab7ce4a51e7209e8615755a48ba94d"
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
MAX_ADAPTER_BYTES = 32 * 1024**2
MAX_HEADER_BYTES = 65536
MAX_PACKAGE_FILES = 20000
MAX_PACKAGE_FILE_BYTES = 2 * 1024**3
MAX_PACKAGE_TOTAL_BYTES = 4 * 1024**3
MEMORY_BYTES = 8 * 1024**3
REPOSITORY = Path(__file__).resolve().parents[1]


class ServingError(ValueError):
    """Stable codes only; callers must not expose third-party exceptions."""


def _require(condition, code):
    if not condition:
        raise ServingError(code)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _unique(items):
    result = {}
    for key, value in items:
        _require(key not in result, "SERVE_INVALID_ADAPTER")
        result[key] = value
    return result


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@contextlib.contextmanager
def _regular_input(path: Path, limit: int, *, allow_empty=False):
    """Open one bounded regular inode, refusing links, replacement and growth."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        minimum = 0 if allow_empty else 1
        _require(stat.S_ISREG(before.st_mode) and minimum <= before.st_size <= limit, "SERVE_UNSAFE_INPUT")
        _require(_file_identity(path.lstat()) == _file_identity(before), "SERVE_INPUT_CHANGED")
        yield descriptor, before.st_size
        _require(
            _file_identity(os.fstat(descriptor)) == _file_identity(before)
            and _file_identity(path.lstat()) == _file_identity(before),
            "SERVE_INPUT_CHANGED",
        )
    finally:
        os.close(descriptor)


def _read_regular(path: Path, limit: int) -> bytes:
    with _regular_input(path, limit) as (descriptor, size):
        chunks, count = [], 0
        while chunk := os.read(descriptor, min(1024**2, limit + 1 - count)):
            chunks.append(chunk)
            count += len(chunk)
            _require(count <= limit, "SERVE_UNSAFE_INPUT")
        _require(count == size, "SERVE_INPUT_CHANGED")
        result = b"".join(chunks)
    return result


def _regular_hash(path: Path, limit: int) -> tuple[str, int]:
    digest, count = hashlib.sha256(), 0
    with _regular_input(path, limit, allow_empty=True) as (descriptor, size):
        while chunk := os.read(descriptor, min(1024**2, limit + 1 - count)):
            count += len(chunk)
            _require(count <= limit, "SERVE_UNSAFE_INPUT")
            digest.update(chunk)
        _require(count == size, "SERVE_INPUT_CHANGED")
    return digest.hexdigest(), count


def expected_tensor_shapes():
    return {
        f"model.layers.{layer}.self_attn.{projection}.lora_{letter}": shape
        for layer in range(20, 28)
        for projection in ("q_proj", "v_proj")
        for letter, shape in (
            ("a", [2048, 8]),
            ("b", [8, 2048 if projection == "q_proj" else 1024]),
        )
    }


def verify_tensor_file(path: Path, expected_sha: str) -> dict:
    return _verify_tensor_bytes(_read_regular(path, MAX_ADAPTER_BYTES), expected_sha)


def _verify_tensor_bytes(body: bytes, expected_sha: str) -> dict:
    """Validate the small safetensors artifact on CPU before importing MLX."""
    try:
        size = len(body)
        _require(8 < size <= MAX_ADAPTER_BYTES, "SERVE_INVALID_ADAPTER")
        _require(
            len(body) == size and hashlib.sha256(body).hexdigest() == expected_sha, "SERVE_ADAPTER_CHANGED"
        )
        length = struct.unpack("<Q", body[:8])[0]
        _require(2 <= length <= MAX_HEADER_BYTES and 8 + length < size, "SERVE_INVALID_ADAPTER")
        header = json.loads(body[8 : 8 + length], object_pairs_hook=_unique)
        _require(isinstance(header, dict), "SERVE_INVALID_ADAPTER")
        tensors = {key: value for key, value in header.items() if key != "__metadata__"}
        expected = expected_tensor_shapes()
        _require(set(tensors) == set(expected), "SERVE_INCOMPATIBLE_TENSORS")
        ranges, parameters = [], 0
        for key, shape in expected.items():
            tensor = tensors[key]
            _require(isinstance(tensor, dict), "SERVE_INVALID_ADAPTER")
            _require(
                tensor.get("dtype") == "F32" and tensor.get("shape") == shape, "SERVE_INCOMPATIBLE_TENSORS"
            )
            offsets = tensor.get("data_offsets")
            _require(
                isinstance(offsets, list)
                and len(offsets) == 2
                and all(type(value) is int for value in offsets),
                "SERVE_INVALID_ADAPTER",
            )
            start, end = offsets
            count = math.prod(shape)
            _require(
                0 <= start < end <= size - 8 - length and end - start == count * 4, "SERVE_INVALID_ADAPTER"
            )
            raw = memoryview(body)[8 + length + start : 8 + length + end]
            _require(
                all(math.isfinite(item[0]) for item in struct.iter_unpack("<f", raw)),
                "SERVE_NONFINITE_ADAPTER",
            )
            ranges.append((start, end))
            parameters += count
        ranges.sort()
        _require(
            ranges[0][0] == 0
            and ranges[-1][1] == size - 8 - length
            and all(left[1] == right[0] for left, right in itertools.pairwise(ranges)),
            "SERVE_INVALID_ADAPTER",
        )
        _require(parameters == 458752, "SERVE_INCOMPATIBLE_TENSORS")
        return {"tensor_count": 32, "parameters": parameters, "dtype": "F32", "finite": True}
    except ServingError:
        raise
    except (OSError, ValueError, TypeError, KeyError, struct.error, UnicodeError, RecursionError):
        raise ServingError("SERVE_INVALID_ADAPTER") from None


def verify_adapter_artifact(adapter: Path, expected_sha: str, base_identity: dict) -> tuple[dict, dict]:
    """Parse each bounded captured input once; never reopen receipt bytes."""
    weights = _read_regular(adapter / "adapters.safetensors", MAX_ADAPTER_BYTES)
    tensors = _verify_tensor_bytes(weights, expected_sha)
    config_bytes = _read_regular(adapter / "adapter_config.json", 65536)
    config = json.loads(config_bytes, object_pairs_hook=_unique)
    _require(
        isinstance(config, dict)
        and config.get("fine_tune_type") == "lora"
        and type(config.get("num_layers")) is int
        and config["num_layers"] == 8
        and config.get("lora_parameters") == LORA_PARAMETERS,
        "SERVE_INCOMPATIBLE_ADAPTER_CONFIG",
    )
    candidates = (adapter.parent / "train.json", adapter / "training.json")
    receipt_path = next(path for path in candidates if path.exists() or path.is_symlink())
    receipt_bytes = _read_regular(receipt_path, 262144)
    receipt = json.loads(receipt_bytes, object_pairs_hook=_unique)
    _require(isinstance(receipt, dict), "SERVE_INVALID_ADAPTER")
    _require(receipt.get("base") == base_identity, "SERVE_TOKENIZER_OR_CONFIG_CHANGED")
    _require(
        receipt.get("adapter_sha256") == expected_sha
        and base_identity.get("repo_id") == BASE_ID
        and base_identity.get("revision") == BASE_REVISION
        and base_identity.get("files", {}).get("model.safetensors") == BASE_WEIGHT_SHA256,
        "SERVE_INCOMPATIBLE_BASE_RECEIPT",
    )
    return {
        "adapter_sha256": expected_sha,
        "adapter_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "training_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "base_weight_sha256": BASE_WEIGHT_SHA256,
        "optimizer_state_loaded": False,
    }, tensors


@dataclass(frozen=True)
class ServeConfig:
    base: Path
    adapter: Path
    adapter_sha256: str
    state: Path
    model_name: str
    port: int


def validate_inputs(
    *, base, adapter_dir, adapter_sha256, state_dir, model_name, port, enabled
) -> tuple[ServeConfig, dict]:
    _require(enabled is True, "SERVE_EXPLICIT_OPT_IN_REQUIRED")
    _require(isinstance(adapter_sha256, str) and _SHA.fullmatch(adapter_sha256), "SERVE_INVALID_SHA256")
    _require(isinstance(model_name, str) and _NAME.fullmatch(model_name), "SERVE_INVALID_MODEL_NAME")
    _require(type(port) is int and 1024 <= port <= 65535, "SERVE_INVALID_PORT")
    _require(
        all(isinstance(value, (str, Path)) for value in (base, adapter_dir, state_dir)), "SERVE_UNSAFE_PATHS"
    )
    base, adapter, state = (Path(value).expanduser() for value in (base, adapter_dir, state_dir))
    _require(not state.is_symlink(), "SERVE_UNSAFE_STATE")
    base, adapter, state = base.resolve(), adapter.resolve(), state.resolve()
    _require(base.is_dir() and adapter.is_dir(), "SERVE_LOCAL_FILES_REQUIRED")
    _require(
        base != adapter and not base.is_relative_to(adapter) and not adapter.is_relative_to(base),
        "SERVE_UNSAFE_PATHS",
    )
    for protected in (base, adapter, REPOSITORY):
        _require(
            not state.is_relative_to(protected) and not protected.is_relative_to(state), "SERVE_UNSAFE_STATE"
        )
    try:
        base_identity = verify_base(base)
        initialization, tensors = verify_adapter_artifact(adapter, adapter_sha256, base_identity)
    except ServingError:
        raise
    except Exception:  # noqa: BLE001 - redact third-party artifact errors at the CLI boundary.
        raise ServingError("SERVE_ARTIFACT_VERIFICATION_FAILED") from None
    config = ServeConfig(base, adapter, adapter_sha256, state, model_name, port)
    return config, {
        "base": base_identity,
        "adapter": initialization,
        "tensors": tensors,
        "model_name": model_name,
        "host": "127.0.0.1",
        "port": port,
        "memory_limit_bytes": MEMORY_BYTES,
        "unsupported_internal_hook": True,
        "installed_files_modified": False,
        "runtime_training": False,
        "base_requantized": False,
    }


def verify_backend_versions() -> dict:
    try:
        actual = {name: importlib.metadata.version(name) for name in TESTED_BACKENDS}
    except importlib.metadata.PackageNotFoundError:
        raise ServingError("SERVE_BACKEND_UNAVAILABLE") from None
    _require(actual == TESTED_BACKENDS, "SERVE_UNTESTED_BACKEND")
    return actual


def hash_package_tree(directory: Path) -> dict:
    """Canonical identity of installed Python, native and resource files."""
    _require(directory.is_dir() and not directory.is_symlink(), "SERVE_UNSAFE_PACKAGE")
    files, directories, total = [], {}, 0
    for current, children, names in os.walk(directory, followlinks=False):
        current = Path(current)
        info = current.lstat()
        _require(stat.S_ISDIR(info.st_mode), "SERVE_UNSAFE_PACKAGE")
        directories[current] = _file_identity(info)
        _require(len(directories) <= MAX_PACKAGE_FILES, "SERVE_PACKAGE_LIMIT")
        for child in children:
            _require(not (current / child).is_symlink(), "SERVE_UNSAFE_PACKAGE")
        children[:] = [name for name in children if name != "__pycache__"]
        for name in names:
            path = current / name
            _require(not path.is_symlink(), "SERVE_UNSAFE_PACKAGE")
            if name == ".DS_Store" or path.suffix in {".pyc", ".pyo"}:
                continue
            files.append(path)
            _require(len(files) <= MAX_PACKAGE_FILES, "SERVE_PACKAGE_LIMIT")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda value: value.relative_to(directory).as_posix()):
        sha, size = _regular_hash(path, min(MAX_PACKAGE_FILE_BYTES, MAX_PACKAGE_TOTAL_BYTES - total))
        total += size
        _require(total <= MAX_PACKAGE_TOTAL_BYTES, "SERVE_PACKAGE_LIMIT")
        digest.update(
            path.relative_to(directory).as_posix().encode("utf-8") + b"\0" + sha.encode("ascii") + b"\n"
        )
    _require(files, "SERVE_UNSAFE_PACKAGE")
    _require(
        all(_file_identity(path.lstat()) == identity for path, identity in directories.items()),
        "SERVE_INPUT_CHANGED",
    )
    return {"file_count": len(files), "total_bytes": total, "sha256": digest.hexdigest()}


def _verify_import_spec(name: str, spec, root: Path, *, top_level=False):
    _require(spec is not None and spec.name == name, "SERVE_BACKEND_IMPORT_SHADOWED")
    locations = spec.submodule_search_locations
    expected = root.joinpath(*name.split(".")[1:])
    if locations is not None:
        _require(
            [Path(path).resolve(strict=True) for path in locations] == [expected],
            "SERVE_BACKEND_IMPORT_SHADOWED",
        )
    if spec.origin is None:
        _require(name == "mlx" and locations is not None, "SERVE_BACKEND_IMPORT_SHADOWED")
    else:
        origin = Path(spec.origin).resolve(strict=True)
        _require(origin.is_file() and origin.is_relative_to(root), "SERVE_BACKEND_IMPORT_SHADOWED")
        if top_level:
            _require(origin == root / "__init__.py", "SERVE_BACKEND_IMPORT_SHADOWED")
    if top_level:
        _require(locations is not None, "SERVE_BACKEND_IMPORT_SHADOWED")


def verify_import_origins(roots: dict[str, Path]):
    """Bind metadata trees to imports without importing any backend package."""
    try:
        for name, module in tuple(sys.modules.items()):
            root = roots.get(name.split(".")[0])
            if root is None:
                continue
            spec = getattr(module, "__spec__", None)
            _verify_import_spec(name, spec, root, top_level=name in roots)
            if spec.origin is not None:
                _require(
                    Path(getattr(module, "__file__", "")).resolve(strict=True)
                    == Path(spec.origin).resolve(strict=True),
                    "SERVE_BACKEND_IMPORT_SHADOWED",
                )
            if spec.submodule_search_locations is not None:
                _require(
                    [Path(path).resolve(strict=True) for path in getattr(module, "__path__", ())]
                    == [Path(path).resolve(strict=True) for path in spec.submodule_search_locations],
                    "SERVE_BACKEND_IMPORT_SHADOWED",
                )
        for name, root in roots.items():
            _verify_import_spec(name, importlib.util.find_spec(name), root, top_level=True)
    except ServingError:
        raise
    except (ImportError, OSError, ValueError, TypeError, AttributeError):
        raise ServingError("SERVE_BACKEND_IMPORT_SHADOWED") from None


def verify_backend_packages() -> dict:
    """Prove the measured installed trees without importing a backend."""
    try:
        distributions = {name: importlib.metadata.distribution(name) for name in TESTED_BACKENDS}
        roots = {
            name: Path(distributions[name].locate_file(package))
            for name, package in PACKAGE_DIRECTORIES.items()
        }
        actual = {name: hash_package_tree(root) for name, root in roots.items()}
        _require(actual == TESTED_PACKAGE_IDENTITIES, "SERVE_BACKEND_SOURCE_CHANGED")
        distribution = distributions["mlx-lm"]
        paths = [
            path for path in distribution.files or () if str(path).endswith(".dist-info/direct_url.json")
        ]
        _require(len(paths) == 1, "SERVE_MLX_LM_SOURCE_CHANGED")
        direct = json.loads(_read_regular(Path(distribution.locate_file(paths[0])), 65536))
        _require(
            direct.get("url") == MLX_LM_SERVING_URL
            and direct.get("vcs_info", {}).get("vcs") == "git"
            and direct.get("vcs_info", {}).get("commit_id") == MLX_LM_SERVING_COMMIT,
            "SERVE_MLX_LM_SOURCE_CHANGED",
        )
        verify_import_origins(
            {PACKAGE_DIRECTORIES[name]: root.resolve(strict=True) for name, root in roots.items()}
        )
        return {
            "format": "package-tree-sha256-v1",
            "packages": actual,
            "mlx_lm_source": {"url": MLX_LM_SERVING_URL, "commit_id": MLX_LM_SERVING_COMMIT},
            "import_origins_verified": True,
        }
    except ServingError:
        raise
    except (OSError, ValueError, TypeError, AttributeError, importlib.metadata.PackageNotFoundError):
        raise ServingError("SERVE_BACKEND_SOURCE_UNVERIFIED") from None


@contextlib.contextmanager
def credentialless_environment():
    original = dict(os.environ)
    safe = offline_environment()
    os.environ.clear()
    os.environ.update(safe)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(original)


def _write_json(path: Path, value: dict):
    _require(not path.is_symlink(), "SERVE_UNSAFE_STATE")
    descriptor, temporary = tempfile.mkstemp(prefix=".serve-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def private_state(config: ServeConfig, identity: dict, versions: dict):
    """Hold a lifetime lock; reject unrelated state and never signal a PID."""
    _require(not config.state.is_symlink(), "SERVE_UNSAFE_STATE")
    config.state.mkdir(parents=True, mode=0o700, exist_ok=True)
    info = config.state.stat()
    _require(
        info.st_uid == os.getuid() and stat.S_IMODE(info.st_mode) & 0o077 == 0, "SERVE_STATE_PERMISSIONS"
    )
    descriptor = os.open(
        config.state / ".serve.lock", os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600
    )
    lock = os.fdopen(descriptor, "a+")
    try:
        lock_info = os.fstat(lock.fileno())
        _require(
            stat.S_ISREG(lock_info.st_mode)
            and lock_info.st_uid == os.getuid()
            and stat.S_IMODE(lock_info.st_mode) & 0o077 == 0,
            "SERVE_UNSAFE_STATE",
        )
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ServingError("SERVE_BUSY") from None
        allowed = {
            ".serve.lock",
            "manifest.json",
            "integrity.json",
            "models",
            "server-state",
            "empty-mcp.json",
        }
        _require(all(path.name in allowed for path in config.state.iterdir()), "SERVE_UNRELATED_STATE")
        profile = _digest(
            {
                "base": str(config.base),
                "adapter": str(config.adapter),
                "identity": identity,
                "backends": versions,
            }
        )
        manifest = {
            "format": "experimental-omlx-adapter-v1",
            "profile_sha256": profile,
            **identity,
            "backend_versions": versions,
        }
        manifest_path = config.state / "manifest.json"
        if manifest_path.exists():
            _require(
                not manifest_path.is_symlink() and json.loads(manifest_path.read_text()) == manifest,
                "SERVE_PROFILE_CHANGED",
            )
        else:
            _write_json(manifest_path, manifest)
        models = config.state / "models"
        _require(not models.is_symlink(), "SERVE_UNSAFE_STATE")
        models.mkdir(mode=0o700, exist_ok=True)
        _require(all(path.name == config.model_name for path in models.iterdir()), "SERVE_UNRELATED_STATE")
        model = models / config.model_name
        if model.is_symlink():
            _require(model.resolve() == config.base, "SERVE_MODEL_LINK_CHANGED")
        elif model.exists():
            raise ServingError("SERVE_MODEL_LINK_CHANGED")
        else:
            model.symlink_to(config.base, target_is_directory=True)
        server = config.state / "server-state"
        _require(not server.is_symlink(), "SERVE_UNSAFE_STATE")
        server.mkdir(mode=0o700, exist_ok=True)
        mcp = config.state / "empty-mcp.json"
        if mcp.exists():
            _require(
                not mcp.is_symlink() and json.loads(mcp.read_text()) == {"mcpServers": {}},
                "SERVE_MCP_CONFIG_CHANGED",
            )
        else:
            _write_json(mcp, {"mcpServers": {}})
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", config.port))
            except OSError:
                raise ServingError("SERVE_PORT_IN_USE") from None
        yield manifest
    finally:
        lock.close()


def _backend_runtime():
    from omlx._torch_stub import install

    install()
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.models import qwen3
    from mlx_lm.tuner.lora import LoRALinear
    from mlx_lm.tuner.utils import load_adapters
    from omlx.engine.batched import BatchedEngine
    from omlx.engine_core import get_mlx_executor
    from omlx.utils import model_loading

    from .infer import verify_loaded_adapter
    from .weight_space import _mlx_tensor_hash

    return SimpleNamespace(
        mx=mx,
        qwen3=qwen3,
        flatten=tree_flatten,
        lora=LoRALinear,
        load_adapters=load_adapters,
        engine=BatchedEngine,
        loading=model_loading,
        executor=get_mlx_executor,
        verify_adapter=verify_loaded_adapter,
        tensor_hash=_mlx_tensor_hash,
    )


@contextlib.contextmanager
def adapter_hook(config: ServeConfig, identity: dict, versions: dict, *, runtime=None):
    """Patch only this foreground process, restoring owned hooks on exit."""
    rt = runtime or _backend_runtime()
    rt.mx.set_memory_limit(MEMORY_BYTES)
    rt.mx.set_cache_limit(512 * 1024**2)
    original_load = rt.loading.lm_load_compat
    original_start, original_chat, original_call = rt.engine.start, rt.engine.chat, rt.lora.__call__
    classes = [
        rt.qwen3.Model,
        rt.qwen3.Qwen3Model,
        rt.qwen3.TransformerBlock,
        rt.qwen3.Attention,
        rt.qwen3.MLP,
    ]
    functions = {cls: cls.__call__ for cls in classes}
    state = {
        "model": None,
        "base_hashes": {},
        "paths": [],
        "ids": {},
        "counts": {},
        "post_start_verified": False,
    }

    def record():
        _write_json(
            config.state / "integrity.json",
            {
                "format": "experimental-omlx-adapter-integrity-v1",
                "adapter_sha256": config.adapter_sha256,
                "backend_versions": versions,
                "serving_packages": identity.get("serving_packages"),
                "post_start_verified": state["post_start_verified"],
                "base_tensor_count": len(state["base_hashes"]),
                "base_tensor_hashes_sha256": _digest(state["base_hashes"]),
                "base_tensors_unchanged": state.get("base_unchanged", False),
                "wrapped_modules": state["paths"],
                "wrapper_trace_invocations": state["counts"],
                "adapter_verification": state.get("adapter_verification"),
                "call_graph_unchanged": all(cls.__call__ is function for cls, function in functions.items()),
                "unsupported_internal_hook": True,
                "installed_files_modified": False,
                "runtime_training": False,
                "base_requantized": False,
                "existing_services_changed": False,
                "limitations": "Invocation counts include tracing; a successful startup is not an extraction-accuracy result.",
            },
        )

    def verify(model):
        _require(model is state["model"], "SERVE_MODEL_HOOK_BYPASSED")
        arrays = dict(rt.flatten(model.parameters()))
        for key, digest in state["base_hashes"].items():
            actual = key
            for prefix in state["paths"]:
                if key.startswith(prefix + "."):
                    actual = prefix + ".linear" + key[len(prefix) :]
                    break
            _require(
                actual in arrays and rt.tensor_hash(arrays[actual]) == digest, "SERVE_BASE_TENSOR_CHANGED"
            )
        modules = dict(model.named_modules())
        _require(
            all(path in modules and state["ids"].get(id(modules[path])) == path for path in state["paths"]),
            "SERVE_ADAPTER_REPLACED",
        )
        _require(
            all(cls.__call__ is function for cls, function in functions.items()), "SERVE_FORWARD_PATH_CHANGED"
        )
        _require(rt.lora.__call__ is counted_call, "SERVE_FORWARD_PATH_CHANGED")
        _require(not rt.flatten(model.trainable_parameters()), "SERVE_TRAINABLE_PARAMETERS")
        result = rt.verify_adapter(model, config.adapter)
        _require(
            result.get("exact_tensor_reload") is True and result.get("tensor_count") == 32,
            "SERVE_ADAPTER_CHANGED",
        )
        state["adapter_verification"], state["base_unchanged"] = result, True

    def verify_structure(model):
        _require(model is state["model"], "SERVE_MODEL_HOOK_BYPASSED")
        modules = dict(model.named_modules())
        _require(
            all(path in modules and state["ids"].get(id(modules[path])) == path for path in state["paths"]),
            "SERVE_ADAPTER_REPLACED",
        )
        _require(
            all(cls.__call__ is function for cls, function in functions.items()), "SERVE_FORWARD_PATH_CHANGED"
        )
        _require(rt.lora.__call__ is counted_call, "SERVE_FORWARD_PATH_CHANGED")

    @functools.wraps(original_call)
    def counted_call(module, *args, **kwargs):
        path = state["ids"].get(id(module))
        if path is not None:
            state["counts"][path] += 1
        return original_call(module, *args, **kwargs)

    @functools.wraps(original_load)
    def load_target(path, **kwargs):
        _require(
            Path(path).resolve() == config.base and state["model"] is None, "SERVE_UNEXPECTED_MODEL_LOAD"
        )
        _require(verify_base(config.base) == identity["base"], "SERVE_BASE_FILES_CHANGED")
        _require(
            verify_adapter_artifact(config.adapter, config.adapter_sha256, identity["base"])[0]
            == identity["adapter"],
            "SERVE_ADAPTER_CHANGED",
        )
        kwargs["trust_remote_code"] = False
        model, tokenizer = original_load(path, **kwargs)
        state["base_hashes"] = {key: rt.tensor_hash(value) for key, value in rt.flatten(model.parameters())}
        _require(len(state["base_hashes"]) == 704, "SERVE_INCOMPATIBLE_BASE_TENSORS")
        model = rt.load_adapters(model, str(config.adapter))
        state["model"] = model
        for module_path, module in model.named_modules():
            if isinstance(module, rt.lora):
                state["paths"].append(module_path)
                state["ids"][id(module)], state["counts"][module_path] = module_path, 0
        expected = {name.rsplit(".", 1)[0] for name in expected_tensor_shapes()}
        _require(
            set(state["paths"]) == expected and len(state["paths"]) == 16,
            "SERVE_INCOMPATIBLE_ADAPTER_MODULES",
        )
        model.freeze()
        model.eval()
        rt.mx.eval(model.parameters())
        verify(model)
        _require(
            verify_adapter_artifact(config.adapter, config.adapter_sha256, identity["base"])[0]
            == identity["adapter"],
            "SERVE_ADAPTER_CHANGED",
        )
        record()
        return model, tokenizer

    @functools.wraps(original_start)
    async def checked_start(engine, *args, **kwargs):
        await original_start(engine, *args, **kwargs)
        await asyncio.get_running_loop().run_in_executor(rt.executor(), verify, engine._model)
        state["post_start_verified"] = True
        record()

    @functools.wraps(original_chat)
    async def checked_chat(engine, *args, **kwargs):
        verify_structure(engine._model)
        result = await original_chat(engine, *args, **kwargs)
        _require(state["post_start_verified"] and all(state["counts"].values()), "SERVE_ADAPTER_NOT_INVOKED")
        verify_structure(engine._model)
        record()
        return result

    rt.lora.__call__ = counted_call
    rt.loading.lm_load_compat = load_target
    rt.engine.start, rt.engine.chat = checked_start, checked_chat
    try:
        yield state
    finally:
        for target, name, installed, original in (
            (rt.loading, "lm_load_compat", load_target, original_load),
            (rt.engine, "start", checked_start, original_start),
            (rt.engine, "chat", checked_chat, original_chat),
            (rt.lora, "__call__", counted_call, original_call),
        ):
            if getattr(target, name) is installed:
                setattr(target, name, original)


def server_arguments(config):
    return [
        "omlx",
        "serve",
        "--model-dir",
        str(config.state / "models"),
        "--base-path",
        str(config.state / "server-state"),
        "--host",
        "127.0.0.1",
        "--port",
        str(config.port),
        "--max-concurrent-requests",
        "1",
        "--memory-guard",
        "safe",
        "--no-cache",
        "--no-hf-cache",
        "--headless",
        "--log-level",
        "warning",
        "--mcp-config",
        str(config.state / "empty-mcp.json"),
    ]


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--experimental-omlx-adapter", action="store_true", required=True)
    result.add_argument("--base", type=Path, required=True)
    result.add_argument("--adapter-dir", type=Path, required=True)
    result.add_argument("--adapter-sha256", required=True)
    result.add_argument("--state-dir", type=Path, required=True)
    result.add_argument("--model-name", default="evidence-scraper-local")
    result.add_argument("--port", type=int, default=8001)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        config, identity = validate_inputs(
            base=args.base,
            adapter_dir=args.adapter_dir,
            adapter_sha256=args.adapter_sha256,
            state_dir=args.state_dir,
            model_name=args.model_name,
            port=args.port,
            enabled=args.experimental_omlx_adapter,
        )
        versions = verify_backend_versions()
        identity = {**identity, "serving_packages": verify_backend_packages()}
        with (
            private_state(config, identity, versions),
            credentialless_environment(),
            adapter_hook(config, identity, versions),
        ):
            from omlx.cli import main as omlx_main

            original_argv = sys.argv
            try:
                sys.argv = server_arguments(config)
                omlx_main()
            finally:
                sys.argv = original_argv
        return 0
    except KeyboardInterrupt:
        return 130
    except ServingError as error:
        print(f"{error}: Local adapter serving could not start or continue", file=sys.stderr)
    except Exception:  # noqa: BLE001 - never emit source paths or third-party credentials.
        print("SERVE_FAILED: Local adapter serving could not start or continue", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
