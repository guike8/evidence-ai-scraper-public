"""CPU controls for the explicitly experimental serving boundary."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.machinery
import json
import os
import struct
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import ClassVar

import pytest

from training import serve_omlx as serve
from training.train import BASE_ID, BASE_REVISION, BASE_WEIGHT_SHA256, LORA_PARAMETERS


def write_weights(path, *, mutate=None, nonfinite=False):
    header, offset = {}, 0
    for key, shape in serve.expected_tensor_shapes().items():
        length = shape[0] * shape[1] * 4
        header[key] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, offset + length]}
        offset += length
    if mutate:
        mutate(header)
    raw = bytearray(offset)
    if nonfinite:
        raw[:4] = struct.pack("<f", float("nan"))
    encoded = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def artifact(tmp_path, monkeypatch):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    base.mkdir()
    adapter.mkdir()
    sha = write_weights(adapter / "adapters.safetensors")
    (adapter / "adapter_config.json").write_text(
        json.dumps({"fine_tune_type": "lora", "num_layers": 8, "lora_parameters": LORA_PARAMETERS})
    )
    base_identity = {
        "repo_id": BASE_ID,
        "revision": BASE_REVISION,
        "files": {"model.safetensors": BASE_WEIGHT_SHA256},
    }
    (adapter / "training.json").write_text(json.dumps({"base": base_identity, "adapter_sha256": sha}))
    monkeypatch.setattr(serve, "verify_base", lambda _: base_identity)
    options = {
        "base": base,
        "adapter_dir": adapter,
        "adapter_sha256": sha,
        "state_dir": tmp_path / "state",
        "model_name": "test-model",
        "port": 8019,
        "enabled": True,
    }
    return options, base_identity


def test_artifact_preflight_verifies_all_known_cpu_tensors_and_portable_metadata(artifact):
    options, _ = artifact
    config, identity = serve.validate_inputs(**options)
    assert identity["tensors"] == {"tensor_count": 32, "parameters": 458752, "dtype": "F32", "finite": True}
    assert str(config.base) not in json.dumps(identity)
    assert str(config.adapter) not in json.dumps(identity)


@pytest.mark.parametrize(
    "change,code",
    [
        ({"enabled": False}, "SERVE_EXPLICIT_OPT_IN_REQUIRED"),
        ({"adapter_sha256": "x" * 64}, "SERVE_INVALID_SHA256"),
        ({"model_name": "../other"}, "SERVE_INVALID_MODEL_NAME"),
        ({"port": True}, "SERVE_INVALID_PORT"),
        ({"port": 80}, "SERVE_INVALID_PORT"),
    ],
)
def test_invalid_options_fail_before_artifact_or_backend_load(artifact, change, code):
    options, _ = artifact
    with pytest.raises(serve.ServingError, match=code):
        serve.validate_inputs(**{**options, **change})


def test_rejects_state_in_model_adapter_or_source_or_an_ancestor(artifact, monkeypatch, tmp_path):
    options, _ = artifact
    source = tmp_path / "source"
    monkeypatch.setattr(serve, "REPOSITORY", source)
    for state in (options["base"] / "state", options["adapter_dir"] / "state", source / "state", tmp_path):
        with pytest.raises(serve.ServingError, match="SERVE_UNSAFE_STATE"):
            serve.validate_inputs(**{**options, "state_dir": state})


def test_rejects_symlink_state_and_modified_tokenizer_receipt(artifact, tmp_path):
    options, base = artifact
    link = tmp_path / "link"
    target = tmp_path / "target"
    target.mkdir()
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(serve.ServingError, match="SERVE_UNSAFE_STATE"):
        serve.validate_inputs(**{**options, "state_dir": link})
    base["files"]["tokenizer.json"] = "1" * 64
    with pytest.raises(serve.ServingError, match="SERVE_TOKENIZER_OR_CONFIG_CHANGED"):
        serve.validate_inputs(**options)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda h: h.pop(next(iter(h))),
        lambda h: h[next(iter(h))].update(dtype="F16"),
        lambda h: h[next(iter(h))].update(shape=[1, 1]),
        lambda h: h[next(iter(h))].update(data_offsets=[0, 4]),
    ],
)
def test_corrupt_or_incompatible_tensor_headers_are_rejected(artifact, mutator):
    options, _ = artifact
    path = options["adapter_dir"] / "adapters.safetensors"
    digest = write_weights(path, mutate=mutator)
    with pytest.raises(serve.ServingError):
        serve.verify_tensor_file(path, digest)


def test_nonfinite_and_changed_weight_bytes_fail_before_mlx(artifact):
    options, _ = artifact
    path = options["adapter_dir"] / "adapters.safetensors"
    digest = write_weights(path, nonfinite=True)
    with pytest.raises(serve.ServingError, match="SERVE_NONFINITE_ADAPTER"):
        serve.verify_tensor_file(path, digest)
    with pytest.raises(serve.ServingError, match="SERVE_ADAPTER_CHANGED"):
        serve.verify_tensor_file(path, options["adapter_sha256"])


@pytest.mark.parametrize("kind", ["tensor", "receipt"])
@pytest.mark.parametrize("mutation", ["replace", "grow"])
def test_bounded_fd_reads_reject_replacement_or_growth(artifact, monkeypatch, kind, mutation):
    options, _ = artifact
    name = "adapters.safetensors" if kind == "tensor" else "training.json"
    path = options["adapter_dir"] / name
    original_read, changed = os.read, False

    def racing_read(descriptor, size):
        nonlocal changed
        value = original_read(descriptor, size)
        if not changed and os.fstat(descriptor).st_ino == path.stat().st_ino:
            changed = True
            if mutation == "replace":
                replacement = path.with_suffix(".replacement")
                replacement.write_bytes(path.read_bytes())
                replacement.replace(path)
            else:
                with path.open("ab") as stream:
                    stream.write(b"x")
        return value

    monkeypatch.setattr(serve.os, "read", racing_read)
    with pytest.raises(serve.ServingError, match="SERVE_INPUT_CHANGED"):
        serve.validate_inputs(**options)
    assert changed


@pytest.mark.parametrize("kind", ["tensor", "receipt"])
def test_bounded_fd_reads_reject_symlinks(artifact, tmp_path, kind):
    options, _ = artifact
    name = "adapters.safetensors" if kind == "tensor" else "training.json"
    path = options["adapter_dir"] / name
    saved = tmp_path / "saved-input"
    path.replace(saved)
    path.symlink_to(saved)
    with pytest.raises(serve.ServingError, match="SERVE_ARTIFACT_VERIFICATION_FAILED"):
        serve.validate_inputs(**options)


def test_bounded_fd_reads_refuse_oversize_and_fifo_without_reading(tmp_path, monkeypatch):
    path = tmp_path / "input"
    path.write_bytes(b"too large")
    monkeypatch.setattr(serve.os, "read", lambda *_: pytest.fail("Must reject before reading"))
    with pytest.raises(serve.ServingError, match="SERVE_UNSAFE_INPUT"):
        serve._read_regular(path, 2)
    path.unlink()
    os.mkfifo(path)
    with pytest.raises(serve.ServingError, match="SERVE_UNSAFE_INPUT"):
        serve._read_regular(path, 20)


def test_artifact_validation_parses_one_captured_receipt(artifact, monkeypatch):
    options, _ = artifact
    original_read, opened = serve._read_regular, []

    def record_read(path, limit):
        opened.append(path.name)
        return original_read(path, limit)

    monkeypatch.setattr(serve, "_read_regular", record_read)
    serve.validate_inputs(**options)
    assert opened == ["adapters.safetensors", "adapter_config.json", "training.json"]


def test_backend_versions_are_exact_and_missing_backends_fail(monkeypatch):
    monkeypatch.setattr(serve.importlib.metadata, "version", serve.TESTED_BACKENDS.__getitem__)
    assert serve.verify_backend_versions() == serve.TESTED_BACKENDS
    monkeypatch.setattr(serve.importlib.metadata, "version", lambda _: "new-version")
    with pytest.raises(serve.ServingError, match="SERVE_UNTESTED_BACKEND"):
        serve.verify_backend_versions()

    def missing(_):
        raise serve.importlib.metadata.PackageNotFoundError("test")

    monkeypatch.setattr(serve.importlib.metadata, "version", missing)
    with pytest.raises(serve.ServingError, match="SERVE_BACKEND_UNAVAILABLE"):
        serve.verify_backend_versions()


@pytest.fixture
def package_install(tmp_path, monkeypatch):
    metadata = tmp_path / "mlx_lm-verified.dist-info"
    metadata.mkdir()
    direct_path = metadata / "direct_url.json"
    direct_path.write_text(
        json.dumps(
            {
                "url": serve.MLX_LM_SERVING_URL,
                "vcs_info": {"vcs": "git", "commit_id": serve.MLX_LM_SERVING_COMMIT},
            }
        )
    )
    for name in serve.PACKAGE_DIRECTORIES.values():
        package = tmp_path / name
        package.mkdir()
        (package / "__init__.py").write_bytes(b"# measured source\n")
        (package / "native.bin").write_bytes(b"\0\1native\xff")
        (package / "empty-resource").touch()
    identities = {
        name: serve.hash_package_tree(tmp_path / directory)
        for name, directory in serve.PACKAGE_DIRECTORIES.items()
    }

    class Distribution:
        files = (Path("mlx_lm-verified.dist-info/direct_url.json"),)

        def locate_file(self, path):
            return tmp_path / path

    monkeypatch.setattr(serve.importlib.metadata, "distribution", lambda _: Distribution())
    monkeypatch.setattr(serve.importlib.metadata, "version", serve.TESTED_BACKENDS.__getitem__)
    monkeypatch.setattr(serve, "TESTED_PACKAGE_IDENTITIES", identities)
    monkeypatch.syspath_prepend(str(tmp_path))
    return tmp_path, direct_path


def test_backend_tree_identity_detects_same_version_source_or_native_change(package_install):
    root, _ = package_install
    assert serve.verify_backend_versions() == serve.TESTED_BACKENDS
    assert serve.verify_backend_packages()["packages"] == serve.TESTED_PACKAGE_IDENTITIES
    (root / "omlx/native.bin").write_bytes(b"modified native code")
    assert serve.verify_backend_versions() == serve.TESTED_BACKENDS
    with pytest.raises(serve.ServingError, match="SERVE_BACKEND_SOURCE_CHANGED"):
        serve.verify_backend_packages()


def test_backend_requires_exact_mlx_lm_vcs_identity(package_install):
    _, direct = package_install
    direct.write_text(
        json.dumps(
            {
                "url": "https://example.org/other",
                "vcs_info": {"vcs": "git", "commit_id": serve.MLX_LM_SERVING_COMMIT},
            }
        )
    )
    with pytest.raises(serve.ServingError, match="SERVE_MLX_LM_SOURCE_CHANGED"):
        serve.verify_backend_packages()


def test_backend_rejects_real_sys_path_shadow_without_importing_it(package_install, tmp_path, monkeypatch):
    root, _ = package_install
    shadow = tmp_path / "earlier-path"
    (shadow / "omlx").mkdir(parents=True)
    (shadow / "omlx/__init__.py").write_text("raise RuntimeError('must never import shadow')\n")
    monkeypatch.syspath_prepend(str(shadow))
    assert serve.importlib.util.find_spec("omlx").origin == str(shadow / "omlx/__init__.py")
    with pytest.raises(serve.ServingError, match="SERVE_BACKEND_IMPORT_SHADOWED"):
        serve.verify_backend_packages()
    assert str(root) not in json.dumps({"packages": serve.TESTED_PACKAGE_IDENTITIES})
    assert "omlx" not in sys.modules


def test_backend_accepts_only_verified_single_root_mlx_namespace(package_install, monkeypatch):
    root, _ = package_install
    (root / "mlx/__init__.py").unlink()
    identities = {**serve.TESTED_PACKAGE_IDENTITIES, "mlx": serve.hash_package_tree(root / "mlx")}
    monkeypatch.setattr(serve, "TESTED_PACKAGE_IDENTITIES", identities)
    spec = serve.importlib.util.find_spec("mlx")
    assert spec.origin is None and list(spec.submodule_search_locations) == [str(root / "mlx")]
    receipt = serve.verify_backend_packages()
    assert receipt["import_origins_verified"] and str(root) not in json.dumps(receipt)
    assert "mlx" not in sys.modules


def test_backend_rejects_preloaded_module_from_unverified_origin(package_install, tmp_path, monkeypatch):
    root, _ = package_install
    unrelated = tmp_path / "unverified.py"
    unrelated.write_text("# unverified\n")
    module = ModuleType("mlx.core")
    module.__spec__ = importlib.machinery.ModuleSpec("mlx.core", loader=None, origin=str(unrelated))
    module.__file__ = str(unrelated)
    monkeypatch.setitem(sys.modules, "mlx.core", module)
    with pytest.raises(serve.ServingError, match="SERVE_BACKEND_IMPORT_SHADOWED"):
        serve.verify_backend_packages()
    assert (root / "mlx").is_dir()


def test_backend_rejects_namespace_with_an_extra_search_root(package_install, tmp_path, monkeypatch):
    root, _ = package_install
    (root / "mlx/__init__.py").unlink()
    other = tmp_path / "other-root"
    (other / "mlx").mkdir(parents=True)
    identities = {**serve.TESTED_PACKAGE_IDENTITIES, "mlx": serve.hash_package_tree(root / "mlx")}
    monkeypatch.setattr(serve, "TESTED_PACKAGE_IDENTITIES", identities)
    monkeypatch.syspath_prepend(str(other))
    with pytest.raises(serve.ServingError, match="SERVE_BACKEND_IMPORT_SHADOWED"):
        serve.verify_backend_packages()


def test_canonical_package_hash_includes_resources_excludes_only_cache_files(tmp_path):
    (tmp_path / "z.native").write_bytes(b"native")
    (tmp_path / "a.py").write_bytes(b"source")
    cache = tmp_path / "__pycache__"
    cache.mkdir()
    (cache / "ignored.pyc").write_bytes(b"ignored")
    for name in (".DS_Store", "ignored.pyc", "ignored.pyo"):
        (tmp_path / name).write_bytes(b"ignored")
    expected = hashlib.sha256()
    for name, body in (("a.py", b"source"), ("z.native", b"native")):
        expected.update(name.encode() + b"\0" + hashlib.sha256(body).hexdigest().encode() + b"\n")
    assert serve.hash_package_tree(tmp_path) == {
        "file_count": 2,
        "total_bytes": 12,
        "sha256": expected.hexdigest(),
    }


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_package_tree_refuses_symlink_files_and_directories(tmp_path, kind):
    package, external = tmp_path / "package", tmp_path / "outside"
    package.mkdir()
    external.mkdir()
    (external / "real.py").write_bytes(b"source")
    target = external / "real.py" if kind == "file" else external
    (package / "linked").symlink_to(target, target_is_directory=kind == "directory")
    with pytest.raises(serve.ServingError, match="SERVE_UNSAFE_PACKAGE"):
        serve.hash_package_tree(package)


class Socket:
    busy = False
    seen: ClassVar[list] = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def bind(self, address):
        self.seen.append(address)
        if self.busy:
            raise OSError("busy")


@pytest.fixture
def profile(artifact, monkeypatch):
    options, _ = artifact
    config, identity = serve.validate_inputs(**options)
    Socket.busy, Socket.seen = False, []
    monkeypatch.setattr(serve, "socket", SimpleNamespace(socket=Socket))
    return config, identity


def test_foreground_state_lock_identity_and_loopback_port_check(profile):
    config, identity = profile
    with serve.private_state(config, identity, serve.TESTED_BACKENDS):
        assert (config.state / "models/test-model").resolve() == config.base
        assert json.loads((config.state / "empty-mcp.json").read_text()) == {"mcpServers": {}}
        with (
            pytest.raises(serve.ServingError, match="SERVE_BUSY"),
            serve.private_state(config, identity, serve.TESTED_BACKENDS),
        ):
            pytest.fail("Second process must not own the state")
    with serve.private_state(config, identity, serve.TESTED_BACKENDS):
        pass  # A clean foreground exit releases only its own lock.
    assert Socket.seen == [("127.0.0.1", 8019), ("127.0.0.1", 8019)]


def test_state_rejects_unrelated_content_changed_model_and_mcp(profile):
    config, identity = profile
    with serve.private_state(config, identity, serve.TESTED_BACKENDS):
        pass
    (config.state / "unrelated.txt").write_text("preserve")
    with (
        pytest.raises(serve.ServingError, match="SERVE_UNRELATED_STATE"),
        serve.private_state(config, identity, serve.TESTED_BACKENDS),
    ):
        pass
    (config.state / "unrelated.txt").unlink()
    (config.state / "empty-mcp.json").write_text('{"mcpServers":{"other":{}}}')
    with (
        pytest.raises(serve.ServingError, match="SERVE_MCP_CONFIG_CHANGED"),
        serve.private_state(config, identity, serve.TESTED_BACKENDS),
    ):
        pass


def test_state_rejects_changed_profile_and_model_link(profile):
    config, identity = profile
    with serve.private_state(config, identity, serve.TESTED_BACKENDS):
        pass
    with (
        pytest.raises(serve.ServingError, match="SERVE_PROFILE_CHANGED"),
        serve.private_state(config, {**identity, "port": 8020}, serve.TESTED_BACKENDS),
    ):
        pass
    link = config.state / "models/test-model"
    link.unlink()
    link.symlink_to(config.adapter, target_is_directory=True)
    with (
        pytest.raises(serve.ServingError, match="SERVE_MODEL_LINK_CHANGED"),
        serve.private_state(config, identity, serve.TESTED_BACKENDS),
    ):
        pass


def test_state_rejects_shared_permissions_and_symlink_lock(profile, tmp_path):
    config, identity = profile
    config.state.mkdir(mode=0o755)
    config.state.chmod(0o755)
    with (
        pytest.raises(serve.ServingError, match="SERVE_STATE_PERMISSIONS"),
        serve.private_state(config, identity, serve.TESTED_BACKENDS),
    ):
        pass
    config.state.chmod(0o700)
    target = tmp_path / "unrelated-file"
    target.write_text("preserve")
    (config.state / ".serve.lock").symlink_to(target)
    with pytest.raises(OSError), serve.private_state(config, identity, serve.TESTED_BACKENDS):
        pass
    assert target.read_text() == "preserve"


def test_state_port_conflict_preserves_existing_service_and_rejects_public_bind(profile):
    config, identity = profile
    Socket.busy = True
    with (
        pytest.raises(serve.ServingError, match="SERVE_PORT_IN_USE"),
        serve.private_state(config, identity, serve.TESTED_BACKENDS),
    ):
        pass
    args = serve.server_arguments(config)
    assert args[args.index("--host") + 1] == "127.0.0.1"
    assert "--no-cache" in args and "--no-hf-cache" in args
    assert args[args.index("--max-concurrent-requests") + 1] == "1"


def test_environment_removes_credentials_without_changing_os_home(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-sensitive-value")
    original_home = os.environ.get("HOME")
    with serve.credentialless_environment():
        assert "HF_TOKEN" not in os.environ
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ.get("HOME") == original_home
    assert os.environ["HF_TOKEN"] == "test-sensitive-value"


def fake_runtime():
    class Tensor:
        def __init__(self, value):
            self.value = value

    class Lora:
        def __call__(self, *args, **kwargs):
            return "computed"

    paths = sorted({key.rsplit(".", 1)[0] for key in serve.expected_tensor_shapes()})

    class Model:
        def __init__(self):
            self.arrays = {f"base.{i}": Tensor(i) for i in range(688)}
            self.arrays.update({path + ".weight": Tensor(path) for path in paths})
            self.modules = {}
            self.frozen = False

        def parameters(self):
            return self.arrays

        def named_modules(self):
            return list(self.modules.items())

        def trainable_parameters(self):
            return {} if self.frozen else {"unexpected": Tensor(1)}

        def freeze(self):
            self.frozen = True

        def eval(self):
            pass

    model = Model()

    def original_load(path, **kwargs):
        assert kwargs["trust_remote_code"] is False
        return model, "tokenizer"

    def load_adapters(value, _):
        for path in paths:
            value.arrays[path + ".linear.weight"] = value.arrays.pop(path + ".weight")
            value.modules[path] = Lora()
        return value

    class Engine:
        def __init__(self):
            self._model = model

        async def start(self):
            pass

        async def chat(self):
            for module in model.modules.values():
                module()
            return "response"

    def call(self):
        return None

    classes = {
        name: type(name, (), {"__call__": call})
        for name in ("Model", "Qwen3Model", "TransformerBlock", "Attention", "MLP")
    }
    return SimpleNamespace(
        mx=SimpleNamespace(
            set_memory_limit=lambda _: None, set_cache_limit=lambda _: None, eval=lambda _: None
        ),
        qwen3=SimpleNamespace(**classes),
        flatten=lambda value: list(value.items()),
        lora=Lora,
        load_adapters=load_adapters,
        engine=Engine,
        loading=SimpleNamespace(lm_load_compat=original_load),
        executor=lambda: None,
        verify_adapter=lambda *_: {"exact_tensor_reload": True, "tensor_count": 32},
        tensor_hash=lambda value: str(value.value),
    ), model


async def test_hook_preserves_base_loads_exact_adapter_counts_all_wrappers_and_restores(profile):
    config, identity = profile
    rt, model = fake_runtime()
    originals = (rt.loading.lm_load_compat, rt.engine.start, rt.engine.chat, rt.lora.__call__)
    config.state.mkdir(mode=0o700)
    with serve.adapter_hook(config, identity, serve.TESTED_BACKENDS, runtime=rt) as state:
        assert rt.loading.lm_load_compat(config.base)[0] is model
        engine = rt.engine()
        await engine.start()
        assert await engine.chat() == "response"
        assert len(state["paths"]) == 16 and all(state["counts"].values())
        receipt = json.loads((config.state / "integrity.json").read_text())
        assert receipt["base_tensor_count"] == 704 and receipt["base_tensors_unchanged"]
        assert receipt["post_start_verified"] and receipt["adapter_verification"]["tensor_count"] == 32
    assert originals == (rt.loading.lm_load_compat, rt.engine.start, rt.engine.chat, rt.lora.__call__)


async def test_hook_refuses_wrong_model_and_base_mutation(profile):
    config, identity = profile
    config.state.mkdir(mode=0o700)
    rt, model = fake_runtime()
    with serve.adapter_hook(config, identity, serve.TESTED_BACKENDS, runtime=rt):
        with pytest.raises(serve.ServingError, match="SERVE_UNEXPECTED_MODEL_LOAD"):
            rt.loading.lm_load_compat(config.adapter)
        rt.loading.lm_load_compat(config.base)
        model.arrays["base.0"].value = "changed"
        with pytest.raises(serve.ServingError, match="SERVE_BASE_TENSOR_CHANGED"):
            await rt.engine().start()


@pytest.mark.parametrize("change", ["wrapper", "forward"])
async def test_hook_refuses_replaced_wrappers_or_qwen_forward(profile, change):
    config, identity = profile
    config.state.mkdir(mode=0o700)
    rt, model = fake_runtime()
    with serve.adapter_hook(config, identity, serve.TESTED_BACKENDS, runtime=rt) as state:
        rt.loading.lm_load_compat(config.base)
        engine = rt.engine()
        await engine.start()
        if change == "wrapper":
            model.modules[state["paths"][0]] = rt.lora()
            expected = "SERVE_ADAPTER_REPLACED"
        else:
            rt.qwen3.Model.__call__ = lambda self: None
            expected = "SERVE_FORWARD_PATH_CHANGED"
        with pytest.raises(serve.ServingError, match=expected):
            await engine.chat()


def test_main_errors_do_not_echo_input_paths_or_credentials(monkeypatch, capsys):
    def fail(**kwargs):
        raise RuntimeError("test-sensitive-value")

    monkeypatch.setattr(serve, "validate_inputs", fail)
    result = serve.main(
        [
            "--experimental-omlx-adapter",
            "--base",
            "base",
            "--adapter-dir",
            "adapter",
            "--adapter-sha256",
            "0" * 64,
            "--state-dir",
            "state",
        ]
    )
    assert result == 1 and "test-sensitive-value" not in capsys.readouterr().err


def test_main_rejects_modified_backend_before_state_or_backend_imports(monkeypatch, profile, capsys):
    config, identity = profile
    monkeypatch.setattr(serve, "validate_inputs", lambda **_: (config, identity))
    monkeypatch.setattr(serve, "verify_backend_versions", lambda: serve.TESTED_BACKENDS)

    def reject_packages():
        raise serve.ServingError("SERVE_BACKEND_SOURCE_CHANGED")

    monkeypatch.setattr(serve, "verify_backend_packages", reject_packages)
    monkeypatch.setattr(serve, "_backend_runtime", lambda: pytest.fail("Must not import a backend"))
    monkeypatch.setattr(serve, "private_state", lambda *_: pytest.fail("Must not create server state"))
    assert (
        serve.main(
            [
                "--experimental-omlx-adapter",
                "--base",
                "base",
                "--adapter-dir",
                "adapter",
                "--adapter-sha256",
                "0" * 64,
                "--state-dir",
                "state",
            ]
        )
        == 1
    )
    assert "SERVE_BACKEND_SOURCE_CHANGED" in capsys.readouterr().err
    assert not config.state.exists()


def test_main_uses_foreground_backend_offline_and_restores_argv(monkeypatch, profile):
    config, identity = profile
    monkeypatch.setattr(serve, "validate_inputs", lambda **_: (config, identity))
    monkeypatch.setattr(serve, "verify_backend_versions", lambda: serve.TESTED_BACKENDS)
    monkeypatch.setattr(serve, "verify_backend_packages", lambda: {"packages": "verified-test-packages"})
    monkeypatch.setattr(serve, "private_state", lambda *_: contextlib.nullcontext())
    monkeypatch.setattr(serve, "adapter_hook", lambda *_: contextlib.nullcontext())
    monkeypatch.setenv("HF_TOKEN", "test-sensitive-value")
    package = ModuleType("omlx")
    package.__path__ = []
    cli = ModuleType("omlx.cli")

    def main():
        assert "HF_TOKEN" not in os.environ
        assert sys.argv[sys.argv.index("--host") + 1] == "127.0.0.1"

    cli.main = main
    monkeypatch.setitem(sys.modules, "omlx", package)
    monkeypatch.setitem(sys.modules, "omlx.cli", cli)
    before = sys.argv
    assert (
        serve.main(
            [
                "--experimental-omlx-adapter",
                "--base",
                "base",
                "--adapter-dir",
                "adapter",
                "--adapter-sha256",
                "0" * 64,
                "--state-dir",
                "state",
            ]
        )
        == 0
    )
    assert sys.argv is before and os.environ["HF_TOKEN"] == "test-sensitive-value"
