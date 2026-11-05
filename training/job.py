"""Detached, bounded local MLX jobs; status requires only the standard library.

A lifetime advisory lock is inherited by the supervisor. PIDs are diagnostic
only: an old PID never authorizes restarting a job or terminating a process.
Interrupted runs require a new directory because MLX checkpoints are weights
only, rather than exact optimizer/RNG continuations.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

STAGES = ("preflight", "sanity", "train")
ACTIVE = {"starting", "running"}
TERMINAL = {"completed", "failed", "timed_out", "interrupted"}
REPOSITORY = Path(__file__).resolve().parents[1]


def offline_environment() -> dict[str, str]:
    """Pass OS essentials and offline flags, without parent credentials.

    This limits inherited environment variables; it is not a process sandbox.
    HOME and temporary/locale settings retain their existing OS meanings.
    """
    allowed = ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TMP", "TEMP")
    return {
        **{name: os.environ[name] for name in allowed if name in os.environ},
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "DO_NOT_TRACK": "1",
    }


def _write_json(path: Path, value: dict) -> None:
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    descriptor, temporary = tempfile.mkstemp(prefix=".job-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError("Invalid job state")
    return value


def _lock(run_dir: Path):
    descriptor = os.open(run_dir / "job.lock", os.O_RDWR | os.O_CREAT, 0o600)
    os.fchmod(descriptor, 0o600)
    stream = os.fdopen(descriptor, "a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        return None
    return stream


def _identity(config: dict) -> str:
    # Only private job-config.json contains paths. The portable state contains a
    # digest, status and measurements, never commands or environment variables.
    stable = {key: value for key, value in config.items() if key != "started_monotonic"}
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def make_config(
    *,
    python: str,
    model: str,
    data: str,
    run_dir: str,
    max_seconds: int = 3600,
    iters: int = 200,
    max_seq_length: int = 4096,
    memory_gib: int = 8,
) -> dict:
    for name, value, low, high in (
        ("max_seconds", max_seconds, 1, 21600),
        ("iters", iters, 4, 2000),
        ("max_seq_length", max_seq_length, 512, 8192),
        ("memory_gib", memory_gib, 4, 16),
    ):
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid bounded job option: {name}")
    if iters % 4:
        raise ValueError("Iterations must contain complete accumulation groups")
    interpreter = Path(python).expanduser().absolute()
    model_dir, data_dir, output = (Path(value).expanduser().resolve() for value in (model, data, run_dir))
    if not interpreter.is_file() or not os.access(interpreter, os.X_OK):
        raise ValueError("A local executable Python is required")
    if not model_dir.is_dir() or not data_dir.is_dir():
        raise ValueError("Model and dataset must be local directories")
    if (
        output == model_dir
        or output == data_dir
        or output in model_dir.parents
        or output in data_dir.parents
        or model_dir in output.parents
        or data_dir in output.parents
    ):
        raise ValueError("Run directory must be separate from model and data")
    return {
        "python": str(interpreter),
        "model": str(model_dir),
        "data": str(data_dir),
        "run_dir": str(output),
        "max_seconds": max_seconds,
        "iters": iters,
        "max_seq_length": max_seq_length,
        "memory_gib": memory_gib,
    }


def job_status(run_dir: str | Path) -> dict:
    directory = Path(run_dir)
    if not directory.is_dir():
        return {"status": "not_started", "running": False}
    if not (directory / "job.lock").exists() and not (directory / "job-state.json").exists():
        return {"status": "not_started", "running": False}
    lock = _lock(directory)
    try:
        state = _read_json(directory / "job-state.json")
        if lock is None:
            return {**(state or {"status": "starting"}), "running": True}
        if state is None:
            return {"status": "not_started", "running": False}
        if state.get("status") in ACTIVE:
            state.update(status="interrupted", exception_type="SupervisorLost", finished_at_unix=time.time())
            _write_json(directory / "job-state.json", state)
        return {**state, "running": False}
    finally:
        if lock is not None:
            lock.close()


def start_job(config: dict) -> dict:
    """Start once. Closing the parent's FD leaves the child's lock intact."""
    config = make_config(**config)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    run_dir.chmod(0o700)
    lock = _lock(run_dir)
    if lock is None:
        state = _read_json(run_dir / "job-state.json") or {"status": "starting"}
        if state.get("config_hash", _identity(config)) != _identity(config):
            raise ValueError("Run directory belongs to another configuration")
        return {**state, "running": True, "started": False}
    try:
        state = _read_json(run_dir / "job-state.json")
        if state is not None:
            if state.get("config_hash") != _identity(config):
                raise ValueError("Run directory belongs to another configuration")
            if state.get("status") in ACTIVE:
                state.update(
                    status="interrupted", exception_type="SupervisorLost", finished_at_unix=time.time()
                )
                _write_json(run_dir / "job-state.json", state)
            return {**state, "running": False, "started": False}
        # Existing receipts/checkpoints must never become an implicit resume.
        if any(
            (run_dir / name).exists() for name in ("adapter", "sanity-adapter", "sanity.json", "train.json")
        ):
            raise ValueError("Existing training artifacts require a new run directory")
        state = {
            "format": "local-training-job-v1",
            "status": "starting",
            "stage": None,
            "config_hash": _identity(config),
            "started_at_unix": time.time(),
            "max_seconds": config["max_seconds"],
            "stages_completed": [],
        }
        _write_json(run_dir / "job-config.json", {**config, "started_monotonic": time.monotonic()})
        _write_json(run_dir / "job-state.json", state)
        log_descriptor = os.open(run_dir / "job.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        os.fchmod(log_descriptor, 0o600)
        try:
            with os.fdopen(log_descriptor, "ab") as log:
                process = subprocess.Popen(
                    [
                        config["python"],
                        "-m",
                        "training.job",
                        "_run",
                        "--run-dir",
                        str(run_dir),
                        "--lock-fd",
                        str(lock.fileno()),
                    ],
                    cwd=REPOSITORY,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=offline_environment(),
                    start_new_session=True,
                    pass_fds=(lock.fileno(),),
                )
        except Exception as error:
            state.update(status="failed", exception_type=type(error).__name__, finished_at_unix=time.time())
            _write_json(run_dir / "job-state.json", state)
            raise
        return {**state, "supervisor_pid": process.pid, "running": True, "started": True}
    finally:
        # Do NOT LOCK_UN: the inherited open-file description owns this lock.
        lock.close()


def _stage_command(config: dict, stage: str) -> list[str]:
    return [
        config["python"],
        "-m",
        "training.train",
        "--model",
        config["model"],
        "--data",
        config["data"],
        "--run-dir",
        config["run_dir"],
        "--stage",
        stage,
        "--iters",
        str(config["iters"]),
        "--max-seq-length",
        str(config["max_seq_length"]),
        "--memory-gib",
        str(config["memory_gib"]),
    ]


def _require_group_absent(process: subprocess.Popen, error: Exception) -> None:
    """After signals cease, accept cleanup only with proof the group is gone."""
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        raise error from None
    deadline = time.monotonic() + 0.25
    while True:
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        except PermissionError:
            pass
        if time.monotonic() >= deadline:
            raise error
        time.sleep(0.01)


def _stop_owned_group(process: subprocess.Popen) -> None:
    """Stop an exclusively owned new session before reaping its leader.

    Keeping the leader unreaped reserves its PID throughout group signaling.
    A reaped leader cannot authorize signals to a possibly reused group ID.
    """
    if process.returncode is not None:
        _require_group_absent(
            process, RuntimeError("Cannot verify process-group ownership after its leader was reaped")
        )
        return
    if os.getpgid(process.pid) != process.pid:
        raise RuntimeError("Owned process must have its own session/process group")
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=2)
        return
    except PermissionError as error:
        _require_group_absent(process, error)
        return
    # Do not poll()/wait() here: both reap the leader and release its PID even
    # when a descendant ignores TERM. Retain the bounded two-second grace.
    time.sleep(2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError as error:
        # Darwin can deny a signal to a zombie-only group. Do not assume that
        # explanation: cease signals and require ESRCH after reaping our child.
        _require_group_absent(process, error)
        return
    process.wait(timeout=2)


def run_stages(config: dict, *, command_factory: Callable | None = None) -> dict:
    """Internal supervisor body; injection exists for subprocess cleanup tests."""
    run_dir = Path(config["run_dir"])
    state = _read_json(run_dir / "job-state.json")
    if state is None or state.get("status") != "starting" or state.get("config_hash") != _identity(config):
        raise ValueError("Supervisor configuration/state mismatch")
    deadline = config["started_monotonic"] + config["max_seconds"]
    started = time.monotonic()
    process = None
    completed = []
    state.update(
        status="running",
        supervisor_pid=os.getpid(),
        supervisor_startup_seconds=started - config["started_monotonic"],
    )
    _write_json(run_dir / "job-state.json", state)
    try:
        with (run_dir / "job.log").open("ab") as log:
            for stage in STAGES:
                state.update(stage=stage, stages_completed=list(completed))
                _write_json(run_dir / "job-state.json", state)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("local-training-stage", config["max_seconds"])
                process = subprocess.Popen(
                    (command_factory or _stage_command)(config, stage),
                    cwd=REPOSITORY,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=offline_environment(),
                    start_new_session=True,
                )
                code = process.wait(timeout=remaining)
                if code:
                    raise subprocess.CalledProcessError(code, "local-training-stage")
                process = None
                completed.append(stage)
        state.update(status="completed", stages_completed=completed)
    except BaseException as error:  # noqa: BLE001 - persist failure and clean up owned processes on every exit
        cleanup_error = None
        if process is not None:
            try:
                _stop_owned_group(process)
            except (OSError, RuntimeError, subprocess.SubprocessError) as termination_error:
                cleanup_error = termination_error
        state.update(
            status="timed_out"
            if isinstance(error, subprocess.TimeoutExpired) and cleanup_error is None
            else "failed",
            exception_type=type(error).__name__,
            stages_completed=completed,
        )
        if cleanup_error is not None:
            state.update(cleanup_confirmed=False, cleanup_exception_type=type(cleanup_error).__name__)
        if isinstance(error, subprocess.CalledProcessError):
            state["exit_code"] = error.returncode
    finally:
        finished = time.monotonic()
        state.update(
            finished_at_unix=time.time(),
            elapsed_seconds=finished - config["started_monotonic"],
            supervisor_elapsed_seconds=finished - started,
        )
        _write_json(run_dir / "job-state.json", state)
    return state


def _run(run_dir: Path, descriptor: int) -> int:
    # A real inherited lifetime lock, rather than PID/name heuristics, proves
    # ownership. Verify its inode so another open descriptor is not mistaken for
    # the job lock. Do not reopen or release the shared open-file description.
    expected = os.stat(run_dir / "job.lock")
    actual = os.fstat(descriptor)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError("Supervisor did not inherit the correct job lock")
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def interrupted(signum, frame):
        raise InterruptedError("Supervisor was stopped")

    previous = {signum: signal.signal(signum, interrupted) for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        config = _read_json(run_dir / "job-config.json")
        if config is None or config.get("run_dir") != str(run_dir.resolve()):
            raise ValueError("Supervisor configuration does not match the locked directory")
        state = run_stages(config)
        return 0 if state["status"] == "completed" else 1
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        os.close(descriptor)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    start = commands.add_parser("start", help="Launch one detached bounded experiment")
    start.add_argument("--python", required=True)
    start.add_argument("--model", required=True)
    start.add_argument("--data", required=True)
    start.add_argument("--run-dir", required=True)
    start.add_argument("--max-seconds", type=int, default=3600)
    start.add_argument("--iters", type=int, default=200)
    start.add_argument("--max-seq-length", type=int, default=4096)
    start.add_argument("--memory-gib", type=int, default=8)
    status = commands.add_parser("status", help="Read durable state without loading a model")
    status.add_argument("--run-dir", required=True)
    worker = commands.add_parser("_run", help=argparse.SUPPRESS)
    worker.add_argument("--run-dir", required=True)
    worker.add_argument("--lock-fd", required=True, type=int)
    options = parser.parse_args(argv)
    os.umask(0o077)
    try:
        if options.action == "_run":
            return _run(Path(options.run_dir), options.lock_fd)
        if options.action == "status":
            result = job_status(options.run_dir)
        else:
            result = start_job(
                make_config(**{key: value for key, value in vars(options).items() if key != "action"})
            )
    except Exception as error:  # noqa: BLE001 - CLI errors expose types only, never private paths or source
        print(json.dumps({"status": "failed", "exception_type": type(error).__name__}))
        return 1
    print(json.dumps(result, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
