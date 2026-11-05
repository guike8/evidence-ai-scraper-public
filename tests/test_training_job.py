"""Real lifetime locks and owned-process cleanup, without importing MLX."""

from __future__ import annotations

import concurrent.futures
import importlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from training import job

SENTINEL_NAMES = (
    "HF_TOKEN",
    "OPENAI_API_KEY",
    "BRIGHTDATA_API_KEY",
    "AWS_SECRET_ACCESS_KEY",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "SSH_AUTH_SOCK",
    "UNLISTED_SECRET",
)


@pytest.fixture(autouse=True)
def private_checkout(tmp_path, monkeypatch):
    """Use a real interpreter and production supervisor, with a stub MLX module.

    Executing a new shebang wrapper and then execing Python a second time adds
    platform startup work absent from the actual --python interpreter workflow.
    The isolated source checkout replaces only the expensive training module.
    """
    checkout = tmp_path / "checkout"
    package = checkout / "training"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    shutil.copyfile(job.__file__, package / "job.py")
    monkeypatch.setattr(job, "REPOSITORY", checkout)


def fake_stage(tmp_path, *, sleep=0.15, failure_stage=None, grandchild=False, clean_environment=False):
    """Substitute only the model stage, never the Python executable."""
    package = tmp_path / "checkout" / "training"
    if clean_environment:
        # -m imports this package in both the supervisor and every stage, before
        # either module can scrub an incorrectly inherited environment itself.
        (package / "__init__.py").write_text(
            f"import os\nassert not set({SENTINEL_NAMES!r}).intersection(os.environ)\n"
        )
    script = package / "train.py"
    script.write_text(
        "import os,sys,time,subprocess\n"
        "from pathlib import Path\n"
        "stage=sys.argv[sys.argv.index('--stage')+1]\n"
        "root=Path(sys.argv[sys.argv.index('--model')+1])\n"
        "with (root/'calls').open('a') as stream: stream.write(stage+'\\n')\n"
        "assert os.environ['HF_HUB_OFFLINE']=='1'\n"
        "assert os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']=='1'\n"
        + (
            "heartbeat=str(root/'heartbeat')\n"
            "code='import time,signal\\nfrom pathlib import Path\\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)"
            "\\np=Path('+repr(heartbeat)+')\\nuntil=time.monotonic()+15\\nwhile time.monotonic()<until:\\n"
            ' with p.open("a") as stream: stream.write(".")\\n time.sleep(0.03)\'\n'
            "child=subprocess.Popen([sys.executable,'-c',code])\n"
            "(root/'grandchild.pid').write_text(str(child.pid))\n"
            if grandchild
            else ""
        )
        + f"time.sleep({sleep!r})\n"
        + f"sys.exit(7 if stage=={failure_stage!r} else 0)\n"
    )


def config(tmp_path, **changes):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)
    fake_stage(tmp_path)
    return job.make_config(
        python=sys.executable,
        model=str(model),
        data=str(data),
        run_dir=str(tmp_path / "private-run"),
        max_seconds=10,
        **changes,
    )


def wait_done(run_dir, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = job.job_status(run_dir)
        if not result["running"] and result["status"] in job.TERMINAL:
            return result
        time.sleep(0.03)
    raise AssertionError(f"Job did not finish: {job.job_status(run_dir)}")


def test_concurrent_start_inherits_lifetime_lock_and_executes_each_stage_once(tmp_path):
    options = config(tmp_path)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: job.start_job(options), range(2)))
    assert sum(result["started"] for result in results) == 1
    assert job.job_status(options["run_dir"])["running"]
    final = wait_done(options["run_dir"])
    assert final["status"] == "completed"
    assert final["stages_completed"] == list(job.STAGES)
    assert (Path(options["model"]) / "calls").read_text().splitlines() == list(job.STAGES)
    assert Path(options["run_dir"]).stat().st_mode & 0o777 == 0o700
    for name in ("job.lock", "job-state.json", "job-config.json", "job.log"):
        assert (Path(options["run_dir"]) / name).stat().st_mode & 0o777 == 0o600


def test_completed_start_is_idempotent_and_public_state_has_no_private_paths(tmp_path):
    options = config(tmp_path)
    job.start_job(options)
    assert wait_done(options["run_dir"])["status"] == "completed"
    again = job.start_job(options)
    assert not again["started"] and not again["running"]
    assert (Path(options["model"]) / "calls").read_text().splitlines() == list(job.STAGES)
    state_text = (Path(options["run_dir"]) / "job-state.json").read_text()
    assert str(tmp_path) not in state_text
    assert "python" not in state_text and "environment" not in state_text


def contaminate_environment(monkeypatch):
    for name in SENTINEL_NAMES:
        monkeypatch.setenv(name, "unit-test-secret-sentinel")
    monkeypatch.setenv("HTTPS_PROXY", "http://" + "test-user:placeholder" + "@proxy.invalid:9")


def test_offline_environment_preserves_os_settings_without_credentials(monkeypatch):
    contaminate_environment(monkeypatch)
    allowed = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TMP", "TEMP")
        if name in os.environ
    }
    environment = job.offline_environment()
    assert not set(SENTINEL_NAMES).intersection(environment)
    assert all(environment[name] == value for name, value in allowed.items())
    assert environment["HF_HUB_OFFLINE"] == environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert environment["HF_HUB_DISABLE_TELEMETRY"] == environment["DO_NOT_TRACK"] == "1"


def test_detached_supervisor_and_stage_processes_do_not_inherit_credentials(tmp_path, monkeypatch):
    contaminate_environment(monkeypatch)
    options = config(tmp_path)
    # Package import checks happen before dispatching the actual supervisor,
    # then again when each stage is launched by the real interpreter.
    fake_stage(tmp_path, clean_environment=True)
    job.start_job(options)
    assert wait_done(options["run_dir"])["status"] == "completed"
    assert (Path(options["model"]) / "calls").read_text().splitlines() == list(job.STAGES)


def test_stage_launch_also_scrubs_a_direct_callers_environment(tmp_path, monkeypatch):
    # Exercise the stage boundary independently: a sanitized supervisor alone
    # must not make a missing stage-specific env argument pass this check.
    contaminate_environment(monkeypatch)
    options = config(tmp_path)
    directory = Path(options["run_dir"])
    directory.mkdir()
    job._write_json(
        directory / "job-state.json", {"status": "starting", "config_hash": job._identity(options)}
    )
    options["started_monotonic"] = time.monotonic()
    code = f"import os; assert not set({SENTINEL_NAMES!r}).intersection(os.environ)"
    final = job.run_stages(options, command_factory=lambda *_: [sys.executable, "-c", code])
    assert final["status"] == "completed"
    assert final["stages_completed"] == list(job.STAGES)


def test_startup_measurement_keeps_the_parent_deadline_and_whole_job_elapsed(tmp_path, monkeypatch):
    options = config(tmp_path)
    options["max_seconds"] = 60
    directory = Path(options["run_dir"])
    directory.mkdir()
    job._write_json(
        directory / "job-state.json", {"status": "starting", "config_hash": job._identity(options)}
    )
    options["started_monotonic"] = 100.0
    clock = iter((110.0, 111.0, 112.0, 113.0, 115.0))
    timeouts = []

    class Child:
        def wait(self, timeout):
            timeouts.append(timeout)
            return 0

    monkeypatch.setattr(job.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(job.subprocess, "Popen", lambda *args, **kwargs: Child())
    final = job.run_stages(options)
    assert final["status"] == "completed"
    assert timeouts == [49.0, 48.0, 47.0]  # Deadline remains parent start 100 + budget 60.
    assert final["supervisor_startup_seconds"] == 10
    assert final["supervisor_elapsed_seconds"] == 5
    assert final["elapsed_seconds"] == 15


def test_stage_failure_is_terminal_and_is_never_automatically_restarted(tmp_path):
    options = config(tmp_path)
    fake_stage(tmp_path, failure_stage="sanity")
    job.start_job(options)
    final = wait_done(options["run_dir"])
    assert final["status"] == "failed"
    assert final["exit_code"] == 7
    assert final["exception_type"] == "CalledProcessError"
    assert final["stages_completed"] == ["preflight"]
    assert not job.start_job(options)["started"]
    assert (Path(options["model"]) / "calls").read_text().splitlines() == ["preflight", "sanity"]


def test_deadline_kills_only_owned_stage_and_its_descendants(tmp_path):
    options = config(tmp_path)
    options["max_seconds"] = 3
    fake_stage(tmp_path, sleep=60, grandchild=True)
    unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(60)"])
    try:
        started = time.monotonic()
        job.start_job(options)
        final = wait_done(options["run_dir"])
        assert final["status"] == "timed_out"
        assert final["stages_completed"] == []
        assert time.monotonic() - started < 8
        heartbeat = Path(options["model"]) / "heartbeat"
        assert heartbeat.exists()  # The grandchild actually ran before the deadline.
        size = heartbeat.stat().st_size
        time.sleep(0.2)
        assert heartbeat.stat().st_size == size
        assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=3)


def test_supervisor_stop_cleans_up_owned_stage_and_persists_failure(tmp_path):
    options = config(tmp_path)
    fake_stage(tmp_path, sleep=60)
    result = job.start_job(options)
    deadline = time.monotonic() + 3
    calls = Path(options["model"]) / "calls"
    while not calls.exists() and time.monotonic() < deadline:
        time.sleep(0.03)
    assert calls.exists()
    os.kill(result["supervisor_pid"], signal.SIGTERM)
    final = wait_done(options["run_dir"])
    assert final["status"] == "failed"
    assert final["exception_type"] == "InterruptedError"
    assert not job.start_job(options)["started"]


def test_cleanup_permission_error_preserves_original_failure_and_reports_unverified(tmp_path, monkeypatch):
    options = config(tmp_path)
    directory = Path(options["run_dir"])
    directory.mkdir()
    job._write_json(
        directory / "job-state.json", {"status": "starting", "config_hash": job._identity(options)}
    )
    options["started_monotonic"] = time.monotonic()

    class Child:
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("owned-test-stage", timeout)

    def denied_cleanup(_):
        raise PermissionError("private details must not enter state")

    monkeypatch.setattr(job.subprocess, "Popen", lambda *args, **kwargs: Child())
    monkeypatch.setattr(job, "_stop_owned_group", denied_cleanup)
    final = job.run_stages(options)
    assert final["status"] == "failed"
    assert final["exception_type"] == "TimeoutExpired"
    assert final["cleanup_exception_type"] == "PermissionError"
    assert final["cleanup_confirmed"] is False
    assert "private details" not in json.dumps(final)


def test_reaped_leader_cannot_authorize_signaling_a_reused_group(monkeypatch):
    attempted = []

    def kill_group(pid, signum):
        attempted.append((pid, signum))

    monkeypatch.setattr(job.os, "killpg", kill_group)
    child = type("ReapedChild", (), {"pid": 123456789, "returncode": 0, "wait": lambda *_args, **_kw: 0})()
    with pytest.raises(RuntimeError, match="ownership"):
        job._stop_owned_group(child)
    assert attempted and all(item == (123456789, 0) for item in attempted)


def test_permission_denial_requires_proof_of_absence_before_cleanup_is_accepted(monkeypatch):
    attempted = []
    waited = []

    class Child:
        pid = 123456789
        returncode = None

        def wait(self, timeout):
            waited.append(timeout)
            self.returncode = -signal.SIGTERM
            return self.returncode

    def kill_group(pid, signum):
        attempted.append(signum)
        if signum == signal.SIGKILL:
            raise PermissionError("test zombie-only group")
        if signum == 0:
            assert waited  # Only the owned leader was reaped, before the probe.
            raise ProcessLookupError

    monkeypatch.setattr(job.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(job.os, "killpg", kill_group)
    monkeypatch.setattr(job.time, "sleep", lambda _: None)
    job._stop_owned_group(Child())
    assert attempted == [signal.SIGTERM, signal.SIGKILL, 0]


def test_permission_denial_with_present_group_stays_unverified(monkeypatch):
    attempted = []

    class Child:
        pid = 123456789
        returncode = None

        def wait(self, timeout):
            self.returncode = -signal.SIGTERM
            return self.returncode

    def kill_group(pid, signum):
        attempted.append(signum)
        if signum != 0:
            raise PermissionError("test denied signal")

    monkeypatch.setattr(job.os, "getpgid", lambda pid: pid)
    monkeypatch.setattr(job.os, "killpg", kill_group)
    with pytest.raises(PermissionError):
        job._stop_owned_group(Child())
    assert attempted[0] == signal.SIGTERM
    assert len(attempted) > 1 and all(signum == 0 for signum in attempted[1:])


def test_stale_pid_does_not_authorize_resume_or_signal_another_process(tmp_path):
    options = config(tmp_path)
    run_dir = Path(options["run_dir"])
    run_dir.mkdir()
    state = {"status": "running", "supervisor_pid": os.getpid(), "config_hash": job._identity(options)}
    job._write_json(run_dir / "job-state.json", state)
    final = job.job_status(run_dir)
    assert final["status"] == "interrupted"
    assert final["exception_type"] == "SupervisorLost"
    assert not job.start_job(options)["started"]


def test_existing_weights_require_new_run_directory(tmp_path):
    options = config(tmp_path)
    run_dir = Path(options["run_dir"])
    (run_dir / "adapter").mkdir(parents=True)
    with pytest.raises(ValueError, match="new run directory"):
        job.start_job(options)


def test_launch_exception_is_persisted_and_does_not_retry(tmp_path, monkeypatch):
    options = config(tmp_path)

    def failed_launch(*args, **kwargs):
        raise OSError("private path must not enter portable state")

    monkeypatch.setattr(job.subprocess, "Popen", failed_launch)
    with pytest.raises(OSError):
        job.start_job(options)
    final = job.job_status(options["run_dir"])
    assert final["status"] == "failed"
    assert final["exception_type"] == "OSError"
    assert "private path" not in json.dumps(final)
    assert not job.start_job(options)["started"]


def test_different_configuration_cannot_reuse_a_completed_directory(tmp_path):
    options = config(tmp_path)
    job.start_job(options)
    wait_done(options["run_dir"])
    with pytest.raises(ValueError, match="another configuration"):
        job.start_job({**options, "iters": 400})


def test_status_imports_no_optional_model_packages(tmp_path, monkeypatch, capsys):
    before = set(sys.modules)
    assert job.main(["status", "--run-dir", str(tmp_path / "absent")]) == 0
    assert json.loads(capsys.readouterr().out) == {"status": "not_started", "running": False}
    assert not any(name.startswith(("mlx", "transformers")) for name in set(sys.modules) - before)
    # Reload itself is also pure stdlib.
    importlib.reload(job)


@pytest.mark.parametrize(
    "bad", [{"iters": 201}, {"max_seconds": 0}, {"max_seconds": True}, {"memory_gib": 64}]
)
def test_invalid_budgets_rejected_before_launch(tmp_path, bad):
    model, data = tmp_path / "model", tmp_path / "data"
    model.mkdir()
    data.mkdir()
    values = {
        "python": sys.executable,
        "model": str(model),
        "data": str(data),
        "run_dir": str(tmp_path / "run"),
        **bad,
    }
    with pytest.raises(ValueError):
        job.make_config(**values)
