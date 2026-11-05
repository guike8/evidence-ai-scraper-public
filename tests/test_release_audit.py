from __future__ import annotations

import io
import json
import stat
import subprocess
import tarfile
import zipfile
from types import SimpleNamespace

import pytest

from scripts import audit_release


def token():
    return "gh" + "p_" + "A" * 32


def write_tar(path, name, data=b"", *, kind=None, size=None):
    with tarfile.open(path, "w:gz") as bundle:
        entry = tarfile.TarInfo(name)
        entry.size = len(data) if size is None else size
        if kind is not None:
            entry.type = kind
            entry.linkname = "outside"
        bundle.addfile(entry, io.BytesIO(data) if size is None else None)


@pytest.mark.parametrize(
    "content,rule",
    [
        (lambda: token(), "github_token"),
        (lambda: "hf" + "_" + "b" * 30, "huggingface_token"),
        (lambda: "-----BEGIN " + "OPENSSH PRIVATE KEY-----", "private_key"),
        (lambda: 'BRIGHTDATA_API_KEY="' + "b" * 32 + '"', "credential_assignment"),
        (lambda: "/" + "Users" + "/operator/cache", "absolute_home_path"),
        (lambda: "https:" + "//operator:nonplaceholder@service.example/", "credential_url"),
    ],
)
def test_source_content_findings_never_echo_the_secret(tmp_path, content, rule):
    secret = content()
    (tmp_path / "source.py").write_text(secret)
    audit = audit_release.Audit()
    audit.source(tmp_path)
    report = audit.report()
    assert {finding["rule"] for finding in report["findings"]} == {rule}
    assert secret not in json.dumps(report)
    assert report["findings"][0]["path"] == "source.py"


def test_public_env_example_and_explicit_security_placeholders_are_allowed(tmp_path):
    (tmp_path / ".env.example").write_text("HF_TOKEN=\nBRIGHTDATA_API_KEY=\nLOCAL_API_KEY=\n")
    (tmp_path / "test_urls.py").write_text(
        "http://user:pass@example.com/\nhttp://user:secret@localhost/v1\n"
        "http://user:password@93.184.216.34/\nhttp://127.0.0.1/private\n"
    )
    audit = audit_release.Audit()
    audit.source(tmp_path)
    assert audit.report()["status"] == "passed"
    assert audit.files_scanned == 2


def test_unquoted_nonplaceholder_env_value_is_detected(tmp_path):
    (tmp_path / ".env.example").write_text("BRIGHTDATA_API_KEY=" + "a" * 32 + "\n")
    audit = audit_release.Audit()
    audit.source(tmp_path)
    assert audit.report()["findings"] == [{"path": ".env.example", "rule": "credential_assignment"}]


@pytest.mark.parametrize("name", [".env", "config.json", "weight.safetensors", "key.pem", "id_ed25519"])
def test_forbidden_source_files_fail_without_reading_their_values(tmp_path, name):
    (tmp_path / name).write_text(token())
    audit = audit_release.Audit()
    audit.source(tmp_path)
    assert audit.report()["findings"] == [{"path": name, "rule": "private_file"}]
    assert audit.files_scanned == 0


def test_source_skips_tool_caches_but_flags_private_directories_and_symlinks(tmp_path):
    for name in [".git", ".venv", "dist", "state"]:
        directory = tmp_path / name
        directory.mkdir()
        (directory / "private.txt").write_text(token())
    (tmp_path / "linked.py").symlink_to(tmp_path / "state/private.txt")
    audit = audit_release.Audit()
    audit.source(tmp_path)
    assert {(item.path, item.rule) for item in audit.findings} == {
        ("state", "private_directory"),
        ("linked.py", "symlink"),
    }
    assert audit.files_scanned == 0


def test_tracked_mode_audits_force_added_ignored_files(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("HF_TOKEN=example")

    def git_index(arguments, **kwargs):
        assert arguments[-2:] == ["ls-files", "-z"]
        return SimpleNamespace(stdout=b".env\0")

    monkeypatch.setattr(subprocess, "run", git_index)
    audit = audit_release.Audit()
    audit.source(tmp_path, tracked=True)
    assert audit.report()["findings"] == [{"path": ".env", "rule": "private_file"}]


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
def test_archive_contents_are_audited_without_extraction(tmp_path, extension):
    path = tmp_path / ("package" + extension)
    if extension == ".whl":
        with zipfile.ZipFile(path, "w") as bundle:
            bundle.writestr("package/source.py", token())
    else:
        write_tar(path, "package/source.py", token().encode())
    audit = audit_release.Audit()
    audit.archive(path)
    assert audit.report()["findings"] == [{"path": path.name + "!package/source.py", "rule": "github_token"}]
    assert not (tmp_path / "package").exists()
    assert token() not in json.dumps(audit.report())


@pytest.mark.parametrize(
    "name", ["../escaped.py", "/absolute.py", "C:\\escaped.py", "package/../../escape.py"]
)
def test_unsafe_archive_paths_are_rejected(tmp_path, name):
    path = tmp_path / "unsafe.tar.gz"
    write_tar(path, name, b"benign")
    audit = audit_release.Audit()
    audit.archive(path)
    assert audit.report()["findings"][0]["rule"] == "unsafe_path"
    assert audit.files_scanned == 0


def test_archive_symlinks_and_private_empty_directories_are_rejected(tmp_path):
    path = tmp_path / "links.whl"
    with zipfile.ZipFile(path, "w") as bundle:
        entry = zipfile.ZipInfo("package/link")
        entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        bundle.writestr(entry, "outside")
        bundle.writestr("package/state/", "")
    audit = audit_release.Audit()
    audit.archive(path)
    assert {item.rule for item in audit.findings} == {"archive_link_or_special_file", "private_directory"}
    assert audit.files_scanned == 0


def test_archive_limits_reject_claimed_large_payload_before_reading_it(tmp_path):
    path = tmp_path / "huge.tar.gz"
    write_tar(path, "package/huge.txt", size=audit_release.MAX_FILE_BYTES + 1)
    audit = audit_release.Audit()
    audit.archive(path)
    assert audit.report()["findings"] == [{"path": "huge.tar.gz!package/huge.txt", "rule": "size_limit"}]
    assert audit.files_scanned == 0


def test_secret_filename_is_redacted_in_findings(tmp_path):
    name = token() + ".py"
    (tmp_path / name).write_text(token())
    audit = audit_release.Audit()
    audit.source(tmp_path)
    report = json.dumps(audit.report())
    assert token() not in report
    assert "[redacted].py" in report


def test_cli_exit_status_and_clean_archive(tmp_path, capsys):
    source = tmp_path / "source"
    source.mkdir()
    (source / ".env.example").write_text("HF_TOKEN=\n")
    path = tmp_path / "clean.whl"
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("package/__init__.py", "__version__ = '0.3.0'\n")
        bundle.writestr("package-0.3.0.dist-info/licenses/LICENSE", "MIT License\n")
    assert audit_release.main([str(source), "--archive", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "passed"
    (source / "settings.py").write_text(token())
    assert audit_release.main([str(source), "--archive", str(path)]) == 1
    failed = capsys.readouterr().out
    assert token() not in failed
    assert json.loads(failed)["status"] == "failed"
