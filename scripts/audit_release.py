"""Audit selected public source files and distribution archives without extracting.

This is a bounded release check, not a guarantee that every possible secret or
personal fact can be recognized. Findings contain redacted paths and rule names,
never matched content. Deliberate placeholder URLs in security fixtures remain
allowed; private network addresses alone are not credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import subprocess
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

MAX_FILE_BYTES = 10_000_000
MAX_TOTAL_BYTES = 50_000_000
MAX_ARCHIVE_BYTES = 25_000_000
MAX_MEMBERS = 5_000
SKIPPED_SOURCE_DIRS = frozenset(
    {".git", ".venv", "venv", "dist", "__pycache__", ".pytest_cache", ".ruff_cache"}
)
PRIVATE_DIRS = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        ".ssh",
        ".scraper",
        "state",
        "snapshots",
        "source-views",
        "reports",
        "logs",
        "models",
        "model-server",
        "secrets",
        "work",
        ".cache",
    }
)
PRIVATE_NAMES = frozenset(
    {".env", ".envrc", ".netrc", ".pypirc", ".npmrc", ".DS_Store", "credentials.json", "secrets.json"}
)
PRIVATE_SUFFIXES = frozenset(
    {
        ".pem",
        ".key",
        ".p12",
        ".pfx",
        ".sqlite3",
        ".db",
        ".log",
        ".safetensors",
        ".gguf",
        ".ggml",
        ".ckpt",
        ".onnx",
        ".pt",
        ".pth",
        ".bin",
        ".pyc",
    }
)
SECRET_PATTERNS = (
    ("private_key", re.compile(r"-----BEGIN (?:OPENSSH|RSA|EC|DSA|ENCRYPTED)? ?PRIVATE KEY-----")),
    ("github_token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{24,}\b")),
    ("provider_token", re.compile(r"\bsk-(?:proj-|ant-api\d+-)?[A-Za-z0-9_-]{24,}\b")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
)
CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?im)\b(?:[A-Z0-9_]*API_KEY|HF_TOKEN|[A-Z0-9_]*ACCESS_TOKEN|[A-Z0-9_]*CLIENT_SECRET)"
    r"[\"']?[\t ]*[:=][\t ]*[\"']([A-Za-z0-9_./+=-]{20,})[\"']"
)
ENV_ASSIGNMENT = re.compile(
    r"(?m)^[\t ]*(?:export[\t ]+)?(?:[A-Z0-9_]*API_KEY|HF_TOKEN|[A-Z0-9_]*ACCESS_TOKEN|[A-Z0-9_]*CLIENT_SECRET)"
    r"[\t ]*=[\t ]*([A-Za-z0-9_./+=-]{20,})[\t ]*(?:#[^\n]*)?$"
)
HOME_PATH = re.compile(r"(?:/(?:Users|home)/[^/\s\"'<>$]+|[A-Za-z]:[\\/]Users[\\/][^\\/\s\"'<>$]+)")
URL = re.compile(r"\b(?:https?|ssh|git\+https)://[^\s\"'<>\\]+", re.IGNORECASE)
PLACEHOLDER_USERS = frozenset({"", "user", "username", "example", "test", "git"})
PLACEHOLDER_PASSWORDS = frozenset({"", "pass", "password", "secret", "test", "example"})


@dataclass(frozen=True)
class Finding:
    path: str
    rule: str


def display_path(value: str) -> str:
    """Avoid logging a token even when it occurs in a malicious filename."""
    for _, pattern in SECRET_PATTERNS:
        value = pattern.sub("[redacted]", value)
    value = CREDENTIAL_ASSIGNMENT.sub(
        lambda match: match.group().replace(match.group(1), "[redacted]"), value
    )
    value = re.sub(r"(?i)((?:https?|ssh)://)[^@\s]+@", r"\1[redacted]@", value)
    return value[:500]


def unsafe_path(name: str) -> bool:
    normalized = name.replace("\\", "/")
    return (
        not normalized
        or normalized.startswith("/")
        or bool(re.match(r"^[A-Za-z]:", normalized))
        or ".." in PurePosixPath(normalized).parts
        or "\x00" in normalized
    )


def forbidden_path(name: str) -> bool:
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if any(part in PRIVATE_DIRS for part in parts[:-1]):
        return True
    basename = parts[-1] if parts else ""
    if basename in PRIVATE_NAMES or basename.startswith(("id_rsa", "id_ed25519")):
        return True
    if basename.startswith(".env.") and basename != ".env.example":
        return True
    if basename == "config.json" or (basename.startswith("config-") and basename.endswith(".json")):
        return True
    return (
        PurePosixPath(basename).suffix.lower() in PRIVATE_SUFFIXES
        or ".sqlite3-" in basename
        or ".log." in basename
    )


def placeholder_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").lower()
        reserved = hostname in {
            "localhost",
            "127.0.0.1",
            "::1",
            "example.com",
            "example.org",
            "example.net",
            "93.184.216.34",
        } or hostname.endswith((".test", ".example", ".invalid"))
        return (
            reserved
            and unquote(parsed.username or "") in PLACEHOLDER_USERS
            and unquote(parsed.password or "") in PLACEHOLDER_PASSWORDS
        )
    except ValueError:
        return False


def content_rules(content: bytes, path: str = "") -> set[str]:
    text = content.decode("utf-8", errors="replace")
    rules = {rule for rule, pattern in SECRET_PATTERNS if pattern.search(text)}
    env_file = PurePosixPath(path.split("!", 1)[-1]).name.startswith(".env")
    if CREDENTIAL_ASSIGNMENT.search(text) or (env_file and ENV_ASSIGNMENT.search(text)):
        rules.add("credential_assignment")
    if HOME_PATH.search(text):
        rules.add("absolute_home_path")
    for match in URL.finditer(text):
        value = match.group()
        try:
            parsed = urlsplit(value)
            has_userinfo = parsed.username is not None or parsed.password is not None
        except ValueError:
            continue
        if has_userinfo and not placeholder_url(value):
            rules.add("credential_url")
    return rules


class Audit:
    def __init__(self):
        self.findings: list[Finding] = []
        self.files_scanned = 0
        self.archives_scanned = 0
        self.total_bytes = 0

    def add(self, path: str, rule: str):
        finding = Finding(display_path(path), rule)
        if finding not in self.findings:
            self.findings.append(finding)

    def check_bytes(self, path: str, data: bytes):
        if not self.permit_size(path, len(data)):
            return
        self.files_scanned += 1
        self.total_bytes += len(data)
        for rule in content_rules(data, path):
            self.add(path, rule)

    def permit_size(self, path: str, size: int) -> bool:
        if size < 0 or size > MAX_FILE_BYTES or self.total_bytes + size > MAX_TOTAL_BYTES:
            self.add(path, "size_limit")
            return False
        return True

    def source_file(self, root: Path, relative: str):
        if unsafe_path(relative):
            self.add(relative, "unsafe_path")
            return
        path = root / relative
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            self.add(relative, "symlink")
            return
        if forbidden_path(relative):
            self.add(relative, "private_file")
            return
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            with os.fdopen(fd, "rb") as stream:
                info = os.fstat(stream.fileno())
                if not stat.S_ISREG(info.st_mode):
                    self.add(relative, "non_regular_file")
                elif self.permit_size(relative, info.st_size):
                    self.check_bytes(relative, stream.read(MAX_FILE_BYTES + 1))
        except OSError:
            self.add(relative, "read_error")

    def source(self, root: Path, *, tracked: bool = False):
        root = root.resolve()
        if tracked:
            try:
                result = subprocess.run(
                    ["git", "-C", str(root), "ls-files", "-z"], capture_output=True, check=True
                )
            except (OSError, subprocess.CalledProcessError):
                self.add("source", "git_index_unavailable")
                return
            names = result.stdout.decode("utf-8", errors="surrogateescape").split("\x00")
            for index, name in enumerate(filter(None, names)):
                if index >= MAX_MEMBERS:
                    self.add("source", "member_limit")
                    break
                self.source_file(root, name)
            return
        count = 0
        for directory, directories, filenames in os.walk(root, followlinks=False):
            parent = Path(directory)
            for name in list(directories):
                count += 1
                if count > MAX_MEMBERS:
                    self.add("source", "member_limit")
                    return
                path = parent / name
                relative = path.relative_to(root).as_posix()
                if path.is_symlink():
                    self.add(relative, "symlink")
                    directories.remove(name)
                elif name in SKIPPED_SOURCE_DIRS:
                    directories.remove(name)
                elif name in PRIVATE_DIRS:
                    self.add(relative, "private_directory")
                    directories.remove(name)
            for name in filenames:
                count += 1
                if count > MAX_MEMBERS:
                    self.add("source", "member_limit")
                    return
                self.source_file(root, (parent / name).relative_to(root).as_posix())

    def archive_member(
        self, archive_name: str, name: str, size: int, *, regular: bool, directory: bool
    ) -> bool:
        label = archive_name + "!" + name
        if unsafe_path(name):
            self.add(label, "unsafe_path")
            return False
        if directory:
            if any(part in PRIVATE_DIRS for part in PurePosixPath(name.replace("\\", "/")).parts):
                self.add(label, "private_directory")
            return False
        if not regular:
            self.add(label, "archive_link_or_special_file")
            return False
        if forbidden_path(name):
            self.add(label, "private_file")
            return False
        return self.permit_size(label, size)

    def archive(self, path: Path):
        name = path.name
        self.archives_scanned += 1
        try:
            if path.is_symlink() or path.stat().st_size > MAX_ARCHIVE_BYTES:
                self.add(name, "archive_size_or_link")
                return
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as bundle:
                    entries = bundle.infolist()
                    if len(entries) > MAX_MEMBERS:
                        self.add(name, "member_limit")
                        return
                    for entry in entries:
                        mode = entry.external_attr >> 16
                        regular = not stat.S_ISLNK(mode) and (not stat.S_IFMT(mode) or stat.S_ISREG(mode))
                        if self.archive_member(
                            name, entry.filename, entry.file_size, regular=regular, directory=entry.is_dir()
                        ):
                            self.check_bytes(name + "!" + entry.filename, bundle.read(entry))
            else:
                with tarfile.open(path, "r:*") as bundle:
                    for index, entry in enumerate(bundle):
                        if index >= MAX_MEMBERS:
                            self.add(name, "member_limit")
                            break
                        if not self.permit_size(name + "!" + entry.name, entry.size):
                            break
                        before = len(self.findings)
                        permitted = self.archive_member(
                            name, entry.name, entry.size, regular=entry.isfile(), directory=entry.isdir()
                        )
                        if not permitted and len(self.findings) > before:
                            break  # do not decompress/seek past a rejected TAR payload
                        if permitted:
                            stream = bundle.extractfile(entry)
                            if stream is None:
                                self.add(name + "!" + entry.name, "read_error")
                            else:
                                self.check_bytes(name + "!" + entry.name, stream.read(MAX_FILE_BYTES + 1))
        except (OSError, ValueError, EOFError, tarfile.TarError, zipfile.BadZipFile, RuntimeError):
            self.add(name, "archive_read_error")

    def report(self) -> dict:
        return {
            "status": "passed" if not self.findings else "failed",
            "files_scanned": self.files_scanned,
            "archives_scanned": self.archives_scanned,
            "bytes_scanned": self.total_bytes,
            "findings": [{"path": item.path, "rule": item.rule} for item in self.findings],
            "scope": "Source tree (or explicit Git index) and bounded archive members; no extraction or network calls",
            "limitations": "Pattern checks do not establish that every secret, private fact or dependency risk has been found; Git history is not scanned",
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument(
        "--tracked", action="store_true", help="Audit Git-index paths instead of the source walk"
    )
    parser.add_argument("--archive", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    audit = Audit()
    if not args.source.is_dir():
        audit.add("source", "source_unavailable")
    else:
        audit.source(args.source, tracked=args.tracked)
    for archive in args.archive:
        audit.archive(archive)
    report = audit.report()
    print(json.dumps(report, indent=2))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
