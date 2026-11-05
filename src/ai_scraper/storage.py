"""Private, atomic files; JSON logs must never contain credentials."""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

SENSITIVE = re.compile(
    r"(authorization|api[_-]?key|(?:^|_)(?:access_|refresh_|hf_)?token$|secret|password|cookie)",
    re.IGNORECASE,
)


def redact(value: Any, secrets: tuple[str, ...] = ()) -> Any:
    if isinstance(value, dict):
        return {k: "[REDACTED]" if SENSITIVE.search(k) else redact(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", value)
        value = re.sub(r"(?i)([?&](?:token|api_key|key|password|secret)=)[^&#\s]+", r"\1[REDACTED]", value)
    return value


def private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def atomic_json(path: Path, value: Any) -> None:
    # Exports may target an existing user directory. Only dedicated state roots
    # use private_dir(); do not chmod a caller's existing output parent.
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temp_name = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
