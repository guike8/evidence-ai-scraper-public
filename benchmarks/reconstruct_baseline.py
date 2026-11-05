"""Reconstruct the historical Python baseline without importing private history.

Only Python source is copied into a new directory outside the checkout. A local
empty Git repository provides a safe patch root; no remotes or commits exist.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="New directory outside this checkout")
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output == project or output.is_relative_to(project) or output.exists():
        parser.error("Output must be a new directory outside this checkout")
    source = project / "src"
    output.mkdir(parents=True, mode=0o700)
    for path in source.rglob("*.py"):
        if path.is_symlink():
            raise ValueError("Source symlinks are not supported")
        target = output / "src" / path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    subprocess.run(["git", "init", "--quiet", "--initial-branch=baseline"], cwd=output, check=True)
    patch = project / "benchmarks/baseline-src.patch"
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=output, check=True)
    subprocess.run(["git", "apply", str(patch)], cwd=output, check=True)
    print("Historical source reconstructed; use its src directory as PYTHONPATH for the current collector.")


if __name__ == "__main__":
    main()
