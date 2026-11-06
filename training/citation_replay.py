"""Reproduce the fixed citation-v3 training replay recipe without model calls.

This accepts only the two measured, authored dataset identities. It is not a
general dataset merger; validation/test rows can never be selected for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
from collections import Counter
from pathlib import Path

from ai_scraper.extraction.huggingface import HuggingFaceExtractor, request_instruction

from .dataset import SCHEMAS, SPLITS, _write, canonical, check_disjoint, digest
from .evaluate import MAX_FILE_BYTES, MAX_ROW_BYTES, MAX_ROWS, _context, _parse

VERSION = "citation-v3-replay-mix-v1"
SEED = 2026110704
VALIDATION_INDICES = frozenset({0, 2, 6, 8, 10, 14, 16, 19})
EXPECTED_FINGERPRINT = "410a7f708a512ca305c1cbd92db55fd1a3938d06a7804484c3932e0b821ee47e"
INPUTS = {
    "original": {
        "generator": "synthetic-pilot-v1",
        "seed": 20261105,
        "per_schema": {"train": 80, "valid": 8, "test": 12},
        "fingerprint": "2f3639f41a4bf6a5d41eaf58a959067abeb61f61abf1c9cad5d222ae80be2499",
    },
    "v3": {
        "generator": "synthetic-citation-v3",
        "seed": 2026110701,
        "per_schema": {"train": 120, "valid": 20, "test": 12},
        "fingerprint": "48f6380a2da4726569fe37972db42787141dae78f5841333b7d3704989a074be",
    },
}


def _read(path: Path, limit: int) -> bytes:
    """Capture a bounded regular file, rejecting symlinks and concurrent changes."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError("Replay input must be a bounded regular file")
        body = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
        current = path.lstat()
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
    if (
        len(body) != before.st_size
        or identity(before) != identity(after)
        or identity(before) != identity(current)
    ):
        raise ValueError("Replay input changed while reading")
    return body


def _jsonl(body: bytes) -> list[dict]:
    rows = []
    for line in body.splitlines():
        if not line.strip():
            continue
        if len(line) > MAX_ROW_BYTES or len(rows) >= MAX_ROWS:
            raise ValueError("Replay input exceeds its row bound")
        row = _parse(line.decode("utf-8"))
        if not isinstance(row, dict):
            raise TypeError("Replay rows must be objects")
        rows.append(row)
    return rows


def _validate_pair(row: dict, case: dict, *, shuffled: bool) -> None:
    request, spec, snapshot, gold = _context(case)
    target = json.dumps(
        gold.model_dump(mode="json"), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    messages = row.get("messages")
    if (
        set(row) != {"messages"}
        or not isinstance(messages, list)
        or len(messages) != 3
        or any(not isinstance(message, dict) or set(message) != {"role", "content"} for message in messages)
        or [message["role"] for message in messages] != ["system", "user", "assistant"]
        or messages[0]["content"] != request_instruction(request, spec)
        or messages[2]["content"] != target
        or not isinstance(messages[1]["content"], str)
        or not messages[1]["content"].startswith("UNTRUSTED_SOURCE_JSON\n")
    ):
        raise ValueError("Replay messages must match their authored schema-ordered target")
    adapter = HuggingFaceExtractor("synthetic-offline", "synthetic-offline", client=object()).for_request(
        request, spec
    )
    source = adapter._source_content(snapshot)
    message_source = messages[1]["content"].split("\n", 1)[1]
    actual = _parse(message_source)
    expected = _parse(source)
    if shuffled:
        if not isinstance(actual, dict) or not isinstance(actual.get("sections"), list):
            raise ValueError("Replay source view differs from its immutable snapshot")
        actual = {**actual, "sections": sorted(actual["sections"], key=lambda section: section["id"])}
        expected = {**expected, "sections": sorted(expected["sections"], key=lambda section: section["id"])}
    # Sorted case JSONL does not retain JSON-LD dictionary insertion order.
    # Compare source structure with the rebuilt view; hash the original message
    # bytes against its recorded pre/post section-shuffle identity separately.
    source_hash = (
        case.get("citation_curriculum", {}).get("training_source_hash")
        if shuffled
        else case.get("normalized_source_hash")
    )
    if actual != expected or source_hash != digest(message_source):
        raise ValueError("Replay source view differs from its immutable snapshot")


def _load(directory: Path, kind: str) -> tuple[dict, dict, str]:
    config = INPUTS[kind]
    manifest_body = _read(directory / "manifest.json", 1_000_000)
    manifest = _parse(manifest_body.decode("utf-8"))
    if (
        not isinstance(manifest, dict)
        or any(manifest.get(key) != config[key] for key in ("generator", "seed", "per_schema"))
        or manifest.get("target_order") != "schema"
        or manifest.get("dataset_fingerprint") != config["fingerprint"]
        or digest(manifest.get("files")) != config["fingerprint"]
    ):
        raise ValueError("Replay requires the exact measured schema-ordered input identity")
    rows, cases = {}, {}
    for split in SPLITS:
        for suffix, destination in (("jsonl", rows), ("cases.jsonl", cases)):
            name = f"{split}.{suffix}"
            body = _read(directory / name, MAX_FILE_BYTES)
            content = _jsonl(body)
            expected_count = config["per_schema"][split] * len(SCHEMAS)
            if len(content) != expected_count or manifest["files"].get(name) != {
                "sha256": hashlib.sha256(body).hexdigest(),
                "rows": expected_count,
            }:
                raise ValueError("Replay source file hash or row count differs")
            destination[split] = content
        expected_ids = {
            f"{'citation-v3-' if kind == 'v3' else ''}{split}-{schema}-{number:05d}"
            for schema in SCHEMAS
            for number in range(config["per_schema"][split])
        }
        if (
            {case.get("case_id") for case in cases[split]} != expected_ids
            or any(case.get("split") != split for case in cases[split])
            or Counter(case.get("schema_id") for case in cases[split])
            != Counter(dict.fromkeys(SCHEMAS, config["per_schema"][split]))
        ):
            raise ValueError("Replay source split or schema membership differs")
        for row, case in zip(rows[split], cases[split], strict=True):
            shuffled = bool(case.get("citation_curriculum", {}).get("train_section_presentation_shuffled"))
            _validate_pair(row, case, shuffled=shuffled)
    check_disjoint(cases)
    return rows, cases, hashlib.sha256(manifest_body).hexdigest()


def build_replay(original_dir: str | Path, v3_dir: str | Path, output_dir: str | Path) -> dict:
    original, v3, output = (
        Path(value).expanduser().resolve() for value in (original_dir, v3_dir, output_dir)
    )
    repository = Path(__file__).resolve().parents[1]
    if output.is_relative_to(repository):
        raise ValueError("Generated data must be outside the public source checkout")
    paths = (original, v3, output)
    if (
        any(left.is_relative_to(right) for left in paths for right in paths if left != right)
        or len(set(paths)) != 3
    ):
        raise ValueError("Replay input and output directories must not overlap")
    if output.exists():
        raise ValueError("Replay output already exists; choose a new directory")
    old_rows, old_cases, old_manifest = _load(original, "original")
    new_rows, new_cases, new_manifest = _load(v3, "v3")
    pairs = [
        *zip(old_rows["train"], old_cases["train"], strict=True),
        *zip(new_rows["train"], new_cases["train"], strict=True),
    ]
    random.Random(SEED).shuffle(pairs)
    selected = [
        (row, case)
        for row, case in zip(new_rows["valid"], new_cases["valid"], strict=True)
        if int(case["case_id"].rsplit("-", 1)[1]) in VALIDATION_INDICES
    ]
    rows = {
        "train": [pair[0] for pair in pairs],
        "valid": [pair[0] for pair in selected],
        "test": new_rows["test"],
    }
    cases = {
        "train": [pair[1] for pair in pairs],
        "valid": [pair[1] for pair in selected],
        "test": new_cases["test"],
    }
    checks = check_disjoint(cases)
    payloads, files = {}, {}
    for split in SPLITS:
        for suffix, content in (("jsonl", rows[split]), ("cases.jsonl", cases[split])):
            name = f"{split}.{suffix}"
            payloads[name] = "".join(canonical(item) + "\n" for item in content)
            files[name] = {
                "sha256": hashlib.sha256(payloads[name].encode()).hexdigest(),
                "rows": len(content),
            }
    if digest(files) != EXPECTED_FINGERPRINT:
        raise ValueError("Replay output differs from the measured recipe")
    manifest = {
        "generator": VERSION,
        "seed": SEED,
        "recipe": {
            "original_train_only": 720,
            "new_balanced_train": 1080,
            "validation_indices_per_schema": sorted(VALIDATION_INDICES),
            "fresh_test_evaluation_performed": False,
        },
        "source_manifests": {"original": old_manifest, "v3": new_manifest},
        "counts": {split: len(rows[split]) for split in SPLITS},
        "counts_by_schema": {
            split: dict(Counter(case["schema_id"] for case in cases[split])) for split in SPLITS
        },
        "quality_checks": checks,
        "files": files,
        "dataset_fingerprint": digest(files),
        "provenance": "Authored synthetic data; old training replay only; no validation/test/model-output rows in training",
        "blend_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    output.mkdir(mode=0o700, parents=True)
    for name, payload in payloads.items():
        _write(output / name, payload)
    _write(output / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--v3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    result = build_replay(**vars(parser.parse_args()))
    print(json.dumps({"counts": result["counts"], "fingerprint": result["dataset_fingerprint"]}))


if __name__ == "__main__":
    main()
