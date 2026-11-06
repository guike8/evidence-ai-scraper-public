"""Reproduce the measured replay and reject changed sources or split leakage."""

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path

import pytest

from training import citation_replay as replay
from training.citation_dataset_v3 import build_case
from training.citation_dataset_v3 import generate_dataset as generate_v3
from training.dataset import generate_dataset as generate_original
from training.evaluate import read_jsonl


@pytest.fixture(scope="module")
def authored_inputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("citation-replay")
    original, v3 = root / "original", root / "v3"
    first = generate_original(original, target_order="schema")
    second = generate_v3(v3, train_per_schema=120)
    assert first["dataset_fingerprint"] == replay.INPUTS["original"]["fingerprint"]
    assert second["dataset_fingerprint"] == replay.INPUTS["v3"]["fingerprint"]
    return original, v3


def test_measured_replay_is_byte_exact_private_and_train_only(authored_inputs, tmp_path):
    original, v3 = authored_inputs
    output = tmp_path / "mixed"
    result = replay.build_replay(original, v3, output)
    assert result["counts"] == {"train": 1800, "valid": 72, "test": 108}
    assert result["dataset_fingerprint"] == replay.EXPECTED_FINGERPRINT
    assert all(result["quality_checks"].values())
    assert all(
        counts == dict.fromkeys(replay.SCHEMAS, 200 if split == "train" else 8 if split == "valid" else 12)
        for split, counts in result["counts_by_schema"].items()
    )
    for name, metadata in result["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == metadata["sha256"]
        assert (output / name).stat().st_mode & 0o777 == 0o600
    assert output.stat().st_mode & 0o777 == 0o700
    assert (output / "test.jsonl").read_bytes() == (v3 / "test.jsonl").read_bytes()
    assert (output / "test.cases.jsonl").read_bytes() == (v3 / "test.cases.jsonl").read_bytes()
    train = read_jsonl(output / "train.cases.jsonl")
    assert all(case["split"] == "train" for case in train)
    assert sum(case["case_id"].startswith("citation-v3-") for case in train) == 1080
    valid = read_jsonl(output / "valid.cases.jsonl")
    assert {int(case["case_id"].rsplit("-", 1)[1]) for case in valid} == replay.VALIDATION_INDICES
    again = tmp_path / "again"
    assert replay.build_replay(original, v3, again) == result
    assert all((again / name).read_bytes() == (output / name).read_bytes() for name in result["files"])


def test_refuses_overwrite_overlap_and_checkout_output(authored_inputs, tmp_path):
    original, v3 = authored_inputs
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "keep"
    sentinel.write_text("untouched")
    with pytest.raises(ValueError, match="already exists"):
        replay.build_replay(original, v3, occupied)
    assert sentinel.read_text() == "untouched"
    with pytest.raises(ValueError, match="must not overlap"):
        replay.build_replay(original, original, tmp_path / "new")
    with pytest.raises(ValueError, match="must not overlap"):
        replay.build_replay(original, v3, original / "new")
    with pytest.raises(ValueError, match="outside the public source checkout"):
        replay.build_replay(original, v3, Path(__file__).resolve().parents[1] / "private-data")


@pytest.mark.parametrize("change", ["seed", "target_order", "dataset_fingerprint", "files"])
def test_rejects_relabeling_input_identity(authored_inputs, tmp_path, change):
    original, _ = authored_inputs
    changed = tmp_path / "changed"
    changed.mkdir()
    manifest = json.loads((original / "manifest.json").read_text())
    if change == "seed":
        manifest[change] = 42
    elif change == "target_order":
        manifest[change] = "canonical"
    elif change == "dataset_fingerprint":
        manifest[change] = "0" * 64
    else:
        manifest["files"]["train.jsonl"]["rows"] = 719
    (changed / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="exact measured"):
        replay._load(changed, "original")


def test_source_file_tampering_fails_before_output(authored_inputs, tmp_path):
    original, v3 = authored_inputs
    changed = tmp_path / "changed"
    shutil.copytree(original, changed)
    target = changed / "train.jsonl"
    # Equal row count and valid JSON are insufficient: recorded bytes must match.
    target.write_text(target.read_text().replace("UNTRUSTED_SOURCE_JSON", "ALTERED_SOURCE_JSON", 1))
    output = tmp_path / "mixed"
    with pytest.raises(ValueError, match="file hash or row count"):
        replay.build_replay(changed, v3, output)
    assert not output.exists()


def test_production_validation_rejects_invented_gold_quote():
    row, case = build_case("train", "article_v1", 0)
    case["expected_candidate"]["evidence"][0]["excerpt"] = "This quote never appeared in the source."
    row["messages"][-1]["content"] = json.dumps(case["expected_candidate"])
    with pytest.raises(ValueError, match="Gold labels fail production source validation"):
        replay._validate_pair(row, case, shuffled=False)


def test_recorded_training_target_and_source_cannot_be_replaced():
    row, case = build_case("valid", "article_v1", 0)
    changed = copy.deepcopy(row)
    changed["messages"][-1]["content"] = json.dumps({"data": {}, "evidence": [], "warnings": []})
    with pytest.raises(ValueError, match="authored schema-ordered target"):
        replay._validate_pair(changed, case, shuffled=False)
    changed = copy.deepcopy(row)
    source = json.loads(changed["messages"][1]["content"].split("\n", 1)[1])
    source["sections"][0]["text"] = "Unrelated source"
    changed["messages"][1]["content"] = "UNTRUSTED_SOURCE_JSON\n" + json.dumps(source)
    with pytest.raises(ValueError, match="immutable snapshot"):
        replay._validate_pair(changed, case, shuffled=False)


def test_combined_mix_checks_original_training_against_new_validation(tmp_path, monkeypatch):
    # Input checks alone cannot establish disjointness across two datasets.
    # Isolate that additional guard with small, individually valid sources.
    from training.dataset import build_case as original_case

    old = {split: [original_case(split, "page_v1", 0)] for split in replay.SPLITS}
    new = {split: [build_case(split, "page_v1", 0)] for split in replay.SPLITS}
    old["train"][0][1]["domain"] = new["valid"][0][1]["domain"]

    def loaded(_, kind):
        pairs = old if kind == "original" else new
        return (
            {split: [pair[0] for pair in values] for split, values in pairs.items()},
            {split: [pair[1] for pair in values] for split, values in pairs.items()},
            "0" * 64,
        )

    monkeypatch.setattr(replay, "_load", loaded)
    output = tmp_path / "mixed"
    with pytest.raises(ValueError, match="Split leakage: domain overlaps train/valid"):
        replay.build_replay(tmp_path / "old", tmp_path / "new", output)
    assert not output.exists()


def test_read_limits_duplicate_keys_and_symlinks_fail(tmp_path, monkeypatch):
    path = tmp_path / "large"
    path.write_bytes(b"123456")
    with pytest.raises(ValueError, match="bounded regular file"):
        replay._read(path, 5)
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(OSError):
        replay._read(link, 100)
    with pytest.raises(ValueError, match="Duplicate JSON keys"):
        replay._jsonl(b'{"same":1,"same":2}\n')
    monkeypatch.setattr(replay, "MAX_ROWS", 1)
    with pytest.raises(ValueError, match="row bound"):
        replay._jsonl(b'{"row":1}\n{"row":2}\n')


def test_fifo_input_fails_without_waiting_for_a_writer(tmp_path):
    fifo = tmp_path / "fifo"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="bounded regular file"):
        replay._read(fifo, 100)
