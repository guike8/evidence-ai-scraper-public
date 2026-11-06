"""Private-page compiler mechanics use authored fixtures, never the web corpus."""

from __future__ import annotations

import copy
import json

import pytest

from ai_scraper.extraction.grounded import build_grounding_task
from ai_scraper.extraction.grounded_view import build_grounded_view
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest
from training.realpage_dataset import (
    compile_dataset,
    file_hash,
    make_training_row,
    validate_training_row,
)


def fixture(domain="train.example", token="Amber"):
    request = ExtractionRequest(
        url=f"https://{domain}/about",
        schema_id="custom_v1",
        instruction="Extract the project name, founding year and fax number.",
        fields=[
            {
                "name": "project_name",
                "type": "string",
                "description": "Project's exact name",
                "critical": True,
            },
            {"name": "founded_year", "type": "integer", "description": "Explicit founding year"},
            {"name": "fax_number", "type": "string", "description": "Public fax number", "required": False},
        ],
    )
    snapshot = normalize_record(
        {
            "final_url": request.url,
            "markdown": f"# {token} Project\nThe {token} Project was founded in 2001.\nContact the project online.",
            "page_html": f"<h1>{token} Project</h1><p>The {token} Project was founded in 2001.</p>",
            "source_preparation": {"kind": "firecrawl_preparation_component_replay"},
        },
        request,
        retrieval_method="firecrawl_preparation_replay",
    )
    spec = get_schema(request)
    source = json.loads(build_grounded_view(snapshot, request, spec, 8000).content)
    data = {"project_name": f"{token} Project", "founded_year": 2001, "fax_number": None}
    evidence = []
    for path, value in (("project_name", f"{token} Project"), ("founded_year", "2001")):
        section = next(
            s
            for s in snapshot.sections
            if s.representation == "markdown" and "founded" in s.text and value in s.text
        )
        evidence.append(
            {
                "field_path": path,
                "section_id": section.id,
                "excerpt": value,
                "semantic_context": "The sentence explicitly binds the project name to its founding year.",
            }
        )
    return request, spec, snapshot, source, data, evidence


def test_both_tasks_reconstruct_production_prompts_and_exact_schema():
    request, spec, _, source, data, _ = fixture()
    business = make_training_row(request, source, "business", data)
    assert validate_training_row(business) == "business"
    task = build_grounding_task(spec, data, source, request.instruction)
    selection = {"selections": {path: values[0].id for path, values in task.candidates.items()}}
    grounding = make_training_row(request, source, "grounding", selection, data=data)
    assert validate_training_row(grounding) == "grounding"
    assert len(grounding["messages"]) == 3


@pytest.mark.parametrize(
    "mutation",
    ["prompt", "schema", "provenance", "missing_key", "integer_string", "extra_metadata", "tools", "source"],
)
def test_business_row_rejects_changed_prompts_schema_types_and_application_targets(mutation):
    request, _, _, source, data, _ = fixture()
    row = make_training_row(request, source, "business", data)
    if mutation == "prompt":
        row["messages"][0]["content"] += " Changed prompt"
    elif mutation == "schema":
        row["grounded"]["schema_sha256"] = "0" * 64
    elif mutation == "provenance":
        row["messages"][-1]["content"] = json.dumps({**data, "provenance": {"model": "invented"}})
    elif mutation == "missing_key":
        row["messages"][-1]["content"] = json.dumps({"project_name": data["project_name"]})
    elif mutation == "integer_string":
        row["messages"][-1]["content"] = json.dumps({**data, "founded_year": "2001"})
    elif mutation == "extra_metadata":
        row["grounded"]["target_schema"] = {"type": "object"}
    elif mutation == "tools":
        row["tools"] = []
    else:
        row["grounded"]["source"]["sections"][0]["text"] = "Changed source"
    with pytest.raises((ValueError, TypeError)):
        validate_training_row(row)


def test_grounding_targets_cannot_invent_cross_field_ids_or_application_keys():
    request, spec, _, source, data, _ = fixture()
    task = build_grounding_task(spec, data, source, request.instruction)
    selected = {path: values[0].id for path, values in task.candidates.items()}
    row = make_training_row(request, source, "grounding", {"selections": selected}, data=data)
    for bad in (
        {"selections": {**selected, "founded_year": selected["project_name"]}},
        {"selections": {**selected, "founded_year": "c9999"}},
        {"selections": {}, "status": "success"},
    ):
        changed = copy.deepcopy(row)
        changed["messages"][-1]["content"] = json.dumps(bad)
        with pytest.raises(ValueError):
            validate_training_row(changed)
    selected["founded_year"] = None
    assert (
        validate_training_row(
            make_training_row(request, source, "grounding", {"selections": selected}, data=data)
        )
        == "grounding"
    )


def corpus(tmp_path):
    pages = []
    snapshots = tmp_path / "snapshots"
    for split, domain, token in [
        ("train", "train.example", "Amber"),
        ("validation", "valid.example", "Birch"),
        ("site_holdout", "test.example", "Cedar"),
    ]:
        request, _, snapshot, _, data, evidence = fixture(domain, token)
        directory = snapshots / token.lower()
        directory.mkdir(parents=True)
        path = directory / "snapshot.json"
        path.write_text(snapshot.model_dump_json())
        pages.append(
            {
                "page_id": token.lower(),
                "split": split,
                "site_family": domain,
                "prepared_source_hash": snapshot.source_hash,
                "prepared_snapshot_sha256": file_hash(path),
                "tasks": [
                    {
                        "task_id": "facts",
                        "request": request.model_dump(mode="json"),
                        "expected_data": data,
                        "evidence": evidence,
                    }
                ],
            }
        )
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps({"pages": pages}))
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps(
            {
                "status": "approved",
                "labels_sha256": file_hash(labels),
                "model_outputs_seen": False,
                "unresolved": [],
            }
        )
    )
    return labels, snapshots, review


def test_compiler_binds_review_sources_and_keeps_site_holdout_out_of_train(tmp_path):
    labels, snapshots, review = corpus(tmp_path)
    manifest = compile_dataset(labels, snapshots, tmp_path / "out", review_receipt=review)
    assert manifest["case_counts"] == {"train": 1, "valid": 1, "test": 1}
    assert manifest["row_counts"] == {"train": 2, "valid": 2, "test": 2}
    assert manifest["target_format"] == "realpage_grounded_v1"
    train = (tmp_path / "out/train.jsonl").read_text()
    assert "Amber" in train and "Birch" not in train and "Cedar" not in train
    assert manifest["independent_review_sha256"] == file_hash(review)
    for split in ("train", "valid", "test"):
        for line in (tmp_path / f"out/{split}.jsonl").read_text().splitlines():
            validate_training_row(json.loads(line))
    with pytest.raises(ValueError, match="new private"):
        compile_dataset(labels, snapshots, tmp_path / "out", review_receipt=review)


def test_compiler_rejects_review_of_different_labels_before_writing(tmp_path):
    labels, snapshots, review = corpus(tmp_path)
    labels.write_text(labels.read_text() + "\n")
    with pytest.raises(ValueError, match="exact labels"):
        compile_dataset(labels, snapshots, tmp_path / "out", review_receipt=review)
    assert not (tmp_path / "out").exists()


def test_compiler_rejects_site_overlap_even_if_snapshots_differ(tmp_path):
    labels, snapshots, review = corpus(tmp_path)
    data = json.loads(labels.read_text())
    data["pages"][1]["site_family"] = "train.example"
    labels.write_text(json.dumps(data))
    rec = json.loads(review.read_text())
    rec["labels_sha256"] = file_hash(labels)
    review.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="site family|Site families overlap"):
        compile_dataset(labels, snapshots, tmp_path / "out", review_receipt=review)
