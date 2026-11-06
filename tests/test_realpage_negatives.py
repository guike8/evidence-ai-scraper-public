"""Counterfactual targets reject a real navigation value, never relabel it gold."""

from __future__ import annotations

import hashlib
import json

import pytest

from ai_scraper.extraction.grounded import build_grounding_task
from ai_scraper.extraction.grounded_view import build_grounded_view
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest, SourceSnapshot
from training import evaluate
from training.realpage_dataset import (
    NEGATIVE_FORMAT,
    _training_visible_data,
    _visible_evidence,
    canonical,
    compile_dataset,
    digest,
    file_hash,
    validate_training_row,
)


@pytest.fixture
def reviewed(tmp_path):
    pages, snapshots = [], tmp_path / "snapshots"
    for split, token, domain in (
        ("train", "Amber", "train.example"),
        ("validation", "Birch", "valid.example"),
        ("site_holdout", "Cedar", "test.example"),
    ):
        request = ExtractionRequest(
            url=f"https://{domain}/about",
            schema_id="custom_v1",
            instruction="Extract the featured project main heading and its founding year.",
            fields=[
                {
                    "name": "project_name",
                    "description": "The featured project main heading",
                    "critical": True,
                },
                {"name": "founded_year", "type": "integer", "description": "Its explicit founding year"},
            ],
        )
        snapshot = normalize_record(
            {
                "final_url": request.url,
                "page_html": f"<nav>Navigation Index</nav><h1>{token} Project</h1><p>Founded in 2001.</p>",
                "markdown": f"Navigation Index\n# {token} Project\nFounded in 2001.",
                "source_preparation": {"kind": "firecrawl_preparation_component_replay"},
            },
            request,
            retrieval_method="firecrawl_preparation_replay",
        )
        folder = snapshots / token.lower()
        folder.mkdir(parents=True)
        path = folder / "snapshot.json"
        path.write_text(snapshot.model_dump_json())
        evidence = []
        for field, literal in (("project_name", f"{token} Project"), ("founded_year", "2001")):
            section = next(
                s for s in snapshot.sections if s.representation == "markdown" and literal in s.text
            )
            evidence.append(
                {
                    "field_path": field,
                    "section_id": section.id,
                    "excerpt": literal,
                    "semantic_context": "Featured project heading or its explicit founding statement.",
                }
            )
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
                        "expected_data": {"project_name": f"{token} Project", "founded_year": 2001},
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
    base = tmp_path / "base"
    compile_dataset(labels, snapshots, base, review_receipt=review)
    case = evaluate.read_jsonl(base / "train.cases.jsonl")[0]
    request = ExtractionRequest.model_validate(case["request"])
    spec = get_schema(request)
    snapshot = SourceSnapshot.model_validate_json(canonical(case["snapshot"]))
    view = build_grounded_view(snapshot, request, spec, 8000)
    source = json.loads(view.content)
    visible, omitted = _training_visible_data(
        case["expected_candidate"]["data"], case["expected_candidate"]["evidence"], source
    )
    predicted = {**visible, "project_name": "Navigation Index"}
    task = build_grounding_task(spec, predicted, source, request.instruction, max_source_chars=8000)
    candidate = task.candidates["project_name"][0]
    anchor = next(e for e in case["expected_candidate"]["evidence"] if e["field_path"] == "founded_year")
    correct = next(c for c in task.candidates["founded_year"] if c.section_id == anchor["section_id"])
    example = {
        "example_id": "negative-project-heading",
        "case_id": case["case_id"],
        "page_id": "amber",
        "split": "train",
        "site_family": "train.example",
        "data_overrides": {"project_name": "Navigation Index"},
        "wrong_paths": ["project_name"],
        "original_gold_value": visible["project_name"],
        "visible_business_data_before_override": visible,
        "counterfactual_business_data": predicted,
        "omitted_gold_paths_in_original_view": omitted,
        "wrong_candidate_count": 1,
        "rejected_contexts": [
            {
                "field_path": "project_name",
                "candidate_id": candidate.id,
                "section_id": candidate.section_id,
                "json_pointer": candidate.json_pointer,
                "excerpt": candidate.excerpt,
                "context": json.loads(candidate.context),
                "reason": "This is navigation, not the featured project heading.",
            }
        ],
        "correct_references_preserved": [
            {
                "field_path": "founded_year",
                "candidate_id": correct.id,
                "section_id": correct.section_id,
                "excerpt": anchor["excerpt"],
            }
        ],
        "expected_selections": {"selections": {"project_name": None, "founded_year": correct.id}},
        "source_view_sha256": hashlib.sha256(view.content.encode()).hexdigest(),
        "source_view_descriptor": view.as_metadata(),
        "grounding_task_sha256": digest(
            {"schema": task.schema, "messages": task.messages, "metadata": task.metadata}
        ),
        "grounding_catalog_sha256": task.metadata["catalog_hash"],
        "request_sha256": digest(request.model_dump(mode="json")),
        "source_hash": snapshot.source_hash,
        "semantic_rejection_reason": "The navigation label is a different semantic role than the featured project name.",
        "annotation_status": "independent_semantic_review_required",
    }
    payload = {
        "format": NEGATIVE_FORMAT,
        "approved_business_labels_sha256": file_hash(labels),
        "base_train_cases_sha256": file_hash(base / "train.cases.jsonl"),
        "source_limit_chars": 8000,
        "curriculum_role": "Derived TRAIN-only grounding supervision",
        "request_policy": "Original request unchanged",
        "target_policy": "Abstain on navigation value",
        "examples": [example],
        "model_outputs_consulted": False,
        "model_calls": 0,
        "training_calls": 0,
        "approved_business_labels_modified": False,
        "raw_redistribution": False,
        "failures": [],
    }
    negatives = tmp_path / "negatives.json"
    negative_review = tmp_path / "negative-review.json"

    def write():
        negatives.write_text(json.dumps(payload))
        negative_review.write_text(
            json.dumps(
                {
                    "status": "approved",
                    "negatives_sha256": file_hash(negatives),
                    "model_outputs_seen": False,
                    "unresolved": [],
                }
            )
        )

    write()
    return labels, snapshots, review, negatives, negative_review, payload, base, write


def compile_augmented(values, output):
    labels, snapshots, review, negatives, negative_review, *_ = values
    return compile_dataset(
        labels,
        snapshots,
        output,
        review_receipt=review,
        grounding_negatives=negatives,
        negative_review_receipt=negative_review,
    )


def test_augments_only_train_grounding_preserving_business_case_bytes(reviewed, tmp_path):
    output = tmp_path / "augmented"
    manifest = compile_augmented(reviewed, output)
    assert manifest["case_counts"] == {"train": 1, "valid": 1, "test": 1}
    assert manifest["row_counts"] == {"train": 3, "valid": 2, "test": 2}
    assert manifest["grounding_negatives"]["count"] == 1
    assert manifest["grounding_negatives"]["negatives_sha256"] == file_hash(reviewed[3])
    assert manifest["grounding_negatives"]["independent_review_sha256"] == file_hash(reviewed[4])
    for split in ("train", "valid", "test"):
        assert (output / f"{split}.cases.jsonl").read_bytes() == (
            reviewed[6] / f"{split}.cases.jsonl"
        ).read_bytes()
    for split in ("valid", "test"):
        assert (output / f"{split}.jsonl").read_bytes() == (reviewed[6] / f"{split}.jsonl").read_bytes()
    rows = evaluate.read_jsonl(output / "train.jsonl")
    negative = next(
        row for row in rows if row["grounded"].get("data", {}).get("project_name") == "Navigation Index"
    )
    assert validate_training_row(negative) == "grounding"
    assert json.loads(negative["messages"][-1]["content"])["selections"]["project_name"] is None
    base_manifest = json.loads((reviewed[6] / "manifest.json").read_text())
    assert base_manifest["dataset_fingerprint"] != manifest["dataset_fingerprint"]


@pytest.mark.parametrize(
    "mutation",
    [
        "nontrain",
        "case_hash",
        "business_hash",
        "nonnull",
        "candidate",
        "schema",
        "same_value",
        "wrong_type",
        "context",
        "reason",
        "correct_ref",
    ],
)
def test_rejects_unreviewable_or_forged_negative_targets_before_writing(reviewed, tmp_path, mutation):
    payload = reviewed[5]
    example = payload["examples"][0]
    if mutation == "nontrain":
        example["case_id"], example["split"] = "birch-facts", "valid"
    elif mutation == "case_hash":
        payload["base_train_cases_sha256"] = "0" * 64
    elif mutation == "business_hash":
        payload["approved_business_labels_sha256"] = "0" * 64
    elif mutation == "nonnull":
        example["expected_selections"]["selections"]["project_name"] = example["rejected_contexts"][0][
            "candidate_id"
        ]
    elif mutation == "candidate":
        example["rejected_contexts"][0]["candidate_id"] = "c9999"
    elif mutation == "schema":
        example["grounding_schema"] = {"type": "object"}
    elif mutation == "same_value":
        example["data_overrides"]["project_name"] = example["original_gold_value"]
    elif mutation == "wrong_type":
        example["data_overrides"]["project_name"] = 42
    elif mutation == "context":
        example["rejected_contexts"][0]["section_id"] = "s9999"
    elif mutation == "reason":
        example["rejected_contexts"][0]["reason"] = " "
    elif mutation == "correct_ref":
        example["correct_references_preserved"][0]["candidate_id"] = "c9999"
    reviewed[7]()
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        compile_augmented(reviewed, output)
    assert not output.exists()


def test_both_negative_paths_and_exact_review_are_required(reviewed, tmp_path):
    labels, snapshots, review, negatives, _negative_review, *_ = reviewed
    with pytest.raises(ValueError, match="together"):
        compile_dataset(
            labels,
            snapshots,
            tmp_path / "only-negative",
            review_receipt=review,
            grounding_negatives=negatives,
        )
    negatives.write_text(negatives.read_text() + "\n")
    with pytest.raises(ValueError, match="exact bytes"):
        compile_augmented(reviewed, tmp_path / "changed-after-review")


def test_visible_list_reindex_uses_original_indices_for_repeated_values():
    data = {"names": ["Amber", "Amber"]}
    evidence = [
        {"field_path": "names.0", "section_id": "s0001", "excerpt": "Amber"},
        {"field_path": "names.1", "section_id": "s0002", "excerpt": "Amber"},
    ]
    result = _visible_evidence(data, evidence, {"sections": [{"id": "s0002", "text": "Amber"}]})
    assert result == [{"field_path": "names.0", "section_id": "s0002", "excerpt": "Amber"}]
