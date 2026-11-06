"""Authored-fixture checks for the private TRAIN-only schema augmentation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ai_scraper.extraction.generic import populated_fields
from ai_scraper.extraction.grounded import build_grounding_task
from ai_scraper.extraction.grounded_view import build_grounded_view
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest
from training import evaluate
from training import realpage_augment as augmentation
from training.realpage_dataset import (
    _training_visible_data,
    _visible_evidence,
    compile_dataset,
    digest,
    file_hash,
    make_training_row,
    validate_training_row,
)


@pytest.fixture
def base(tmp_path):
    snapshots, pages = tmp_path / "snapshots", []
    for split, domain, token in (
        ("train", "train.example", "Amber"),
        ("validation", "valid.example", "Birch"),
        ("site_holdout", "test.example", "Cedar"),
    ):
        request = ExtractionRequest(
            url=f"https://{domain}/about",
            schema_id="custom_v1",
            instruction="Extract project identity, founding year, hidden note and fax number.",
            fields=[
                {
                    "name": "project_name",
                    "description": "Main featured project heading",
                    "critical": True,
                },
                {
                    "name": "founded_year",
                    "type": "integer",
                    "description": "The explicit founding year",
                },
                {
                    "name": "hidden_note",
                    "description": "Exact note stated in the long body",
                },
                {
                    "name": "fax_number",
                    "description": "An explicit support fax number",
                    "required": False,
                },
            ],
        )
        long_note = "Note: long-body-marker. " + "Padding for this large indivisible paragraph. " * 300
        snapshot = normalize_record(
            {
                "final_url": request.url,
                "markdown": f"Navigation Index\n# {token} Project\nFounded in 2001.\n{long_note}",
                "page_html": f"<nav>Navigation Index</nav><h1>{token} Project</h1><p>Founded in 2001.</p><p>{long_note}</p>",
                "source_preparation": {"kind": "firecrawl_preparation_component_replay"},
            },
            request,
            retrieval_method="firecrawl_preparation_replay",
        )
        folder = snapshots / token.lower()
        folder.mkdir(parents=True)
        snapshot_path = folder / "snapshot.json"
        snapshot_path.write_text(snapshot.model_dump_json())
        evidence = []
        for field, literal in (
            ("project_name", f"{token} Project"),
            ("founded_year", "2001"),
            ("hidden_note", "long-body-marker"),
        ):
            section = next(
                s for s in snapshot.sections if s.representation == "markdown" and literal in s.text
            )
            evidence.append(
                {
                    "field_path": field,
                    "section_id": section.id,
                    "excerpt": literal,
                    "semantic_context": "Authored fixture explicitly labels this fact's semantic role.",
                }
            )
        pages.append(
            {
                "page_id": token.lower(),
                "split": split,
                "site_family": domain,
                "prepared_source_hash": snapshot.source_hash,
                "prepared_snapshot_sha256": file_hash(snapshot_path),
                "tasks": [
                    {
                        "task_id": "facts",
                        "request": request.model_dump(mode="json"),
                        "expected_data": {
                            "project_name": f"{token} Project",
                            "founded_year": 2001,
                            "hidden_note": "long-body-marker",
                            "fax_number": None,
                        },
                        "evidence": evidence,
                    }
                ],
            }
        )
    labels, review = tmp_path / "labels.json", tmp_path / "review.json"
    labels.write_text(json.dumps({"pages": pages}))
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
    directory = tmp_path / "base"
    compile_dataset(labels, snapshots, directory, review_receipt=review)
    return directory


def test_projection_and_alias_preserve_facts_and_reserved_bytes_without_parsing_them(
    base, tmp_path, monkeypatch
):
    reads = []
    original = evaluate.read_jsonl

    def train_only(path):
        reads.append(Path(path).name)
        assert Path(path).name in {"train.jsonl", "train.cases.jsonl"}
        return original(path)

    monkeypatch.setattr(evaluate, "read_jsonl", train_only)
    output = tmp_path / "derived"
    result = augmentation.augment_business(base, output)
    assert reads == ["train.cases.jsonl", "train.jsonl"]
    assert result["row_counts"] == {"train": 7, "valid": 2, "test": 2}
    assert result["derived_business_transformations"]["added_single_field_projection"] == 4
    assert result["derived_business_transformations"]["added_full_schema_alias"] == 1
    assert result["validation_and_holdout_content_parsed"] is False
    for name in augmentation.FILES - {"train.jsonl"}:
        assert (output / name).read_bytes() == (base / name).read_bytes()
    rows = original(output / "train.jsonl")
    for row in rows:
        validate_training_row(row)
        assert row["grounded"]["request"]["url"].startswith("https://train.example/")
        if row["grounded"]["task"] == "business":
            target = json.loads(row["messages"][-1]["content"])
            assert target.get("hidden_note") is None
            assert target.get("requested_hidden_note") is None
            assert "Birch Project" not in augmentation.canonical(row)
            assert "Cedar Project" not in augmentation.canonical(row)
    alias = next(
        row
        for row in rows
        if row["grounded"]["task"] == "business"
        and any(field["name"].startswith("requested_") for field in row["grounded"]["request"]["fields"])
    )
    target = json.loads(alias["messages"][-1]["content"])
    assert target == {
        "requested_project_name": "Amber Project",
        "requested_founded_year": 2001,
        "requested_hidden_note": None,
        "requested_fax_number": None,
    }
    assert type(target["requested_founded_year"]) is int
    assert result["original_TRAIN_cases"] == result["original_TRAIN_pages"] == 1
    assert result["original_TRAIN_positive_facts"] == 3
    assert result["source_workflow_hashes"]["realpage_augment.py"] == file_hash(Path(augmentation.__file__))


def test_supplemental_grounding_deduplicates_existing_rows_by_canonical_hash(base, tmp_path, monkeypatch):
    original_rows = evaluate.read_jsonl(base / "train.jsonl")
    existing = next(row for row in original_rows if row["grounded"]["task"] == "grounding")
    request = ExtractionRequest.model_validate(existing["grounded"]["request"])
    schema = get_schema(request)
    wrong_data = {**existing["grounded"]["data"], "project_name": "Navigation Index"}
    task = build_grounding_task(schema, wrong_data, existing["grounded"]["source"], request.instruction)
    selected = {
        path: None if path == "project_name" else candidates[0].id
        for path, candidates in task.candidates.items()
    }
    new = make_training_row(
        request,
        existing["grounded"]["source"],
        "grounding",
        {"selections": selected},
        data=wrong_data,
    )

    def supplement(path, review, cases, labels_hash, bound):
        assert len(cases) == 1 and cases[0]["split"] == "train"
        assert bound == 8000 and len(labels_hash) == 64
        return (
            [copy.deepcopy(existing), new],
            [],
            {"count": 2, "negatives_sha256": "a" * 64},
        )

    monkeypatch.setattr(augmentation, "_negative_rows", supplement)
    result = augmentation.augment_business(
        base,
        tmp_path / "derived",
        grounding_negatives=tmp_path / "negatives.json",
        negative_review_receipt=tmp_path / "negative-review.json",
    )
    assert result["row_counts"]["train"] == 8
    assert result["derived_business_transformations"]["deduplicated_supplemental_grounding_negative"] == 1
    assert result["derived_business_transformations"]["added_supplemental_grounding_negative"] == 1
    assert result["derived_business_transformations"]["task_business"] == 6
    assert result["derived_business_transformations"]["task_grounding"] == 2


def test_parent_checksums_and_train_split_identity_are_enforced_before_output(base, tmp_path):
    path = base / "train.cases.jsonl"
    values = evaluate.read_jsonl(path)
    values[0]["split"] = "test"
    path.write_text("".join(augmentation.canonical(value) + "\n" for value in values))
    with pytest.raises(ValueError, match="checksum"):
        augmentation.augment_business(base, tmp_path / "changed")
    manifest = json.loads((base / "manifest.json").read_text())
    manifest["files"][path.name] = file_hash(path)
    (base / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="original custom TRAIN"):
        augmentation.augment_business(base, tmp_path / "wrong-split")
    assert not (tmp_path / "changed").exists()
    assert not (tmp_path / "wrong-split").exists()


def test_negative_review_pair_and_separate_output_are_required(base, tmp_path):
    with pytest.raises(ValueError, match="together"):
        augmentation.augment_business(
            base,
            tmp_path / "missing-review",
            grounding_negatives=tmp_path / "negatives.json",
        )
    with pytest.raises(ValueError, match="separate"):
        augmentation.augment_business(base, base / "nested")
    with pytest.raises(ValueError, match="new private"):
        augmentation.augment_business(base, base)


def test_canonical_train_rows_remain_unique_and_reproducible(base, tmp_path):
    first = augmentation.augment_business(base, tmp_path / "first")
    second = augmentation.augment_business(base, tmp_path / "second")
    assert first["dataset_fingerprint"] == second["dataset_fingerprint"]
    rows = (tmp_path / "first/train.jsonl").read_text().splitlines()
    assert len(rows) == len(set(rows))
    for name, expected in first["files"].items():
        assert file_hash(tmp_path / "first" / name) == expected


def test_pair_grounding_is_opt_in_and_preserves_default_row_bytes(base, tmp_path):
    implicit = augmentation.augment_business(base, tmp_path / "implicit")
    explicit = augmentation.augment_business(base, tmp_path / "explicit", pair_grounding=False)
    assert implicit["dataset_fingerprint"] == explicit["dataset_fingerprint"]
    assert "paired_grounding" not in implicit
    assert (tmp_path / "implicit/train.jsonl").read_bytes() == (
        tmp_path / "explicit/train.jsonl"
    ).read_bytes()
    with pytest.raises(ValueError, match="explicit boolean"):
        augmentation.augment_business(base, tmp_path / "invalid", pair_grounding="true")
    assert not (tmp_path / "invalid").exists()


def test_pair_grounding_covers_every_business_schema_and_keeps_reserved_bytes(base, tmp_path, monkeypatch):
    original = evaluate.read_jsonl
    reads = []

    def train_only(path):
        reads.append(Path(path).name)
        assert Path(path).name in {"train.jsonl", "train.cases.jsonl"}
        return original(path)

    monkeypatch.setattr(evaluate, "read_jsonl", train_only)
    output = tmp_path / "paired"
    manifest = augmentation.augment_business(base, output, pair_grounding=True)
    assert reads == ["train.cases.jsonl", "train.jsonl"]
    assert manifest["format"] == augmentation.PAIRED_FORMAT
    assert manifest["row_counts"] == {"train": 12, "valid": 2, "test": 2}
    assert manifest["derived_business_transformations"]["task_business"] == 6
    assert manifest["derived_business_transformations"]["task_grounding"] == 6
    assert manifest["paired_grounding"]["added_rows"] == 5
    assert manifest["paired_grounding"]["new_business_gold_facts"] == 0
    assert manifest["prediction_exposure"] == augmentation.EXPOSURE
    for name in augmentation.FILES - {"train.jsonl"}:
        assert (output / name).read_bytes() == (base / name).read_bytes()
    rows = original(output / "train.jsonl")
    business = [row for row in rows if row["grounded"]["task"] == "business"]
    grounding = [row for row in rows if row["grounded"]["task"] == "grounding"]
    for row in business:
        meta = row["grounded"]
        target = json.loads(row["messages"][-1]["content"])
        matches = [
            candidate
            for candidate in grounding
            if candidate["grounded"]["request"] == meta["request"]
            and candidate["grounded"]["source"] == meta["source"]
            and candidate["grounded"]["data"] == target
        ]
        assert len(matches) == 1
        paired = matches[0]
        assert validate_training_row(paired) == "grounding"
        picks = json.loads(paired["messages"][-1]["content"])["selections"]
        assert set(picks) == set(populated_fields(target))
        assert all(identifier is not None for identifier in picks.values())
        if any(name.startswith("requested_") for name in target):
            assert all(name.startswith("requested_") for name in picks)
        if target.get("hidden_note", "not-this-projection") is None and len(target) == 1:
            assert picks == {}
    audits = json.loads((output / "derived-business-audit.json").read_text())
    paired_audits = [audit for audit in audits if audit["kind"] == "paired_positive_grounding"]
    assert len(paired_audits) == 5
    assert all(len(audit["business_row_sha256"]) == 64 for audit in paired_audits)


def _derived_pair_fixture(*, repeated_list=False):
    field = "members" if repeated_list else "project_name"
    request = ExtractionRequest(
        url="https://train.example/identity",
        schema_id="custom_v1",
        fields=[{"name": field, "type": "string_list" if repeated_list else "string"}],
    )
    if repeated_list:
        markdown = "Amber appears in an omitted paragraph. " + "Long archival material. " * 600
        markdown += "\nMember: Amber"
        data = {field: ["Amber", "Amber"]}
    else:
        markdown = "Navigation mentions Amber Project\n# Amber Project"
        data = {field: "Amber Project"}
    snapshot = normalize_record(
        {"final_url": request.url, "markdown": markdown}, request, retrieval_method="fixture_replay"
    )
    sections = [section for section in snapshot.sections if section.representation == "markdown"]
    assert len(sections) == 2
    evidence = (
        [
            {"field_path": f"{field}.{index}", "section_id": section.id, "excerpt": "Amber"}
            for index, section in enumerate(sections)
        ]
        if repeated_list
        else [{"field_path": field, "section_id": sections[1].id, "excerpt": "Amber Project"}]
    )
    case = {"case_id": "authored-pair", "page_id": "authored-page"}
    mapping = {field: "requested_" + field}
    row, audit = augmentation._derive(
        case, request, snapshot, data, evidence, "full_schema_alias", mapping, 8000
    )
    return snapshot, data, evidence, row, audit, sections


def test_positive_pair_uses_reviewed_context_instead_of_first_literal_occurrence():
    snapshot, data, evidence, row, audit, sections = _derived_pair_fixture()
    paired, _ = augmentation._paired_grounding(row, audit, snapshot, data, evidence)
    meta = paired["grounded"]
    request = ExtractionRequest.model_validate(meta["request"])
    task = build_grounding_task(get_schema(request), meta["data"], meta["source"], request.instruction)
    candidates = task.candidates["requested_project_name"]
    assert candidates[0].section_id == sections[0].id
    identifier = json.loads(paired["messages"][-1]["content"])["selections"]["requested_project_name"]
    chosen = next(candidate for candidate in candidates if candidate.id == identifier)
    assert chosen.section_id == sections[1].id
    assert chosen.id != candidates[0].id
    forged = [{**evidence[0], "excerpt": "Not present in the reviewed source"}]
    with pytest.raises(ValueError, match="Reviewed citation"):
        augmentation._paired_grounding(row, audit, snapshot, data, forged)


def test_positive_pair_reindexes_repeated_list_values_from_original_gold_indices():
    snapshot, data, evidence, row, audit, sections = _derived_pair_fixture(repeated_list=True)
    assert json.loads(row["messages"][-1]["content"]) == {"requested_members": ["Amber"]}
    paired, paired_audit = augmentation._paired_grounding(row, audit, snapshot, data, evidence)
    assert paired_audit["omitted_original_gold_paths"] == ["requested_members.0"]
    request = ExtractionRequest.model_validate(paired["grounded"]["request"])
    task = build_grounding_task(
        get_schema(request),
        paired["grounded"]["data"],
        paired["grounded"]["source"],
        request.instruction,
    )
    picks = json.loads(paired["messages"][-1]["content"])["selections"]
    assert set(picks) == {"requested_members.0"}
    chosen = next(
        candidate
        for candidate in task.candidates["requested_members.0"]
        if candidate.id == picks["requested_members.0"]
    )
    assert chosen.section_id == sections[1].id
    assert sections[0].id not in task.metadata["included_section_ids"]


def missing_curriculum(base, tmp_path, *, only_missing=False):
    cases = evaluate.read_jsonl(base / "train.cases.jsonl")
    case = copy.deepcopy(cases[0])
    if only_missing:
        request = ExtractionRequest.model_validate(case["request"])
        request.fields = [field for field in request.fields if field.name == "hidden_note"]
        schema = get_schema(request)
        data = {"hidden_note": case["expected_candidate"]["data"]["hidden_note"]}
        case.update(
            request=request.model_dump(mode="json"),
            schema_fingerprint=schema.fingerprint,
            expected_candidate={
                "data": data,
                "evidence": [
                    e for e in case["expected_candidate"]["evidence"] if e["field_path"] == "hidden_note"
                ],
                "warnings": [],
            },
            expected_fields=populated_fields(data),
            expected_status="success",
        )
        cases = [case]
    request, schema, snapshot, gold = evaluate._context(case)
    evidence = case["expected_candidate"]["evidence"]
    data = gold.data.model_dump(mode="json")
    view = build_grounded_view(snapshot, request, schema, 8000)
    source = json.loads(view.content)
    visible, omitted = _training_visible_data(data, evidence, source)
    assert visible["hidden_note"] is None and "hidden_note" in omitted
    predicted = {**visible, "hidden_note": "Navigation Index"}
    task = build_grounding_task(schema, predicted, source, request.instruction, max_source_chars=8000)
    anchors = {e["field_path"]: e for e in _visible_evidence(data, evidence, source)}
    selected, preserved = {}, []
    for path, candidates in task.candidates.items():
        if path == "hidden_note":
            selected[path] = None
            continue
        anchor = anchors[path]
        match = next(
            candidate
            for candidate in candidates
            if candidate.section_id == anchor["section_id"] and anchor["excerpt"] in candidate.excerpt
        )
        selected[path] = match.id
        preserved.append(
            {
                "field_path": path,
                "candidate_id": match.id,
                "section_id": match.section_id,
                "excerpt": anchor["excerpt"],
            }
        )
    example = {
        "example_id": "missing-context-authored-note",
        "case_id": case["case_id"],
        "page_id": case["page_id"],
        "split": "train",
        "site_family": case["domain"],
        "data_overrides": {"hidden_note": "Navigation Index"},
        "wrong_paths": ["hidden_note"],
        "original_gold_value": data["hidden_note"],
        "visible_business_data_before_override": visible,
        "counterfactual_business_data": predicted,
        "omitted_gold_paths_in_original_view": omitted,
        "wrong_candidate_count": len(task.candidates["hidden_note"]),
        "rejected_contexts": [
            {
                "field_path": "hidden_note",
                "candidate_id": candidate.id,
                "section_id": candidate.section_id,
                "json_pointer": candidate.json_pointer,
                "excerpt": candidate.excerpt,
                "context": json.loads(candidate.context),
                "reason": "Navigation is not the requested note; the reviewed note's source paragraph is out of view.",
            }
            for candidate in task.candidates["hidden_note"]
        ],
        "correct_references_preserved": preserved,
        "expected_selections": {"selections": selected},
        "source_view_sha256": view.as_metadata()["view_hash"],
        "source_view_descriptor": view.as_metadata(),
        "grounding_task_sha256": digest(
            {"schema": task.schema, "messages": task.messages, "metadata": task.metadata}
        ),
        "grounding_catalog_sha256": task.metadata["catalog_hash"],
        "request_sha256": digest(request.model_dump(mode="json")),
        "source_hash": snapshot.source_hash,
        "semantic_rejection_reason": "The only supplied value is navigation, while the full-page note's reviewed anchor was omitted.",
        "annotation_status": "independent_semantic_review_required",
    }
    parent = json.loads((base / "manifest.json").read_text())
    payload = {
        "format": augmentation.MISSING_FORMAT,
        "approved_business_labels_sha256": parent["labels_sha256"],
        "base_train_cases_sha256": augmentation.hashlib.sha256(
            "".join(augmentation.canonical(value) + "\n" for value in cases).encode()
        ).hexdigest(),
        "source_limit_chars": 8000,
        "curriculum_role": "TRAIN-only missing-context counterfactual",
        "request_policy": "Unchanged reviewed request",
        "target_policy": "Abstain on an in-view wrong-role literal",
        "examples": [example],
        "model_outputs_consulted": True,
        "model_calls": 0,
        "training_calls": 0,
        "approved_business_labels_modified": False,
        "raw_redistribution": False,
        "failures": [],
        "annotation_scope": "after_validation_before_final_site_test; TRAIN source annotations only",
        "prediction_exposure": dict(augmentation.EXPOSURE),
    }
    path, review_path = tmp_path / "missing.json", tmp_path / "missing-review.json"
    review = {
        "status": "approved",
        "model_outputs_seen": True,
        "review_protocol": "validation_informed_train_only_v1",
        **augmentation.EXPOSURE,
        "unresolved": [],
    }

    def write():
        path.write_text(json.dumps(payload))
        review["negatives_sha256"] = file_hash(path)
        review_path.write_text(json.dumps(review))

    write()
    return path, review_path, payload, cases, parent["labels_sha256"], write, review


def test_missing_context_branch_requires_reviewed_abstention_and_preserves_reserved_bytes(base, tmp_path):
    path, review, *_ = missing_curriculum(base, tmp_path)
    output = tmp_path / "with-missing"
    result = augmentation.augment_business(
        base, output, missing_context_negatives=path, missing_context_review=review
    )
    assert result["row_counts"]["train"] == 8
    assert result["missing_context_grounding_negatives"]["count"] == 1
    assert result["prediction_exposure"] == augmentation.EXPOSURE
    assert result["prediction_exposure"]["holdout_predictions_seen"] is False
    assert "model_calls_before_freeze" not in result
    assert result["builder_model_calls"] == result["builder_training_calls"] == 0
    assert result["augmentation_module_sha256"] == file_hash(Path(augmentation.__file__))
    for name in augmentation.FILES - {"train.jsonl"}:
        assert (output / name).read_bytes() == (base / name).read_bytes()


def test_missing_context_can_have_no_other_visible_populated_fields(base, tmp_path):
    path, review, _, cases, labels_hash, *_ = missing_curriculum(base, tmp_path, only_missing=True)
    rows, audit, metadata = augmentation._missing_context_rows(path, review, cases, labels_hash)
    assert metadata["count"] == 1
    assert audit[0]["correct_reference_count"] == 0
    assert json.loads(rows[0]["messages"][-1]["content"]) == {"selections": {"hidden_note": None}}
    assert validate_training_row(rows[0]) == "grounding"


@pytest.mark.parametrize(
    "change",
    [
        "gold_hint",
        "no_omission",
        "nonnull",
        "candidate",
        "wrong_type",
        "same_gold",
        "review_protocol",
        "holdout_exposure",
        "gold_from_predictions",
        "file_exposure",
    ],
)
def test_missing_context_rejects_forged_targets_and_hidden_prediction_exposure(base, tmp_path, change):
    path, review_path, payload, cases, labels_hash, write, review = missing_curriculum(base, tmp_path)
    example = payload["examples"][0]
    if change == "gold_hint":
        example["visible_business_data_before_override"]["hidden_note"] = "long-body-marker"
    elif change == "no_omission":
        example["omitted_gold_paths_in_original_view"] = []
    elif change == "nonnull":
        example["expected_selections"]["selections"]["hidden_note"] = example["rejected_contexts"][0][
            "candidate_id"
        ]
    elif change == "candidate":
        example["rejected_contexts"][0]["candidate_id"] = "c9999"
    elif change == "wrong_type":
        example["data_overrides"]["hidden_note"] = 17
    elif change == "same_gold":
        example["data_overrides"]["hidden_note"] = example["original_gold_value"]
    elif change == "review_protocol":
        review.pop("review_protocol")
    elif change == "holdout_exposure":
        review["holdout_predictions_seen"] = True
    elif change == "gold_from_predictions":
        review["training_predictions_used_as_gold"] = True
    elif change == "file_exposure":
        payload["prediction_exposure"]["validation_predictions_seen"] = False
    write()
    with pytest.raises(ValueError):
        augmentation._missing_context_rows(path, review_path, cases, labels_hash)


def test_missing_context_files_are_a_separate_explicit_pair(base, tmp_path):
    with pytest.raises(ValueError, match="together"):
        augmentation.augment_business(
            base, tmp_path / "missing-review", missing_context_negatives=tmp_path / "missing.json"
        )
