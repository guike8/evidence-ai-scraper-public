"""Derive TRAIN-only business schema variations from already reviewed facts.

This optional compiler never retrieves pages, calls a model, invents labels,
or parses validation/holdout rows. Projections and aliases are derived
supervision over the same training pages and facts, not independent examples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from ai_scraper.extraction.generic import populated_fields
from ai_scraper.extraction.grounded import (
    GROUNDED_POLICY_VERSION,
    bind_grounding_selection,
    build_grounding_task,
)
from ai_scraper.extraction.grounded_view import build_grounded_view
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest
from training import evaluate
from training.realpage_dataset import (
    TARGET_FORMAT,
    _gold_selections,
    _negative_rows,
    _training_visible_data,
    _visible_evidence,
    approved_negative_exposure,
    canonical,
    digest,
    file_hash,
    make_training_row,
    require,
    validate_training_row,
)
from training.train import workflow_hashes

FORMAT = "realpage-v5-derived-business-curriculum-v1"
PAIRED_FORMAT = "realpage-v6-paired-grounding-curriculum-v1"
MISSING_FORMAT = "realpage-v5-missing-context-grounding-negatives-v1"
EXPOSURE = {
    "training_predictions_used_as_gold": False,
    "validation_predictions_seen": True,
    "holdout_predictions_seen": False,
}
PREFIX = "requested_"
FILES = frozenset(
    f"{split}{suffix}" for split in ("train", "valid", "test") for suffix in (".jsonl", ".cases.jsonl")
)


def _in_family(url, family):
    host = urlsplit(url).hostname or ""
    return host == family or host.endswith("." + family)


def _remap_evidence(evidence, mapping):
    result = []
    for anchor in evidence:
        root, dot, suffix = anchor["field_path"].partition(".")
        if root in mapping:
            result.append({**anchor, "field_path": mapping[root] + (dot + suffix if dot else "")})
    return result


def _derive(case, request, snapshot, data, evidence, kind, mapping, max_source_chars):
    descriptors = []
    for field in request.fields:
        if field.name in mapping:
            name = mapping[field.name]
            require(len(name) <= 64, "Derived field alias exceeds the trusted name bound")
            descriptors.append({**field.model_dump(mode="json"), "name": name})
    require(bool(descriptors), "Projection must keep at least one original field")
    projected_request = ExtractionRequest.model_validate(
        {**request.model_dump(mode="json"), "fields": descriptors}
    )
    spec = get_schema(projected_request)
    # The selector sees only the trusted request and immutable snapshot, never
    # business values, reviewed anchors or the eventual expected answer.
    view = build_grounded_view(snapshot, projected_request, spec, max_source_chars)
    source = json.loads(view.content)
    projected_data = {mapping[name]: copy.deepcopy(value) for name, value in data.items() if name in mapping}
    anchors = _remap_evidence(evidence, mapping)
    visible, omitted = _training_visible_data(projected_data, anchors, source)
    included = {section["id"] for section in source["sections"]}
    by_path = {anchor["field_path"]: anchor for anchor in anchors}
    # Verify retained source-conditioned facts through original list indices,
    # including repeated values whose earlier occurrence was out of view.
    for field, value in projected_data.items():
        if isinstance(value, list):
            retained = [
                item
                for index, item in enumerate(value)
                if by_path[f"{field}.{index}"]["section_id"] in included
            ]
            require(
                visible[field] == retained,
                "Projected list target changed its reviewed facts",
            )
        elif visible[field] is not None:
            require(
                visible[field] == value and by_path[field]["section_id"] in included,
                "Projected target lacks its reviewed included anchor",
            )
    row = make_training_row(projected_request, source, "business", visible)
    require(
        validate_training_row(row) == "business",
        "Derived row must remain business-only supervision",
    )
    return row, {
        "case_id": case["case_id"],
        "page_id": case["page_id"],
        "kind": kind,
        "field_mapping": mapping,
        "original_source_hash": snapshot.source_hash,
        "original_gold_leaf_count": len(populated_fields(projected_data)),
        "visible_target_leaf_count": len(populated_fields(visible)),
        "omitted_original_gold_paths": omitted,
        "source_view": view.as_metadata(),
        "row_sha256": digest(row),
        "provenance": "derived_schema_supervision_over_unchanged_reviewed_TRAIN_facts",
    }


def _paired_grounding(business_row, audit, snapshot, data, evidence):
    """Pair a derived business task with its exact reviewed visible anchors."""
    require(validate_training_row(business_row) == "business", "Only business targets may be paired")
    meta = business_row["grounded"]
    request = ExtractionRequest.model_validate(meta["request"])
    spec = get_schema(request)
    source = meta["source"]
    mapping = audit["field_mapping"]
    projected = {mapping[name]: copy.deepcopy(value) for name, value in data.items() if name in mapping}
    anchors = _remap_evidence(evidence, mapping)
    visible, omitted = _training_visible_data(projected, anchors, source)
    require(
        canonical(visible) == canonical(evaluate._parse(business_row["messages"][-1]["content"])),
        "Paired grounding must use the exact corresponding business target",
    )
    task = build_grounding_task(spec, visible, source, request.instruction, max_source_chars=8000)
    # Reindex through original gold positions, including repeated list values.
    visible_anchors = _visible_evidence(projected, anchors, source)
    selection = _gold_selections(task, visible_anchors)
    row = make_training_row(request, source, "grounding", selection, data=visible)
    require(validate_training_row(row) == "grounding", "Paired grounding target must reconstruct exactly")
    # Like the primary compiler, conditionally omitted critical values remain
    # valid supervised abstentions; complete gold receives the production gate.
    if not omitted:
        bound = bind_grounding_selection(task, selection, snapshot, spec, request.locale)
        require(canonical(bound["data"]) == canonical(visible), "Grounding binding changed reviewed facts")
    return row, {
        **audit,
        "kind": "paired_positive_grounding",
        "business_row_sha256": digest(business_row),
        "row_sha256": digest(row),
        "grounding_catalog_sha256": task.metadata["catalog_hash"],
        "selected_anchor_count": len(selection["selections"]),
        "omitted_original_gold_paths": omitted,
        "provenance": "paired_grounding_over_exact_reviewed_derived_TRAIN_business_targets",
    }


def _missing_context_rows(path, review_path, train_cases, labels_sha256):
    """Separate reviewed abstention targets with full-gold context out of view."""
    path, review_path = Path(path), Path(review_path)
    require(
        path.stat().st_size <= 10_000_000,
        "Missing-context curriculum exceeds its file bound",
    )
    payload, review = (
        evaluate._parse(path.read_text()),
        evaluate._parse(review_path.read_text()),
    )
    require(
        isinstance(review, dict)
        and review.get("status") == "approved"
        and review.get("negatives_sha256") == file_hash(path)
        and (review.get("model_outputs_seen") is False or approved_negative_exposure(review))
        and review.get("unresolved") == [],
        "Exact missing-context review and truthful prediction exposure are required",
    )
    top_keys = {
        "format",
        "approved_business_labels_sha256",
        "base_train_cases_sha256",
        "source_limit_chars",
        "curriculum_role",
        "request_policy",
        "target_policy",
        "examples",
        "model_outputs_consulted",
        "model_calls",
        "training_calls",
        "approved_business_labels_modified",
        "raw_redistribution",
        "failures",
        "annotation_scope",
        "prediction_exposure",
    }
    require(
        isinstance(payload, dict) and set(payload) == top_keys,
        "Unexpected missing-context curriculum keys",
    )
    require(
        payload["format"] == MISSING_FORMAT
        and payload["approved_business_labels_sha256"] == labels_sha256
        and payload["base_train_cases_sha256"]
        == hashlib.sha256("".join(canonical(case) + "\n" for case in train_cases).encode()).hexdigest()
        and type(payload["source_limit_chars"]) is int
        and payload["source_limit_chars"] == 8000,
        "Missing-context curriculum must bind original TRAIN cases, labels and source policy",
    )
    require(
        payload["annotation_scope"]
        == "after_validation_before_final_site_test; TRAIN source annotations only"
        and isinstance(payload["prediction_exposure"], dict)
        and set(payload["prediction_exposure"]) == set(EXPOSURE)
        and all(payload["prediction_exposure"][key] is value for key, value in EXPOSURE.items())
        and payload["model_outputs_consulted"] is True
        and payload["approved_business_labels_modified"] is False
        and type(payload["model_calls"]) is int
        and payload["model_calls"] == 0
        and type(payload["training_calls"]) is int
        and payload["training_calls"] == 0
        and payload["raw_redistribution"] is False
        and payload["failures"] == [],
        "Missing-context annotations must disclose validation-informed TRAIN-only derivation",
    )
    examples = payload["examples"]
    require(
        isinstance(examples, list) and 1 <= len(examples) <= 1000,
        "Invalid missing-context example count",
    )
    example_keys = {
        "example_id",
        "case_id",
        "page_id",
        "split",
        "site_family",
        "data_overrides",
        "wrong_paths",
        "original_gold_value",
        "visible_business_data_before_override",
        "counterfactual_business_data",
        "omitted_gold_paths_in_original_view",
        "wrong_candidate_count",
        "rejected_contexts",
        "correct_references_preserved",
        "expected_selections",
        "source_view_sha256",
        "source_view_descriptor",
        "grounding_task_sha256",
        "grounding_catalog_sha256",
        "request_sha256",
        "source_hash",
        "semantic_rejection_reason",
        "annotation_status",
    }
    indexed, seen, rows, audits = (
        {case["case_id"]: case for case in train_cases},
        set(),
        [],
        [],
    )
    for example in examples:
        require(
            isinstance(example, dict) and set(example) == example_keys,
            "Unexpected missing-context example keys",
        )
        example_id = example["example_id"]
        require(
            isinstance(example_id, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", example_id) is not None
            and example_id not in seen
            and len(example_id) <= 100,
            "Invalid missing-context example ID",
        )
        seen.add(example_id)
        require(isinstance(example["case_id"], str), "Invalid missing-context case ID")
        case = indexed.get(example["case_id"])
        require(
            case is not None
            and case["split"] == example["split"] == "train"
            and case["page_id"] == example["page_id"]
            and case["domain"] == example["site_family"],
            "Missing-context examples may use only their original TRAIN source",
        )
        request, spec, snapshot, gold = evaluate._context(case)
        data, evidence = (
            gold.data.model_dump(mode="json"),
            case["expected_candidate"]["evidence"],
        )
        view = build_grounded_view(snapshot, request, spec, 8000)
        source = json.loads(view.content)
        visible, omitted = _training_visible_data(data, evidence, source)
        wrong, overrides = example["wrong_paths"], example["data_overrides"]
        require(
            isinstance(wrong, list)
            and len(wrong) == 1
            and isinstance(wrong[0], str)
            and isinstance(overrides, dict)
            and set(overrides) == set(wrong),
            "Missing-context examples override exactly one original scalar",
        )
        field = wrong[0]
        require(
            field in data
            and data[field] is not None
            and not isinstance(data[field], (list, dict))
            and visible[field] is None
            and field in omitted
            and type(overrides[field]) is type(data[field])
            and overrides[field] != data[field],
            "Missing-context targets require a wrong value of the full-gold type for an omitted scalar",
        )
        anchor = next(item for item in evidence if item["field_path"] == field)
        require(
            anchor["section_id"] in view.as_metadata()["excluded_section_ids"]
            and anchor["section_id"] not in {item["id"] for item in source["sections"]},
            "Missing full-gold anchor must be explicitly excluded from the shown source",
        )
        predicted = {**visible, **overrides}
        require(
            canonical(spec.data_model.model_validate(predicted).model_dump(mode="json"))
            == canonical(predicted),
            "Counterfactual changed the complete trusted business schema",
        )
        require(
            canonical(example["original_gold_value"]) == canonical(data[field])
            and canonical(example["visible_business_data_before_override"]) == canonical(visible)
            and canonical(example["counterfactual_business_data"]) == canonical(predicted)
            and example["omitted_gold_paths_in_original_view"] == omitted,
            "Missing-context data changed original source-conditioned facts",
        )
        task = build_grounding_task(spec, predicted, source, request.instruction, max_source_chars=8000)
        require(
            example["source_view_sha256"] == hashlib.sha256(view.content.encode()).hexdigest()
            and example["source_view_descriptor"] == view.as_metadata()
            and example["source_hash"] == snapshot.source_hash
            and example["request_sha256"] == digest(request.model_dump(mode="json"))
            and example["grounding_catalog_sha256"] == task.metadata["catalog_hash"]
            and example["grounding_task_sha256"]
            == digest(
                {
                    "schema": task.schema,
                    "messages": task.messages,
                    "metadata": task.metadata,
                }
            ),
            "Missing-context source, task, schema or catalog identity changed",
        )
        candidates, rejected = task.candidates[field], example["rejected_contexts"]
        require(
            bool(candidates)
            and type(example["wrong_candidate_count"]) is int
            and example["wrong_candidate_count"] == len(candidates)
            and isinstance(rejected, list)
            and len(rejected) == len(candidates),
            "All eligible wrong-value candidates require semantic review",
        )
        for recorded, candidate in zip(rejected, candidates, strict=True):
            expected = {
                "field_path": field,
                "candidate_id": candidate.id,
                "section_id": candidate.section_id,
                "json_pointer": candidate.json_pointer,
                "excerpt": candidate.excerpt,
                "context": json.loads(candidate.context),
            }
            require(
                isinstance(recorded, dict)
                and set(recorded) == {*expected, "reason"}
                and {key: recorded[key] for key in expected} == expected
                and isinstance(recorded["reason"], str)
                and 0 < len(recorded["reason"].strip()) <= 2000,
                "A missing-context candidate or reviewed reason changed",
            )
        anchors = {item["field_path"]: item for item in _visible_evidence(data, evidence, source)}
        selected, preserved = {}, []
        for path_name, available in task.candidates.items():
            if path_name == field:
                selected[path_name] = None
                continue
            correct = anchors[path_name]
            matches = [
                candidate
                for candidate in available
                if candidate.section_id == correct["section_id"] and correct["excerpt"] in candidate.excerpt
            ]
            require(bool(matches), "Other visible fields must retain reviewed anchors")
            selected[path_name] = matches[0].id
            preserved.append(
                {
                    "field_path": path_name,
                    "candidate_id": matches[0].id,
                    "section_id": matches[0].section_id,
                    "excerpt": correct["excerpt"],
                }
            )
        require(
            example["expected_selections"] == {"selections": selected}
            and example["correct_references_preserved"] == preserved
            and isinstance(example["semantic_rejection_reason"], str)
            and 0 < len(example["semantic_rejection_reason"].strip()) <= 2000,
            "Missing-context targets must abstain and preserve all other visible anchors",
        )
        row = make_training_row(request, source, "grounding", {"selections": selected}, data=predicted)
        rows.append(row)
        audits.append(
            {
                "example_id": example_id,
                "case_id": case["case_id"],
                "wrong_paths": wrong,
                "omitted_full_gold_anchor": anchor["section_id"],
                "rejected_candidate_count": len(candidates),
                "correct_reference_count": len(preserved),
                "row_sha256": digest(row),
                "provenance": "validation_informed_TRAIN_only_missing_context_counterfactual_grounding",
            }
        )
    return (
        rows,
        audits,
        {
            "count": len(rows),
            "negatives_sha256": file_hash(path),
            "independent_review_sha256": file_hash(review_path),
            "approved_business_labels_sha256": labels_sha256,
            "base_train_cases_sha256": payload["base_train_cases_sha256"],
            "prediction_exposure": dict(EXPOSURE),
            "provenance": "Validation-informed TRAIN-only omitted-context grounding; unchanged full-page business gold",
        },
    )


def augment_business(
    base_dir: Path,
    output_dir: Path,
    *,
    grounding_negatives: Path | None = None,
    negative_review_receipt: Path | None = None,
    missing_context_negatives: Path | None = None,
    missing_context_review: Path | None = None,
    pair_grounding: bool = False,
    seed: int = 2026110805,
):
    os.umask(0o077)
    base_dir, output_dir = Path(base_dir).resolve(), Path(output_dir).resolve()
    require(not output_dir.exists(), "Choose a new private output directory")
    require(
        not output_dir.is_relative_to(Path(__file__).resolve().parents[1]),
        "Keep private derived datasets outside the source checkout",
    )
    require(
        not output_dir.is_relative_to(base_dir) and not base_dir.is_relative_to(output_dir),
        "Parent and derived datasets must remain separate",
    )
    require(
        (grounding_negatives is None) == (negative_review_receipt is None),
        "Supplemental negatives and their review must be supplied together",
    )
    require(
        (missing_context_negatives is None) == (missing_context_review is None),
        "Missing-context negatives and their review must be supplied together",
    )
    require(type(seed) is int and 0 <= seed < 2**32, "A bounded shuffle seed is required")
    require(type(pair_grounding) is bool, "Paired grounding must be an explicit boolean")
    manifest_path = base_dir / "manifest.json"
    parent = evaluate._parse(manifest_path.read_text())
    require(
        isinstance(parent, dict)
        and parent.get("target_format") == TARGET_FORMAT
        and parent.get("policy_version") == GROUNDED_POLICY_VERSION,
        "Parent dataset has a different training or prompt policy",
    )
    require(
        parent.get("source_limit_chars") == 8000,
        "Derived recipe requires the declared 8000-character source policy",
    )
    require(
        isinstance(parent.get("files"), dict) and set(parent["files"]) == FILES,
        "Parent manifest must bind exactly all six original split files",
    )
    for name in FILES:
        path = base_dir / name
        require(
            path.is_file() and not path.is_symlink() and path.stat().st_size <= evaluate.MAX_FILE_BYTES,
            "Parent split file is missing or exceeds its bound",
        )
        require(
            file_hash(path) == parent["files"][name],
            "Parent split file checksum changed",
        )
    # Deliberately parse TRAIN only. Other split files are checksum-verified and
    # copied byte-for-byte below, without inspecting their requests or labels.
    cases = evaluate.read_jsonl(base_dir / "train.cases.jsonl")
    base_rows = evaluate.read_jsonl(base_dir / "train.jsonl")
    require(
        len(cases) == parent["case_counts"]["train"] and len(base_rows) == parent["row_counts"]["train"],
        "Parent TRAIN counts changed",
    )
    evaluate.validate_cases(cases)
    train_families = set(parent["split_site_families"]["train"])
    require(
        {case["domain"] for case in cases} == train_families,
        "TRAIN case families differ from their manifest",
    )
    for split in ("valid", "test"):
        require(
            not any(
                a == b or a.endswith("." + b) or b.endswith("." + a)
                for a in train_families
                for b in parent["split_site_families"][split]
            ),
            "TRAIN family overlaps a reserved split",
        )
    allowed_urls = {case["request"]["url"] for case in cases}
    rows, seen, provenance, counts = [], {}, [], Counter()

    def add(row, kind, audit=None):
        task = validate_training_row(row)
        require(
            row["grounded"]["request"]["url"] in allowed_urls,
            "A derived row must use an original TRAIN URL",
        )
        require(
            any(_in_family(row["grounded"]["source"]["final_url"], family) for family in train_families),
            "A derived source is outside TRAIN families",
        )
        serialized, row_hash = canonical(row), digest(row)
        if row_hash in seen:
            require(seen[row_hash] == serialized, "Canonical row digest collision")
            counts[f"deduplicated_{kind}"] += 1
            return
        seen[row_hash] = serialized
        rows.append(row)
        counts[f"added_{kind}"] += 1
        counts[f"task_{task}"] += 1
        if audit:
            provenance.append(audit)

    for row in base_rows:
        add(row, "parent")
    for case in cases:
        request, spec, snapshot, gold = evaluate._context(case)
        require(
            case["split"] == "train"
            and spec.schema_id == "custom_v1"
            and _in_family(snapshot.final_url, case["domain"]),
            "Only original custom TRAIN cases may be transformed",
        )
        data, evidence = (
            gold.data.model_dump(mode="json"),
            case["expected_candidate"]["evidence"],
        )
        for field in request.fields:
            row, audit = _derive(
                case,
                request,
                snapshot,
                data,
                evidence,
                "single_field_projection",
                {field.name: field.name},
                8000,
            )
            add(row, "single_field_projection", audit)
            if pair_grounding:
                paired, paired_audit = _paired_grounding(row, audit, snapshot, data, evidence)
                add(paired, "paired_positive_grounding", paired_audit)
        aliases = {field.name: PREFIX + field.name for field in request.fields}
        row, audit = _derive(case, request, snapshot, data, evidence, "full_schema_alias", aliases, 8000)
        add(row, "full_schema_alias", audit)
        if pair_grounding:
            paired, paired_audit = _paired_grounding(row, audit, snapshot, data, evidence)
            add(paired, "paired_positive_grounding", paired_audit)
    negative_metadata = None
    negative_audits = []
    if grounding_negatives is not None:
        supplemental, negative_audits, negative_metadata = _negative_rows(
            Path(grounding_negatives),
            Path(negative_review_receipt),
            cases,
            parent["labels_sha256"],
            8000,
        )
        for row in supplemental:
            add(row, "supplemental_grounding_negative")
    missing_metadata, missing_audits = None, []
    if missing_context_negatives is not None:
        supplemental, missing_audits, missing_metadata = _missing_context_rows(
            missing_context_negatives, missing_context_review, cases, parent["labels_sha256"]
        )
        for row in supplemental:
            add(row, "missing_context_grounding_negative")
    require(counts["task_grounding"] > 0, "Two-stage training requires grounding supervision")
    random.Random(seed).shuffle(rows)
    source_hashes = workflow_hashes(TARGET_FORMAT)
    source_hashes["realpage_augment.py"] = file_hash(Path(__file__))
    output_dir.mkdir(parents=True, mode=0o700)
    files = {}
    for name in sorted(FILES - {"train.jsonl"}):
        shutil.copyfile(base_dir / name, output_dir / name)
        require(
            file_hash(output_dir / name) == parent["files"][name],
            "Copied reserved split bytes changed",
        )
        files[name] = file_hash(output_dir / name)
    (output_dir / "train.jsonl").write_text("".join(canonical(row) + "\n" for row in rows))
    files["train.jsonl"] = file_hash(output_dir / "train.jsonl")
    (output_dir / "derived-business-audit.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n"
    )
    files["derived-business-audit.json"] = file_hash(output_dir / "derived-business-audit.json")
    if negative_audits:
        (output_dir / "supplemental-grounding-audit.json").write_text(
            json.dumps(negative_audits, indent=2) + "\n"
        )
        files["supplemental-grounding-audit.json"] = file_hash(
            output_dir / "supplemental-grounding-audit.json"
        )
    if missing_audits:
        (output_dir / "missing-context-grounding-audit.json").write_text(
            json.dumps(missing_audits, indent=2) + "\n"
        )
        files["missing-context-grounding-audit.json"] = file_hash(
            output_dir / "missing-context-grounding-audit.json"
        )
    manifest = {
        **parent,
        "format": PAIRED_FORMAT if pair_grounding else FORMAT,
        "parent_dataset_fingerprint": parent["dataset_fingerprint"],
        "parent_manifest_sha256": file_hash(manifest_path),
        "parent_split_hashes": parent["files"],
        "shuffle_seed": seed,
        "files": files,
        "row_counts": {**parent["row_counts"], "train": len(rows)},
        "derived_business_transformations": dict(counts),
        "source_workflow_hashes": source_hashes,
        "augmentation_module_sha256": file_hash(Path(__file__)),
        "original_TRAIN_cases": len(cases),
        "original_TRAIN_pages": len({case["page_id"] for case in cases}),
        "original_TRAIN_positive_facts": sum(len(case["expected_fields"]) for case in cases),
        "training_business_to_grounding_ratio": counts["task_business"] / counts["task_grounding"],
        "validation_and_holdout_content_parsed": False,
        "reserved_rows_and_all_case_files_byte_unchanged": True,
        "supplemental_grounding_negatives": negative_metadata,
        "missing_context_grounding_negatives": missing_metadata,
        "annotation_scope": "after_validation_before_final_site_test; TRAIN source annotations only",
        "prediction_exposure": dict(EXPOSURE),
        "builder_model_calls": 0,
        "builder_training_calls": 0,
        "curriculum_provenance": "TRAIN-only field projections and aliases over the same reviewed real-page facts; not new pages or independent labels.",
    }
    if pair_grounding:
        manifest["paired_grounding"] = {
            "enabled": True,
            "policy": "Every derived business target is paired with exact reviewed visible grounding anchors",
            "design_provenance": "Validation-informed response to v5 grounding rejection regression; no site-test predictions seen during design",
            "added_rows": counts["added_paired_positive_grounding"],
            "deduplicated_rows": counts["deduplicated_paired_positive_grounding"],
            "new_business_gold_facts": 0,
        }
    manifest.pop("model_calls_before_freeze", None)
    manifest["dataset_fingerprint"] = digest(
        {
            "files": files,
            "parent_dataset_fingerprint": parent["dataset_fingerprint"],
            "transformations": dict(counts),
            "source_workflow_hashes": source_hashes,
            "shuffle_seed": seed,
            "supplemental_grounding_negatives": negative_metadata,
            "missing_context_grounding_negatives": missing_metadata,
            "prediction_exposure": dict(EXPOSURE),
            **({"paired_grounding": manifest["paired_grounding"]} if pair_grounding else {}),
        }
    )
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grounding-negatives", type=Path)
    parser.add_argument("--negative-review-receipt", type=Path)
    parser.add_argument("--missing-context-negatives", type=Path)
    parser.add_argument("--missing-context-review", type=Path)
    parser.add_argument(
        "--pair-grounding",
        action="store_true",
        help="Pair derived business tasks with reviewed positive grounding",
    )
    parser.add_argument("--seed", type=int, default=2026110805)
    args = parser.parse_args(argv)
    manifest = augment_business(
        args.base_dir,
        args.output_dir,
        grounding_negatives=args.grounding_negatives,
        negative_review_receipt=args.negative_review_receipt,
        missing_context_negatives=args.missing_context_negatives,
        missing_context_review=args.missing_context_review,
        pair_grounding=args.pair_grounding,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "dataset_fingerprint": manifest["dataset_fingerprint"],
                "row_counts": manifest["row_counts"],
                "transformations": manifest["derived_business_transformations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
