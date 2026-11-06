"""Compile reviewed saved-page labels into explicit two-stage training tasks.

Raw pages, annotations and generated rows stay outside the source checkout.
Production task builders define the prompts, schemas and evidence catalog;
labels never choose the model's source window or eligible citation candidates.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from ai_scraper.contracts import ScraperError
from ai_scraper.extraction.generic import populated_fields
from ai_scraper.extraction.grounded import (
    GROUNDED_POLICY_VERSION,
    bind_grounding_selection,
    build_business_task,
    build_grounding_task,
)
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest, SourceSnapshot
from training import evaluate

FORMAT = "realpage-grounded-training-row-v1"
TARGET_FORMAT = "realpage_grounded_v1"
MAX_ROW_CHARS = 300_000
SPLITS = {"train": "train", "validation": "valid", "site_holdout": "test"}
NEGATIVE_FORMAT = "realpage-v4-counterfactual-grounding-negatives-v1"


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _complete_business(spec, value):
    require(isinstance(value, dict), "Business target must be an object")
    parsed = spec.data_model.model_validate(value).model_dump(mode="json")
    require(
        canonical(parsed) == canonical(value),
        "Business target must contain every trusted schema key and type",
    )
    return parsed


def rebuild_training_task(row: dict):
    require(isinstance(row, dict) and set(row) == {"messages", "grounded"}, "Unexpected training row keys")
    require(len(canonical(row)) <= MAX_ROW_CHARS, "Training row exceeds its character bound")
    meta = row["grounded"]
    require(isinstance(meta, dict), "Missing trusted task metadata")
    task = meta.get("task")
    keys = {"format", "task", "request", "source", "schema_sha256", "task_sha256"}
    if task == "grounding":
        keys.add("data")
    require(task in {"business", "grounding"} and set(meta) == keys, "Unexpected grounded task metadata")
    require(meta["format"] == FORMAT, "Unknown grounded training format")
    request = ExtractionRequest.model_validate(meta["request"])
    spec = get_schema(request)
    require(spec.schema_id == "custom_v1", "Real-page pilot requires a trusted custom-field request")
    if task == "business":
        built = build_business_task(spec, meta["source"], request.instruction, max_source_chars=48_000)
    else:
        built = build_grounding_task(
            spec, meta["data"], meta["source"], request.instruction, max_source_chars=48_000
        )
    require(meta["schema_sha256"] == digest(built.schema), "Training schema identity changed")
    require(
        meta["task_sha256"] == digest({"messages": built.messages, "metadata": built.metadata}),
        "Training task identity changed",
    )
    messages = row["messages"]
    require(
        isinstance(messages, list) and len(messages) == 3,
        "Training requires exact two-message prompt and one target",
    )
    require(messages[:-1] == built.messages, "Training prompt differs from the production task")
    require(
        isinstance(messages[-1], dict)
        and set(messages[-1]) == {"role", "content"}
        and messages[-1]["role"] == "assistant"
        and isinstance(messages[-1]["content"], str),
        "Invalid supervised assistant target",
    )
    return task, spec, built, evaluate._parse(messages[-1]["content"])


def validate_training_row(row: dict) -> str:
    """Reconstruct exact production tasks; never trust a row-supplied schema."""
    try:
        task, spec, built, target = rebuild_training_task(row)
        if task == "business":
            _complete_business(spec, target)
        else:
            require(
                isinstance(target, dict) and set(target) == {"selections"}, "Unexpected grounding target keys"
            )
            selected = target["selections"]
            require(
                isinstance(selected, dict) and set(selected) == set(built.candidates),
                "Grounding target must include exactly every populated leaf",
            )
            for path, value in selected.items():
                require(
                    value is None
                    or (type(value) is str and value in {item.id for item in built.candidates[path]}),
                    "Grounding target contains an out-of-catalog selection",
                )
        return task
    except (ScraperError, TypeError, KeyError, RecursionError, UnicodeError) as exc:
        raise ValueError("Invalid bounded real-page training task") from exc


def make_training_row(request, source, task, target, *, data=None):
    spec = get_schema(request)
    built = (
        build_business_task(spec, source, request.instruction)
        if task == "business"
        else build_grounding_task(spec, data, source, request.instruction)
    )
    meta = {
        "format": FORMAT,
        "task": task,
        "request": request.model_dump(mode="json"),
        "source": copy.deepcopy(source),
        "schema_sha256": digest(built.schema),
        "task_sha256": digest({"messages": built.messages, "metadata": built.metadata}),
    }
    if task == "grounding":
        meta["data"] = copy.deepcopy(data)
    # Grammar emits schema property order; retain trusted declaration order.
    if task == "business":
        target = _complete_business(spec, target)
    row = {
        "messages": [
            *built.messages,
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
            },
        ],
        "grounded": meta,
    }
    validate_training_row(row)
    return row


def _evidence_for_label(snapshot, task):
    evidence = []
    indexed = {section.id: section for section in snapshot.sections}
    for anchor in task["evidence"]:
        require(
            set(anchor) >= {"field_path", "section_id", "excerpt", "semantic_context"},
            "Gold evidence needs reviewed field and semantic context",
        )
        section = indexed.get(anchor["section_id"])
        require(
            section is not None and section.representation == "markdown",
            "Gold evidence must use the saved preparation Markdown",
        )
        require(anchor["excerpt"] in section.text, "Gold excerpt absent from its saved section")
        require(bool(anchor["semantic_context"]), "Gold requires a semantic review note")
        evidence.append(
            {
                "field_path": anchor["field_path"],
                "section_id": section.id,
                "json_pointer": None,
                "excerpt": anchor["excerpt"],
            }
        )
    return evidence


def _training_visible_data(data, evidence, source):
    """Absent-from-view gold becomes null only in training, with logged losses.

    Validation/test case labels always keep the full saved-page denominator.
    Source selection itself never sees gold. The pilot has flat custom fields.
    """
    included = {section["id"] for section in source["sections"]}
    by_path = {item["field_path"]: item for item in evidence}
    visible = copy.deepcopy(data)
    omitted = []
    for field, value in data.items():
        if isinstance(value, list):
            kept = []
            for index, item in enumerate(value):
                path = f"{field}.{index}"
                if by_path[path]["section_id"] in included:
                    kept.append(item)
                else:
                    omitted.append(path)
            visible[field] = kept
        elif value is not None and by_path[field]["section_id"] not in included:
            visible[field] = None
            omitted.append(field)
    return visible, omitted


def _gold_selections(task, evidence):
    anchors = {item["field_path"]: item for item in evidence}
    result = {}
    for path, candidates in task.candidates.items():
        anchor = anchors[path]
        matching = [
            item
            for item in candidates
            if item.section_id == anchor["section_id"] and anchor["excerpt"] in item.excerpt
        ]
        require(bool(matching), "Reviewed citation is outside the exact predicted-value catalog")
        result[path] = matching[0].id
    return {"selections": result}


def _visible_evidence(data, evidence, source):
    """Preserve original list indices while reindexing visible training leaves."""
    included = {section["id"] for section in source["sections"]}
    anchors = {anchor["field_path"]: anchor for anchor in evidence}
    result = []
    for field, value in data.items():
        if isinstance(value, list):
            visible_index = 0
            for original_index, _ in enumerate(value):
                anchor = anchors[f"{field}.{original_index}"]
                if anchor["section_id"] in included:
                    result.append({**anchor, "field_path": f"{field}.{visible_index}"})
                    visible_index += 1
        elif value is not None and anchors[field]["section_id"] in included:
            result.append(anchors[field])
    return result


def _negative_rows(path, review_path, train_cases, labels_sha256, max_source_chars):
    """Add reviewed semantic-rejection supervision, never new business gold."""
    from ai_scraper.extraction.grounded_view import build_grounded_view

    require(path.stat().st_size <= 10_000_000, "Negative curriculum exceeds its file bound")
    negatives = evaluate._parse(path.read_text())
    review = evaluate._parse(review_path.read_text())
    require(
        isinstance(review, dict)
        and review.get("status") == "approved"
        and review.get("negatives_sha256") == file_hash(path)
        and review.get("model_outputs_seen") is False
        and review.get("unresolved") == [],
        "Independent negative review must approve these exact bytes",
    )
    require(isinstance(negatives, dict), "Negative curriculum must be an object")
    require(
        set(negatives)
        == {
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
        },
        "Unexpected negative curriculum keys",
    )
    require(
        negatives["format"] == NEGATIVE_FORMAT
        and negatives["approved_business_labels_sha256"] == labels_sha256
        and negatives["base_train_cases_sha256"]
        == hashlib.sha256("".join(canonical(case) + "\n" for case in train_cases).encode()).hexdigest()
        and type(negatives["source_limit_chars"]) is int
        and negatives["source_limit_chars"] == max_source_chars == 8000,
        "Negative curriculum labels, train cases or source limit differ",
    )
    require(
        negatives["model_outputs_consulted"] is False
        and negatives["model_calls"] == 0
        and negatives["training_calls"] == 0
        and negatives["approved_business_labels_modified"] is False
        and negatives["raw_redistribution"] is False
        and negatives["failures"] == [],
        "Negative curriculum must be reviewed before model calls without modifying business gold",
    )
    examples = negatives["examples"]
    require(isinstance(examples, list) and 1 <= len(examples) <= 1000, "Invalid negative example count")
    indexed = {case["case_id"]: case for case in train_cases}
    rows, audits, seen = [], [], set()
    keys = {
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
    for example in examples:
        require(isinstance(example, dict) and set(example) == keys, "Unexpected negative example keys")
        example_id = example["example_id"]
        require(
            isinstance(example_id, str)
            and re.fullmatch(r"[a-z0-9][a-z0-9-]{0,99}", example_id) is not None
            and example_id not in seen,
            "Invalid or duplicate negative example ID",
        )
        seen.add(example_id)
        case = indexed.get(example["case_id"])
        require(
            case is not None
            and case["split"] == example["split"] == "train"
            and example["page_id"] == case["page_id"]
            and example["site_family"] == case["domain"],
            "Negative examples may link only their original TRAIN case",
        )
        request, spec, snapshot, gold = evaluate._context(case)
        data = gold.data.model_dump(mode="json")
        evidence = case["expected_candidate"]["evidence"]
        view = build_grounded_view(snapshot, request, spec, max_source_chars)
        source = json.loads(view.content)
        visible, omitted = _training_visible_data(data, evidence, source)
        overrides, wrong = example["data_overrides"], example["wrong_paths"]
        require(
            isinstance(overrides, dict)
            and isinstance(wrong, list)
            and len(wrong) == 1
            and isinstance(wrong[0], str)
            and set(overrides) == set(wrong),
            "A negative must override exactly one populated scalar",
        )
        wrong_path = wrong[0]
        require(
            wrong_path in visible
            and visible[wrong_path] is not None
            and not isinstance(visible[wrong_path], (list, dict))
            and type(overrides[wrong_path]) is type(visible[wrong_path])
            and overrides[wrong_path] != visible[wrong_path]
            and canonical(example["original_gold_value"]) == canonical(data[wrong_path]),
            "Counterfactual must change a visible scalar without changing its trusted type",
        )
        predicted = _complete_business(spec, {**visible, **overrides})
        require(
            canonical(example["visible_business_data_before_override"]) == canonical(visible)
            and canonical(example["counterfactual_business_data"]) == canonical(predicted)
            and example["omitted_gold_paths_in_original_view"] == omitted,
            "Counterfactual business data differs from the original source-conditioned target",
        )
        task = build_grounding_task(
            spec, predicted, source, request.instruction, max_source_chars=max_source_chars
        )
        require(
            example["source_view_sha256"] == hashlib.sha256(view.content.encode()).hexdigest()
            and example["source_view_descriptor"] == view.as_metadata()
            and example["source_hash"] == snapshot.source_hash
            and example["request_sha256"] == digest(request.model_dump(mode="json"))
            and example["grounding_catalog_sha256"] == task.metadata["catalog_hash"]
            and example["grounding_task_sha256"]
            == digest({"schema": task.schema, "messages": task.messages, "metadata": task.metadata}),
            "Negative source, prompt, schema or candidate identity changed",
        )
        wrong_candidates = task.candidates[wrong_path]
        require(
            bool(wrong_candidates)
            and type(example["wrong_candidate_count"]) is int
            and example["wrong_candidate_count"] == len(wrong_candidates),
            "A reviewed negative requires eligible but semantically rejected literal candidates",
        )
        rejected = example["rejected_contexts"]
        require(
            isinstance(rejected, list) and len(rejected) == len(wrong_candidates),
            "Incomplete rejected contexts",
        )
        for recorded, candidate in zip(rejected, wrong_candidates, strict=True):
            expected = {
                "field_path": wrong_path,
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
                "A rejected candidate context or semantic reason differs from the reviewed catalog",
            )
        anchors = {anchor["field_path"]: anchor for anchor in _visible_evidence(data, evidence, source)}
        selected, preserved = {}, []
        for field, candidates in task.candidates.items():
            if field == wrong_path:
                selected[field] = None
                continue
            anchor = anchors[field]
            matching = [
                c
                for c in candidates
                if c.section_id == anchor["section_id"] and anchor["excerpt"] in c.excerpt
            ]
            require(bool(matching), "Correct negative-example fields must preserve reviewed gold anchors")
            selected[field] = matching[0].id
            preserved.append(
                {
                    "field_path": field,
                    "candidate_id": matching[0].id,
                    "section_id": matching[0].section_id,
                    "excerpt": anchor["excerpt"],
                }
            )
        require(
            bool(preserved)
            and example["expected_selections"] == {"selections": selected}
            and example["correct_references_preserved"] == preserved
            and isinstance(example["semantic_rejection_reason"], str)
            and 0 < len(example["semantic_rejection_reason"].strip()) <= 2000,
            "Negative targets must abstain on the wrong value and retain every other correct reference",
        )
        rows.append(make_training_row(request, source, "grounding", {"selections": selected}, data=predicted))
        audits.append(
            {
                "example_id": example_id,
                "case_id": case["case_id"],
                "split": "train",
                "provenance": "counterfactual_derived_grounding_only",
                "wrong_paths": wrong,
                "rejected_candidate_count": len(wrong_candidates),
            }
        )
    return (
        rows,
        audits,
        {
            "count": len(rows),
            "negatives_sha256": file_hash(path),
            "independent_review_sha256": file_hash(review_path),
            "provenance": "Reviewed TRAIN-only counterfactual grounding supervision; not business gold facts",
            "approved_business_labels_sha256": labels_sha256,
            "base_train_cases_sha256": negatives["base_train_cases_sha256"],
        },
    )


def compile_dataset(
    labels_path: Path,
    snapshots: Path,
    output: Path,
    *,
    review_receipt: Path,
    max_source_chars=8000,
    seed=2026110801,
    grounding_negatives: Path | None = None,
    negative_review_receipt: Path | None = None,
):
    from ai_scraper.extraction.grounded_view import build_grounded_view

    os.umask(0o077)
    require(
        (grounding_negatives is None) == (negative_review_receipt is None),
        "Negative curriculum and review must be supplied together",
    )
    require(not output.exists(), "Choose a new private output directory")
    labels = evaluate._parse(labels_path.read_text())
    review = evaluate._parse(review_receipt.read_text())
    require(
        isinstance(review, dict)
        and review.get("status") == "approved"
        and review.get("labels_sha256") == file_hash(labels_path)
        and review.get("model_outputs_seen") is False
        and review.get("unresolved") == [],
        "Independent source review must approve these exact labels before inference",
    )
    pages = labels["pages"]
    require(isinstance(pages, list) and 3 <= len(pages) <= 1000, "A bounded reviewed page list is required")
    rows = {split: [] for split in SPLITS.values()}
    cases = {split: [] for split in SPLITS.values()}
    audits = []
    groups = {split: set() for split in SPLITS.values()}
    source_groups = {split: set() for split in SPLITS.values()}
    task_ids = set()
    page_ids = set()
    for page in pages:
        split = SPLITS[page["split"]]
        groups[split].add(page["site_family"])
        require(re.fullmatch(r"[a-z0-9][a-z0-9-]{0,80}", page["page_id"]) is not None, "Invalid page ID")
        require(page["page_id"] not in page_ids, "Duplicate reviewed page ID")
        page_ids.add(page["page_id"])
        snapshot_path = snapshots / page["page_id"] / "snapshot.json"
        snapshot = SourceSnapshot.model_validate_json(snapshot_path.read_text())
        require(
            page.get("prepared_source_hash") == snapshot.source_hash
            and page.get("prepared_snapshot_sha256") == file_hash(snapshot_path),
            "Reviewed source identity differs from its immutable snapshot",
        )
        family = page["site_family"]
        host = urlsplit(snapshot.final_url).hostname or ""
        require(
            isinstance(family, str)
            and len(family.split(".")) >= 2
            and (host == family or host.endswith("." + family)),
            "Declared site family does not match the saved source host",
        )
        require(
            snapshot.retrieval_method == "firecrawl_preparation_replay"
            and snapshot.raw_record.get("source_preparation", {}).get("kind")
            == "firecrawl_preparation_component_replay",
            "The real-page pilot requires declared Firecrawl preparation snapshots",
        )
        source_groups[split].add(snapshot.source_hash)
        for label in page["tasks"]:
            task_id = f"{page['page_id']}-{label['task_id']}"
            require(len(task_id) <= 100, "Reviewed task ID exceeds its bound")
            require(task_id not in task_ids, "Duplicate reviewed task ID")
            task_ids.add(task_id)
            request = ExtractionRequest.model_validate(label["request"])
            spec = get_schema(request)
            require(
                request.url == snapshot.requested_url and spec.schema_id == "custom_v1",
                "Task/source identity mismatch",
            )
            data = _complete_business(spec, label["expected_data"])
            evidence = _evidence_for_label(snapshot, label)
            payload = {"data": data, "evidence": evidence, "warnings": []}
            candidate = spec.candidate_model.model_validate(payload)
            assessment = evaluate._assess(candidate, snapshot, request, spec)
            require(
                not assessment.errors and assessment.status in {"success", "partial"},
                "Reviewed gold fails production validation",
            )
            require(
                {item["field_path"] for item in evidence} == set(populated_fields(data)),
                "Gold evidence must bind every populated leaf exactly",
            )
            view = build_grounded_view(snapshot, request, spec, max_source_chars)
            source = json.loads(view.content)
            case = {
                "case_id": task_id,
                "split": split,
                "domain": page["site_family"],
                "page_id": page["page_id"],
                "schema_id": spec.schema_id,
                "schema_fingerprint": spec.fingerprint,
                "request": request.model_dump(mode="json"),
                "snapshot": snapshot.model_dump(mode="json"),
                "source_hash": snapshot.source_hash,
                "expected_candidate": candidate.model_dump(mode="json"),
                "expected_fields": populated_fields(data),
                "expected_status": assessment.status,
                "reviewed_evidence": label["evidence"],
                "source_view": view.as_metadata(),
                "provenance": {
                    "kind": "reviewed_public_page_capture",
                    "label_origin": "agent_authored_and_independently_reviewed_before_model_inference",
                    "firecrawl_api_exercised": False,
                    "source_preparation": "actual_pinned_Firecrawl_component",
                    "raw_redistribution": False,
                },
            }
            cases[split].append(case)
            visible, omitted = _training_visible_data(data, evidence, source)
            # The supervised rows are always conditional on the actual source
            # shown, including token-preflight rows from non-training splits.
            rows[split].append(make_training_row(request, source, "business", visible))
            grounding_task = build_grounding_task(spec, visible, source, request.instruction)
            # Reindex surviving list leaf paths after source omissions.
            visible_evidence = _visible_evidence(data, evidence, source)
            selection = _gold_selections(grounding_task, visible_evidence)
            rows[split].append(make_training_row(request, source, "grounding", selection, data=visible))
            # All-null missing-critical training examples remain valid targets;
            # only complete gold receives a production binding check here.
            if not omitted:
                bind_grounding_selection(grounding_task, selection, snapshot, spec, request.locale)
            audits.append(
                {
                    "case_id": task_id,
                    "split": split,
                    "site_family": page["site_family"],
                    "snapshot_sha256": file_hash(snapshot_path),
                    "source_hash": snapshot.source_hash,
                    "gold_fact_count": len(populated_fields(data)),
                    "visible_training_fact_count": len(populated_fields(visible)),
                    "omitted_gold_paths": omitted,
                    "source_view": view.as_metadata(),
                    "grounding_candidates": grounding_task.metadata["candidate_count"],
                    "optional_absent_controls": sum(value is None or value == [] for value in data.values()),
                }
            )
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        require(
            not any(
                a == b or a.endswith("." + b) or b.endswith("." + a)
                for a in groups[left]
                for b in groups[right]
            ),
            "Site families overlap between splits",
        )
        require(not source_groups[left] & source_groups[right], "Saved sources overlap between splits")
    negative_audits, negative_metadata = [], None
    if grounding_negatives is not None:
        added, negative_audits, negative_metadata = _negative_rows(
            grounding_negatives,
            negative_review_receipt,
            cases["train"],
            file_hash(labels_path),
            max_source_chars,
        )
        rows["train"].extend(added)
    for split, split_rows in rows.items():
        require(bool(split_rows), "All three splits are required")
        evaluate.validate_cases(cases[split])
        random.Random(seed).shuffle(split_rows)
    output.mkdir(parents=True, mode=0o700)
    files = {}
    for split, split_rows in rows.items():
        for name, values in ((f"{split}.jsonl", split_rows), (f"{split}.cases.jsonl", cases[split])):
            path = output / name
            path.write_text("".join(canonical(row) + "\n" for row in values))
            files[name] = file_hash(path)
    manifest = {
        "format": "reviewed-realpage-grounded-corpus-v1",
        "target_format": TARGET_FORMAT,
        "policy_version": GROUNDED_POLICY_VERSION,
        "labels_sha256": file_hash(labels_path),
        "independent_review_sha256": file_hash(review_receipt),
        "source_limit_chars": max_source_chars,
        "shuffle_seed": seed,
        "page_count": len(pages),
        "case_counts": {split: len(values) for split, values in cases.items()},
        "row_counts": {split: len(values) for split, values in rows.items()},
        "split_site_families": {split: sorted(values) for split, values in groups.items()},
        "split_site_disjoint": True,
        "split_source_disjoint": True,
        "files": files,
        "model_calls_before_freeze": 0,
        "raw_redistribution": False,
        "provenance": "Actual public captures and Firecrawl preparation; agent-authored, independently reviewed labels",
        "source_loss_policy": "Automatic request-only whole-section window. Supervised out-of-view facts become null; evaluation retains all full-page gold facts.",
    }
    manifest["dataset_fingerprint"] = digest(
        {
            "files": files,
            "target_format": TARGET_FORMAT,
            "policy_version": GROUNDED_POLICY_VERSION,
            "source_limit_chars": max_source_chars,
            **({"grounding_negatives": negative_metadata} if negative_metadata else {}),
        }
    )
    if negative_metadata:
        manifest["grounding_negatives"] = negative_metadata
        (output / "grounding-negative-review.json").write_text(json.dumps(negative_audits, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "source-review.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--review-receipt", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-source-chars", type=int, default=8000)
    parser.add_argument("--grounding-negatives", type=Path)
    parser.add_argument("--negative-review-receipt", type=Path)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            compile_dataset(
                args.labels,
                args.snapshots,
                args.output_dir,
                review_receipt=args.review_receipt,
                max_source_chars=args.max_source_chars,
                grounding_negatives=args.grounding_negatives,
                negative_review_receipt=args.negative_review_receipt,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
