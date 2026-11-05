"""Opt-in authored citation curriculum, separate from the reproducible v1 pilot.

No model output, private snapshot, real-web evaluation or benchmark label is
read here. The original generator remains unchanged. Only train messages may
shuffle section presentation; immutable snapshots retain normalizer IDs/order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from ai_scraper.extraction.generic import populated_fields
from ai_scraper.extraction.huggingface import HuggingFaceExtractor, request_instruction
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest

from .dataset import (
    LAYOUTS,
    ROW_TEMPLATES,
    SCHEMAS,
    SPLITS,
    _apply_mode,
    _facts,
    _render,
    _write,
    bind_evidence,
    canonical,
    check_disjoint,
    digest,
    validate_gold,
)

VERSION = "synthetic-citation-v2"
DEFAULT_SEED = 2026110603
PROVENANCE = {
    "kind": "synthetic_authored_citation_templates",
    "review": "automatically validated; not independently reviewed real-web data",
    "label_origin": "authored facts before rendering; no model-generated labels",
    "external_sources": [],
    "benchmark_training_rows": 0,
    "private_source_rows": 0,
    "license": "MIT",
}


def _without_identity_mentions(value, identity: str):
    """Do not contradict a critical-null label with its name in another field."""
    if isinstance(value, str):
        return value.replace(identity, "This unnamed record") if identity else value
    if isinstance(value, list):
        return [_without_identity_mentions(item, identity) for item in value]
    if isinstance(value, dict):
        return {key: _without_identity_mentions(item, identity) for key, item in value.items()}
    return value


def build_case(split: str, schema_id: str, number: int, seed: int = DEFAULT_SEED) -> tuple[dict, dict]:
    if split not in SPLITS or schema_id not in SCHEMAS or type(number) is not int or number < 0:
        raise ValueError("Unsupported split, schema, or case index")
    facts, request_dict, token, relative = _facts(schema_id, split, number, seed)
    # Positive short descriptions are separately authored and labeled. Most
    # products have only long descriptions; copying those into short is wrong.
    if schema_id == "product_v1" and number % 4 == 0:
        facts["short_description"] = f"Compact reference edition {token}."
    if schema_id == "event_v1" and number % 4 == 2:
        facts["start_date"] = None  # End date never establishes a start date.
    if schema_id == "product_v1" and number % 11 == 0:
        facts["long_description"] = None
    request = ExtractionRequest.model_validate(request_dict)
    spec = get_schema(request)
    gold, mode, warnings = _apply_mode(facts, schema_id, number, split)
    if mode == "missing_critical":
        identity = facts.get("name") or facts.get("title")
        if schema_id == "custom_v1":
            identity = facts.get(request.fields[0].name)
        gold = _without_identity_mentions(gold, identity or "")
    gold = spec.data_model.model_validate(gold).model_dump()
    record, template_id = _render(schema_id, split, number, gold, facts, request, token, relative, mode)
    # Seventy percent of each balanced split has no JSON-LD. The remainder uses
    # the original authored native representation (Product uses Schema.org).
    without_ld = number % 10 < 7
    if without_ld:
        record["ld_json"] = []
    decoration = number % 3
    if decoration:
        prefix = (
            '<nav><a href="/help">Help</a><span>Reference</span></nav>'
            if decoration == 1
            else '<aside><span>Reference</span><span>Reference</span><a href="/index">Index</a></aside>'
        )
        record["page_html"] = record["page_html"].replace("<body>", "<body>" + prefix, 1)
    if number % 5 == 0 and mode not in {"ambiguous", "missing_critical"}:
        # Duplicate labels create sparse included IDs after real source-view
        # deduplication. They do not invent additional business facts.
        record["page_html"] = record["page_html"].replace(
            "</body>", "<footer><span>Reference</span><span>Reference</span></footer></body>", 1
        )
    snapshot = normalize_record(record, request, retrieval_method="synthetic_authored_fixture")
    snapshot.retrieved_at = datetime(2026, 1, 1, tzinfo=UTC)
    adapter = HuggingFaceExtractor("synthetic-offline", "synthetic-offline", client=object())
    adapter = adapter.for_request(request, spec)
    source = adapter._source_content(snapshot)
    source_payload = json.loads(source)
    included_ids = {section["id"] for section in source_payload["sections"]}
    binding_snapshot = snapshot.model_copy(
        update={"sections": [s for s in snapshot.sections if s.id in included_ids]}
    )
    candidate = {
        "data": gold,
        "evidence": bind_evidence(
            gold,
            binding_snapshot,
            relative_fields=relative,
            prefer_pointers=not without_ld,
            schema_id=schema_id,
        ),
        "warnings": warnings,
    }
    assessment = validate_gold(candidate, snapshot, request)
    missing = [name for name in spec.required_fields if gold.get(name) in (None, [], "")]
    expected_status = (
        "failed"
        if any(name in missing for name in spec.critical_fields)
        else "partial"
        if missing
        else "success"
    )
    if assessment.status != expected_status:
        raise ValueError("Production validation disagrees with authored completeness")
    # Copying actual IDs is necessary when source order is deliberately
    # permuted. Only train presentation is augmented; replay cases keep exact
    # production rendering and normalized snapshot identity.
    message_source = source
    shuffled = split == "train" and number % 2 == 0
    if shuffled:
        random.Random(f"{seed}:{schema_id}:{number}:sections").shuffle(source_payload["sections"])
        message_source = json.dumps(source_payload, ensure_ascii=False)
    target = json.dumps(
        spec.candidate_model.model_validate(candidate).model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    row = {
        "messages": [
            {"role": "system", "content": request_instruction(request, spec)},
            {"role": "user", "content": "UNTRUSTED_SOURCE_JSON\n" + message_source},
            {"role": "assistant", "content": target},
        ]
    }
    family = number % len(LAYOUTS[split])
    case = {
        "case_id": f"citation-v2-{split}-{schema_id}-{number:05d}",
        "split": split,
        "schema_id": schema_id,
        "mode": mode,
        "template_id": template_id,
        "template_fingerprint": digest([LAYOUTS[split][family], ROW_TEMPLATES[split][family]]),
        "domain": urlsplit(request.url).hostname,
        "entity_group": token,
        "request": request.model_dump(mode="json"),
        "snapshot": snapshot.model_dump(mode="json"),
        "source_hash": snapshot.source_hash,
        "normalized_source_hash": digest(source),
        "expected_candidate": candidate,
        "expected_status": expected_status,
        "expected_validation_errors": assessment.errors,
        "expected_fields": populated_fields(gold),
        "schema_fingerprint": spec.fingerprint,
        "candidate_json_schema": spec.candidate_model.model_json_schema(),
        "source_view": adapter.last_source_view,
        "provenance": PROVENANCE,
        "citation_curriculum": {
            "version": VERSION,
            "without_json_ld": without_ld,
            "train_section_presentation_shuffled": shuffled,
            "training_source_hash": digest(message_source),
            "source_decoration": decoration,
        },
    }
    return row, case


def generate_dataset(
    output_dir: str | Path,
    *,
    seed: int = DEFAULT_SEED,
    train_per_schema: int = 200,
    valid_per_schema: int = 16,
    test_per_schema: int = 12,
) -> dict:
    output = Path(output_dir).expanduser().resolve()
    repository = Path(__file__).resolve().parents[1]
    if output == repository or repository in output.parents:
        raise ValueError("Generated data must be outside the public source checkout")
    sizes = dict(zip(SPLITS, (train_per_schema, valid_per_schema, test_per_schema), strict=True))
    if any(type(count) is not int or not 1 <= count <= 5000 for count in sizes.values()):
        raise ValueError("Per-schema counts must be integers between 1 and 5000")
    if output.exists() and any(output.iterdir()):
        try:
            previous = json.loads((output / "manifest.json").read_text())
        except (OSError, ValueError):
            raise ValueError("Refusing to overwrite a nonempty unrelated directory") from None
        if (
            previous.get("generator") != VERSION
            or previous.get("seed") != seed
            or previous.get("per_schema") != sizes
        ):
            raise ValueError("Existing dataset configuration differs; choose a new output directory")
    rows, cases = {}, {}
    for split, count in sizes.items():
        pairs = [build_case(split, schema, n, seed) for schema in SCHEMAS for n in range(count)]
        random.Random(f"{seed}:{split}").shuffle(pairs)
        rows[split], cases[split] = [p[0] for p in pairs], [p[1] for p in pairs]
    checks = check_disjoint(cases)
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    files = {}
    for split in SPLITS:
        for suffix, content in (("jsonl", rows[split]), ("cases.jsonl", cases[split])):
            name = f"{split}.{suffix}"
            serialized = "".join(canonical(item) + "\n" for item in content)
            _write(output / name, serialized)
            files[name] = {"sha256": hashlib.sha256(serialized.encode()).hexdigest(), "rows": len(content)}
    manifest = {
        "generator": VERSION,
        "seed": seed,
        "per_schema": sizes,
        "target_order": "schema",
        "provenance": PROVENANCE,
        "format": "MLX-LM messages; plain JSON assistant; schema property order",
        "chat_template_kwargs": {"enable_thinking": False},
        "counts": {split: len(rows[split]) for split in SPLITS},
        "counts_by_schema": {s: dict(Counter(c["schema_id"] for c in cases[s])) for s in SPLITS},
        "counts_by_mode": {s: dict(Counter(c["mode"] for c in cases[s])) for s in SPLITS},
        "counts_by_status": {s: dict(Counter(c["expected_status"] for c in cases[s])) for s in SPLITS},
        "citation_variants": {
            s: dict(
                Counter(
                    "no_ld" if c["citation_curriculum"]["without_json_ld"] else "with_ld" for c in cases[s]
                )
            )
            for s in SPLITS
        },
        "quality_checks": {
            **checks,
            "all_candidates_pydantic_valid": True,
            "all_populated_leaves_source_verified": True,
            "gold_references_in_actual_source_view": True,
            "failed_targets_only_expected_completeness": True,
            "application_provenance_not_in_assistant_targets": True,
        },
        "files": files,
        "dataset_fingerprint": digest(files),
        "generator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "limitations": [
            "Synthetic authored curriculum; not representative real-web accuracy.",
            "Source templates and simple task families are reused from v1, with new entities/seed.",
            "Train-only section-order augmentation differs from production serialization order; IDs/text remain exact.",
            "Critical-null narratives remove explicit identity mentions to avoid contradictory supervision.",
        ],
    }
    _write(output / "manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--train-per-schema", type=int, default=200)
    parser.add_argument("--valid-per-schema", type=int, default=16)
    parser.add_argument("--test-per-schema", type=int, default=12)
    result = generate_dataset(**vars(parser.parse_args()))
    print(json.dumps({"counts": result["counts"], "fingerprint": result["dataset_fingerprint"]}))


if __name__ == "__main__":
    main()
