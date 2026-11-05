"""Explicit-label scoring. Failed and pending outcomes remain in recall denominators."""

from __future__ import annotations

import json
import math
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib.parse import urlsplit

from .extraction.deterministic import deterministic_extract
from .normalize import detect_content_problem, normalize_record
from .schemas import REQUIRED_FIELDS, ExtractionRequest, ExtractionResult, ProductV1
from .validate import populated_fields, validate_candidate


def _equal(path: str, predicted: Any, expected: Any) -> bool:
    if predicted is None or expected is None:
        return predicted is expected
    if path == "price" and isinstance(predicted, str) and isinstance(expected, str):
        try:
            return Decimal(predicted) == Decimal(expected)
        except InvalidOperation:
            return False
    return type(predicted) is type(expected) and predicted == expected


def _counter() -> dict[str, int | float | None]:
    return {"true_positive": 0, "predicted": 0, "present": 0}


def _finish(counter: dict) -> dict:
    result = dict(counter)
    result["false_positive"] = result["predicted"] - result["true_positive"]
    result["missed"] = result["present"] - result["true_positive"]
    result["precision"] = result["true_positive"] / result["predicted"] if result["predicted"] else None
    result["recall"] = result["true_positive"] / result["present"] if result["present"] else None
    return result


def score_predictions(
    cases: list[dict[str, Any]], results: list[ExtractionResult | dict[str, Any]]
) -> dict[str, Any]:
    """Score outcomes at the caller's completion deadline, without dropping failures.

    Each case has url and expected (a complete or partial ProductV1 label). Each
    result has status and data; optional retrieval_usable, latency_ms, and
    validation_passed carry measured execution evidence. No source support or
    latency is assumed when those execution fields are absent.
    """
    if len(cases) != len(results):
        raise ValueError("Every labeled case must have exactly one outcome, including pending or failed")
    totals = _counter()
    by_field = {field: _counter() for field in REQUIRED_FIELDS}
    by_domain: dict[str, dict] = {}
    statuses: Counter = Counter()
    unsupported = 0
    usable = 0
    latency: list[float] = []
    all_success_validated = True
    for case, outcome in zip(cases, results, strict=True):
        result = outcome.model_dump() if isinstance(outcome, ExtractionResult) else outcome
        status = result.get("status", "failed")
        statuses[status] += 1
        expected = ProductV1.model_validate(case["expected"]).model_dump()
        raw_data = result.get("data")
        data = raw_data if status in {"success", "partial"} and isinstance(raw_data, dict) else {}
        truth_fields = populated_fields(expected)
        actual_fields = populated_fields(data)
        for path, value in actual_fields.items():
            unsupported += not _equal(path, value, truth_fields.get(path))
        source_hash = (result.get("provenance") or {}).get("source_hash")
        usable += bool(result.get("retrieval_usable", bool(source_hash)))
        elapsed = result.get("latency_ms")
        if (
            isinstance(elapsed, (float, int))
            and not isinstance(elapsed, bool)
            and math.isfinite(elapsed)
            and elapsed >= 0
        ):
            latency.append(float(elapsed))
        if status == "success" and result.get("validation_passed") is not True:
            all_success_validated = False
        domain = urlsplit(case["url"]).hostname or "unknown"
        domain_counts = by_domain.setdefault(
            domain, {"aggregate": _counter(), "by_field": {field: _counter() for field in REQUIRED_FIELDS}}
        )
        for field in REQUIRED_FIELDS:
            present = expected[field] is not None
            predicted = data.get(field) is not None
            correct = present and predicted and _equal(field, data[field], expected[field])
            for counter in (
                totals,
                by_field[field],
                domain_counts["aggregate"],
                domain_counts["by_field"][field],
            ):
                counter["present"] += present
                counter["predicted"] += predicted
                counter["true_positive"] += correct
    total_metrics = _finish(totals)
    field_metrics = {field: _finish(value) for field, value in by_field.items()}
    domain_metrics = {
        domain: {
            **_finish(value["aggregate"]),
            "by_field": {field: _finish(counts) for field, counts in value["by_field"].items()},
        }
        for domain, value in by_domain.items()
    }
    sorted_latency = sorted(latency)
    p95 = sorted_latency[math.ceil(0.95 * len(sorted_latency)) - 1] if sorted_latency else None
    precision_ok = total_metrics["precision"] is not None and total_metrics["precision"] >= 0.95
    recall_ok = total_metrics["recall"] is not None and total_metrics["recall"] >= 0.95
    return {
        "fixture_count": len(cases),
        "all": total_metrics,
        "by_field": field_metrics,
        "by_domain": domain_metrics,
        "retrieval_coverage": {"usable": usable, "total": len(cases)},
        "statuses": dict(statuses),
        "unsupported_populated_facts": unsupported,
        "p95_latency_ms": p95,
        "latency_samples": len(latency),
        "cost_estimate_usd": None,
        "pricing_basis": None,
        "gates": {
            "precision_at_least_95pct": precision_ok,
            "recall_at_least_95pct": recall_ok,
            "zero_unsupported_facts": unsupported == 0,
            "all_success_validated": all_success_validated,
            "passed": precision_ok and recall_ok and unsupported == 0 and all_success_validated,
        },
    }


def evaluate_manifest(path: Path) -> dict[str, Any]:
    """Run local deterministic extraction; never fetch or call a paid model."""
    manifest = json.loads(path.read_text())
    cases = manifest["cases"]
    results = []
    for case in cases:
        start = perf_counter()
        request = ExtractionRequest(url=case["url"], locale=case.get("locale"))
        snapshot = normalize_record(case["record"], request)
        problem = detect_content_problem(snapshot)
        if problem:
            result = {
                "status": "failed",
                "data": None,
                "error": {"code": problem},
                "retrieval_usable": False,
                "validation_passed": False,
            }
        else:
            candidate = deterministic_extract(snapshot, request.locale)
            assessment = validate_candidate(candidate, snapshot, request.locale)
            result = {
                "status": assessment.status,
                "data": assessment.data.model_dump() if assessment.data else None,
                "retrieval_usable": True,
                "validation_passed": not assessment.errors,
            }
        result["latency_ms"] = (perf_counter() - start) * 1000
        results.append(result)
    report = score_predictions(cases, results)
    report.update(
        mode="synthetic_deterministic_offline",
        corpus_version=manifest.get("corpus_version"),
        corpus_provenance=manifest.get("provenance"),
        inference_provider=None,
        inference_model=None,
        measured_cost_usd=0.0,
        cost_note="Local fixture replay only; excludes external retrieval and inference.",
        completion_deadline_note="All local fixtures terminal; no pending jobs were omitted.",
    )
    return report
