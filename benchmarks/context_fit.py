"""Measure inference context fit without calling a model or retrieving a URL.

Execute this same file with PYTHONPATH selecting each application revision.
Character counts describe the serialized source payload, not prompt token usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

from ai_scraper.contracts import ScraperError
from ai_scraper.extraction.huggingface import HuggingFaceExtractor
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest


def measure(manifest: dict, limit: int, repetitions: int) -> list[dict]:
    rows = []
    for case in manifest["cases"]:
        request = ExtractionRequest(url=case["url"], **case.get("request", {}))
        spec = get_schema(request)
        snapshot = normalize_record(case["record"], request)
        adapter = HuggingFaceExtractor(
            "local", "measurement-only", client=object(), max_source_chars=limit
        ).for_request(request, spec)
        durations = []
        content = None
        error = None
        for _ in range(repetitions):
            started = perf_counter()
            try:
                content = adapter._source_content(snapshot)
            except ScraperError as exc:
                error = exc.error.code
            durations.append((perf_counter() - started) * 1000)
        full_payload = json.dumps(
            {
                "final_url": snapshot.final_url,
                "sections": [{"id": section.id, "text": section.text} for section in snapshot.sections],
                "ld_json": snapshot.ld_json,
                "excluded_sections": snapshot.excluded_sections,
            },
            ensure_ascii=False,
        )
        descriptor = getattr(adapter, "last_source_view", None)
        rows.append(
            {
                "case_id": case["id"],
                "schema_id": spec.schema_id,
                "source_hash": snapshot.source_hash,
                "original_chars": len(full_payload),
                "view_chars": len(content) if content is not None else None,
                "fits": content is not None,
                "error_code": error,
                "view_hash": hashlib.sha256(content.encode()).hexdigest() if content is not None else None,
                "scope": descriptor["scope"] if descriptor else "full_source",
                "coverage": descriptor["coverage"] if descriptor else "full",
                "included_sections": len(descriptor["included_section_ids"])
                if descriptor
                else len(snapshot.sections),
                "excluded_sections": len(descriptor["excluded_section_ids"]) if descriptor else 0,
                "planning_median_ms": statistics.median(durations),
                "planning_runs": repetitions,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--limit", type=int, default=48_000)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 < args.limit <= 10_000_000 or not 0 < args.repetitions <= 100:
        parser.error("limit and repetitions must be positive and bounded")
    manifest = json.loads(args.manifest.read_text())
    report = {
        "candidate": args.candidate,
        "mode": "context_planning_offline",
        "runtime": platform.python_version(),
        "hardware": {"system": platform.system(), "machine": platform.machine()},
        "corpus_id": manifest["corpus_id"],
        "input_character_limit": args.limit,
        "network_calls": 0,
        "model_calls": 0,
        "token_usage": None,
        "limitations": "Authored fixtures and serialized-source fit only; no model latency or factual accuracy measurement",
        "results": measure(manifest, args.limit, args.repetitions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidate": args.candidate, "fits": sum(row["fits"] for row in report["results"])}))


if __name__ == "__main__":
    main()
