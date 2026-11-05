"""Offline parser/selector components, not end-to-end scraper benchmarks.

Run this file with each backend's isolated Python environment. Every backend
receives the same authored HTML and manually predefined CSS schema. No HTTP,
browser, model, business validation or source-evidence work is measured here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = {
    "name": "Controlled main content",
    "baseSelector": "main",
    "fields": [
        {"name": "heading", "selector": "h1", "type": "text"},
        {
            "name": "paragraphs",
            "selector": "p",
            "type": "list",
            "fields": [{"name": "text", "type": "text"}],
        },
        {
            "name": "links",
            "selector": "a[href]",
            "type": "list",
            "fields": [
                {"name": "label", "type": "text"},
                {"name": "href", "type": "attribute", "attribute": "href"},
            ],
        },
        {
            "name": "cells",
            "selector": "th,td",
            "type": "list",
            "fields": [{"name": "text", "type": "text"}],
        },
    ],
}


def fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def fixtures() -> list[dict[str, Any]]:
    """Authored labels; none of these inputs is a fetched real webpage."""
    heading = "Main research record"
    paragraph = "One exact paragraph about useful public facts."
    base_expected = {
        "heading": heading,
        "paragraphs": [{"text": paragraph}],
        "links": [{"label": "Reference page", "href": "/reference"}],
        "cells": [{"text": "Name"}, {"text": "Value"}, {"text": "Ada"}, {"text": "37"}],
    }
    main = (
        f"<main><h1>{heading}</h1><p>{paragraph}</p>"
        '<a href="/reference">Reference page</a>'
        "<table><tr><th>Name</th><th>Value</th></tr><tr><td>Ada</td><td>37</td></tr></table></main>"
    )
    cases = [{"id": "small", "html": main, "expected": base_expected}]
    for count in (1000, 2100):
        navigation = "".join(f"<p>Unrelated navigation destination {index}.</p>" for index in range(count))
        cases.append(
            {"id": f"navigation-{count}", "html": f"<nav>{navigation}</nav>{main}", "expected": base_expected}
        )
    inline = "<span>" * 64 + "exact" + "</span>" * 64
    cases.append({"id": "nested-inline-64", "html": main.replace("exact", inline), "expected": base_expected})
    rows = "".join(
        "<tr>" + "".join(f"<td>r{row}c{column}</td>" for column in range(10)) + "</tr>" for row in range(20)
    )
    cases.append(
        {
            "id": "table-20x10",
            "html": f"<main><h1>{heading}</h1><p>{paragraph}</p><table>{rows}</table></main>",
            "expected": {
                "heading": heading,
                "paragraphs": [{"text": paragraph}],
                "links": [],
                "cells": [{"text": f"r{row}c{column}"} for row in range(20) for column in range(10)],
            },
        }
    )
    links = "".join(f'<a href="/record/{index}">Record {index}</a>' for index in range(100))
    cases.append(
        {
            "id": "links-100",
            "html": f"<main><h1>{heading}</h1><p>{paragraph}</p>{links}</main>",
            "expected": {
                "heading": heading,
                "paragraphs": [{"text": paragraph}],
                "links": [{"label": f"Record {index}", "href": f"/record/{index}"} for index in range(100)],
                "cells": [],
            },
        }
    )
    return cases


def disable_python_network() -> list[str]:
    """Block common Python socket entry points; this is not OS isolation."""
    attempts: list[str] = []

    def blocked(name):
        def reject(*args, **kwargs):
            attempts.append(name)
            raise OSError("Network disabled for offline selector component benchmark")

        return reject

    socket.socket.connect = blocked("socket.connect")
    socket.socket.connect_ex = blocked("socket.connect_ex")
    socket.create_connection = blocked("socket.create_connection")
    socket.getaddrinfo = blocked("socket.getaddrinfo")
    return attempts


def load_backend(name: str):
    if name == "beautifulsoup":
        from bs4 import BeautifulSoup

        def extract(html):
            root = BeautifulSoup(html, "html.parser").select_one("main")
            return {
                "heading": root.select_one("h1").get_text(" ", strip=True),
                "paragraphs": [{"text": node.get_text(" ", strip=True)} for node in root.select("p")],
                "links": [
                    {"label": node.get_text(" ", strip=True), "href": node["href"]}
                    for node in root.select("a[href]")
                ],
                "cells": [{"text": node.get_text(" ", strip=True)} for node in root.select("th,td")],
            }

        return extract, "beautifulsoup4", {"parser": "html.parser"}
    if name == "scrapling":
        from scrapling import Selector

        def extract(html):
            root = Selector(html).css("main")[0]

            def text(node):
                return str(node.get_all_text(separator=" ", strip=True))

            return {
                "heading": text(root.css("h1")[0]),
                "paragraphs": [{"text": text(node)} for node in root.css("p")],
                "links": [{"label": text(node), "href": node.attrib["href"]} for node in root.css("a[href]")],
                "cells": [{"text": text(node)} for node in root.css("th,td")],
            }

        return extract, "scrapling", {"adaptive": False, "auto_save": False}
    from crawl4ai.extraction_strategy import JsonLxmlExtractionStrategy

    # Disable cross-document result caching. Selector compilation remains warm,
    # as it does in the other backends. The HTML itself is reparsed every call.
    strategy = JsonLxmlExtractionStrategy(SCHEMA, use_caching=False, verbose=False)

    def extract(html):
        result = strategy.extract("https://component.example.test/record", html)
        if len(result) != 1:
            raise ValueError("Expected exactly one main element")
        return result[0]

    return extract, "crawl4ai", {"strategy": "JsonLxmlExtractionStrategy", "use_caching": False}


def mismatch_paths(expected: Any, actual: Any, path: str = "") -> list[str]:
    if type(expected) is not type(actual):
        return [path or "$type"]
    if isinstance(expected, dict):
        paths = []
        for key in sorted(set(expected) | set(actual)):
            child = f"{path}.{key}" if path else key
            if key not in expected or key not in actual:
                paths.append(child)
            else:
                paths.extend(mismatch_paths(expected[key], actual[key], child))
        return paths
    if isinstance(expected, list):
        paths = [f"{path}.length"] if len(expected) != len(actual) else []
        for index, (left, right) in enumerate(zip(expected, actual)):
            paths.extend(mismatch_paths(left, right, f"{path}.{index}"))
        return paths
    return [] if expected == actual else [path]


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("beautifulsoup", "scrapling", "crawl4ai"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--warmups", type=int, default=3, choices=range(1, 11))
    parser.add_argument("--iterations", type=int, default=25, choices=range(5, 101))
    args = parser.parse_args()
    started = time.perf_counter()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    os.environ["CRAWL4_AI_BASE_DIRECTORY"] = str(args.state_dir.resolve())
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["SCRAPEGRAPHAI_TELEMETRY_ENABLED"] = "false"
    attempts = disable_python_network()
    import_started = time.perf_counter()
    cases = fixtures()
    report: dict[str, Any] = {
        "format": "selector-component-receipt-v1",
        "checked_at": datetime.now(UTC).isoformat(),
        "candidate": args.backend,
        "track": "offline_predefined_css_component",
        "corpus_id": "synthetic-selector-components-v1",
        "corpus_hash": fingerprint(cases),
        "selector_schema": SCHEMA,
        "selector_schema_hash": fingerprint(SCHEMA),
        "runtime": f"Python {platform.python_version()}",
        "hardware": {"system": platform.system(), "machine": platform.machine()},
        "warmups_per_case": args.warmups,
        "iterations_per_correct_case": args.iterations,
        "results": [],
        "limitations": [
            "Authored synthetic HTML with one explicit main element and manually predefined selectors.",
            "Components only: no fetching, JavaScript, browser, model, business validation or evidence verification.",
            "Warm timings include reparsing and selection; imports and backend construction are measured separately.",
            "Python socket entry points are blocked; this is not OS-level network isolation.",
            "Process RSS includes backend imports and all previous cases; no isolated per-case RSS is claimed.",
        ],
    }
    try:
        extract, package, configuration = load_backend(args.backend)
        report["import_and_setup_ms"] = (time.perf_counter() - import_started) * 1000
        report["package"] = package
        report["version"] = importlib.metadata.version(package)
        report["configuration"] = configuration
        for case in cases:
            row: dict[str, Any] = {
                "id": case["id"],
                "input_sha256": hashlib.sha256(case["html"].encode()).hexdigest(),
                "input_bytes": len(case["html"].encode()),
                "expected_hash": fingerprint(case["expected"]),
            }
            try:
                warm_correct = True
                for _ in range(args.warmups):
                    output = extract(case["html"])
                    warm_correct = warm_correct and output == case["expected"]
                row["correct"] = warm_correct
                row["output_hash"] = fingerprint(output)
                row["mismatch_paths"] = mismatch_paths(case["expected"], output)[:30]
                if warm_correct:
                    timings = []
                    all_correct = True
                    for _ in range(args.iterations):
                        sample_started = time.perf_counter()
                        output = extract(case["html"])
                        timings.append((time.perf_counter() - sample_started) * 1000)
                        all_correct = all_correct and output == case["expected"]
                    row["correct"] = all_correct
                    row["timing_samples_ms"] = timings
                    row["p50_ms"] = statistics.median(timings)
                    row["p95_ms"] = percentile(timings, 0.95)
                    row["min_ms"] = min(timings)
                    row["max_ms"] = max(timings)
                else:
                    row["timing_samples_ms"] = []
                    row["timing_note"] = "Incorrect output is not ranked by speed."
            except Exception as error:  # noqa: BLE001 -- retain third-party failure in the receipt
                row.update(correct=False, error_type=type(error).__name__, timing_samples_ms=[])
            report["results"].append(row)
    except Exception as error:  # noqa: BLE001 -- setup failures must remain measurable outcomes
        report["setup_error_type"] = type(error).__name__
        report["import_and_setup_ms"] = (time.perf_counter() - import_started) * 1000
    report["blocked_python_network_attempts"] = len(attempts)
    report["blocked_python_network_entry_points"] = sorted(set(attempts))
    report["python_socket_network_calls_completed"] = 0
    report["wall_ms"] = (time.perf_counter() - started) * 1000
    try:
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report["peak_process_rss_bytes"] = rss if sys.platform == "darwin" else rss * 1024
    except ImportError:
        report["peak_process_rss_bytes"] = None
    report["all_outputs_correct"] = len(report["results"]) == len(cases) and all(
        row["correct"] for row in report["results"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "candidate": args.backend,
                "all_outputs_correct": report["all_outputs_correct"],
                "blocked_network_attempts": len(attempts),
                "cases": len(report["results"]),
            }
        )
    )
    return 0 if report["all_outputs_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
