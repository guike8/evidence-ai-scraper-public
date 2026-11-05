from pathlib import Path

import pytest

from ai_scraper.evaluation import evaluate_manifest, score_predictions


def label(name="Product", price="10.00", currency="USD"):
    return {
        "url": "https://shop.test/product",
        "expected": {"name": name, "price": price, "currency": currency},
    }


def outcome(status="success", name="Product", price="10.00", currency="USD"):
    return {
        "status": status,
        "data": {"name": name, "price": price, "currency": currency},
        "validation_passed": True,
        "retrieval_usable": True,
        "latency_ms": 1.0,
    }


def test_fixture_report_is_honest_about_mode_and_measured_latency():
    report = evaluate_manifest(Path(__file__).parent / "fixtures/products.json")
    assert report["fixture_count"] == 27
    assert report["all"] == {
        "true_positive": 63,
        "predicted": 63,
        "present": 63,
        "false_positive": 0,
        "missed": 0,
        "precision": 1.0,
        "recall": 1.0,
    }
    assert report["retrieval_coverage"] == {"usable": 25, "total": 27}
    assert report["gates"]["passed"]
    assert report["p95_latency_ms"] > 0
    assert report["latency_samples"] == 27
    assert report["mode"] == "synthetic_deterministic_offline"
    assert report["inference_provider"] is None
    assert report["measured_cost_usd"] == 0
    assert report["cost_estimate_usd"] is None
    assert report["by_field"]["name"]["true_positive"] == 23
    assert report["by_domain"]["apparel.test"]["true_positive"] == 14


@pytest.mark.parametrize("status", ["failed", "pending"])
def test_failure_and_timeout_are_misses_not_filtered_out(status):
    report = score_predictions([label(), label()], [outcome(), {"status": status, "data": None}])
    assert report["all"]["present"] == 6
    assert report["all"]["true_positive"] == 3
    assert report["all"]["recall"] == 0.5
    assert report["all"]["precision"] == 1
    assert not report["gates"]["passed"]


def test_wrong_values_count_false_positives_and_misses():
    report = score_predictions([label()], [outcome(price="11.00")])
    assert report["all"]["true_positive"] == 2
    assert report["all"]["predicted"] == report["all"]["present"] == 3
    assert report["all"]["false_positive"] == report["all"]["missed"] == 1
    assert report["unsupported_populated_facts"] == 1
    assert not report["gates"]["zero_unsupported_facts"]


def test_null_ground_truth_is_not_a_recall_target_but_guessed_fact_is_false_positive():
    report = score_predictions([label(currency=None)], [outcome()])
    assert report["by_field"]["currency"]["present"] == 0
    assert report["by_field"]["currency"]["predicted"] == 1
    assert report["by_field"]["currency"]["precision"] == 0
    assert report["by_field"]["currency"]["recall"] is None
    assert report["unsupported_populated_facts"] == 1


def test_all_optional_populated_facts_are_compared_against_labels():
    result = outcome()
    result["data"]["sku"] = "INVENTED"
    report = score_predictions([label()], [result])
    assert report["all"]["precision"] == 1
    assert report["unsupported_populated_facts"] == 1
    assert not report["gates"]["passed"]


def test_missing_outcome_rejected_and_success_validation_not_assumed():
    with pytest.raises(ValueError, match="Every labeled case"):
        score_predictions([label()], [])
    result = outcome()
    del result["validation_passed"]
    del result["latency_ms"]
    report = score_predictions([label()], [result])
    assert report["p95_latency_ms"] is None
    assert not report["gates"]["all_success_validated"]


def test_price_equivalent_decimal_strings_are_same_fact():
    report = score_predictions([label()], [outcome(price="10.0")])
    assert report["all"]["true_positive"] == 3
    assert report["unsupported_populated_facts"] == 0
