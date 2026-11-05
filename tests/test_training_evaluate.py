"""Label accuracy is independent of verbatim source validation."""

import copy
import json

import pytest

from ai_scraper.contracts import InferenceResponse, ScraperError
from training import evaluate
from training.dataset import SCHEMAS, build_case

MODEL = {"id": "example/scraper-1.7b", "revision": "test-revision"}
INFERENCE = {
    "temperature": 0,
    "max_tokens": 4096,
    "enable_thinking": False,
    "response_format": "none",
    "repair_attempts": 0,
}


def _case(schema="custom_v1", number=0):
    return build_case("test", schema, number)[1]


def _row(case, **kwargs):
    return {"case_id": case["case_id"], "candidate": copy.deepcopy(case["expected_candidate"]), **kwargs}


def _score(cases, outputs):
    return evaluate.score_outputs(cases, outputs, model=MODEL, inference=INFERENCE)


def test_gold_across_schemas_and_negative_modes():
    cases = [_case(schema, index) for schema in SCHEMAS for index in range(8)]
    report = _score(cases, [_row(case) for case in cases])
    assert report["case_count"] == 72
    assert report["raw_facts"]["recall"] == 1
    assert report["raw_facts"]["precision"] == 1
    assert report["source_evidence"]["correct_with_source_support"] == report["raw_facts"]["present"]
    assert report["expected_status_matches"] == 72
    assert report["abstention"]["accuracy"] == 1


def test_wrong_fact_can_be_in_source_and_pass_application_validation():
    case = _case()
    row = _row(case)
    candidate = row["candidate"]
    name, *_, themes = candidate["data"]
    # One valid source string is used for the wrong custom business field.
    candidate["data"][name] = candidate["data"][themes][0]
    theme_anchor = copy.deepcopy(
        next(item for item in candidate["evidence"] if item["field_path"] == themes + ".0")
    )
    theme_anchor["field_path"] = name
    candidate["evidence"] = [item for item in candidate["evidence"] if item["field_path"] != name]
    candidate["evidence"].append(theme_anchor)
    report = _score([case], [row])
    assert report["accepted_cases"] == 1
    assert report["source_evidence"]["source_supported_fields"] == 6
    assert report["source_evidence"]["correct_with_source_support"] == 5
    assert report["source_evidence"]["authored_reference_matches"] == 5
    assert report["raw_facts"]["wrong_or_unlabeled"] == 1
    assert report["raw_facts"]["recall"] == 5 / 6
    assert report["required_facts"]["raw"]["precision"] == 5 / 6
    assert report["required_facts"]["accepted"]["wrong_or_unlabeled"] == 1


def test_missing_failure_and_truncation_remain_in_recall():
    cases = [_case(number=index) for index in range(3)]
    expected = sum(len(case["expected_fields"]) for case in cases)
    report = _score(
        cases,
        [
            {"case_id": cases[0]["case_id"], "error_code": "INFERENCE_TIMEOUT"},
            _row(cases[1], finish_reason="length", elapsed_ms=120),
        ],
    )
    assert report["case_count"] == 3
    assert report["missing_outputs"] == 1
    assert report["accepted_cases"] == 0
    assert report["raw_facts"]["present"] == expected
    assert report["raw_facts"]["correct"] == len(cases[1]["expected_fields"])
    assert report["accepted_facts"]["recall"] == 0
    assert report["diagnostics"]["token_limit_cases"] == 1
    assert report["latency"] == {"measured_cases": 1, "p50_ms": 120, "p95_ms": 120, "unknown_cases": 2}
    assert report["tokens"]["output_tokens"] == {"total": None, "measured_cases": 0}


def test_false_zero_and_float_are_distinct_facts():
    case = _case()
    assert len(case["expected_fields"]) == 6
    report = _score([case], [_row(case)])
    assert report["raw_facts"]["correct"] == 6
    row = _row(case)
    integer = next(key for key, value in row["candidate"]["data"].items() if type(value) is int)
    row["candidate"]["data"][integer] = False
    report = _score([case], [row])
    assert report["valid_schema_cases"] == 0
    assert report["raw_facts"]["correct"] == 5
    assert report["accepted_cases"] == 0


def test_array_members_count_as_leaves_and_invalid_evidence_rejects_record():
    case = _case("page_v1")
    row = _row(case)
    assert any(path.startswith("tables.0.rows.0.") for path in case["expected_fields"])
    row["candidate"]["evidence"][0]["excerpt"] = "This quote is absent."
    report = _score([case], [row])
    assert report["raw_facts"]["recall"] == 1
    assert report["source_evidence"]["source_supported_fields"] == report["raw_facts"]["present"] - 1
    assert report["accepted_facts"]["correct"] == 0
    assert report["cases"][0]["validation_codes"] == ["evidence"]


@pytest.mark.parametrize("raw", ['{"data": {}, "data": {}}', '{"data": NaN}', "```json\n{}\n```"])
def test_strict_json_rejects_duplicate_keys_nonfinite_and_markdown(raw):
    case = _case()
    report = _score([case], [{"case_id": case["case_id"], "raw_output": raw}])
    assert report["valid_json_cases"] == 0
    assert report["valid_schema_cases"] == 0
    assert report["raw_facts"]["recall"] == 0


def test_json_arrays_are_json_but_not_candidates():
    case = _case()
    report = _score([case], [{"case_id": case["case_id"], "raw_output": "[]"}])
    assert report["valid_json_cases"] == 1
    assert report["valid_schema_cases"] == 0


def test_repetition_is_only_a_diagnostic():
    case = _case()
    raw = " ".join([str(index) for index in range(32)] * 3)
    report = _score([case], [{"case_id": case["case_id"], "raw_output": raw, "finish_reason": "max_tokens"}])
    assert report["diagnostics"]["repetition_heuristic_cases"] == 1
    assert report["diagnostics"]["token_limit_cases"] == 1
    assert report["raw_facts"]["recall"] == 0


def test_receipt_does_not_embed_outputs_sources_paths_or_unknown_model_keys():
    case = _case()
    secret_marker = "/private/example/secret-value"
    row = _row(case)
    row["candidate"]["data"][secret_marker] = secret_marker
    row["debug"] = secret_marker
    report = _score([case], [row])
    dumped = json.dumps(report)
    assert secret_marker not in dumped
    assert case["snapshot"]["sections"][0]["text"] not in dumped
    assert report["cases"][0]["wrong_paths"][0].startswith("unknown_field_sha256:")
    with pytest.raises(ValueError):
        evaluate.score_outputs([case], [row], model={**MODEL, "api_key": secret_marker}, inference=INFERENCE)
    with pytest.raises(ValueError):
        evaluate.score_outputs([case], [row], model={**MODEL, "id": secret_marker}, inference=INFERENCE)
    with pytest.raises(ValueError):
        evaluate.score_outputs([case], [row], model=MODEL, inference={**INFERENCE, "token": secret_marker})


def test_duplicate_unknown_ids_and_tampered_sources_are_rejected():
    case = _case()
    with pytest.raises(ValueError):
        _score([case], [_row(case), _row(case)])
    with pytest.raises(ValueError):
        _score([case], [{"case_id": "unknown", "error_code": "TIMEOUT"}])
    altered = copy.deepcopy(case)
    altered["snapshot"]["sections"][0]["text"] = "tampered"
    with pytest.raises(ValueError):
        _score([altered], [])


async def test_loopback_driver_checkpoints_and_resumes_without_duplicate_calls(tmp_path, monkeypatch):
    cases = [_case(number=index) for index in range(2)]
    calls = []

    class FakeExtractor:
        def __init__(self, *args, **kwargs):
            self.last_usage = {}

        def for_request(self, request, spec):
            return self

        async def extract(self, snapshot):
            calls.append(snapshot.source_hash)
            self.last_usage = {"input_tokens": 30, "output_tokens": 20}
            if snapshot.source_hash == cases[1]["source_hash"]:
                raise ScraperError("INFERENCE_TIMEOUT", "private exception must never be echoed")
            return InferenceResponse(cases[0]["expected_candidate"])

        async def close(self):
            pass

    monkeypatch.setattr(evaluate, "LocalExtractor", FakeExtractor)
    checkpoint = tmp_path / "private-checkpoint.json"
    rows, report = await evaluate.run_local(
        cases,
        model_id=MODEL["id"],
        revision=MODEL["revision"],
        base_url="http://127.0.0.1:8001/v1",
        output_path=checkpoint,
    )
    assert len(calls) == 2
    assert report["accepted_cases"] == 1
    assert report["diagnostics"]["errors"] == {"INFERENCE_TIMEOUT": 1}
    assert "private exception" not in checkpoint.read_text()
    assert checkpoint.stat().st_mode & 0o777 == 0o600
    resumed, _ = await evaluate.run_local(
        cases,
        model_id=MODEL["id"],
        revision=MODEL["revision"],
        base_url="http://127.0.0.1:8001/v1",
        output_path=checkpoint,
    )
    assert resumed == rows
    assert len(calls) == 2
    with pytest.raises(ValueError):
        await evaluate.run_local(
            cases,
            model_id=MODEL["id"],
            revision="other-revision",
            base_url="http://127.0.0.1:8001/v1",
            output_path=checkpoint,
        )


async def test_local_driver_rejects_remote_endpoints_before_any_call():
    with pytest.raises(ValueError):
        await evaluate.run_local(
            [_case()], model_id=MODEL["id"], revision=MODEL["revision"], base_url="https://example.com/v1"
        )


def test_required_nested_array_metrics_keep_partial_and_missing_cases():
    cases = [_case("listing_v1", number) for number in (0, 1)]
    row = _row(cases[0])
    row["candidate"]["data"]["items"] = row["candidate"]["data"]["items"][:1]
    report = _score(cases, [row])
    # title + five item leaves are required; description is optional.
    assert report["required_facts"]["raw"]["present"] == 12
    assert report["required_facts"]["raw"]["correct"] == 4
    assert report["required_facts"]["raw"]["recall"] == 1 / 3
    assert report["required_facts"]["accepted"]["correct"] == 4
    assert report["optional_facts"]["raw"]["present"] == 2
    assert report["optional_facts"]["raw"]["correct"] == 1
    assert report["by_schema"]["listing_v1"]["case_count"] == 2
    assert report["by_schema"]["listing_v1"]["required_facts"] == report["required_facts"]
    field = report["by_field"]["listing_v1:items.*.name"]
    assert field["raw_facts"]["present"] == 4
    assert field["raw_facts"]["predicted"] == 1
    assert field["raw_facts"]["recall"] == 0.25
    assert len(report["by_domain"]) == 2
    assert sum(item["required_facts"]["raw"]["present"] for item in report["by_domain"].values()) == 12
    assert sorted(item["required_facts"]["raw"]["correct"] for item in report["by_domain"].values()) == [0, 4]


def test_required_metrics_distinguish_source_supported_wrong_from_rejected_record():
    case = _case("article_v1")
    row = _row(case)
    row["candidate"]["evidence"][0]["excerpt"] = "Absent quote"
    report = _score([case], [row])
    assert report["required_facts"]["raw"]["recall"] == 1
    assert report["required_facts"]["accepted"]["recall"] == 0
    assert report["optional_facts"]["raw"]["present"] == 3
    assert report["by_schema"]["article_v1"]["accepted_facts"]["correct"] == 0
    assert report["by_field"]["article_v1:text.*"]["required_facts"]["raw"]["present"] == 1


@pytest.mark.parametrize("failure", ["timeout", "truncated", "invalid_schema", "invalid_json"])
def test_execution_failures_never_receive_expected_negative_status_credit(failure):
    case = _case("article_v1", 6)
    assert case["expected_status"] == "failed"
    assert _score([case], [_row(case)])["expected_status_matches"] == 1
    if failure == "timeout":
        row = {"case_id": case["case_id"], "error_code": "INFERENCE_TIMEOUT"}
    elif failure == "truncated":
        row = _row(case, finish_reason="length")
    elif failure == "invalid_schema":
        row = _row(case)
        row["candidate"]["data"]["title"] = False
    else:
        row = {"case_id": case["case_id"], "raw_output": "{"}
    report = _score([case], [row])
    assert report["expected_status_matches"] == 0
    assert report["raw_facts"]["present"] == len(case["expected_fields"])
    assert report["accepted_facts"]["correct"] == 0
    if failure == "truncated":
        assert report["raw_facts"]["correct"] == len(case["expected_fields"])
