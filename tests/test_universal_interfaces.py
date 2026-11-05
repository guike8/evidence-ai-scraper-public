from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime

import httpx
import pytest

from ai_scraper.cli import main
from ai_scraper.contracts import ScraperError
from ai_scraper.extraction.huggingface import HuggingFaceExtractor
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.mcp_server import build_server
from ai_scraper.registry import get_schema
from ai_scraper.schemas import CustomField, ExtractionRequest, ExtractionResult, Provenance, SourceSnapshot


def custom(name, **kwargs):
    return get_schema(
        ExtractionRequest(
            url="https://example.com",
            schema_id="custom_v1",
            fields=[CustomField(name=name, **kwargs)],
        )
    )


def response(payload, tokens=3):
    return {
        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": tokens, "completion_tokens": 2},
    }


@pytest.mark.asyncio
async def test_concurrent_custom_contracts_have_isolated_grammars_prompts_usage_and_ownership():
    calls = []

    class Client:
        closed = False

        async def chat_completion(self, **kwargs):
            calls.append(kwargs)
            definitions = kwargs["response_format"]["json_schema"]["schema"]["$defs"]
            name = next(iter(definitions["CustomV1"]["properties"]))
            await asyncio.sleep(0 if name == "second" else 0.01)
            return response({"data": {name: name}, "evidence": [], "warnings": []}, len(name))

        async def close(self):
            self.closed = True

    client = Client()
    base = HuggingFaceExtractor("test", "test", client=client)
    original = base.response_format
    first = base.for_schema(custom("first"), "Find the first fact.")
    second = base.for_schema(custom("second"), "Find the second fact.")
    snapshot = SourceSnapshot(
        requested_url="https://example.com",
        final_url="https://example.com",
        retrieved_at=datetime.now(UTC),
        raw_record={},
        sections=[],
        source_hash="a" * 64,
    )
    outcomes = await asyncio.gather(first.extract(snapshot), second.extract(snapshot))
    assert [value.payload["data"] for value in outcomes] == [{"first": "first"}, {"second": "second"}]
    assert [first.last_usage["input_tokens"], second.last_usage["input_tokens"]] == [5, 6]
    assert base.last_usage == {}
    assert base.response_format == original
    assert "Find the first fact." in calls[0]["messages"][0]["content"]
    assert "Find the second fact." not in calls[0]["messages"][0]["content"]
    external = first.response_format
    external["json_schema"]["schema"].clear()
    assert first.schema
    await first.close()
    assert client.closed is False


@pytest.mark.asyncio
async def test_local_scopes_preserve_decimal_only_for_product_and_isolate_error_state():
    async def handler(request):
        payload = json.loads(request.content)
        fields = payload["structured_outputs"]["json_schema"]["$defs"]["CustomV1"]["properties"]
        if "bad" in fields:
            return httpx.Response(429, json={"error": "busy"})
        return httpx.Response(200, json=response({"data": {"good": "value"}, "evidence": [], "warnings": []}))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        base = LocalExtractor("Studio-MLX", client=client)
        good = base.for_schema(custom("good"))
        bad = base.for_schema(custom("bad"))
        assert "ProductV1" not in good.schema["$defs"]
        product = base.for_schema(get_schema(ExtractionRequest(url="https://example.com")))
        assert product.schema["$defs"]["ProductV1"]["properties"]["price"]["anyOf"][0]["pattern"]
        snapshot = SourceSnapshot(
            requested_url="https://example.com",
            final_url="https://example.com",
            retrieved_at=datetime.now(UTC),
            raw_record={},
            sections=[],
            source_hash="a" * 64,
        )
        results = await asyncio.gather(good.extract(snapshot), bad.extract(snapshot), return_exceptions=True)
        assert results[0].payload["data"] == {"good": "value"}
        assert isinstance(results[1], ScraperError)
        assert results[1].error.code == "LOCAL_INFERENCE_HTTP_ERROR"
        assert good._local_client.last_error is None
        assert base._local_client.last_error is None
        await good.close()
        assert not client.is_closed


@pytest.mark.parametrize(
    "schema_id",
    [
        "page_v1",
        "article_v1",
        "organization_v1",
        "person_v1",
        "job_v1",
        "event_v1",
        "listing_v1",
        "custom_v1",
    ],
)
@pytest.mark.asyncio
async def test_selected_schema_probe_uses_production_evidence_validation(schema_id):
    def leaves(value, path="", pointer="/0"):
        if isinstance(value, dict):
            return [
                entry
                for key, item in value.items()
                for entry in leaves(item, f"{path}.{key}".strip("."), pointer + "/" + key)
            ]
        if isinstance(value, list):
            return [
                entry
                for index, item in enumerate(value)
                for entry in leaves(item, f"{path}.{index}", pointer + f"/{index}")
            ]
        if value is None:
            return []
        excerpt = value if isinstance(value, str) else json.dumps(value)
        return [{"field_path": path, "json_pointer": pointer, "section_id": None, "excerpt": excerpt}]

    class Client:
        async def chat_completion(self, **kwargs):
            source = json.loads(kwargs["messages"][1]["content"].split("\n", 1)[1])
            data = source["ld_json"][0]
            return response({"data": data, "evidence": leaves(data), "warnings": []})

    request = ExtractionRequest(url="https://example.com", schema_id=schema_id)
    if schema_id == "custom_v1":
        request.fields = [
            CustomField(name="count", type="integer"),
            CustomField(name="active", type="boolean"),
        ]
    adapter = HuggingFaceExtractor("test", "test", client=Client()).for_schema(get_schema(request))
    report = await adapter.probe()
    assert report["passed"] is True
    assert report["schema_id"] == schema_id
    assert "synthetic" in report["limitations"]


def test_cli_discovery_and_custom_descriptor_file_are_credentialless(monkeypatch, capsys, tmp_path):
    from ai_scraper import pipeline

    assert main(["schemas"]) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["default_schema_id"] == "page_v1"
    assert {"page_v1", "article_v1", "custom_v1"} <= {item["schema_id"] for item in listing["schemas"]}
    seen = []

    class Pipeline:
        async def extract_url(self, request):
            seen.append(request)
            return ExtractionResult(
                job_id="test",
                status="partial",
                data={"rating": 4},
                provenance=Provenance(requested_url=request.url),
            )

        async def close(self):
            pass

    monkeypatch.setattr(pipeline, "create_pipeline", lambda _: Pipeline())
    fields = tmp_path / "fields.json"
    fields.write_text(json.dumps([{"name": "rating", "type": "integer", "description": "Published rating"}]))
    assert (
        main(
            [
                "extract",
                "https://example.com",
                "--fields-file",
                str(fields),
                "--instruction",
                "Read its published rating",
            ]
        )
        == 2
    )
    assert seen[0].schema_id == "custom_v1"
    assert seen[0].fields[0].type == "integer"
    assert seen[0].instruction == "Read its published rating"
    assert json.loads(capsys.readouterr().out)["data"] == {"rating": 4}


@pytest.mark.asyncio
async def test_mcp_custom_fields_remain_typed_and_generic_data_serializes_once():
    seen = []

    class Pipeline:
        async def extract_url(self, request):
            seen.append(request)
            return ExtractionResult(
                job_id="test",
                status="success",
                data={"rating": 4},
                provenance=Provenance(
                    requested_url=request.url, retrieved_at=datetime.now(UTC), schema_version="custom_v1"
                ),
            )

        async def close(self):
            pass

    server = build_server(Pipeline)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert tools["extract_url"].inputSchema["properties"]["schema_id"]["default"] == "page_v1"
    result = await server.call_tool(
        "extract_url",
        {
            "url": "https://example.com",
            "fields": [{"name": "rating", "type": "integer"}],
            "instruction": "Read its published rating",
        },
    )
    payload = result[1] if isinstance(result, tuple) else json.loads(result[0].text)
    assert payload["data"] == {"rating": 4}
    assert isinstance(ExtractionResult.model_validate_json(json.dumps(payload)).data, dict)
    assert isinstance(seen[0].fields[0], CustomField)
    assert seen[0].schema_id == "custom_v1"


@pytest.mark.asyncio
async def test_probe_rejects_observed_numeric_context_excerpts_and_reports_safe_diagnostics():
    class Client:
        async def chat_completion(self, **kwargs):
            instruction = kwargs["messages"][0]["content"]
            assert "ONLY the scalar" in instruction
            assert '"17" and "true"' in instruction
            assert "pointers /0/count and /0/active" in instruction
            return response(
                {
                    "data": {"count": 17, "active": True},
                    "evidence": [
                        {
                            "field_path": "count",
                            "section_id": "s1",
                            "json_pointer": None,
                            "excerpt": '"count": 17',
                        },
                        {
                            "field_path": "active",
                            "section_id": "s1",
                            "json_pointer": None,
                            "excerpt": '"active": true',
                        },
                    ],
                    "warnings": [],
                }
            )

    request = ExtractionRequest(
        url="https://example.com",
        schema_id="custom_v1",
        fields=[CustomField(name="count", type="integer"), CustomField(name="active", type="boolean")],
    )
    adapter = HuggingFaceExtractor("test", "test", client=Client()).for_schema(get_schema(request))
    assert adapter.prompt_version == "general-evidence-v2"
    with pytest.raises(ScraperError) as caught:
        await adapter.probe()
    assert caught.value.error.code == "CAPABILITY_PROBE_FAILED"
    errors = caught.value.error.vendor_details["validation_errors"]
    assert len(errors) == 2
    assert all(error.startswith(("evidence:count:", "evidence:active:")) for error in errors)
    assert "payload" not in caught.value.error.vendor_details


def test_custom_probe_failure_retains_validated_contract_and_reads_descriptors_once(
    monkeypatch, capsys, tmp_path
):
    from ai_scraper import cli

    fields = tmp_path / "fields.json"
    fields.write_text(json.dumps([{"name": "count", "type": "integer"}]))
    reads = []
    original_request = cli._request

    def record_request(args):
        reads.append(True)
        result = original_request(args)
        fields.unlink()
        return result

    async def fail(settings, request):
        assert request.schema_id == "custom_v1"
        raise ScraperError(
            "CAPABILITY_PROBE_FAILED",
            "HF schema probe failed production evidence validation",
            vendor_details={
                "validation_errors": ["evidence:count: missing, non-verbatim, or unsupported value"]
            },
        )

    monkeypatch.setattr(cli, "_request", record_request)
    monkeypatch.setattr(cli, "_probe", fail)
    assert main(["probe", "--schema", "custom", "--fields-file", str(fields)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert reads == [True]
    assert payload["provenance"]["schema_version"] == "custom_v1"
    assert payload["provenance"]["prompt_version"] == "general-evidence-v2"
    assert payload["error"]["code"] == "CAPABILITY_PROBE_FAILED"
    assert payload["error"]["vendor_details"]["validation_errors"][0].startswith("evidence:count:")


def test_configuration_failure_after_valid_request_keeps_general_contract(monkeypatch, capsys):
    monkeypatch.setenv("HF_TIMEOUT_SECONDS", "invalid-secret-value")
    assert main(["extract", "https://example.com", "--schema", "article"]) == 1
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert "invalid-secret-value" not in captured.out + captured.err
    assert payload["error"]["code"] == "LOCAL_ERROR"
    assert payload["provenance"]["schema_version"] == "article_v1"
    assert payload["provenance"]["prompt_version"] == "general-evidence-v2"
