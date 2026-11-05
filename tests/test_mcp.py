from __future__ import annotations

import asyncio
import json
import sys
from datetime import UTC, datetime

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ai_scraper.mcp_server import build_server
from ai_scraper.schemas import EvidenceCandidate, ExtractionResult, ProductV1, Provenance


def decode_tool_result(value):
    if isinstance(value, tuple):
        return value[1]
    if isinstance(value, dict):
        return value
    return json.loads(value[0].text)


@pytest.mark.asyncio
async def test_mcp_discovery_is_lazy_and_describes_typed_contract():
    def must_not_initialize():
        raise AssertionError("Discovery must not initialize vendor clients")

    server = build_server(must_not_initialize)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert set(tools) == {
        "extract_url",
        "resume_extraction",
        "list_schemas",
        "crawl_site",
        "resume_crawl",
        "crawl_status",
    }
    assert tools["extract_url"].inputSchema["required"] == ["url"]
    assert tools["extract_url"].inputSchema["properties"]["schema_id"]["default"] == "page_v1"
    assert tools["extract_url"].outputSchema["properties"]["status"]["enum"] == [
        "pending",
        "success",
        "partial",
        "failed",
    ]
    invalid = await server.call_tool("extract_url", {"url": "file:///etc/passwd"})
    assert decode_tool_result(invalid)["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_mcp_extraction_and_resume_share_pipeline_contract_and_cleanup():
    seen = []

    class FakePipeline:
        async def extract_url(self, request):
            seen.append((request.url, request.schema_id, request.country))
            return ExtractionResult(
                job_id="one", status="pending", provenance=Provenance(requested_url=request.url)
            )

        async def resume(self, job_id):
            seen.append(job_id)
            return ExtractionResult(
                job_id=job_id, status="partial", provenance=Provenance(requested_url="https://example.com")
            )

        async def close(self):
            seen.append("closed")

    server = build_server(FakePipeline)
    first = decode_tool_result(
        await server.call_tool(
            "extract_url", {"url": "https://example.com", "schema_id": "product_v1", "country": "US"}
        )
    )
    second = decode_tool_result(await server.call_tool("resume_extraction", {"job_id": "one"}))
    assert first["status"] == "pending"
    assert second["status"] == "partial"
    assert seen == [("https://example.com", "product_v1", "US"), "closed", "one", "closed"]


@pytest.mark.asyncio
async def test_mcp_success_and_resume_serialize_typed_datetimes_data_and_evidence():
    retrieved_at = datetime(2026, 11, 5, 16, 30, tzinfo=UTC)
    calls = []

    def success(job_id, url):
        return ExtractionResult(
            job_id=job_id,
            status="success",
            data=ProductV1(name="Hoodie", price="69.00", currency="USD"),
            provenance=Provenance(
                requested_url=url,
                final_url=url,
                retrieved_at=retrieved_at,
                source_hash="a" * 64,
                retrieval_method="direct_http",
                provider="local",
                model="Studio-MLX",
            ),
            evidence={
                field: EvidenceCandidate(field_path=field, section_id="s0001", excerpt=excerpt)
                for field, excerpt in (("name", "Hoodie"), ("price", "USD 69.00"), ("currency", "USD 69.00"))
            },
        )

    class SuccessfulPipeline:
        async def extract_url(self, request):
            calls.append(("extract", request.url))
            return success("successful-job", request.url)

        async def resume(self, job_id):
            calls.append(("resume", job_id))
            return success(job_id, "https://example.com/product")

        async def close(self):
            calls.append("closed")

    server = build_server(SuccessfulPipeline)
    for tool, arguments in (
        ("extract_url", {"url": "https://example.com/product", "schema_id": "product_v1"}),
        ("resume_extraction", {"job_id": "successful-job"}),
    ):
        payload = decode_tool_result(await server.call_tool(tool, arguments))
        assert payload["status"] == "success"
        assert payload["data"]["name"] == "Hoodie"
        assert payload["data"]["price"] == "69.00"
        assert payload["evidence"]["name"]["section_id"] == "s0001"
        # The framework is responsible for the one JSON serialization boundary.
        # Strict Pydantic accepts ISO datetimes from JSON, but not from a dict.
        parsed = ExtractionResult.model_validate_json(json.dumps(payload))
        assert parsed.provenance.retrieved_at == retrieved_at
        assert isinstance(parsed.data, ProductV1)
        assert isinstance(parsed.evidence["price"], EvidenceCandidate)
    assert calls == [
        ("extract", "https://example.com/product"),
        "closed",
        ("resume", "successful-job"),
        "closed",
    ]


@pytest.mark.asyncio
async def test_real_stdio_handshake_discovery_and_safe_tool_call(tmp_path):
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ai_scraper.mcp_server"],
        env={"SCRAPER_STATE_DIR": str(tmp_path / "state")},
    )
    async with (
        asyncio.timeout(20),
        stdio_client(parameters) as (read_stream, write_stream),
        ClientSession(read_stream, write_stream) as session,
    ):
        initialized = await session.initialize()
        assert initialized.serverInfo.name == "Evidence AI Scraper"
        listing = await session.list_tools()
        assert {tool.name for tool in listing.tools} == {
            "extract_url",
            "resume_extraction",
            "list_schemas",
            "crawl_site",
            "resume_crawl",
            "crawl_status",
        }
        result = await session.call_tool("extract_url", {"url": "file:///etc/passwd"})
        assert not result.isError
        payload = result.structuredContent or json.loads(result.content[0].text)
        assert payload["status"] == "failed"
        assert payload["error"]["code"] == "INVALID_REQUEST"


@pytest.mark.asyncio
async def test_mcp_defaults_general_and_lists_custom_contracts_without_clients():
    server = build_server(lambda: (_ for _ in ()).throw(AssertionError("Discovery should remain lazy")))
    payload = decode_tool_result(await server.call_tool("list_schemas", {}))
    assert payload["default_schema_id"] == "page_v1"
    assert {"page_v1", "custom_v1", "event_v1"} <= {row["schema_id"] for row in payload["schemas"]}
    tools = {tool.name: tool for tool in await server.list_tools()}
    props = tools["extract_url"].inputSchema["properties"]
    assert {"fields", "instruction"} <= props.keys()
    assert props["schema_id"]["default"] == "page_v1"


@pytest.mark.asyncio
async def test_mcp_custom_fields_and_instruction_share_cli_request_validation():
    seen = []

    class UniversalPipeline:
        async def extract_url(self, request):
            seen.append(request)
            return ExtractionResult(
                job_id="general",
                status="success",
                data={"title": "Reference", "text": ["Source"]},
                provenance=Provenance(
                    requested_url=request.url,
                    schema_version=request.schema_id,
                    retrieved_at=datetime.now(UTC),
                ),
            )

        async def close(self):
            seen.append("closed")

    server = build_server(UniversalPipeline)
    payload = decode_tool_result(await server.call_tool("extract_url", {"url": "https://example.com"}))
    assert payload["status"] == "success" and payload["data"]["text"] == ["Source"]
    assert seen[0].schema_id == "page_v1"
    payload = decode_tool_result(
        await server.call_tool(
            "extract_url",
            {
                "url": "https://example.com",
                "fields": [{"name": "year", "type": "integer"}],
                "instruction": "Printed year only",
            },
        )
    )
    assert seen[2].schema_id == "custom_v1" and seen[2].fields[0].type == "integer"
    assert seen[2].instruction == "Printed year only"
    payload = decode_tool_result(
        await server.call_tool(
            "extract_url", {"url": "https://example.com", "instruction": "Find the published email addresses"}
        )
    )
    assert seen[4].schema_id == "custom_v1" and seen[4].fields[0].name == "matches"
    bad = decode_tool_result(
        await server.call_tool(
            "extract_url",
            {"url": "https://example.com", "schema_id": "product_v1", "fields": [{"name": "year"}]},
        )
    )
    assert bad["error"]["code"] == "INVALID_REQUEST"
    assert len(seen) == 6  # Invalid requests never initialize clients.
