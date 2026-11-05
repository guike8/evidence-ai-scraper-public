"""Small stdio MCP interface over the same extraction and resume library."""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from .cli import build_request, failure_result
from .contracts import ScraperError
from .registry import list_schemas as registered_schemas
from .schemas import DEFAULT_SCHEMA_ID, CustomField, ExtractionResult


def build_server(pipeline_factory: Callable | None = None) -> FastMCP:
    # Lazy construction keeps import and tool discovery independent of secrets.
    server = FastMCP(
        "Evidence AI Scraper",
        instructions="Extract public webpages with validated source evidence. Use page_v1 for general content; article_v1, organization_v1, person_v1, job_v1, event_v1, listing_v1 or product_v1 for specific records. Supply typed fields with custom_v1 for other tasks. Resume pending jobs using their local job ID.",
        log_level="WARNING",
    )

    def pipeline():
        if pipeline_factory is not None:
            return pipeline_factory()
        from .pipeline import create_pipeline

        return create_pipeline()

    async def execute(
        *,
        url: str = "",
        schema_id: str = DEFAULT_SCHEMA_ID,
        country: str | None = None,
        fields: list[CustomField] | None = None,
        instruction: str | None = None,
        job_id: str | None = None,
    ) -> ExtractionResult:
        instance = None
        try:
            request = (
                None
                if job_id is not None
                else build_request(
                    url=url,
                    schema_id=schema_id,
                    country=country,
                    fields=fields,
                    instruction=instruction,
                )
            )
            instance = pipeline()
            result = (
                await instance.resume(job_id) if job_id is not None else await instance.extract_url(request)
            )
            return result
        except ScraperError as exc:
            return failure_result(exc.error.code, exc.error.message, url=url, job_id=job_id or "")
        except (ValidationError, ValueError, TypeError):
            return failure_result("INVALID_REQUEST", "Invalid request or configuration")
        except OSError:
            return failure_result("LOCAL_STATE_ERROR", "Could not read local state", url=url)
        finally:
            if instance is not None:
                await instance.close()

    @server.tool()
    async def extract_url(
        url: str,
        schema_id: str = DEFAULT_SCHEMA_ID,
        country: str | None = None,
        fields: list[CustomField] | None = None,
        instruction: str | None = None,
    ) -> ExtractionResult:
        """Extract a public webpage with evidence. Default page_v1 returns text, links, headings and tables.

        Presets: article_v1, organization_v1, person_v1, job_v1, event_v1,
        listing_v1, product_v1. For custom_v1 supply up to 32 fields with name,
        type (string/number/integer/boolean/string_list), description, required,
        and critical. An instruction without fields requests exact matching
        source strings in custom_v1.matches. Values must occur in source.
        """
        return await execute(
            url=url, schema_id=schema_id, country=country, fields=fields, instruction=instruction
        )

    @server.tool()
    async def list_schemas() -> dict:
        """List built-in extraction contracts and the bounded custom-field interface."""
        return {"default_schema_id": DEFAULT_SCHEMA_ID, "schemas": registered_schemas()}

    @server.tool()
    async def resume_extraction(job_id: str) -> ExtractionResult:
        """Resume an existing local job without submitting duplicate vendor work."""
        return await execute(job_id=job_id)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
