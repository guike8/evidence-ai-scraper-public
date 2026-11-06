"""Small stdio MCP interface over the same extraction and resume library."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Literal
from urllib.parse import urlsplit

from mcp.server.fastmcp import FastMCP
from pydantic import ValidationError

from .cli import build_request, failure_result
from .config import Settings
from .contracts import ScraperError
from .registry import list_schemas as registered_schemas
from .schemas import DEFAULT_SCHEMA_ID, AppError, BrowserAction, CustomField, ExtractionResult


def build_server(
    pipeline_factory: Callable | None = None, *, crawler_factory: Callable | None = None
) -> FastMCP:
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
        browser_actions: list[BrowserAction] | None = None,
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
                    browser_actions=browser_actions or [],
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
        browser_actions: list[BrowserAction] | None = None,
    ) -> ExtractionResult:
        """Extract a public webpage with evidence. Default page_v1 returns text, links, headings and tables.

        Presets: article_v1, organization_v1, person_v1, job_v1, event_v1,
        listing_v1, product_v1. For custom_v1 supply up to 32 fields with name,
        type (string/number/integer/boolean/string_list), description, required,
        and critical. An instruction without fields requests exact matching
        source strings in custom_v1.matches. Values must occur in source.
        """
        return await execute(
            url=url,
            schema_id=schema_id,
            country=country,
            fields=fields,
            instruction=instruction,
            browser_actions=browser_actions,
        )

    @server.tool()
    async def list_schemas() -> dict:
        """List built-in extraction contracts and the bounded custom-field interface."""
        return {"default_schema_id": DEFAULT_SCHEMA_ID, "schemas": registered_schemas()}

    @server.tool()
    async def resume_extraction(job_id: str) -> ExtractionResult:
        """Resume an existing local job without submitting duplicate vendor work."""
        return await execute(job_id=job_id)

    def crawl_failure(code: str, message: str, crawl_id: str = "") -> dict:
        return {
            "crawl_id": crawl_id,
            "status": "failed",
            "counts": {},
            "entries": [],
            "warnings": [],
            "error": AppError(code=code, message=message).model_dump(mode="json"),
        }

    async def crawl_execute(
        *,
        request=None,
        crawl_id: str | None = None,
        inspect_only=False,
        wait_seconds: float | None = None,
        retrieval_mode: str = "configured",
    ) -> dict:
        instance = None
        try:
            settings = Settings.from_env()
            updates = {}
            if not os.environ.get("SCRAPER_RETRIEVAL_BACKEND"):
                updates["retrieval_backend"] = "direct"
            if retrieval_mode != "configured":
                updates["retrieval_backend"] = retrieval_mode
            if request is not None and not os.environ.get("SCRAPER_ALLOWED_DOMAINS"):
                updates["allowed_domains"] = sorted(
                    {urlsplit(seed).hostname or "" for seed in request.seeds} | set(request.scope_domains)
                )
            settings = settings.model_copy(update=updates)
            if (
                request is not None
                and request.browser_actions
                and settings.retrieval_backend not in {"browser", "auto"}
            ):
                raise ScraperError("INVALID_REQUEST", "Browser actions require browser or auto retrieval")
            if crawler_factory is not None:
                instance = crawler_factory(settings)
            else:
                from .crawl import Crawler

                instance = Crawler(settings)
            if request is not None:
                result = await instance.start(request)
            elif inspect_only:
                result = instance.status(crawl_id)
            else:
                result = await instance.resume(crawl_id, wait_seconds=wait_seconds)
            return result.model_dump(mode="json")
        except ScraperError as exc:
            return crawl_failure(exc.error.code, exc.error.message, crawl_id or "")
        except (ValidationError, ValueError, TypeError):
            return crawl_failure("INVALID_REQUEST", "Invalid crawl request or configuration", crawl_id or "")
        except OSError:
            return crawl_failure("LOCAL_STATE_ERROR", "Could not read crawl state", crawl_id or "")
        finally:
            if instance is not None:
                await instance.close()

    @server.tool()
    async def crawl_site(
        seeds: list[str],
        schema_id: str = DEFAULT_SCHEMA_ID,
        fields: list[CustomField] | None = None,
        instruction: str | None = None,
        country: str | None = None,
        browser_actions: list[BrowserAction] | None = None,
        max_pages: int = 20,
        max_depth: int = 2,
        max_frontier: int = 1000,
        max_links_per_page: int = 500,
        delay_seconds: float = 1.0,
        wait_seconds: float = 180.0,
        scope_domains: list[str] | None = None,
        retrieval_mode: Literal["configured", "direct", "browser", "auto", "firecrawl"] = "configured",
    ) -> dict:
        """Crawl public seed hosts with robots, bounded breadth/depth and durable per-page jobs.

        Results contain page job IDs and saved-result paths. Resume using the
        crawl ID; pending vendor work is retained. Browser actions are trusted
        caller plans with click, wait_for or scroll steps, never page instructions.
        An explicit scope_domains list must include every seed host.
        """
        try:
            from .crawl import CrawlRequest

            if not seeds:
                raise ScraperError("INVALID_REQUEST", "At least one public seed URL is required")
            page = build_request(
                seeds[0],
                schema_id=schema_id,
                fields=fields,
                instruction=instruction,
                country=country,
                browser_actions=browser_actions or [],
            )
            request = CrawlRequest(
                seeds=seeds,
                schema_id=page.schema_id,
                fields=page.fields,
                instruction=page.instruction,
                country=page.country,
                browser_actions=page.browser_actions,
                max_pages=max_pages,
                max_depth=max_depth,
                max_frontier=max_frontier,
                max_links_per_page=max_links_per_page,
                delay_seconds=delay_seconds,
                wait_seconds=wait_seconds,
                scope_domains=scope_domains or [],
            )
        except ScraperError as exc:
            return crawl_failure(exc.error.code, exc.error.message)
        except (ValidationError, ValueError, TypeError):
            return crawl_failure("INVALID_REQUEST", "Invalid crawl request")
        return await crawl_execute(request=request, retrieval_mode=retrieval_mode)

    @server.tool()
    async def resume_crawl(
        crawl_id: str,
        wait_seconds: float | None = None,
        retrieval_mode: Literal["configured", "direct", "browser", "auto", "firecrawl"] = "configured",
    ) -> dict:
        """Resume a saved crawl with its original retrieval mode and per-page jobs."""
        return await crawl_execute(
            crawl_id=crawl_id, wait_seconds=wait_seconds, retrieval_mode=retrieval_mode
        )

    @server.tool()
    async def crawl_status(crawl_id: str) -> dict:
        """Inspect saved crawl progress without submitting retrieval or model work."""
        return await crawl_execute(crawl_id=crawl_id, inspect_only=True)

    return server


def main() -> None:
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
