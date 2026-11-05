from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from ai_scraper.cli import main, parser
from ai_scraper.config import Settings
from ai_scraper.mcp_server import build_server
from ai_scraper.schemas import ExtractionResult, Provenance


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    for key in (
        "SCRAPER_RETRIEVAL_BACKEND",
        "SCRAPER_ALLOWED_DOMAINS",
        "SCRAPER_INFERENCE_BACKEND",
        "LOCAL_MODEL",
        "HF_TOKEN",
        "HF_MODEL",
        "HF_PROVIDER",
        "BRIGHTDATA_API_KEY",
        "SCRAPER_USER_AGENT",
        "BROWSER_TIMEOUT_SECONDS",
        "BROWSER_MAX_REQUESTS",
        "BROWSER_MAX_TOTAL_BYTES",
        "BROWSER_MAX_ATTEMPTS",
        "BROWSER_RESOURCE_DOMAINS",
        "BROWSER_EXECUTABLE_PATH",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SCRAPER_STATE_DIR", str(tmp_path / "state"))


class SavedCrawl:
    def __init__(self, status="pending", crawl_id="crawl-one"):
        self.payload = {
            "crawl_id": crawl_id,
            "status": status,
            "counts": {"discovered": 2},
            "entries": [],
            "warnings": [],
            "error": None,
        }

    def model_dump(self, **_):
        return self.payload


def decode(value):
    if isinstance(value, tuple):
        return value[1]
    if isinstance(value, dict):
        return value
    return json.loads(value[0].text)


def test_browser_settings_are_bounded_and_user_agent_cannot_inject_headers(monkeypatch):
    monkeypatch.setenv("SCRAPER_RETRIEVAL_BACKEND", "auto")
    monkeypatch.setenv("BROWSER_MAX_REQUESTS", "12")
    monkeypatch.setenv("BROWSER_RESOURCE_DOMAINS", "cdn.example.org, static.example.org")
    settings = Settings.from_env()
    assert settings.retrieval_backend == "auto" and settings.browser_max_requests == 12
    assert settings.browser_resource_domains == ["cdn.example.org", "static.example.org"]
    assert settings.user_agent == "EvidenceAIScraper/0.5"
    for value in ("bad\r\nAuthorization: value", "非ASCII", "a" * 201, " ", "agent\x7f"):
        with pytest.raises(ValidationError):
            Settings(user_agent=value)
    with pytest.raises(ValidationError):
        Settings(browser_max_requests=501)


def test_browser_modes_are_mutually_exclusive():
    with pytest.raises(SystemExit) as error:
        parser().parse_args(["extract", "https://example.org", "--browser", "--render-fallback"])
    assert error.value.code == 2


def test_extract_browser_plan_reaches_pipeline_with_custom_schema(monkeypatch, tmp_path, capsys):
    from ai_scraper import pipeline

    seen = []

    class FakePipeline:
        async def extract_url(self, request):
            seen.append(request)
            return ExtractionResult(
                job_id="browser-page", status="partial", provenance=Provenance(requested_url=request.url)
            )

        async def close(self):
            seen.append("closed")

    def factory(settings):
        seen.append(settings)
        return FakePipeline()

    monkeypatch.setattr(pipeline, "create_pipeline", factory)
    actions = tmp_path / "actions.json"
    actions.write_text(
        json.dumps(
            [{"kind": "click", "selector": "button.more"}, {"kind": "wait_for", "selector": "#details"}]
        )
    )
    assert (
        main(
            [
                "extract",
                "https://example.org",
                "--browser",
                "--browser-actions",
                str(actions),
                "--fields",
                "year:integer",
            ]
        )
        == 2
    )
    assert json.loads(capsys.readouterr().out)["job_id"] == "browser-page"
    assert seen[0].retrieval_backend == "browser"
    assert seen[1].schema_id == "custom_v1" and seen[1].fields[0].name == "year"
    assert [x.kind for x in seen[1].browser_actions] == ["click", "wait_for"]
    assert seen[-1] == "closed"


@pytest.mark.parametrize(
    "contents",
    ["{}", '[{"kind":"fill","selector":"input"}]', json.dumps([{"kind": "scroll"}] * 11), " " * 65537],
)
def test_invalid_browser_plans_fail_before_client_creation(contents, monkeypatch, tmp_path, capsys):
    from ai_scraper import pipeline

    def must_not_create(_):
        raise AssertionError("Invalid caller plans must not initialize clients")

    monkeypatch.setattr(pipeline, "create_pipeline", must_not_create)
    actions = tmp_path / "bad-actions.json"
    actions.write_text(contents)
    assert main(["extract", "https://example.org", "--browser", "--browser-actions", str(actions)]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["error"]["code"] == "INVALID_REQUEST"


def test_browser_plan_requires_browser_backend_before_client_creation(monkeypatch, tmp_path, capsys):
    from ai_scraper import pipeline

    monkeypatch.setattr(pipeline, "create_pipeline", lambda _: pytest.fail("Should not create a client"))
    actions = tmp_path / "actions.json"
    actions.write_text('[{"kind":"scroll","pixels":300}]')
    assert main(["extract", "https://example.org", "--browser-actions", str(actions)]) == 1
    assert json.loads(capsys.readouterr().out)["error"]["code"] == "INVALID_REQUEST"


def test_crawl_cli_defaults_direct_and_seed_scope_and_preserves_custom_fields(monkeypatch, capsys):
    from ai_scraper import crawl

    seen = []

    class FakeCrawler:
        def __init__(self, settings):
            seen.append(settings)

        async def start(self, request):
            seen.append(request)
            return SavedCrawl()

        async def close(self):
            seen.append("closed")

    monkeypatch.setattr(crawl, "Crawler", FakeCrawler)
    assert (
        main(
            [
                "crawl",
                "https://example.org/start",
                "--scope-domain",
                "docs.example.org",
                "--max-pages",
                "3",
                "--max-depth",
                "1",
                "--fields",
                "name,year:integer",
            ]
        )
        == 3
    )
    assert json.loads(capsys.readouterr().out)["crawl_id"] == "crawl-one"
    assert seen[0].retrieval_backend == "direct"
    assert seen[0].allowed_domains == ["docs.example.org", "example.org"]
    assert seen[0].hf_token is seen[0].brightdata_api_key is None
    request = seen[1]
    assert request.schema_id == "custom_v1" and request.max_pages == 3 and request.max_depth == 1
    assert [field.name for field in request.fields] == ["name", "year"]
    assert seen[-1] == "closed"


def test_crawl_cli_rejects_invalid_limits_before_constructing_crawler(monkeypatch, capsys):
    from ai_scraper import crawl

    monkeypatch.setattr(crawl, "Crawler", lambda _: pytest.fail("Invalid limits must fail first"))
    assert main(["crawl", "https://example.org", "--max-pages", "1001"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed" and payload["error"]["code"] == "INVALID_REQUEST"
    assert "crawl_id" in payload and "job_id" not in payload


def test_resume_crawl_and_status_share_lifecycle_without_new_start(monkeypatch, capsys):
    from ai_scraper import crawl

    seen = []

    class FakeCrawler:
        def __init__(self, settings):
            assert settings.retrieval_backend == "direct"

        async def resume(self, crawl_id, *, wait_seconds):
            seen.append(("resume", crawl_id, wait_seconds))
            return SavedCrawl("partial", crawl_id)

        def status(self, crawl_id):
            seen.append(("status", crawl_id))
            return SavedCrawl("partial", crawl_id)

        async def close(self):
            seen.append("closed")

    monkeypatch.setattr(crawl, "Crawler", FakeCrawler)
    assert main(["resume-crawl", "crawl-existing", "--wait-seconds", "2"]) == 2
    capsys.readouterr()
    assert main(["crawl-status", "crawl-existing"]) == 2
    capsys.readouterr()
    assert seen == [("resume", "crawl-existing", 2.0), "closed", ("status", "crawl-existing"), "closed"]


@pytest.mark.asyncio
async def test_mcp_crawl_discovery_is_lazy_and_exposes_browser_plans():
    def must_not_initialize(*_):
        raise AssertionError("Tool discovery must not initialize clients")

    server = build_server(must_not_initialize, crawler_factory=must_not_initialize)
    tools = {tool.name: tool for tool in await server.list_tools()}
    assert {"crawl_site", "resume_crawl", "crawl_status"} <= tools.keys()
    assert tools["crawl_site"].inputSchema["required"] == ["seeds"]
    assert "browser_actions" in tools["extract_url"].inputSchema["properties"]


@pytest.mark.asyncio
async def test_mcp_crawl_custom_browser_request_resume_and_status_close_clients():
    seen = []

    class FakeCrawler:
        def __init__(self, settings):
            seen.append(settings)

        async def start(self, request):
            seen.append(request)
            return SavedCrawl()

        async def resume(self, crawl_id, *, wait_seconds):
            seen.append(("resume", crawl_id, wait_seconds))
            return SavedCrawl("partial", crawl_id)

        def status(self, crawl_id):
            seen.append(("status", crawl_id))
            return SavedCrawl("partial", crawl_id)

        async def close(self):
            seen.append("closed")

    server = build_server(crawler_factory=FakeCrawler)
    result = decode(
        await server.call_tool(
            "crawl_site",
            {
                "seeds": ["https://example.org"],
                "fields": [{"name": "year", "type": "integer"}],
                "max_pages": 2,
                "retrieval_mode": "auto",
                "browser_actions": [{"kind": "scroll", "pixels": 300}],
            },
        )
    )
    assert result["status"] == "pending"
    assert seen[0].retrieval_backend == "auto" and seen[0].allowed_domains == ["example.org"]
    assert seen[1].schema_id == "custom_v1" and seen[1].browser_actions[0].pixels == 300
    assert seen[2] == "closed"
    result = decode(
        await server.call_tool(
            "resume_crawl",
            {"crawl_id": "crawl-one", "wait_seconds": 2.0, "retrieval_mode": "auto"},
        )
    )
    assert result["status"] == "partial"
    assert seen[3].retrieval_backend == "auto"
    result = decode(await server.call_tool("crawl_status", {"crawl_id": "crawl-one"}))
    assert result["status"] == "partial"
    assert seen[4:6] == [("resume", "crawl-one", 2.0), "closed"]
    assert seen[6].retrieval_backend == "direct"
    assert seen[7:] == [("status", "crawl-one"), "closed"]


@pytest.mark.asyncio
async def test_mcp_invalid_crawl_limits_and_actions_never_create_clients():
    def must_not_initialize(*_):
        raise AssertionError("Invalid crawl must not initialize clients")

    server = build_server(crawler_factory=must_not_initialize)
    bad = decode(await server.call_tool("crawl_site", {"seeds": ["https://example.org"], "max_pages": 1001}))
    assert bad["error"]["code"] == "INVALID_REQUEST"
    bad = decode(
        await server.call_tool(
            "crawl_site",
            {
                "seeds": ["https://example.org"],
                "browser_actions": [{"kind": "scroll"}] * 11,
                "retrieval_mode": "browser",
            },
        )
    )
    assert bad["error"]["code"] == "INVALID_REQUEST"
