from __future__ import annotations

import asyncio
import copy
import hashlib
import json
from datetime import UTC, datetime

import httpx
import pytest

from ai_scraper.contracts import ScraperError
from ai_scraper.extraction import source_view
from ai_scraper.extraction.huggingface import HuggingFaceExtractor
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.extraction.source_view import SOURCE_VIEW_VERSION, build_source_view
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest, SourceSection, SourceSnapshot

URL = "https://public.test/report"


def snapshot(html="", **record_values):
    return normalize_record(
        {"page_html": html, "final_url": URL, **record_values},
        ExtractionRequest(url=URL, schema_id="page_v1"),
        retrieval_method="fixture_replay",
    )


def noisy_page(lines=1_000, *, tag="main", suffix=""):
    return (
        "<title>Research brief</title><nav>"
        + "".join(f"<p>Navigation entry {index}: unrelated link.</p>" for index in range(lines))
        + f"</nav><{tag}><h1>Research brief</h1><p>Volunteer count: 37</p></{tag}>"
        + suffix
    )


def scoped(adapter, schema="page_v1", **request_values):
    request = ExtractionRequest(url=URL, schema_id=schema, **request_values)
    return adapter.for_request(request, get_schema(request))


def legacy_content(saved):
    return json.dumps(
        {
            "final_url": saved.final_url,
            "sections": [{"id": section.id, "text": section.text} for section in saved.sections],
            "ld_json": saved.ld_json,
            "excluded_sections": saved.excluded_sections,
        },
        ensure_ascii=False,
    )


@pytest.mark.parametrize("schema,tag", [("page_v1", "main"), ("article_v1", "article")])
def test_oversized_page_uses_explicit_contract_scope_with_original_ids(schema, tag):
    saved = snapshot(noisy_page(tag=tag))
    before = copy.deepcopy(saved.model_dump())
    base = HuggingFaceExtractor("fixture", "model", client=object())
    adapter = scoped(base, schema)
    original = legacy_content(saved)
    assert len(original) > 48_000
    content = adapter._source_content(saved)
    payload = json.loads(content)
    assert len(content) < 2_000
    assert payload["ld_json"] == saved.ld_json
    assert {section["text"] for section in payload["sections"]} == {
        "Research brief",
        "Volunteer count: 37",
    }
    original_ids = {section.id for section in saved.sections}
    assert {section["id"] for section in payload["sections"]} <= original_ids
    metadata = adapter.last_source_view
    assert metadata["selection_version"] == SOURCE_VIEW_VERSION
    assert metadata["scope"] == tag
    assert metadata["coverage"] == "schema_scope"
    assert metadata["source_hash"] == saved.source_hash
    assert metadata["schema_id"] == schema
    assert len(metadata["out_of_scope_section_ids"]) == 1_000
    assert metadata["view_hash"] == hashlib.sha256(content.encode()).hexdigest()
    assert saved.model_dump() == before
    assert base.last_source_view is None
    assert base.source_view_version is None
    assert adapter.source_view_version == SOURCE_VIEW_VERSION


def test_textual_representation_dedup_is_lossless_and_keeps_metadata_attributes():
    saved = SourceSnapshot(
        requested_url=URL,
        final_url=URL,
        retrieved_at=datetime.now(UTC),
        raw_record={},
        sections=[
            SourceSection(id="first", text="Repeated fact", representation="html_text_block"),
            SourceSection(id="second", text="Repeated fact", representation="html_text"),
            SourceSection(id="third", text="Repeated fact", representation="markdown"),
            SourceSection(id="attr1", text="name: Repeated fact", representation="html_attribute"),
            SourceSection(id="attr2", text="name: Repeated fact", representation="html_attribute"),
            SourceSection(id="opaque", text="Repeated fact", representation="vendor_custom"),
        ],
        ld_json=[{"name": "Repeated fact"}, {"name": "Repeated fact"}],
        source_hash="a" * 64,
    )
    first = build_source_view(saved, "custom_v1", 48_000)
    second = build_source_view(saved, "custom_v1", 48_000)
    payload = json.loads(first.content)
    assert [section["id"] for section in payload["sections"]] == ["first", "attr1", "attr2", "opaque"]
    assert payload["ld_json"] == saved.ld_json
    assert first.as_metadata() == second.as_metadata()
    assert first.as_metadata()["coverage"] == "full"
    assert first.as_metadata()["duplicate_section_ids"] == ["second", "third"]
    metadata = first.as_metadata()
    metadata["included_section_ids"].clear()
    assert first.as_metadata()["included_section_ids"] == ["first", "attr1", "attr2", "opaque"]


def test_small_general_source_keeps_full_scope_even_with_main_present():
    saved = snapshot(
        "<header>Public header</header><main><p>Main body</p></main><footer>Footer note</footer>"
    )
    view = build_source_view(saved, "page_v1", 48_000)
    payload = json.loads(view.content)
    assert {section["text"] for section in payload["sections"]} == {
        "Public header",
        "Main body",
        "Footer note",
    }
    assert view.scope == "full_source"
    assert view.coverage == "full"


def test_scoping_retains_conflicting_jsonld_all_attributes_and_secondary_representations():
    ld = [
        {"@type": "Article", "headline": "First variant", "url": URL},
        {"@type": "Article", "headline": "Second variant", "url": URL},
    ]
    html = '<meta property="og:title" content="Conflicting metadata">' + noisy_page()
    html += '<footer><a href="mailto:contact@public.test">Contact</a></footer>'
    saved = snapshot(html, ld_json=ld, markdown="Secondary representation remains available.")
    view = build_source_view(saved, "article_v1", 48_000)
    payload = json.loads(view.content)
    assert payload["ld_json"] == ld
    text = [section["text"] for section in payload["sections"]]
    assert "og:title: Conflicting metadata" in text
    assert "href: mailto:contact@public.test" in text
    assert "Secondary representation remains available." in text
    assert "Research brief" in text
    assert "Volunteer count: 37" in text


def test_oversized_custom_goal_does_not_drop_footer_or_choose_main_arbitrarily():
    saved = snapshot(noisy_page(suffix="<footer>Opening hours: Monday 09:00–17:00.</footer>"))
    adapter = scoped(
        HuggingFaceExtractor("fixture", "model", client=object()),
        "custom_v1",
        fields=[{"name": "opening_hours"}],
        instruction="Find the published opening hours in the footer.",
    )
    with pytest.raises(ScraperError) as error:
        adapter._source_content(saved)
    assert error.value.error.code == "CONTENT_TOO_LARGE"
    metadata = adapter.last_source_view
    footer_id = next(section.id for section in saved.sections if section.text.startswith("Opening hours:"))
    assert footer_id in metadata["included_section_ids"]
    assert metadata["out_of_scope_section_ids"] == []
    assert metadata["scope"] == "full_source"


@pytest.mark.parametrize(
    "html",
    [
        noisy_page(tag="section"),
        noisy_page() + "<main><p>Second possible main.</p></main>",
        noisy_page(tag="article") + "<article><p>Second possible article.</p></article>",
    ],
)
def test_missing_or_ambiguous_scope_fails_instead_of_truncating(html):
    saved = snapshot(html)
    adapter = scoped(HuggingFaceExtractor("fixture", "model", client=object()))
    with pytest.raises(ScraperError) as error:
        adapter._source_content(saved)
    assert error.value.error.code == "CONTENT_TOO_LARGE"
    assert adapter.last_source_view["scope"] == "full_source"
    assert adapter.last_source_view["out_of_scope_section_ids"] == []


def test_scope_dom_budget_exhaustion_fails_closed(monkeypatch):
    saved = snapshot(noisy_page())
    monkeypatch.setattr(source_view, "MAX_SCOPE_DOM_VISITS", 1)
    view = build_source_view(saved, "page_v1", 48_000)
    assert len(view.content) > 48_000
    assert view.scope == "full_source"
    assert view.out_of_scope_section_ids == ()


def test_jsonld_alone_cannot_be_silently_truncated_to_fit():
    ld = [{"description": "Long structured fact. " * 4_000}]
    saved = snapshot(noisy_page(), ld_json=ld)
    adapter = scoped(HuggingFaceExtractor("fixture", "model", client=object()))
    with pytest.raises(ScraperError) as error:
        adapter._source_content(saved)
    assert error.value.error.code == "CONTENT_TOO_LARGE"
    view = build_source_view(saved, "page_v1", 48_000)
    assert json.loads(view.content)["ld_json"] == ld
    assert view.scope == "main"
    assert len(view.content) > 48_000


def test_product_content_and_boundary_remain_exactly_legacy():
    saved = snapshot(noisy_page())
    base = HuggingFaceExtractor("fixture", "model", client=object(), max_source_chars=100_000)
    adapter = scoped(base, "product_v1")
    assert adapter._source_content(saved) == legacy_content(saved)
    assert adapter.last_source_view is None
    assert adapter.source_view_version is None
    adapter.max_source_chars = 48_000
    with pytest.raises(ScraperError) as error:
        adapter._source_content(saved)
    assert error.value.error.code == "CONTENT_TOO_LARGE"
    assert adapter.last_source_view is None


def test_budget_includes_json_framing_and_selection_metadata():
    saved = snapshot("<p>A short paragraph.</p>")
    view = build_source_view(saved, "page_v1", 48_000)
    adapter = scoped(HuggingFaceExtractor("fixture", "model", client=object()))
    adapter.max_source_chars = len(view.content)
    assert adapter._source_content(saved) == view.content
    adapter.max_source_chars -= 1
    with pytest.raises(ScraperError) as error:
        adapter._source_content(saved)
    assert error.value.error.code == "CONTENT_TOO_LARGE"


def test_general_source_can_be_prepared_before_any_rpc_and_returns_independent_metadata():
    saved = snapshot(noisy_page())
    adapter = scoped(HuggingFaceExtractor("fixture", "model", client=object()))
    prepared = adapter.prepare_source(saved)
    assert prepared == adapter.last_source_view
    assert prepared["scope"] == "main"
    prepared["included_section_ids"].clear()
    assert adapter.last_source_view["included_section_ids"]
    assert adapter.last_usage == {}
    adapter.max_source_chars = 1
    with pytest.raises(ScraperError) as error:
        adapter.prepare_source(saved)
    assert error.value.error.code == "CONTENT_TOO_LARGE"
    assert adapter.last_source_view["view_chars"] > 1


def test_product_prepare_source_does_not_inspect_or_validate_source_early():
    adapter = HuggingFaceExtractor("fixture", "model", client=object(), max_source_chars=1)
    assert adapter.prepare_source(snapshot(noisy_page())) is None
    assert adapter.last_source_view is None
    assert adapter.last_usage == {}


def test_upstream_exclusions_remain_explicit_and_snapshot_is_unchanged():
    saved = snapshot("<main><p>Known source.</p></main>")
    saved.excluded_sections = ["earlier-omitted-section"]
    before = copy.deepcopy(saved.model_dump())
    view = build_source_view(saved, "page_v1", 48_000)
    assert view.coverage == "incomplete"
    assert view.as_metadata()["upstream_excluded_section_ids"] == ["earlier-omitted-section"]
    assert json.loads(view.content)["excluded_sections"] == ["earlier-omitted-section"]
    assert saved.model_dump() == before


@pytest.mark.parametrize("local", [False, True])
async def test_concurrent_request_views_keep_independent_selection_metadata(local):
    requests = []
    ready = asyncio.Event()

    async def completion(**kwargs):
        requests.append(kwargs)
        if len(requests) == 2:
            ready.set()
        await ready.wait()
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": json.dumps(
                            {
                                "data": {"title": None, "text": []},
                                "evidence": [],
                                "warnings": [],
                            }
                        )
                    },
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        }

    if local:

        async def handler(request):
            return httpx.Response(200, json=await completion(**json.loads(request.content)))

        transport = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        base = LocalExtractor("fixture-model", client=transport)
    else:

        class Client:
            chat_completion = staticmethod(completion)

        base = HuggingFaceExtractor("fixture", "model", client=Client())
        transport = None
    first = scoped(base)
    second = scoped(base, "article_v1")
    first_source = snapshot(noisy_page())
    second_source = snapshot(noisy_page(tag="article", suffix="<footer>Extra source.</footer>"))
    before = [copy.deepcopy(value.model_dump()) for value in [first_source, second_source]]
    try:
        await asyncio.gather(first.extract(first_source), second.extract(second_source))
        assert len(requests) == 2
        assert base.last_source_view is None
        assert first.last_source_view["scope"] == "main"
        assert second.last_source_view["scope"] == "article"
        assert first.last_source_view["schema_id"] == "page_v1"
        assert second.last_source_view["schema_id"] == "article_v1"
        assert first.last_source_view["selection_hash"] != second.last_source_view["selection_hash"]
        assert first.last_source_view is not second.last_source_view
        assert first_source.model_dump() == before[0]
        assert second_source.model_dump() == before[1]
        assert all("source_scope" in call["messages"][1]["content"] for call in requests)
    finally:
        await base.close()
        if transport:
            await transport.aclose()
