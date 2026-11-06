from __future__ import annotations

import copy
import hashlib
import json

import pytest

from ai_scraper.contracts import ScraperError
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.normalize import normalize_record
from ai_scraper.registry import get_schema
from ai_scraper.schemas import ExtractionRequest
from training.firecrawl_views import MARKDOWN_VIEW_VERSION, FirecrawlViewExtractor, markdown_view


def source(markdown="# Field report\n\nVolunteer count: 17\n", method="firecrawl_self_hosted"):
    request = ExtractionRequest(url="https://public.test/report", schema_id="page_v1")
    snapshot = normalize_record(
        {
            "final_url": request.url,
            "page_html": "<title>Field report</title><nav>Hidden navigation</nav><p>Volunteer count: 17</p>",
            "markdown": markdown,
            "ld_json": [{"count": 17}],
        },
        request,
        retrieval_method=method,
    )
    return request, snapshot


def test_markdown_view_keeps_exact_ids_and_declares_excluded_representations():
    _, snapshot = source()
    before = copy.deepcopy(snapshot.model_dump())
    view = markdown_view(snapshot, "page_v1")
    payload, meta = json.loads(view.content), view.metadata()
    selected = [section for section in snapshot.sections if section.representation == "markdown"]
    assert payload["sections"] == [{"id": s.id, "text": s.text} for s in selected]
    assert payload["ld_json"] == [] and meta["json_ld_excluded"]
    assert meta["coverage"] == "selected_representation"
    assert meta["selection_version"] == MARKDOWN_VIEW_VERSION
    assert meta["view_hash"] == hashlib.sha256(view.content.encode()).hexdigest()
    assert set(meta["included_section_ids"]) | set(meta["excluded_section_ids"]) == {
        section.id for section in snapshot.sections
    }
    assert not set(meta["included_section_ids"]) & set(meta["excluded_section_ids"])
    assert snapshot.model_dump() == before
    meta["included_section_ids"].clear()
    assert view.metadata()["included_section_ids"]


@pytest.mark.parametrize("markdown,method", [("", "firecrawl_self_hosted"), ("# Report", "direct_http")])
def test_absent_or_wrong_source_fails(markdown, method):
    _, snapshot = source(markdown, method)
    with pytest.raises(ScraperError, match="Saved Firecrawl"):
        markdown_view(snapshot, "page_v1")


def test_modified_markdown_sections_fail():
    _, snapshot = source()
    section = next(s for s in snapshot.sections if s.representation == "markdown")
    section.text = "Invented content"
    with pytest.raises(ScraperError, match="section identity"):
        markdown_view(snapshot, "page_v1")


def test_full_record_control_is_byte_identical_to_existing_extractor():
    request, snapshot = source()
    actual = FirecrawlViewExtractor("offline", source_policy="full_record", client=object()).for_request(
        request, get_schema(request)
    )
    control = LocalExtractor("offline", client=object()).for_request(request, get_schema(request))
    assert actual._source_content(snapshot) == control._source_content(snapshot)
    assert actual.schema == control.schema
    assert actual.system_instruction == control.system_instruction


def test_source_limit_rejects_without_clipping_and_preserves_descriptor():
    request, snapshot = source("# Report\n" + "Large paragraph " * 400)
    actual = FirecrawlViewExtractor(
        "offline", source_policy="markdown", max_source_chars=100, client=object()
    ).for_request(request, get_schema(request))
    with pytest.raises(ScraperError, match="exceeds the input budget"):
        actual._source_content(snapshot)
    assert actual.last_source_view["view_chars"] > 100
    assert actual.last_source_view["included_section_ids"]
    assert actual.last_source_view["coverage"] != "full"


@pytest.mark.parametrize(
    "options",
    [{"source_policy": "unknown"}, {"source_bound_evidence": True}, {"source_quote_evidence": True}],
)
def test_unknown_or_mixed_policy_fails_before_client_creation(options):
    with pytest.raises(ValueError):
        FirecrawlViewExtractor("offline", **options)


def test_component_replay_is_explicitly_eligible_without_claiming_live_retrieval():
    _, snapshot = source(method="firecrawl_preparation_replay")
    view = markdown_view(snapshot, "custom_v1")
    assert view.metadata()["coverage"] == "selected_representation"
    assert snapshot.retrieval_method == "firecrawl_preparation_replay"
