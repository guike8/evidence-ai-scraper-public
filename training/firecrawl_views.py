"""Explicit experimental model input views over saved Firecrawl sources.

This does not retrieve pages, train a model or select a production policy.
Markdown-only comparisons retain the complete immutable snapshot for validation
and declare every excluded source section. No source text is clipped.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from ai_scraper.contracts import ScraperError
from ai_scraper.extraction.local import LocalExtractor
from ai_scraper.schemas import SourceSnapshot

MARKDOWN_VIEW_VERSION = "firecrawl-markdown-only-v1"


def _hash(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class MarkdownView:
    content: str
    descriptor_json: str

    def metadata(self) -> dict:
        return json.loads(self.descriptor_json)


def markdown_view(snapshot: SourceSnapshot, schema_id: str) -> MarkdownView:
    """Select all normalized Markdown sections, preserving their original IDs.

    Callers must validate snapshot immutability first (e.g. evaluate._context).
    JSON-LD/HTML are deliberately excluded from this model-only control, while
    remaining available in the saved source and the unchanged factual scorer.
    """
    markdown = snapshot.raw_record.get("markdown")
    if (
        snapshot.retrieval_method not in {"firecrawl_self_hosted", "firecrawl_preparation_replay"}
        or not isinstance(markdown, str)
        or not markdown.strip()
    ):
        raise ScraperError("FIRECRAWL_MARKDOWN_UNAVAILABLE", "Saved Firecrawl Markdown is required")
    lines = {line.strip() for line in markdown.splitlines() if line.strip()}
    selected = [section for section in snapshot.sections if section.representation == "markdown"]
    ids = [section.id for section in snapshot.sections]
    if not selected or len(ids) != len(set(ids)) or {section.text for section in selected} != lines:
        raise ScraperError("FIRECRAWL_MARKDOWN_INVALID", "Markdown section identity is inconsistent")
    included = [section.id for section in selected]
    excluded = [section.id for section in snapshot.sections if section.representation != "markdown"]
    upstream = list(snapshot.excluded_sections)
    payload = {
        "final_url": snapshot.final_url,
        "sections": [{"id": section.id, "text": section.text} for section in selected],
        "ld_json": [],
        "excluded_sections": [*excluded, *upstream],
        "source_scope": {
            "selection_version": MARKDOWN_VIEW_VERSION,
            "scope": "firecrawl_markdown",
            "coverage": "selected_representation",
            "excluded_section_count": len(excluded),
            "json_ld_excluded": bool(snapshot.ld_json),
        },
    }
    content = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    original = json.dumps(
        {
            "final_url": snapshot.final_url,
            "sections": [{"id": section.id, "text": section.text} for section in snapshot.sections],
            "ld_json": snapshot.ld_json,
            "excluded_sections": upstream,
        },
        ensure_ascii=False,
        allow_nan=False,
    )
    identity = {
        "selection_version": MARKDOWN_VIEW_VERSION,
        "source_hash": snapshot.source_hash,
        "schema_id": schema_id,
        "scope": "firecrawl_markdown",
        "coverage": "selected_representation",
        "view_hash": _hash(content.encode()),
        "included_section_ids": included,
        "excluded_section_ids": excluded,
        "duplicate_section_ids": [],
        "out_of_scope_section_ids": excluded,
        "upstream_excluded_section_ids": upstream,
        "json_ld_excluded": bool(snapshot.ld_json),
    }
    selection_hash = _hash(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode())
    descriptor = {
        **identity,
        "selection_hash": selection_hash,
        "original_chars": len(original),
        "view_chars": len(content),
        "deduplicated_chars": len(content),
    }
    return MarkdownView(content, json.dumps(descriptor, allow_nan=False))


class FirecrawlViewExtractor(LocalExtractor):
    """Benchmark-only input ablation using the regular evidence grammar.

    Full-record mode delegates unchanged to LocalExtractor. Markdown mode is not
    wired into the application factory and never claims full-page coverage.
    """

    def __init__(self, *args, source_policy="full_record", **kwargs):
        if source_policy not in {"full_record", "markdown"}:
            raise ValueError("Unknown Firecrawl benchmark source policy")
        if kwargs.get("source_bound_evidence") or kwargs.get("source_quote_evidence"):
            raise ValueError("Compare source views with the same original evidence grammar")
        self.source_policy = source_policy
        super().__init__(*args, **kwargs)

    def for_request(self, request, spec):
        scoped = super().for_request(request, spec)
        if self.source_policy == "markdown":
            scoped.source_view_version = MARKDOWN_VIEW_VERSION
        return scoped

    def _source_content(self, snapshot):
        if self.source_policy == "full_record":
            return super()._source_content(snapshot)
        self.last_source_view = None
        view = markdown_view(snapshot, self.schema_id)
        self.last_source_view = view.metadata()
        if len(view.content) > self.max_source_chars:
            raise ScraperError("CONTENT_TOO_LARGE", "Complete Firecrawl Markdown exceeds the input budget")
        return view.content
