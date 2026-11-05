"""Immutable, explicit inference views over a complete saved source snapshot.

The snapshot and evidence IDs never change. General contracts can collapse
identical textual representations. Oversized page/article contracts may use one
explicit main/article scope, retaining structured data and metadata. Arbitrary
custom tasks keep the complete page so footer facts cannot disappear silently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup, CData, NavigableString

from ..schemas import SourceSection, SourceSnapshot

SOURCE_VIEW_VERSION = "source-view-v1"
MAX_SCOPE_DOM_VISITS = 200_000
MAX_SCOPE_TEXT_CHARS = 10_000_000
TEXT_REPRESENTATIONS = frozenset(
    {"html_text", "html_text_block", "markdown", "html2text", "text", "page_title"}
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _original_payload(snapshot: SourceSnapshot) -> dict[str, Any]:
    return {
        "final_url": snapshot.final_url,
        "sections": [{"id": section.id, "text": section.text} for section in snapshot.sections],
        "ld_json": snapshot.ld_json,
        "excluded_sections": snapshot.excluded_sections,
    }


@dataclass(frozen=True)
class SourceView:
    content: str
    source_hash: str
    schema_id: str
    scope: str
    coverage: str
    original_chars: int
    deduplicated_chars: int
    included_section_ids: tuple[str, ...]
    excluded_section_ids: tuple[str, ...]
    duplicate_section_ids: tuple[str, ...]
    out_of_scope_section_ids: tuple[str, ...]
    upstream_excluded_section_ids: tuple[str, ...]

    def as_metadata(self) -> dict[str, Any]:
        """Return a fresh descriptor; callers may persist it outside the model."""
        view_hash = hashlib.sha256(self.content.encode()).hexdigest()
        identity = {
            "selection_version": SOURCE_VIEW_VERSION,
            "source_hash": self.source_hash,
            "schema_id": self.schema_id,
            "scope": self.scope,
            "view_hash": view_hash,
            "included_section_ids": self.included_section_ids,
            "excluded_section_ids": self.excluded_section_ids,
            "duplicate_section_ids": self.duplicate_section_ids,
            "out_of_scope_section_ids": self.out_of_scope_section_ids,
            "upstream_excluded_section_ids": self.upstream_excluded_section_ids,
        }
        selection_hash = hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return {
            **identity,
            "selection_hash": selection_hash,
            "coverage": self.coverage,
            "original_chars": self.original_chars,
            "deduplicated_chars": self.deduplicated_chars,
            "view_chars": len(self.content),
            "included_section_ids": list(self.included_section_ids),
            "excluded_section_ids": list(self.excluded_section_ids),
            "duplicate_section_ids": list(self.duplicate_section_ids),
            "out_of_scope_section_ids": list(self.out_of_scope_section_ids),
            "upstream_excluded_section_ids": list(self.upstream_excluded_section_ids),
        }


def _scope_lines(snapshot: SourceSnapshot) -> tuple[str, set[str]] | None:
    html = snapshot.raw_record.get("page_html")
    if not isinstance(html, str) or not html:
        return None
    soup = BeautifulSoup(html, "html.parser")
    for node in soup.select("script, style, noscript, template"):
        node.decompose()
    # Multiple roots cannot establish one schema scope. Match the deterministic
    # preference for main, then article, without arbitrarily picking among peers.
    mains = soup.find_all("main", limit=2)
    articles = soup.find_all("article", limit=2) if not mains else []
    roots = mains or articles
    if len(roots) != 1:
        return None
    scope = "main" if mains else "article"
    allowed: set[str] = set()
    visits = 0
    chars = 0
    # A document title is valid metadata even when it sits outside main/article.
    nodes = [roots[0]] + ([soup.title] if soup.title is not None else [])
    for node in nodes:
        for child in node.descendants:
            if visits >= MAX_SCOPE_DOM_VISITS:
                return None
            visits += 1
            if type(child) not in {NavigableString, CData}:
                continue
            text = str(child)
            chars += len(text)
            if chars > MAX_SCOPE_TEXT_CHARS:
                return None
            allowed.update(line.strip() for line in text.strip().splitlines() if line.strip())
    return scope, allowed


def build_source_view(snapshot: SourceSnapshot, schema_id: str, max_chars: int) -> SourceView:
    """Plan a view without truncation or mutation, even when it cannot fit.

    The caller enforces its input limit *after* recording the descriptor. A view
    that remains oversized is returned intact so failure diagnostics explain
    what was considered; this function never clips text, JSON-LD, or attributes.
    """
    original = _original_payload(snapshot)
    original_chars = len(_json(original))
    unique: list[SourceSection] = []
    duplicates: list[str] = []
    seen_text: set[str] = set()
    for section in snapshot.sections:
        if section.representation in TEXT_REPRESENTATIONS:
            if section.text in seen_text:
                duplicates.append(section.id)
                continue
            seen_text.add(section.text)
        # All attributes and unknown representations are retained, including
        # repeated values that could have distinct structural meanings.
        unique.append(section)

    def serialize(sections: list[SourceSection], scope: str, outside: list[str]) -> str:
        return _json(
            {
                **original,
                "sections": [{"id": section.id, "text": section.text} for section in sections],
                "source_scope": {
                    "selection_version": SOURCE_VIEW_VERSION,
                    "scope": scope,
                    "deduplicated_section_count": len(duplicates),
                    "out_of_scope_section_count": len(outside),
                },
            }
        )

    scope = "full_source"
    outside: list[str] = []
    included = unique
    content = serialize(included, scope, outside)
    deduplicated_chars = len(content)
    if len(content) > max_chars and schema_id in {"page_v1", "article_v1"}:
        scoped = _scope_lines(snapshot)
        if scoped is not None:
            scope, allowed = scoped
            included = []
            for section in unique:
                # Only HTML text known to originate outside the declared scope
                # is removed. Keep other vendor representations, all metadata,
                # all HTML blocks and the entire JSON-LD representation.
                if section.representation == "html_text" and section.text not in allowed:
                    outside.append(section.id)
                else:
                    included.append(section)
            content = serialize(included, scope, outside)
    excluded = set(duplicates) | set(outside)
    coverage = "schema_scope" if outside else "full"
    if snapshot.excluded_sections:
        coverage = "incomplete"
    return SourceView(
        content=content,
        source_hash=snapshot.source_hash,
        schema_id=schema_id,
        scope=scope,
        coverage=coverage,
        original_chars=original_chars,
        deduplicated_chars=deduplicated_chars,
        included_section_ids=tuple(section.id for section in included),
        excluded_section_ids=tuple(section.id for section in snapshot.sections if section.id in excluded),
        duplicate_section_ids=tuple(duplicates),
        out_of_scope_section_ids=tuple(outside),
        upstream_excluded_section_ids=tuple(snapshot.excluded_sections),
    )
