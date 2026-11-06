"""Deterministic, declared whole-section windows for two-stage extraction.

Only the trusted request defines the relevance query. The saved snapshot and
its evidence IDs remain unchanged. Omissions are reported explicitly, including
duplicate representations and JSON-LD; no selected paragraph is clipped.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from ..contracts import ScraperError
from ..registry import SchemaSpec
from ..schemas import ExtractionRequest, SourceSection, SourceSnapshot

GROUNDED_VIEW_VERSION = "grounded-keyword-window-v1"
MAX_SECTIONS = 10_000
MAX_QUERY_CHARS = 8_000
MAX_SOURCE_CHARS = 10_000_000
_TEXT = frozenset({"html_text_block", "html_text", "text", "page_title"})
_WORDS = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "each",
        "exact",
        "extract",
        "field",
        "fields",
        "for",
        "from",
        "has",
        "in",
        "is",
        "it",
        "its",
        "literal",
        "of",
        "on",
        "only",
        "or",
        "page",
        "preserve",
        "requested",
        "return",
        "stated",
        "string",
        "the",
        "their",
        "this",
        "to",
        "use",
        "value",
        "verbatim",
        "was",
        "were",
        "what",
        "when",
        "which",
        "with",
    ]
)


def _json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise ScraperError(
            "GROUNDING_VIEW_INVALID", "Grounded source selection requires strict JSON"
        ) from None


def _hash(value: str) -> str:
    try:
        return hashlib.sha256(value.encode()).hexdigest()
    except UnicodeError:
        raise ScraperError(
            "GROUNDING_VIEW_INVALID", "Grounded source selection requires valid text"
        ) from None


def _limit() -> None:
    raise ScraperError("GROUNDING_VIEW_LIMIT", "Grounded source selection exceeds a configured bound")


@dataclass(frozen=True)
class GroundedSourceView:
    content: str
    descriptor_json: str

    def as_metadata(self) -> dict[str, Any]:
        return json.loads(self.descriptor_json)


def _query(request: ExtractionRequest, spec: SchemaSpec) -> tuple[str, frozenset[str]]:
    if request.fields:
        fields = [{"name": field.name, "description": field.description} for field in request.fields]
    else:
        fields = [
            {"name": name, "description": field.description or ""}
            for name, field in spec.data_model.model_fields.items()
        ]
    trusted = _json({"instruction": request.instruction or "", "fields": fields})
    if len(trusted) > MAX_QUERY_CHARS:
        _limit()
    query_text = " ".join([request.instruction or "", *(f"{f['name']} {f['description']}" for f in fields)])
    terms = frozenset(
        match.group().casefold()
        for match in _WORDS.finditer(query_text)
        if len(match.group()) > 1 and match.group().casefold() not in _STOPWORDS
    )
    return _hash(trusted), terms


def _heading(section: SourceSection) -> bool:
    return bool(re.match(r"^\s{0,3}#{1,6}\s+", section.text)) or (
        section.representation in {"page_title", "html_text_block"} and len(section.text) <= 160
    )


def build_grounded_view(
    snapshot: SourceSnapshot,
    request: ExtractionRequest,
    spec: SchemaSpec,
    max_chars: int = 8_000,
) -> GroundedSourceView:
    """Choose complete source sections without consulting gold or predictions.

    The character bound applies to the complete serialized source JSON. A
    section that cannot fit whole is excluded and remains a recall opportunity
    in evaluation. The descriptor separately records every original omitted ID.
    """
    if type(max_chars) is not int or not 1_024 <= max_chars <= 48_000:
        _limit()
    if len(snapshot.sections) > MAX_SECTIONS or len(snapshot.excluded_sections) > MAX_SECTIONS:
        _limit()
    ids = [section.id for section in snapshot.sections]
    if len(ids) != len(set(ids)) or any(not item or len(item) > 256 for item in ids):
        raise ScraperError("GROUNDING_VIEW_INVALID", "Grounded source section identity is invalid")
    if any(len(item) > 256 for item in snapshot.excluded_sections):
        _limit()
    if sum(len(section.text) for section in snapshot.sections) > MAX_SOURCE_CHARS:
        _limit()
    query_hash, terms = _query(request, spec)
    upstream = list(snapshot.excluded_sections)
    original = {
        "final_url": snapshot.final_url,
        "sections": [{"id": section.id, "text": section.text} for section in snapshot.sections],
        "ld_json": snapshot.ld_json,
        "excluded_sections": upstream,
    }
    original_json = _json(original)
    if len(original_json) > MAX_SOURCE_CHARS:
        _limit()
    has_markdown = isinstance(snapshot.raw_record.get("markdown"), str) and bool(
        snapshot.raw_record["markdown"].strip()
    )
    markdown = [section for section in snapshot.sections if section.representation == "markdown"]
    preferred = (
        markdown
        if has_markdown and markdown
        else [section for section in snapshot.sections if section.representation in _TEXT]
    )
    candidates, duplicates, seen = [], [], set()
    for section in preferred:
        if section.text in seen:
            duplicates.append(section.id)
        else:
            seen.add(section.text)
            candidates.append(section)
    duplicate_ids = set(duplicates)
    json_ld_excluded = bool(snapshot.ld_json)

    def serialize(selected: list[SourceSection], *, conservative=False) -> str:
        omitted = len(snapshot.sections) - len(selected)
        return _json(
            {
                "final_url": snapshot.final_url,
                "sections": [{"id": section.id, "text": section.text} for section in selected],
                "ld_json": [],
                "excluded_sections": upstream,
                "source_scope": {
                    "selection_version": GROUNDED_VIEW_VERSION,
                    "scope": "full_source",
                    "coverage": "incomplete" if omitted or upstream or json_ld_excluded else "full",
                    "excluded_section_count": MAX_SECTIONS if conservative else omitted,
                    "json_ld_excluded": json_ld_excluded,
                },
            }
        )

    all_content = serialize(candidates)
    selected = candidates
    if len(all_content) > max_chars:
        # Reserve the largest count representation, so incremental section
        # costs cannot exceed the exact final JSON character budget.
        remaining = max_chars - len(serialize([], conservative=True))
        if remaining < 0:
            _limit()
        costs = [len(_json({"id": section.id, "text": section.text})) for section in candidates]
        bases = []
        for section, cost in zip(candidates, costs, strict=True):
            if cost > remaining:
                bases.append(0)
                continue
            matches = {
                match.group().casefold()
                for match in _WORDS.finditer(section.text)
                if match.group().casefold() in terms
            }
            bases.append(100 * len(matches) + (8 if _heading(section) else 0))
        scores = [
            score
            + (bases[index - 1] // 4 if index else 0)
            + (bases[index + 1] // 4 if index + 1 < len(bases) else 0)
            for index, score in enumerate(bases)
        ]
        chosen = set()
        for index in sorted(range(len(candidates)), key=lambda item: (-scores[item], item)):
            cost = costs[index] + bool(chosen)
            if cost <= remaining:
                chosen.add(index)
                remaining -= cost
        selected = [section for index, section in enumerate(candidates) if index in chosen]
    content = serialize(selected)
    if len(content) > max_chars:
        _limit()
    included = [section.id for section in selected]
    included_ids = set(included)
    excluded = [section.id for section in snapshot.sections if section.id not in included_ids]
    outside = [item for item in excluded if item not in duplicate_ids]
    coverage = "incomplete" if excluded or upstream or json_ld_excluded else "full"
    identity = {
        "selection_version": GROUNDED_VIEW_VERSION,
        "source_hash": snapshot.source_hash,
        "schema_id": spec.schema_id,
        "schema_fingerprint": spec.fingerprint,
        "scope": "full_source",
        "coverage": coverage,
        "view_hash": _hash(content),
        "query_hash": query_hash,
        "max_chars": max_chars,
        "included_section_ids": included,
        "excluded_section_ids": excluded,
        "duplicate_section_ids": duplicates,
        "out_of_scope_section_ids": outside,
        "upstream_excluded_section_ids": upstream,
        "json_ld_excluded": json_ld_excluded,
        "preferred_representation": "markdown" if has_markdown and markdown else "saved_text",
    }
    descriptor = {
        **identity,
        "selection_hash": _hash(_json(identity)),
        "original_chars": len(original_json),
        "deduplicated_chars": len(all_content),
        "view_chars": len(content),
    }
    return GroundedSourceView(content, _json(descriptor))
