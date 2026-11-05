"""Retain the vendor record and create stable, numbered evidence sections."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from bs4 import BeautifulSoup

from .contracts import ScraperError
from .schemas import ExtractionRequest, SourceSection, SourceSnapshot


def _ld_items(value: Any) -> list[Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return []
    if isinstance(value, dict):
        return [value]
    return value if isinstance(value, list) else []


def normalize_record(
    record: dict[str, Any],
    request: ExtractionRequest,
    snapshot_id: str | None = None,
    retrieval_method: str = "brightdata_fetch",
    *,
    max_record_bytes: int = 5_000_000,
    max_normalized_bytes: int = 10_000_000,
) -> SourceSnapshot:
    raw_json = json.dumps(record, ensure_ascii=False, allow_nan=False)
    if len(raw_json.encode()) > max_record_bytes:
        raise ScraperError("CONTENT_TOO_LARGE", "Raw source record exceeds its configured byte limit")
    raw = json.loads(raw_json)
    html = record.get("page_html") or ""
    soup = BeautifulSoup(html, "html.parser")
    ld_json = _ld_items(record.get("ld_json"))
    if not ld_json:
        for script in soup.select('script[type="application/ld+json"]'):
            ld_json.extend(_ld_items(script.string or script.get_text()))
    sections: list[SourceSection] = []
    seen: set[tuple[str, str]] = set()
    normalized_bytes = 0

    def add(text: str, representation: str) -> None:
        nonlocal normalized_bytes
        text = text.strip()
        if text and (text, representation) not in seen:
            normalized_bytes += len(text.encode())
            if normalized_bytes > max_normalized_bytes or len(sections) >= 50_000:
                raise ScraperError("CONTENT_TOO_LARGE", "Normalized source exceeds its byte or section limit")
            seen.add((text, representation))
            sections.append(
                SourceSection(id=f"s{len(sections) + 1:04d}", text=text, representation=representation)
            )

    for element in soup.select("meta[content], img[src], img[data-src], a[href], [itemprop][content]"):
        for attribute in ("content", "src", "data-src", "href"):
            value = element.get(attribute)
            if value:
                label = element.get("property") or element.get("name") or element.get("itemprop") or attribute
                add(f"{label}: {value}", "html_attribute")
    for element in soup.select("script, style, noscript, template"):
        element.decompose()
    for element in soup.select(
        "h1, .price, .product-price, .sku, .woocommerce-product-details__short-description, #tab-description, .posted_in, .woocommerce-product-attributes td"
    ):
        add(element.get_text(" ", strip=True), "html_text_block")
    for line in soup.get_text("\n", strip=True).splitlines():
        add(line, "html_text")
    for name in ("markdown", "html2text"):
        value = record.get(name)
        if isinstance(value, str):
            for line in value.splitlines():
                add(line, name)
    if not sections and record.get("page_title"):
        add(str(record["page_title"]), "page_title")
    # The raw record is immutable on disk. Hash includes every retained representation,
    # not the retrieval timestamp, so extraction replay has a stable cache identity.
    digest = hashlib.sha256(
        json.dumps(raw, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    return SourceSnapshot(
        requested_url=request.url,
        final_url=record.get("final_url") or request.url,
        retrieved_at=datetime.now(UTC),
        country=request.country,
        snapshot_id=snapshot_id,
        retrieval_method=retrieval_method,
        raw_record=raw,
        sections=sections,
        ld_json=ld_json,
        source_hash=digest,
    )


def detect_content_problem(snapshot: SourceSnapshot) -> str | None:
    text = "\n".join(s.text for s in snapshot.sections if s.representation != "html_attribute")
    lowered = text.lower()
    challenge_markers = (
        "verify you are human",
        "checking your browser",
        "just a moment...",
        "access denied",
        "complete the captcha",
        "cf-chl-",
        "enable cookies to continue",
    )
    if any(marker in lowered for marker in challenge_markers):
        return "CONTENT_CHALLENGE"
    html = snapshot.raw_record.get("page_html") or ""
    if (
        len(text.strip()) < 80
        and not snapshot.ld_json
        and "<script" in html.lower()
        and any(
            marker in html.lower() for marker in ('id="root"', 'id="app"', 'id="__next"', "enable javascript")
        )
    ):
        return "CONTENT_JS_SHELL"
    if not text.strip() and not snapshot.ld_json:
        return "CONTENT_EMPTY"
    return None
