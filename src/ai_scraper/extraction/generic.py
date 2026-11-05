"""Conservative, source-aligned extraction for registered general contracts.

This module never follows links or adds evidence sections. Every emitted value
has a reference into the already saved snapshot, including resolved link URLs.
"""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import unquote, urljoin, urlsplit, urlunsplit

from bs4 import BeautifulSoup, CData, NavigableString
from pydantic import BaseModel, ValidationError

from ..registry import SchemaSpec
from ..schemas import EvidenceCandidate, SourceSnapshot, http_url

MAX_TEXT_BLOCKS = 2_000
MAX_LINKS = 500
MAX_TABLES = 100
MAX_TABLE_ROWS = 100
MAX_TABLE_COLUMNS = 50
MAX_HEADINGS = 500
MAX_METADATA_NODES = 200
MAX_TIME_NODES = 100
MAX_SOURCE_INDEX_SECTIONS = 50_000
MAX_SOURCE_INDEX_BYTES = 10_000_000
MAX_REFERENCE_ATTEMPTS = 10_000
MAX_SUBSTRING_CHECKS = 50_000
MAX_SUBSTRING_BYTES = 20_000_000
MAX_DOM_TEXT_VISITS = 200_000
MAX_DOM_TEXT_BYTES = 20_000_000
MAX_JSONLD_NODES = 50_000
MAX_JSONLD_DEPTH = 100


def scalar_text(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def populated_fields(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten every populated scalar, recursively, including nested arrays."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(populated_fields(item, f"{prefix}.{key}" if prefix else key))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(populated_fields(item, f"{prefix}.{index}"))
        return result
    # Empty table cells are structural placeholders, not extracted facts.
    return {prefix: value} if value is not None and value != "" else {}


def _walk(value: Any, path: str, *, on_limit=None):
    """Walk JSON-LD lazily without Python recursion or eager sibling expansion."""

    def children(current, pointer, depth):
        if isinstance(current, dict):
            for key, item in current.items():
                escaped = key.replace("~", "~0").replace("/", "~1")
                yield f"{pointer}/{escaped}", item, depth + 1
        else:
            for index, item in enumerate(current):
                yield f"{pointer}/{index}", item, depth + 1

    stack = [iter(((path, value, 0),))]
    examined = 0
    while stack:
        try:
            pointer, current, depth = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if examined >= MAX_JSONLD_NODES:
            if on_limit:
                on_limit("JSON-LD nodes")
            return
        examined += 1
        if depth > MAX_JSONLD_DEPTH:
            if on_limit:
                on_limit("JSON-LD depth")
            continue
        if isinstance(current, dict):
            yield pointer, current
            stack.append(children(current, pointer, depth))
        elif isinstance(current, list):
            stack.append(children(current, pointer, depth))


def _canonical(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), parsed.query, "")
    )


LD_TYPES = {
    "page_v1": {"WebPage", "AboutPage", "ContactPage", "ProfilePage"},
    "article_v1": {"Article", "NewsArticle", "BlogPosting", "TechArticle", "ScholarlyArticle", "Report"},
    "organization_v1": {
        "Organization",
        "Corporation",
        "LocalBusiness",
        "NGO",
        "EducationalOrganization",
        "CollegeOrUniversity",
        "GovernmentOrganization",
    },
    "person_v1": {"Person"},
    "job_v1": {"JobPosting"},
    "event_v1": {"Event", "BusinessEvent", "EducationEvent", "MusicEvent", "SportsEvent", "SocialEvent"},
    "listing_v1": {"ItemList", "CollectionPage"},
}


def _entity(snapshot: SourceSnapshot, spec: SchemaSpec, *, on_limit=None):
    types = LD_TYPES.get(spec.schema_id, set())
    entities = []
    limits = []
    for pointer, obj in _walk(snapshot.ld_json, "", on_limit=limits.append):
        actual = obj.get("@type", [])
        actual = [actual] if isinstance(actual, str) else actual
        if isinstance(actual, list) and types.intersection(
            x.rsplit("/", 1)[-1] for x in actual if isinstance(x, str)
        ):
            entities.append((pointer, obj))
    if limits:
        if on_limit:
            for name in dict.fromkeys(limits):
                on_limit(name)
        # Truncated traversal cannot establish uniqueness even if a relevant
        # entity was encountered before the unexamined branch.
        return None, bool(entities)
    targets = {_canonical(snapshot.requested_url), _canonical(snapshot.final_url)}
    matching = []
    for pointer, obj in entities:
        identities = [obj.get("url"), obj.get("@id"), obj.get("mainEntityOfPage")]
        identities = [
            identity.get("@id") if isinstance(identity, dict) else identity for identity in identities
        ]
        if any(
            isinstance(identity, str) and _canonical(urljoin(snapshot.final_url, identity)) in targets
            for identity in identities
        ):
            matching.append((pointer, obj))
    if len(matching) == 1:
        return matching[0], False
    if len(entities) == 1:
        return entities[0], False
    return None, bool(entities)


class _Builder:
    def __init__(self, snapshot: SourceSnapshot, spec: SchemaSpec):
        self.snapshot = snapshot
        self.spec = spec
        self.values = spec.data_model().model_dump()
        self.evidence: list[EvidenceCandidate] = []
        self.warnings: list[str] = []
        self.reference_attempts = 0
        self.substring_checks = 0
        self.substring_bytes = 0
        self.dom_text_visits = 0
        self.dom_text_bytes = 0
        self._literal_sections: dict[str, str] = {}
        self._substring_matches: dict[str, str | None] = {}
        self._indexed_sections = []
        self._node_text: dict[int, dict[str, str | None]] = {}
        indexed_bytes = 0
        for index, section in enumerate(snapshot.sections):
            if index >= MAX_SOURCE_INDEX_SECTIONS:
                self.limit("source section index")
                break
            size = len(section.text.encode())
            if indexed_bytes + size > MAX_SOURCE_INDEX_BYTES:
                self.limit("source section index bytes")
                break
            indexed_bytes += size
            self._indexed_sections.append((section, size))
            self._literal_sections.setdefault(section.text, section.id)
            if section.representation == "html_attribute":
                _, separator, value = section.text.partition(": ")
                if separator and value:
                    self._literal_sections.setdefault(value, section.id)

    def reference(self, path: str, source: str, pointer: str | None = None):
        if self.reference_attempts >= MAX_REFERENCE_ATTEMPTS:
            self.limit("evidence attempts")
            return None
        self.reference_attempts += 1
        if pointer:
            return EvidenceCandidate(field_path=path, json_pointer=pointer, excerpt=source)
        if not source:
            return None
        section_id = self._literal_sections.get(source)
        if section_id is None and source not in self._substring_matches:
            source_bytes = len(source.encode())
            for section, size in self._indexed_sections:
                cost = size + source_bytes
                if (
                    self.substring_checks >= MAX_SUBSTRING_CHECKS
                    or self.substring_bytes + cost > MAX_SUBSTRING_BYTES
                ):
                    self.limit("source substring search")
                    break
                self.substring_checks += 1
                self.substring_bytes += cost
                if source in section.text:
                    section_id = section.id
                    break
            self._substring_matches[source] = section_id
        elif section_id is None:
            section_id = self._substring_matches[source]
        if section_id is None:
            return None
        return EvidenceCandidate(field_path=path, section_id=section_id, excerpt=source)

    def text(self, node, separator: str = " ") -> str | None:
        """Match BeautifulSoup's stripped text with a total traversal budget."""
        cached = self._node_text.setdefault(id(node), {})
        if separator in cached:
            return cached[separator]
        parts = []
        for child in node.descendants:
            if self.dom_text_visits >= MAX_DOM_TEXT_VISITS:
                self.limit("DOM text traversal")
                cached[separator] = None
                return None
            self.dom_text_visits += 1
            if type(child) not in {NavigableString, CData}:
                continue
            value = str(child)
            size = len(value.encode())
            if self.dom_text_bytes + size > MAX_DOM_TEXT_BYTES:
                self.limit("DOM text bytes")
                cached[separator] = None
                return None
            self.dom_text_bytes += size
            if value.strip():
                parts.append(value.strip())
        cached[separator] = separator.join(parts)
        return cached[separator]

    def put(self, path: str, value: Any, *, source: str | None = None, pointer: str | None = None) -> bool:
        if value is None or value == "":
            return False
        reference = self.reference(path, source if source is not None else scalar_text(value), pointer)
        if reference is None:
            return False
        if "." in path:
            raise ValueError("put expects a top-level scalar field")
        old = self.values.get(path)
        if old is not None and old != [] and old != value:
            self.values[path] = None
            self.evidence = [item for item in self.evidence if item.field_path != path]
            self.warnings.append(f"conflict:{path}: structured and visible source values disagree")
            return False
        if any(warning.startswith(f"conflict:{path}:") for warning in self.warnings):
            return False
        try:
            self.spec.data_model.model_validate({**self.values, path: value})
        except (ValidationError, ValueError, TypeError):
            self.warnings.append(f"invalid_source:{path}: value does not meet the trusted field contract")
            return False
        self.values[path] = value
        self.evidence = [item for item in self.evidence if item.field_path != path]
        self.evidence.append(reference)
        return True

    def append(self, field: str, value: Any, *, source: str | None = None, pointer: str | None = None):
        if value is None or value == "" or value in self.values[field]:
            return
        path = f"{field}.{len(self.values[field])}"
        reference = self.reference(path, source if source is not None else scalar_text(value), pointer)
        if reference:
            try:
                self.spec.data_model.model_validate({**self.values, field: self.values[field] + [value]})
            except (ValidationError, ValueError, TypeError):
                self.warnings.append(
                    f"invalid_source:{field}: value does not meet the trusted field contract"
                )
                return
            self.values[field].append(value)
            self.evidence.append(reference)

    def limit(self, name: str):
        warning = f"incomplete_source: {name} exceeded deterministic extraction limits"
        if warning not in self.warnings:
            self.warnings.append(warning)


def _nodes(builder: _Builder, root, selector: str, limit: int, name: str):
    """Bound examined matches, including invalid, empty and duplicate nodes."""
    nodes = root.select(selector, limit=limit + 1)
    if len(nodes) > limit:
        builder.limit(name)
    return nodes[:limit]


def _ld_extract(builder: _Builder, pointer: str, obj: dict[str, Any]):
    schema_id = builder.spec.schema_id
    mappings = {
        "page_v1": {
            "title": "name",
            "description": "description",
            "published_at": "datePublished",
            "language": "inLanguage",
        },
        "article_v1": {"title": "headline", "description": "description", "published_at": "datePublished"},
        "organization_v1": {"name": "name", "description": "description", "website": "url"},
        "person_v1": {"name": "name", "role": "jobTitle", "biography": "description", "website": "url"},
        "job_v1": {
            "title": "title",
            "description": "description",
            "employment_type": "employmentType",
            # A literal salary string is already an exact source fact. Do not
            # synthesize one from structured amount/currency components.
            "salary_text": "baseSalary",
        },
        "event_v1": {
            "name": "name",
            "description": "description",
            "start_date": "startDate",
            "end_date": "endDate",
        },
        "listing_v1": {"title": "name", "description": "description"},
    }
    for field, key in mappings.get(schema_id, {}).items():
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            if field == "website":
                try:
                    value = http_url(urljoin(builder.snapshot.final_url, value))
                except ValueError:
                    continue
            builder.put(field, value, source=scalar_text(obj[key]), pointer=f"{pointer}/{key}")
    nested = {
        "page_v1": {"author": "author"},
        "article_v1": {"author": "author"},
        "person_v1": {"organization": "worksFor"},
        "job_v1": {"organization": "hiringOrganization", "location": "jobLocation"},
        "event_v1": {"organizer": "organizer", "location": "location"},
    }
    for field, key in nested.get(schema_id, {}).items():
        value = obj.get(key)
        path = f"{pointer}/{key}"
        if isinstance(value, dict):
            value, path = value.get("name"), path + "/name"
        if isinstance(value, str) and value.strip():
            builder.put(field, value, pointer=path)
    if schema_id in {"page_v1", "article_v1"}:
        for key in ("articleBody", "text"):
            if isinstance(obj.get(key), str) and obj[key].strip():
                builder.append("text", obj[key], pointer=f"{pointer}/{key}")
                break
    for field, key in (("emails", "email"), ("phone_numbers", "telephone"), ("addresses", "address")):
        if field not in builder.values:
            continue
        value = obj.get(key)
        if isinstance(value, str):
            builder.append(field, value, pointer=f"{pointer}/{key}")
        elif isinstance(value, list):
            for index, item in enumerate(value[:MAX_LINKS]):
                if isinstance(item, str) and item.strip():
                    builder.append(field, item, pointer=f"{pointer}/{key}/{index}")
            if len(value) > MAX_LINKS:
                builder.limit(f"{field} items")
        elif field == "addresses" and isinstance(value, dict):
            street = value.get("streetAddress")
            if isinstance(street, str) and street.strip():
                builder.append(field, street, pointer=f"{pointer}/{key}/streetAddress")
    if schema_id == "listing_v1":
        items = obj.get("itemListElement", [])
        if not isinstance(items, list):
            return
        for index, item in enumerate(items[:MAX_LINKS]):
            if not isinstance(item, dict):
                continue
            base = f"{pointer}/itemListElement/{index}"
            if isinstance(item.get("item"), dict):
                item, base = item["item"], base + "/item"
            values = {"name": None, "url": None, "description": None}
            references = []
            output_index = len(builder.values["items"])
            for field in values:
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    continue
                if field == "url":
                    try:
                        value = http_url(urljoin(builder.snapshot.final_url, value))
                    except ValueError:
                        continue
                reference = builder.reference(f"items.{output_index}.{field}", item[field], f"{base}/{field}")
                if reference:
                    values[field] = value
                    references.append(reference)
            if references:
                builder.values["items"].append(values)
                builder.evidence.extend(references)
        if len(items) > MAX_LINKS:
            builder.limit("listing items")


def _page_structure(builder: _Builder, soup: BeautifulSoup, root):
    for node in _nodes(builder, root, "h1, h2, h3, h4, h5, h6", MAX_HEADINGS, "headings"):
        value = builder.text(node)
        builder.append("headings", value)
    for node in _nodes(builder, root, "a[href]", MAX_LINKS, "links"):
        href = node.get("href")
        if not isinstance(href, str) or not href.strip():
            continue
        try:
            url = http_url(urljoin(builder.snapshot.final_url, href))
        except ValueError:
            continue
        if any(item["url"] == url for item in builder.values["links"]):
            continue
        index = len(builder.values["links"])
        ref = builder.reference(f"links.{index}.url", href)
        if not ref:
            continue
        label = builder.text(node) or None
        label_ref = builder.reference(f"links.{index}.label", label) if label else None
        builder.values["links"].append({"url": url, "label": label if label_ref else None})
        builder.evidence.append(ref)
        if label_ref:
            builder.evidence.append(label_ref)
    for table in _nodes(builder, root, "table", MAX_TABLES, "tables"):
        index = len(builder.values["tables"])
        headers: list[str] = []
        rows: list[list[str]] = []
        references = []
        examined_rows = table.find_all("tr", limit=MAX_TABLE_ROWS + 2)
        if len(examined_rows) > MAX_TABLE_ROWS + 1:
            builder.limit("table rows")
        for row in examined_rows[: MAX_TABLE_ROWS + 1]:
            cells = row.find_all(["td", "th"], recursive=False, limit=MAX_TABLE_COLUMNS + 1)
            if not cells:
                continue
            is_header = not rows and not headers and all(cell.name == "th" for cell in cells)
            if len(cells) > MAX_TABLE_COLUMNS:
                builder.limit("table columns")
            if not is_header and len(rows) >= MAX_TABLE_ROWS:
                builder.limit("table rows")
                break
            output = []
            row_references = []
            for column, cell in enumerate(cells[:MAX_TABLE_COLUMNS]):
                value = builder.text(cell)
                path = (
                    f"tables.{index}.headers.{column}"
                    if is_header
                    else f"tables.{index}.rows.{len(rows)}.{column}"
                )
                ref = builder.reference(path, value) if value else None
                if value is None or (value and ref is None):
                    # Preserve its shape without pretending an uncitable combined
                    # inline fragment was successfully extracted.
                    value = ""
                    builder.limit("unverifiable table cells")
                output.append(value)
                if ref:
                    row_references.append(ref)
            if is_header:
                headers = output
            else:
                rows.append(output)
            references.extend(row_references)
        if references:
            builder.values["tables"].append({"headers": headers, "rows": rows})
            builder.evidence.extend(references)


def _visible_contacts(builder: _Builder, root):
    """Use contact links in the identified entity's visible primary content."""
    for node in _nodes(builder, root, 'a[href^="mailto:"], a[href^="tel:"]', MAX_LINKS, "contact links"):
        href = node.get("href")
        if not isinstance(href, str):
            continue
        scheme = href.split(":", 1)[0]
        field = "emails" if scheme == "mailto" else "phone_numbers"
        if field not in builder.values:
            continue
        try:
            value = unquote(href.split(":", 1)[1].split("?", 1)[0], errors="strict")
        except (ValueError, UnicodeError):
            continue
        builder.append(field, value, source=href)
    for node in _nodes(builder, root, 'a[itemprop="url"][href]', MAX_LINKS, "website links"):
        href = node.get("href")
        if not isinstance(href, str):
            continue
        try:
            value = http_url(urljoin(builder.snapshot.final_url, href))
        except ValueError:
            continue
        builder.put("website", value, source=href)


def _visible_listing(builder: _Builder, root):
    """List source-page records without requesting any linked resource."""
    seen = set()
    for node in _nodes(builder, root, "li a[href]", MAX_LINKS, "listing items"):
        href = node.get("href")
        if not isinstance(href, str) or not href.strip():
            continue
        try:
            url = http_url(urljoin(builder.snapshot.final_url, href))
        except ValueError:
            continue
        if url in seen:
            continue
        index = len(builder.values["items"])
        reference = builder.reference(f"items.{index}.url", href)
        if reference is None:
            continue
        label = builder.text(node)
        label_reference = builder.reference(f"items.{index}.name", label) if label else None
        builder.values["items"].append(
            {"name": label if label_reference else None, "url": url, "description": None}
        )
        builder.evidence.append(reference)
        if label_reference:
            builder.evidence.append(label_reference)
        seen.add(url)


def deterministic_extract(snapshot: SourceSnapshot, spec: SchemaSpec) -> BaseModel:
    """Extract only trusted literal metadata, page structure and unique JSON-LD."""
    builder = _Builder(snapshot, spec)
    selected, ambiguous = _entity(snapshot, spec, on_limit=builder.limit)
    if ambiguous:
        builder.warnings.append("ambiguous_entity: no unique entity matches the requested URL")
    if selected:
        _ld_extract(builder, *selected)
    soup = BeautifulSoup(snapshot.raw_record.get("page_html") or "", "html.parser")
    for node in soup.select("script, style, noscript, template"):
        node.decompose()
    root = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    headings = _nodes(builder, root, "h1", MAX_HEADINGS, "title headings")
    heading = builder.text(headings[0]) if len(headings) == 1 else None
    metadata_nodes = _nodes(builder, soup, "meta[content]", MAX_METADATA_NODES, "metadata nodes")
    metadata_values: dict[str, list[str]] = {}
    for node in metadata_nodes:
        value = node.get("content")
        if not isinstance(value, str) or not value.strip():
            continue
        for attribute in ("name", "property"):
            name = node.get(attribute)
            if isinstance(name, str):
                values = metadata_values.setdefault(name, [])
                if value not in values:
                    values.append(value)
    if (
        selected
        and not ambiguous
        and spec.schema_id in {"organization_v1", "person_v1"}
        and len(headings) == 1
        and heading == selected[1].get("name")
    ):
        # A publisher's JSON-LD on an article page does not make the article's
        # author/contact links the publisher's own contact information.
        _visible_contacts(builder, root)
    # Entity presets require unique JSON-LD or semantic model interpretation;
    # an arbitrary page heading cannot establish a person/company/event name.
    if "title" in builder.values and not ambiguous:
        title = heading
        for name in ("og:title", "twitter:title"):
            if title:
                break
            if metadata_values.get(name):
                title = metadata_values[name][0]
        if not title and soup.title:
            title = builder.text(soup.title)
        if not title and isinstance(snapshot.raw_record.get("page_title"), str):
            title = snapshot.raw_record["page_title"]
        if title:
            builder.put("title", title)
    metadata = {
        "description": ["description", "og:description", "twitter:description"],
        "author": ["author", "article:author"],
        "published_at": ["article:published_time", "datePublished", "date"],
        "language": ["language", "inLanguage", "og:locale"],
    }
    for field, names in metadata.items():
        if field not in builder.values or ambiguous:
            continue
        for name in names:
            values = metadata_values.get(name, [])
            if len(values) > 1:
                builder.values[field] = None
                builder.evidence = [item for item in builder.evidence if item.field_path != field]
                builder.warnings.append(f"conflict:{field}: multiple metadata values")
                break
            if values:
                builder.put(field, values[0])
                break
    for field, selector in (
        ("published_at", "time[itemprop=datePublished], time.published, time[pubdate]"),
        ("start_date", "time[itemprop=startDate]"),
        ("end_date", "time[itemprop=endDate]"),
    ):
        if field in builder.values and not ambiguous:
            nodes = _nodes(builder, root, selector, MAX_TIME_NODES, "time nodes")
            values = list(dict.fromkeys(value for node in nodes if (value := builder.text(node))))
            for value in values:
                builder.put(field, value)
    if "text" in builder.values:
        # Source lines remain individually citable. Main/article markup limits
        # unrelated navigation where present; Markdown-only records use saved
        # source blocks without changing their evidence IDs.
        root_text = builder.text(root, "\n") if snapshot.raw_record.get("page_html") else None
        allowed = set((root_text or "").splitlines()) if snapshot.raw_record.get("page_html") else None
        examined = 0
        seen_text: set[str] = set()
        for section, _size in builder._indexed_sections:
            if section.representation not in {"html_text", "markdown", "html2text", "text"}:
                continue
            if allowed is not None and section.text not in allowed:
                continue
            if section.text in seen_text:
                continue
            if examined >= MAX_TEXT_BLOCKS:
                builder.limit("text blocks")
                break
            examined += 1
            seen_text.add(section.text)
            builder.append("text", section.text)
    if spec.schema_id == "page_v1":
        _page_structure(builder, soup, root)
    if spec.schema_id == "listing_v1" and not builder.values["items"]:
        _visible_listing(builder, root)
    if not snapshot.ld_json and spec.schema_id not in {"page_v1", "custom_v1"}:
        builder.warnings.append("representation: JSON-LD unavailable; semantic extraction may be needed")
    return spec.candidate_model(
        data=spec.data_model.model_validate(builder.values),
        evidence=builder.evidence,
        warnings=builder.warnings,
    )


def merge_semantic_gaps(payload: dict[str, Any], baseline: BaseModel, spec: SchemaSpec) -> dict[str, Any]:
    """Fill missing model values from trusted data without hiding disagreements."""
    try:
        parsed = spec.candidate_model.model_validate(payload)
    except (ValidationError, ValueError, TypeError):
        return payload
    result = parsed.model_dump()
    base = baseline.model_dump()
    filled = []
    for key, value in result["data"].items():
        if value is None or value == []:
            result["data"][key] = base["data"][key]
            filled.append(key)
    for reference in base["evidence"]:
        if reference["field_path"].split(".", 1)[0] in filled:
            result["evidence"].append(reference)
    result["warnings"] = list(dict.fromkeys(result["warnings"] + base["warnings"]))
    return result
