"""Immutable, trusted extraction contracts shared by CLI, MCP and inference.

Custom contracts are built from a bounded set of typed field descriptors. They
cannot import code, load external references, or insert arbitrary JSON Schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field, create_model, field_validator

from .contracts import ScraperError
from .schemas import (
    CRITICAL_FIELDS,
    PROMPT_VERSION,
    REQUIRED_FIELDS,
    EvidenceCandidate,
    ExtractionCandidate,
    ExtractionRequest,
    ProductV1,
    StrictModel,
    http_url,
)

GENERAL_PROMPT_VERSION = "general-evidence-v2"


class TextRecord(StrictModel):
    """Literal page strings; missing facts use null or an empty collection."""

    @field_validator("*")
    @classmethod
    def populated_text(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("Populated numbers must be finite")
        if isinstance(value, str) and not value.strip():
            raise ValueError("Populated text must not be blank")
        if isinstance(value, list) and any(isinstance(item, str) and not item.strip() for item in value):
            raise ValueError("Populated text items must not be blank")
        return value


class PageLink(TextRecord):
    label: str | None = None
    url: str

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str) -> str:
        return http_url(value)


class PageTable(StrictModel):
    # Empty cells are structural placeholders, not populated business facts.
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)


class PageV1(TextRecord):
    title: str | None = None
    text: list[str] = Field(default_factory=list)
    headings: list[str] = Field(default_factory=list)
    links: list[PageLink] = Field(default_factory=list)
    tables: list[PageTable] = Field(default_factory=list)
    description: str | None = None
    author: str | None = None
    published_at: str | None = None
    language: str | None = None


class ArticleV1(TextRecord):
    title: str | None = None
    author: str | None = None
    published_at: str | None = None
    description: str | None = None
    text: list[str] = Field(default_factory=list)


class OrganizationV1(TextRecord):
    name: str | None = None
    description: str | None = None
    website: str | None = None
    emails: list[str] = Field(default_factory=list)
    phone_numbers: list[str] = Field(default_factory=list)
    addresses: list[str] = Field(default_factory=list)

    @field_validator("website")
    @classmethod
    def valid_website(cls, value: str | None) -> str | None:
        return http_url(value) if value is not None else None

    @field_validator("emails")
    @classmethod
    def valid_emails(cls, values: list[str]) -> list[str]:
        if any(not re.fullmatch(r"[^@\s]+@[^@\s]+", value) for value in values):
            raise ValueError("Email addresses must have a local part and domain")
        return values


class PersonV1(TextRecord):
    name: str | None = None
    role: str | None = None
    organization: str | None = None
    biography: str | None = None
    website: str | None = None
    emails: list[str] = Field(default_factory=list)

    @field_validator("website")
    @classmethod
    def valid_website(cls, value: str | None) -> str | None:
        return http_url(value) if value is not None else None

    @field_validator("emails")
    @classmethod
    def valid_emails(cls, values: list[str]) -> list[str]:
        return OrganizationV1.valid_emails(values)


class JobV1(TextRecord):
    title: str | None = None
    organization: str | None = None
    location: str | None = None
    employment_type: str | None = None
    salary_text: str | None = None
    description: str | None = None


class EventV1(TextRecord):
    name: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    location: str | None = None
    organizer: str | None = None
    description: str | None = None


class ListingItem(TextRecord):
    name: str | None = None
    url: str | None = None
    description: str | None = None

    @field_validator("url")
    @classmethod
    def valid_url(cls, value: str | None) -> str | None:
        return http_url(value) if value is not None else None


class ListingV1(TextRecord):
    title: str | None = None
    description: str | None = None
    items: list[ListingItem] = Field(default_factory=list)


@dataclass(frozen=True)
class SchemaSpec:
    schema_id: str
    data_model: type[BaseModel]
    candidate_model: type[BaseModel]
    required_fields: tuple[str, ...]
    critical_fields: tuple[str, ...]
    instruction: str
    prompt_version: str
    fingerprint: str


def _candidate(data_model: type[BaseModel], name: str) -> type[BaseModel]:
    return create_model(
        name,
        __base__=StrictModel,
        data=(data_model, ...),
        evidence=(list[EvidenceCandidate], Field(default_factory=list)),
        warnings=(list[str], Field(default_factory=list)),
    )


def _spec(
    schema_id: str,
    data_model: type[BaseModel],
    required: tuple[str, ...],
    critical: tuple[str, ...],
    instruction: str,
    *,
    candidate_model: type[BaseModel] | None = None,
    prompt_version: str = GENERAL_PROMPT_VERSION,
) -> SchemaSpec:
    candidate_model = candidate_model or _candidate(data_model, data_model.__name__ + "Extraction")
    payload = {
        "schema_id": schema_id,
        "schema": candidate_model.model_json_schema(),
        "required_fields": required,
        "critical_fields": critical,
        "instruction": instruction,
        "prompt_version": prompt_version,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return SchemaSpec(
        schema_id,
        data_model,
        candidate_model,
        required,
        critical,
        instruction,
        prompt_version,
        fingerprint,
    )


_SPECS = {
    "product_v1": _spec(
        "product_v1",
        ProductV1,
        REQUIRED_FIELDS,
        CRITICAL_FIELDS,
        "Extract one unambiguous product. Preserve displayed price text. The price is a flat decimal string, "
        "and currency is a separate explicit ISO code. Never infer currency from a dollar symbol alone.",
        candidate_model=ExtractionCandidate,
        prompt_version=PROMPT_VERSION,
    ),
    "page_v1": _spec(
        "page_v1",
        PageV1,
        ("text",),
        ("text",),
        "Extract the public webpage's literal content, metadata, headings, links and tables. "
        "Keep text as exact source-aligned blocks and table cells as displayed. "
        "Dates are original source text. Do not invent summaries or missing metadata.",
    ),
    "article_v1": _spec(
        "article_v1",
        ArticleV1,
        ("title", "text"),
        ("title",),
        "Extract the main article's title, author, date, description and literal text blocks. "
        "Exclude unrelated articles or advertisements. Preserve source date text and do not paraphrase.",
    ),
    "organization_v1": _spec(
        "organization_v1",
        OrganizationV1,
        ("name",),
        ("name",),
        "Extract one unambiguous organization and its publicly displayed business description, website, "
        "email addresses, phone numbers and postal addresses. Do not mix different organizations.",
    ),
    "person_v1": _spec(
        "person_v1",
        PersonV1,
        ("name",),
        ("name",),
        "Extract one unambiguous publicly described person's name, role, organization, biography, website "
        "and explicitly published email addresses. Keep biography verbatim and do not infer personal facts.",
    ),
    "job_v1": _spec(
        "job_v1",
        JobV1,
        ("title", "description"),
        ("title",),
        "Extract one job posting's title, hiring organization, location, employment type, salary text "
        "and literal description. Preserve salary and currency text instead of guessing normalized amounts.",
    ),
    "event_v1": _spec(
        "event_v1",
        EventV1,
        ("name", "start_date"),
        ("name",),
        "Extract one event's name, source-exact start and end date strings, location, organizer and "
        "literal description. Do not infer a timezone, year or date absent from the supplied source.",
    ),
    "listing_v1": _spec(
        "listing_v1",
        ListingV1,
        ("title", "items"),
        ("title",),
        "Extract the listing page's title, literal description and explicitly listed items. Each item "
        "may contain a name, URL and source-exact description. Do not visit linked pages or invent records.",
    ),
}

_ALIASES = {
    "auto": "page_v1",
    "general": "page_v1",
    "page": "page_v1",
    "document": "page_v1",
    "article": "article_v1",
    "news": "article_v1",
    "organization": "organization_v1",
    "org": "organization_v1",
    "company": "organization_v1",
    "person": "person_v1",
    "contact": "person_v1",
    "job": "job_v1",
    "event": "event_v1",
    "listing": "listing_v1",
    "product": "product_v1",
    "custom": "custom_v1",
}
_CUSTOM_TYPES = {"string": str, "number": float, "integer": int, "boolean": bool, "string_list": list[str]}
_FIELD_NAME = re.compile(r"[a-z][a-z0-9_]{0,63}")


def _custom(request: ExtractionRequest) -> SchemaSpec:
    descriptors = getattr(request, "fields", None)
    if not isinstance(descriptors, list) or not 1 <= len(descriptors) <= 32:
        raise ScraperError(
            "INVALID_REQUEST", "Custom extraction requires between one and 32 field descriptors"
        )
    fields: dict[str, Any] = {}
    required: list[str] = []
    critical: list[str] = []
    for descriptor in descriptors:
        item = descriptor.model_dump() if isinstance(descriptor, BaseModel) else descriptor
        if not isinstance(item, dict) or set(item) - {"name", "type", "description", "required", "critical"}:
            raise ScraperError("INVALID_REQUEST", "Unsupported custom field descriptor")
        name = item.get("name")
        kind = item.get("type", "string")
        description = item.get("description", "")
        needed, essential = item.get("required", True), item.get("critical", False)
        if (
            not isinstance(name, str)
            or not _FIELD_NAME.fullmatch(name)
            or name.startswith("model_")
            or hasattr(StrictModel, name)
            or name in fields
            or not isinstance(kind, str)
            or kind not in _CUSTOM_TYPES
            or not isinstance(description, str)
            or len(description) > 500
            or type(needed) is not bool
            or type(essential) is not bool
        ):
            raise ScraperError("INVALID_REQUEST", "Invalid or duplicate custom field descriptor")
        data_type = _CUSTOM_TYPES[kind]
        if kind == "string_list":
            fields[name] = (data_type, Field(default_factory=list, description=description))
        else:
            fields[name] = (data_type | None, Field(default=None, description=description))
        if needed or essential:
            required.append(name)
        if essential:
            critical.append(name)
    model = create_model("CustomV1", __base__=TextRecord, **fields)
    instruction = (
        "Extract only the requested custom fields. Field descriptions define their meanings. "
        "Use null or empty arrays for missing or ambiguous facts, preserve source strings exactly, "
        "and never invent a value to satisfy a required field. Every populated scalar or array item "
        "needs an exact supporting source reference."
    )
    return _spec("custom_v1", model, tuple(required), tuple(critical), instruction)


def get_schema(request: ExtractionRequest) -> SchemaSpec:
    """Resolve an immutable trusted contract; custom definitions are request-scoped."""
    schema_id = _ALIASES.get(request.schema_id, request.schema_id)
    if schema_id == "custom_v1":
        return _custom(request)
    try:
        spec = _SPECS[schema_id]
    except KeyError:
        raise ScraperError("UNSUPPORTED_SCHEMA", "Unknown registered schema") from None
    if getattr(request, "fields", None):
        raise ScraperError("INVALID_REQUEST", "Custom fields require schema_id=custom_v1")
    return spec


def list_schemas() -> list[dict[str, Any]]:
    """Return public schema metadata without mutable shared model instances."""
    result = [
        {
            "schema_id": spec.schema_id,
            "required_fields": list(spec.required_fields),
            "critical_fields": list(spec.critical_fields),
            "description": spec.instruction,
            "fingerprint": spec.fingerprint,
            "custom_fields": False,
        }
        for spec in _SPECS.values()
    ]
    result.append(
        {
            "schema_id": "custom_v1",
            "required_fields": [],
            "critical_fields": [],
            "description": "Request-specific bounded typed fields with exact source evidence.",
            "fingerprint": None,
            "custom_fields": True,
        }
    )
    return result
