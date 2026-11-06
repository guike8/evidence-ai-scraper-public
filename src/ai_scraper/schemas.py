"""Trusted request/result contracts and the backward-compatible product model."""

from __future__ import annotations

import re
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROMPT_VERSION = "product-evidence-v3"
SCHEMA_ID = "product_v1"
DEFAULT_SCHEMA_ID = "page_v1"
REQUIRED_FIELDS = ("name", "price", "currency")
CRITICAL_FIELDS = ("name",)
ISO_CURRENCIES = frozenset(
    [
        "AED",
        "AFN",
        "ALL",
        "AMD",
        "ANG",
        "AOA",
        "ARS",
        "AUD",
        "AWG",
        "AZN",
        "BAM",
        "BBD",
        "BDT",
        "BGN",
        "BHD",
        "BIF",
        "BMD",
        "BND",
        "BOB",
        "BOV",
        "BRL",
        "BSD",
        "BTN",
        "BWP",
        "BYN",
        "BZD",
        "CAD",
        "CDF",
        "CHE",
        "CHF",
        "CHW",
        "CLF",
        "CLP",
        "CNY",
        "COP",
        "COU",
        "CRC",
        "CUP",
        "CVE",
        "CZK",
        "DJF",
        "DKK",
        "DOP",
        "DZD",
        "EGP",
        "ERN",
        "ETB",
        "EUR",
        "FJD",
        "FKP",
        "GBP",
        "GEL",
        "GHS",
        "GIP",
        "GMD",
        "GNF",
        "GTQ",
        "GYD",
        "HKD",
        "HNL",
        "HTG",
        "HUF",
        "IDR",
        "ILS",
        "INR",
        "IQD",
        "IRR",
        "ISK",
        "JMD",
        "JOD",
        "JPY",
        "KES",
        "KGS",
        "KHR",
        "KMF",
        "KPW",
        "KRW",
        "KWD",
        "KYD",
        "KZT",
        "LAK",
        "LBP",
        "LKR",
        "LRD",
        "LSL",
        "LYD",
        "MAD",
        "MDL",
        "MGA",
        "MKD",
        "MMK",
        "MNT",
        "MOP",
        "MRU",
        "MUR",
        "MVR",
        "MWK",
        "MXN",
        "MXV",
        "MYR",
        "MZN",
        "NAD",
        "NGN",
        "NIO",
        "NOK",
        "NPR",
        "NZD",
        "OMR",
        "PAB",
        "PEN",
        "PGK",
        "PHP",
        "PKR",
        "PLN",
        "PYG",
        "QAR",
        "RON",
        "RSD",
        "RUB",
        "RWF",
        "SAR",
        "SBD",
        "SCR",
        "SDG",
        "SEK",
        "SGD",
        "SHP",
        "SLE",
        "SLL",
        "SOS",
        "SRD",
        "SSP",
        "STN",
        "SVC",
        "SYP",
        "SZL",
        "THB",
        "TJS",
        "TMT",
        "TND",
        "TOP",
        "TRY",
        "TTD",
        "TWD",
        "TZS",
        "UAH",
        "UGX",
        "USD",
        "USN",
        "UYI",
        "UYU",
        "UYW",
        "UZS",
        "VED",
        "VES",
        "VND",
        "VUV",
        "WST",
        "XAF",
        "XAG",
        "XAU",
        "XBA",
        "XBB",
        "XBC",
        "XBD",
        "XCD",
        "XDR",
        "XOF",
        "XPD",
        "XPF",
        "XPT",
        "XSU",
        "XTS",
        "XUA",
        "XXX",
        "YER",
        "ZAR",
        "ZMW",
        "ZWG",
    ]
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


def http_url(value: str) -> str:
    parsed = urlsplit(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError("Expected an HTTP(S) URL without credentials")
    if any(ord(c) < 33 for c in value):
        raise ValueError("URL contains whitespace or control characters")
    return value


class Category(StrictModel):
    name: str | None = None
    url: str | None = None

    @field_validator("name")
    @classmethod
    def nonempty_name(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("Category name must not be empty")
        return v

    @field_validator("url")
    @classmethod
    def valid_url(cls, v: str | None) -> str | None:
        return http_url(v) if v is not None else None


class AdditionalInfo(StrictModel):
    size: list[str] = Field(default_factory=list)
    color: list[str] = Field(default_factory=list)

    @field_validator("size", "color")
    @classmethod
    def nonempty_items(cls, v: list[str]) -> list[str]:
        if any(not item.strip() for item in v):
            raise ValueError("Attribute values must not be empty")
        return v


class ProductV1(StrictModel):
    name: str | None = None
    price_text: str | None = None
    price: Annotated[str, Field(pattern=r"^[0-9]+(?:[.][0-9]+)?$")] | None = Field(
        default=None,
        description="Flat decimal string such as '19.95'; currency is the separate top-level currency field.",
    )
    currency: str | None = None
    sku: str | None = None
    category: Category | None = None
    images: list[str] = Field(default_factory=list)
    short_description: str | None = None
    long_description: str | None = None
    additional_info: AdditionalInfo | None = None

    @field_validator("name", "sku", "short_description", "long_description", "price_text")
    @classmethod
    def nonempty(cls, v: str | None) -> str | None:
        if v is not None and not v.strip():
            raise ValueError("Populated text must not be empty")
        return v

    @field_validator("price")
    @classmethod
    def valid_price(cls, v: str | None) -> str | None:
        if v is not None:
            if not re.fullmatch(r"\d+(?:\.\d+)?", v):
                raise ValueError("Price must be an ungrouped nonnegative decimal string")
            try:
                amount = Decimal(v)
            except InvalidOperation as e:
                raise ValueError("Price must be a decimal string") from e
            if not amount.is_finite() or amount < 0 or "e" in v.lower() or "," in v:
                raise ValueError("Price must be finite, nonnegative, ungrouped decimal text")
        return v

    @field_validator("currency")
    @classmethod
    def valid_currency(cls, v: str | None) -> str | None:
        if v is not None and v not in ISO_CURRENCIES:
            raise ValueError("Currency must be an explicit ISO 4217 code")
        return v

    @field_validator("images")
    @classmethod
    def valid_images(cls, v: list[str]) -> list[str]:
        return [http_url(x) for x in v]


class EvidenceCandidate(StrictModel):
    field_path: str
    section_id: str | None = None
    excerpt: str
    json_pointer: str | None = None


class ExtractionCandidate(StrictModel):
    data: ProductV1
    evidence: list[EvidenceCandidate] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class CustomField(StrictModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[a-z][a-z0-9_]*$")
    type: Literal["string", "number", "integer", "boolean", "string_list"] = "string"
    description: str = Field(default="", max_length=500)
    required: bool = True
    critical: bool = False

    @field_validator("name")
    @classmethod
    def safe_field_name(cls, value: str) -> str:
        if value.startswith("model_") or value in {
            "provenance",
            "evidence",
            "usage",
            "status",
            "error",
            "job_id",
        }:
            raise ValueError("Field name is reserved for application metadata")
        return value


class BrowserAction(StrictModel):
    """A bounded caller-authored action; page content never supplies actions."""

    kind: Literal["click", "wait_for", "scroll"]
    selector: str | None = Field(default=None, min_length=1, max_length=500)
    timeout_ms: int = Field(default=5000, ge=1, le=10000)
    pixels: int = Field(default=600, ge=-4000, le=4000)

    @model_validator(mode="after")
    def check_action(self):
        if self.kind in {"click", "wait_for"}:
            if self.selector is None or not self.selector.strip():
                raise ValueError("Click and wait actions require a CSS selector")
            if any(ord(char) < 32 for char in self.selector):
                raise ValueError("Browser selectors cannot contain control characters")
        elif self.selector is not None:
            raise ValueError("Scroll actions do not accept a selector")
        return self


class ExtractionRequest(StrictModel):
    url: str
    browser_actions: list[BrowserAction] = Field(
        default_factory=list, max_length=10, exclude_if=lambda actions: not actions
    )
    schema_id: str = SCHEMA_ID
    country: str | None = None
    freshness_seconds: int = Field(default=3600, ge=0)
    locale: Literal["en_US", "en_GB", "de_DE", "fr_FR"] | None = None
    fields: list[CustomField] = Field(default_factory=list, max_length=32)
    instruction: str | None = Field(default=None, max_length=2000)

    @field_validator("url")
    @classmethod
    def valid_url(cls, v: str) -> str:
        return http_url(v)

    @field_validator("country")
    @classmethod
    def valid_country(cls, v: str | None) -> str | None:
        if v is not None and (len(v) != 2 or not v.isascii() or not v.isupper() or not v.isalpha()):
            raise ValueError("country must be a two-letter uppercase country code")
        return v


class SourceSection(StrictModel):
    id: str
    text: str
    representation: str = "text"


class SourceSnapshot(StrictModel):
    requested_url: str
    final_url: str
    retrieved_at: datetime
    country: str | None = None
    snapshot_id: str | None = None
    retrieval_method: str = "brightdata_fetch"
    raw_record: dict[str, Any]
    sections: list[SourceSection]
    ld_json: list[Any] = Field(default_factory=list)
    source_hash: str
    excluded_sections: list[str] = Field(default_factory=list)


class AppError(StrictModel):
    code: str
    message: str
    vendor_details: dict[str, Any] | None = None
    retryable: bool = False


class SourceViewSummary(StrictModel):
    selection_version: str = Field(min_length=1, max_length=64)
    selection_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    view_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    scope: Literal["full_source", "main", "article"]
    coverage: Literal["full", "schema_scope", "incomplete"]
    original_chars: int = Field(ge=0)
    view_chars: int = Field(ge=0)
    included_sections: int = Field(ge=0)
    excluded_sections: int = Field(ge=0)
    duplicate_sections: int = Field(ge=0)
    out_of_scope_sections: int = Field(ge=0)
    descriptor_path: str


class Provenance(StrictModel):
    requested_url: str
    final_url: str | None = None
    retrieved_at: datetime | None = None
    country: str | None = None
    source_hash: str | None = None
    snapshot_id: str | None = None
    snapshot_path: str | None = None
    retrieval_method: str | None = None
    schema_version: str = SCHEMA_ID
    prompt_version: str = PROMPT_VERSION
    provider: str | None = None
    model: str | None = None
    inference_source: SourceViewSummary | None = None


class Usage(StrictModel):
    retrieval_attempts: int = 0
    inference_attempts: int = 0
    model_calls: int = Field(default=0, ge=0)
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_estimate_usd: float | None = None
    pricing_basis: str | None = None
    cache_hit: bool = False


class ExtractionResult(StrictModel):
    job_id: str
    status: Literal["pending", "success", "partial", "failed"]
    data: ProductV1 | dict[str, Any] | None = None
    missing_fields: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    provenance: Provenance
    evidence: dict[str, EvidenceCandidate] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)
    error: AppError | None = None

    @model_validator(mode="after")
    def preserve_registered_data_shape(self):
        # A generic object containing only `name` must not acquire product-only
        # defaults through Pydantic's union selection, including on cache reload.
        if self.provenance.schema_version != SCHEMA_ID and isinstance(self.data, ProductV1):
            self.data = self.data.model_dump(exclude_unset=True)
        return self
