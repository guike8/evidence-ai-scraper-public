"""Local schema, source occurrence, field-support and completeness checks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import ValidationError

from .extraction.deterministic import currency_from_text, deterministic_extract, parse_price
from .schemas import (
    CRITICAL_FIELDS,
    REQUIRED_FIELDS,
    EvidenceCandidate,
    ExtractionCandidate,
    ProductV1,
    SourceSnapshot,
)


@dataclass
class Assessment:
    status: str
    data: ProductV1 | dict[str, Any] | None = None
    evidence: dict[str, EvidenceCandidate] = field(default_factory=dict)
    missing_fields: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def populated_fields(product: ProductV1 | dict[str, Any], prefix: str = "") -> dict[str, Any]:
    data = product.model_dump() if isinstance(product, ProductV1) else product
    result: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(populated_fields(value, path))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                result[f"{path}.{i}"] = item
        elif value is not None:
            result[path] = value
    return result


def _pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be absolute")
    value = document
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            if not re.fullmatch(r"0|[1-9]\d*", key):
                raise ValueError("Invalid JSON pointer array index")
            value = value[int(key)]
        else:
            value = value[key]
    return value


def _supports(
    field_path: str, value: Any, excerpt: str, pointer: str | None, locale: str | None = None
) -> bool:
    if not excerpt:
        return False
    if pointer:
        # A matching scalar in an unrelated structured field is not evidence.
        key = pointer.rsplit("/", 1)[-1]
        allowed = {
            "name": {"name"},
            "sku": {"sku"},
            "price": {"price"},
            "price_text": {"price"},
            "currency": {"priceCurrency"},
            "short_description": {"description"},
            "long_description": {"description"},
            "category.name": {"category", "name"},
            "category.url": {"url"},
        }
        if field_path.startswith("images."):
            if not ("/image" in pointer and (key.isdigit() or key in {"image", "url", "contentUrl"})):
                return False
        elif field_path in allowed and key not in allowed[field_path]:
            return False
        elif field_path.startswith("additional_info."):
            return False  # no generic JSON-LD mapping for site-specific attributes
    if field_path == "price":
        parsed = parse_price(excerpt, locale)
        # Explicit localization may have normalized a saved comma price. It is only
        # accepted when the candidate provides that locale normalization in source.
        candidates = [parsed]
        if pointer:
            candidates = [excerpt]
        for candidate in candidates:
            try:
                if candidate is not None and Decimal(candidate) == Decimal(value):
                    return True
            except InvalidOperation:
                pass
        return False
    if field_path == "currency":
        return currency_from_text(excerpt) == value
    return isinstance(value, str) and value in excerpt


def validate_candidate(
    candidate: dict[str, Any] | ExtractionCandidate, snapshot: SourceSnapshot, locale: str | None = None
) -> Assessment:
    try:
        parsed = ExtractionCandidate.model_validate(candidate)
    except (ValidationError, ValueError, TypeError) as exc:
        # Validation errors can include untrusted input; callers should not log raw exceptions.
        details = (
            exc.errors(include_input=False)
            if isinstance(exc, ValidationError)
            else [{"msg": "invalid candidate"}]
        )
        return Assessment(status="failed", errors=[f"schema:{x.get('loc', '')}:{x['msg']}" for x in details])
    values = populated_fields(parsed.data)
    errors: list[str] = []
    verified: dict[str, EvidenceCandidate] = {}
    sections = {s.id: s.text for s in snapshot.sections}
    candidates: dict[str, list[EvidenceCandidate]] = {}
    for item in parsed.evidence:
        candidates.setdefault(item.field_path, []).append(item)
    for path, value in values.items():
        accepted = None
        for evidence in candidates.get(path, []):
            if bool(evidence.section_id) == bool(evidence.json_pointer):
                continue
            occurrence = False
            if evidence.section_id:
                occurrence = evidence.excerpt in sections.get(evidence.section_id, "")
            else:
                try:
                    scalar = _pointer(snapshot.ld_json, evidence.json_pointer or "")
                    scalar_text = (
                        scalar
                        if isinstance(scalar, str)
                        else json.dumps(scalar, ensure_ascii=False, separators=(",", ":"))
                    )
                    occurrence = not isinstance(scalar, (dict, list)) and evidence.excerpt == scalar_text
                except (KeyError, IndexError, ValueError, TypeError):
                    pass
            if occurrence and _supports(path, value, evidence.excerpt, evidence.json_pointer, locale):
                accepted = evidence
                break
        if accepted:
            verified[path] = accepted
        else:
            errors.append(f"evidence:{path}: missing, non-verbatim, or unsupported value")
    # Model warnings cannot resolve a source conflict. Repeat trusted conflict discovery.
    deterministic = deterministic_extract(snapshot, locale)
    trusted_values = populated_fields(deterministic.data)
    for path, trusted in trusted_values.items():
        actual = values.get(path)
        if actual is not None and actual != trusted:
            # Verbatim values from another entity cannot replace trusted selected facts.
            if path == "price" and Decimal(actual) == Decimal(trusted):
                continue
            errors.append(f"identity:{path}: differs from the unambiguous deterministic value")
    trusted_warnings = [w for w in deterministic.warnings if w.startswith(("conflict:", "ambiguous_entity:"))]
    for warning in trusted_warnings:
        if warning.startswith("conflict:"):
            path = warning.split(":", 2)[1]
            if path in values:
                errors.append(f"conflict:{path}: populated value is ambiguous in source")
        elif parsed.data.name is not None:
            errors.append("ambiguous_entity: cannot establish a unique product")
    warnings = list(dict.fromkeys(parsed.warnings + trusted_warnings))
    missing = [path for path in REQUIRED_FIELDS if getattr(parsed.data, path) is None]
    if errors:
        return Assessment(status="failed", missing_fields=missing, warnings=warnings, errors=errors)
    if any(path in missing for path in CRITICAL_FIELDS):
        return Assessment(
            status="failed",
            missing_fields=missing,
            warnings=warnings,
            errors=["completeness: critical product name missing"],
        )
    if snapshot.excluded_sections:
        warnings.append("incomplete_source: excluded sections prevent complete extraction")
    status = "partial" if missing or snapshot.excluded_sections else "success"
    return Assessment(
        status=status, data=parsed.data, evidence=verified, missing_fields=missing, warnings=warnings
    )
