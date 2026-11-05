"""Strict schemas, nested source evidence and completeness for general tasks.

Literal occurrence is necessary, not a proof of arbitrary semantic meaning.
Known presets also check structured field meaning and trusted entity identity;
custom field semantics should be evaluated against reviewed task examples.
"""

from __future__ import annotations

import json
import math
import re
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import unquote, urljoin

from pydantic import BaseModel, ValidationError

from .extraction.generic import _entity, deterministic_extract, populated_fields
from .registry import SchemaSpec
from .schemas import EvidenceCandidate, SourceSnapshot
from .validate import Assessment, _pointer


def _pointer_meaning(path: str, pointer: str, schema_id: str) -> bool:
    if schema_id == "custom_v1":
        return True
    parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/") if part]
    # A native structured representation may already use the business contract's
    # field names (including nested tables/links). That exact field alignment is
    # stronger evidence than a scalar found in an unrelated property.
    if parts and parts[0].isdigit() and parts[1:] == path.split("."):
        return True
    keys = [part for part in parts if not part.isdigit()]
    root = path.split(".", 1)[0]
    allowed = {
        "title": {"headline", "title", "name"},
        "name": {"name"},
        "description": {"description"},
        "biography": {"description", "biography"},
        "text": {"articleBody", "text"},
        "headings": {"headline", "name"},
        "author": {"author"},
        "published_at": {"datePublished", "dateCreated"},
        "language": {"inLanguage"},
        "website": {"url", "sameAs"},
        "emails": {"email"},
        "phone_numbers": {"telephone"},
        "addresses": {"address"},
        "role": {"jobTitle"},
        "organization": {"worksFor", "hiringOrganization"},
        "location": {"location", "jobLocation"},
        "employment_type": {"employmentType"},
        "salary_text": {"baseSalary", "salary"},
        "start_date": {"startDate"},
        "end_date": {"endDate"},
        "organizer": {"organizer"},
    }
    if root == "items":
        return "itemListElement" in keys and bool(keys) and keys[-1] in {"name", "url", "description"}
    if root == "links":
        return bool(keys) and keys[-1] in {"url", "sameAs", "name"}
    if root == "tables":
        return False
    matches = allowed.get(root, set())
    if root in {"author", "organization", "location", "organizer", "addresses", "salary_text"}:
        return bool(matches.intersection(keys))
    return bool(keys) and keys[-1] in matches


def _supports(
    value: Any,
    excerpt: str,
    *,
    resolved_url: bool,
    snapshot: SourceSnapshot,
    contact_scheme: str | None = None,
) -> bool:
    if not excerpt:
        return False
    if isinstance(value, str):
        if contact_scheme:
            if not excerpt.startswith(contact_scheme + ":"):
                return False
            try:
                return value == unquote(excerpt.split(":", 1)[1].split("?", 1)[0], errors="strict")
            except (ValueError, UnicodeError):
                return False
        if resolved_url:
            return value == excerpt or value == urljoin(snapshot.final_url, excerpt)
        return value in excerpt
    if type(value) is bool:
        return excerpt == ("true" if value else "false")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and not math.isfinite(value):
            return False
        # Numeric normalization is deliberately narrow. A number embedded in a
        # word, localized grouping, date or currency amount is not silently parsed.
        if not re.fullmatch(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?", excerpt):
            return False
        try:
            return Decimal(excerpt) == Decimal(str(value))
        except InvalidOperation:
            return False
    return False


def validate_candidate(
    candidate: dict[str, Any] | BaseModel, snapshot: SourceSnapshot, spec: SchemaSpec
) -> Assessment:
    try:
        parsed = spec.candidate_model.model_validate(candidate)
    except (ValidationError, ValueError, TypeError) as exc:
        details = (
            exc.errors(include_input=False)
            if isinstance(exc, ValidationError)
            else [{"msg": "invalid candidate"}]
        )
        return Assessment(
            status="failed", errors=[f"schema:{item.get('loc', '')}:{item['msg']}" for item in details]
        )
    data = parsed.data.model_dump()
    values = populated_fields(data)
    missing = [
        field
        for field in spec.required_fields
        if data.get(field) is None or data.get(field) == [] or data.get(field) == ""
    ]
    errors: list[str] = []
    verified: dict[str, EvidenceCandidate] = {}
    sections = {section.id: section for section in snapshot.sections}
    evidence_by_path: dict[str, list[EvidenceCandidate]] = {}
    for item in parsed.evidence:
        evidence_by_path.setdefault(item.field_path, []).append(item)
    selected, _ = _entity(snapshot, spec)
    for path, value in values.items():
        accepted = None
        url_field = path.rsplit(".", 1)[-1] in {"url", "website"}
        for evidence in evidence_by_path.get(path, []):
            if bool(evidence.section_id) == bool(evidence.json_pointer):
                continue
            occurrence = False
            can_resolve_url = False
            contact_scheme = None
            if evidence.section_id:
                section = sections.get(evidence.section_id)
                if section:
                    occurrence = evidence.excerpt in section.text
                    # Only saved URL attributes can support resolving a relative
                    # link. Arbitrary body text cannot become a guessed URL.
                    can_resolve_url = (
                        url_field
                        and section.representation == "html_attribute"
                        and section.text.endswith(": " + evidence.excerpt)
                    )
                    if (
                        section.representation == "html_attribute"
                        and section.text == "href: " + evidence.excerpt
                    ):
                        root = path.split(".", 1)[0]
                        if root == "emails" and evidence.excerpt.startswith("mailto:"):
                            contact_scheme = "mailto"
                        elif root == "phone_numbers" and evidence.excerpt.startswith("tel:"):
                            contact_scheme = "tel"
            else:
                try:
                    scalar = _pointer(snapshot.ld_json, evidence.json_pointer or "")
                    scalar_text = (
                        scalar
                        if isinstance(scalar, str)
                        else json.dumps(scalar, ensure_ascii=False, separators=(",", ":"))
                    )
                    occurrence = not isinstance(scalar, (dict, list)) and evidence.excerpt == scalar_text
                    occurrence = occurrence and _pointer_meaning(
                        path, evidence.json_pointer or "", spec.schema_id
                    )
                    if selected and spec.schema_id not in {"page_v1", "custom_v1"}:
                        occurrence = occurrence and (evidence.json_pointer or "").startswith(
                            selected[0] + "/"
                        )
                    can_resolve_url = url_field
                except (KeyError, IndexError, ValueError, TypeError):
                    pass
            if occurrence and _supports(
                value,
                evidence.excerpt,
                resolved_url=can_resolve_url,
                snapshot=snapshot,
                contact_scheme=contact_scheme,
            ):
                accepted = evidence
                break
        if accepted:
            verified[path] = accepted
        else:
            errors.append(f"evidence:{path}: missing, non-verbatim, or unsupported value")
    baseline = deterministic_extract(snapshot, spec)
    trusted = baseline.data.model_dump()
    # Comparing scalars preserves entity identity while allowing a semantic
    # extractor to select relevant blocks from a larger source collection.
    for path, value in trusted.items():
        actual = data.get(path)
        if (
            value is not None
            and not isinstance(value, (list, dict))
            and actual is not None
            and actual != value
        ):
            errors.append(f"identity:{path}: differs from the unambiguous deterministic value")
    trusted_warnings = [
        warning
        for warning in baseline.warnings
        if warning.startswith(("conflict:", "ambiguous_entity:", "incomplete_source:"))
    ]
    for warning in trusted_warnings:
        if warning.startswith("conflict:"):
            path = warning.split(":", 2)[1]
            if data.get(path) is not None and data.get(path) != []:
                errors.append(f"conflict:{path}: populated value is ambiguous in source")
        elif (
            spec.schema_id not in {"page_v1", "custom_v1"}
            and warning.startswith("ambiguous_entity:")
            and any(data.get(field) not in (None, [], "") for field in spec.critical_fields)
        ):
            errors.append("ambiguous_entity: cannot establish a unique entity")
    warnings = list(dict.fromkeys(parsed.warnings + trusted_warnings))
    if any(field in missing for field in spec.critical_fields):
        errors.append("completeness: critical fields missing")
    if errors:
        return Assessment(status="failed", missing_fields=missing, warnings=warnings, errors=errors)
    if snapshot.excluded_sections:
        warnings.append("incomplete_source: excluded sections prevent complete extraction")
    incomplete = snapshot.excluded_sections or any(
        warning.startswith("incomplete_source:") for warning in warnings
    )
    return Assessment(
        status="partial" if missing or incomplete else "success",
        data=data,
        evidence=verified,
        missing_fields=missing,
        warnings=warnings,
    )
