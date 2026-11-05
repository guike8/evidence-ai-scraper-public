"""Experimental excerpt-first grammar over literal included source quotes.

The grammar makes a selected quote and its source locator inseparable. It does
not establish field meaning, select business values, repair model output or
replace application validation. Whole short sections, literal numeric/boolean
label suffixes, href/src attribute suffixes and JSON-LD scalars are the only
quote candidates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from .evidence_schema import constrain_evidence_references, source_evidence_catalog

SOURCE_QUOTE_SCHEMA_VERSION = "source-quotes-excerpt-first-v1"
MAX_QUOTE_CHARS = 512
MAX_QUOTE_BRANCHES = 512
MAX_SCHEMA_BYTES = 1_000_000
MAX_SOURCE_TEXT_BYTES = 2_000_000
_DEFINITION = "EvidenceCandidate"
_REFERENCE = "#/$defs/" + _DEFINITION
_ORDER = ("field_path", "excerpt", "section_id", "json_pointer")
_FIELDS = frozenset(_ORDER)
_NUMBER = re.compile(r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z")


class SourceQuoteSchemaError(ValueError):
    """Invalid or over-budget source/schema; contains no source content."""


@dataclass(frozen=True)
class SourceQuoteSchema:
    schema: dict[str, Any]
    _metadata: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        """Return counters and a digest without literal source content."""
        return copy.deepcopy(self._metadata)


def _encoded(value: Any) -> bytes:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
    except (ValueError, TypeError, RecursionError, UnicodeError):
        raise SourceQuoteSchemaError("Source quote schema is not bounded strict JSON") from None


def _canonical_branches(schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Copy only the exact canonical fixed-object evidence contract."""
    if not isinstance(schema, dict):
        raise SourceQuoteSchemaError("Source quote binding requires a candidate schema")
    try:
        result = constrain_evidence_references(schema)
        evidence = result["properties"]["evidence"]
        definition = result["$defs"][_DEFINITION]
    except (KeyError, TypeError, ValueError, RecursionError):
        raise SourceQuoteSchemaError("Source quote binding requires canonical evidence") from None
    if (
        not isinstance(evidence, dict)
        or evidence.get("type") != "array"
        or evidence.get("items") != {"$ref": _REFERENCE}
        or not isinstance(definition, dict)
        or not isinstance(definition.get("anyOf"), list)
        or not 1 <= len(definition["anyOf"]) <= 2
    ):
        raise SourceQuoteSchemaError("Source quote binding requires canonical evidence")
    forms = {}
    for branch in definition["anyOf"]:
        if not isinstance(branch, dict):
            raise SourceQuoteSchemaError("Malformed source quote evidence branch")
        fields = branch.get("properties")
        if (
            branch.get("type") != "object"
            or branch.get("additionalProperties") is not False
            or not isinstance(fields, dict)
            or set(fields) != _FIELDS
            or not isinstance(branch.get("required"), list)
            or len(branch["required"]) != len(_FIELDS)
            or set(branch["required"]) != _FIELDS
            or any(not isinstance(fields[name], dict) for name in _FIELDS)
            or fields["field_path"].get("type") != "string"
            or fields["excerpt"].get("type") != "string"
        ):
            raise SourceQuoteSchemaError("Malformed source quote evidence branch")
        form = (fields["section_id"].get("type"), fields["json_pointer"].get("type"))
        if form not in {("string", "null"), ("null", "string")} or form in forms:
            raise SourceQuoteSchemaError("Malformed source quote reference exclusivity")
        forms[form] = branch
    return result, forms


def _literal_allowed(schema: dict[str, Any], value: str | None) -> bool:
    """Filter known scalar restrictions while retaining every original keyword."""
    if schema.get("type") != ("null" if value is None else "string"):
        return False
    if "const" in schema and schema["const"] != value:
        return False
    if "enum" in schema:
        if not isinstance(schema["enum"], list):
            raise SourceQuoteSchemaError("Malformed source quote scalar enum")
        if value not in schema["enum"]:
            return False
    if value is not None:
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", math.inf):
            return False
        if "pattern" in schema:
            try:
                if re.search(schema["pattern"], value) is None:
                    return False
            except (re.error, TypeError):
                raise SourceQuoteSchemaError("Malformed source quote scalar pattern") from None
    return True


def _resolve(document: Any, pointer: str) -> Any:
    value = document
    for part in pointer[1:].split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        value = value[int(key)] if isinstance(value, list) else value[key]
    return value


def _scalar_suffix(text: str, maximum: int) -> str | None:
    label, delimiter, suffix = text.rpartition(": ")
    if not label or not delimiter or len(suffix) > maximum:
        return None
    if suffix in {"true", "false"}:
        return suffix
    if _NUMBER.fullmatch(suffix):
        try:
            value = json.loads(suffix)
        except ValueError:
            return None
        if type(value) is int or (type(value) is float and math.isfinite(value)):
            return suffix
    return None


def bind_source_quote_evidence(
    compiled_schema: dict[str, Any],
    source: dict[str, Any],
    *,
    max_quote_chars: int = MAX_QUOTE_CHARS,
    max_branches: int = MAX_QUOTE_BRANCHES,
    max_schema_bytes: int = MAX_SCHEMA_BYTES,
) -> SourceQuoteSchema:
    """Bind excerpt-first literal branches to the exact inference source view.

    Long or empty literals are omitted explicitly in coverage metadata; they
    are never clipped into invented quotes. Branch, node, depth, source-text and
    schema-byte exhaustion raises instead of silently selecting a catalog subset.
    The caller must opt in to this research protocol and inspect its coverage.
    """
    for value, maximum in (
        (max_quote_chars, MAX_QUOTE_CHARS),
        (max_branches, MAX_QUOTE_BRANCHES),
        (max_schema_bytes, MAX_SCHEMA_BYTES),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise SourceQuoteSchemaError("Invalid source quote schema limit")
    if len(_encoded(compiled_schema)) > max_schema_bytes:
        raise SourceQuoteSchemaError("Source quote schema byte budget exceeded")
    result, forms = _canonical_branches(compiled_schema)
    try:
        catalog = source_evidence_catalog(source)
    except (TypeError, ValueError, RecursionError, UnicodeError):
        raise SourceQuoteSchemaError("Invalid or over-budget included source catalog") from None
    metadata = {
        "policy_version": SOURCE_QUOTE_SCHEMA_VERSION,
        "max_quote_chars": max_quote_chars,
        "max_branches": max_branches,
        "max_schema_bytes": max_schema_bytes,
        "section_id_count": len(catalog.section_ids),
        "json_pointer_count": len(catalog.json_pointers),
        "section_quote_count": 0,
        "scalar_suffix_quote_count": 0,
        "attribute_suffix_quote_count": 0,
        "json_scalar_quote_count": 0,
        "excluded_long_sections": 0,
        "excluded_long_scalars": 0,
        "excluded_empty_literals": 0,
        "excluded_by_schema": 0,
    }
    branches, identities = [], []
    seen: set[tuple[str, str | None, str | None]] = set()
    source_bytes = 0
    # Remove only the canonical evidence definition. Incremental accounting
    # keeps a large copied field constraint from multiplying beyond the budget.
    result["$defs"][_DEFINITION] = {"anyOf": []}
    encoded_size = len(_encoded(result))

    def count_text(text: str) -> None:
        nonlocal source_bytes
        if len(text) > MAX_SOURCE_TEXT_BYTES:
            raise SourceQuoteSchemaError("Included source text byte budget exceeded")
        source_bytes += len(text.encode())
        if source_bytes > MAX_SOURCE_TEXT_BYTES:
            raise SourceQuoteSchemaError("Included source text byte budget exceeded")

    def add(quote: str, section_id: str | None, pointer: str | None, counter: str) -> None:
        nonlocal encoded_size
        identity = (quote, section_id, pointer)
        if identity in seen:
            return
        seen.add(identity)
        form = ("string", "null") if section_id is not None else ("null", "string")
        template = forms.get(form)
        if template is None:
            metadata["excluded_by_schema"] += 1
            return
        literals = {"excerpt": quote, "section_id": section_id, "json_pointer": pointer}
        if not all(_literal_allowed(template["properties"][key], value) for key, value in literals.items()):
            metadata["excluded_by_schema"] += 1
            return
        branch = copy.deepcopy(template)
        branch["properties"] = {key: branch["properties"][key] for key in _ORDER}
        branch["required"] = list(_ORDER)
        for key, value in literals.items():
            branch["properties"][key]["const"] = value
        if len(branches) >= max_branches:
            raise SourceQuoteSchemaError("Source quote branch budget exceeded")
        encoded_size += len(_encoded(branch)) + bool(branches)
        if encoded_size > max_schema_bytes:
            raise SourceQuoteSchemaError("Source quote schema byte budget exceeded")
        branches.append(branch)
        identities.append(identity)
        metadata[counter] += 1

    try:
        for section in source["sections"]:
            text = section["text"]
            count_text(text)
            if not text:
                metadata["excluded_empty_literals"] += 1
            elif len(text) > max_quote_chars:
                metadata["excluded_long_sections"] += 1
            else:
                add(text, section["id"], None, "section_quote_count")
            suffix = _scalar_suffix(text, max_quote_chars)
            if suffix is not None:
                add(suffix, section["id"], None, "scalar_suffix_quote_count")
            for prefix in ("href: ", "src: "):
                if text.startswith(prefix):
                    attribute = text[len(prefix) :]
                    if attribute and len(attribute) <= max_quote_chars:
                        add(attribute, section["id"], None, "attribute_suffix_quote_count")
        for pointer in catalog.json_pointers:
            value = _resolve(source["ld_json"], pointer)
            text = value if isinstance(value, str) else _encoded(value).decode()
            count_text(text)
            if not text:
                metadata["excluded_empty_literals"] += 1
            elif len(text) > max_quote_chars:
                metadata["excluded_long_scalars"] += 1
            else:
                add(text, None, pointer, "json_scalar_quote_count")
    except (KeyError, IndexError, TypeError, UnicodeError, RecursionError):
        raise SourceQuoteSchemaError("Invalid included source literals") from None
    evidence = result["properties"]["evidence"]
    if branches:
        result["$defs"][_DEFINITION] = {"anyOf": branches}
    else:
        if evidence.get("minItems", 0) > 0:
            raise SourceQuoteSchemaError("Required evidence has no permitted source quotes")
        evidence["maxItems"] = 0
        # Keep a valid canonical definition: its array is empty-only.
        result["$defs"][_DEFINITION] = copy.deepcopy(next(iter(forms.values())))
    actual_bytes = len(_encoded(result))
    if actual_bytes > max_schema_bytes:
        raise SourceQuoteSchemaError("Source quote schema byte budget exceeded")
    omitted = sum(
        metadata[key]
        for key in (
            "excluded_long_sections",
            "excluded_long_scalars",
            "excluded_empty_literals",
            "excluded_by_schema",
        )
    )
    metadata.update(
        quote_pair_count=len(identities),
        branch_count=len(branches),
        schema_bytes=actual_bytes,
        source_text_bytes=source_bytes,
        catalog_hash=hashlib.sha256(_encoded(identities)).hexdigest(),
        coverage="partial" if omitted else ("complete" if branches else "empty"),
    )
    return SourceQuoteSchema(result, metadata)
