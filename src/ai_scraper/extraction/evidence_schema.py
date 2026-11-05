"""Optional transport constraint for the application's two evidence forms.

This does not verify evidence or rewrite model output. Production validation
still checks reference exclusivity, saved-source occurrence and field support.
The adapter must explicitly opt in after its provider supports this schema.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

_DEFINITION = "EvidenceCandidate"
_REFERENCE = "#/$defs/" + _DEFINITION
_FIELDS = frozenset({"field_path", "section_id", "excerpt", "json_pointer"})
SOURCE_BOUND_EVIDENCE_VERSION = "source-bound-evidence-v1"
MAX_CATALOG_LOCATORS = 4096
MAX_CATALOG_BYTES = 65536
MAX_CATALOG_NODES = 32768
MAX_CATALOG_DEPTH = 64
MAX_LOCATOR_LENGTH = 2048


class EvidenceCatalogLimitError(ValueError):
    """The complete source locator catalog cannot fit the declared budget."""


@dataclass(frozen=True)
class EvidenceCatalog:
    """Locators from the exact inference payload, never the full excluded source."""

    section_ids: tuple[str, ...]
    json_pointers: tuple[str, ...]

    def metadata(self) -> dict[str, Any]:
        identity = {"section_ids": self.section_ids, "json_pointers": self.json_pointers}
        encoded = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode()
        return {
            "policy_version": SOURCE_BOUND_EVIDENCE_VERSION,
            "catalog_hash": hashlib.sha256(encoded).hexdigest(),
            "section_id_count": len(self.section_ids),
            "json_pointer_count": len(self.json_pointers),
        }


def source_evidence_catalog(source: dict[str, Any]) -> EvidenceCatalog:
    """Enumerate included section IDs and existing non-null JSON-LD scalars.

    Null represents absent data and is not a usable populated-field citation.
    Iterative traversal has explicit node, depth, locator and byte budgets; an
    exhausted budget raises rather than silently omitting possible references.
    """
    if not isinstance(source, dict):
        raise TypeError("Source evidence catalog requires an object")
    sections, ld_json = source.get("sections"), source.get("ld_json")
    if not isinstance(sections, list) or not isinstance(ld_json, list):
        raise TypeError("Source evidence catalog requires section and JSON-LD arrays")
    if len(sections) + len(ld_json) > MAX_CATALOG_NODES:
        raise EvidenceCatalogLimitError("Source evidence catalog node budget exceeded")
    section_ids, pointers = [], []
    seen_sections, seen_pointers = set(), set()
    locator_bytes = 0

    def add(value: str, output: list[str], seen: set[str]) -> None:
        nonlocal locator_bytes
        if len(value) > MAX_LOCATOR_LENGTH:
            raise EvidenceCatalogLimitError("Source evidence locator length exceeded")
        if value in seen:
            return
        if len(seen_sections) + len(seen_pointers) >= MAX_CATALOG_LOCATORS:
            raise EvidenceCatalogLimitError("Source evidence locator budget exceeded")
        locator_bytes += len(value.encode("utf-8"))
        if locator_bytes > MAX_CATALOG_BYTES:
            raise EvidenceCatalogLimitError("Source evidence catalog byte budget exceeded")
        seen.add(value)
        output.append(value)

    for section in sections:
        if (
            not isinstance(section, dict)
            or not isinstance(section.get("id"), str)
            or not section["id"]
            or not isinstance(section.get("text"), str)
        ):
            raise ValueError("Malformed included source section")
        if section["id"] in seen_sections:
            raise ValueError("Included source section IDs must be unique")
        add(section["id"], section_ids, seen_sections)

    stack = [("", ld_json, 0)]
    visited = len(sections) + 1
    while stack:
        pointer, value, depth = stack.pop()
        if depth > MAX_CATALOG_DEPTH:
            raise EvidenceCatalogLimitError("Source evidence catalog depth exceeded")
        if isinstance(value, (dict, list)):
            visited += len(value)
            if visited > MAX_CATALOG_NODES:
                raise EvidenceCatalogLimitError("Source evidence catalog node budget exceeded")
            children = value.items() if isinstance(value, dict) else enumerate(value)
            # The node check precedes materialization, keeping a wide object
            # from allocating an unbounded traversal stack.
            for key, child in reversed(list(children)):
                if not isinstance(key, (str, int)) or isinstance(key, bool):
                    raise TypeError("Malformed JSON-LD property")
                escaped = str(key).replace("~", "~0").replace("/", "~1")
                child_pointer = pointer + "/" + escaped
                if len(child_pointer) > MAX_LOCATOR_LENGTH:
                    raise EvidenceCatalogLimitError("Source evidence locator length exceeded")
                stack.append((child_pointer, child, depth + 1))
        elif value is None:
            continue
        elif type(value) in {str, int, float, bool}:
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("JSON-LD evidence scalars must be finite")
            add(pointer, pointers, seen_pointers)
        else:
            raise ValueError("Unsupported JSON-LD evidence scalar")
    return EvidenceCatalog(tuple(section_ids), tuple(pointers))


def _reference_branch(schema: dict[str, Any], kind: str) -> dict[str, Any]:
    """Select a nullable reference branch without dropping scalar constraints."""
    alternatives = schema.get("anyOf")
    if (
        not isinstance(alternatives, list)
        or len(alternatives) != 2
        or any(not isinstance(item, dict) for item in alternatives)
        or {item.get("type") for item in alternatives} != {"string", "null"}
    ):
        raise ValueError("Canonical evidence references must be nullable strings")
    selected = next(item for item in alternatives if item["type"] == kind)
    result = {}
    for key, value in schema.items():
        if key == "anyOf":
            for branch_key, branch_value in selected.items():
                if branch_key in result and result[branch_key] != branch_value:
                    raise ValueError("Conflicting evidence reference constraints cannot be combined")
                result[branch_key] = copy.deepcopy(branch_value)
        else:
            if key in result and result[key] != value:
                raise ValueError("Conflicting evidence reference constraints cannot be combined")
            result[key] = copy.deepcopy(value)
    return result


def constrain_evidence_references(compiled_schema: dict[str, Any]) -> dict[str, Any]:
    """Copy a trusted provider schema and require exactly one evidence locator.

    Only the canonical ``EvidenceCandidate`` definition directly referenced by
    the root candidate's evidence array is eligible. Unrelated definitions and
    schemas with a different property shape are preserved. A canonical shape
    with malformed types/fixed-object constraints raises a bounded error rather
    than silently weakening its constraints. Already constrained schemas are
    unchanged, making this operation idempotent.
    """
    if not isinstance(compiled_schema, dict):
        raise TypeError("Provider schema must be an object")
    result = copy.deepcopy(compiled_schema)
    properties = result.get("properties")
    if not isinstance(properties, dict):
        return result
    evidence = properties.get("evidence")
    if not isinstance(evidence, dict) or evidence.get("type") != "array":
        return result
    items = evidence.get("items")
    definitions = result.get("$defs")
    if not isinstance(items, dict) or items.get("$ref") != _REFERENCE or not isinstance(definitions, dict):
        return result
    original = definitions.get(_DEFINITION)
    if not isinstance(original, dict):
        return result
    fields = original.get("properties")
    if not isinstance(fields, dict) or set(fields) != _FIELDS:
        return result
    if (
        original.get("type") != "object"
        or original.get("additionalProperties") is not False
        or not isinstance(original.get("required"), list)
        or len(original["required"]) != len(_FIELDS)
        or set(original["required"]) != _FIELDS
        or any(not isinstance(fields[name], dict) for name in _FIELDS)
        or fields["field_path"].get("type") != "string"
        or fields["excerpt"].get("type") != "string"
    ):
        raise ValueError("Canonical evidence definition must be a fixed four-property object")
    branches = []
    for section_kind, pointer_kind in (("string", "null"), ("null", "string")):
        branch = copy.deepcopy(original)
        # Updating existing keys preserves the schema's declared property order.
        branch["properties"]["section_id"] = _reference_branch(fields["section_id"], section_kind)
        branch["properties"]["json_pointer"] = _reference_branch(fields["json_pointer"], pointer_kind)
        branch["required"] = list(fields)
        branch["additionalProperties"] = False
        branches.append(branch)
    definitions[_DEFINITION] = {"anyOf": branches}
    return result


def bind_source_evidence_references(
    compiled_schema: dict[str, Any], catalog: EvidenceCatalog
) -> dict[str, Any]:
    """Restrict a trusted evidence grammar to this exact inference source.

    This prevents invented locators, not wrong quotes or field meanings. Keep
    the independent application validators unchanged. A source without any
    usable locator permits an empty evidence array only; it cannot make a
    populated field pass evidence validation.
    """
    result = constrain_evidence_references(compiled_schema)
    properties, definitions = result.get("properties"), result.get("$defs")
    if not isinstance(properties, dict) or not isinstance(definitions, dict):
        raise TypeError("Source binding requires the canonical evidence array")
    evidence = properties.get("evidence")
    definition = definitions.get(_DEFINITION)
    if (
        not isinstance(evidence, dict)
        or evidence.get("type") != "array"
        or evidence.get("items") != {"$ref": _REFERENCE}
        or not isinstance(definition, dict)
    ):
        raise ValueError("Source binding requires the canonical evidence array")
    alternatives = definition.get("anyOf")
    if not isinstance(alternatives, list) or len(alternatives) != 2:
        raise ValueError("Source binding requires exclusive canonical evidence branches")
    branches = []
    seen_forms = set()
    for original in alternatives:
        if not isinstance(original, dict):
            raise TypeError("Malformed exclusive evidence branch")
        fields = original.get("properties", {})
        if (
            original.get("type") != "object"
            or original.get("additionalProperties") is not False
            or set(fields) != _FIELDS
            or original.get("required") != list(fields)
            or any(not isinstance(fields[name], dict) for name in _FIELDS)
        ):
            raise ValueError("Source binding requires fixed canonical evidence branches")
        form = (fields["section_id"].get("type"), fields["json_pointer"].get("type"))
        if form not in {("string", "null"), ("null", "string")} or form in seen_forms:
            raise ValueError("Malformed exclusive evidence reference types")
        seen_forms.add(form)
        field, values = (
            ("section_id", catalog.section_ids)
            if form[0] == "string"
            else ("json_pointer", catalog.json_pointers)
        )
        existing = fields[field].get("enum")
        if existing is not None:
            if not isinstance(existing, list):
                raise ValueError("Evidence reference enum must be an array")
            values = tuple(value for value in values if value in existing)
        if values:
            branch = copy.deepcopy(original)
            branch["properties"][field]["enum"] = list(values)
            branches.append(branch)
    if branches:
        result["$defs"][_DEFINITION] = {"anyOf": branches}
    else:
        if evidence.get("minItems", 0) > 0:
            raise ValueError("Required evidence cannot be bound to an empty source")
        evidence["maxItems"] = 0
    return result
