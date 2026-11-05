"""Optional transport constraint for the application's two evidence forms.

This does not verify evidence or rewrite model output. Production validation
still checks reference exclusivity, saved-source occurrence and field support.
The adapter must explicitly opt in after its provider supports this schema.
"""

from __future__ import annotations

import copy
from typing import Any

_DEFINITION = "EvidenceCandidate"
_REFERENCE = "#/$defs/" + _DEFINITION
_FIELDS = frozenset({"field_path", "section_id", "excerpt", "json_pointer"})


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
