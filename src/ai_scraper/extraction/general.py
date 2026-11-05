"""Compatibility facade for the general extraction and validation boundary."""

from ..validate_generic import validate_candidate
from .generic import deterministic_extract, populated_fields
from .generic import merge_semantic_gaps as merge_candidates

__all__ = ["deterministic_extract", "merge_candidates", "populated_fields", "validate_candidate"]
