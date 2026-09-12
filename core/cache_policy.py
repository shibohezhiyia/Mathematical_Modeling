"""Explicit policies for reusable intermediate artifacts and failures."""

from __future__ import annotations

from typing import Any, Mapping


class CachePolicyError(ValueError):
    pass


_NON_CACHEABLE_FAILURES = frozenset({
    "timeout", "network_failure", "resource_exhausted", "budget_exhausted",
    "counterexample_search_incomplete", "not_assessed",
})


def classify_cache_failure(status: str) -> dict[str, Any]:
    if not isinstance(status, str) or not status.strip():
        raise CachePolicyError("failure_status_required")
    normalized = status.strip().lower()
    cacheable = normalized in {"deterministic_type_error", "schema_error", "invalid_input_contract"}
    return {
        "schema_version": "mathmodel.cache-policy/v1",
        "status": "cacheable_failure" if cacheable else "do_not_cache_failure",
        "failure_status": normalized,
        "cacheable": cacheable,
        "policy": "timeouts_network_resource_and_incomplete_search_never_become_permanent_verdicts",
    }


def validate_cache_hit(
    cached_identity: Mapping[str, Any],
    expected_identity: Mapping[str, Any],
    *,
    conclusion: bool = False,
) -> dict[str, Any]:
    """Check source/version/domain dimensions before reusing an artifact."""
    if not isinstance(cached_identity, Mapping) or not isinstance(expected_identity, Mapping):
        raise CachePolicyError("cache_identities_must_be_mappings")
    dimensions = ("source_signature", "compiler_version", "domain_signature", "data_view_digest", "preprocessing_digest")
    missing = [key for key in dimensions if key not in cached_identity or key not in expected_identity]
    mismatched = [key for key in dimensions if key in cached_identity and key in expected_identity
                  and cached_identity[key] != expected_identity[key]]
    if conclusion and missing:
        return {"status": "not_reusable", "reason": "incomplete_evidence_key", "missing": missing, "mismatched": mismatched}
    return {
        "status": "reusable" if not missing and not mismatched else "not_reusable",
        "reason": "all_identity_dimensions_match" if not missing and not mismatched else "identity_mismatch_or_incomplete",
        "missing": missing,
        "mismatched": mismatched,
        "conclusion_reuse_allowed": bool(not conclusion or (not missing and not mismatched)),
    }


def warm_start_gate(
    parameter_mapping: Mapping[str, str] | None,
    *,
    structure_unchanged: bool,
    cold_start_available: bool,
) -> dict[str, Any]:
    if parameter_mapping is not None and not isinstance(parameter_mapping, Mapping):
        raise CachePolicyError("parameter_mapping_must_be_mapping")
    if not isinstance(structure_unchanged, bool) or not isinstance(cold_start_available, bool):
        raise CachePolicyError("warm_start_flags_must_be_boolean")
    mapping_ok = bool(parameter_mapping) if parameter_mapping is not None else False
    allowed = structure_unchanged and mapping_ok and cold_start_available
    return {
        "schema_version": "mathmodel.cache-policy/v1",
        "status": "warm_start_allowed" if allowed else "cold_start_required",
        "structure_unchanged": structure_unchanged,
        "parameter_mapping_valid": mapping_ok,
        "cold_start_comparison_required": cold_start_available,
        "policy": "warm_start_is_optional_and_never_replaces_cold_start_comparison",
    }


__all__ = ["CachePolicyError", "classify_cache_failure", "validate_cache_hit", "warm_start_gate"]
