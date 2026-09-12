"""Evidence-gated reuse plan for views, features, sparse blocks and warm starts."""

from __future__ import annotations

from typing import Any, Mapping


class ReusePlanError(ValueError):
    pass


def build_reuse_plan(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    parameter_mapping: Mapping[str, str] | None = None,
    cold_start_available: bool = True,
) -> dict[str, Any]:
    if not isinstance(previous, Mapping) or not isinstance(current, Mapping):
        raise ReusePlanError("snapshots_must_be_mappings")
    if not isinstance(cold_start_available, bool):
        raise ReusePlanError("cold_start_available_must_be_boolean")
    dimensions = {
        "data_view": "data_view_digest", "training_features": "training_features_digest",
        "sparse_structure": "sparse_structure_digest", "key_partition": "key_partition_digest",
    }
    matches = {name: previous.get(key) is not None and previous.get(key) == current.get(key)
               for name, key in dimensions.items()}
    structure_unchanged = all(matches.values())
    mapping_valid = parameter_mapping is not None and isinstance(parameter_mapping, Mapping) and bool(parameter_mapping)
    warm_allowed = bool(structure_unchanged and mapping_valid and cold_start_available)
    reusable = [name for name, matched in matches.items() if matched]
    return {
        "schema_version": "mathmodel.reuse-plan/v1",
        "status": "reusable_intermediates" if reusable else "cold_compile_required",
        "matches": matches,
        "reusable_intermediates": reusable,
        "parameter_mapping_valid": mapping_valid,
        "warm_start": "allowed" if warm_allowed else "denied",
        "cold_start_comparison_required": cold_start_available,
        "policy": "only_matching_view_feature_sparse_and_key_signatures_reuse; final_verdict_never_cached",
    }


__all__ = ["ReusePlanError", "build_reuse_plan"]
