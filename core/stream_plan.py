"""Plans for memory-bounded streaming, joins, and external-state operations."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class StreamPlanError(ValueError):
    pass


def estimate_join_cardinality(
    left_key_counts: Mapping[Any, int],
    right_key_counts: Mapping[Any, int],
    *,
    max_output_rows: int = 2_000_000,
    aggregation_candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Estimate an exact upper bound for an equi-join from key histograms.

    A many-to-many key contributes ``left_count * right_count`` rows.  If the
    bound is too large, the planner refuses to aggregate unless the caller
    supplies a semantically named candidate; it never silently changes the
    problem.
    """
    if not isinstance(left_key_counts, Mapping) or not isinstance(right_key_counts, Mapping):
        raise StreamPlanError("key_counts_must_be_mappings")
    if type(max_output_rows) is not int or max_output_rows < 1:
        raise StreamPlanError("invalid_max_output_rows")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0
           for value in (*left_key_counts.values(), *right_key_counts.values())):
        raise StreamPlanError("key_counts_must_be_nonnegative_integers")
    total = 0
    overlapping = []
    for key in left_key_counts.keys() & right_key_counts.keys():
        left, right = left_key_counts[key], right_key_counts[key]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in (left, right)):
            raise StreamPlanError("key_counts_must_be_nonnegative_integers")
        total += left * right
        overlapping.append({"key": key, "left_rows": left, "right_rows": right, "output_rows": left * right})
    candidates = []
    for candidate in aggregation_candidates or ():
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("id"), str) or not candidate["id"].strip():
            raise StreamPlanError("aggregation_candidate_requires_id")
        if candidate.get("status") not in {"proposed", "approved"}:
            raise StreamPlanError("aggregation_candidate_status_invalid")
        candidates.append(dict(candidate))
    return {
        "schema_version": "mathmodel.stream-plan/v1",
        "status": "admitted" if total <= max_output_rows else "needs_semantic_aggregation",
        "estimated_output_rows": total,
        "max_output_rows": max_output_rows,
        "overlapping_key_count": len(overlapping),
        "largest_key_output_rows": max((item["output_rows"] for item in overlapping), default=0),
        "aggregation_candidates": candidates,
        "policy": "cardinality_is_an_upper_bound_from_key_histograms; aggregation_requires_semantic_approval",
    }


def plan_external_state_operation(
    operation: str,
    *,
    projected_columns: Sequence[str],
    state_protocol: str | None = None,
    estimated_rows: int | None = None,
) -> dict[str, Any]:
    """Return an explicit plan for operations that cannot be chunk-independent."""
    if not isinstance(operation, str) or operation not in {"sort", "group", "join", "window"}:
        raise StreamPlanError("unsupported_stream_operation")
    if not isinstance(projected_columns, Sequence) or isinstance(projected_columns, (str, bytes)) or not projected_columns:
        raise StreamPlanError("projected_columns_required")
    columns = [str(column).strip() for column in projected_columns]
    if any(not column for column in columns) or len(set(columns)) != len(columns):
        raise StreamPlanError("projected_columns_must_be_unique")
    if estimated_rows is not None and (type(estimated_rows) is not int or estimated_rows < 0):
        raise StreamPlanError("estimated_rows_invalid")
    requires_state = operation in {"sort", "join", "window"}
    accepted = (not requires_state) or (isinstance(state_protocol, str) and bool(state_protocol.strip()))
    return {
        "schema_version": "mathmodel.stream-plan/v1",
        "status": "admitted" if accepted else "needs_external_state_protocol",
        "operation": operation,
        "projected_columns": columns,
        "estimated_rows": estimated_rows,
        "state_protocol": state_protocol.strip() if isinstance(state_protocol, str) and state_protocol.strip() else None,
        "policy": "projection_is_explicit; global-order_and_cross-block_state_require_external_protocol",
    }


__all__ = ["StreamPlanError", "estimate_join_cardinality", "plan_external_state_operation"]
