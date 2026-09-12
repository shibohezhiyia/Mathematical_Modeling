"""Typed, read-only data contracts for interactive mathematical surfaces."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class InteractiveSurfaceError(ValueError):
    pass


def _finite(value: Any, error: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InteractiveSurfaceError(error) from exc
    if not math.isfinite(result):
        raise InteractiveSurfaceError(error)
    return result


def build_constraint_state(constraints: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(constraints, Sequence) or isinstance(constraints, (str, bytes)) or len(constraints) > 512:
        raise InteractiveSurfaceError("constraints_must_be_bounded_sequence")
    rows = []
    for item in constraints:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise InteractiveSurfaceError("constraint_id_required")
        relation = item.get("relation", "<=")
        if relation not in {"<=", ">=", "="}:
            raise InteractiveSurfaceError("unsupported_constraint_relation")
        value = _finite(item.get("value"), "constraint_value_must_be_finite")
        bound = _finite(item.get("bound"), "constraint_bound_must_be_finite")
        tolerance = _finite(item.get("tolerance", 0.0), "constraint_tolerance_must_be_finite")
        if tolerance < 0:
            raise InteractiveSurfaceError("constraint_tolerance_must_be_nonnegative")
        violation = (max(0.0, value - bound - tolerance) if relation == "<=" else
                     max(0.0, bound - value - tolerance) if relation == ">=" else
                     max(0.0, abs(value - bound) - tolerance))
        rows.append({"id": item["id"].strip(), "relation": relation, "value": value,
                     "bound": bound, "tolerance": tolerance, "violation": violation,
                     "status": "pass" if violation == 0 else "fail"})
    return {"schema_version": "mathmodel.constraint-state/v1", "constraints": rows,
            "all_pass": all(row["status"] == "pass" for row in rows),
            "policy": "display_only_until_trusted_solver_evidence_is_attached"}


def build_sensitivity_surface(points: Sequence[Mapping[str, Any]], *, axes: Sequence[str], metric: str,
                              max_points: int = 512) -> dict[str, Any]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or not 1 <= len(points) <= max_points:
        raise InteractiveSurfaceError("sensitivity_points_out_of_bounds")
    if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)) or not 1 <= len(axes) <= 4:
        raise InteractiveSurfaceError("sensitivity_axes_out_of_bounds")
    if not isinstance(metric, str) or not metric.strip():
        raise InteractiveSurfaceError("sensitivity_metric_required")
    rows = []
    for item in points:
        if not isinstance(item, Mapping):
            raise InteractiveSurfaceError("sensitivity_point_must_be_object")
        values = {str(axis): _finite(item.get(axis), "sensitivity_axis_must_be_finite") for axis in axes}
        values["metric"] = _finite(item.get(metric), "sensitivity_metric_must_be_finite")
        rows.append(values)
    return {"schema_version": "mathmodel.sensitivity-surface/v1", "axes": list(axes),
            "metric": metric, "points": rows,
            "policy": "finite_what_if_surface_not_global_sensitivity_or_causal_effect"}


def build_pareto_front(points: Sequence[Mapping[str, Any]], *, objectives: Mapping[str, str], max_points: int = 512) -> dict[str, Any]:
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or not 1 <= len(points) <= max_points:
        raise InteractiveSurfaceError("pareto_points_out_of_bounds")
    if not isinstance(objectives, Mapping) or not objectives:
        raise InteractiveSurfaceError("pareto_objectives_required")
    if any(direction not in {"min", "max"} for direction in objectives.values()):
        raise InteractiveSurfaceError("pareto_direction_invalid")
    rows = []
    for item in points:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or not item["id"].strip():
            raise InteractiveSurfaceError("pareto_point_id_required")
        rows.append({"id": item["id"].strip(), "objectives": {
            name: _finite(item.get(name), "pareto_objective_must_be_finite") for name in objectives
        }})
    def dominates(left, right):
        comparisons = []
        for name, direction in objectives.items():
            a, b = left["objectives"][name], right["objectives"][name]
            comparisons.append(a <= b if direction == "min" else a >= b)
        strict = any(left["objectives"][name] != right["objectives"][name]
                     for name in objectives)
        return all(comparisons) and strict
    front = [row for row in rows if not any(dominates(other, row) for other in rows if other is not row)]
    return {"schema_version": "mathmodel.pareto-surface/v1", "objectives": dict(objectives),
            "points": rows, "front_ids": [row["id"] for row in front],
            "policy": "unweighted_non_dominated_set_not_a_truth_or_single_winner"}


__all__ = ["InteractiveSurfaceError", "build_constraint_state", "build_sensitivity_surface", "build_pareto_front"]
