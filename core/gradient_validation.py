"""Backend-independent gradient cross-checks for differentiable model plans."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Callable, Mapping


SCHEMA_VERSION = "mathmodel.gradient-check/v1"


class GradientCheckError(ValueError):
    pass


def _finite(value: Any, code: str = "value_must_be_finite") -> float:
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise GradientCheckError(code) from exc
    if not math.isfinite(value):
        raise GradientCheckError(code)
    return value


def check_gradient(
    objective: Callable[[Mapping[str, float]], float],
    gradient: Callable[[Mapping[str, float]], Mapping[str, float]],
    point: Mapping[str, Any], *, bounds: Mapping[str, Any] | None = None,
    absolute_tolerance: float = 1e-5, relative_tolerance: float = 1e-4,
    step: float | None = None, max_evaluations: int = 512,
) -> dict[str, Any]:
    """Compare a supplied gradient against central/one-sided finite differences."""
    if not callable(objective) or not callable(gradient):
        raise GradientCheckError("objective_and_gradient_must_be_callable")
    if not isinstance(point, Mapping) or not 1 <= len(point) <= 128:
        raise GradientCheckError("point_must_bind_1_to_128_variables")
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 4096:
        raise GradientCheckError("invalid_gradient_evaluation_budget")
    absolute_tolerance, relative_tolerance = _finite(absolute_tolerance, "invalid_tolerance"), _finite(relative_tolerance, "invalid_tolerance")
    if absolute_tolerance < 0 or relative_tolerance < 0 or (absolute_tolerance == 0 and relative_tolerance == 0):
        raise GradientCheckError("positive_gradient_tolerance_required")
    base = {str(name): _finite(value, "point_must_be_finite") for name, value in point.items()}
    if len(base) != len(point) or any(not name for name in base):
        raise GradientCheckError("point_variable_names_must_be_nonempty_and_unique")
    normalized_bounds = {}
    if bounds is not None:
        if not isinstance(bounds, Mapping) or set(bounds) != set(base):
            raise GradientCheckError("bounds_must_cover_point_variables")
        for name, interval in bounds.items():
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise GradientCheckError(f"invalid_bound:{name}")
            lower, upper = _finite(interval[0], "invalid_bound"), _finite(interval[1], "invalid_bound")
            if lower >= upper or not lower <= base[name] <= upper:
                raise GradientCheckError(f"point_outside_bound:{name}")
            normalized_bounds[name] = (lower, upper)
    h0 = None if step is None else _finite(step, "invalid_gradient_step")
    if h0 is not None and h0 <= 0:
        raise GradientCheckError("invalid_gradient_step")
    if max_evaluations < 1 + len(base):
        return {"schema_version": SCHEMA_VERSION, "status": "not_assessed",
                "reason": "gradient_evaluation_budget_exhausted", "evaluations": 0,
                "variables": list(base), "point": base}
    try:
        objective_value = _finite(objective(dict(base)), "objective_returned_non_finite")
        raw_gradient = gradient(dict(base))
        if not isinstance(raw_gradient, Mapping) or set(raw_gradient) != set(base):
            raise GradientCheckError("gradient_must_bind_exactly_point_variables")
        supplied = {name: _finite(raw_gradient[name], "gradient_returned_non_finite") for name in base}
    except GradientCheckError as exc:
        return {"schema_version": SCHEMA_VERSION, "status": "not_assessed", "reason": str(exc),
                "evaluations": 1, "variables": list(base), "point": base}
    evaluations = 1
    numeric, methods, failures = {}, {}, []
    epsilon = math.sqrt(2.220446049250313e-16)
    for name in base:
        h = h0 if h0 is not None else epsilon * max(1.0, abs(base[name]))
        lower, upper = normalized_bounds.get(name, (-math.inf, math.inf))
        can_forward, can_backward = base[name] + h <= upper, base[name] - h >= lower
        if not can_forward and not can_backward:
            failures.append({"variable": name, "reason": "no_feasible_finite_difference_step"})
            continue
        plus = dict(base); minus = dict(base)
        if can_forward and can_backward:
            plus[name] += h; minus[name] -= h
            try:
                plus_value, minus_value = _finite(objective(plus), "objective_returned_non_finite"), _finite(objective(minus), "objective_returned_non_finite")
            except GradientCheckError as exc:
                failures.append({"variable": name, "reason": str(exc)})
                continue
            evaluations += 2
            numeric[name], methods[name] = (plus_value - minus_value) / (2.0 * h), "central"
        else:
            if can_forward:
                plus[name] += h
            else:
                minus[name] -= h
            nearby = plus if can_forward else minus
            try:
                nearby_value = _finite(objective(nearby), "objective_returned_non_finite")
            except GradientCheckError as exc:
                failures.append({"variable": name, "reason": str(exc)})
                continue
            evaluations += 1
            numeric[name] = (nearby_value - objective_value) / h if can_forward else (objective_value - nearby_value) / h
            methods[name] = "forward" if can_forward else "backward"
    rows = []
    for name in base:
        if name not in numeric:
            continue
        difference = abs(supplied[name] - numeric[name])
        tolerance = absolute_tolerance + relative_tolerance * max(abs(supplied[name]), abs(numeric[name]))
        rows.append({"variable": name, "supplied": supplied[name], "finite_difference": numeric[name],
                     "absolute_error": difference, "tolerance": tolerance, "method": methods[name],
                     "status": "pass" if difference <= tolerance else "fail"})
    status = "not_assessed" if failures or len(rows) != len(base) else "fail" if any(row["status"] == "fail" for row in rows) else "pass"
    return {"schema_version": SCHEMA_VERSION, "status": status, "objective_value": objective_value,
            "point": base, "variables": list(base), "checks": rows, "failures": failures,
            "evaluations": evaluations, "policy": "finite_difference_cross_check_not_a_global_gradient_proof",
            "input_sha256": sha256(json.dumps(base, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def gradient_optimization_gate(report: Mapping[str, Any]) -> dict[str, Any]:
    """Allow derivative-accelerated optimisation only after a passing check."""
    if not isinstance(report, Mapping):
        raise GradientCheckError("gradient_report_required")
    status = str(report.get("status", "not_assessed"))
    allowed = status == "pass"
    return {"status": "allowed" if allowed else "blocked", "gradient_status": status,
            "use_gradient": allowed,
            "reason": "finite_difference_check_passed" if allowed else "gradient_check_not_passed",
            "policy": "finite_difference_check_is_a local gate, not a global gradient proof"}


__all__ = ["SCHEMA_VERSION", "GradientCheckError", "check_gradient", "gradient_optimization_gate"]
