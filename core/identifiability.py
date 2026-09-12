"""Bounded local identifiability diagnostics for parameterized predictions."""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.identifiability/v1"


class IdentifiabilityError(ValueError):
    pass


def _prediction(value: Any) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise IdentifiabilityError("predictor_must_return_numeric_array") from exc
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise IdentifiabilityError("predictor_must_return_finite_1d_array")
    return result


def assess_local_identifiability(
    predict: Callable[[Mapping[str, float]], Sequence[float]],
    parameters: Mapping[str, float],
    *, bounds: Mapping[str, Sequence[float]] | None = None,
    relative_step: float = 1e-5, rank_tolerance: float = 1e-8,
    max_evaluations: int = 128,
) -> dict[str, Any]:
    """Estimate local parameter identifiability using finite-difference responses.

    ``identified`` means only that the local response Jacobian has numerical
    rank at this point. It is not a global identifiability or causal certificate.
    """
    if not parameters or len(parameters) > 32:
        raise IdentifiabilityError("parameters_require_1_to_32_entries")
    names = list(parameters)
    try:
        point = {name: float(parameters[name]) for name in names}
    except (TypeError, ValueError, OverflowError) as exc:
        raise IdentifiabilityError("parameters_must_be_finite_numbers") from exc
    if not all(math.isfinite(value) for value in point.values()):
        raise IdentifiabilityError("parameters_must_be_finite_numbers")
    if not 0 < float(relative_step) < 0.5 or not 0 < float(rank_tolerance) < 1:
        raise IdentifiabilityError("invalid_identifiability_tolerance")
    if type(max_evaluations) is not int or max_evaluations < 1:
        raise IdentifiabilityError("invalid_evaluation_budget")
    normalized_bounds: dict[str, tuple[float, float]] = {}
    for name in names:
        if bounds is not None and name in bounds:
            raw = bounds[name]
            if len(raw) != 2:
                raise IdentifiabilityError("bounds_must_have_two_values")
            lower, upper = float(raw[0]), float(raw[1])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise IdentifiabilityError("invalid_parameter_bounds")
            if not lower <= point[name] <= upper:
                raise IdentifiabilityError("parameter_outside_bounds")
            normalized_bounds[name] = (lower, upper)

    evaluations = 0
    baseline = _prediction(predict(point))
    evaluations += 1
    jacobian = np.empty((baseline.size, len(names)), dtype=float)
    methods: list[str] = []
    for column, name in enumerate(names):
        if evaluations + 2 > max_evaluations:
            return {
                "schema_version": SCHEMA_VERSION, "status": "not_assessed",
                "reason": "evaluation_budget_exhausted", "parameter_names": names,
                "evaluations": evaluations, "required_evaluations": 1 + 2 * len(names),
                "policy": "local_jacobian_is_not_global_identifiability_proof",
            }
        step = float(relative_step) * max(1.0, abs(point[name]))
        plus, minus = dict(point), dict(point)
        plus[name] += step
        minus[name] -= step
        if name in normalized_bounds:
            lower, upper = normalized_bounds[name]
            if plus[name] > upper or minus[name] < lower:
                # A one-sided derivative remains valid at a bound and avoids
                # silently evaluating an impossible parameter value.
                if plus[name] <= upper:
                    jacobian[:, column] = (_prediction(predict(plus)) - baseline) / step
                    methods.append("forward")
                elif minus[name] >= lower:
                    jacobian[:, column] = (baseline - _prediction(predict(minus))) / step
                    methods.append("backward")
                else:
                    raise IdentifiabilityError("parameter_step_outside_bounds")
                evaluations += 1
                continue
        jacobian[:, column] = (_prediction(predict(plus)) - _prediction(predict(minus))) / (2.0 * step)
        methods.append("central")
        evaluations += 2
    if not np.isfinite(jacobian).all():
        raise IdentifiabilityError("nonfinite_local_jacobian")
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    maximum = float(singular_values[0]) if singular_values.size else 0.0
    threshold = float(rank_tolerance) * max(maximum, 1.0)
    rank = int(np.sum(singular_values > threshold))
    fisher = jacobian.T @ jacobian
    fisher_eigenvalues = np.linalg.eigvalsh(fisher)
    positive = singular_values[singular_values > threshold]
    condition_number = float(maximum / positive[-1]) if positive.size else math.inf
    if rank == len(names):
        status = "locally_identified"
    elif rank > 0:
        status = "weakly_identified"
    else:
        status = "not_identified"
    return {
        "schema_version": SCHEMA_VERSION, "status": status,
        "parameter_names": names, "response_size": int(baseline.size),
        "evaluations": evaluations, "difference_methods": methods,
        "jacobian_rank": rank, "parameter_count": len(names),
        "singular_values": [float(value) for value in singular_values],
        "fisher_eigenvalues": [float(value) for value in fisher_eigenvalues],
        "condition_number": condition_number,
        "rank_threshold": threshold,
        "policy": "local_jacobian_is_not_global_identifiability_proof",
    }


__all__ = ["SCHEMA_VERSION", "IdentifiabilityError", "assess_local_identifiability"]
