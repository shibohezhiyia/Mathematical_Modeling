"""Normalize heterogeneous solver outputs without inventing accuracy claims."""
from __future__ import annotations

import math
from typing import Any, Mapping


class SolverResultContractError(ValueError):
    pass


_STATUSES = {"executed", "partial", "not_executed", "failed", "unresolved", "cancelled"}


def _number(value: Any, code: str) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SolverResultContractError(code) from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise SolverResultContractError(code)
    return parsed


def normalize_solver_result(
    raw: Mapping[str, Any], *, operation: str, resource_usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a stable result envelope for UI, evidence and model comparison.

    Missing residual/error bounds remain ``not_assessed``. The adapter does not
    infer an error bound from a fit score and does not upgrade a partial result.
    """
    if not isinstance(raw, Mapping) or type(operation) is not str or not 1 <= len(operation) <= 128:
        raise SolverResultContractError("invalid_solver_result_input")
    status = str(raw.get("status", "not_executed"))
    if status not in _STATUSES:
        raise SolverResultContractError("invalid_solver_status")
    residual = _number(raw.get("residual", raw.get("residual_inf", raw.get("residual_l2"))), "invalid_residual")
    error_bound = _number(raw.get("error_bound"), "invalid_error_bound")
    iterations = raw.get("iterations", raw.get("iteration_count"))
    if iterations is not None and (type(iterations) is not int or not 0 <= iterations <= 10_000_000):
        raise SolverResultContractError("invalid_iteration_count")
    usage = dict(resource_usage or raw.get("resource_usage") or {})
    if len(usage) > 32:
        raise SolverResultContractError("resource_usage_too_large")
    normalized_usage = {}
    for key, value in usage.items():
        if type(key) is not str or len(key) > 64:
            raise SolverResultContractError("invalid_resource_usage_key")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if not math.isfinite(float(value)) or value < 0:
                raise SolverResultContractError("invalid_resource_usage_value")
            normalized_usage[key] = float(value)
        elif isinstance(value, str) and len(value) <= 256:
            normalized_usage[key] = value
        else:
            raise SolverResultContractError("invalid_resource_usage_value")
    return {
        "schema_version": "mathmodel.solver-result/v1", "operation": operation,
        "status": status, "result": raw.get("result", raw.get("solution", raw.get("value"))),
        "residual": residual, "residual_status": "assessed" if residual is not None else "not_assessed",
        "error_bound": error_bound, "error_bound_status": "assessed" if error_bound is not None else "not_assessed",
        "iterations": iterations, "resource_usage": normalized_usage,
        "policy": "normalization_does_not_upgrade_status_or_infer_error_bounds",
    }


__all__ = ["SolverResultContractError", "normalize_solver_result"]
