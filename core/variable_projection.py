"""Finite variable-projection search for separable model parameters."""

from __future__ import annotations

import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class VariableProjectionError(ValueError):
    pass


def variable_projection_search(
    nonlinear_points: Sequence[Any],
    design_builder: Callable[[Any], Any],
    target: Any,
    *,
    max_points: int = 256,
    condition_limit: float = 1e12,
) -> dict[str, Any]:
    """Eliminate linear coefficients for each bounded nonlinear candidate.

    The nonlinear part is deliberately supplied as a finite candidate set; a
    caller may obtain it from a validated grid, local search or structure
    proposal. Every least-squares solve is recorded, including rank deficiency,
    so this helper cannot silently turn an underidentified block into proof.
    """
    if not isinstance(nonlinear_points, Sequence) or isinstance(nonlinear_points, (str, bytes)):
        raise VariableProjectionError("nonlinear_points_must_be_a_sequence")
    if not 1 <= len(nonlinear_points) <= max_points or type(max_points) is not int or not 1 <= max_points <= 2048:
        raise VariableProjectionError("nonlinear_point_budget_invalid")
    if type(condition_limit) not in (int, float) or not math.isfinite(float(condition_limit)) or condition_limit <= 1:
        raise VariableProjectionError("condition_limit_invalid")
    try:
        y = np.asarray(target, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise VariableProjectionError("target_must_be_numeric") from exc
    if y.ndim != 1 or not 1 <= len(y) <= 1_000_000 or not np.isfinite(y).all():
        raise VariableProjectionError("target_shape_or_finite_invalid")
    attempts: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for index, point in enumerate(nonlinear_points):
        try:
            design = np.asarray(design_builder(point), dtype=float)
            if design.ndim != 2 or design.shape[0] != len(y) or not 1 <= design.shape[1] <= 512:
                raise VariableProjectionError("design_shape_invalid")
            if design.size > 2_000_000 or not np.isfinite(design).all():
                raise VariableProjectionError("design_shape_or_finite_invalid")
            coefficients, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
            residual = float(np.linalg.norm(design @ coefficients - y) ** 2)
            positive = singular[singular > np.finfo(float).eps]
            condition = float(positive.max() / positive.min()) if len(positive) else math.inf
            rank_deficient = int(rank) < design.shape[1]
            usable = math.isfinite(residual) and condition <= float(condition_limit)
            row = {"index": index, "nonlinear_point": point, "loss": residual,
                   "coefficients": coefficients.tolist(), "rank": int(rank),
                   "column_count": int(design.shape[1]), "condition_number": condition,
                   "rank_deficient": rank_deficient, "usable": usable}
            attempts.append(row)
            if usable and (best is None or residual < best["loss"]):
                best = row
        except VariableProjectionError as exc:
            attempts.append({"index": index, "nonlinear_point": point,
                             "status": "not_assessed", "reason": str(exc)})
        except Exception as exc:
            attempts.append({"index": index, "nonlinear_point": point,
                             "status": "not_assessed", "reason": f"solve_error:{type(exc).__name__}"})
    return {
        "schema_version": "mathmodel.variable-projection/v1",
        "status": "assessed" if best is not None else "not_assessed",
        "best": best,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "rank_deficient_attempts": sum(bool(row.get("rank_deficient")) for row in attempts),
        "policy": "finite_grid_variable_projection; rank_and_condition_are_diagnostics_not_identifiability_proof",
    }


__all__ = ["VariableProjectionError", "variable_projection_search"]
