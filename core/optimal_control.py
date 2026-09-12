"""Bounded finite-horizon linear-quadratic control candidate.

The Riccati recursion is exact for the unconstrained finite-horizon LQR. Box
clipping is reported as a constrained candidate rather than silently called an
optimal solution; callers must revalidate dynamics and constraints.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


class OptimalControlError(ValueError):
    pass


def _matrix(value: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise OptimalControlError(f"{name}_must_be_numeric") from exc
    if result.ndim != 2 or (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise OptimalControlError(f"{name}_shape_or_finite_invalid")
    return result


def solve_linear_quadratic_control(
    A: Any, B: Any, Q: Any, R: Any, x0: Any, *, horizon: int,
    control_lower: Any | None = None, control_upper: Any | None = None,
) -> dict[str, Any]:
    """Solve a bounded candidate for ``x[k+1]=A x[k]+B u[k]``."""
    if type(horizon) is not int or not 1 <= horizon <= 256:
        raise OptimalControlError("invalid_horizon")
    a, b = _matrix(A, "A"), _matrix(B, "B")
    if a.shape[0] != a.shape[1] or b.shape[0] != a.shape[0] or b.shape[1] < 1:
        raise OptimalControlError("dynamics_shape_invalid")
    n, m = a.shape[0], b.shape[1]
    if n > 32 or m > 16:
        raise OptimalControlError("control_dimension_budget_exceeded")
    q, r = _matrix(Q, "Q", (n, n)), _matrix(R, "R", (m, m))
    x = np.asarray(x0, dtype=float)
    if x.ndim != 1 or len(x) != n or not np.isfinite(x).all():
        raise OptimalControlError("initial_state_invalid")
    if not np.allclose(q, q.T) or not np.allclose(r, r.T) or np.min(np.linalg.eigvalsh(q)) < -1e-9 or np.min(np.linalg.eigvalsh(r)) <= 0:
        raise OptimalControlError("cost_matrices_must_be_psd_pd")
    lower = np.full(m, -np.inf) if control_lower is None else np.asarray(control_lower, dtype=float)
    upper = np.full(m, np.inf) if control_upper is None else np.asarray(control_upper, dtype=float)
    if lower.shape != (m,) or upper.shape != (m,) or np.any(lower > upper) or np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise OptimalControlError("control_bounds_invalid")
    # Backward Riccati recursion with zero terminal cost.
    p = q.copy()
    gains = []
    for _ in range(horizon):
        gram = r + b.T @ p @ b
        try:
            gain = np.linalg.solve(gram, b.T @ p @ a)
        except np.linalg.LinAlgError as exc:
            raise OptimalControlError("riccati_solve_failed") from exc
        gains.append(gain)
        p = q + a.T @ p @ (a - b @ gain)
        p = 0.5 * (p + p.T)
        if not np.isfinite(p).all():
            raise OptimalControlError("riccati_nonfinite")
    states = [x.copy()]
    controls = []
    total_cost = 0.0
    for gain in reversed(gains):
        control = -gain @ x
        clipped = np.clip(control, lower, upper)
        controls.append(clipped.copy())
        total_cost += float(x @ q @ x + clipped @ r @ clipped)
        x = a @ x + b @ clipped
        if not np.isfinite(x).all():
            raise OptimalControlError("trajectory_nonfinite")
        states.append(x.copy())
    violation = max((float(max(lo - value, value - hi, 0.0))
                     for control in controls for value, lo, hi in zip(control, lower, upper)), default=0.0)
    constrained = bool(np.any(np.isfinite(lower)) or np.any(np.isfinite(upper)))
    return {"schema_version": "mathmodel.linear-quadratic-control/v1", "status": "candidate",
            "solution_type": "clipped_lqr_candidate" if constrained else "finite_horizon_lqr",
            "horizon": horizon, "states": [item.tolist() for item in states],
            "controls": [item.tolist() for item in controls], "objective": total_cost,
            "maximum_constraint_violation": violation,
            "policy": "unconstrained_riccati_is_exact;clipped_bounds_require_independent_optimality_check"}


__all__ = ["OptimalControlError", "solve_linear_quadratic_control"]
