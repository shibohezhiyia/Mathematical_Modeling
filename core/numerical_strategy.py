"""Evidence-gated numerical backend strategy selection.

This module chooses a *reviewable strategy hint* from diagnostics.  It does
not run a solver and never treats a heuristic diagnostic as a proof of
stiffness, conditioning, or gradient correctness.  The selected strategy is
therefore carried with its evidence status and can be independently checked
by the backend before execution.
"""

from __future__ import annotations

import math
from typing import Any, Mapping


class NumericalStrategyError(ValueError):
    pass


def select_numerical_strategy(
    diagnostics: Mapping[str, Any] | None = None,
    *,
    condition_threshold: float = 1e8,
    stiffness_threshold: float = 1e4,
) -> dict[str, Any]:
    """Return a conservative strategy hint from finite, explicit diagnostics.

    Accepted diagnostics are ``condition_number``, ``stiffness_ratio``,
    ``event_count`` and ``gradient_status``.  Missing values remain
    ``not_assessed``.  Thresholds only select a safer route; they do not
    certify that a chosen integrator or derivative is correct.
    """
    if diagnostics is None:
        diagnostics = {}
    if not isinstance(diagnostics, Mapping):
        raise NumericalStrategyError("diagnostics_must_be_mapping")
    for name, value in (("condition_threshold", condition_threshold),
                        ("stiffness_threshold", stiffness_threshold)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NumericalStrategyError(f"{name}_must_be_numeric")
        if not math.isfinite(float(value)) or float(value) <= 1:
            raise NumericalStrategyError(f"{name}_must_be_finite_above_one")

    def finite_metric(key: str) -> float | None:
        value = diagnostics.get(key)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise NumericalStrategyError(f"{key}_must_be_finite_nonnegative")
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise NumericalStrategyError(f"{key}_must_be_finite_nonnegative")
        return value

    condition = finite_metric("condition_number")
    stiffness = finite_metric("stiffness_ratio")
    events = finite_metric("event_count")
    gradient_status = diagnostics.get("gradient_status")
    if gradient_status not in {"validated", "unvalidated", "failed", "not_assessed"}:
        gradient_status = "not_assessed"

    flags = {
        "ill_conditioned": condition is not None and condition >= float(condition_threshold),
        "stiff": stiffness is not None and stiffness >= float(stiffness_threshold),
        "event_structure": events is not None and events > 0,
    }
    if flags["stiff"]:
        backend = "stiff_aware_integrator"
    elif flags["event_structure"]:
        backend = "event_aware_integrator"
    elif flags["ill_conditioned"]:
        backend = "condition_robust_optimizer"
    else:
        backend = "standard_bounded_solver"
    derivative = "gradient_accelerated" if gradient_status == "validated" else "independent_difference_check"
    assessed = any(value is not None for value in (condition, stiffness, events)) or gradient_status != "not_assessed"
    return {
        "schema_version": "mathmodel.numerical-strategy/v1",
        "status": "assessed" if assessed else "not_assessed",
        "backend_strategy": backend,
        "derivative_strategy": derivative,
        "flags": flags,
        "diagnostics": {
            "condition_number": condition,
            "stiffness_ratio": stiffness,
            "event_count": events,
            "gradient_status": gradient_status,
        },
        "thresholds": {
            "condition_number": float(condition_threshold),
            "stiffness_ratio": float(stiffness_threshold),
        },
        "policy": "routing_hint_requires_backend_validation; diagnostics_are_not_certificates",
    }


__all__ = ["NumericalStrategyError", "select_numerical_strategy"]
