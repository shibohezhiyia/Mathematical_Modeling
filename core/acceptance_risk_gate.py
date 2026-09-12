"""Conservative pre-registered gate for repair/search acceptance claims."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


class AcceptanceRiskGateError(ValueError):
    pass


def _wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 1.0)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - radius), min(1.0, centre + radius)


def assess_repair_acceptance_risk(
    rows: Sequence[Mapping[str, Any]], *, min_cases: int = 20,
    min_valid_rate_gain: float = 0.05, max_treatment_error_rate: float = 0.05,
    bootstrap_replicates: int = 1000, seed: int = 20260910,
) -> dict[str, Any]:
    """Assess a treatment against a baseline without turning it into proof.

    Each row must contain ``baseline`` and ``treatment`` mappings with boolean
    ``valid`` and ``error_accept`` fields. Failed/timeout rows remain in the
    denominator. A pass requires a conservative bootstrap lower bound for the
    paired valid-rate gain and a Wilson upper bound for treatment error accepts.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise AcceptanceRiskGateError("rows_required")
    if type(min_cases) is not int or not 1 <= min_cases <= 100_000:
        raise AcceptanceRiskGateError("invalid_min_cases")
    for value, code in ((min_valid_rate_gain, "invalid_min_valid_rate_gain"),
                        (max_treatment_error_rate, "invalid_max_treatment_error_rate")):
        if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise AcceptanceRiskGateError(code)
    if type(bootstrap_replicates) is not int or not 100 <= bootstrap_replicates <= 20_000:
        raise AcceptanceRiskGateError("invalid_bootstrap_replicates")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise AcceptanceRiskGateError("invalid_seed")
    if not rows or len(rows) > 100_000:
        raise AcceptanceRiskGateError("rows_out_of_bounds")
    paired_gain: list[float] = []
    baseline_valid = treatment_valid = treatment_errors = 0
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("baseline"), Mapping) or not isinstance(row.get("treatment"), Mapping):
            raise AcceptanceRiskGateError("paired_result_mappings_required")
        base, treatment = row["baseline"], row["treatment"]
        if type(base.get("valid")) is not bool or type(treatment.get("valid")) is not bool:
            raise AcceptanceRiskGateError("valid_boolean_required")
        if type(base.get("error_accept", False)) is not bool or type(treatment.get("error_accept", False)) is not bool:
            raise AcceptanceRiskGateError("error_accept_boolean_required")
        base_valid, treat_valid = int(base["valid"]), int(treatment["valid"])
        baseline_valid += base_valid
        treatment_valid += treat_valid
        treatment_errors += int(treatment.get("error_accept", False))
        paired_gain.append(float(treat_valid - base_valid))
    n = len(paired_gain)
    baseline_rate = baseline_valid / n
    treatment_rate = treatment_valid / n
    error_rate = treatment_errors / n
    error_interval = _wilson(treatment_errors, n)
    if n < min_cases:
        return {
            "schema_version": "mathmodel.acceptance-risk-gate/v1", "status": "not_assessed",
            "sample_count": n, "baseline_valid_rate": baseline_rate,
            "treatment_valid_rate": treatment_rate, "observed_valid_rate_gain": treatment_rate - baseline_rate,
            "valid_rate_gain_interval": None, "treatment_error_acceptance_rate": error_rate,
            "treatment_error_acceptance_interval": list(error_interval),
            "reason": "below_predeclared_min_cases",
            "policy": "risk_gate_is_not_generalization_proof",
        }
    if bootstrap_replicates * n > 20_000_000:
        raise AcceptanceRiskGateError("bootstrap_memory_budget_exceeded")
    values = np.asarray(paired_gain, dtype=float)
    rng = np.random.default_rng(seed)
    sampled = values[rng.integers(0, n, size=(bootstrap_replicates, n))].mean(axis=1)
    gain_interval = [float(np.quantile(sampled, 0.025)), float(np.quantile(sampled, 0.975))]
    passed = gain_interval[0] >= float(min_valid_rate_gain) and error_interval[1] <= float(max_treatment_error_rate)
    return {
        "schema_version": "mathmodel.acceptance-risk-gate/v1", "status": "pass" if passed else "fail",
        "sample_count": n, "baseline_valid_rate": baseline_rate,
        "treatment_valid_rate": treatment_rate, "observed_valid_rate_gain": treatment_rate - baseline_rate,
        "valid_rate_gain_interval": gain_interval, "treatment_error_acceptance_rate": error_rate,
        "treatment_error_acceptance_interval": list(error_interval),
        "thresholds": {"min_cases": min_cases, "min_valid_rate_gain": float(min_valid_rate_gain),
                        "max_treatment_error_rate": float(max_treatment_error_rate)},
        "bootstrap_seed": seed, "bootstrap_replicates": bootstrap_replicates,
        "policy": "predeclared_risk_gate;_paired_bootstrap_and_wilson_interval;_not_generalization_proof",
    }


__all__ = ["AcceptanceRiskGateError", "assess_repair_acceptance_risk"]
