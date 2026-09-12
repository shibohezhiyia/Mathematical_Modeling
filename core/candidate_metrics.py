"""Typed multi-objective candidate metric contract.

The score vector is intentionally kept as separate objectives.  A weighted
sum would hide whether a candidate won by trading a severe constraint or
dimension violation for a small fit improvement.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .candidate_gate import CandidateGateError, gate_candidates


METRIC_DIRECTIONS = {
    "fit": "min", "complexity": "min", "dimension_violation": "min",
    "constraint_violation": "min", "stability": "min", "theory_support": "max",
    "cost": "min",
}


def compare_unified_candidate_metrics(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate and Pareto-rank a homogeneous seven-objective score vector."""
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise CandidateGateError("candidates_required")
    normalized = []
    required = set(METRIC_DIRECTIONS)
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise CandidateGateError("candidate_must_be_mapping")
        metrics = candidate.get("metrics")
        if not isinstance(metrics, Mapping) or set(metrics) != required:
            raise CandidateGateError("unified_metric_schema_required")
        clean = {}
        for name in required:
            try:
                value = float(metrics[name])
            except (TypeError, ValueError, OverflowError) as exc:
                raise CandidateGateError(f"metric_not_finite:{name}") from exc
            if not math.isfinite(value) or value < 0 and name != "theory_support":
                raise CandidateGateError(f"metric_not_finite:{name}")
            clean[name] = value
        normalized.append({"id": candidate.get("id"), "hard_checks": candidate.get("hard_checks", {}),
                           "metrics": clean})
    result = gate_candidates(normalized, metric_directions=METRIC_DIRECTIONS)
    result["schema_version"] = "mathmodel.candidate-metrics/v1"
    result["metric_directions"] = dict(METRIC_DIRECTIONS)
    result["policy"] = "separate_objectives_pareto_only_no_uncalibrated_scalar_score"
    return result


__all__ = ["METRIC_DIRECTIONS", "compare_unified_candidate_metrics"]
