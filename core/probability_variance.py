"""Probability-weighted law-of-total-variance adapter."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .uncertainty_propagation import UncertaintyPropagationError, propagate_layered_uncertainty


class ProbabilityVarianceError(ValueError):
    pass


def propagate_probability_variance(
    evaluations: Sequence[Mapping[str, Any]], *, value_key: str = "value",
    probability_key: str = "probability", unit_signature: str | None = None,
    mass_tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Apply an explicit probability mass to the four-layer variance ledger.

    Each row is one mutually exclusive leaf event. Probabilities must be
    nonnegative and sum to one within ``mass_tolerance``; no scores with
    different units are added. The result is empirical/declared-model
    probability accounting, not a posterior unless the caller supplies a
    justified probability model.
    """
    if not isinstance(evaluations, Sequence) or isinstance(evaluations, (str, bytes)) or not evaluations:
        raise ProbabilityVarianceError("evaluations_required")
    if type(mass_tolerance) not in (int, float) or not math.isfinite(float(mass_tolerance)) or mass_tolerance < 0:
        raise ProbabilityVarianceError("invalid_mass_tolerance")
    rows = []
    total = 0.0
    for index, item in enumerate(evaluations):
        if not isinstance(item, Mapping):
            raise ProbabilityVarianceError(f"evaluation_{index}_must_be_mapping")
        try:
            probability = float(item[probability_key])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ProbabilityVarianceError("probability_required") from exc
        if not math.isfinite(probability) or probability < 0:
            raise ProbabilityVarianceError("probability_must_be_nonnegative_finite")
        total += probability
        row = dict(item)
        row["weight"] = probability
        rows.append(row)
    if total <= 0 or abs(total - 1.0) > float(mass_tolerance):
        raise ProbabilityVarianceError("probabilities_must_sum_to_one")
    try:
        result = propagate_layered_uncertainty(rows, value_key=value_key,
                                               unit_signature=unit_signature)
    except UncertaintyPropagationError as exc:
        raise ProbabilityVarianceError(str(exc)) from exc
    result = dict(result)
    result["schema_version"] = "mathmodel.probability-variance/v1"
    result["probability_mass"] = total
    result["law_total_variance"] = result.get("reconstructed_variance") is not None
    result["policy"] = "declared_probability_weighted_total_variance_not_automatic_posterior"
    return result


__all__ = ["ProbabilityVarianceError", "propagate_probability_variance"]
