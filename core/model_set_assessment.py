"""Evidence-preserving comparison of competing model candidates."""
from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.model-set-assessment/v1"
_REQUIRED_METRIC_NAMES = ("validation_loss", "complexity", "constraint_violation", "instability")
_OPTIONAL_METRIC_NAMES = ("compute_cost",)


class ModelSetAssessmentError(ValueError):
    pass


def _finite_vector(value: Any, name: str, *, maximum: int = 100_000) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelSetAssessmentError(f"{name}_must_be_numeric") from exc
    if result.ndim != 1 or result.size == 0 or result.size > maximum or not np.isfinite(result).all():
        raise ModelSetAssessmentError(f"{name}_must_be_finite_1d_array")
    return result


def _decision_key(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return repr(value)


def _dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    return all(a <= b for a, b in zip(left, right)) and any(a < b for a, b in zip(left, right))


def assess_model_set(
    candidates: Sequence[Mapping[str, Any]], *, max_candidates: int = 32,
    max_prediction_points: int = 100_000,
) -> dict[str, Any]:
    """Compare candidates without converting scores into uncalibrated posteriors.

    Each candidate must contain ``id``, ``predictions`` and a ``metrics`` mapping
    with non-negative finite values for the four lower-is-better axes.  A
    candidate may include an arbitrary JSON-like ``decision`` for consensus
    analysis.  Pareto membership is not a correctness certificate.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ModelSetAssessmentError("candidates_must_be_a_sequence")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 64:
        raise ModelSetAssessmentError("invalid_candidate_budget")
    if type(max_prediction_points) is not int or not 1 <= max_prediction_points <= 1_000_000:
        raise ModelSetAssessmentError("invalid_prediction_budget")
    if not 1 <= len(candidates) <= max_candidates:
        raise ModelSetAssessmentError("candidate_count_out_of_bounds")
    normalized = []
    metric_names = list(_REQUIRED_METRIC_NAMES)
    if all(isinstance(candidate, Mapping) and isinstance(candidate.get("metrics"), Mapping)
           and "compute_cost" in candidate["metrics"] for candidate in candidates):
        metric_names.extend(_OPTIONAL_METRIC_NAMES)
    common_size = None
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ModelSetAssessmentError("candidate_must_be_a_mapping")
        identifier = candidate.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise ModelSetAssessmentError("candidate_id_required")
        if any(item["id"] == identifier for item in normalized):
            raise ModelSetAssessmentError("duplicate_candidate_id")
        predictions = _finite_vector(candidate.get("predictions"), f"predictions:{identifier}", maximum=max_prediction_points)
        if common_size is None:
            common_size = predictions.size
        elif predictions.size != common_size:
            raise ModelSetAssessmentError("prediction_lengths_must_match")
        metrics = candidate.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ModelSetAssessmentError("candidate_metrics_required")
        axis_values = []
        for metric in metric_names:
            try:
                value = float(metrics[metric])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise ModelSetAssessmentError(f"missing_or_invalid_metric:{metric}") from exc
            if not math.isfinite(value) or value < 0:
                raise ModelSetAssessmentError(f"invalid_metric:{metric}")
            axis_values.append(value)
        normalized.append({"id": identifier, "predictions": predictions,
                           "metrics": dict(zip(metric_names, axis_values)),
                           "decision": candidate.get("decision")})
    pareto = [item for index, item in enumerate(normalized)
              if not any(_dominates(other["metrics"].values(), item["metrics"].values())
                         for other_index, other in enumerate(normalized) if index != other_index)]
    pareto_ids = [item["id"] for item in pareto]
    pareto_predictions = np.vstack([item["predictions"] for item in pareto])
    lower = np.min(pareto_predictions, axis=0)
    upper = np.max(pareto_predictions, axis=0)
    spread = upper - lower
    baseline_scale = max(float(np.median(np.abs(pareto_predictions))), 1e-12)
    normalized_spread = float(np.max(spread) / baseline_scale) if spread.size else 0.0
    decision_keys = [_decision_key(item["decision"]) for item in pareto if item["decision"] is not None]
    decision_consensus = bool(pareto) and len(decision_keys) == len(pareto) and len(set(decision_keys)) == 1
    decision_assessed = len(decision_keys) == len(pareto)
    pairwise_count = 0
    max_pairwise_rmse = 0.0
    for index in range(len(pareto)):
        for other_index in range(index):
            max_pairwise_rmse = max(max_pairwise_rmse, float(np.sqrt(np.mean(
                (pareto[index]["predictions"] - pareto[other_index]["predictions"]) ** 2))))
            pairwise_count += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "decision_not_assessed" if not decision_assessed else
            ("decision_consensus" if decision_consensus else "model_disagreement")
        ),
        "candidate_count": len(normalized), "pareto_candidate_ids": pareto_ids,
        "pareto_count": len(pareto), "prediction_points": int(common_size),
        "decision_assessed": decision_assessed,
        "decision_consensus": decision_consensus,
        "comparison_axes": metric_names,
        "decision_values": [item["decision"] for item in pareto],
        "prediction_envelope": {"lower": lower.tolist(), "upper": upper.tolist()},
        "prediction_spread": {"max_absolute": float(np.max(spread)),
                               "max_relative_to_median_abs": normalized_spread,
                               "max_pairwise_rmse": max_pairwise_rmse,
                               "pairwise_comparisons": pairwise_count},
        "uncertainty_decomposition": {
            "semantic": "not_assessed",
            "structural": "supported_by_multiple_pareto_candidates" if len(pareto) > 1 else "single_pareto_candidate",
            "parameter": "not_assessed",
            "numerical": "not_assessed",
        },
        "minimum_common_conclusion": {
            "kind": "pointwise_prediction_envelope",
            "supported_candidate_ids": pareto_ids,
            "interpretation": "Only claims valid across this envelope are shared candidate support; it is not a confidence interval.",
        },
        "policy": "pareto_and_disagreement_without_uncalibrated_posterior",
    }


def identify_observationally_equivalent_models(
    candidates: Sequence[Mapping[str, Any]], *, prediction_tolerance: float = 1e-8,
    max_candidates: int = 64,
) -> dict[str, Any]:
    """Return groups whose finite predictions are indistinguishable on a set.

    This is an observational equivalence report only. It does not prove the
    equations are algebraically equivalent or that the models agree outside
    the supplied evaluation points; callers should preserve every group as a
    competing mechanism when extrapolation matters.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise ModelSetAssessmentError("candidates_must_be_a_sequence")
    if len(candidates) > max_candidates:
        raise ModelSetAssessmentError("candidate_count_out_of_bounds")
    try:
        tolerance = float(prediction_tolerance)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ModelSetAssessmentError("invalid_prediction_tolerance") from exc
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ModelSetAssessmentError("invalid_prediction_tolerance")
    normalized = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("id"), str):
            raise ModelSetAssessmentError("candidate_id_required")
        vector = _finite_vector(candidate.get("predictions"), f"predictions:{candidate['id']}")
        normalized.append((candidate["id"], vector, candidate.get("equivalence_key")))
    parent = list(range(len(normalized)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for index, (_identifier, vector, key) in enumerate(normalized):
        for other_index in range(index):
            other_key = normalized[other_index][2]
            if vector.size != normalized[other_index][1].size:
                continue
            if key is not None and other_key is not None and key == other_key:
                union(index, other_index)
            elif float(np.max(np.abs(vector - normalized[other_index][1]))) <= tolerance:
                union(index, other_index)
    groups: dict[int, list[str]] = {}
    for index, (identifier, _vector, _key) in enumerate(normalized):
        groups.setdefault(find(index), []).append(identifier)
    classes = [sorted(values) for values in groups.values()]
    classes.sort(key=lambda values: values[0])
    return {"schema_version": "mathmodel.observational-equivalence/v1",
            "status": "equivalent_set_found" if any(len(item) > 1 for item in classes) else "no_equivalence_observed",
            "groups": classes, "prediction_tolerance": tolerance,
            "policy": "finite_observational_equivalence_not_algebraic_or_out_of_domain_proof"}


__all__ = ["SCHEMA_VERSION", "ModelSetAssessmentError", "assess_model_set",
           "identify_observationally_equivalent_models"]
