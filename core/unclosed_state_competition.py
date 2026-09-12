"""Competing explanations for an apparently unclosed observed system.

Residual memory is not evidence of a hidden state by itself.  This module
computes bounded, descriptive screening signals for several alternatives so a
search controller can compare them before expanding the state space.  Scores
are not probabilities, causal effects, or model-selection certificates.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.unclosed-state-competition/v1"


class UnclosedStateCompetitionError(ValueError):
    pass


def _matrix(value: Any, name: str, *, max_columns: int = 64) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UnclosedStateCompetitionError(f"{name}_must_be_numeric") from exc
    if array.ndim != 2 or not 16 <= array.shape[0] <= 5000 or not 1 <= array.shape[1] <= max_columns:
        raise UnclosedStateCompetitionError(f"{name}_shape_or_budget_invalid")
    if not np.isfinite(array).all():
        raise UnclosedStateCompetitionError(f"{name}_must_be_finite")
    return array


def _absolute_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 8 or right.size != left.size:
        return 0.0
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return 0.0
    value = float(abs(np.corrcoef(left, right)[0, 1]))
    return float(min(1.0, max(0.0, value))) if math.isfinite(value) else 0.0


def _hash_payload(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def compete_unclosed_state_explanations(
    residuals: Sequence[Sequence[float]], observations: Sequence[Sequence[float]],
    external_inputs: Sequence[Sequence[float]] | None = None,
    *, state_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a bounded screening comparison for competing residual explanations.

    The residual and observation arrays must refer to the same development
    window.  If external inputs are unavailable, that branch is explicitly
    marked ``not_assessed`` rather than treated as absent.
    """
    residual = _matrix(residuals, "residuals")
    observed = _matrix(observations, "observations")
    if observed.shape != residual.shape:
        raise UnclosedStateCompetitionError("residuals_and_observations_must_align")
    if state_names is None:
        names = [f"state_{index}" for index in range(residual.shape[1])]
    else:
        names = list(state_names)
        if len(names) != residual.shape[1] or any(type(item) is not str or not item for item in names):
            raise UnclosedStateCompetitionError("state_names_must_match_columns")
    exogenous = None
    if external_inputs is not None:
        exogenous = _matrix(external_inputs, "external_inputs", max_columns=32)
        if exogenous.shape[0] != residual.shape[0]:
            raise UnclosedStateCompetitionError("external_inputs_must_align_rows")

    candidates: list[dict[str, Any]] = []
    per_state: list[dict[str, Any]] = []
    for index, name in enumerate(names):
        series = residual[:, index]
        observed_series = observed[:, index]
        lag_memory = _absolute_correlation(series[:-1], series[1:])
        scale_pattern = _absolute_correlation(np.abs(series), np.abs(observed_series))
        split = max(8, residual.shape[0] // 2)
        first, second = series[:split], series[split:]
        scale = max(float(np.std(series)), 1e-12)
        nonstationarity = min(1.0, abs(float(np.mean(first) - np.mean(second))) / scale)
        nonstationarity = max(nonstationarity, min(1.0, abs(float(np.std(first) - np.std(second))) / scale))
        state_record = {
            "state": name,
            "lag1_memory_signal": lag_memory,
            "heteroscedasticity_signal": scale_pattern,
            "nonstationarity_signal": nonstationarity,
            "residual_rmse": float(np.sqrt(np.mean(series ** 2))),
            "observation_scale": float(np.std(observed_series)),
        }
        per_state.append(state_record)
        candidates.extend([
            {"id": "memory_or_latent_state", "state": name, "score": lag_memory,
             "status": "candidate", "evidence": {"lag1_memory_signal": lag_memory},
             "alternative_to": ["correlated_observation_noise", "time_alignment_error"],
             "not_claimed": ["hidden_state_exists", "physical_identity"]},
            {"id": "heteroscedastic_observation", "state": name, "score": scale_pattern,
             "status": "candidate", "evidence": {"absolute_scale_signal": scale_pattern},
             "alternative_to": ["missing_nonlinearity", "state_dependent_mechanism"],
             "not_claimed": ["noise_distribution_identified"]},
            {"id": "nonstationary_or_regime_change", "state": name, "score": nonstationarity,
             "status": "candidate", "evidence": {"split_shift_signal": nonstationarity},
             "alternative_to": ["unmodeled_input", "parameter_drift"],
             "not_claimed": ["change_point_proven"]},
        ])
        if exogenous is not None:
            corr = max((_absolute_correlation(series, exogenous[:, col])
                        for col in range(exogenous.shape[1])), default=0.0)
            candidates.append({"id": "external_input", "state": name, "score": corr,
                               "status": "candidate", "evidence": {"max_exogenous_signal": corr},
                               "alternative_to": ["latent_state", "omitted_observation_bias"],
                               "not_claimed": ["exogenous_cause"]})
    scale = np.maximum(np.std(observed, axis=0), 1e-12)
    normalized_misfit = float(min(1.0, np.mean(np.std(residual, axis=0) / scale)))
    candidates.append({"id": "library_or_numerical_misfit", "state": "all", "score": normalized_misfit,
                       "status": "candidate", "evidence": {"normalized_residual_scale": normalized_misfit},
                       "alternative_to": ["all_structural_explanations"],
                       "not_claimed": ["function_library_insufficient", "numerical_failure"]})
    candidates.sort(key=lambda item: (-float(item["score"]), item["id"], item["state"]))
    top_score = float(candidates[0]["score"]) if candidates else 0.0
    near_ties = [item["id"] + ":" + item["state"] for item in candidates
                 if top_score - float(item["score"]) <= 0.05]
    fingerprint = _hash_payload({"residuals": np.round(residual, 12).tolist(),
                                 "observations": np.round(observed, 12).tolist(),
                                 "external_inputs": None if exogenous is None else np.round(exogenous, 12).tolist(),
                                 "state_names": names})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "screening_only",
        "proof_status": "tested_not_falsified",
        "state_signals": per_state,
        "candidate_explanations": candidates[:128],
        "near_tied_explanations": near_ties[:16],
        "external_input_status": "assessed" if exogenous is not None else "not_assessed",
        "input_fingerprint": fingerprint,
        "policy": {
            "scores_are_not_probabilities": True,
            "latent_state_not_proven": True,
            "must_compare_noise_input_nonstationarity_and_misfit": True,
            "may_modify_ir": False,
            "may_change_hard_constraints": False,
        },
    }


__all__ = ["SCHEMA_VERSION", "UnclosedStateCompetitionError", "compete_unclosed_state_explanations"]
