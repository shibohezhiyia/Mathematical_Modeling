"""Bounded discovery of candidate linear invariants from observed trajectories."""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.conservation-discovery/v1"


class ConservationDiscoveryError(ValueError):
    pass


def _finite_array(value: Any, name: str, *, ndim: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ConservationDiscoveryError(f"{name}_must_be_numeric") from exc
    if result.ndim != ndim or result.size == 0 or not np.isfinite(result).all():
        raise ConservationDiscoveryError(f"{name}_must_be_finite_{ndim}d_array")
    return result


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ConservationDiscoveryError("degenerate_invariant_vector")
    vector = vector / norm
    first = next((item for item in vector if abs(float(item)) > 1e-12), None)
    if first is not None and first < 0:
        vector = -vector
    return vector


def _residual(derivatives: np.ndarray, coefficients: np.ndarray) -> float:
    values = derivatives @ coefficients
    return float(np.sqrt(np.mean(values ** 2)))


def discover_linear_conservation(
    times: Sequence[float], states: Sequence[Sequence[float]], state_names: Sequence[str] | None = None,
    *, singular_value_tolerance: float = 1e-8, residual_tolerance: float = 1e-5,
    validation_fraction: float = 0.3, max_candidates: int = 8,
) -> dict[str, Any]:
    """Find stable null-space directions of a finite-difference derivative matrix."""
    t = _finite_array(times, "times", ndim=1)
    x = _finite_array(states, "states", ndim=2)
    if x.shape[0] != t.size or t.size < 6 or x.shape[1] < 2 or x.shape[1] > 64:
        raise ConservationDiscoveryError("trajectory_requires_at_least_6_rows_and_2_to_64_states")
    if not np.all(np.diff(t) > 0):
        raise ConservationDiscoveryError("times_must_be_strictly_increasing")
    if t.size * x.shape[1] > 1_000_000:
        raise ConservationDiscoveryError("trajectory_size_limit_exceeded")
    if not 0 < float(validation_fraction) < 0.5:
        raise ConservationDiscoveryError("validation_fraction_must_be_between_0_and_0_5")
    if not 0 < float(singular_value_tolerance) < 1 or not 0 < float(residual_tolerance):
        raise ConservationDiscoveryError("invalid_discovery_tolerance")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 16:
        raise ConservationDiscoveryError("invalid_candidate_budget")
    names = list(state_names) if state_names is not None else [f"x_{i}" for i in range(x.shape[1])]
    if len(names) != x.shape[1] or len(set(names)) != len(names) or any(not isinstance(name, str) or not name for name in names):
        raise ConservationDiscoveryError("state_names_must_match_and_be_unique")

    derivatives = (x[2:] - x[:-2]) / (t[2:, None] - t[:-2, None])
    row_count = derivatives.shape[0]
    split = max(3, min(row_count - 2, int(math.floor(row_count * (1.0 - float(validation_fraction))))))
    train, validation = derivatives[:split], derivatives[split:]
    scale = np.maximum(np.linalg.norm(train, axis=0), 1e-15)
    scaled = train / scale
    # Keep the complete right-singular basis.  With more state variables than
    # derivative rows, ``full_matrices=False`` silently drops the exact
    # null-space directions we are explicitly trying to discover.
    _u, singular_values, vh = np.linalg.svd(scaled, full_matrices=True)
    full_singular_values = np.concatenate(
        [singular_values, np.zeros(max(0, scaled.shape[1] - singular_values.size))]
    )
    max_singular = float(singular_values[0]) if singular_values.size else 0.0
    threshold = float(singular_value_tolerance) * max(max_singular, 1.0)
    null_indices = [index for index, value in enumerate(full_singular_values) if value <= threshold]
    if not null_indices and singular_values.size and singular_values[-1] <= threshold * 10:
        null_indices = [len(singular_values) - 1]
    candidates = []
    for index in null_indices[:max_candidates]:
        coefficients = _normalized(vh[index] / scale)
        train_error = _residual(train, coefficients)
        validation_error = _residual(validation, coefficients) if validation.size else math.inf
        relative_scale = max(float(np.linalg.norm(train, axis=1).mean()), 1e-12)
        candidates.append({
            "id": f"invariant_{len(candidates) + 1}",
            "coefficients": {name: float(value) for name, value in zip(names, coefficients) if abs(float(value)) > 1e-10},
            "train_derivative_rmse": train_error,
            "validation_derivative_rmse": validation_error,
            "relative_validation_error": validation_error / relative_scale,
            "status": "candidate" if validation_error <= float(residual_tolerance) else "rejected_by_validation",
            "proof_status": "tested_not_falsified" if validation_error <= float(residual_tolerance) else "counterexample_found",
        })
    matrix_payload = np.asarray(x, dtype=float).round(15).tolist()
    digest = sha256(json.dumps({"times": np.asarray(t).round(15).tolist(), "states": matrix_payload,
                                "state_names": names}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_found" if any(item["status"] == "candidate" for item in candidates) else "no_stable_candidate",
        "state_names": names,
        "sample_count": int(t.size),
        "derivative_rows": int(row_count),
        "train_rows": int(split),
        "validation_rows": int(row_count - split),
        "rank": int(np.sum(full_singular_values > threshold)),
        "nullity": int(len(null_indices)),
        "singular_values": [float(value) for value in full_singular_values],
        "candidates": candidates,
        "trajectory_sha256": digest,
        "policy": "finite_difference_nullspace_candidate_not_conservation_proof",
    }


__all__ = ["SCHEMA_VERSION", "ConservationDiscoveryError", "discover_linear_conservation"]
