"""Delay-embedding candidates for an unobserved state space.

This module constructs reproducible mathematical latent coordinates from observed
trajectories.  It does not identify a physical hidden variable: that requires an
observation model, independent measurements, or additional identifiability
constraints.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.latent-state-discovery/v1"


class LatentStateDiscoveryError(ValueError):
    pass


def _finite(value: Any, name: str, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise LatentStateDiscoveryError(f"{name}_must_be_numeric") from exc
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise LatentStateDiscoveryError(f"{name}_must_be_finite_{ndim}d_array")
    return array


def _embedding(values: np.ndarray, delays: list[int]) -> np.ndarray:
    usable = values.shape[0] - delays[-1]
    return np.column_stack([values[delay:delay + usable] for delay in delays])


def discover_delay_latent_states(
    times: Sequence[float], observations: Sequence[Sequence[float]],
    observation_names: Sequence[str] | None = None, *, delay_count: int = 3,
    delay_stride: int = 1, latent_dim: int | None = None,
    validation_fraction: float = 0.25, explained_variance_target: float = 0.95,
) -> dict[str, Any]:
    """Construct a bounded delay-embedding/PCA latent-state candidate."""
    t = _finite(times, "times", 1)
    y = _finite(observations, "observations", 2)
    if y.shape[0] != t.size or t.size < 16 or y.shape[1] < 1 or y.shape[1] > 32:
        raise LatentStateDiscoveryError("observations_require_16_to_32_aligned_columns")
    if not np.all(np.diff(t) > 0):
        raise LatentStateDiscoveryError("times_must_be_strictly_increasing")
    if type(delay_count) is not int or not 2 <= delay_count <= 32:
        raise LatentStateDiscoveryError("invalid_delay_count")
    if type(delay_stride) is not int or not 1 <= delay_stride <= 32:
        raise LatentStateDiscoveryError("invalid_delay_stride")
    if not 0.05 <= float(validation_fraction) <= 0.4:
        raise LatentStateDiscoveryError("validation_fraction_must_be_between_0_05_and_0_4")
    if not 0.5 <= float(explained_variance_target) < 1.0:
        raise LatentStateDiscoveryError("invalid_explained_variance_target")
    names = list(observation_names) if observation_names is not None else [f"y_{i}" for i in range(y.shape[1])]
    if len(names) != y.shape[1] or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise LatentStateDiscoveryError("observation_names_must_match_and_be_unique")
    delays = [index * delay_stride for index in range(delay_count)]
    if delays[-1] >= t.size - 8:
        raise LatentStateDiscoveryError("delay_window_leaves_too_few_rows")
    embedded = _embedding(y, delays)
    if embedded.shape[0] * embedded.shape[1] > 1_000_000:
        raise LatentStateDiscoveryError("embedding_size_limit_exceeded")
    split = max(8, min(embedded.shape[0] - 4,
                        int(math.floor(embedded.shape[0] * (1.0 - float(validation_fraction))))))
    train, validation = embedded[:split], embedded[split:]
    center = np.mean(train, axis=0)
    scale = np.maximum(np.std(train, axis=0), 1e-12)
    train_scaled = (train - center) / scale
    validation_scaled = (validation - center) / scale
    _u, singular_values, vt = np.linalg.svd(train_scaled, full_matrices=False)
    variance = singular_values ** 2
    explained = variance / max(float(np.sum(variance)), 1e-15)
    cumulative = np.cumsum(explained)
    selected = int(np.searchsorted(cumulative, float(explained_variance_target)) + 1)
    selected = max(1, min(selected, min(8, vt.shape[0])))
    if latent_dim is not None:
        if type(latent_dim) is not int or not 1 <= latent_dim <= min(8, vt.shape[0]):
            raise LatentStateDiscoveryError("invalid_latent_dim")
        selected = latent_dim
    basis = vt[:selected]
    train_latent = train_scaled @ basis.T
    validation_latent = validation_scaled @ basis.T
    train_reconstruction = train_latent @ basis
    validation_reconstruction = validation_latent @ basis
    train_error = float(np.sqrt(np.mean((train_scaled - train_reconstruction) ** 2)))
    validation_error = float(np.sqrt(np.mean((validation_scaled - validation_reconstruction) ** 2)))
    payload = {"times": np.round(t, 15).tolist(), "observations": np.round(y, 15).tolist(), "names": names,
               "delay_count": delay_count, "delay_stride": delay_stride}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_generated",
        "proof_status": "tested_not_falsified",
        "observation_names": names,
        "delays": delays,
        "embedded_rows": int(embedded.shape[0]),
        "embedded_columns": int(embedded.shape[1]),
        "train_rows": int(split),
        "validation_rows": int(validation.shape[0]),
        "latent_dim": int(selected),
        "explained_variance": [float(v) for v in explained[:selected]],
        "basis": basis.round(12).tolist(),
        "center": center.round(12).tolist(),
        "scale": scale.round(12).tolist(),
        "train_reconstruction_rmse": train_error,
        "validation_reconstruction_rmse": validation_error,
        "latent_coordinates": np.vstack((train_latent, validation_latent)).round(12).tolist(),
        "trajectory_sha256": digest,
        "policy": "delay_embedding_is_math_latent_not_physical_hidden_variable",
    }


__all__ = ["SCHEMA_VERSION", "LatentStateDiscoveryError", "discover_delay_latent_states"]
