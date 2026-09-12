"""Bounded multivariate interaction screening.

This is a deterministic, dependency-light graph pre-screen.  It estimates
regularised partial correlations and bootstrap edge stability; an edge is an
association candidate only, never a causal claim or a learned GNN graph.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "mathmodel.interaction-graph-screen/v1"


class InteractionGraphScreenError(ValueError):
    pass


def _partial_edges(values: np.ndarray, names: Sequence[str], ridge: float) -> dict[tuple[str, str], float]:
    centered = values - np.mean(values, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, keepdims=True)
    scaled = centered / np.maximum(scale, 1e-12)
    covariance = (scaled.T @ scaled) / max(len(scaled) - 1, 1)
    covariance = covariance + np.eye(covariance.shape[0]) * float(ridge)
    try:
        precision = np.linalg.pinv(covariance, hermitian=True)
    except TypeError:
        precision = np.linalg.pinv(covariance)
    edges: dict[tuple[str, str], float] = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            denominator = max(float(precision[i, i] * precision[j, j]), 1e-15) ** 0.5
            value = float(-precision[i, j] / denominator)
            if np.isfinite(value):
                edges[(str(names[i]), str(names[j]))] = float(np.clip(value, -1.0, 1.0))
    return edges


def discover_interaction_graph(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    max_variables: int = 24,
    max_rows: int = 4_000,
    bootstrap: int = 24,
    ridge: float = 0.05,
    edge_threshold: float = 0.15,
    stability_threshold: float = 0.6,
    random_state: int = 0,
) -> dict[str, Any]:
    if not isinstance(frame, pd.DataFrame):
        raise InteractionGraphScreenError("frame_must_be_dataframe")
    if type(max_variables) is not int or not 3 <= max_variables <= 64:
        raise InteractionGraphScreenError("invalid_max_variables")
    if type(max_rows) is not int or not 100 <= max_rows <= 100_000:
        raise InteractionGraphScreenError("invalid_max_rows")
    if type(bootstrap) is not int or not 4 <= bootstrap <= 200:
        raise InteractionGraphScreenError("invalid_bootstrap")
    if not np.isfinite(float(ridge)) or float(ridge) <= 0:
        raise InteractionGraphScreenError("ridge_must_be_positive_finite")
    if not 0 < float(edge_threshold) <= 1 or not 0 < float(stability_threshold) <= 1:
        raise InteractionGraphScreenError("invalid_threshold")
    numeric = list(columns) if columns is not None else [
        str(c) for c in frame.columns if pd.api.types.is_numeric_dtype(frame[c])
    ]
    numeric = [str(c) for c in numeric if str(c) in frame.columns]
    numeric = [c for c in numeric if pd.api.types.is_numeric_dtype(frame[c])]
    numeric = [c for c in numeric if frame[c].nunique(dropna=True) >= 3]
    if len(numeric) < 3:
        return {
            "schema_version": SCHEMA_VERSION, "status": "not_assessed",
            "reason": "fewer_than_three_nonconstant_numeric_variables", "variables": numeric,
            "edges": [],
        }
    # Keep the most variable columns to bound inversion cost while preserving a
    # deterministic selection independent of row order.
    numeric = sorted(numeric, key=lambda c: (-float(pd.to_numeric(frame[c], errors="coerce").std()), c))[:max_variables]
    work = frame[numeric].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = work.dropna(axis=0, how="any")
    if len(work) < 30:
        return {
            "schema_version": SCHEMA_VERSION, "status": "not_assessed",
            "reason": "fewer_than_thirty_complete_rows", "variables": numeric,
            "rows": int(len(work)), "edges": [],
        }
    rng = np.random.default_rng(int(random_state))
    if len(work) > max_rows:
        work = work.iloc[np.sort(rng.choice(len(work), size=max_rows, replace=False))]
    values = work.to_numpy(dtype=float)
    base = _partial_edges(values, numeric, float(ridge))
    counts = {edge: 0 for edge in base}
    for _ in range(bootstrap):
        sample = values[rng.integers(0, len(values), len(values))]
        current = _partial_edges(sample, numeric, float(ridge))
        for edge, score in current.items():
            if abs(score) >= float(edge_threshold):
                counts[edge] += 1
    edges = []
    for (left, right), score in base.items():
        stability = counts[(left, right)] / float(bootstrap)
        if abs(score) >= float(edge_threshold) and stability >= float(stability_threshold):
            edges.append({
                "source": left, "target": right, "partial_correlation": round(score, 6),
                "absolute_strength": round(abs(score), 6), "stability": round(stability, 6),
                "interpretation": "stable conditional association; not a causal edge",
            })
    edges.sort(key=lambda item: (-item["stability"], -item["absolute_strength"], item["source"], item["target"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "tested_not_falsified",
        "proof_status": "finite_resampling_association_only",
        "variables": numeric,
        "rows": int(len(values)),
        "bootstrap": int(bootstrap),
        "ridge": float(ridge),
        "edge_threshold": float(edge_threshold),
        "stability_threshold": float(stability_threshold),
        "edges": edges,
        "policy": "partial_correlation_screen_is_not_causal_discovery_or_gnn",
    }


__all__ = ["SCHEMA_VERSION", "InteractionGraphScreenError", "discover_interaction_graph"]
