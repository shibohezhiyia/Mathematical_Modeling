"""Bounded, model-agnostic diagnostics that propose structure-search directions.

The outputs are intentionally signals rather than claims about the data-generating
mechanism.  They are cheap enough to run before expensive symbolic or simulation
search and retain the evidence used for each suggestion.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.structure-diagnostics/v1"


class StructureDiagnosticsError(ValueError):
    """Raised when a diagnostic input contract cannot be satisfied."""


def _finite(value: Any, name: str, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise StructureDiagnosticsError(f"{name}_must_be_numeric") from exc
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise StructureDiagnosticsError(f"{name}_must_be_finite_{ndim}d_array")
    return array


def _scale(values: np.ndarray) -> float:
    centered = values - float(np.median(values))
    mad = float(np.median(np.abs(centered)))
    return max(1.4826 * mad, float(np.std(values)), 1e-12)


def _series_diagnostic(values: np.ndarray, *, max_lag: int) -> dict[str, Any]:
    y = np.asarray(values, dtype=float)
    centered = y - float(np.mean(y))
    differences = np.diff(y)
    if not differences.size:
        return {"status": "not_assessed", "reason": "too_few_points"}
    positive = float(np.mean(differences >= 0))
    negative = float(np.mean(differences <= 0))
    monotone_score = max(positive, negative)
    monotone_direction = "increasing" if positive >= negative else "decreasing"

    lags = min(max_lag, max(1, y.size // 2))
    denominator = float(np.dot(centered, centered))
    autocorrelations: list[float] = []
    if denominator > 1e-15:
        autocorrelations = [float(np.dot(centered[:-lag], centered[lag:]) /
                                  math.sqrt(max(denominator, 1e-15) *
                                             max(float(np.dot(centered[:-lag], centered[:-lag])), 1e-15)))
                            for lag in range(1, lags + 1)]
    periodic_lag = None
    periodic_score = 0.0
    if autocorrelations:
        periodic_lag = int(np.argmax(autocorrelations) + 1)
        periodic_score = float(autocorrelations[periodic_lag - 1])

    window = max(3, min(y.size // 4, 32))
    change_index = None
    change_score = 0.0
    if y.size >= 2 * window:
        scores = []
        for index in range(window, y.size - window + 1):
            left, right = y[index - window:index], y[index:index + window]
            scores.append(abs(float(np.mean(right) - np.mean(left))) / _scale(y))
        if scores:
            offset = int(np.argmax(scores))
            change_index = window + offset
            change_score = float(scores[offset])

    signals: list[str] = []
    if monotone_score >= 0.9:
        signals.append("monotone_candidate")
    if periodic_lag is not None and periodic_score >= 0.45:
        signals.append("periodic_candidate")
    if change_index is not None and change_score >= 1.0:
        signals.append("change_point_candidate")
    return {
        "status": "signal" if signals else "no_signal",
        "signals": signals,
        "monotone": {"direction": monotone_direction, "score": monotone_score},
        "periodicity": {"best_lag": periodic_lag, "autocorrelation": periodic_score,
                         "tested_lags": len(autocorrelations)},
        "change_point": {"index": change_index, "score": change_score, "window": window},
        "policy": "diagnostic_signal_not_mechanism_proof",
    }


def diagnose_series_structure(
    times: Sequence[float], values: Sequence[Sequence[float]] | Sequence[float],
    series_names: Sequence[str] | None = None, *, max_lag: int = 64,
) -> dict[str, Any]:
    """Return bounded structure signals for one or more aligned series."""
    t = _finite(times, "times", 1)
    try:
        value_ndim = np.asarray(values).ndim
    except (TypeError, ValueError):
        raise StructureDiagnosticsError("values_must_be_numeric") from None
    y = _finite(values, "values", 1 if value_ndim == 1 else 2)
    if y.ndim == 1:
        y = y[:, None]
    if y.shape[0] != t.size or t.size < 8 or y.shape[1] < 1 or y.shape[1] > 128:
        raise StructureDiagnosticsError("values_requires_8_to_128_aligned_series")
    if not np.all(np.diff(t) > 0):
        raise StructureDiagnosticsError("times_must_be_strictly_increasing")
    if type(max_lag) is not int or not 1 <= max_lag <= 128:
        raise StructureDiagnosticsError("invalid_lag_budget")
    names = list(series_names) if series_names is not None else [f"series_{i}" for i in range(y.shape[1])]
    if len(names) != y.shape[1] or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise StructureDiagnosticsError("series_names_must_match_and_be_unique")
    diagnostics = {name: _series_diagnostic(y[:, index], max_lag=max_lag)
                   for index, name in enumerate(names)}
    payload = {"times": np.round(t, 15).tolist(), "values": np.round(y, 15).tolist(), "series_names": names}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "sample_count": int(t.size),
        "series_count": int(y.shape[1]),
        "diagnostics": diagnostics,
        "trajectory_sha256": digest,
        "policy": "signals_generate_search_hints_only",
    }


__all__ = ["SCHEMA_VERSION", "StructureDiagnosticsError", "diagnose_series_structure"]
