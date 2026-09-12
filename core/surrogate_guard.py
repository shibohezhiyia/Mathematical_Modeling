"""Small, dependency-light surrogate with explicit domain monitoring.

This is a safe approximation route for expensive simulations, not a silent
replacement for the simulator.  Predictions outside the fitted convex box
are rejected by default and callers can provide a full-fidelity fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping, Sequence

import numpy as np


class SurrogateGuardError(ValueError):
    """Raised for malformed training data or unsafe prediction requests."""


@dataclass(frozen=True)
class SurrogateModel:
    feature_min: np.ndarray
    feature_max: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    ridge: float
    residual_rmse: float
    schema_version: str = "mathmodel.surrogate/v1"

    @classmethod
    def fit(cls, features: Sequence[Sequence[float]], targets: Sequence[float], *, ridge: float = 1e-8) -> "SurrogateModel":
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float).reshape(-1)
        if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(x) < 2 or x.shape[1] < 1 or x.shape[1] > 128:
            raise SurrogateGuardError("training_shape_invalid")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not math.isfinite(float(ridge)) or ridge < 0:
            raise SurrogateGuardError("training_values_invalid")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        z = (x - mean) / scale
        design = np.column_stack([np.ones(len(z)), z])
        gram = design.T @ design + float(ridge) * np.eye(design.shape[1])
        try:
            coefficients = np.linalg.solve(gram, design.T @ y)
        except np.linalg.LinAlgError as exc:
            raise SurrogateGuardError("training_solve_failed") from exc
        residual = design @ coefficients - y
        return cls(x.min(axis=0), x.max(axis=0), mean, scale, coefficients,
                   float(ridge), float(np.sqrt(np.mean(residual ** 2))))

    def _normalise(self, features: Sequence[Sequence[float]]) -> tuple[np.ndarray, bool, np.ndarray]:
        x = np.asarray(features, dtype=float)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise SurrogateGuardError("prediction_shape_or_values_invalid")
        outside = np.logical_or(x < self.feature_min, x > self.feature_max).any(axis=1)
        return x, bool(outside.any()), (x - self.mean) / self.scale

    def predict(self, features: Sequence[Sequence[float]], *, allow_extrapolation: bool = False) -> dict[str, Any]:
        x, outside_any, z = self._normalise(features)
        if outside_any and not allow_extrapolation:
            return {"status": "out_of_domain", "predictions": None,
                    "outside_rows": np.flatnonzero(np.logical_or(x < self.feature_min, x > self.feature_max).any(axis=1)).tolist(),
                    "fallback_required": True, "residual_rmse": self.residual_rmse}
        predictions = np.column_stack([np.ones(len(z)), z]) @ self.coefficients
        if not np.isfinite(predictions).all():
            return {"status": "nonfinite", "predictions": None, "fallback_required": True,
                    "outside_rows": [], "residual_rmse": self.residual_rmse}
        return {"status": "extrapolated" if outside_any else "in_domain",
                "predictions": predictions.tolist(), "outside_rows": [],
                "fallback_required": False, "residual_rmse": self.residual_rmse}

    def predict_or_fallback(self, features: Sequence[Sequence[float]], fallback: Callable[[np.ndarray], Sequence[float]]) -> dict[str, Any]:
        if not callable(fallback):
            raise SurrogateGuardError("fallback_must_be_callable")
        x, _, _ = self._normalise(features)
        result = self.predict(x)
        if not result["fallback_required"]:
            return result
        values = np.asarray(fallback(x), dtype=float).reshape(-1)
        if len(values) != len(x) or not np.isfinite(values).all():
            raise SurrogateGuardError("fallback_result_invalid")
        return {"status": "full_fidelity_fallback", "predictions": values.tolist(),
                "fallback_required": False, "outside_rows": result.get("outside_rows", []),
                "residual_rmse": self.residual_rmse}


__all__ = ["SurrogateGuardError", "SurrogateModel"]
