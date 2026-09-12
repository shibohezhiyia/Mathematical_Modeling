"""Reversible, unit-aware scaling for numerically ill-conditioned models."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


class ReversibleScalingError(ValueError):
    pass


def _matrix(values: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReversibleScalingError(f"{name}_must_be_numeric") from exc
    if array.ndim not in {1, 2} or array.size == 0 or array.size > 2_000_000 or not np.isfinite(array).all():
        raise ReversibleScalingError(f"{name}_shape_or_finite_invalid")
    return array.reshape((-1, 1)) if array.ndim == 1 else array


@dataclass(frozen=True)
class ReversibleScaler:
    center: tuple[float, ...]
    scale: tuple[float, ...]
    unit_signature: tuple[str, ...]
    method: str
    constant_columns: tuple[int, ...]
    digest: str

    def _check_width(self, values: Any) -> np.ndarray:
        array = _matrix(values, name="values")
        if array.shape[1] != len(self.center):
            raise ReversibleScalingError("column_count_mismatch")
        return array

    def transform(self, values: Any) -> list[list[float]] | list[float]:
        array = self._check_width(values)
        output = (array - np.asarray(self.center)) / np.asarray(self.scale)
        if not np.isfinite(output).all():
            raise ReversibleScalingError("scaled_values_nonfinite")
        return output[:, 0].tolist() if np.asarray(values).ndim == 1 else output.tolist()

    def inverse_transform(self, values: Any) -> list[list[float]] | list[float]:
        array = self._check_width(values)
        output = array * np.asarray(self.scale) + np.asarray(self.center)
        if not np.isfinite(output).all():
            raise ReversibleScalingError("inverse_values_nonfinite")
        return output[:, 0].tolist() if np.asarray(values).ndim == 1 else output.tolist()

    def audit_roundtrip(self, values: Any, *, tolerance: float = 1e-10) -> dict[str, Any]:
        if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) or tolerance < 0:
            raise ReversibleScalingError("invalid_roundtrip_tolerance")
        original = _matrix(values, name="values")
        transformed = np.asarray(self.transform(values), dtype=float)
        recovered = np.asarray(self.inverse_transform(transformed), dtype=float)
        error = float(np.max(np.abs(original - recovered)))
        return {"status": "pass" if error <= float(tolerance) else "fail",
                "max_absolute_error": error, "tolerance": float(tolerance),
                "unit_signature": list(self.unit_signature),
                "policy": "roundtrip_check_only; scaling_does_not_make_units_interchangeable"}


def fit_reversible_scaler(values: Any, *, unit_signature: str | Sequence[str], method: str = "standard") -> ReversibleScaler:
    array = _matrix(values, name="values")
    if method not in {"standard", "minmax"}:
        raise ReversibleScalingError("unsupported_scaling_method")
    if isinstance(unit_signature, str):
        units = (unit_signature.strip(),) * array.shape[1]
    elif isinstance(unit_signature, Sequence) and not isinstance(unit_signature, (bytes, str)):
        units = tuple(str(item).strip() for item in unit_signature)
    else:
        raise ReversibleScalingError("unit_signature_required")
    if len(units) != array.shape[1] or any(not unit for unit in units):
        raise ReversibleScalingError("unit_signature_width_mismatch")
    if method == "standard":
        center = array.mean(axis=0)
        scale = array.std(axis=0)
    else:
        center = array.min(axis=0)
        scale = array.max(axis=0) - center
    constant = tuple(int(index) for index, value in enumerate(scale) if value == 0)
    scale = np.where(scale == 0, 1.0, scale)
    payload = {"center": center.tolist(), "scale": scale.tolist(), "units": units, "method": method}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return ReversibleScaler(tuple(float(item) for item in center), tuple(float(item) for item in scale),
                            units, method, constant, digest)


__all__ = ["ReversibleScaler", "ReversibleScalingError", "fit_reversible_scaler"]
