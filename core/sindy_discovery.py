"""Small, dependency-light SINDy-style sparse ODE candidate discovery."""
from __future__ import annotations

from hashlib import sha256
import itertools
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.sindy-discovery/v1"

def _integrate(values: np.ndarray, grid: np.ndarray, *, axis: int = 0) -> np.ndarray:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(values, grid, axis=axis)
    # Compatibility fallback for older NumPy releases; all callers use axis 0.
    if axis != 0:
        raise SINDyDiscoveryError("weak_form_integration_axis_must_be_zero")
    return np.sum((values[1:] + values[:-1]) * np.diff(grid)[:, None] / 2.0, axis=0)


class SINDyDiscoveryError(ValueError):
    pass


def _finite(value: Any, name: str, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SINDyDiscoveryError(f"{name}_must_be_numeric") from exc
    if array.ndim != ndim or array.size == 0 or not np.isfinite(array).all():
        raise SINDyDiscoveryError(f"{name}_must_be_finite_{ndim}d_array")
    return array


def _powers(variable_count: int, degree: int) -> list[tuple[int, ...]]:
    powers = [(0,) * variable_count]
    for total in range(1, degree + 1):
        for positions in itertools.combinations_with_replacement(range(variable_count), total):
            exponent = [0] * variable_count
            for position in positions:
                exponent[position] += 1
            powers.append(tuple(exponent))
    return powers


def _term_name(power: tuple[int, ...], names: Sequence[str]) -> str:
    if not any(power):
        return "1"
    factors = []
    for name, exponent in zip(names, power):
        if exponent == 1:
            factors.append(name)
        elif exponent > 1:
            factors.append(f"{name}^{exponent}")
    return "*".join(factors)


def _library(states: np.ndarray, powers: Sequence[tuple[int, ...]]) -> np.ndarray:
    columns = []
    for power in powers:
        column = np.ones(states.shape[0], dtype=float)
        for index, exponent in enumerate(power):
            if exponent:
                column *= states[:, index] ** exponent
        columns.append(column)
    return np.column_stack(columns)


def _stlsq(matrix: np.ndarray, target: np.ndarray, *, threshold: float, max_terms: int) -> np.ndarray:
    scale = np.maximum(np.linalg.norm(matrix, axis=0), 1e-15)
    scaled = matrix / scale
    try:
        coefficient = np.linalg.lstsq(scaled, target, rcond=None)[0]
    except np.linalg.LinAlgError as exc:
        raise SINDyDiscoveryError("library_least_squares_failed") from exc
    for _ in range(12):
        active = np.abs(coefficient) >= threshold
        if not np.any(active):
            return np.zeros_like(coefficient)
        updated = np.zeros_like(coefficient)
        try:
            updated[active] = np.linalg.lstsq(scaled[:, active], target, rcond=None)[0]
        except np.linalg.LinAlgError as exc:
            raise SINDyDiscoveryError("library_least_squares_failed") from exc
        if np.array_equal(active, np.abs(updated) >= threshold):
            coefficient = updated
            break
        coefficient = updated
    coefficient = coefficient / scale
    nonzero = np.flatnonzero(np.abs(coefficient) > 0)
    if nonzero.size > max_terms:
        keep = nonzero[np.argsort(np.abs(coefficient[nonzero]))[-max_terms:]]
        filtered = np.zeros_like(coefficient)
        filtered[keep] = coefficient[keep]
        coefficient = filtered
    return coefficient


def _counterexamples(times: np.ndarray, residuals: np.ndarray, names: Sequence[str],
                     tolerance: float, *, offset: int = 0, maximum: int = 32) -> list[dict[str, Any]]:
    """Return bounded, deterministic witnesses for validation residuals."""
    rows = []
    for row_index, residual_row in enumerate(np.abs(residuals)):
        for equation_index, error in enumerate(residual_row):
            if float(error) > tolerance:
                rows.append((float(error), row_index, equation_index))
    rows.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [{"row": int(row + offset), "time": float(times[row + 1 + offset]),
             "equation": f"d{names[equation]}/dt", "absolute_residual": error,
             "origin": "heldout_validation"} for error, row, equation in rows[:maximum]]


def _normalize_dimensions(raw: Sequence[Mapping[str, int]] | None, count: int,
                          label: str) -> list[dict[str, int]] | None:
    if raw is None:
        return None
    if not isinstance(raw, Sequence) or len(raw) != count:
        raise SINDyDiscoveryError(f"{label}_must_match_state_count")
    normalized = []
    for dimension in raw:
        if not isinstance(dimension, Mapping):
            raise SINDyDiscoveryError(f"{label}_must_contain_mappings")
        result = {}
        for key, exponent in dimension.items():
            if not isinstance(key, str) or not key.strip() or type(exponent) is not int or abs(exponent) > 32:
                raise SINDyDiscoveryError(f"{label}_contains_invalid_exponent")
            if exponent:
                result[key] = exponent
        normalized.append(result)
    return normalized


def _compatible_library_indices(
    powers: Sequence[tuple[int, ...]], state_dimensions: list[dict[str, int]],
    target_dimension: Mapping[str, int], time_dimension: Mapping[str, int],
) -> list[int]:
    derivative_dimension = dict(target_dimension)
    for key, exponent in time_dimension.items():
        derivative_dimension[key] = derivative_dimension.get(key, 0) - exponent
    derivative_dimension = {key: value for key, value in derivative_dimension.items() if value}
    compatible = []
    for index, power in enumerate(powers):
        dimension: dict[str, int] = {}
        for state_index, exponent in enumerate(power):
            for key, value in state_dimensions[state_index].items():
                dimension[key] = dimension.get(key, 0) + exponent * value
        dimension = {key: value for key, value in dimension.items() if value}
        if dimension == derivative_dimension:
            compatible.append(index)
    return compatible


def discover_sparse_dynamics(
    times: Sequence[float], states: Sequence[Sequence[float]],
    state_names: Sequence[str] | None = None, *, polynomial_degree: int = 2,
    sparsity_threshold: float = 1e-5, max_terms_per_equation: int = 12,
    validation_fraction: float = 0.25, residual_tolerance: float = 1e-4,
    state_dimensions: Sequence[Mapping[str, int]] | None = None,
    time_dimension: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Discover polynomial ODE candidates with finite differences and STLSQ.

    When ``state_dimensions`` is provided, only library terms with the same
    dimensions as ``d(state)/dt`` are fitted for each equation.  This is a
    generation-time filter, not a proof that the chosen units or mechanism are
    correct.
    """
    t = _finite(times, "times", 1)
    x = _finite(states, "states", 2)
    if x.shape[0] != t.size or t.size < 16 or x.shape[1] < 1 or x.shape[1] > 8:
        raise SINDyDiscoveryError("trajectory_requires_16_to_8_states")
    if not np.all(np.diff(t) > 0):
        raise SINDyDiscoveryError("times_must_be_strictly_increasing")
    if type(polynomial_degree) is not int or not 1 <= polynomial_degree <= 3:
        raise SINDyDiscoveryError("invalid_polynomial_degree")
    if not 0 < float(sparsity_threshold) < 1 or not 0 < float(residual_tolerance):
        raise SINDyDiscoveryError("invalid_sparsity_or_residual_tolerance")
    if type(max_terms_per_equation) is not int or not 1 <= max_terms_per_equation <= 32:
        raise SINDyDiscoveryError("invalid_term_budget")
    if not 0.05 <= float(validation_fraction) <= 0.4:
        raise SINDyDiscoveryError("validation_fraction_must_be_between_0_05_and_0_4")
    names = list(state_names) if state_names is not None else [f"x_{i}" for i in range(x.shape[1])]
    if len(names) != x.shape[1] or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise SINDyDiscoveryError("state_names_must_match_and_be_unique")
    powers = _powers(x.shape[1], polynomial_degree)
    if len(powers) > 128:
        raise SINDyDiscoveryError("polynomial_library_too_large")
    dimensions = _normalize_dimensions(state_dimensions, x.shape[1], "state_dimensions")
    if time_dimension is not None and not isinstance(time_dimension, Mapping):
        raise SINDyDiscoveryError("time_dimension_must_be_mapping")
    normalized_time_dimension = ({str(key): int(value) for key, value in time_dimension.items()}
                                 if time_dimension is not None else {"T": 1})
    if time_dimension is not None:
        normalized_time_dimension = _normalize_dimensions([normalized_time_dimension], 1, "time_dimension")[0]
    derivatives = (x[2:] - x[:-2]) / (t[2:, None] - t[:-2, None])
    row_count = derivatives.shape[0]
    split = max(8, min(row_count - 4, int(math.floor(row_count * (1.0 - float(validation_fraction))))))
    theta = _library(x[1:-1], powers)
    if not np.isfinite(theta).all():
        raise SINDyDiscoveryError("polynomial_library_overflow")
    train_theta, validation_theta = theta[:split], theta[split:]
    train_derivative, validation_derivative = derivatives[:split], derivatives[split:]
    allowed_indices = [list(range(len(powers))) for _ in range(x.shape[1])]
    if dimensions is not None:
        allowed_indices = [_compatible_library_indices(powers, dimensions, dimensions[index], normalized_time_dimension)
                           for index in range(x.shape[1])]
        if any(not indices for indices in allowed_indices):
            raise SINDyDiscoveryError("no_dimension_compatible_library_terms")
    coefficient_columns = []
    for index in range(x.shape[1]):
        fitted = _stlsq(train_theta[:, allowed_indices[index]], train_derivative[:, index],
                        threshold=float(sparsity_threshold), max_terms=max_terms_per_equation)
        column = np.zeros(len(powers), dtype=float)
        column[allowed_indices[index]] = fitted
        coefficient_columns.append(column)
    coefficients = np.column_stack(coefficient_columns)
    train_prediction = train_theta @ coefficients
    validation_prediction = validation_theta @ coefficients
    validation_residuals = validation_prediction - validation_derivative
    equations = []
    train_errors, validation_errors = [], []
    for index, name in enumerate(names):
        train_error = float(np.sqrt(np.mean((train_prediction[:, index] - train_derivative[:, index]) ** 2)))
        validation_error = float(np.sqrt(np.mean((validation_prediction[:, index] - validation_derivative[:, index]) ** 2)))
        train_errors.append(train_error)
        validation_errors.append(validation_error)
        terms = [{"term": _term_name(power, names), "coefficient": float(coefficients[term_index, index])}
                 for term_index, power in enumerate(powers) if abs(float(coefficients[term_index, index])) > 1e-10]
        equations.append({"lhs": f"d{name}/dt", "terms": terms,
                          "dimension_filter": "enabled" if dimensions is not None else "not_provided",
                          "allowed_library_terms": [_term_name(powers[item], names) for item in allowed_indices[index]]})
    payload = {"times": np.round(t, 15).tolist(), "states": np.round(x, 15).tolist(), "names": names,
               "degree": polynomial_degree, "state_dimensions": dimensions,
               "time_dimension": normalized_time_dimension}
    digest = sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    passed = all(error <= float(residual_tolerance) for error in validation_errors)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_found" if passed else "candidate_rejected_by_validation",
        "proof_status": "tested_not_falsified" if passed else "counterexample_found",
        "state_names": names, "library_size": len(powers),
        "library_terms": [_term_name(power, names) for power in powers],
        "dimension_filter": {"enabled": dimensions is not None, "state_dimensions": dimensions,
                             "time_dimension": normalized_time_dimension if dimensions is not None else None},
        "train_rows": int(split), "validation_rows": int(row_count - split),
        "equations": equations,
        "train_derivative_rmse": train_errors,
        "validation_derivative_rmse": validation_errors,
        "counterexamples": _counterexamples(t, validation_residuals, names, float(residual_tolerance), offset=split),
        "trajectory_sha256": digest,
        "policy": "finite_difference_polynomial_sindy_candidate_not_ode_proof",
    }


def discover_weak_form_dynamics(
    times: Sequence[float], states: Sequence[Sequence[float]],
    state_names: Sequence[str] | None = None, *, polynomial_degree: int = 2,
    sparsity_threshold: float = 1e-5, max_terms_per_equation: int = 12,
    validation_fraction: float = 0.25, residual_tolerance: float = 1e-4,
    window_count: int = 16,
) -> dict[str, Any]:
    """Fit a SINDy candidate through compactly supported weak-form integrals.

    For a window test function ``w`` that is zero at both endpoints,
    ``integral(w * x_dot) = -integral(w_dot * x)``.  This avoids directly
    differentiating every noisy observation, but it does not make the result
    noise-proof or identify the true dynamics.
    """
    t = _finite(times, "times", 1)
    x = _finite(states, "states", 2)
    if x.shape[0] != t.size or t.size < 32 or x.shape[1] < 1 or x.shape[1] > 8:
        raise SINDyDiscoveryError("weak_form_trajectory_requires_32_to_8_states")
    if not np.all(np.diff(t) > 0):
        raise SINDyDiscoveryError("times_must_be_strictly_increasing")
    if type(window_count) is not int or not 8 <= window_count <= 32:
        raise SINDyDiscoveryError("invalid_weak_form_window_budget")
    if not 0.05 <= float(validation_fraction) <= 0.4:
        raise SINDyDiscoveryError("validation_fraction_must_be_between_0_05_and_0_4")
    # Reuse the same library/contract checks as the pointwise route.
    if type(polynomial_degree) is not int or not 1 <= polynomial_degree <= 3:
        raise SINDyDiscoveryError("invalid_polynomial_degree")
    if not 0 < float(sparsity_threshold) < 1:
        raise SINDyDiscoveryError("invalid_sparsity_threshold")
    if not 0 < float(residual_tolerance):
        raise SINDyDiscoveryError("invalid_residual_tolerance")
    if type(max_terms_per_equation) is not int or not 1 <= max_terms_per_equation <= 32:
        raise SINDyDiscoveryError("invalid_term_budget")
    names = list(state_names) if state_names is not None else [f"x_{i}" for i in range(x.shape[1])]
    if len(names) != x.shape[1] or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
        raise SINDyDiscoveryError("state_names_must_match_and_be_unique")
    powers = _powers(x.shape[1], polynomial_degree)
    if len(powers) > 128:
        raise SINDyDiscoveryError("polynomial_library_too_large")
    window_length = max(8, min(t.size // 3, t.size // max(2, window_count // 2)))
    starts = np.linspace(0, t.size - window_length, num=window_count, dtype=int)
    starts = list(dict.fromkeys(int(value) for value in starts))
    feature_rows, target_rows = [], []
    for start in starts:
        stop = start + window_length
        local_t, local_x = t[start:stop], x[start:stop]
        span = float(local_t[-1] - local_t[0])
        if span <= 0:
            raise SINDyDiscoveryError("weak_form_window_has_no_time_span")
        s = (local_t - local_t[0]) / span
        weight = np.sin(np.pi * s) ** 2
        weight_derivative = np.pi * np.sin(2.0 * np.pi * s) / span
        library_values = _library(local_x, powers)
        feature_rows.append(_integrate(weight[:, None] * library_values, local_t, axis=0))
        target_rows.append(-_integrate(weight_derivative[:, None] * local_x, local_t, axis=0))
    features, targets = np.asarray(feature_rows), np.asarray(target_rows)
    if not np.isfinite(features).all() or not np.isfinite(targets).all():
        raise SINDyDiscoveryError("weak_form_integral_overflow")
    split = max(4, min(features.shape[0] - 2, int(math.floor(features.shape[0] * (1.0 - float(validation_fraction))))))
    coefficients = np.column_stack([
        _stlsq(features[:split], targets[:split, index], threshold=float(sparsity_threshold),
               max_terms=max_terms_per_equation)
        for index in range(x.shape[1])
    ])
    train_prediction, validation_prediction = features[:split] @ coefficients, features[split:] @ coefficients
    validation_residuals = validation_prediction - targets[split:]
    equations = []
    train_errors, validation_errors = [], []
    for index, name in enumerate(names):
        train_errors.append(float(np.sqrt(np.mean((train_prediction[:, index] - targets[:split, index]) ** 2))))
        validation_errors.append(float(np.sqrt(np.mean((validation_prediction[:, index] - targets[split:, index]) ** 2))))
        equations.append({"lhs": f"d{name}/dt", "terms": [
            {"term": _term_name(power, names), "coefficient": float(coefficients[term_index, index])}
            for term_index, power in enumerate(powers) if abs(float(coefficients[term_index, index])) > 1e-10
        ]})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_generated",
        "proof_status": "tested_not_falsified",
        "state_names": names, "library_size": len(powers), "window_count": len(starts),
        "train_windows": int(split), "validation_windows": int(features.shape[0] - split),
        "equations": equations, "train_weak_form_rmse": train_errors,
        "validation_weak_form_rmse": validation_errors,
        "counterexamples": [{"window": int(split + row), "equation": f"d{names[equation]}/dt",
                             "absolute_residual": float(abs(error)), "origin": "heldout_weak_form"}
                            for row, values in enumerate(np.abs(validation_residuals))
                            for equation, error in enumerate(values) if float(error) > float(residual_tolerance)][:32],
        "policy": "weak_form_candidate_not_noise_proof_or_ode_proof",
    }


__all__ = ["SCHEMA_VERSION", "SINDyDiscoveryError", "discover_sparse_dynamics", "discover_weak_form_dynamics"]
