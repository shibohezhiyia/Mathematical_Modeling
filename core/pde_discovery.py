"""Bounded 1-D space/time PDE candidate discovery.

The implementation intentionally covers a small, explicit library (reaction,
advection and diffusion terms).  It is a real fit/holdout computation for
regular grids, not a claim to solve arbitrary PDEs or infer boundary physics.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.pde-discovery/v1"


class PDEDiscoveryError(ValueError):
    pass


def _forward_rollout(times: np.ndarray, coordinates: np.ndarray, initial: np.ndarray,
                     coefficients: np.ndarray, terms: list[str],
                     boundary_values: np.ndarray | None = None) -> dict[str, Any]:
    """Explicitly integrate the discovered finite-difference PDE.

    Boundaries are held at their initial values.  This is deliberately a
    forward diagnostic, not a general PDE solver; unstable or non-finite
    rollouts are returned as evidence instead of being hidden.
    """
    dt, dx = float(np.mean(np.diff(times))), float(np.mean(np.diff(coordinates)))
    values = np.zeros((times.size, coordinates.size), dtype=float)
    values[0] = initial
    if boundary_values is not None:
        values[:, 0], values[:, -1] = boundary_values[:, 0], boundary_values[:, 1]
    else:
        values[:, 0], values[:, -1] = initial[0], initial[-1]
    coeff = {name: float(value) for name, value in zip(terms, coefficients)}
    diffusion = abs(coeff.get("u_xx", 0.0)) * dt / max(dx * dx, 1e-15)
    advection = abs(coeff.get("u_x", 0.0)) * dt / max(dx, 1e-15)
    stability = {"diffusion_number": diffusion, "advection_number": advection,
                 "explicit_cfl_satisfied": bool(diffusion <= 0.5 and advection <= 1.0)}
    for index in range(1, times.size):
        previous = values[index - 1]
        u = previous[1:-1]
        ux = (previous[2:] - previous[:-2]) / (2.0 * dx)
        uxx = (previous[2:] - 2.0 * previous[1:-1] + previous[:-2]) / (dx * dx)
        rhs = coeff.get("u", 0.0) * u + coeff.get("u_x", 0.0) * ux + coeff.get("u_xx", 0.0) * uxx
        values[index, 1:-1] = u + dt * rhs
        if boundary_values is not None:
            values[index, 0], values[index, -1] = boundary_values[index, 0], boundary_values[index, 1]
        if not np.isfinite(values[index]).all() or np.max(np.abs(values[index])) > 1e8:
            return {"status": "rejected_nonfinite_rollout", "stability": stability,
                    "failed_time_index": index}
    return {"status": "rollout_completed", "values": values,
            "stability": stability}


def _coarse_grid_check(times: np.ndarray, coordinates: np.ndarray, values: np.ndarray,
                       split: int, coefficients: np.ndarray, terms: list[str],
                       fine_rmse: float) -> dict[str, Any]:
    """Recompute finite-difference residual on a deterministic coarse grid."""
    if coordinates.size < 9 or times.size < 16:
        return {"status": "not_assessed", "reason": "grid_too_small"}
    coarse_x, coarse_values = coordinates[::2], values[:, ::2]
    dx = float(np.mean(np.diff(coarse_x)))
    dt = float(np.mean(np.diff(times)))
    if coarse_x.size < 5:
        return {"status": "not_assessed", "reason": "coarse_space_too_small"}
    u_t = np.gradient(coarse_values, dt, axis=0, edge_order=2)
    u_x = np.gradient(coarse_values, dx, axis=1, edge_order=2)
    u_xx = np.gradient(u_x, dx, axis=1, edge_order=2)
    values_by_term = {"u": coarse_values, "u_x": u_x, "u_xx": u_xx}
    rows = np.arange(split, times.size - 1)
    if rows.size < 3:
        return {"status": "not_assessed", "reason": "coarse_holdout_too_small"}
    matrix = np.column_stack([values_by_term[name][rows, 1:-1].reshape(-1) for name in terms])
    residual = matrix @ coefficients - u_t[rows, 1:-1].reshape(-1)
    coarse_rmse = float(np.sqrt(np.mean(residual ** 2)))
    ratio = coarse_rmse / fine_rmse if fine_rmse > 1e-15 else (0.0 if coarse_rmse <= 1e-15 else None)
    return {"status": "assessed", "coarse_validation_rmse": coarse_rmse,
            "fine_validation_rmse": float(fine_rmse), "rmse_ratio": ratio,
            "coarse_space_points": int(coarse_x.size), "stride": 2}


def _array(value: Any, name: str, ndim: int) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PDEDiscoveryError(f"{name}_must_be_numeric") from exc
    if result.ndim != ndim or result.size == 0 or not np.isfinite(result).all():
        raise PDEDiscoveryError(f"{name}_must_be_finite_{ndim}d_array")
    return result


def discover_1d_pde(
    times: Sequence[float], coordinates: Sequence[float], field: Sequence[Sequence[float]],
    *, validation_fraction: float = 0.25, ridge: float = 1e-8,
    residual_tolerance: float = 1e-3, boundary_tolerance: float = 1e-2,
    include_advection: bool = True, max_terms: int = 4, sparsity_threshold: float = 0.0,
    field_mask: Sequence[Sequence[bool]] | None = None, noise_scale: float | None = None,
    field_dimensions: Mapping[str, float] | None = None,
    coordinate_dimensions: Mapping[str, float] | None = None,
    boundary_values: Sequence[Sequence[float]] | None = None,
) -> dict[str, Any]:
    """Fit ``u_t = a*u + b*u_x + c*u_xx`` on a regular 1-D grid."""
    t, x, values = _array(times, "times", 1), _array(coordinates, "coordinates", 1), _array(field, "field", 2)
    if t.size < 16 or x.size < 5 or values.shape != (t.size, x.size):
        raise PDEDiscoveryError("pde_requires_at_least_16_times_5_space_points")
    if not np.all(np.diff(t) > 0) or not np.all(np.diff(x) > 0):
        raise PDEDiscoveryError("pde_grids_must_be_strictly_increasing")
    if field_mask is not None:
        mask = np.asarray(field_mask)
        if mask.shape != values.shape or mask.dtype.kind not in {"b", "i", "u"}:
            raise PDEDiscoveryError("pde_field_mask_shape_invalid")
        if not np.all(mask.astype(bool)):
            # Finite differences cannot infer an arbitrary missing-data pattern;
            # reject it rather than silently interpolating across a boundary.
            raise PDEDiscoveryError("pde_missing_field_values_not_supported")
    if noise_scale is not None and (type(noise_scale) not in (int, float) or not math.isfinite(float(noise_scale)) or float(noise_scale) < 0):
        raise PDEDiscoveryError("pde_noise_scale_invalid")
    for dimensions, code in ((field_dimensions, "pde_field_dimensions_invalid"),
                             (coordinate_dimensions, "pde_coordinate_dimensions_invalid")):
        if dimensions is not None and (not isinstance(dimensions, Mapping) or len(dimensions) > 16 or
                                       any(type(key) is not str or type(value) not in (int, float) or not math.isfinite(float(value))
                                           for key, value in dimensions.items())):
            raise PDEDiscoveryError(code)
    declared_unit_status = "declared" if field_dimensions is not None and coordinate_dimensions is not None else "not_assessed"
    if boundary_values is not None:
        declared_boundary = _array(boundary_values, "boundary_values", 2)
        if declared_boundary.shape != (t.size, 2) or not np.isfinite(declared_boundary).all():
            raise PDEDiscoveryError("pde_boundary_values_shape_invalid")
    else:
        declared_boundary = None
    dt, dx = np.diff(t), np.diff(x)
    if np.max(dt) - np.min(dt) > max(1e-10, 1e-6 * float(np.mean(dt))):
        raise PDEDiscoveryError("time_grid_must_be_regular")
    if np.max(dx) - np.min(dx) > max(1e-10, 1e-6 * float(np.mean(dx))):
        raise PDEDiscoveryError("space_grid_must_be_regular")
    if not 0.1 <= float(validation_fraction) <= 0.4:
        raise PDEDiscoveryError("validation_fraction_invalid")
    if not math.isfinite(float(ridge)) or not 0 < float(ridge) <= 1e6:
        raise PDEDiscoveryError("ridge_invalid")
    if not math.isfinite(float(residual_tolerance)) or not 0 < float(residual_tolerance):
        raise PDEDiscoveryError("residual_tolerance_invalid")
    if not math.isfinite(float(boundary_tolerance)) or not 0 < float(boundary_tolerance):
        raise PDEDiscoveryError("boundary_tolerance_invalid")
    if not math.isfinite(float(sparsity_threshold)) or not 0 <= float(sparsity_threshold) <= 1:
        raise PDEDiscoveryError("sparsity_threshold_invalid")
    if type(include_advection) is not bool or type(max_terms) is not int or not 1 <= max_terms <= 4:
        raise PDEDiscoveryError("pde_options_invalid")

    dt_value, dx_value = float(np.mean(dt)), float(np.mean(dx))
    u_t = np.gradient(values, dt_value, axis=0, edge_order=2)
    u_x = np.gradient(values, dx_value, axis=1, edge_order=2)
    u_xx = np.gradient(u_x, dx_value, axis=1, edge_order=2)
    term_values = [values, u_x, u_xx] if include_advection else [values, u_xx]
    term_names = ["u", "u_x", "u_xx"] if include_advection else ["u", "u_xx"]
    if len(term_names) > max_terms:
        term_values, term_names = term_values[:max_terms], term_names[:max_terms]
    split = max(8, min(t.size - 4, int(math.floor(t.size * (1.0 - float(validation_fraction))))))
    train_rows = np.arange(1, split)
    valid_rows = np.arange(split, t.size - 1)
    if len(train_rows) < len(term_names) or len(valid_rows) < 3:
        raise PDEDiscoveryError("pde_holdout_too_small")
    train_matrix = np.column_stack([item[train_rows, 1:-1].reshape(-1) for item in term_values])
    train_target = u_t[train_rows, 1:-1].reshape(-1)
    gram = train_matrix.T @ train_matrix + float(ridge) * np.eye(len(term_names))
    try:
        coefficients = np.linalg.solve(gram, train_matrix.T @ train_target)
    except np.linalg.LinAlgError as exc:
        raise PDEDiscoveryError("pde_fit_solve_failed") from exc
    if not np.isfinite(coefficients).all():
        raise PDEDiscoveryError("pde_fit_nonfinite")
    selected = np.ones(len(term_names), dtype=bool)
    if float(sparsity_threshold) > 0 and np.max(np.abs(coefficients)) > 0:
        selected = np.abs(coefficients) >= float(sparsity_threshold) * np.max(np.abs(coefficients))
        if not selected.any():
            selected[np.argmax(np.abs(coefficients))] = True
        if selected.sum() < len(coefficients):
            reduced = train_matrix[:, selected]
            reduced_gram = reduced.T @ reduced + float(ridge) * np.eye(int(selected.sum()))
            reduced_coefficients = np.linalg.solve(reduced_gram, reduced.T @ train_target)
            coefficients = np.zeros(len(term_names), dtype=float)
            coefficients[selected] = reduced_coefficients
    def residual(rows: np.ndarray, interior: bool = True) -> np.ndarray:
        section = slice(1, -1) if interior else slice(None)
        matrix = np.column_stack([item[rows, section].reshape(-1) for item in term_values])
        return matrix @ coefficients - u_t[rows, section].reshape(-1)
    train_residual = residual(train_rows)
    valid_residual = residual(valid_rows)
    boundary_indices = np.ix_(valid_rows, np.array([0, values.shape[1] - 1]))
    boundary_matrix = np.column_stack([item[boundary_indices].reshape(-1) for item in term_values])
    boundary_residual = boundary_matrix @ coefficients - u_t[boundary_indices].reshape(-1)
    validation_rmse = float(np.sqrt(np.mean(valid_residual ** 2)))
    boundary_rmse = float(np.sqrt(np.mean(boundary_residual ** 2)))
    grid_check = _coarse_grid_check(t, x, values, split, coefficients, term_names, validation_rmse)
    rollout = _forward_rollout(t, x, values[0], coefficients, term_names, declared_boundary)
    if rollout["status"] == "rollout_completed":
        valid_slice = slice(split, t.size)
        forward_rmse = float(np.sqrt(np.mean((rollout["values"][valid_slice] - values[valid_slice]) ** 2)))
        forward_max_error = float(np.max(np.abs(rollout["values"][valid_slice] - values[valid_slice])))
    else:
        forward_rmse, forward_max_error = None, None
    payload = {"times": np.round(t, 12).tolist(), "coordinates": np.round(x, 12).tolist(),
               "field": np.round(values, 12).tolist(), "terms": term_names,
               "field_dimensions": dict(field_dimensions or {}),
               "coordinate_dimensions": dict(coordinate_dimensions or {}),
               "noise_scale": None if noise_scale is None else float(noise_scale),
               "boundary_values": None if declared_boundary is None else np.round(declared_boundary, 12).tolist()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    boundary_condition_rmse = None
    if declared_boundary is not None:
        observed_boundary = values[valid_rows][:, [0, -1]]
        boundary_condition_rmse = float(np.sqrt(np.mean((observed_boundary - declared_boundary[valid_rows]) ** 2)))
    passed = validation_rmse <= float(residual_tolerance) and boundary_rmse <= float(boundary_tolerance)
    if boundary_condition_rmse is not None:
        passed = passed and boundary_condition_rmse <= float(boundary_tolerance)
    witnesses = []
    for index, error in sorted(enumerate(np.abs(valid_residual)), key=lambda item: -item[1])[:32]:
        if float(error) > float(residual_tolerance):
            witnesses.append({"time_index": int(valid_rows[index // max(1, x.size - 2)]),
                              "absolute_residual": float(error), "origin": "pde_validation"})
    return {
        "schema_version": SCHEMA_VERSION, "status": "candidate_found" if passed else "candidate_rejected_by_validation",
        "proof_status": "tested_not_falsified" if passed else "counterexample_found",
        "terms": term_names, "coefficients": [float(value) for value in coefficients],
        "active_terms": [name for name, keep in zip(term_names, selected) if keep],
        "sparsity_threshold": float(sparsity_threshold),
        "train_time_rows": int(len(train_rows)), "validation_time_rows": int(len(valid_rows)),
        "validation_rmse": validation_rmse, "boundary_rmse": boundary_rmse,
        "forward_rollout_status": rollout["status"], "forward_rollout_rmse": forward_rmse,
        "forward_rollout_max_error": forward_max_error, "forward_stability": rollout["stability"],
        "boundary_condition_rmse": boundary_condition_rmse,
        "grid_convergence": grid_check,
        "grid": {"time_step": dt_value, "space_step": dx_value, "field_shape": list(values.shape)},
        "unit_status": declared_unit_status,
        "noise_status": "declared" if noise_scale is not None else "not_assessed",
        "missing_status": "none_observed" if field_mask is None or np.all(np.asarray(field_mask, dtype=bool)) else "not_supported",
        "boundary_status": "declared_applied" if declared_boundary is not None else "initial_boundary_only",
        "counterexamples": witnesses, "field_sha256": digest,
        "policy": "regular_grid_finite_difference_pde_candidate; not_arbitrary_pde_solver_or_boundary_proof",
    }


def discover_2d_pde(
    times: Sequence[float], x_coordinates: Sequence[float], y_coordinates: Sequence[float],
    field: Sequence[Sequence[Sequence[float]]], *, validation_fraction: float = 0.25,
    ridge: float = 1e-8, residual_tolerance: float = 1e-3,
    include_advection: bool = True, sparsity_threshold: float = 0.0,
    max_rows: int = 2_000_000,
    boundary_values: Mapping[str, Sequence[Sequence[float]]] | None = None,
    boundary_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Fit a bounded 2-D regular-grid PDE candidate.

    The library is limited to u, u_x, u_y, u_xx and u_yy. It uses a temporal
    holdout and an explicit forward Euler diagnostic; it does not infer
    arbitrary mixed derivatives or irregular boundary physics. When declared,
    time-varying Dirichlet edges are applied during the rollout and scored
    against the observed validation field.
    """
    t, x, y, values = (_array(times, "times", 1), _array(x_coordinates, "x_coordinates", 1),
                       _array(y_coordinates, "y_coordinates", 1), _array(field, "field", 3))
    if t.size < 12 or x.size < 5 or y.size < 5 or values.shape != (t.size, x.size, y.size):
        raise PDEDiscoveryError("pde_2d_requires_at_least_12_times_5_by_5_space_points")
    if not np.all(np.diff(t) > 0) or not np.all(np.diff(x) > 0) or not np.all(np.diff(y) > 0):
        raise PDEDiscoveryError("pde_2d_grids_must_be_strictly_increasing")
    if type(max_rows) is not int or max_rows < 100 or int(values.size) > int(max_rows):
        raise PDEDiscoveryError("pde_2d_size_budget_exceeded")
    if not 0.1 <= float(validation_fraction) <= 0.4 or not math.isfinite(float(ridge)) or ridge <= 0:
        raise PDEDiscoveryError("pde_2d_options_invalid")
    if not math.isfinite(float(residual_tolerance)) or residual_tolerance <= 0:
        raise PDEDiscoveryError("pde_2d_residual_tolerance_invalid")
    if not math.isfinite(float(sparsity_threshold)) or not 0 <= sparsity_threshold <= 1:
        raise PDEDiscoveryError("pde_2d_sparsity_threshold_invalid")
    if not math.isfinite(float(boundary_tolerance)) or boundary_tolerance <= 0:
        raise PDEDiscoveryError("pde_2d_boundary_tolerance_invalid")
    declared_boundary: dict[str, np.ndarray] | None = None
    if boundary_values is not None:
        if not isinstance(boundary_values, Mapping):
            raise PDEDiscoveryError("pde_2d_boundary_values_must_be_mapping")
        allowed_edges = {"left", "right", "bottom", "top"}
        if not boundary_values:
            raise PDEDiscoveryError("pde_2d_boundary_values_empty")
        unknown_edges = set(boundary_values) - allowed_edges
        if unknown_edges:
            raise PDEDiscoveryError("pde_2d_boundary_edge_unknown")
        declared_boundary = {}
        expected_shapes = {
            "left": (t.size, y.size), "right": (t.size, y.size),
            "bottom": (t.size, x.size), "top": (t.size, x.size),
        }
        for edge, raw in boundary_values.items():
            edge_values = _array(raw, f"boundary_values.{edge}", 2)
            if edge_values.shape != expected_shapes[edge] or not np.isfinite(edge_values).all():
                raise PDEDiscoveryError("pde_2d_boundary_values_shape_or_finite_invalid")
            declared_boundary[edge] = edge_values
        for corner in ("bottom", "top"):
            horizontal = declared_boundary.get(corner)
            for edge, index in (("left", 0), ("right", -1)):
                vertical = declared_boundary.get(edge)
                if horizontal is not None and vertical is not None:
                    if not np.allclose(horizontal[:, index], vertical[:, 0 if corner == "bottom" else -1],
                                       rtol=0.0, atol=float(boundary_tolerance)):
                        raise PDEDiscoveryError("pde_2d_boundary_corner_conflict")
    dt, dx, dy = float(np.mean(np.diff(t))), float(np.mean(np.diff(x))), float(np.mean(np.diff(y)))
    if (np.max(np.abs(np.diff(t) - dt)) > max(1e-10, abs(dt) * 1e-6)
            or np.max(np.abs(np.diff(x) - dx)) > max(1e-10, abs(dx) * 1e-6)
            or np.max(np.abs(np.diff(y) - dy)) > max(1e-10, abs(dy) * 1e-6)):
        raise PDEDiscoveryError("pde_2d_grid_must_be_regular")
    u_t = np.gradient(values, dt, axis=0, edge_order=2)
    u_x = np.gradient(values, dx, axis=1, edge_order=2)
    u_y = np.gradient(values, dy, axis=2, edge_order=2)
    u_xx = np.gradient(u_x, dx, axis=1, edge_order=2)
    u_yy = np.gradient(u_y, dy, axis=2, edge_order=2)
    all_terms = [("u", values), ("u_x", u_x), ("u_y", u_y), ("u_xx", u_xx), ("u_yy", u_yy)]
    terms = all_terms if include_advection else [all_terms[0], all_terms[3], all_terms[4]]
    split = max(6, min(t.size - 3, int(math.floor(t.size * (1.0 - float(validation_fraction))))))
    train_rows, valid_rows = np.arange(1, split), np.arange(split, t.size - 1)
    interior = (slice(1, -1), slice(1, -1))
    matrix = np.column_stack([term[train_rows, interior[0], interior[1]].reshape(-1) for _, term in terms])
    target = u_t[train_rows, interior[0], interior[1]].reshape(-1)
    gram = matrix.T @ matrix + float(ridge) * np.eye(len(terms))
    try:
        coefficients = np.linalg.solve(gram, matrix.T @ target)
    except np.linalg.LinAlgError as exc:
        raise PDEDiscoveryError("pde_2d_fit_solve_failed") from exc
    selected = np.ones(len(terms), dtype=bool)
    if sparsity_threshold > 0 and np.max(np.abs(coefficients)) > 0:
        selected = np.abs(coefficients) >= float(sparsity_threshold) * np.max(np.abs(coefficients))
        if not selected.any():
            selected[np.argmax(np.abs(coefficients))] = True
        reduced = matrix[:, selected]
        coefficients = np.zeros(len(terms), dtype=float)
        coefficients[selected] = np.linalg.solve(
            reduced.T @ reduced + float(ridge) * np.eye(int(selected.sum())), reduced.T @ target)
    valid_matrix = np.column_stack([term[valid_rows, interior[0], interior[1]].reshape(-1) for _, term in terms])
    valid_residual = valid_matrix @ coefficients - u_t[valid_rows, interior[0], interior[1]].reshape(-1)
    validation_rmse = float(np.sqrt(np.mean(valid_residual ** 2)))
    diffusion_number = 0.0
    for index, (name, _) in enumerate(terms):
        if name == "u_xx":
            diffusion_number += abs(float(coefficients[index])) * dt / max(dx * dx, 1e-15)
        elif name == "u_yy":
            diffusion_number += abs(float(coefficients[index])) * dt / max(dy * dy, 1e-15)
    rollout = np.zeros_like(values)
    rollout[0] = values[0]
    for index in range(1, t.size):
        previous = rollout[index - 1]
        ux_now = np.gradient(previous, dx, axis=0, edge_order=2)
        uy_now = np.gradient(previous, dy, axis=1, edge_order=2)
        derivs = {"u": previous, "u_x": ux_now, "u_y": uy_now,
                  "u_xx": np.gradient(ux_now, dx, axis=0, edge_order=2),
                  "u_yy": np.gradient(uy_now, dy, axis=1, edge_order=2)}
        rhs = sum(float(coef) * derivs[name] for coef, (name, _) in zip(coefficients, terms))
        rollout[index] = previous + dt * rhs
        if declared_boundary is None:
            rollout[index, 0, :] = values[0, 0, :]
            rollout[index, -1, :] = values[0, -1, :]
            rollout[index, :, 0] = values[0, :, 0]
            rollout[index, :, -1] = values[0, :, -1]
        else:
            if "left" in declared_boundary:
                rollout[index, 0, :] = declared_boundary["left"][index]
            if "right" in declared_boundary:
                rollout[index, -1, :] = declared_boundary["right"][index]
            if "bottom" in declared_boundary:
                rollout[index, :, 0] = declared_boundary["bottom"][index]
            if "top" in declared_boundary:
                rollout[index, :, -1] = declared_boundary["top"][index]
        if not np.isfinite(rollout[index]).all() or np.max(np.abs(rollout[index])) > 1e8:
            return {"schema_version": SCHEMA_VERSION, "status": "rejected_nonfinite_rollout",
                    "terms": [name for name, _ in terms], "coefficients": coefficients.tolist(),
                    "validation_rmse": validation_rmse,
                    "forward_stability": {"diffusion_number": diffusion_number,
                                          "explicit_cfl_satisfied": diffusion_number <= 0.5}}
    forward_rmse = float(np.sqrt(np.mean((rollout[split:] - values[split:]) ** 2)))
    boundary_condition_rmse = None
    if declared_boundary is not None:
        errors: list[np.ndarray] = []
        valid_boundary_rows = np.arange(split, t.size)
        edge_specs = {
            "left": lambda row: values[row, 0, :],
            "right": lambda row: values[row, -1, :],
            "bottom": lambda row: values[row, :, 0],
            "top": lambda row: values[row, :, -1],
        }
        for edge, getter in edge_specs.items():
            if edge in declared_boundary:
                errors.append(np.concatenate([
                    getter(row) - declared_boundary[edge][row]
                    for row in valid_boundary_rows
                ]))
        if errors:
            boundary_condition_rmse = float(np.sqrt(np.mean(np.concatenate(errors) ** 2)))
    cell_count = max(1, (x.size - 2) * (y.size - 2))
    witnesses = [{"time_index": int(valid_rows[index // cell_count]),
                  "absolute_residual": float(abs(error)), "origin": "pde_2d_validation"}
                 for index, error in sorted(enumerate(valid_residual), key=lambda item: -abs(item[1]))[:32]
                 if abs(error) > residual_tolerance]
    if boundary_condition_rmse is not None and boundary_condition_rmse > float(boundary_tolerance):
        witnesses.append({"origin": "pde_2d_boundary_validation",
                          "absolute_residual": boundary_condition_rmse,
                          "tolerance": float(boundary_tolerance)})
    if x.size >= 9 and y.size >= 9:
        coarse = values[:, ::2, ::2]
        coarse_dx = float(np.mean(np.diff(x[::2])))
        coarse_dy = float(np.mean(np.diff(y[::2])))
        coarse_ut = np.gradient(coarse, dt, axis=0, edge_order=2)
        coarse_ux = np.gradient(coarse, coarse_dx, axis=1, edge_order=2)
        coarse_uy = np.gradient(coarse, coarse_dy, axis=2, edge_order=2)
        coarse_terms = {
            "u": coarse, "u_x": coarse_ux, "u_y": coarse_uy,
            "u_xx": np.gradient(coarse_ux, coarse_dx, axis=1, edge_order=2),
            "u_yy": np.gradient(coarse_uy, coarse_dy, axis=2, edge_order=2),
        }
        coarse_matrix = np.column_stack([
            coarse_terms[name][valid_rows, 1:-1, 1:-1].reshape(-1) for name, _ in terms
        ])
        coarse_target = coarse_ut[valid_rows, 1:-1, 1:-1].reshape(-1)
        coarse_residual = coarse_matrix @ coefficients - coarse_target
        grid_convergence = {
            "status": "assessed", "coarse_validation_rmse": float(np.sqrt(np.mean(coarse_residual ** 2))),
            "fine_validation_rmse": validation_rmse,
            "rmse_ratio": float(np.sqrt(np.mean(coarse_residual ** 2)) / max(validation_rmse, 1e-15)),
            "coarse_x_points": int(coarse.shape[1]), "coarse_y_points": int(coarse.shape[2]),
            "stride": 2,
        }
    else:
        grid_convergence = {"status": "not_assessed", "reason": "grid_too_small"}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_found" if not witnesses else "candidate_rejected_by_validation",
        "proof_status": "tested_not_falsified" if not witnesses else "counterexample_found",
        "terms": [name for name, _ in terms], "coefficients": coefficients.astype(float).tolist(),
        "active_terms": [name for (name, _), keep in zip(terms, selected) if keep],
        "validation_rmse": validation_rmse, "forward_rollout_rmse": forward_rmse,
        "boundary_condition_rmse": boundary_condition_rmse,
        "boundary_status": "declared_applied" if declared_boundary is not None else "initial_boundary_only",
        "grid_convergence": grid_convergence,
        "forward_stability": {"diffusion_number": diffusion_number,
                              "explicit_cfl_satisfied": diffusion_number <= 0.5},
        "grid": {"time_step": dt, "x_step": dx, "y_step": dy, "field_shape": list(values.shape)},
        "train_time_rows": int(len(train_rows)), "validation_time_rows": int(len(valid_rows)),
        "counterexamples": witnesses,
        "policy": "regular_grid_finite_difference_2d_pde_candidate; not_arbitrary_pde_solver_or_boundary_proof",
    }


def discover_3d_pde(
    times: Sequence[float], x_coordinates: Sequence[float], y_coordinates: Sequence[float],
    z_coordinates: Sequence[float], field: Sequence[Sequence[Sequence[Sequence[float]]]], *,
    validation_fraction: float = 0.25, ridge: float = 1e-8,
    residual_tolerance: float = 1e-3, include_advection: bool = True,
    sparsity_threshold: float = 0.0, max_rows: int = 2_000_000,
) -> dict[str, Any]:
    """Fit a bounded 3-D regular-grid PDE candidate.

    The supported library is ``u, u_x, u_y, u_z, u_xx, u_yy, u_zz`` with a
    temporal holdout and explicit Euler rollout.  It deliberately excludes
    mixed derivatives, irregular meshes, variable coefficients and inferred
    boundary physics.
    """
    t = _array(times, "times", 1); x = _array(x_coordinates, "x_coordinates", 1)
    y = _array(y_coordinates, "y_coordinates", 1); z = _array(z_coordinates, "z_coordinates", 1)
    values = _array(field, "field", 4)
    if t.size < 12 or min(x.size, y.size, z.size) < 5 or values.shape != (t.size, x.size, y.size, z.size):
        raise PDEDiscoveryError("pde_3d_requires_at_least_12_times_5_by_5_by_5_space_points")
    if any(np.any(np.diff(axis) <= 0) for axis in (t, x, y, z)):
        raise PDEDiscoveryError("pde_3d_grids_must_be_strictly_increasing")
    if type(max_rows) is not int or max_rows < 100 or int(values.size) > int(max_rows):
        raise PDEDiscoveryError("pde_3d_size_budget_exceeded")
    if not 0.1 <= float(validation_fraction) <= 0.4 or not math.isfinite(float(ridge)) or ridge <= 0:
        raise PDEDiscoveryError("pde_3d_options_invalid")
    if not math.isfinite(float(residual_tolerance)) or residual_tolerance <= 0:
        raise PDEDiscoveryError("pde_3d_residual_tolerance_invalid")
    if not math.isfinite(float(sparsity_threshold)) or not 0 <= sparsity_threshold <= 1:
        raise PDEDiscoveryError("pde_3d_sparsity_threshold_invalid")
    dt, dx, dy, dz = (float(np.mean(np.diff(axis))) for axis in (t, x, y, z))
    for axis, step in ((t, dt), (x, dx), (y, dy), (z, dz)):
        if np.max(np.abs(np.diff(axis) - step)) > max(1e-10, abs(step) * 1e-6):
            raise PDEDiscoveryError("pde_3d_grid_must_be_regular")
    u_t = np.gradient(values, dt, axis=0, edge_order=2)
    u_x = np.gradient(values, dx, axis=1, edge_order=2); u_y = np.gradient(values, dy, axis=2, edge_order=2)
    u_z = np.gradient(values, dz, axis=3, edge_order=2)
    u_xx = np.gradient(u_x, dx, axis=1, edge_order=2); u_yy = np.gradient(u_y, dy, axis=2, edge_order=2)
    u_zz = np.gradient(u_z, dz, axis=3, edge_order=2)
    all_terms = [("u", values), ("u_x", u_x), ("u_y", u_y), ("u_z", u_z),
                 ("u_xx", u_xx), ("u_yy", u_yy), ("u_zz", u_zz)]
    terms = all_terms if include_advection else [all_terms[0], all_terms[4], all_terms[5], all_terms[6]]
    split = max(6, min(t.size - 3, int(math.floor(t.size * (1.0 - float(validation_fraction))))))
    train_rows, valid_rows = np.arange(1, split), np.arange(split, t.size - 1)
    interior = (slice(1, -1), slice(1, -1), slice(1, -1))
    matrix = np.column_stack([term[train_rows, *interior].reshape(-1) for _, term in terms])
    target = u_t[train_rows, *interior].reshape(-1)
    coefficients = np.linalg.solve(matrix.T @ matrix + float(ridge) * np.eye(len(terms)), matrix.T @ target)
    selected = np.ones(len(terms), dtype=bool)
    if sparsity_threshold > 0 and np.max(np.abs(coefficients)) > 0:
        selected = np.abs(coefficients) >= float(sparsity_threshold) * np.max(np.abs(coefficients))
        if not selected.any(): selected[np.argmax(np.abs(coefficients))] = True
        reduced = matrix[:, selected]
        coefficients = np.zeros(len(terms), dtype=float)
        coefficients[selected] = np.linalg.solve(reduced.T @ reduced + float(ridge) * np.eye(int(selected.sum())), reduced.T @ target)
    valid_matrix = np.column_stack([term[valid_rows, *interior].reshape(-1) for _, term in terms])
    valid_residual = valid_matrix @ coefficients - u_t[valid_rows, *interior].reshape(-1)
    validation_rmse = float(np.sqrt(np.mean(valid_residual ** 2)))
    diffusion_number = (abs(float(coefficients[[name for name, _ in terms].index("u_xx")])) * dt / max(dx * dx, 1e-15) if "u_xx" in [name for name, _ in terms] else 0.0)
    if "u_yy" in [name for name, _ in terms]: diffusion_number += abs(float(coefficients[[name for name, _ in terms].index("u_yy")])) * dt / max(dy * dy, 1e-15)
    if "u_zz" in [name for name, _ in terms]: diffusion_number += abs(float(coefficients[[name for name, _ in terms].index("u_zz")])) * dt / max(dz * dz, 1e-15)
    rollout = np.zeros_like(values); rollout[0] = values[0]
    for index in range(1, t.size):
        previous = rollout[index - 1]
        ux = np.gradient(previous, dx, axis=0, edge_order=2); uy = np.gradient(previous, dy, axis=1, edge_order=2); uz = np.gradient(previous, dz, axis=2, edge_order=2)
        derivs = {"u": previous, "u_x": ux, "u_y": uy, "u_z": uz,
                  "u_xx": np.gradient(ux, dx, axis=0, edge_order=2), "u_yy": np.gradient(uy, dy, axis=1, edge_order=2), "u_zz": np.gradient(uz, dz, axis=2, edge_order=2)}
        rhs = sum(float(coef) * derivs[name] for coef, (name, _) in zip(coefficients, terms))
        rollout[index] = previous + dt * rhs
        rollout[index, 0, :, :] = values[0, 0, :, :]; rollout[index, -1, :, :] = values[0, -1, :, :]
        rollout[index, :, 0, :] = values[0, :, 0, :]; rollout[index, :, -1, :] = values[0, :, -1, :]
        rollout[index, :, :, 0] = values[0, :, :, 0]; rollout[index, :, :, -1] = values[0, :, :, -1]
        if not np.isfinite(rollout[index]).all() or np.max(np.abs(rollout[index])) > 1e8:
            return {"schema_version": SCHEMA_VERSION, "status": "rejected_nonfinite_rollout", "terms": [name for name, _ in terms], "coefficients": coefficients.tolist(), "validation_rmse": validation_rmse}
    forward_rmse = float(np.sqrt(np.mean((rollout[split:] - values[split:]) ** 2)))
    witnesses = [{"time_index": int(valid_rows[index // max(1, (x.size - 2) * (y.size - 2) * (z.size - 2))]), "absolute_residual": float(abs(error)), "origin": "pde_3d_validation"}
                 for index, error in sorted(enumerate(valid_residual), key=lambda item: -abs(item[1]))[:32] if abs(error) > residual_tolerance]
    return {"schema_version": SCHEMA_VERSION, "status": "candidate_found" if not witnesses else "candidate_rejected_by_validation", "proof_status": "tested_not_falsified" if not witnesses else "counterexample_found", "terms": [name for name, _ in terms], "coefficients": coefficients.astype(float).tolist(), "active_terms": [name for (name, _), keep in zip(terms, selected) if keep], "validation_rmse": validation_rmse, "forward_rollout_rmse": forward_rmse, "forward_stability": {"diffusion_number": diffusion_number, "explicit_cfl_satisfied": diffusion_number <= 0.5}, "grid": {"time_step": dt, "x_step": dx, "y_step": dy, "z_step": dz, "field_shape": list(values.shape)}, "train_time_rows": int(len(train_rows)), "validation_time_rows": int(len(valid_rows)), "counterexamples": witnesses, "policy": "regular_grid_finite_difference_3d_pde_candidate; not_arbitrary_pde_solver_or_boundary_proof"}


__all__ = ["SCHEMA_VERSION", "PDEDiscoveryError", "discover_1d_pde", "discover_2d_pde", "discover_3d_pde"]
