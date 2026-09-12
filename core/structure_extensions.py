"""Optional structure-discovery contracts for interaction graphs, UDE, and PDE.

These routines provide bounded, non-causal screening and typed contracts.  A
screened edge, neural correction term, or derivative library is a candidate
for later validation, never a claimed causal relation or discovered law.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


class StructureExtensionError(ValueError):
    pass


def screen_interactions(
    values: Sequence[Sequence[float]],
    variable_names: Sequence[str],
    *,
    threshold: float = 0.25,
    max_edges: int = 128,
) -> dict[str, Any]:
    """Screen pairwise linear interactions without interpreting them causally."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[0] < 3 or array.shape[1] < 2:
        raise StructureExtensionError("interaction_matrix_too_small")
    if not isinstance(variable_names, Sequence) or len(variable_names) != array.shape[1] or any(not isinstance(item, str) or not item.strip() for item in variable_names):
        raise StructureExtensionError("variable_names_invalid")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 <= float(threshold) <= 1:
        raise StructureExtensionError("invalid_interaction_threshold")
    if type(max_edges) is not int or not 1 <= max_edges <= 10_000:
        raise StructureExtensionError("invalid_interaction_budget")
    if not np.all(np.isfinite(array)):
        raise StructureExtensionError("interaction_values_must_be_finite")
    correlations = np.corrcoef(array, rowvar=False)
    edges = []
    for left in range(array.shape[1]):
        for right in range(left + 1, array.shape[1]):
            score = float(abs(correlations[left, right])) if math.isfinite(float(correlations[left, right])) else 0.0
            if score >= float(threshold):
                edges.append({"source": variable_names[left], "target": variable_names[right], "score": score})
    edges.sort(key=lambda item: (-item["score"], item["source"], item["target"]))
    return {
        "schema_version": "mathmodel.interaction-screen/v1",
        "status": "assessed",
        "edges": edges[:max_edges],
        "candidate_count": len(edges),
        "truncated": len(edges) > max_edges,
        "threshold": float(threshold),
        "policy": "screened_interactions_are_not_causal_edges; independent direction_and_mechanism_checks_required",
    }


def build_ude_contract(
    state_variables: Sequence[str],
    known_rhs: Sequence[str],
    *,
    correction_basis: Sequence[str] = ("linear", "quadratic"),
    max_parameters: int = 64,
) -> dict[str, Any]:
    """Create a typed Universal Differential Equation correction contract."""
    if not isinstance(state_variables, Sequence) or isinstance(state_variables, (str, bytes)) or not state_variables:
        raise StructureExtensionError("state_variables_required")
    if not isinstance(known_rhs, Sequence) or isinstance(known_rhs, (str, bytes)) or len(known_rhs) != len(state_variables):
        raise StructureExtensionError("known_rhs_must_match_state_variables")
    if not isinstance(correction_basis, Sequence) or isinstance(correction_basis, (str, bytes)) or not correction_basis:
        raise StructureExtensionError("correction_basis_required")
    if type(max_parameters) is not int or not 1 <= max_parameters <= 10_000:
        raise StructureExtensionError("invalid_parameter_budget")
    variables = [str(item).strip() for item in state_variables]
    rhs = [str(item).strip() for item in known_rhs]
    basis = [str(item).strip() for item in correction_basis]
    if any(not item for item in (*variables, *rhs, *basis)) or len(set(variables)) != len(variables):
        raise StructureExtensionError("ude_contract_fields_invalid")
    return {
        "schema_version": "mathmodel.ude-contract/v1",
        "status": "typed_not_fitted",
        "state_variables": variables,
        "known_rhs": rhs,
        "correction_basis": basis,
        "max_parameters": max_parameters,
        "equations": [f"d{variable}/dt = {formula} + correction({variable})" for variable, formula in zip(variables, rhs)],
        "policy": "neural_correction_requires_holdout_residual_and_stability_checks; no automatic physical interpretation",
    }


def fit_ude_correction(
    target_rhs: Sequence[float], known_rhs: Sequence[float], correction_features: Sequence[Sequence[float]],
    *, validation_fraction: float = 0.25, ridge: float = 1e-6, max_rows: int = 200_000,
    max_condition_number: float = 1e12,
) -> dict[str, Any]:
    """Fit a bounded linear correction basis and audit a temporal holdout.

    This is the dependency-light numerical core of a UDE contract.  It does
    not pretend that a fitted correction is a neural network or a physical
    law; a later JAX/torch adapter may replace the basis while preserving the
    same train/holdout and stability gates.
    """
    target = np.asarray(target_rhs, dtype=float).reshape(-1)
    known = np.asarray(known_rhs, dtype=float).reshape(-1)
    features = np.asarray(correction_features, dtype=float)
    if target.ndim != 1 or known.shape != target.shape or features.ndim != 2 or features.shape[0] != target.size:
        raise StructureExtensionError("ude_fit_shapes_invalid")
    if type(max_rows) is not int or not 8 <= max_rows <= 1_000_000:
        raise StructureExtensionError("ude_max_rows_invalid")
    if not 8 <= target.size <= max_rows or features.shape[1] < 1 or features.shape[1] > 256:
        raise StructureExtensionError("ude_fit_size_invalid")
    if not np.isfinite(target).all() or not np.isfinite(known).all() or not np.isfinite(features).all():
        raise StructureExtensionError("ude_fit_values_must_be_finite")
    if type(validation_fraction) not in (int, float) or not 0.05 <= float(validation_fraction) <= 0.4:
        raise StructureExtensionError("ude_validation_fraction_invalid")
    if type(ridge) not in (int, float) or not math.isfinite(float(ridge)) or not 0 < float(ridge) <= 1e6:
        raise StructureExtensionError("ude_ridge_invalid")
    if (type(max_condition_number) not in (int, float) or isinstance(max_condition_number, bool) or
            not math.isfinite(float(max_condition_number)) or not 1 < float(max_condition_number) <= 1e16):
        raise StructureExtensionError("ude_condition_limit_invalid")
    split = int(np.floor(target.size * (1.0 - float(validation_fraction))))
    if split < max(4, features.shape[1]) or target.size - split < 2:
        raise StructureExtensionError("ude_holdout_too_small")
    x_train = features[:split]
    residual = target[:split] - known[:split]
    gram = x_train.T @ x_train + float(ridge) * np.eye(features.shape[1])
    condition_number = float(np.linalg.cond(gram))
    if not math.isfinite(condition_number) or condition_number > float(max_condition_number):
        raise StructureExtensionError("ude_design_ill_conditioned")
    try:
        coefficients = np.linalg.solve(gram, x_train.T @ residual)
    except np.linalg.LinAlgError as exc:
        raise StructureExtensionError("ude_fit_solve_failed") from exc
    correction = features @ coefficients
    prediction = known + correction
    if not np.isfinite(coefficients).all() or not np.isfinite(prediction).all():
        raise StructureExtensionError("ude_fit_nonfinite")
    train_rmse = float(np.sqrt(np.mean((prediction[:split] - target[:split]) ** 2)))
    holdout_rmse = float(np.sqrt(np.mean((prediction[split:] - target[split:]) ** 2)))
    perturbation = np.clip(features[split:] * 1.01, -1e12, 1e12)
    perturbed = known[split:] + perturbation @ coefficients
    stability_ratio = float(np.max(np.abs(perturbed - prediction[split:])) / max(1.0, np.max(np.abs(prediction[split:]))))
    return {
        "schema_version": "mathmodel.ude-fit/v1", "status": "fitted",
        "coefficients": coefficients.round(12).tolist(), "train_rows": split,
        "holdout_rows": target.size - split, "train_rmse": train_rmse,
        "holdout_rmse": holdout_rmse, "stability_ratio_under_1pct_feature_perturbation": stability_ratio,
        "design_condition_number": condition_number,
        "stability_status": "not_assessed" if not math.isfinite(stability_ratio) else "assessed",
        "policy": "bounded_basis_correction_with_temporal_holdout; not_neural_or_physical_law_proof",
    }


def build_pde_library_contract(
    field_shape: Sequence[int],
    coordinate_names: Sequence[str],
    *,
    derivative_orders: Sequence[int] = (1, 2),
    max_terms: int = 64,
) -> dict[str, Any]:
    """Describe a bounded PDE derivative library without fitting coefficients."""
    if not isinstance(field_shape, Sequence) or isinstance(field_shape, (str, bytes)) or len(field_shape) < 2:
        raise StructureExtensionError("field_shape_requires_grid")
    if any(type(size) is not int or size < 3 or size > 10_000 for size in field_shape):
        raise StructureExtensionError("field_shape_invalid")
    if not isinstance(coordinate_names, Sequence) or isinstance(coordinate_names, (str, bytes)) or len(coordinate_names) != len(field_shape):
        raise StructureExtensionError("coordinate_names_mismatch")
    if any(not isinstance(name, str) or not name.strip() for name in coordinate_names):
        raise StructureExtensionError("coordinate_names_invalid")
    orders = tuple(derivative_orders)
    if not orders or any(type(order) is not int or not 1 <= order <= 4 for order in orders):
        raise StructureExtensionError("derivative_orders_invalid")
    if type(max_terms) is not int or not 1 <= max_terms <= 10_000:
        raise StructureExtensionError("invalid_pde_term_budget")
    terms = ["field"]
    for coordinate in coordinate_names:
        for order in orders:
            terms.append(f"d{order}_{coordinate}(field)")
    return {
        "schema_version": "mathmodel.pde-library/v1",
        "status": "typed_not_fitted",
        "field_shape": list(field_shape),
        "coordinate_names": list(coordinate_names),
        "derivative_orders": list(orders),
        "terms": terms[:max_terms],
        "truncated": len(terms) > max_terms,
        "policy": "derivative_library_is_a_candidate_space; boundary_conditions_units_and_holdout_validation_required",
    }


__all__ = ["StructureExtensionError", "screen_interactions", "build_ude_contract", "fit_ude_correction",
           "build_pde_library_contract"]
