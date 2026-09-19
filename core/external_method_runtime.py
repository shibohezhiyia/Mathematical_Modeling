"""Safe execution adapters for the external-method plan.

This module is intentionally narrow: it dispatches only to local, typed
implementations already owned by this project.  It never imports a repository
named by a model, evaluates generated source, or treats a proposal as a
solved result.
"""

from __future__ import annotations

from typing import Any, Mapping

from .sindy_discovery import discover_sparse_dynamics, discover_weak_form_dynamics
from .structure_extensions import build_pde_library_contract, fit_ude_correction
from .pde_discovery import discover_1d_pde, discover_2d_pde, discover_3d_pde
from .ude_neural import fit_neural_ude, fit_joint_neural_ude, simulate_neural_ude_stiff
from .typed_symbolic_regression import fit_typed_symbolic_candidates, fit_typed_symbolic_expression, search_typed_symbolic_candidates
from .conclusion_certificate import ConclusionCertificateError, build_conclusion_certificate


class ExternalMethodRuntimeError(ValueError):
    """Raised for an invalid method execution request."""


_METHODS = {"sindy", "weak_sindy", "pde_find", "ude", "ude_neural", "ude_joint", "ude_stiff", "llm_sr"}
_COMMON = {"method", "payload"}


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalMethodRuntimeError(f"{name}_must_be_mapping")
    return value


def _result(method: str, status: str, **fields: Any) -> dict[str, Any]:
    return {
        "schema_version": "mathmodel.external-method-runtime/v1",
        "method": method,
        "status": status,
        "policy": "local_typed_adapter_only; result_requires_candidate_gate_and_independent_validation",
        **fields,
    }


def _readiness(*, units: str = "not_assessed", source: str = "not_assessed") -> dict[str, str]:
    """Return explicit execution checks; missing provenance never defaults to pass."""
    return {
        "type": "ready",
        "unit": units,
        "source": source,
        "resource": "ready",
        "security": "ready",
    }


def _certificate(method: str, result: Mapping[str, Any], boundary: Any, assumptions: Any) -> dict[str, Any]:
    """Build a report certificate when the caller supplies an applicability boundary."""
    if boundary is None:
        return {"status": "not_assessed", "reason": "applicability_boundary_required",
                "policy": "no_boundary_no_publishable_conclusion"}
    if assumptions is None:
        assumptions = ()
    if not isinstance(assumptions, (list, tuple)):
        return {"status": "blocked", "reason": "assumptions_must_be_sequence"}
    if method == "sindy":
        values = result.get("validation_derivative_rmse", ())
    elif method == "weak_sindy":
        values = result.get("validation_weak_form_rmse", ())
    elif method in ("ude", "ude_neural", "ude_joint"):
        values = [result.get("holdout_rmse")]
    elif method == "ude_stiff":
        # The stiff simulator returns a trajectory; an external certificate
        # can only be built when the caller supplies an independently checked
        # residual field, so leave the value absent rather than inventing one.
        values = [result.get("trajectory_rmse")]
    elif method in ("pde_find", "llm_sr"):
        values = ([result.get("validation_rmse"), result.get("boundary_rmse"), result.get("forward_rollout_rmse")] if method == "pde_find"
                  else [result.get("holdout_rmse")])
    else:
        values = ()
    try:
        residuals = [{"id": f"{method}_validation_{index}", "value": value}
                     for index, value in enumerate(values)]
        return build_conclusion_certificate(
            applicability_boundary=boundary,
            residuals=residuals,
            constraint_checks=[],
            counterexamples=result.get("counterexamples", ()),
            assumptions=assumptions,
        )
    except (ConclusionCertificateError, TypeError, ValueError) as exc:
        return {"status": "blocked", "reason": type(exc).__name__,
                "policy": "certificate_build_failed_no_publishable_conclusion"}


def execute_external_method(method: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one allow-listed local method adapter.

    Errors from a numerical adapter are returned as a structured ``rejected``
    result so a batch can retain its failure denominator.  ``llm_sr`` is
    deliberately never executable here; its output must enter the structured
    proposal/compiler path first.
    """
    if not isinstance(method, str) or method not in _METHODS:
        raise ExternalMethodRuntimeError("method_invalid")
    data = _mapping(payload, "payload")
    boundary = data.get("applicability_boundary")
    assumptions = data.get("assumptions", ())
    try:
        if method == "sindy":
            allowed = {"times", "states", "state_names", "polynomial_degree",
                       "sparsity_threshold", "max_terms_per_equation", "validation_fraction",
                       "residual_tolerance", "state_dimensions", "time_dimension",
                       "applicability_boundary", "assumptions"}
            if set(data) - allowed:
                raise ExternalMethodRuntimeError("sindy_payload_contains_unknown_fields")
            numerical_data = {key: value for key, value in data.items()
                              if key not in {"applicability_boundary", "assumptions"}}
            result = discover_sparse_dynamics(**numerical_data)
            unit_status = "ready" if data.get("state_dimensions") is not None else "not_assessed"
            return _result(method, "executed", result=result,
                           execution_readiness=_readiness(units=unit_status),
                           conclusion_certificate=_certificate(method, result, boundary, assumptions))
        if method == "weak_sindy":
            allowed = {"times", "states", "state_names", "polynomial_degree",
                       "sparsity_threshold", "max_terms_per_equation", "validation_fraction",
                       "residual_tolerance", "window_count", "applicability_boundary", "assumptions"}
            if set(data) - allowed:
                raise ExternalMethodRuntimeError("weak_sindy_payload_contains_unknown_fields")
            numerical_data = {key: value for key, value in data.items()
                              if key not in {"applicability_boundary", "assumptions"}}
            result = discover_weak_form_dynamics(**numerical_data)
            return _result(method, "executed", result=result,
                           execution_readiness=_readiness(),
                           conclusion_certificate=_certificate(method, result, boundary, assumptions))
        if method == "pde_find":
            allowed = {"field_shape", "coordinate_names", "derivative_orders", "max_terms", "sparsity_threshold",
                       "times", "coordinates", "field", "validation_fraction", "ridge",
                       "residual_tolerance", "boundary_tolerance", "include_advection", "field_mask",
                       "noise_scale", "field_dimensions", "coordinate_dimensions", "boundary_values",
                       "x_coordinates", "y_coordinates", "z_coordinates", "max_rows"}
            if set(data) - allowed:
                raise ExternalMethodRuntimeError("pde_payload_contains_unknown_fields")
            if all(key in data for key in ("times", "x_coordinates", "y_coordinates", "field")) and "z_coordinates" not in data:
                numerical_data = {key: value for key, value in data.items()
                                  if key not in {"applicability_boundary", "assumptions", "field_shape", "coordinate_names", "derivative_orders", "coordinates", "boundary_tolerance", "field_mask", "noise_scale", "field_dimensions", "coordinate_dimensions"}}
                result = discover_2d_pde(**numerical_data)
                return _result(method, "executed", result=result,
                               execution_readiness=_readiness(),
                               conclusion_certificate=_certificate(method, result, boundary, assumptions))
            if all(key in data for key in ("times", "x_coordinates", "y_coordinates", "z_coordinates", "field")):
                numerical_data = {key: value for key, value in data.items()
                                  if key not in {"applicability_boundary", "assumptions", "field_shape", "coordinate_names", "derivative_orders", "coordinates", "boundary_tolerance", "field_mask", "noise_scale", "field_dimensions", "coordinate_dimensions"}}
                result = discover_3d_pde(**numerical_data)
                return _result(method, "executed", result=result,
                               execution_readiness=_readiness(),
                               conclusion_certificate=_certificate(method, result, boundary, assumptions))
            if all(key in data for key in ("times", "coordinates", "field")):
                numerical_data = {key: value for key, value in data.items()
                                  if key not in {"applicability_boundary", "assumptions", "field_shape", "coordinate_names", "derivative_orders", "max_terms", "x_coordinates", "y_coordinates", "max_rows"}}
                result = discover_1d_pde(**numerical_data)
                return _result(method, "executed", result=result,
                               execution_readiness=_readiness(),
                               conclusion_certificate=_certificate(method, result, boundary, assumptions))
            allowed_plan = {"field_shape", "coordinate_names", "derivative_orders", "max_terms"}
            if set(data) - allowed_plan:
                raise ExternalMethodRuntimeError("pde_plan_requires_field_or_contract_fields")
            result = build_pde_library_contract(**dict(data))
            return _result(method, "planned", result=result)
        if method == "ude":
            allowed = {"target_rhs", "known_rhs", "correction_features", "validation_fraction",
                       "ridge", "max_rows", "max_condition_number", "applicability_boundary", "assumptions"}
            if set(data) - allowed:
                raise ExternalMethodRuntimeError("ude_payload_contains_unknown_fields")
            numerical_data = {key: value for key, value in data.items()
                              if key not in {"applicability_boundary", "assumptions"}}
            result = fit_ude_correction(**numerical_data)
            return _result(method, "executed", result=result,
                           execution_readiness=_readiness(),
                           conclusion_certificate=_certificate(method, result, boundary, assumptions))
        if method == "ude_neural":
            allowed = {"target_rhs", "known_rhs", "correction_features", "validation_fraction", "epochs",
                       "hidden_dim", "restarts", "learning_rate", "max_rows", "random_state",
                       "feature_domains", "target_dimensions", "known_rhs_dimensions",
                       "applicability_boundary", "assumptions"}
            if set(data) - allowed:
                raise ExternalMethodRuntimeError("ude_neural_payload_contains_unknown_fields")
            numerical_data = {key: value for key, value in data.items()
                              if key not in {"applicability_boundary", "assumptions"}}
            result = fit_neural_ude(**numerical_data)
            return _result(method, "executed", result=result,
                           execution_readiness=_readiness(),
                           conclusion_certificate=_certificate(method, result, boundary, assumptions))
        if method == "ude_joint":
            allowed = {"times", "observations", "known_matrix", "validation_fraction", "epochs",
                       "hidden_dim", "restarts", "learning_rate", "max_rows", "max_abs_state",
                       "random_state", "applicability_boundary", "assumptions"}
            if set(data) - allowed:
                raise ExternalMethodRuntimeError("ude_joint_payload_contains_unknown_fields")
            numerical_data = {key: value for key, value in data.items()
                              if key not in {"applicability_boundary", "assumptions"}}
            result = fit_joint_neural_ude(**numerical_data)
            return _result(method, "executed", result=result,
                           execution_readiness=_readiness(),
                           conclusion_certificate=_certificate(method, result, boundary, assumptions))
        if method == "ude_stiff":
            allowed = {"times", "initial_state", "linear_matrix", "fit_results", "method", "rtol", "atol",
                       "max_step", "max_abs_state", "applicability_boundary", "assumptions"}
            if set(data) - allowed:
                raise ExternalMethodRuntimeError("ude_stiff_payload_contains_unknown_fields")
            numerical_data = {key: value for key, value in data.items()
                              if key not in {"applicability_boundary", "assumptions"}}
            result = simulate_neural_ude_stiff(**numerical_data)
            return _result(method, "executed", result=result,
                           execution_readiness=_readiness(),
                           conclusion_certificate=_certificate(method, result, boundary, assumptions))
        if method == "llm_sr":
            # A model proposal may enter execution only as a typed JSON tree;
            # source strings and repository paths remain proposal-only.
            if "expression" not in data and "candidates" not in data:
                return _result(method, "proposal_only", reason="llm_generated_code_is_not_an_execution_input")
            allowed = {"target", "features", "expression", "parameter_bounds", "candidates", "validation_fraction",
                       "max_nfev", "random_state", "feature_dimensions", "target_dimensions", "parameter_dimensions",
                       "allowed_operators", "allowed_variables", "search",
                       "applicability_boundary", "assumptions"}
            if set(data) - allowed:
                raise ExternalMethodRuntimeError("llm_sr_payload_contains_unknown_fields")
            if "expression" in data and "candidates" in data:
                raise ExternalMethodRuntimeError("llm_sr_expression_and_candidates_are_mutually_exclusive")
            numerical_data = {key: value for key, value in data.items()
                              if key not in {"applicability_boundary", "assumptions"}}
            search_options = numerical_data.pop("search", None)
            if search_options is not None:
                if "candidates" not in numerical_data or not isinstance(search_options, Mapping):
                    raise ExternalMethodRuntimeError("llm_sr_search_requires_candidate_pool")
                allowed_search = {"generations", "max_candidates", "beam_width"}
                if set(search_options) - allowed_search:
                    raise ExternalMethodRuntimeError("llm_sr_search_fields_invalid")
                numerical_data.update(dict(search_options))
                result = search_typed_symbolic_candidates(**numerical_data)
            else:
                result = (fit_typed_symbolic_candidates(**numerical_data) if "candidates" in numerical_data
                          else fit_typed_symbolic_expression(**numerical_data))
            return _result(method, "executed", result=result,
                           execution_readiness=_readiness(),
                           conclusion_certificate=_certificate(method, result, boundary, assumptions))
    except ExternalMethodRuntimeError:
        raise
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        return _result(method, "rejected", reason=type(exc).__name__)


__all__ = ["ExternalMethodRuntimeError", "execute_external_method"]
