"""Typed adapters for optional numerical backends.

Only structured contracts enter this module.  No backend receives generated
source, callbacks, ``eval`` or an unbounded solver configuration.  Missing
packages and missing external solvers are explicit ``unavailable`` results.
"""
from __future__ import annotations

import importlib.util
import math
from typing import Any, Mapping

from .differentiable_ir import evaluate_differentiable_ir
from .sympy_backend import compile_sympy_expression
from .universal_math_solvers import UniversalSolverRegistry


class OptionalBackendRuntimeError(ValueError):
    pass


def _missing(module: str, reason: str) -> dict[str, Any]:
    return {"schema_version": "mathmodel.optional-backend-runtime/v1", "status": "unavailable",
            "backend": module, "reason": reason, "policy": "optional_dependency_or_solver_missing"}


def _module_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _finite_vector(values: Any, name: str) -> list[float]:
    if not isinstance(values, (list, tuple)) or not values or len(values) > 1000:
        raise OptionalBackendRuntimeError(f"{name}_must_be_bounded_vector")
    result = [float(value) for value in values]
    if not all(math.isfinite(value) for value in result):
        raise OptionalBackendRuntimeError(f"{name}_must_be_finite")
    return result


def _cvxpy_lp(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import cvxpy as cp
    except (ImportError, ModuleNotFoundError):
        return _missing("cvxpy", "cvxpy_not_installed")
    variables = payload.get("variables")
    if not isinstance(variables, list) or not variables or len(variables) > 256 or any(not isinstance(item, str) or not item.strip() for item in variables):
        raise OptionalBackendRuntimeError("variables_invalid")
    c = _finite_vector(payload.get("objective_coefficients"), "objective_coefficients")
    if len(c) != len(variables):
        raise OptionalBackendRuntimeError("objective_dimension_mismatch")
    direction = payload.get("direction", "minimize")
    if direction not in {"minimize", "maximize"}:
        raise OptionalBackendRuntimeError("direction_invalid")
    x = cp.Variable(len(variables))
    constraints = []
    bounds = payload.get("bounds", [[0.0, None] for _ in variables])
    if not isinstance(bounds, list) or len(bounds) != len(variables):
        raise OptionalBackendRuntimeError("bounds_dimension_mismatch")
    for value, bound in zip(x, bounds):
        if not isinstance(bound, (list, tuple)) or len(bound) != 2:
            raise OptionalBackendRuntimeError("bound_invalid")
        if bound[0] is not None: constraints.append(value >= float(bound[0]))
        if bound[1] is not None: constraints.append(value <= float(bound[1]))
    for matrix_name, vector_name, relation in (("A_ub", "b_ub", "le"), ("A_eq", "b_eq", "eq")):
        matrix = payload.get(matrix_name, [])
        rhs = payload.get(vector_name, [])
        if not isinstance(matrix, list) or not isinstance(rhs, list) or len(matrix) != len(rhs):
            raise OptionalBackendRuntimeError(f"{matrix_name}_dimension_mismatch")
        for row, target in zip(matrix, rhs):
            values = _finite_vector(row, matrix_name)
            if len(values) != len(variables) or not math.isfinite(float(target)):
                raise OptionalBackendRuntimeError(f"{matrix_name}_row_invalid")
            expression = sum(values[index] * x[index] for index in range(len(values)))
            constraints.append(expression <= float(target) if relation == "le" else expression == float(target))
    objective = cp.Maximize(c @ x) if direction == "maximize" else cp.Minimize(c @ x)
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=payload.get("solver", None), verbose=False)
    except Exception as exc:
        return {"schema_version": "mathmodel.optional-backend-runtime/v1", "status": "error", "backend": "cvxpy", "reason": type(exc).__name__}
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE} or x.value is None:
        return {"schema_version": "mathmodel.optional-backend-runtime/v1", "status": "incomplete", "backend": "cvxpy", "solver_status": str(problem.status)}
    solution = [float(value) for value in x.value]
    return {"schema_version": "mathmodel.optional-backend-runtime/v1", "status": "executed", "backend": "cvxpy",
            "solver_status": str(problem.status), "objective_value": float(problem.value),
            "solution": dict(zip(variables, solution)),
            "policy": "structured_linear_program_only; numerical_result_requires_independent_recheck"}


def execute_optional_backend(backend: str, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one explicitly typed operation through an optional backend."""
    if not isinstance(backend, str) or backend not in {"scipy", "sympy", "cvxpy", "pyomo", "jax"}:
        raise OptionalBackendRuntimeError("backend_invalid")
    if not isinstance(operation, str) or not operation.strip() or not isinstance(payload, Mapping):
        raise OptionalBackendRuntimeError("operation_and_payload_required")
    operation = operation.strip()
    if backend == "scipy":
        if not _module_available("scipy"):
            return _missing("scipy", "scipy_not_installed")
        registry = UniversalSolverRegistry()
        if not registry.has(operation):
            raise OptionalBackendRuntimeError("scipy_operation_not_registered")
        return {**registry.execute(operation, payload), "backend_family": "scipy"}
    if backend == "sympy":
        if operation != "compile_expression":
            raise OptionalBackendRuntimeError("sympy_operation_not_allowed")
        expression, symbols = payload.get("expression"), payload.get("symbols")
        if not isinstance(expression, str) or not isinstance(symbols, (list, tuple)):
            raise OptionalBackendRuntimeError("sympy_expression_contract_invalid")
        return {**compile_sympy_expression(expression, symbols), "backend_family": "sympy"}
    if backend == "jax":
        if operation != "differentiable_ir":
            raise OptionalBackendRuntimeError("jax_operation_not_allowed")
        return {**evaluate_differentiable_ir(payload.get("node"), payload.get("values", {}), backend="jax"), "backend_family": "jax"}
    if backend == "cvxpy":
        if operation != "linear_program":
            raise OptionalBackendRuntimeError("cvxpy_operation_not_allowed")
        return _cvxpy_lp(payload)
    if not _module_available("pyomo"):
        return _missing("pyomo", "pyomo_not_installed")
    return _missing("pyomo", "no_bounded_solver_adapter")


__all__ = ["OptionalBackendRuntimeError", "execute_optional_backend"]
