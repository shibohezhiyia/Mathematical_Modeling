import pytest

from core.optional_backend_runtime import execute_optional_backend


def test_scipy_adapter_uses_registered_typed_linear_program():
    result = execute_optional_backend("scipy", "linear_program/v1", {
        "variables": ["x"], "objective_coefficients": [2.0], "direction": "maximize",
        "A_ub": [[1.0]], "b_ub": [3.0], "A_eq": [], "b_eq": [], "bounds": [[0.0, None]],
    })
    assert result["status"] == "executed"
    assert result["solution"]["x"] == pytest.approx(3.0)
    assert result["backend_family"] == "scipy"


def test_sympy_adapter_rejects_untrusted_expression():
    with pytest.raises(ValueError):
        execute_optional_backend("sympy", "compile_expression", {"expression": "__import__('os')", "symbols": ["x"]})


def test_jax_and_pyomo_are_explicit_when_unavailable_or_bounded():
    result = execute_optional_backend("jax", "differentiable_ir", {
        "node": {"op": "mul", "args": [{"op": "variable", "name": "x"}, {"op": "constant", "value": 2}]},
        "values": {"x": 3.0},
    })
    assert result["status"] in {"executed", "unavailable"}
    pyomo = execute_optional_backend("pyomo", "linear_program", {})
    assert pyomo["status"] == "unavailable"
