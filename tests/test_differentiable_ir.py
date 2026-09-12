import pytest

from core.differentiable_ir import DifferentiableIRError, evaluate_differentiable_ir, validate_differentiable_ir


def _model():
    return {"op": "add", "args": [
        {"op": "pow", "args": [{"op": "variable", "name": "x"}, {"op": "constant", "value": 2}]},
        {"op": "mul", "args": [{"op": "constant", "value": 3}, {"op": "variable", "name": "y"}]},
    ]}


def test_typed_ir_has_bounded_finite_difference_gradient():
    result = evaluate_differentiable_ir(_model(), {"x": 2.0, "y": 4.0}, backend="finite_difference")
    assert result["value"] == pytest.approx(16.0)
    assert result["gradient"] == pytest.approx([4.0, 3.0], rel=1e-4)


def test_ir_rejects_unknown_operations_and_undeclared_variables():
    with pytest.raises(DifferentiableIRError, match="invalid_ir_node"):
        validate_differentiable_ir({"op": "eval", "args": []})
    with pytest.raises(DifferentiableIRError, match="undeclared_variable"):
        validate_differentiable_ir({"op": "variable", "name": "x"}, variables=["y"])


def test_jax_route_is_explicitly_unavailable_or_executes_without_changing_policy():
    result = evaluate_differentiable_ir(_model(), {"x": 1.0, "y": 2.0}, backend="jax")
    assert result["backend"] == "jax"
    assert result["status"] in {"executed", "unavailable"}
