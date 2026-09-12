import pytest

from core.solver_result_contract import SolverResultContractError, normalize_solver_result


def test_normalizer_preserves_missing_error_bound_as_unassessed():
    result = normalize_solver_result({"status": "executed", "solution": [1, 2], "residual_inf": 1e-9, "iterations": 4}, operation="linear_system")
    assert result["residual_status"] == "assessed"
    assert result["error_bound_status"] == "not_assessed"
    assert result["iterations"] == 4


def test_normalizer_rejects_invalid_status_and_negative_resource():
    with pytest.raises(SolverResultContractError):
        normalize_solver_result({"status": "success"}, operation="x")
    with pytest.raises(SolverResultContractError):
        normalize_solver_result({"status": "failed"}, operation="x", resource_usage={"seconds": -1})
