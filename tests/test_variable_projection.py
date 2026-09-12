import pytest

from core.variable_projection import VariableProjectionError, variable_projection_search


def test_variable_projection_eliminates_linear_block_and_records_rank():
    result = variable_projection_search(
        [0.0, 1.0, 2.0],
        lambda theta: [[1, theta], [1, 2 * theta], [1, 3 * theta]],
        [1, 2, 3],
    )
    assert result["status"] == "assessed"
    assert result["best"]["rank"] == 2
    assert result["best"]["loss"] == pytest.approx(0.0)


def test_variable_projection_keeps_rank_deficiency_visible():
    result = variable_projection_search([1], lambda _: [[1, 1], [1, 1]], [2, 2])
    assert result["status"] == "assessed"
    assert result["rank_deficient_attempts"] == 1


def test_variable_projection_rejects_unbounded_contracts():
    with pytest.raises(VariableProjectionError):
        variable_projection_search([], lambda _: [[1]], [1])
