import numpy as np
import pytest

from core.optimal_control import OptimalControlError, solve_linear_quadratic_control


def test_finite_horizon_lqr_returns_finite_dynamics_and_bounded_controls():
    result = solve_linear_quadratic_control([[1.0]], [[1.0]], [[1.0]], [[1.0]], [2.0], horizon=8,
                                            control_lower=[-0.5], control_upper=[0.5])
    assert result["status"] == "candidate"
    assert result["solution_type"] == "clipped_lqr_candidate"
    assert result["maximum_constraint_violation"] == pytest.approx(0.0)
    assert len(result["states"]) == 9
    assert np.isfinite(result["objective"])


def test_control_rejects_nonpositive_control_cost_and_invalid_bounds():
    with pytest.raises(OptimalControlError, match="cost_matrices"):
        solve_linear_quadratic_control([[1.0]], [[1.0]], [[1.0]], [[0.0]], [1.0], horizon=2)
    with pytest.raises(OptimalControlError, match="control_bounds"):
        solve_linear_quadratic_control([[1.0]], [[1.0]], [[1.0]], [[1.0]], [1.0], horizon=2,
                                       control_lower=[1.0], control_upper=[0.0])
