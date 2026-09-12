import pytest

from core.numerical_strategy import NumericalStrategyError, select_numerical_strategy


def test_numerical_strategy_prioritises_stiffness_events_and_gradient_gate():
    result = select_numerical_strategy({
        "condition_number": 1e12,
        "stiffness_ratio": 1e6,
        "event_count": 2,
        "gradient_status": "unvalidated",
    })
    assert result["backend_strategy"] == "stiff_aware_integrator"
    assert result["derivative_strategy"] == "independent_difference_check"
    assert result["flags"]["ill_conditioned"] is True


def test_numerical_strategy_does_not_invent_missing_diagnostics():
    result = select_numerical_strategy()
    assert result["status"] == "not_assessed"
    assert result["backend_strategy"] == "standard_bounded_solver"
    with pytest.raises(NumericalStrategyError):
        select_numerical_strategy({"condition_number": -1})
