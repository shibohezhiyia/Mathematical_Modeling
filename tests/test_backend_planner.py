from types import SimpleNamespace

import pytest

from core.backend_planner import BackendPlanError, plan_backend


def _spec(executor="adaptive_ode/v1", variables=10, points=1000, seconds=30):
    return SimpleNamespace(
        executor_key=executor,
        solver_family="test solver",
        max_variables=variables,
        max_evaluations=10000,
        timeout_seconds=seconds,
    )


def _node(**updates):
    value = {
        "mathematical_form": "initial_value_problem",
        "status": "executable",
        "execution_contract": {
            "state_variables": ["x"],
            "output_points": 100,
        },
    }
    value.update(updates)
    return value


def test_scale_aware_plan_selects_supervised_worker_without_granting_code_execution():
    result = plan_backend(_node(), _spec())
    assert result.status == "ready"
    public = result.public()
    assert public["route"] == "trusted_worker"
    assert public["resource_budget"]["process_isolated"] is True
    assert public["resource_budget"]["arbitrary_code_allowed"] is False


@pytest.mark.parametrize("executor", ["linear_ode/v1", "threshold_event/v1", "quadratic_program/v1"])
def test_registered_primitive_routes_are_also_supervised(executor):
    result = plan_backend(_node(), _spec(executor=executor))
    assert result.route == "trusted_worker"
    assert result.resource_budget["process_isolated"] is True


def test_unverified_node_and_oversized_contract_are_deferred():
    unverified = plan_backend(_node(status="deferred"), _spec())
    assert unverified.status == "deferred"
    assert unverified.reason == "mathematical_contract_not_verified"

    oversized = _node(execution_contract={"state_variables": ["x"], "output_points": 2_000_001})
    plan = plan_backend(oversized, _spec(), max_points=2_000_000)
    assert plan.status == "deferred"
    assert plan.reason == "contract_scale_exceeds_safe_plan"


def test_dynamics_contract_scale_counts_matrix_and_time_grid():
    node = _node(
        execution_contract={
            "state_variables": ["x", "y"],
            "matrix": [[1, 0], [0, 1]],
            "times": [0, 1, 2, 3],
        }
    )
    result = plan_backend(node, _spec(executor="linear_ode/v1"), max_points=3)
    assert result.status == "deferred"
    assert result.reason == "contract_scale_exceeds_safe_plan"


def test_missing_solver_is_not_converted_to_approximate_result():
    plan = plan_backend(_node(), None)
    assert plan.route == "defer"
    assert plan.executor_key is None
    assert plan.fidelity == "none"


def test_malformed_scale_or_matrix_is_deferred_not_coerced_to_zero():
    malformed_scale = plan_backend(_node(execution_contract={"output_points": "2000001"}), _spec())
    assert malformed_scale.status == "deferred"
    assert malformed_scale.reason == "contract_scale_exceeds_safe_plan"
    malformed_matrix = plan_backend(_node(execution_contract={"matrix": "not-a-matrix"}), _spec())
    assert malformed_matrix.status == "deferred"
    assert malformed_matrix.reason == "contract_scale_exceeds_safe_plan"


def test_backend_plan_carries_condition_and_event_strategy_without_claiming_proof():
    plan = plan_backend(_node(execution_contract={
        "state_variables": ["x"], "output_points": 10,
        "diagnostics": {"stiffness_ratio": 2e5, "event_count": 1,
                         "gradient_status": "unvalidated"},
    }), _spec())
    strategy = plan.resource_budget["numerical_strategy"]
    assert strategy["backend_strategy"] == "stiff_aware_integrator"
    assert strategy["derivative_strategy"] == "independent_difference_check"
    assert "certificate" in strategy["policy"]


@pytest.mark.parametrize("kwargs", [{"max_cells": 0}, {"max_points": 0}, {"max_cells": 50_000_001}])
def test_budget_parameters_are_bounded(kwargs):
    with pytest.raises(BackendPlanError):
        plan_backend(_node(), _spec(), **kwargs)
