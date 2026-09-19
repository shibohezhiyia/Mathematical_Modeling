from core.symbolic_portfolio import (_observed_transition_ambiguity,
                                     run_same_split_portfolio_arm, run_validation_routed_portfolio)


def _solver(slope):
    def solve(payload):
        queries = payload["query_inputs"]
        return {"model": {"structure": "affine", "input_variables": ["x"],
                          "coefficients": [slope, 0.0]},
                "predictions": [slope * row[0] for row in queries],
                "execution_supervision": {"process_isolated": True,
                    "permission_isolated": True, "elapsed_seconds": 0.1,
                    "limits": {"memory_mb": 1024}, "memory_backend": "fixture"}}
    return solve


def test_training_only_router_selects_better_arm_and_executes_its_model():
    rows = [{"x": float(index), "response": float(2 * index)} for index in range(80)]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": [[3.5], [10.5]]}
    output = run_validation_routed_portfolio(
        payload, seed=7, current_solver=_solver(1.0), gplearn_solver=_solver(2.0),
    )
    assert output["predictions"] == [7.0, 21.0]
    assert output["model"]["portfolio_selection"]["selected_arm"] == "official_gplearn"
    assert output["model"]["portfolio_selection"]["training_row_count"] == 64
    assert output["model"]["portfolio_selection"]["validation_row_count"] == 16
    assert output["execution_supervision"]["child_runs"] == 2


def test_router_uses_a_stable_name_tie_break_without_test_answers():
    rows = [{"x": float(index), "response": float(index)} for index in range(80)]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": [[2.0]]}
    output = run_validation_routed_portfolio(
        payload, seed=9, current_solver=_solver(1.0), gplearn_solver=_solver(1.0),
    )
    assert output["model"]["portfolio_selection"]["selected_arm"] == "current_bounded_grammar"


def test_router_abstains_when_every_computable_arm_fails_validation():
    rows = [{"x": float(index), "response": float(2 * index)} for index in range(80)]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": [[2.0]]}
    output = run_validation_routed_portfolio(
        payload, seed=9, current_solver=_solver(0.1), gplearn_solver=_solver(-0.2),
    )
    assert output["status"] == "needs_input"
    assert output["result_grade"] == "abstain"
    assert output["reason"] == "portfolio_no_validated_candidate"
    assert "predictions" not in output


def test_same_split_arm_uses_the_identical_80_percent_training_partition():
    rows = [{"x": float(index), "response": float(2 * index)} for index in range(80)]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": [[3.0]]}
    seen = []
    def recording_solver(local_payload):
        seen.append([row["x"] for row in local_payload["attachments"][0]["rows"]])
        return _solver(2.0)(local_payload)
    first = run_same_split_portfolio_arm(
        payload, arm="current_bounded_grammar", seed=12, current_solver=recording_solver)
    second = run_same_split_portfolio_arm(
        payload, arm="official_gplearn", seed=12, gplearn_solver=recording_solver)
    assert seen[0] == seen[1]
    assert len(seen[0]) == 64
    assert first["predictions"] == second["predictions"] == [6.0]


def test_portfolio_reports_transition_identification_gap_without_claiming_structure_exclusion():
    import numpy as np

    calls = []
    x = np.linspace(-2.0, 2.0, 96)
    payload = {
        "attachments": [{"name": "switch", "format": "records", "rows": [
            {"threshold_input": float(value), "response": 2.0 if value >= 0.0 else -1.0}
            for value in x
        ]}],
        "query_inputs": [[0.75]],
    }
    def solver(_payload):
        calls.append(True)
        raise AssertionError("out-of-scope solver must not run")
    result = run_validation_routed_portfolio(
        payload, current_solver=solver, gplearn_solver=solver,
    )
    assert result["status"] == "needs_input"
    assert result["result_grade"] == "abstain"
    assert result["reason"] == "portfolio_transition_region_underobserved"
    assert result["recommended_action"] == "collect_observations_within_suggested_transition_interval"
    assert result["routing_evidence"]["hypotheses_not_distinguished"] == [
        "steep_smooth_transition", "discontinuous_or_quantized_transition",
    ]
    assert len(result["routing_evidence"]["suggested_observation_interval"]) == 2
    assert result["usage"]["numerical_solver_calls"] == 0
    assert calls == []


def test_discontinuity_gate_can_be_ablated_without_changing_solver_budget():
    import numpy as np

    calls = []
    x = np.linspace(-2.0, 2.0, 96)
    payload = {
        "attachments": [{"name": "switch", "format": "records", "rows": [
            {"x": float(value), "response": 2.0 if value >= 0.0 else -1.0}
            for value in x
        ]}], "query_inputs": [[0.75]],
    }
    def solver(local_payload):
        calls.append(len(local_payload["attachments"][0]["rows"]))
        return _solver(0.0)(local_payload)
    result = run_validation_routed_portfolio(
        payload, enable_discontinuity_gate=False,
        current_solver=solver, gplearn_solver=solver,
    )
    assert result["reason"] == "portfolio_no_validated_candidate"
    assert calls == [77, 77]


def test_saturated_smooth_tanh_is_described_as_underobserved_not_outside_grammar():
    import numpy as np

    x = np.linspace(-15.0, 15.0, 96)
    payload = {"attachments": [{"name": "smooth", "format": "records", "rows": [
        {"x": float(value), "response": float(np.tanh(5.0 * value))} for value in x
    ]}], "query_inputs": [[0.1]]}
    result = run_validation_routed_portfolio(payload)
    assert result["status"] == "needs_input"
    assert result["reason"] == "portfolio_transition_region_underobserved"
    assert result["routing_evidence"]["identification_status"] == "smooth_vs_discontinuous_not_distinguished"


def test_quantized_smooth_and_noisy_step_request_local_resampling():
    import numpy as np

    x = np.linspace(-8.0, 8.0, 96)
    quantized = np.round(np.tanh(x), 1)
    rng = np.random.default_rng(4)
    noisy_step = np.where(x >= 0.0, 2.0, -1.0) + rng.normal(0.0, 0.01, len(x))
    for name, y in (("quantized", quantized), ("noisy", noisy_step)):
        payload = {"attachments": [{"name": name, "format": "records", "rows": [
            {"x": float(value), "response": float(response)} for value, response in zip(x, y)
        ]}], "query_inputs": [[0.1]]}
        result = run_validation_routed_portfolio(payload)
        assert result["reason"] == "portfolio_transition_region_underobserved"
        assert result["usage"]["numerical_solver_calls"] == 0


def test_invalid_observation_is_not_misreported_as_transition_evidence():
    rows = [{"x": float(index), "response": float(index)} for index in range(80)]
    rows[20]["response"] = "unreadable"
    payload = {"attachments": [{"name": "invalid", "format": "records", "rows": rows}],
               "query_inputs": [[3.0]]}
    assert _observed_transition_ambiguity(payload) is None


def test_on_demand_router_skips_second_solver_when_current_passes():
    rows = [{"x": float(index), "response": float(2 * index)} for index in range(80)]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": [[3.0]]}
    calls = []
    def unused(_payload):
        calls.append("gplearn")
        raise AssertionError("second solver should not run")
    result = run_validation_routed_portfolio(
        payload, routing_policy="current_then_fallback",
        current_solver=_solver(2.0), gplearn_solver=unused,
    )
    assert result["predictions"] == [6.0]
    assert result["usage"]["numerical_solver_calls"] == 1
    assert result["routing_evidence"]["validation_metrics"]["official_gplearn"]["status"] == "not_run"
    assert calls == []


def test_on_demand_router_uses_second_solver_when_current_fails_validation():
    rows = [{"x": float(index), "response": float(2 * index)} for index in range(80)]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": [[3.0]]}
    result = run_validation_routed_portfolio(
        payload, routing_policy="current_then_fallback",
        current_solver=_solver(0.1), gplearn_solver=_solver(2.0),
    )
    assert result["predictions"] == [6.0]
    assert result["model"]["portfolio_selection"]["selected_arm"] == "official_gplearn"
    assert result["usage"]["numerical_solver_calls"] == 2


def test_on_demand_router_recovers_when_current_solver_cannot_produce_model():
    rows = [{"x": float(index), "response": float(2 * index)} for index in range(80)]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": [[3.0]]}
    result = run_validation_routed_portfolio(
        payload, routing_policy="current_then_fallback",
        current_solver=lambda _payload: {"status": "not_assessed", "reason": "worker_timeout"},
        gplearn_solver=_solver(2.0),
    )
    assert result["status"] == "completed"
    assert result["predictions"] == [6.0]
    assert result["model"]["portfolio_selection"]["selected_arm"] == "official_gplearn"
    assert result["usage"]["numerical_solver_calls"] == 2


def test_one_arm_budget_abstains_instead_of_accepting_failed_first_candidate():
    rows = [{"x": float(index), "response": float(2 * index)} for index in range(80)]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": [[3.0]]}
    calls = []
    def second(_payload):
        calls.append(True)
        return _solver(2.0)(_payload)
    result = run_validation_routed_portfolio(
        payload, solver_arm_budget=1, routing_policy="current_then_fallback",
        current_solver=_solver(0.1), gplearn_solver=second,
    )
    assert result["status"] == "needs_input"
    assert result["result_grade"] == "abstain"
    assert result["reason"] == "portfolio_second_arm_budget_unavailable"
    assert result["usage"]["numerical_solver_calls"] == 1
    assert "predictions" not in result
    assert calls == []


def test_one_arm_budget_can_accept_validated_first_candidate():
    rows = [{"x": float(index), "response": float(2 * index)} for index in range(80)]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": [[3.0]]}
    result = run_validation_routed_portfolio(
        payload, solver_arm_budget=1, routing_policy="current_then_fallback",
        current_solver=_solver(2.0), gplearn_solver=lambda _: (_ for _ in ()).throw(AssertionError()),
    )
    assert result["status"] == "completed"
    assert result["predictions"] == [6.0]
    assert result["usage"]["numerical_solver_calls"] == 1
