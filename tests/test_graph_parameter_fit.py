"""Restarts repair numeric starts, not the independently frozen judge."""
import math

import numpy as np
import pytest

from core.graph_evaluator import ScalarGraphProgram
from core.graph_experiments import SearchExperiment
from core.graph_parameter_fit import _starts, fit_training_parameters
from core.model_hypotheses import HypothesisIR
from core.solver_runtime import EvaluationCounter, SolverLimits, SolverProcessRunner
from tests.test_graph_search import cases, direct, experiment, fixture_graph, node, request


def nonlinear(kind):
    contract, seed = fixture_graph("multiply", parameter=True)
    payload = seed.payload()
    if kind == "square":
        payload["nodes"][-1:] = [node("a2", "multiply", ["a", "a"]), node("y", "multiply", ["a2", "x"])]
        fn, bounds = lambda x: 4*x, [-5, 5, 0]
    else:
        payload["nodes"][-1:] = [node("ax", "multiply", ["a", "x"]), node("y", "log", ["ax"])]
        fn, bounds = lambda x: math.log(2*x), [-5, 5, 0]
    graph = HypothesisIR.from_payload(payload, contract)
    exp = SearchExperiment.create(contract, graph, domain={"x": [0.1, 5]}, probe_count=4,
                     training_cases=cases([0.5, 1, 2], fn, "train"),
                     search_cases=cases([0.2, 1.5, 3], fn, "search"), parameter_bounds={"a": bounds})
    return contract, graph, exp


@pytest.mark.parametrize("kind", ["square", "log"])
def test_recovers_stationary_or_invalid_initial_point(kind):
    contract, graph, exp = nonlinear(kind)
    result = direct(contract, graph, exp)
    assert result["status"] == "eligible_for_confirmation"
    assert abs(result["parameters"]["a"]) == pytest.approx(2)
    assert result["fit"]["selected_attempt"] > 0
    assert result["fit"]["selection_data"] == "training_only"
    assert result["fit"]["global_optimum_proven"] is False
    assert len(result["fit"]["attempts"]) <= 3
    if kind == "log":
        assert result["fit"]["attempts"][0]["status"] == "numeric_failure"
    assert not result["witnesses"]


def test_search_labels_and_tolerances_do_not_choose_restart():
    contract, graph, exp = nonlinear("square")
    original = direct(contract, graph, exp)
    changed = exp.public()
    for case in changed["search_cases"]:
        case["expected"]["y"] += 900
    changed.update(absolute_tolerance=1e-12, relative_tolerance=0)
    other = direct(contract, graph, SearchExperiment.from_payload(changed, contract))
    assert original["parameters"] == other["parameters"]
    assert original["fit"] == other["fit"]
    assert other["status"] == "rejected_by_checks"


def test_easy_affine_case_uses_structurally_proven_linear_fit():
    contract, graph = fixture_graph("multiply", parameter=True)
    exp = experiment(contract, graph, lambda x: 2*x, parameter_bounds={"a": [0, 5, 1]})
    result = direct(contract, graph, exp)
    assert len(result["fit"]["attempts"]) == 1
    assert result["fit"]["status"] == "bounded_linear_fit_completed"
    assert result["fit"]["convex_subproblem"] is True
    assert result["fit"]["eliminated_nonlinear_search_dimensions"] == 1
    assert result["fit"]["jacobian_rank"] == 1
    assert result["fit"]["attempts"][0]["structurally_affine"] is True
    assert result["fit"]["attempts"][0]["evaluations_used"] == 2


def test_bounded_linear_solution_does_not_bypass_search_checks():
    contract, graph = fixture_graph("multiply", parameter=True)
    exp = experiment(contract, graph, lambda x: 9*x, parameter_bounds={"a": [0, 5, 1]})
    result = direct(contract, graph, exp)
    assert result["parameters"]["a"] == pytest.approx(5)
    assert result["fit"]["status"] == "bounded_linear_fit_completed"
    assert result["status"] == "rejected_by_checks"
    assert result["witnesses"]


def test_structural_affine_detector_declines_parameter_product():
    from core.graph_parameter_fit import _all_parameters_affine
    contract, graph, exp = nonlinear("square")
    program = ScalarGraphProgram(graph, EvaluationCounter(100))
    assert _all_parameters_affine(program) is False
    result = direct(contract, graph, exp)
    assert result["fit"]["status"] == "local_fit_completed"


def test_affine_solver_handles_narrow_representable_large_bounds():
    contract, graph = fixture_graph("multiply", parameter=True)
    lo, hi = 1e10, 1e10 + 0.001
    expected = lo + 0.0005
    exp = experiment(contract, graph, lambda x: expected*x,
                     parameter_bounds={"a": [lo, hi, lo]})
    result = direct(contract, graph, exp)
    assert result["fit"]["status"] == "bounded_linear_fit_completed"
    assert result["parameters"]["a"] == pytest.approx(expected, abs=2e-5)


def test_variable_projection_keeps_linear_amplitude_inside_nonlinear_rate_search():
    contract, seed = fixture_graph("multiply", parameter=True)
    payload = seed.payload()
    payload["nodes"] = [
        node("x", "variable", attrs={"name": "x", "role": "observed"}),
        node("a", "parameter", attrs={"name": "a", "role": "parameter"}),
        node("b", "parameter", attrs={"name": "b", "role": "parameter"}),
        node("bx", "multiply", ["b", "x"]), node("ebx", "exp", ["bx"]),
        node("y", "multiply", ["a", "ebx"]),
    ]
    graph = HypothesisIR.from_payload(payload, contract)
    fn = lambda x: 3 * math.exp(0.5 * x)
    exp = SearchExperiment.create(contract, graph, domain={"x": [0.1, 3]}, probe_count=4,
        training_cases=cases([0.2, 0.8, 1.7], fn, "train"),
        search_cases=cases([0.4, 1.2, 2.6], fn, "search"),
        parameter_bounds={"a": [0, 10, 1], "b": [-2, 2, 0]})
    result = direct(contract, graph, exp)
    assert result["status"] == "eligible_for_confirmation"
    assert result["fit"]["status"] == "variable_projection_completed"
    assert result["fit"]["linear_parameter_ids"] == ["a"]
    assert result["fit"]["outer_parameter_ids"] == ["b"]
    assert result["parameters"]["a"] == pytest.approx(3, rel=1e-5)
    assert result["parameters"]["b"] == pytest.approx(0.5, rel=1e-5)
    assert result["fit"]["global_optimum_proven"] is False


def test_rank_deficient_affine_design_is_recorded_not_called_identifiable():
    contract, seed = fixture_graph("multiply", parameter=True)
    payload = seed.payload()
    payload["nodes"][-1:] = [
        node("b", "parameter", attrs={"name": "b", "role": "parameter"}),
        node("ax", "multiply", ["a", "x"]), node("bx", "multiply", ["b", "x"]),
        node("y", "add", ["ax", "bx"]),
    ]
    graph = HypothesisIR.from_payload(payload, contract)
    exp = experiment(contract, graph, lambda x: 2*x,
                     parameter_bounds={"a": [-5, 5, 0], "b": [-5, 5, 0]})
    result = direct(contract, graph, exp)
    assert result["status"] == "eligible_for_confirmation"
    assert result["fit"]["status"] == "bounded_linear_fit_completed"
    assert result["fit"]["jacobian_rank"] == 1
    assert result["fit"]["parameter_count"] == 2
    assert result["fit"]["global_optimum_proven"] is False


def test_deterministic_starts_stay_local_bounded_and_unique():
    bounds = [[-1e12, 1e12, 0], [0, 5, 0], [-5, 0, 0]]
    starts, lower, upper = _starts(bounds)
    assert len(starts) == 3
    assert all(np.all(s >= lower) and np.all(s <= upper) for s in starts)
    assert all(abs(s[0]) <= 1 for s in starts)
    assert all(np.array_equal(a, b) for a, b in zip(starts, _starts(bounds)[0]))
    assert len(_starts([[0, 5, 0]])[0]) == 2


def test_fit_quota_reserves_checks_and_counts_failed_calls(monkeypatch):
    contract, graph, exp = nonlinear("square")
    spec = exp.public()
    counter = EvaluationCounter(40)
    program = ScalarGraphProgram(graph, counter)

    def exhaust(fun, start, **kwargs):
        while True:
            fun(start)

    monkeypatch.setattr("scipy.optimize.least_squares", exhaust)
    params, report = fit_training_parameters(program, [c["bindings"] for c in spec["training_cases"]],
        np.array([[c["expected"]["y"]] for c in spec["training_cases"]]),
        spec["parameter_bounds"], reserved_evaluations=10)
    assert params is None
    assert report["status"] == "fit_budget_exhausted"
    assert counter.used == 30
    assert sum(a["evaluations_used"] for a in report["attempts"]) == 30
    assert len(report["attempts"]) == 3
    assert all(a["evaluations_used"] <= a["evaluation_quota"] for a in report["attempts"])


def test_insufficient_budget_is_not_structural_counterexample():
    contract, graph, exp = nonlinear("square")
    result = direct(contract, graph, exp, maximum=2)
    assert result["fit"]["status"] == "fit_budget_exhausted"
    assert result["status"] == "not_assessed"
    assert result["evaluations_used"] == 0
    assert result["witnesses"] == []


def test_later_exhausted_starts_do_not_discard_converged_fit(monkeypatch):
    from scipy.optimize import least_squares
    calls = 0

    def first_only(fun, start, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return least_squares(fun, start, **kwargs)
        while True:
            fun(start)

    monkeypatch.setattr("scipy.optimize.least_squares", first_only)
    contract, graph, exp = nonlinear("square")
    result = direct(contract, graph, exp, maximum=100)
    assert result["fit"]["selected_attempt"] == 0
    assert result["fit"]["status"] == "local_fit_completed"
    assert result["status"] == "rejected_by_checks"  # poor fit is independently rejected
    assert result["check_count"] >= len(exp.public()["search_cases"])
    assert result["evaluations_used"] <= 100
    assert result["fit"]["attempts"][-1]["status"] == "attempt_budget_exhausted"


def test_invalid_all_starts_retains_numeric_failure_without_witness():
    contract, graph, exp = nonlinear("log")
    changed = exp.public()
    changed["parameter_bounds"]["a"] = [-5, -1, -2]
    result = direct(contract, graph, SearchExperiment.from_payload(changed, contract))
    assert result["fit"]["status"] == "numeric_failure"
    assert len(result["fit"]["attempts"]) == 3
    assert result["witnesses"] == []


def test_cache_switch_preserves_restart_and_verdict():
    from core.graph_evaluator import evaluate_graph_request
    contract, graph, exp = nonlinear("square")
    payload = request(contract, graph, exp)
    cached = evaluate_graph_request(payload, max_evaluations=500)
    uncached = evaluate_graph_request({**payload, "reuse_intermediates": False}, max_evaluations=500)
    assert cached["parameters"] == uncached["parameters"]
    assert cached["fit"] == uncached["fit"]
    assert cached["status"] == uncached["status"]


def test_real_supervised_worker_supports_restarts():
    contract, graph, exp = nonlinear("log")
    result = SolverProcessRunner().execute("scalar_graph/v1", request(contract, graph, exp),
                                           limits=SolverLimits(max_evaluations=500))
    assert result["status"] == "eligible_for_confirmation"
    assert result["fit"]["selected_attempt"] == 1
