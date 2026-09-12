import pytest

from core.cegis_controller import CEGISControllerError, minimize_counterexample


def test_counterexample_minimization_keeps_violation_and_prefers_smallest_candidate():
    def violation(item):
        return item["x"] >= 3

    def shrink(item):
        return [{"x": item["x"] - 1}, {"x": 3}, {"x": item["x"] + 1}]

    result = minimize_counterexample({"x": 7}, violation, shrink)
    assert result["status"] == "minimized"
    assert result["final"] == {"x": 3}
    assert result["proof_status"] == "locally_minimized_under_declared_shrinker"


def test_counterexample_minimization_does_not_promote_nonviolation_or_unbounded_shrinker():
    result = minimize_counterexample({"x": 1}, lambda item: item["x"] > 2,
                                     lambda item: [{"x": 0}])
    assert result["status"] == "not_assessed"
    with pytest.raises(CEGISControllerError, match="budget"):
        minimize_counterexample({"x": 3}, lambda item: True, lambda item: [], max_steps=0)


def test_counterexample_minimization_evaluates_initial_witness_once():
    calls = []
    result = minimize_counterexample(
        {"x": 3}, lambda item: (calls.append(dict(item)) or item["x"] > 1),
        lambda item: [{"x": 2}, {"x": 1}], max_steps=8)
    assert result["status"] == "minimized"
    assert calls[0] == {"x": 3}
