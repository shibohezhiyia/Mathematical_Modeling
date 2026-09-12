import pytest

from core.paired_budget_benchmark import (
    PairedBudget,
    PairedBudgetBenchmarkError,
    run_paired_budget_benchmark,
)


def _evaluate(case, *, method, seed, wall_seconds):
    del seed, wall_seconds
    score = float(case["baseline"] if method == "baseline" else case["treatment"])
    return {"status": "completed", "score": score, "valid": True,
            "api_calls": 0 if method == "baseline" else 1, "cost": 0.0 if method == "baseline" else 0.2}


def test_paired_runner_keeps_same_case_seed_and_resource_denominator():
    report = run_paired_budget_benchmark(
        [{"id": "a", "baseline": 2, "treatment": 1}, {"id": "b", "baseline": 3, "treatment": 2}],
        _evaluate, budget=PairedBudget(max_repeats=2, seed=4),
    )
    assert report["status"] == "completed"
    assert report["baseline"]["rows"] == report["treatment"]["rows"] == 4
    assert report["baseline"]["valid_rate"] == report["treatment"]["valid_rate"] == 1
    assert report["treatment"]["api_calls"] == 4
    assert all(row["seed"] == row["seed"] for row in report["rows"])
    assert "denominator" in report["policy"]


def test_paired_runner_retains_failures_and_rejects_bad_budget():
    with pytest.raises(PairedBudgetBenchmarkError, match="wall"):
        PairedBudget(wall_seconds_per_method=0).validate()

    def fail(case, **kwargs):
        del case, kwargs
        raise RuntimeError("no result")

    report = run_paired_budget_benchmark([{"id": "a"}], fail, budget=PairedBudget(max_repeats=1))
    assert report["baseline"]["valid_rate"] == 0
    assert report["treatment"]["valid_rate"] == 0
    assert report["baseline"]["rows"] == 1
