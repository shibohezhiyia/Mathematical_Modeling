import time
import pytest

from core.benchmark_runner import BenchmarkBudget, BenchmarkRunnerError, run_fixed_benchmark


def test_fixed_benchmark_retains_statuses_and_latency_summary():
    result = run_fixed_benchmark(
        [{"id": "a"}, {"id": "b"}],
        lambda case: {"status": "completed", "score": 1.0},
        budget=BenchmarkBudget(max_repeats=2),
    )
    assert result["status_counts"]["completed"] == 4
    assert result["latency_seconds"]["p50"] is not None


def test_fixed_benchmark_records_failures_and_rejects_duplicates():
    result = run_fixed_benchmark([{"id": "a"}], lambda case: (_ for _ in ()).throw(RuntimeError("x")))
    assert result["status_counts"]["failed"] == 3
    with pytest.raises(BenchmarkRunnerError):
        run_fixed_benchmark([{"id": "a"}, {"id": "a"}], lambda case: {})


def test_fixed_benchmark_isolates_invalid_evaluator_payloads():
    result = run_fixed_benchmark(
        [{"id": "a"}, {"id": " b "}],
        lambda case: {"status": " "} if case["id"] == "a" else {"status": "completed"},
        budget=BenchmarkBudget(max_repeats=1),
    )
    assert result["status_counts"]["failed"] == 1
    assert result["status_counts"]["completed"] == 1
    assert [row["id"] for row in result["rows"]] == ["a", "b"]


def test_fixed_benchmark_rejects_unknown_status_instead_of_silently_counting_it():
    result = run_fixed_benchmark([{"id": "a"}], lambda case: {"status": "made_up"},
                                 budget=BenchmarkBudget(max_repeats=1))
    assert result["status_counts"]["failed"] == 1
    assert result["rows"][0]["error_code"] == "invalid_evaluator_status"
