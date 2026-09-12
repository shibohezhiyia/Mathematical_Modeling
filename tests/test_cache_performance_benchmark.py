import pytest

from core.cache_performance_benchmark import (
    CachePerformanceBenchmarkError,
    run_cache_performance_benchmark,
)


def _runner(case, *, cache_enabled, incremental, seed):
    del cache_enabled, incremental, seed
    return {"value": case["value"]}


def test_cache_benchmark_records_modes_and_consistency():
    report = run_cache_performance_benchmark([{"id": "a", "value": 3}], _runner, seed=9)
    assert report["status"] == "assessed"
    assert report["consistent_count"] == 1
    row = report["rows"][0]
    assert set(row) >= {"cold", "warm_first", "warm_second", "incremental", "consistency"}
    assert "timing_is_not_speedup" in report["policy"]


def test_cache_benchmark_keeps_failures_unassessed_and_rejects_invalid_case():
    with pytest.raises(CachePerformanceBenchmarkError, match="invalid_case"):
        run_cache_performance_benchmark([{"id": ""}], _runner)

    def fail(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("cache failure")

    report = run_cache_performance_benchmark([{"id": "a", "value": 1}], fail)
    assert report["status"] == "partial"
    assert report["assessed_count"] == 0
