import pytest

from core.cache_consistency import CacheConsistencyError, compare_cached_uncached


def test_cached_and_uncached_results_match_except_telemetry():
    result = compare_cached_uncached(
        lambda: {"status": "executed", "value": [1.0, 2.0], "duration_seconds": 9},
        lambda: {"status": "executed", "value": [1.0 + 1e-10, 2.0], "duration_seconds": 1, "cache_hit": True},
    )
    assert result["status"] == "tested_not_falsified"


def test_cache_difference_is_a_counterexample():
    result = compare_cached_uncached(lambda: {"value": 1}, lambda: {"value": 2})
    assert result["status"] == "counterexample"


def test_cyclic_outputs_are_reported_as_counterexamples_with_bounded_traversal():
    left = {}
    right = {}
    left["self"] = left
    right["self"] = right
    result = compare_cached_uncached(lambda: left, lambda: right)
    assert result["status"] == "counterexample"
    assert result["mismatches"][0]["reason"] == "cycle_detected"


def test_cache_comparison_budgets_are_validated():
    with pytest.raises(CacheConsistencyError, match="invalid_comparison_depth_budget"):
        compare_cached_uncached(lambda: {"x": 1}, lambda: {"x": 1}, max_depth=0)
