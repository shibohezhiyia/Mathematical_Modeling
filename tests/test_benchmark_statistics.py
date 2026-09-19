import pytest

from core.benchmark_statistics import BenchmarkStatisticsError, paired_benchmark_effect


def test_paired_benchmark_effect_keeps_small_samples_descriptive():
    result = paired_benchmark_effect([{"baseline": 1, "treatment": 2}] * 2, min_samples=5)
    assert result["status"] == "descriptive_only"
    assert result["confidence_interval"] is None


def test_paired_benchmark_effect_is_reproducible():
    samples = [{"baseline": 1, "treatment": 2}, {"baseline": 2, "treatment": 3},
               {"baseline": 4, "treatment": 3}, {"baseline": 5, "treatment": 7},
               {"baseline": 8, "treatment": 9}]
    a = paired_benchmark_effect(samples, bootstrap_replicates=100)
    b = paired_benchmark_effect(samples, bootstrap_replicates=100)
    assert a["confidence_interval"] == b["confidence_interval"]


def test_paired_benchmark_effect_resamples_structure_clusters():
    samples = [
        {"baseline": 0, "treatment": 1, "structure": "a"},
        {"baseline": 0, "treatment": 1, "structure": "a"},
        {"baseline": 1, "treatment": 0, "structure": "b"},
    ]
    result = paired_benchmark_effect(samples, cluster="structure", min_samples=2,
                                     bootstrap_replicates=100)
    assert result["sample_count"] == 3
    assert result["independent_unit_count"] == 2
    assert result["resampling_unit"] == "cluster"
    assert result["paired_effect"] == 0.0


@pytest.mark.parametrize("kwargs", [
    {"seed": -1}, {"min_samples": 0}, {"bootstrap_replicates": 99},
    {"baseline": "x", "treatment": "x"},
])
def test_paired_benchmark_contract_is_bounded(kwargs):
    with pytest.raises(BenchmarkStatisticsError):
        paired_benchmark_effect([{"baseline": 1, "treatment": 2}] * 5, **kwargs)


def test_paired_benchmark_rejects_bootstrap_memory_explosion():
    with pytest.raises(BenchmarkStatisticsError, match="bootstrap_memory_budget_exceeded"):
        paired_benchmark_effect([{"baseline": 1, "treatment": 2}] * 2000, bootstrap_replicates=20_000)
