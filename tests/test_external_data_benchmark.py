from pathlib import Path

import pandas as pd

from core.external_data_benchmark import _prepare, run_external_data_benchmark


def test_external_benchmark_reports_missing_data_without_claiming_accuracy(tmp_path):
    catalog = Path(__file__).parents[1] / "examples" / "external_dataset_catalog.json"
    result = run_external_data_benchmark(catalog, tmp_path)
    assert result["evaluation_status"] == "descriptive_public_labels"
    assert result["completed_count"] == 0
    assert result["paired_relative_rmse_effect_treatment_minus_baseline"] is None


def test_external_benchmark_statistics_are_bounded_and_reproducible(tmp_path):
    catalog = Path(__file__).parents[1] / "examples" / "external_dataset_catalog.json"
    result = run_external_data_benchmark(catalog, tmp_path)
    stats = result["paired_relative_rmse_statistics"]
    assert stats["status"] == "not_assessed"
    assert stats["permutation_p_two_sided"] is None


def test_external_benchmark_exposes_independent_scorer_contract(tmp_path):
    catalog = Path(__file__).parents[1] / "examples" / "external_dataset_catalog.json"
    result = run_external_data_benchmark(catalog, tmp_path)
    assert result["data_contract"]["scorer"] == "independent_holdout_scorer_v1"


def test_prepare_removes_target_sentinel_and_declared_leakage_columns():
    frame = pd.DataFrame({"target": [1.0, -200.0] + list(range(3, 42)),
                          "casual": list(range(41)), "registered": list(range(41, 82)),
                          "instant": list(range(41)), "x": list(range(82, 123))})
    X, y, audit = _prepare(frame, "target", drop_columns=("casual", "registered", "instant"))
    assert len(y) == 40 and "casual" not in X.columns and "instant" not in X.columns
    assert audit["target_missing_sentinel_rows"] == 1
    assert audit["dropped_target_rows"] == 1
