import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from core.data_graph_bridge import build_scalar_graph_bundle, build_tabular_graph_bundle
from core.graph_search_artifacts import run_search_bundle


def test_data_bridge_creates_disjoint_typed_search_bundle():
    frame = pd.DataFrame({"feature": np.arange(-4, 5), "target": np.arange(-4, 5) ** 2})
    bundle, audit = build_scalar_graph_bundle(frame, "feature", "target", random_state=7)
    assert bundle["schema_version"] == "mathmodel.graph-search-bundle/v1"
    assert audit["unit_status"].startswith("assumed_dimensionless")
    training = {case["bindings"]["x"] for case in bundle["experiment"]["training_cases"]}
    search = {case["bindings"]["x"] for case in bundle["experiment"]["search_cases"]}
    assert training.isdisjoint(search)
    assert bundle["experiment"]["template"]["nodes"][1]["op"] == "unknown_mechanism"


def test_data_bridge_aggregates_duplicate_inputs_and_rejects_small_data():
    frame = pd.DataFrame({"x": [0, 0, 1, 2, 3, 4], "y": [1, 3, 2, 4, 5, np.nan]})
    _, audit = build_scalar_graph_bundle(frame, "x", "y")
    assert audit["duplicate_feature_rows"] == 1
    assert audit["aggregation"] == "median_target_per_feature"
    with pytest.raises(ValueError, match="four"):
        build_scalar_graph_bundle(pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]}), "x", "y")


def test_data_bridge_supports_two_feature_interaction_search():
    frame = pd.DataFrame({"a": [0, 1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1, 0],
                         "y": [0, 4, 6, 6, 4, 0]})
    bundle, audit = build_tabular_graph_bundle(frame, ["a", "b"], "y", random_state=1)
    assert audit["features"] == ["a", "b"]
    assert set(bundle["experiment"]["domain"]) == {"x0", "x1"}
    assert bundle["experiment"]["template"]["nodes"][-1]["inputs"] == ["x0", "x1"]


def test_data_bridge_lowers_three_dimensionless_features_to_executable_fold(tmp_path):
    frame = pd.DataFrame({
        "a": np.arange(8, dtype=float),
        "b": np.arange(1, 9, dtype=float),
        "c": np.array([2, 1, 0, 1, 2, 3, 4, 5], dtype=float),
        "y": np.arange(1, 9, dtype=float),
    })
    bundle, audit = build_tabular_graph_bundle(frame, ["a", "b", "c"], "y")
    assert audit["features"] == ["a", "b", "c"]
    assert bundle["experiment"]["template"]["nodes"][-1]["inputs"] == ["x0", "x1", "x2"]
    result, _ = run_search_bundle(bundle, output_root=Path(tmp_path))
    assert any(item["status"] != "not_executable" for item in result["reports"])


def test_data_bridge_allows_same_dimension_multi_input_additive_fold(tmp_path):
    frame = pd.DataFrame({
        "a": np.arange(8, dtype=float), "b": np.arange(1, 9, dtype=float),
        "c": np.arange(2, 10, dtype=float), "y": np.arange(3, 11, dtype=float),
    })
    dimensions = {"a": {"L": 1}, "b": {"L": 1}, "c": {"L": 1}}
    bundle, _ = build_tabular_graph_bundle(
        frame, ["a", "b", "c"], "y", feature_dimensions=dimensions,
        target_dimensions={"y": {"L": 1}},
    )
    result, _ = run_search_bundle(bundle, output_root=Path(tmp_path))
    assert any(item["status"] != "not_executable" for item in result["reports"])


def test_data_bridge_allows_dimensionally_valid_multi_input_product(tmp_path):
    frame = pd.DataFrame({
        "length": np.arange(1, 9, dtype=float),
        "time": np.arange(1, 9, dtype=float),
        "scale": np.arange(1, 9, dtype=float),
        "output": np.arange(1, 9, dtype=float) ** 2,
    })
    bundle, _ = build_tabular_graph_bundle(
        frame, ["length", "time", "scale"], "output",
        feature_dimensions={"length": {"L": 1}, "time": {"T": 1}, "scale": {}},
        target_dimensions={"output": {"L": 1, "T": 1}},
    )
    result, _ = run_search_bundle(bundle, output_root=Path(tmp_path))
    assert any(item["status"] != "not_executable" for item in result["reports"])


def test_data_bridge_rejects_dimensionally_invalid_product_fold(tmp_path):
    frame = pd.DataFrame({
        "length": np.arange(1, 9, dtype=float),
        "time": np.arange(1, 9, dtype=float),
        "scale": np.arange(1, 9, dtype=float),
        "output": np.arange(1, 9, dtype=float),
    })
    bundle, _ = build_tabular_graph_bundle(
        frame, ["length", "time", "scale"], "output",
        feature_dimensions={"length": {"L": 1}, "time": {"T": 1}, "scale": {}},
        target_dimensions={"output": {"L": 1}},
    )
    result, _ = run_search_bundle(bundle, output_root=Path(tmp_path))
    assert result["status"] == "no_candidate_passed"
    assert all(item["status"] == "not_executable" for item in result["reports"])


def test_data_bridge_audit_preserves_pre_sampling_counts():
    frame = pd.DataFrame({"x": np.arange(20), "y": np.arange(20) ** 2})
    _, audit = build_scalar_graph_bundle(frame, "x", "y", max_rows=8)
    assert audit["rows_finite"] == 20
    assert audit["rows_distinct_feature"] == 20
    assert audit["rows_used"] == 8
    assert audit["dropped_nonfinite_rows"] == 0


def test_data_bridge_accepts_explicit_dimensions_and_records_them():
    frame = pd.DataFrame({"distance": [1, 2, 3, 4], "time": [1, 2, 3, 4]})
    bundle, audit = build_scalar_graph_bundle(
        frame, "time", "distance",
        feature_dimensions={"time": {"T": 1}},
        target_dimensions={"distance": {"L": 1}},
    )
    assert audit["unit_status"] == "explicit_dimensions_bound"
    assert bundle["experiment"]["template"]["nodes"][0]["type"]["dimensions"] == {"T": 1}
    assert bundle["experiment"]["template"]["nodes"][1]["type"]["dimensions"] == {"L": 1}
