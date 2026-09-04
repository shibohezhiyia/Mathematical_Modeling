"""交互式多维图形编译器的边界、审计与大表安全测试。"""

import numpy as np
import pandas as pd
import pytest

from core.interactive_visualization import (
    InteractiveVisualizationCompiler,
    InteractiveVisualizationError,
)
from web.app import app, user_sessions


def _sample_frame(rows: int = 120) -> pd.DataFrame:
    return pd.DataFrame({
        "时间": pd.date_range("2025-01-01", periods=rows, freq="h"),
        "地区": np.resize(np.array(["东", "西", "南", "北"]), rows),
        "批次": np.resize(np.array(["A", "B", "C"]), rows),
        "投入": np.arange(rows, dtype=float),
        "产出": np.arange(rows, dtype=float) * 1.5 + 2,
        "规模": np.linspace(1, 50, rows),
    })


def test_schema_profiles_dimensions_and_recommends_time_chart():
    schema = InteractiveVisualizationCompiler().describe(_sample_frame())
    fields = {item["name"]: item for item in schema["fields"]}
    assert schema["schema_version"] == "mathmodel.interactive-visualization/v2"
    assert fields["时间"]["kind"] == "datetime"
    assert fields["地区"]["kind"] == "categorical"
    assert fields["投入"]["kind"] == "numeric"
    assert fields["投入"]["range"] == [0.0, 119.0]
    assert schema["recommendation"]["chart_type"] == "line"
    assert schema["recommendation"]["encodings"]["x"] == "时间"

    string_dates = InteractiveVisualizationCompiler().describe(pd.DataFrame({
        "when": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "value": [1.0, 2.0, 3.0],
    }))
    assert {item["name"]: item["kind"] for item in string_dates["fields"]}["when"] == "datetime"


def test_numeric_codes_and_high_cardinality_names_are_not_recommended_as_measures():
    rows = 251
    frame = pd.DataFrame({
        "商品编码": 102_900_051_115_000 + np.arange(rows),
        "商品名称": [f"商品{i}" for i in range(rows)],
        "分类编码": np.resize(np.array([401_000_000, 701_000_000, 1_001_000_000]), rows),
        "分类名称": np.resize(np.array(["叶菜", "根茎", "菌菇"]), rows),
    })
    schema = InteractiveVisualizationCompiler().describe(frame)
    fields = {item["name"]: item for item in schema["fields"]}
    recommendation = schema["recommendation"]
    assert fields["商品编码"]["kind"] == "identifier"
    assert fields["分类编码"]["semantic_role"] == "identifier"
    assert fields["商品名称"]["semantic_role"] == "label"
    assert fields["分类名称"]["semantic_role"] == "dimension"
    assert recommendation["chart_type"] == "bar"
    assert recommendation["encodings"] == {"x": "分类名称", "y": None}
    assert recommendation["aggregation"]["function"] == "count"
    assert schema["capability"]["level"] == "composition_only"
    assert schema["capability"]["enabled_charts"] == ["bar"]

    with_measure = frame.assign(销量=np.arange(rows, dtype=float))
    guarded = InteractiveVisualizationCompiler().compile(with_measure, {
        "chart_type": "bar",
        "encodings": {"x": "商品编码", "y": "销量", "color": "商品名称"},
        "aggregation": {"function": "none"},
    })
    assert guarded["field_types"]["商品编码"] == "identifier"
    assert isinstance(guarded["records"][0]["商品编码"], str)
    assert any("不应解释为连续数量" in warning for warning in guarded["warnings"])
    assert any("合并为“其他”" in warning for warning in guarded["warnings"])

    with pytest.raises(InteractiveVisualizationError, match="Y轴必须是真实数值度量"):
        InteractiveVisualizationCompiler().compile(frame, {
            "chart_type": "scatter",
            "encodings": {"x": "商品编码", "y": "分类编码", "color": "商品名称"},
            "aggregation": {"function": "none"},
        })

    compacted = InteractiveVisualizationCompiler().compile(frame, {
        "chart_type": "bar",
        "encodings": {"x": "分类名称", "color": "商品名称"},
        "aggregation": {"function": "count"},
    })
    colors = {record["商品名称"] for record in compacted["records"]}
    assert len(colors) <= 12
    assert "其他" in colors
    assert sum(record["__count__"] for record in compacted["records"]) == rows
    color_audit = compacted["audit"]["aggregation"]["color_compaction"]
    assert color_audit["source_levels"] == rows
    assert color_audit["dropped_records"] == 0

    with pytest.raises(InteractiveVisualizationError, match="几乎一行一值"):
        InteractiveVisualizationCompiler().compile(frame, {
            "chart_type": "bar",
            "encodings": {"x": "商品名称"},
            "aggregation": {"function": "count"},
        })


def test_unique_result_labels_are_displayed_without_reaggregation():
    frame = pd.DataFrame({
        "地区": ["东", "西", "南", "北"],
        "产出合计": [20.0, 28.0, 38.0, 50.0],
    })
    compiler = InteractiveVisualizationCompiler()
    schema = compiler.describe(frame)
    recommendation = schema["recommendation"]
    assert recommendation["chart_type"] == "bar"
    assert recommendation["aggregation"]["function"] == "none"
    result = compiler.compile(frame, {
        "chart_type": recommendation["chart_type"],
        "encodings": recommendation["encodings"],
        "aggregation": recommendation["aggregation"],
    })
    assert result["audit"]["aggregation"]["function"] == "none"
    assert [row["产出合计"] for row in result["records"]] == [20.0, 28.0, 38.0, 50.0]


def test_redundant_visual_channels_and_count_scatter_are_rejected():
    frame = _sample_frame()
    compiler = InteractiveVisualizationCompiler()
    with pytest.raises(InteractiveVisualizationError, match="重复编码不会增加信息"):
        compiler.compile(frame, {
            "chart_type": "scatter",
            "encodings": {"x": "投入", "y": "投入", "color": "地区", "facet": "地区"},
            "aggregation": {"function": "none"},
        })
    with pytest.raises(InteractiveVisualizationError, match="计数聚合只能使用柱状图"):
        compiler.compile(frame, {
            "chart_type": "scatter",
            "encodings": {"x": "地区", "y": "产出"},
            "aggregation": {"function": "count"},
        })


def test_scatter_filters_facets_and_browser_payload_are_bounded_and_read_only():
    frame = _sample_frame(1_000)
    original = frame.copy(deep=True)
    result = InteractiveVisualizationCompiler(max_points=600).compile(frame, {
        "chart_type": "scatter",
        "encodings": {
            "x": "投入", "y": "产出", "color": "地区",
            "size": "规模", "facet": "批次", "tooltip": ["时间"],
        },
        "filters": [
            {"field": "投入", "kind": "range", "min": 100, "max": 899},
            {"field": "地区", "kind": "in", "values": ["东", "西"]},
        ],
        "aggregation": {"function": "none"},
        "max_points": 500,
    })
    assert result["chart_type"] == "scatter"
    assert 0 < len(result["records"]) <= 500
    assert set(result["facet_levels"]) <= {"A", "B", "C"}
    assert result["audit"]["filtered_rows"] == 400
    assert result["audit"]["source_mutated"] is False
    pd.testing.assert_frame_equal(frame, original)


def test_aggregation_compiles_time_grain_and_count_without_fake_y_field():
    compiler = InteractiveVisualizationCompiler()
    mean_result = compiler.compile(_sample_frame(), {
        "chart_type": "line",
        "encodings": {"x": "时间", "y": "产出", "color": "地区"},
        "aggregation": {"function": "mean", "time_unit": "day", "bins": 20},
    })
    assert mean_result["audit"]["aggregation"]["function"] == "mean"
    assert mean_result["audit"]["aggregation"]["group_by"] == ["时间", "地区"]
    assert len(mean_result["records"]) == 20

    guarded = compiler.compile(_sample_frame(), {
        "chart_type": "bar",
        "encodings": {
            "x": "地区", "y": "产出", "color": "投入", "size": "规模",
        },
        "aggregation": {"function": "mean"},
    })
    assert guarded["encodings"]["color"] is None
    assert guarded["encodings"]["size"] is None
    assert set(guarded["audit"]["aggregation"]["dropped_encodings"]) == {"color", "size"}
    assert any("已停用" in warning for warning in guarded["warnings"])

    count_result = compiler.compile(_sample_frame(), {
        "chart_type": "bar",
        "encodings": {"x": "地区"},
        "aggregation": {"function": "count"},
    })
    assert count_result["encodings"]["y"] == "__count__"
    assert sum(row["__count__"] for row in count_result["records"]) == 120


def test_large_chart_uses_coverage_scan_and_marks_aggregate_as_exploratory():
    rows = 50_000
    frame = pd.DataFrame({
        "组": np.resize(np.array(["A", "B", "C", "D"]), rows),
        "x": np.arange(rows, dtype=float),
        "y": np.arange(rows, dtype=float) * 2,
    })
    result = InteractiveVisualizationCompiler(
        max_scan_rows=5_000, max_points=1_000
    ).compile(frame, {
        "chart_type": "bar",
        "encodings": {"x": "组", "y": "y"},
        "aggregation": {"function": "sum"},
    })
    assert result["audit"]["scan_scope"] == "coverage_sample"
    assert result["audit"]["scanned_rows"] == 5_000
    assert any("不是精确总量" in warning for warning in result["warnings"])


def test_parallel_and_invalid_requests_fail_or_compile_deterministically():
    frame = _sample_frame()
    result = InteractiveVisualizationCompiler().compile(frame, {
        "chart_type": "parallel",
        "encodings": {"parallel": ["投入", "产出", "规模"], "color": "地区"},
    })
    assert result["chart_type"] == "parallel"
    assert len(result["records"]) == len(frame)

    with pytest.raises(InteractiveVisualizationError, match="至少需要选择2个字段"):
        InteractiveVisualizationCompiler().compile(frame, {
            "chart_type": "parallel",
            "encodings": {"parallel": ["投入"]},
        })
    with pytest.raises(InteractiveVisualizationError, match="不存在字段"):
        InteractiveVisualizationCompiler().compile(frame, {
            "chart_type": "scatter",
            "encodings": {"x": "不存在", "y": "产出"},
        })
    with pytest.raises(InteractiveVisualizationError, match="下界不能大于上界"):
        InteractiveVisualizationCompiler().compile(frame, {
            "chart_type": "scatter",
            "encodings": {"x": "投入", "y": "产出"},
            "filters": [{"field": "投入", "kind": "range", "min": 5, "max": 2}],
        })


def test_interactive_visualization_web_api_is_bounded_and_read_only():
    sid = "interactive-visualization-api-test"
    original = _sample_frame(2_000)
    user_sessions[sid] = {
        "df": original,
        "df_info": {},
        "train_events": [],
        "train_live_results": [],
    }
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid
            schema_response = client.get("/api/visualization/explore/schema")
            assert schema_response.status_code == 200
            schema = schema_response.get_json()["schema"]
            assert schema["source_rows"] == 2_000

            data_response = client.post("/api/visualization/explore/data", json={
                "chart_type": "scatter",
                "encodings": {
                    "x": "投入", "y": "产出", "color": "地区", "facet": "批次",
                },
                "aggregation": {"function": "none"},
                "max_points": 500,
            })
            assert data_response.status_code == 200
            result = data_response.get_json()["result"]
            assert result["audit"]["output_rows"] == 500
            assert user_sessions[sid]["df"] is original

            rejected = client.post("/api/visualization/explore/data", json={
                "chart_type": "scatter",
                "encodings": {"x": "投入", "y": "missing"},
            })
            assert rejected.status_code == 400
            assert "不存在字段" in rejected.get_json()["error"]

            redundant = client.post("/api/visualization/explore/data", json={
                "chart_type": "scatter",
                "encodings": {"x": "投入", "y": "投入", "color": "地区", "facet": "地区"},
                "aggregation": {"function": "none"},
            })
            assert redundant.status_code == 400
            assert "重复编码不会增加信息" in redundant.get_json()["error"]
    finally:
        user_sessions.pop(sid, None)
