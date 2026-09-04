"""通用表变换注册表、组合执行与 Web 提交契约测试。"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.table_transformer import (
    TableTransformError,
    TableTransformationEngine,
    TableTransformationPlanner,
)
from web.app import app, user_sessions, _session_research_datasets


def test_composed_formula_time_and_panel_window_pipeline_is_auditable():
    frame = pd.DataFrame({
        "地区": ["甲", "甲", "乙", "乙"],
        "日期": ["2025-01-02", "2025-01-01", "2025-01-02", "2025-01-01"],
        "收入": [120.0, 100.0, 80.0, 60.0],
        "成本": [70.0, 60.0, 50.0, 40.0],
    })
    pipeline = [
        {"operation": "convert_types", "params": {"mapping": {"日期": "datetime"}}},
        {"operation": "derive_columns", "params": {"expressions": {
            "利润": "收入 - 成本",
            "利润率": "利润 / 收入",
            "安全利润率": 'fillna(col("利润率"), 0)',
        }}},
        {"operation": "time_features", "params": {
            "time_column": "日期", "features": ["month", "dayofweek", "month_sin"]
        }},
        {"operation": "window_features", "params": {
            "order_by": "日期",
            "partition_by": ["地区"],
            "value_columns": ["收入"],
            "features": [
                {"kind": "lag", "periods": 1},
                {"kind": "rolling_mean", "window": 2, "shift": 1},
            ],
        }},
    ]
    result = TableTransformationEngine().execute(frame, pipeline)

    assert result.output_shape == (4, 12)
    region_a = result.data[result.data["地区"] == "甲"].sort_values("日期").reset_index(drop=True)
    region_b = result.data[result.data["地区"] == "乙"].sort_values("日期").reset_index(drop=True)
    assert region_a.loc[0, "日期"] == pd.Timestamp("2025-01-01")
    assert pd.isna(region_a.loc[0, "收入__lag_1"])
    assert region_a.loc[1, "收入__lag_1"] == 100.0
    assert pd.isna(region_b.loc[0, "收入__lag_1"])
    assert result.audit[-1]["columns_added"] == ["收入__lag_1", "收入__rolling_mean_2_s1"]
    assert [item["operation"] for item in result.audit] == [
        "convert_types", "derive_columns", "time_features", "window_features"
    ]


def test_aggregate_weighted_share_pivot_and_melt_cover_common_modeling_shapes():
    frame = pd.DataFrame({
        "地区": ["甲", "甲", "乙"],
        "产品": ["A", "B", "A"],
        "销量": [10.0, 30.0, 20.0],
        "价格": [2.0, 4.0, 5.0],
    })
    engine = TableTransformationEngine()
    aggregated = engine.execute(frame, [{
        "operation": "aggregate",
        "params": {
            "group_by": ["地区"],
            "aggregations": [
                {"column": "销量", "function": "sum", "output": "销量合计"},
                {"column": "价格", "function": "weighted_mean", "weight": "销量", "output": "加权价格"},
                {"column": "销量", "function": "share_of_total", "output": "销量占比"},
            ],
        },
    }]).data
    by_region = aggregated.set_index("地区")
    assert by_region.loc["甲", "销量合计"] == 40.0
    assert by_region.loc["乙", "销量合计"] == 20.0
    assert by_region.loc["甲", "加权价格"] == pytest.approx(3.5)
    assert aggregated["销量占比"].sum() == pytest.approx(1.0)

    wide = engine.execute(frame, [{
        "operation": "pivot",
        "params": {"index": ["地区"], "columns": ["产品"], "values": ["销量"], "aggfunc": "sum", "fill_value": 0},
    }]).data
    assert wide.shape == (2, 3)
    assert set(wide.columns) == {"地区", "销量__A", "销量__B"}

    long = engine.execute(wide, [{
        "operation": "melt",
        "params": {"id_vars": ["地区"], "value_vars": ["销量__A", "销量__B"], "var_name": "产品指标", "value_name": "值"},
    }]).data
    assert long.shape == (4, 3)


def test_filters_missing_encoding_scaling_and_bins_are_composable():
    frame = pd.DataFrame({
        "组": ["甲", "甲", "乙", None],
        "值": [1.0, np.nan, 5.0, 10.0],
        "标签": ["保留-a", "删除", "保留-b", "保留-c"],
    })
    result = TableTransformationEngine().execute(frame, [
        {"operation": "fill_missing", "params": {"columns": ["值"], "strategy": "median"}},
        {"operation": "fill_missing", "params": {"columns": ["组"], "strategy": "constant", "value": "未知"}},
        {"operation": "filter_rows", "params": {"conditions": [
            {"column": "标签", "operator": "contains", "value": "保留"},
            {"column": "值", "operator": "between", "value": ["1", "10"]},
        ]}},
        {"operation": "normalize", "params": {"columns": ["值"], "method": "minmax", "suffix": "_01"}},
        {"operation": "bin_numeric", "params": {"column": "值", "method": "equal_width", "bins": 2, "output": "值层级"}},
        {"operation": "encode_categorical", "params": {"columns": ["组"], "method": "frequency"}},
    ])
    assert result.output_shape == (3, 6)
    assert result.data["值_01"].between(0, 1).all()
    assert result.data["组__frequency"].between(0, 1).all()


def test_structural_operations_types_and_one_hot_handle_messy_tables():
    frame = pd.DataFrame(
        [[2, "是", "甲", 9], [1, "否", "乙", 8], [1, "否", "乙", 8]],
        columns=["序号", "启用", "类别", "类别"],
    )
    result = TableTransformationEngine().execute(frame, [
        {"operation": "rename_columns", "params": {"mapping": {"类别__2": "分数"}}},
        {"operation": "convert_types", "params": {"mapping": {"序号": "integer", "启用": "boolean"}}},
        {"operation": "deduplicate", "params": {"subset": ["序号", "类别"]}},
        {"operation": "sort_rows", "params": {"by": ["序号"], "ascending": [True]}},
        {"operation": "encode_categorical", "params": {"columns": ["类别"], "method": "one_hot", "max_categories": 5}},
        {"operation": "drop_columns", "params": {"columns": ["启用"]}},
        {"operation": "select_columns", "params": {"columns": ["序号", "分数", "类别_甲", "类别_乙"]}},
    ])
    assert result.output_shape == (2, 4)
    assert result.data["序号"].tolist() == [1, 2]
    assert result.data[["类别_甲", "类别_乙"]].sum().sum() == 2
    assert result.warnings == ["检测到重复字段名，已追加__2、__3后缀以保证字段可寻址。"]


def test_time_resampling_and_pairwise_distance_compile_model_ready_tables():
    observations = pd.DataFrame({
        "站点": ["A", "A", "A", "B"],
        "时间": ["2025-01-01 00:01", "2025-01-01 00:08", "2025-01-01 00:16", "2025-01-01 00:02"],
        "流量": [1.0, 2.0, 4.0, 8.0],
    })
    resampled = TableTransformationEngine().execute(observations, [{
        "operation": "resample_time",
        "params": {
            "time_column": "时间",
            "frequency": "15min",
            "group_by": ["站点"],
            "aggregations": [{"column": "流量", "function": "sum", "output": "周期流量"}],
        },
    }]).data
    station_a = resampled[resampled["站点"] == "A"].sort_values("时间")
    assert station_a["周期流量"].tolist() == [3.0, 4.0]

    locations = pd.DataFrame({
        "节点": ["O", "A", "B"],
        "x": [0.0, 3.0, 0.0],
        "y": [0.0, 4.0, 5.0],
    })
    edges = TableTransformationEngine().execute(locations, [{
        "operation": "pairwise_distance",
        "params": {"id_column": "节点", "coordinate_columns": ["x", "y"], "metric": "euclidean"},
    }]).data
    assert edges.shape == (3, 3)
    origin_a = edges[(edges["source"] == "O") & (edges["target"] == "A")]
    assert origin_a["distance"].iloc[0] == pytest.approx(5.0)

    with pytest.raises(TableTransformError, match="超过预算"):
        TableTransformationEngine().execute(locations, [{
            "operation": "pairwise_distance",
            "params": {"id_column": "节点", "coordinate_columns": ["x", "y"], "max_pairs": 2},
        }])


def test_unsafe_expressions_and_dimension_explosions_are_rejected_transactionally():
    frame = pd.DataFrame({"id": range(20), "category": [f"c{i}" for i in range(20)], "value": range(20)})
    engine = TableTransformationEngine(max_result_cells=100, max_columns=10)

    with pytest.raises(TableTransformError, match="不安全|不支持"):
        engine.execute(frame, [{
            "operation": "derive_columns",
            "params": {"expressions": {"x": '__import__("os").system("echo unsafe")'}},
        }])
    with pytest.raises(TableTransformError, match="透视预计"):
        engine.execute(frame, [{
            "operation": "pivot",
            "params": {"index": ["id"], "columns": ["category"], "values": ["value"]},
        }])
    with pytest.raises(TableTransformError, match="只允许与size"):
        engine.execute(frame, [{
            "operation": "aggregate",
            "params": {"group_by": [], "aggregations": [
                {"column": "*", "function": "sum", "output": "错误汇总"}
            ]},
        }])
    assert list(frame.columns) == ["id", "category", "value"]
    assert frame.shape == (20, 3)


def test_large_vectorized_pipeline_aggregates_every_row_without_sampling():
    rows = 200_000
    frame = pd.DataFrame({
        "实体": np.arange(rows) % 1_000,
        "数量": np.ones(rows, dtype=np.float64),
        "单价": 2.5,
    })
    result = TableTransformationEngine().execute(frame, [
        {"operation": "derive_columns", "params": {"expressions": {"金额": "数量 * 单价"}}},
        {"operation": "aggregate", "params": {
            "group_by": ["实体"],
            "aggregations": [
                {"column": "数量", "function": "sum", "output": "总数量"},
                {"column": "金额", "function": "sum", "output": "总金额"},
            ],
        }},
    ])
    assert result.output_shape == (1_000, 3)
    assert result.data["总数量"].sum() == rows
    assert result.data["总金额"].sum() == pytest.approx(rows * 2.5)
    assert result.audit[0]["input_shape"][0] == rows


def test_planner_uses_problem_mentions_but_keeps_assumption_warnings():
    frame = pd.DataFrame({
        "日期": pd.date_range("2025-01-01", periods=10),
        "地区": ["甲", "乙"] * 5,
        "销量": range(10),
    })
    plan = TableTransformationPlanner.suggest(frame, "分析各地区销量的时间趋势和增长率")
    names = [item["name"] for item in plan["recommendations"]]
    assert "时序/面板数据基础" in names
    assert "分组指标汇总" in names
    time_plan = next(item for item in plan["recommendations"] if item["name"] == "时序/面板数据基础")
    window = time_plan["pipeline"][-1]
    assert window["params"]["partition_by"] == ["地区"]
    assert "粒度" in time_plan["risk"]

    duplicate_columns = pd.DataFrame([["甲", "乙", 1]], columns=["组", "组", "数值"])
    duplicate_plan = TableTransformationPlanner.suggest(duplicate_columns, "按组统计数值")
    assert duplicate_plan["profile"]["columns"] == 3


def test_operation_templates_bind_current_schema_and_never_invent_coordinates():
    product_table = pd.DataFrame({
        "单品编码": [102900005115168, 102900005115199, 102900005115625],
        "单品名称": ["牛首生菜", "四川红香椿", "本地小毛白菜"],
        "分类编码": [1011010101, 1011010101, 1011010101],
        "分类名称": ["花叶类", "花叶类", "花叶类"],
    })
    capabilities = {item["name"]: item for item in TableTransformationEngine.capabilities(product_table)}

    assert capabilities["pairwise_distance"]["availability"] == "unavailable"
    assert capabilities["pairwise_distance"]["template"] is None
    assert capabilities["aggregate"]["availability"] == "unavailable"
    assert capabilities["select_columns"]["template"]["columns"] == list(product_table.columns)

    coordinate_table = pd.DataFrame({
        "节点": ["A", "B", "C"],
        "经度": [116.1, 116.2, 116.3],
        "纬度": [39.8, 39.9, 40.0],
    })
    spatial = {item["name"]: item for item in TableTransformationEngine.capabilities(coordinate_table)}
    template = spatial["pairwise_distance"]["template"]

    assert spatial["pairwise_distance"]["availability"] == "review"
    assert template["id_column"] == "节点"
    assert template["coordinate_columns"] == ["经度", "纬度"]
    result = TableTransformationEngine().execute(
        coordinate_table,
        [{"operation": "pairwise_distance", "params": template}],
    )
    assert result.output_shape == (3, 3)


def test_transform_web_preview_apply_failure_and_undo_are_consistent():
    sid = "table-transform-api-test"
    original = pd.DataFrame({"地区": ["甲", "甲", "乙"], "销量": [1.0, 2.0, 4.0]})
    user_sessions[sid] = {
        "df": original,
        "df_info": {},
        "train_events": [],
        "train_live_results": [],
    }
    pipeline = [{
        "operation": "aggregate",
        "params": {"group_by": ["地区"], "aggregations": [
            {"column": "销量", "function": "sum", "output": "销量合计"}
        ]},
    }]
    try:
        with app.test_client() as client:
            with client.session_transaction() as flask_session:
                flask_session["sid"] = sid

            capabilities = client.get("/api/data/transform/capabilities").get_json()
            assert capabilities["success"] is True
            assert len(capabilities["capabilities"]) >= 15
            pairwise = next(item for item in capabilities["capabilities"] if item["name"] == "pairwise_distance")
            assert pairwise["availability"] == "unavailable"
            assert pairwise["template"] is None

            validation = client.post("/api/data/transform/validate", json={"pipeline": [{
                "operation": "pairwise_distance",
                "params": {"id_column": "节点", "coordinate_columns": ["经度", "纬度"]},
            }]})
            assert validation.status_code == 400
            validation_data = validation.get_json()
            assert validation_data["invalid_step"] == 1
            assert validation_data["suggested_pipeline"] == []
            assert validation_data["current_columns"] == ["地区", "销量"]

            preview = client.post("/api/data/transform/preview", json={"pipeline": pipeline})
            assert preview.status_code == 200
            preview_data = preview.get_json()
            assert preview_data["shape"] == [2, 2]
            assert preview_data["visual_preview"]["available"] is True
            assert preview_data["visual_preview"]["chart_type"] == "bar"
            assert preview_data["visual_preview"]["audit"]["output_rows"] == 2
            assert len(preview_data["visual_preview"]["records"]) == 2
            assert preview_data["visual_preview"]["presets"][0]["label"].startswith("排名")
            assert user_sessions[sid]["df"] is original

            invalid = client.post("/api/data/transform/apply", json={"pipeline": [{
                "operation": "select_columns", "params": {"columns": ["不存在"]}
            }]})
            assert invalid.status_code == 400
            assert user_sessions[sid]["df"] is original

            applied = client.post("/api/data/transform/apply", json={"pipeline": pipeline})
            assert applied.status_code == 200
            assert applied.get_json()["committed"] is True
            assert user_sessions[sid]["df"].shape == (2, 2)

            undone = client.post("/api/data/transform/undo")
            assert undone.status_code == 200
            assert undone.get_json()["shape"] == [3, 2]
            pd.testing.assert_frame_equal(user_sessions[sid]["df"], original)
    finally:
        user_sessions.pop(sid, None)


def test_transformed_current_table_is_explicitly_prioritized_for_auto_research(tmp_path):
    source_path = tmp_path / "raw.csv"
    pd.DataFrame({"原始值": [1, 2, 3]}).to_csv(source_path, index=False)
    transformed = pd.DataFrame({"实体": ["甲", "乙"], "优化指标": [10.0, 20.0]})
    session_data = {
        "df": transformed,
        "use_transformed_for_research": True,
        "transform_lineage": [{"audit": [{"operation": "aggregate"}]}],
        "uploaded_files": [{
            "filename": source_path.name,
            "path": str(source_path),
            "ext": ".csv",
            "shape": [3, 1],
            "sheets": None,
            "columns": ["原始值"],
        }],
    }
    datasets = _session_research_datasets(session_data, max_rows=1_000)
    assert next(iter(datasets)) == "当前处理结果（优先）"
    assert datasets["当前处理结果（优先）"].attrs["preferred_modeling_dataset"] is True
    assert datasets["当前处理结果（优先）"].attrs["transformation_lineage_steps"] == 1
    pd.testing.assert_frame_equal(datasets["当前处理结果（优先）"], transformed)


def test_transform_workbench_is_exposed_in_the_page():
    template = Path("web/templates/index.html").read_text(encoding="utf-8")
    script = Path("web/static/js/app.js").read_text(encoding="utf-8")
    echarts_path = Path("web/static/vendor/echarts.min.js")
    assert 'value="transform"' in template
    assert 'id="transform-pipeline"' in template
    assert 'id="transform-pipeline-cards"' in template
    assert 'id="transform-quick-actions"' in template
    assert 'class="transform-advanced-editor"' in template
    assert 'class="transform-professional-options"' in template
    assert "suggestTransformPipeline" in script
    assert "/api/data/transform/apply" in script
    assert 'id="transform-target"' in template
    assert 'id="math-data-compilation-result"' in template
    assert 'id="transform-semantic-hints"' in template
    assert 'id="interactive-viz-chart"' in template
    assert 'id="interactive-viz-recommendation"' in template
    assert 'id="viz-intent"' in template
    assert 'id="viz-dataset-source"' in template
    assert 'id="interactive-viz-config-note"' in template
    assert '多层编码' in template
    assert 'id="viz-range-min"' in template
    assert 'id="viz-animation-slider"' in template
    assert "runMathematicalDataCompilation" in script
    assert "/api/data/math-compile" in script
    assert "semantic_hints: semanticHints" in script
    assert "/api/visualization/explore/schema" in script
    assert "/api/visualization/explore/data" in script
    assert "groupInteractiveVizRecords" in script
    assert "synchronizeInteractiveVizControls" in script
    assert "validateInteractiveVizControls" in script
    assert "switchInteractiveVisualizationDataset" in script
    assert "/api/data/transform/validate" in script
    assert "renderTransformValidationFailure" in script
    assert "renderTransformPreviewChart" in script
    assert "setTransformPreviewView" in script
    assert "selectTransformPreviewPreset" in script
    assert "syncTransformPipelineCards" in script
    assert "renderTransformQuickActions" in script
    assert "setTransformGoalExample" in script
    assert "toggleTransformStepEditor" in script
    assert "applyAggregateStepEditor" in script
    assert "applyFillMissingStepEditor" in script
    assert "applyWindowStepEditor" in script
    assert "transform-step-editor" in script
    assert "点“设置”直接修改字段与算法" in template
    assert 'id="transform-preview-chart"' in script
    assert "visual_preview" in script
    assert "编码（不建议作为轴）" in script
    assert "vendor/echarts.min.js" in template
    assert "cdn.jsdelivr.net/npm/echarts" not in template
    assert echarts_path.stat().st_size == 1_029_203
    assert hashlib.sha256(echarts_path.read_bytes()).hexdigest() == (
        "42f8329d989b6f6539dd2b15bbdf0d82025762ac112fbb60dc57b27d7bcf3946"
    )


def test_mathematical_data_compile_api_is_read_only_and_reports_reversal():
    sid = "math-data-compile-api-test"
    rows = []
    for group in range(5):
        for within in range(20):
            rows.append({
                "组别": f"G{group}",
                "解释变量": group * 10 + within,
                "目标变量": (4 - group) * 100 + within * 2,
            })
    original = pd.DataFrame(rows)
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
            response = client.post("/api/data/math-compile", json={
                "problem": "分析各组解释变量与目标变量的关系",
                "target": "目标变量",
            })
            assert response.status_code == 200
            payload = response.get_json()
            assert payload["success"] is True
            assert payload["result"]["status"] == "contradicted"
            assert payload["result"]["summary"]["direction_reversals"] >= 1
            assert user_sessions[sid]["df"] is original
    finally:
        user_sessions.pop(sid, None)


def test_mathematical_data_compile_api_accepts_validated_semantic_contract():
    sid = "math-data-semantic-contract-api-test"
    original = pd.DataFrame({
        "k0": [1, 2, 3, 4],
        "x0": [10.0, 11.0, 12.0, 13.0],
        "y0": [3.0, 5.0, 7.0, 9.0],
    })
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
            response = client.post("/api/data/math-compile", json={
                "problem": "评估 x0 与 y0 的关系",
                "semantic_hints": {
                    "target": "y0",
                    "grain": ["k0"],
                    "columns": {
                        "k0": {"role": "technical_id", "semantic_id": "entity"},
                        "x0": {"role": "measure", "unit": "m"},
                        "y0": {"role": "measure", "unit": "s", "additivity": "additive"},
                    },
                },
            })
            assert response.status_code == 200
            result = response.get_json()["result"]
            assert result["schema_version"] == "mathmodel.data-compilation/v2"
            assert result["contract"]["target"] == "y0"
            assert result["contract"]["observed_grain"] == ["k0"]
            assert (
                result["contract"]["semantic_contract_source"]
                == "hybrid_explicit_and_heuristic"
            )
            assert user_sessions[sid]["df"] is original

            rejected = client.post("/api/data/math-compile", json={
                "semantic_hints": {"target": "不存在的字段"},
            })
            assert rejected.status_code == 400
            assert "不存在" in rejected.get_json()["error"]
    finally:
        user_sessions.pop(sid, None)
