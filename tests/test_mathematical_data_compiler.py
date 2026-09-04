"""数学数据编译、多视图不变量和结论翻转审计测试。"""

import numpy as np
import pandas as pd

from core.mathematical_data_compiler import MathematicalDataCompiler


def test_contract_infers_grain_additivity_and_preserves_additive_totals():
    dates = pd.date_range("2025-01-01", periods=12, freq="D")
    frame = pd.DataFrame([
        {
            "日期": date,
            "地区": region,
            "销量(件)": float(day + 1),
            "销售单价(元)": float(10 + day),
        }
        for day, date in enumerate(dates)
        for region in ("甲", "乙", "丙")
    ])
    result = MathematicalDataCompiler().compile(
        frame,
        problem="预测各地区销量的日变化趋势",
        target="销量(件)",
    )

    contract = result["contract"]
    semantics = {item["column"]: item for item in contract["columns_semantics"]}
    assert contract["observed_grain"] == ["日期", "地区"]
    assert contract["grain_uniqueness"] == 1.0
    assert semantics["销量(件)"]["additivity"] == "additive"
    assert semantics["销售单价(元)"]["additivity"] == "non_additive"
    assert semantics["销量(件)"]["unit"] == "件"

    entity_view = next(view for view in result["views"] if view["view_id"] == "entity_level_estimand")
    conservation = next(item for item in entity_view["conservation_audit"] if item["column"] == "销量(件)")
    assert conservation["status"] == "pass"
    assert conservation["before_total"] == conservation["after_total"]
    time_view = next(view for view in result["views"] if view["view_id"] == "time_available_features")
    assert time_view["leakage_audit"]["status"] == "pass"
    assert time_view["admissible"] is True


def test_simpson_direction_reversal_is_reported_as_counterevidence():
    rows = []
    for group in range(5):
        for within in range(20):
            rows.append({
                "组别": f"G{group}",
                "解释变量": group * 10 + within,
                # 每组内部正相关，但组间截距下降，整体呈负相关。
                "目标变量": (4 - group) * 100 + within * 2,
            })
    frame = pd.DataFrame(rows)
    result = MathematicalDataCompiler().compile(
        frame,
        problem="分析各组解释变量与目标变量的关系",
        target="目标变量",
    )

    relationship = next(
        item for item in result["conclusion_stress"]["relationships"]
        if item["predictor"] == "解释变量"
    )
    contexts = {item["view"]: item for item in relationship["contexts"]}
    assert contexts["global_complete_case"]["rho"] < 0
    assert contexts["within_group:组别"]["rho"] > 0
    assert relationship["status"] == "contradicted"
    assert relationship["simpson_risk"] is True
    assert result["summary"]["direction_reversals"] >= 1
    assert result["status"] == "contradicted"
    stability_check = next(
        item for item in result["credibility_audit"]["checks"]
        if item["id"] == "conclusion_view_stability"
    )
    assert result["credibility_audit"]["status"] == "fail"
    assert stability_check["status"] == "fail"


def test_robust_views_separate_estimands_instead_of_ranking_incompatible_grains():
    frame = pd.DataFrame({
        "月份": pd.date_range("2024-01-01", periods=18, freq="MS"),
        "区域": ["甲", "乙", "丙"] * 6,
        "需求量": np.arange(18, dtype=float) + 1,
        "缺失特征": [np.nan if i % 4 == 0 else float(i) for i in range(18)],
    })
    result = MathematicalDataCompiler().compile(
        frame,
        problem="按月预测各区域需求量",
        target="需求量",
    )
    views = {item["view_id"]: item for item in result["views"]}

    assert views["observed_baseline"]["row_relation"] == "baseline"
    assert views["missing_robustness"]["row_relation"] == "row_preserving"
    assert views["entity_level_estimand"]["row_relation"] == "grain_changing"
    assert "区域" in views["entity_level_estimand"]["estimand"]
    assert views["time_grain_estimand"]["output_grain"] == ["区域", "月份"]
    assert all("score" not in view for view in result["views"])


def test_missing_target_or_grain_is_explicitly_unresolved():
    frame = pd.DataFrame({
        "随机文本": [f"row-{index}" for index in range(20)],
        "数值A": np.arange(20, dtype=float),
        "数值B": np.arange(20, dtype=float) ** 2,
    })
    result = MathematicalDataCompiler().compile(frame, problem="探索数据结构")

    assert result["contract"]["target"] is None
    assert result["status"] == "needs_input"
    assert result["conclusion_stress"]["status"] == "not_assessed"
    assert any("目标字段" in item for item in result["contract"]["unresolved"])


def test_all_missing_additive_measure_does_not_create_false_conservation_failure():
    frame = pd.DataFrame({
        "地区": ["甲", "甲", "乙", "乙"],
        "销量": pd.Series([np.nan, np.nan, np.nan, np.nan], dtype=float),
        "可用指标": [1.0, 2.0, 3.0, 4.0],
    })
    result = MathematicalDataCompiler().compile(frame, target="销量")
    entity_view = next(
        view for view in result["views"] if view["view_id"] == "entity_level_estimand"
    )
    conservation = next(
        item for item in entity_view["conservation_audit"] if item["column"] == "销量"
    )
    assert conservation["status"] == "not_assessed"
    assert "销量总量守恒失败" not in entity_view["blocking_reasons"]


def test_large_table_compilation_executes_views_only_on_bounded_coverage_sample():
    size = 5_000
    frame = pd.DataFrame({
        "日期": pd.date_range("2020-01-01", periods=size, freq="h"),
        "地区": np.resize(np.array(["甲", "乙", "丙", "丁"]), size),
        "销量": np.arange(size, dtype=float),
        "特征": np.sin(np.arange(size, dtype=float) / 20),
    })
    compiler = MathematicalDataCompiler(max_analysis_rows=1_000)
    executed_rows = []
    original_execute = compiler.engine.execute

    def bounded_execute(data, pipeline):
        executed_rows.append(len(data))
        return original_execute(data, pipeline)

    compiler.engine.execute = bounded_execute
    result = compiler.compile(frame, problem="按地区预测销量", target="销量")

    assert executed_rows and max(executed_rows) <= 1_000
    assert result["summary"]["source_rows"] == size
    assert result["summary"]["audited_rows"] == 1_000
    assert result["summary"]["sampled_execution"] is True
    assert result["status"] == "restricted"
    assert result["contract"]["audit_scope"]["full_execution_reaudit_required"] is True


def test_multi_table_compiler_distinguishes_feature_enrichment_from_additive_join():
    customers = pd.DataFrame({
        "customer_id": np.arange(20),
        "地区": np.resize(np.array(["东", "西"]), 20),
        "满意度": np.linspace(60, 90, 20),
    })
    orders = pd.DataFrame({
        "customer_id": np.repeat(np.arange(20), 4),
        "订单金额": np.arange(80, dtype=float) + 1,
    })
    result = MathematicalDataCompiler().compile_many(
        {"customers": customers, "orders": orders},
        problem="分析订单金额与客户满意度的关系",
        target="orders.订单金额",
    )
    contract = result["cross_dataset_contracts"][0]

    assert contract["relationship"] in {"one_to_many", "many_to_one"}
    assert contract["safe_feature_enrichment_direction"] == "customers_to_orders"
    assert contract["combined_additive_analysis"] == "requires_preaggregation_to_estimand_grain"
    assert contract["status"] == "restricted"


def test_multi_table_compiler_uses_composite_key_and_blocks_raw_many_to_many():
    dates = pd.date_range("2025-01-01", periods=10, freq="D")
    left = pd.DataFrame([
        {"日期": date, "地区": region, "指标A": float(index)}
        for index, date in enumerate(dates)
        for region in ("甲", "乙")
    ])
    right = pd.DataFrame([
        {"日期": date, "地区": region, "指标B": float(index * 2)}
        for index, date in enumerate(dates)
        for region in ("甲", "乙")
    ])
    composite = MathematicalDataCompiler().compile_many(
        {"left": left, "right": right}, target="left.指标A"
    )["cross_dataset_contracts"][0]
    assert composite["relationship"] == "one_to_one"
    assert len(composite["key_pairs"]) == 2
    assert composite["status"] == "admissible"

    many_left = pd.DataFrame({
        "组别": np.repeat(["A", "B", "C"], 5),
        "指标A": np.arange(15, dtype=float),
    })
    many_right = pd.DataFrame({
        "组别": np.repeat(["A", "B", "C"], 7),
        "指标B": np.arange(21, dtype=float),
    })
    blocked_result = MathematicalDataCompiler().compile_many(
        {"left": many_left, "right": many_right}, target="left.指标A"
    )
    blocked = blocked_result["cross_dataset_contracts"][0]
    assert blocked["relationship"] == "many_to_many"
    assert blocked["status"] == "blocked"
    assert blocked_result["summary"]["blocked_cross_dataset_contracts"] == 1


def test_missing_sensitivity_never_imputes_target_and_noise_is_not_refuted():
    rng = np.random.default_rng(2026)
    size = 120
    target = rng.normal(size=size)
    predictor = rng.normal(size=size)
    target[[1, 7, 13, 19]] = np.nan
    predictor[[2, 8, 14, 20]] = np.nan
    frame = pd.DataFrame({
        "组别": np.resize(np.array(["A", "B", "C"]), size),
        "解释变量": predictor,
        "目标变量": target,
    })
    result = MathematicalDataCompiler().compile(frame, target="目标变量")
    relationship = next(
        item for item in result["conclusion_stress"]["relationships"]
        if item["predictor"] == "解释变量"
    )
    contexts = {item["view"]: item for item in relationship["contexts"]}

    assert contexts["median_imputed"]["n"] == int(pd.Series(target).notna().sum())
    assert relationship["status"] != "contradicted"
    assert relationship["global_significant_fdr_0_05"] is False


def test_identifier_and_join_type_inference_avoid_false_ids_and_normalize_values():
    frame = pd.DataFrame({
        "valid": [1.0, 2.0, 3.0],
        "customer_id": [1, 2, 3],
        "目标": [2.0, 4.0, 6.0],
    })
    contract = MathematicalDataCompiler().compile(frame, target="目标")["contract"]
    assert "valid" not in contract["technical_ids"]
    assert "customer_id" in contract["technical_ids"]

    left = pd.DataFrame({
        "customer_id": [1, 2, 3],
        "日期": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "指标A": [1.0, 2.0, 3.0],
    })
    right = pd.DataFrame({
        "customer_id": [1.0, 2.0, 3.0],
        "日期": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"], utc=True),
        "指标B": [4.0, 5.0, 6.0],
    })
    cross = MathematicalDataCompiler().compile_many(
        {"left": left, "right": right}, target="left.指标A"
    )["cross_dataset_contracts"][0]
    assert cross["relationship"] == "one_to_one"
    assert cross["overlap_coverage"] == 1.0


def test_cross_key_search_stops_after_strong_single_key():
    size = 200
    left = pd.DataFrame({
        "entity_id": np.arange(size),
        "地区": np.resize(np.array(["东", "西", "南", "北"]), size),
        "类别": np.resize(np.array(["A", "B", "C", "D", "E"]), size),
        "指标A": np.arange(size, dtype=float),
    })
    right = pd.DataFrame({
        "entity_id": np.arange(size),
        "地区": np.resize(np.array(["东", "西", "南", "北"]), size),
        "类别": np.resize(np.array(["A", "B", "C", "D", "E"]), size),
        "指标B": np.arange(size, dtype=float),
    })
    result = MathematicalDataCompiler().compile_many(
        {"left": left, "right": right}, target="left.指标A"
    )
    cross = result["cross_dataset_contracts"][0]
    assert cross["key_pairs"] == [{"left": "entity_id", "right": "entity_id"}]
    assert cross["key_candidates_evaluated"] <= 3
    assert result["summary"]["timing_ms"]["multi_table_total"] >= 0


def test_high_cardinality_join_overlap_is_corrected_for_independent_samples():
    size = 5_000
    left = pd.DataFrame({
        "entity_id": np.arange(size),
        "指标A": np.arange(size, dtype=float),
    })
    right = pd.DataFrame({
        "entity_id": np.random.default_rng(9).permutation(np.arange(size)).astype(float),
        "指标B": np.arange(size, dtype=float),
    })
    result = MathematicalDataCompiler(max_analysis_rows=1_000).compile_many(
        {"left": left, "right": right}, target="left.指标A"
    )
    cross = result["cross_dataset_contracts"][0]

    assert cross["relationship"] == "one_to_one"
    assert cross["capture_recapture_used"] is True
    assert cross["overlap_coverage"] >= 0.8
    assert cross["full_cardinality_reaudit_required"] is True
    assert cross["status"] == "restricted"


def test_grain_inference_supports_three_dimensions_and_labels_near_unique_keys():
    frame = pd.DataFrame([
        {
            "地区": f"R{region}",
            "产品": f"P{product}",
            "方案": f"S{scenario}",
            "目标值": float(region + product + scenario),
        }
        for region in range(5)
        for product in range(5)
        for scenario in range(4)
    ])
    contract = MathematicalDataCompiler().compile(frame, target="目标值")["contract"]
    assert contract["observed_grain"] == ["地区", "产品", "方案"]
    assert contract["grain_status"] == "verified_unique"

    entities = [f"E{index}" for index in range(100)]
    entities[-1] = entities[-2]
    near_unique = pd.DataFrame({
        "实体": entities,
        "目标值": np.arange(100, dtype=float),
    })
    near_contract = MathematicalDataCompiler().compile(
        near_unique, target="目标值"
    )["contract"]
    assert near_contract["observed_grain"] == ["实体"]
    assert near_contract["grain_status"] == "near_unique_candidate"


def test_explicit_semantic_contract_overrides_names_without_bypassing_audits():
    frame = pd.DataFrame({
        "when_raw": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "where_raw": [101, 102, 103],
        "metric_raw": [2.0, 3.0, 4.0],
    })
    result = MathematicalDataCompiler().compile(
        frame,
        semantic_hints={
            "target": "metric_raw",
            "grain": ["when_raw", "where_raw"],
            "columns": {
                "when_raw": {"role": "time", "semantic_id": "event_time"},
                "where_raw": {"role": "dimension", "semantic_id": "site"},
                "metric_raw": {
                    "role": "measure", "unit": "kg", "additivity": "additive",
                },
            },
        },
    )
    contract = result["contract"]
    semantics = {item["column"]: item for item in contract["columns_semantics"]}
    assert result["schema_version"] == "mathmodel.data-compilation/v2"
    assert contract["target"] == "metric_raw"
    assert contract["observed_grain"] == ["when_raw", "where_raw"]
    assert contract["grain_status"] == "verified_unique"
    assert contract["time_columns"] == ["when_raw"]
    assert contract["dimensions"] == ["where_raw"]
    assert semantics["metric_raw"]["unit"] == "kg"
    assert semantics["metric_raw"]["additivity"] == "additive"
    assert semantics["metric_raw"]["semantic_source"] == "explicit_hint"


def test_semantic_ids_join_different_column_names_and_auto_alias_is_restricted():
    dates = pd.date_range("2025-01-01", periods=4, freq="D")
    left = pd.DataFrame([
        {"客户主键": customer, "发生日": date, "指标A": float(customer)}
        for customer in (1, 2, 3)
        for date in dates
    ])
    right = pd.DataFrame([
        {"对象代码": float(customer), "记录时间": str(date.date()), "指标B": float(customer * 2)}
        for customer in (1, 2, 3)
        for date in dates
    ])
    hinted = MathematicalDataCompiler().compile_many(
        {"left": left, "right": right},
        target="left.指标A",
        semantic_hints={"datasets": {
            "left": {"columns": {
                "客户主键": {"role": "technical_id", "semantic_id": "entity"},
                "发生日": {"role": "time", "semantic_id": "event_time"},
            }},
            "right": {"columns": {
                "对象代码": {"role": "technical_id", "semantic_id": "entity"},
                "记录时间": {"role": "time", "semantic_id": "event_time"},
            }},
        }},
    )["cross_dataset_contracts"][0]
    assert hinted["relationship"] == "one_to_one"
    assert len(hinted["key_pairs"]) == 2
    assert all(
        item["match_source"] == "explicit_semantic_id"
        for item in hinted["key_pairs"]
    )

    automatic = MathematicalDataCompiler().compile_many(
        {
            "left": pd.DataFrame({"customer_id": range(20), "指标A": range(20)}),
            "right": pd.DataFrame({"buyer_code": np.arange(20, dtype=float), "指标B": range(20)}),
        },
        target="left.指标A",
    )["cross_dataset_contracts"][0]
    assert automatic["relationship"] == "one_to_one"
    assert automatic["key_pairs"][0]["match_source"] == "value_overlap_inferred_alias"
    assert automatic["semantic_alias_reaudit_required"] is True
    assert automatic["status"] == "restricted"


def test_invalid_semantic_hints_fail_closed():
    frame = pd.DataFrame({"x": [1.0, 2.0], "y": [2.0, 3.0]})
    with np.testing.assert_raises_regex(Exception, "不存在字段"):
        MathematicalDataCompiler().compile(
            frame, semantic_hints={"columns": {"missing": {"role": "measure"}}}
        )
    with np.testing.assert_raises_regex(Exception, "semantic_id必须唯一"):
        MathematicalDataCompiler().compile(
            frame,
            semantic_hints={"columns": {
                "x": {"semantic_id": "metric"},
                "y": {"semantic_id": "METRIC"},
            }},
        )


def test_single_hinted_target_selects_primary_dataset_without_name_conventions():
    result = MathematicalDataCompiler().compile_many(
        {
            "wide_support": pd.DataFrame({
                "a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5],
            }),
            "opaque_primary": pd.DataFrame({
                "k": [1, 2, 3], "out": [0.1, 0.2, 0.3],
            }),
        },
        semantic_hints={"datasets": {
            "opaque_primary": {
                "target": "out",
                "grain": ["k"],
                "columns": {
                    "k": {"role": "technical_id", "semantic_id": "entity"},
                    "out": {"role": "measure"},
                },
            },
        }},
    )
    assert result["dataset"] == "opaque_primary"
    assert result["contract"]["target"] == "out"
