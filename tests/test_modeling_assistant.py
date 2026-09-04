from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from core.modeling_assistant import (
    DatasetRelation,
    InteractionFinding,
    MathModelingAssistant,
    run_modeling_study,
)
from core.problem_solver import analyze_problem


def _customer_order_data(seed=7, n_customers=160, n_orders=1200):
    rng = np.random.default_rng(seed)
    customer_effect = rng.normal(70, 9, n_customers)
    customers = pd.DataFrame({
        "customer_id": np.arange(n_customers),
        "region": rng.choice(["north", "south", "east", "west"], n_customers),
        "satisfaction": customer_effect,
    })
    order_customer = rng.integers(0, n_customers, n_orders)
    orders = pd.DataFrame({
        "customer_id": order_customer,
        "amount": customer_effect[order_customer] * 2 + rng.normal(0, 5, n_orders),
        "quantity": rng.integers(1, 6, n_orders),
    })
    return customers, orders


def test_discovers_one_to_many_relation_and_cross_table_interaction(tmp_path):
    customers, orders = _customer_order_data()
    assistant = MathModelingAssistant(output_dir=str(tmp_path), max_analysis_rows=5_000)
    result = assistant.run(
        "分析订单金额与客户满意度的关系，并预测 satisfaction",
        {"customers": customers, "orders": orders},
        target="customers.satisfaction",
        run_modeling=False,
        generate_plots=False,
    )

    relation = next(r for r in result.relationships if r.left_key == "customer_id")
    assert relation.relationship == "one_to_many"
    assert relation.confidence >= 90
    assert relation.safe_to_join
    strongest = result.interactions[0]
    assert {strongest.left_variable, strongest.right_variable} == {"satisfaction", "amount"}
    assert strongest.strength > 0.9
    assert Path(result.report_path).is_file()


def test_many_to_many_relation_is_flagged_before_join(tmp_path):
    left = pd.DataFrame({"group_id": np.repeat(np.arange(20), 5), "x": np.arange(100)})
    right = pd.DataFrame({"group_id": np.repeat(np.arange(20), 7), "y": np.arange(140)})
    assistant = MathModelingAssistant(output_dir=str(tmp_path))
    assistant._datasets = {"left": left, "right": right}
    assistant.profile_datasets("分析变量关系")

    relation = assistant.discover_relationships()[0]
    assert relation.relationship == "many_to_many"
    assert not relation.safe_to_join
    assert "先按关联键聚合" in relation.warning


def test_problem_analysis_returns_multi_task_and_web_compatibility_keys():
    result = analyze_problem("问题1：分析影响因素；问题2：预测未来销量；问题3：优化配送成本")
    assert len(result["task_candidates"]) >= 2
    assert len(result["subproblems"]) == 3
    assert result["formulas"]
    assert result["model"] == result["model_class"]
    assert result["approach"] == result["steps"]
    assert result["code_template"] == result["code_framework"]
    assert [node["task_type"] for node in result["task_graph"]] == [
        "statistical_inference", "prediction_forecast", "optimization",
    ]
    assert result["task_graph"][2]["depends_on"] == ["task_2"]


def test_one_subproblem_can_expand_into_a_composable_pipeline():
    result = analyze_problem("先进行主成分降维并检测异常值")

    assert [node["task_type"] for node in result["task_graph"]] == [
        "dimension_reduction", "anomaly_detection",
    ]
    assert result["task_graph"][1]["depends_on"] == ["task_1"]


def test_entropy_topsis_runs_for_evaluation_problem(tmp_path):
    cities = pd.DataFrame({
        "city_id": ["A", "B", "C", "D"],
        "income": [10.0, 14.0, 8.0, 12.0],
        "service": [70.0, 82.0, 65.0, 78.0],
        "pollution_cost": [9.0, 5.0, 11.0, 7.0],
    })
    result = run_modeling_study(
        "对城市发展质量进行综合评价和排名，成本及污染越低越好",
        {"cities": cities},
        output_dir=str(tmp_path),
        run_modeling=False,
        generate_plots=False,
    )
    ranking = result["ranking_result"]
    assert ranking["method"] == "entropy_weight_topsis"
    assert ranking["directions"]["pollution_cost"] == "negative"
    assert [row["rank"] for row in ranking["ranking"]] == [1, 2, 3, 4]
    assert ranking["credibility_audit"]["checks"][0]["id"] == "weight_sensitivity"
    assert 0 <= ranking["sensitivity"]["winner_retention"] <= 1
    assert ranking["pareto_analysis"]["front_size"] >= 1
    assert ranking["credibility_audit"]["checks"][1]["id"] == "pareto_tradeoff"


def test_automatic_supervised_model_uses_cross_table_features(tmp_path):
    customers, orders = _customer_order_data(n_customers=120, n_orders=800)
    assistant = MathModelingAssistant(output_dir=str(tmp_path), max_analysis_rows=3_000, feedback_optimization=False)
    result = assistant.run(
        "根据客户及订单数据预测 satisfaction",
        {"customers": customers, "orders": orders},
        target="customers.satisfaction",
        run_modeling=True,
        generate_plots=False,
    )
    model = result.model_result
    assert model is not None
    assert model["task_type"] == "regression"
    assert model["n_features"] > len(customers.columns) - 1
    assert model["best_model"]
    assert "r2" in model["metrics"]
    assert model["credibility_audit"]["enabled"]
    assert {check["id"] for check in model["credibility_audit"]["checks"]} >= {
        "validation_protocol", "target_leakage", "naive_baseline",
        "prediction_permutation", "fold_stability", "subgroup_error",
        "prediction_interval_coverage",
    }
    assert model["prediction_interval"]["target_coverage"] == 0.9


def test_generates_relationship_and_interaction_charts(tmp_path):
    customers, orders = _customer_order_data(n_customers=80, n_orders=400)
    assistant = MathModelingAssistant(output_dir=str(tmp_path), max_analysis_rows=2_000, feedback_optimization=False)
    result = assistant.run(
        "分析订单金额与 satisfaction 的交互作用",
        {"customers": customers, "orders": orders},
        target="customers.satisfaction",
        run_modeling=False,
        generate_plots=True,
    )
    chart_types = {chart["type"] for chart in result.charts}
    assert {"dataset_overview", "relationship_graph", "interaction_bar"} <= chart_types
    assert all(Path(chart["path"]).is_file() for chart in result.charts)


def test_clustering_problem_runs_without_target_and_selects_k(tmp_path):
    rng = np.random.default_rng(18)
    values = np.vstack([
        rng.normal((-4, -4), 0.35, size=(60, 2)),
        rng.normal((0, 4), 0.35, size=(60, 2)),
        rng.normal((4, -2), 0.35, size=(60, 2)),
    ])
    data = pd.DataFrame(values, columns=["indicator_a", "indicator_b"])
    assistant = MathModelingAssistant(output_dir=str(tmp_path), max_analysis_rows=2_000, feedback_optimization=False)
    result = assistant.run(
        "对样本进行聚类分组，分析各群体特征",
        {"samples": data},
        run_modeling=True,
        generate_plots=True,
    )
    assert result.model_result["task_type"] == "clustering"
    assert result.model_result["best_k"] == 3
    assert result.model_result["metrics"]["silhouette"] > 0.7
    assert result.model_result["credibility_audit"]["status"] == "pass"
    assert {check["id"] for check in result.model_result["credibility_audit"]["checks"]} == {
        "cluster_separation", "cluster_seed_stability", "cluster_size_balance",
    }
    assert any(chart["type"] == "clustering" for chart in result.charts)


def test_structure_analysis_finds_latent_dimension_and_bounds_anomaly_rate(tmp_path):
    rng = np.random.default_rng(19)
    n_rows = 600
    latent = rng.normal(size=n_rows)
    data = pd.DataFrame({
        "entity_id": np.arange(n_rows),
        "indicator_a": latent + rng.normal(0, 0.05, n_rows),
        "indicator_b": 2 * latent + rng.normal(0, 0.05, n_rows),
        "indicator_c": -latent + rng.normal(0, 0.05, n_rows),
        "independent": rng.normal(size=n_rows),
    })
    data.loc[[3, 17], ["indicator_a", "indicator_b", "indicator_c"]] = [12, -15, 20]
    result = MathModelingAssistant(
        output_dir=str(tmp_path), feedback_optimization=False,
    ).run(
        "问题1：提取主成分并降维；问题2：基于上述结构检测异常值",
        {"samples": data}, run_modeling=False, generate_plots=True,
    )

    structure = result.specialized_results["data_structure"][0]
    assert structure["dimensions_90"] < structure["original_dimensions"]
    assert 0 < structure["anomaly_count"] <= int(np.ceil(n_rows * 0.01))
    assert structure["anomaly_threshold_robust_z"] >= 3.5
    flagged_rows = {
        row["row_index"] for row in structure["top_anomalies"] if row["flagged"]
    }
    assert {"3", "17"} <= flagged_rows
    assert {check["id"] for check in structure["credibility_audit"]["checks"]} == {
        "sample_adequacy", "subspace_stability", "anomaly_perturbation",
    }
    assert any(chart["type"] == "data_structure" for chart in result.charts)
    assert "projection" not in structure
    assert [node["status"] for node in result.problem_analysis["task_graph"]] == [
        "executed", "executed",
    ]
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "多子问题执行图" in report
    assert "潜在结构与稳健异常" in report


def test_near_optimal_model_disagreement_is_explicitly_rejected(tmp_path):
    from core.modeling_engine import TaskType

    actual = pd.Series(np.linspace(-2, 2, 80))

    def cv_result(key, score, prediction):
        return SimpleNamespace(
            model_key=key,
            model_name=key,
            mean_scores={"rmse": score},
            fold_scores={"rmse": [score, score]},
            oof_pred=np.asarray(prediction),
        )

    assistant = MathModelingAssistant(output_dir=str(tmp_path))
    consistent = SimpleNamespace(cv_results=[
        cv_result("ridge", 1.0, actual + 0.01),
        cv_result("hist_gb", 1.05, actual + 0.02),
    ])
    consistent_check = assistant._model_hypothesis_check(
        consistent, actual, TaskType.REGRESSION, False,
    )
    assert consistent_check["status"] == "pass"

    contradictory = SimpleNamespace(cv_results=[
        cv_result("ridge", 1.0, actual),
        cv_result("hist_gb", 1.05, -actual.to_numpy()),
    ])
    contradictory_check = assistant._model_hypothesis_check(
        contradictory, actual, TaskType.REGRESSION, False,
    )
    assert contradictory_check["status"] == "fail"
    assert contradictory_check["details"]["minimum_prediction_spearman"] < 0


def test_integral_sparse_dynamics_recovers_known_driver_without_differentiation(tmp_path):
    rng = np.random.default_rng(21)
    time = np.arange(0, 60, 0.2)
    data = pd.DataFrame({
        "date": pd.Timestamp("2024-01-01") + pd.to_timedelta(time, unit="D"),
        "state": np.sin(0.2 * time) + rng.normal(0, 0.002, len(time)),
        "driver": np.cos(0.2 * time),
    })
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "从时间数据发现 state 的动力学微分方程",
        {"series": data}, target="series.state",
        run_modeling=False, generate_plots=True,
    )

    equation = result.specialized_results["equation_discovery"]
    assert equation["method"] == "derivative_free_integral_sparse_dynamics"
    assert equation["metrics"]["validation_r2"] > 0.98
    assert equation["metrics"]["validation_rmse"] < equation["metrics"]["baseline_rmse"] * 0.1
    assert equation["active_terms"][0]["term"] == "z(driver)"
    assert "validation_actual" not in equation
    assert any(chart["type"] == "equation_discovery" for chart in result.charts)
    assert result.problem_analysis["task_graph"][0]["status"] == "partial"


def test_integral_sparse_dynamics_rejects_structureless_noise(tmp_path):
    rng = np.random.default_rng(23)
    n_rows = 260
    data = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n_rows),
        "state": rng.normal(size=n_rows),
        "driver": rng.normal(size=n_rows),
    })
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "发现 state 的动力学微分方程",
        {"series": data}, target="series.state",
        run_modeling=False, generate_plots=False,
    )

    equation = result.specialized_results["equation_discovery"]
    assert equation["credibility_audit"]["status"] == "fail"
    assert equation["metrics"]["validation_rmse"] >= equation["metrics"]["baseline_rmse"] * 0.99


def test_cross_fitted_causal_effect_recovers_known_effect_and_keeps_assumption_warning(tmp_path):
    rng = np.random.default_rng(22)
    n_rows = 1_200
    confounder = rng.normal(size=n_rows)
    propensity = 1 / (1 + np.exp(-confounder))
    treatment = rng.binomial(1, propensity)
    outcome = 2.5 * treatment + 1.5 * confounder + rng.normal(size=n_rows)
    data = pd.DataFrame({
        "confounder": confounder,
        "treatment": treatment,
        "outcome": outcome,
    })
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "估计因果效应，处理变量=treatment，结果变量=outcome，使用处理前协变量",
        {"study": data}, target="study.outcome",
        run_modeling=False, generate_plots=True,
    )

    causal = result.specialized_results["causal_effect"]
    assert result.problem_analysis["task_type"] == "causal_inference"
    assert abs(causal["effect"] - 2.5) < 0.25
    assert causal["confidence_interval_95"][0] < 2.5 < causal["confidence_interval_95"][1]
    assert causal["placebo_p_value"] < 0.05
    assert causal["credibility_audit"]["label"] == "有条件可信"
    assert next(
        check for check in causal["credibility_audit"]["checks"]
        if check["id"] == "unobserved_confounding"
    )["status"] == "not_assessed"
    assert result.problem_analysis["task_graph"][0]["status"] == "executed"
    assert any(chart["type"] == "causal_effect" for chart in result.charts)


def test_causal_task_refuses_to_guess_treatment_direction(tmp_path):
    data = pd.DataFrame({
        "x": np.arange(100, dtype=float),
        "y": np.arange(100, dtype=float) * 2,
    })
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "分析 x 和 y 之间的因果效应",
        {"study": data}, target="study.y",
        run_modeling=False, generate_plots=False,
    )

    assert "causal_effect" not in result.specialized_results
    assert result.problem_analysis["task_graph"][0]["status"] == "needs_input"
    assert any("不会根据相关性自行指定因果方向" in warning for warning in result.warnings)


def test_forecast_target_uses_time_ordered_validation(tmp_path):
    rng = np.random.default_rng(22)
    n_rows = 120
    data = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n_rows, freq="D"),
        "price": rng.normal(20, 2, n_rows),
    })
    data["sales"] = 200 - 4 * data["price"] + np.arange(n_rows) * 0.3 + rng.normal(0, 2, n_rows)
    assistant = MathModelingAssistant(output_dir=str(tmp_path), max_analysis_rows=2_000, feedback_optimization=False)
    result = assistant.run(
        "根据日期和价格预测未来 sales 销量趋势",
        {"sales_history": data},
        target="sales_history.sales",
        run_modeling=True,
        generate_plots=False,
    )
    assert result.model_result["validation"] == "time_ordered_cv"
    assert result.specialized_results["time_dynamics"]["n_time_points"] == n_rows


def test_dirty_schema_and_nested_cells_degrade_without_aborting(tmp_path):
    dirty = pd.DataFrame(
        [[1, {"tag": "a"}, 10], [2, {"tag": "b"}, 20], [3, ["x", "y"], 30]],
        columns=["id", "metadata", "id"],
    )
    assistant = MathModelingAssistant(output_dir=str(tmp_path))
    result = assistant.run(
        "分析数据质量和变量关系",
        {"dirty": dirty, "empty": pd.DataFrame()},
        run_modeling=False,
        generate_plots=False,
    )
    assert result.dataset_profiles[0].n_columns == 3
    assert list(assistant._datasets["dirty"].columns) == ["id", "metadata", "id__2"]
    assert any("规范化" in warning for warning in result.warnings)
    assert any("为空" in warning for warning in result.warnings)


def test_discovers_composite_key_when_single_keys_are_many_to_many(tmp_path):
    dimension = pd.DataFrame({
        "region": np.repeat(["A", "B", "C"], 4),
        "year": np.tile([2021, 2022, 2023, 2024], 3),
        "policy_score": np.arange(12, dtype=float),
    })
    fact = pd.DataFrame({
        "region": np.repeat(np.repeat(["A", "B", "C"], 4), 5),
        "year": np.repeat(np.tile([2021, 2022, 2023, 2024], 3), 5),
        "sales": np.arange(60, dtype=float),
    })
    assistant = MathModelingAssistant(output_dir=str(tmp_path))
    assistant._datasets = {"dimension": dimension, "fact": fact}
    assistant.profile_datasets("分析各地区年度销量")
    relations = assistant.discover_relationships()
    composite = next(relation for relation in relations if len(relation.left_keys) == 2)
    assert composite.left_keys == ["region", "year"]
    assert composite.right_keys == ["region", "year"]
    assert composite.relationship == "one_to_many"


def test_mixed_type_and_nonlinear_interactions_use_appropriate_effect_sizes(tmp_path):
    rng = np.random.default_rng(31)
    n_entities = 240
    x = np.linspace(-3, 3, n_entities)
    segment = np.where(np.arange(n_entities) < n_entities // 2, "A", "B")
    entities = pd.DataFrame({
        "entity_id": np.arange(n_entities),
        "x": x,
        "segment": segment,
    })
    observations = pd.DataFrame({
        "entity_id": np.arange(n_entities),
        "quadratic_response": x ** 2 + rng.normal(0, 0.05, n_entities),
        "segment_response": np.where(segment == "A", 5.0, 25.0) + rng.normal(0, 0.2, n_entities),
        "channel": np.where(segment == "A", "online", "offline"),
    })
    assistant = MathModelingAssistant(output_dir=str(tmp_path), max_analysis_rows=2_000)
    result = assistant.run(
        "分析两个数据集之间的非线性与类别交互",
        {"entities": entities, "observations": observations},
        run_modeling=False,
        generate_plots=False,
    )
    methods = {finding.method for finding in result.interactions}
    assert "binned_normalized_mutual_information" in methods
    assert "correlation_ratio_eta_squared" in methods
    assert "bias_corrected_cramers_v" in methods
    corrected = [finding for finding in result.interactions if finding.p_value is not None]
    assert corrected and all(finding.q_value is not None for finding in corrected)
    assert any(finding.significant for finding in corrected)


def test_capability_report_distinguishes_execution_from_missing_assumptions(tmp_path):
    data = pd.DataFrame({"resource": [1, 2, 3], "cost": [5.0, 4.0, 3.0]})
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "优化资源分配，使总成本最小，但尚未给出约束",
        {"inputs": data},
        run_modeling=False,
        generate_plots=False,
    )
    optimization = next(
        item for item in result.capability_report["tasks"] if item["task_type"] == "optimization"
    )
    assert optimization["status"] == "planning_only"
    assert "目标函数" in optimization["requirement"]
    assert result.model_result is None


def test_graph_problem_executes_bounded_network_structure_analysis(tmp_path):
    edges = pd.DataFrame({
        "source": ["A", "A", "B", "C", "D"],
        "target": ["B", "C", "C", "D", "A"],
        "distance": [2.0, 5.0, 1.0, 3.0, 4.0],
    })
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "建立有向交通网络，分析节点和最短路径",
        {"roads": edges},
        run_modeling=False,
        generate_plots=False,
    )
    graph = result.specialized_results["graph_network"]
    assert graph["n_nodes"] == 4
    assert graph["n_unique_edges"] == 5
    assert graph["connected_components"] == 1
    assert graph["weight_column"] == "distance"
    capability = next(item for item in result.capability_report["tasks"] if item["task_type"] == "graph_network")
    assert capability["status"] == "executed"


def test_simulation_problem_reports_bootstrap_uncertainty_without_claiming_mechanism(tmp_path):
    rng = np.random.default_rng(44)
    losses = pd.DataFrame({"loss": rng.lognormal(2.0, 0.4, 600)})
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "使用蒙特卡洛模拟分析 loss 损失的不确定性和风险",
        {"risk_samples": losses},
        run_modeling=False,
        generate_plots=False,
    )
    simulation = result.specialized_results["simulation"]
    assert simulation["method"] == "nonparametric_bootstrap"
    assert simulation["iterations"] == 1000
    lower, upper = simulation["mean_confidence_interval_95"]
    assert lower < simulation["observed_mean"] < upper
    capability = next(item for item in result.capability_report["tasks"] if item["task_type"] == "simulation")
    assert capability["status"] == "partial"
    assert "机理" in capability["requirement"]


def test_multiple_targets_are_modeled_independently(tmp_path):
    rng = np.random.default_rng(52)
    n_rows = 110
    feature = rng.normal(size=n_rows)
    data = pd.DataFrame({
        "feature": feature,
        "sales": 3 * feature + rng.normal(0, 0.2, n_rows),
        "profit": -2 * feature + rng.normal(0, 0.2, n_rows),
    })
    result = MathModelingAssistant(output_dir=str(tmp_path), max_analysis_rows=2_000, feedback_optimization=False).run(
        "分别预测 sales 销量和 profit 利润，并比较影响因素",
        {"business": data},
        run_modeling=True,
        generate_plots=True,
    )
    assert [model["target"] for model in result.model_results] == ["sales", "profit"]
    assert result.model_result == result.model_results[0]
    assert all(model["best_model"] for model in result.model_results)
    assert sum("完成自动regression" in conclusion for conclusion in result.conclusions) == 2
    validation_charts = [chart for chart in result.charts if chart["type"] == "model_validation"]
    assert {chart["target"] for chart in validation_charts} == {"sales", "profit"}


def test_auto_target_context_does_not_model_mentioned_predictors(tmp_path):
    rng = np.random.default_rng(53)
    n_rows = 100
    temperature = rng.normal(size=n_rows)
    rainfall = rng.normal(size=n_rows)
    data = pd.DataFrame({
        "temperature": temperature,
        "rainfall": rainfall,
        "sales": 2 * temperature - rainfall + rng.normal(0, 0.2, n_rows),
    })
    result = MathModelingAssistant(output_dir=str(tmp_path), feedback_optimization=False).run(
        "根据 temperature 和 rainfall 预测 sales",
        {"observations": data},
        run_modeling=True,
        generate_plots=False,
    )
    assert [model["target"] for model in result.model_results] == ["sales"]


def test_repeated_numeric_codes_never_enter_metric_geometry_or_target_selection(tmp_path):
    n_rows = 90
    sales = pd.DataFrame({
        "销售日期": pd.date_range("2024-01-01", periods=n_rows, freq="D"),
        "单品编码": np.resize([16400001, 16400002, 16400003], n_rows),
        "分类编码": np.resize([101, 102, 103], n_rows),
        "销量(千克)": np.linspace(10, 30, n_rows),
        "销售单价(元/千克)": np.linspace(5, 8, n_rows),
    })
    loss = pd.DataFrame({
        "单品编码": [16400001, 16400002, 16400003],
        "损耗率(%)": [4.0, 8.0, 12.0],
    })
    assistant = MathModelingAssistant(output_dir=str(tmp_path), feedback_optimization=False)
    assistant._datasets = {"附件2.xlsx::Sheet1": sales, "附件4.xlsx::Sheet1": loss}
    profiles = assistant.profile_datasets(
        "预测各品类未来一周日销售量并优化补货；损耗率只用于成本修正"
    )

    sales_profile = next(item for item in profiles if item.name == "附件2.xlsx::Sheet1")
    assert {"单品编码", "分类编码"} <= set(sales_profile.id_candidates)
    assert "单品编码" not in sales_profile.numeric_columns
    assert "分类编码" not in sales_profile.numeric_columns
    assert assistant._select_targets(
        None, "预测各品类未来一周日销售量并优化补货；损耗率只用于成本修正"
    ) == [("附件2.xlsx::Sheet1", "销量(千克)")]

    dynamics = assistant._run_time_dynamics(None)
    assert dynamics["variable"] == "销量(千克)"
    structures = assistant._run_data_structure_analysis()
    assert all(
        column not in {"单品编码", "分类编码"}
        for result in structures
        for column in result["features"]
    )


def test_data_collection_question_is_executable_audit_not_blocked_optimization(tmp_path):
    data = pd.DataFrame({
        "日期": pd.date_range("2024-01-01", periods=40),
        "商品编码": np.resize([1, 2, 3, 4], 40),
        "销量": np.linspace(10, 20, 40),
        "售价": np.linspace(5, 7, 40),
    })
    problem = (
        "问题1：预测未来销量。问题2：制定补货与定价方案。"
        "问题3：为了更好地解决上述问题，还需要采集哪些数据？"
        "这些数据有什么帮助？请给出意见和理由。"
    )
    result = MathModelingAssistant(
        output_dir=str(tmp_path), feedback_optimization=False,
    ).run(
        problem, {"sales": data}, run_modeling=False, generate_plots=False,
    )

    audit_node = next(
        node for node in result.problem_analysis["task_graph"]
        if node["task_type"] == "data_requirements"
    )
    assert audit_node["status"] == "executed"
    assert audit_node["depends_on"] == []
    audit = result.specialized_results["data_requirements"]
    assert audit["recommendations"]
    assert all(
        {"data_role", "reason", "collection_design", "gap_source"} <= set(item)
        for item in audit["recommendations"]
    )
    assert result.mathematical_model_spec["readiness_by_track"]["mechanistic_structure"] == "not_applicable"
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "数据需求与可识别性审计" in report


def test_grouped_forecast_compiles_requested_grain_before_sampling(tmp_path):
    rng = np.random.default_rng(91)
    dates = pd.date_range("2024-01-01", periods=84, freq="D")
    item_codes = np.arange(1001, 1009)
    master = pd.DataFrame({
        "单品编码": item_codes,
        "分类名称": np.repeat(["叶菜", "根茎"], 4),
    })
    rows = []
    for day_index, date in enumerate(dates):
        for item_index, item in enumerate(item_codes):
            category_level = 8.0 if item_index < 4 else 13.0
            weekday = 2.0 if date.dayofweek >= 5 else 0.0
            rows.append({
                "销售日期": date,
                "单品编码": item,
                "销量(千克)": max(
                    0.1, category_level + weekday + 0.03 * day_index + rng.normal(0, 0.4)
                ),
                "销售单价": 6.0 + item_index * 0.1,
            })
    sales = pd.DataFrame(rows)
    result = MathModelingAssistant(
        output_dir=str(tmp_path), max_analysis_rows=100,
        feedback_optimization=False,
    ).run(
        "预测各蔬菜品类未来一周的日销售总量，并给出预测区间",
        {"销售明细": sales, "商品主数据": master},
        run_modeling=True, generate_plots=False,
    )

    grouped = result.specialized_results["grouped_forecast"]
    assert grouped["aggregation"] == "daily_sum_before_any_sampling"
    assert grouped["source_rows_aggregated"] == len(sales)
    assert grouped["groups_forecast"] == 2
    assert grouped["horizon_days"] == 7
    assert len(grouped["forecasts"]) == 14
    assert grouped["group_column"] == "分类名称"
    assert grouped["dimension_join_audit"]["validation"] == "many_to_one"
    assert grouped["metrics"]["terminal_block_rmse"] <= grouped["metrics"]["seasonal_naive_rmse"] + 1e-12
    assert result.model_results == []
    forecast_node = next(
        node for node in result.problem_analysis["task_graph"]
        if node["task_type"] == "prediction_forecast"
    )
    assert forecast_node["status"] == "executed"
    assert any(
        claim["claim_type"] == "predictive" and "日×分类名称" in claim["statement"]
        for claim in result.evidence_bundle["claims"]
    )
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "分组时间粒度预测" in report


def test_forecast_feedback_compiles_to_generic_milp_decision(tmp_path):
    rng = np.random.default_rng(92)
    dates = pd.date_range("2024-01-01", periods=70, freq="D")
    item_codes = np.arange(2001, 2007)
    master = pd.DataFrame({
        "单品编码": item_codes,
        "分类名称": np.repeat(["A类", "B类"], 3),
        "单位成本": [3.0, 3.2, 3.1, 4.0, 4.2, 4.1],
        "损耗率(%)": [5.0, 6.0, 5.5, 8.0, 7.5, 8.5],
    })
    rows = []
    for day_index, date in enumerate(dates):
        for item_index, item in enumerate(item_codes):
            rows.append({
                "销售日期": date,
                "单品编码": item,
                "销量(千克)": max(
                    0.1, 9 + (item_index >= 3) * 4 + date.dayofweek * 0.2
                    + day_index * 0.02 + rng.normal(0, 0.25)
                ),
                "销售单价": 6.0 + (item_index >= 3) * 1.5,
            })
    result = MathModelingAssistant(
        output_dir=str(tmp_path), max_analysis_rows=100,
        feedback_optimization=False,
    ).run(
        "预测各商品品类未来一周的日销售总量，并制定补货和定价策略，使收益最大",
        {"销售明细": pd.DataFrame(rows), "商品主数据": master},
        run_modeling=True, generate_plots=False,
    )

    decision = result.specialized_results["prescriptive_decision"]
    assert decision["mathematical_form"] == "multiple_choice_mixed_integer_linear_program"
    assert decision["compiled_contract_summary"]["parse_status"] == "machine_verified"
    assert decision["solver_result"]["solver"] == "scipy.milp.highs"
    assert decision["decision_count"] == 14
    assert decision["cost_coverage"] == 1.0
    assert decision["loss_coverage"] == 1.0
    stress = decision["risk_aware_stress_test"]
    assert stress["status"] == "pass"
    assert stress["weights_are_calibrated_probabilities"] is False
    assert stress["adopted"] is False
    assert np.isfinite(stress["risk_aware_selection"]["lower_tail_cvar"])
    assert any(
        check["id"] == "scenario_cvar_stress_test"
        and check["status"] == "pass"
        for check in decision["credibility_audit"]["checks"]
    )
    assert all(row["replenishment"] >= row["forecast_demand"] for row in decision["decision_rows"])
    assert all(
        row["lower_replenishment_90"] <= row["replenishment"] <= row["upper_replenishment_90"]
        for row in decision["decision_rows"]
    )
    optimization_node = next(
        node for node in result.problem_analysis["task_graph"]
        if node["task_type"] == "optimization"
    )
    assert optimization_node["status"] == "partial"
    assert any(
        claim["claim_type"] == "optimization" and "条件性补货/价格候选" in claim["statement"]
        for claim in result.evidence_bundle["claims"]
    )


def test_multi_subproblem_grains_compile_category_and_item_decisions(tmp_path):
    rng = np.random.default_rng(923)
    dates = pd.date_range("2023-04-01", "2023-06-30", freq="D")
    item_codes = np.arange(3001, 3041)
    master = pd.DataFrame({
        "单品编码": item_codes,
        "分类名称": np.repeat(["叶菜", "根茎", "茄类", "食用菌"], 10),
        "单位成本": np.linspace(2.5, 5.0, len(item_codes)),
        "损耗率(%)": np.linspace(3.0, 12.0, len(item_codes)),
    })
    rows = []
    wholesale_rows = []
    for day_index, date in enumerate(dates):
        for item_index, item in enumerate(item_codes):
            base = 1.2 + (item_index % 10) * 0.08 + (date.dayofweek >= 5) * 0.3
            rows.append({
                "销售日期": date,
                "单品编码": item,
                "销量(千克)": max(0.05, base + 0.002 * day_index + rng.normal(0, 0.08)),
                "销售单价(元/千克)": 6.0 + (item_index % 10) * 0.2 + 0.1 * np.sin(day_index / 5),
            })
            wholesale_rows.append({
                "销售日期": date,
                "单品编码": item,
                "批发价格(元/千克)": (
                    2.5 + item_index * 0.05 + 0.08 * np.cos(day_index / 6)
                ),
            })
    problem = (
        "问题1：分析蔬菜各品类及单品销售量的分布规律及相互关系。"
        "问题2：考虑商超以品类为单位做补货计划，预测各蔬菜品类未来一周"
        "（2023年7月1-7日）的日销售总量，并给出补货总量和定价策略，使收益最大。"
        "问题3：根据2023年6月24-30日的可售品种，给出7月1日的单品补货量和定价策略，"
        "可售单品总数控制在27-33个，各单品最小陈列量2.5千克，使收益最大。"
    )
    result = MathModelingAssistant(
        output_dir=str(tmp_path), max_analysis_rows=120,
        feedback_optimization=False,
    ).run(
        problem,
        {
            "销售明细": pd.DataFrame(rows),
            "商品主数据": master,
            "批发价格": pd.DataFrame(wholesale_rows),
        },
        run_modeling=True, generate_plots=True,
    )

    forecasts = result.specialized_results["grouped_forecasts"]
    assert {item["requested_grain"] for item in forecasts} == {"category", "item"}
    category = next(item for item in forecasts if item["requested_grain"] == "category")
    item = next(item for item in forecasts if item["requested_grain"] == "item")
    assert category["target"] == "销量(千克)"
    assert category["forecast_period"] == ["2023-07-01", "2023-07-07"]
    assert category["horizon_days"] == 7
    assert item["forecast_period"] == ["2023-07-01", "2023-07-01"]
    assert item["horizon_days"] == 1
    assert result.model_results == []
    hierarchy = result.specialized_results["hierarchical_sales"]
    assert hierarchy["source_rows_aggregated"] == len(rows)
    assert len(hierarchy["category_summary"]) == 4
    assert hierarchy["concentration"]["item_count"] == 40

    decisions = result.specialized_results["prescriptive_decisions"]
    assert {item["requested_grain"] for item in decisions} == {"category", "item"}
    category_decision = next(item for item in decisions if item["requested_grain"] == "category")
    item_decision = next(item for item in decisions if item["requested_grain"] == "item")
    assert category_decision["cost_dataset"] == "批发价格"
    assert category_decision["cost_plus_pricing_tested_groups"] == 4
    assert category_decision["risk_aware_stress_test"]["status"] == "pass"
    assert all(
        np.isfinite(row["q_value"])
        for row in category_decision["cost_plus_pricing_relationship"]
    )
    assert any(
        check["id"] == "cost_plus_pricing_alignment"
        and check["status"] == "pass"
        for check in category_decision["credibility_audit"]["checks"]
    )
    assert item_decision["selection_bounds"] == [27, 33]
    assert item_decision["minimum_display"] == 2.5
    assert 0.0 <= item_decision["cost_coverage"] <= 1.0
    assert 0.0 <= item_decision["loss_coverage"] <= 1.0
    assert 27 <= item_decision["decision_count"] <= 33
    assert all(row["replenishment"] >= 2.5 for row in item_decision["decision_rows"])
    assert item_decision["solver_result"]["solver"] == "scipy.milp.highs"
    assert item_decision["hierarchical_lexicographic_stage_one"]["solver"] == "scipy.milp.highs"
    assert item_decision["hierarchical_lexicographic_verified"] is True
    assert len(item_decision["hierarchical_demand_coverage"]) == 4
    assert 0.0 <= item_decision["aggregate_parent_demand_coverage"] <= 1.0
    assert any(
        check["id"] == "parent_demand_coverage" and check["status"] == "pass"
        for check in item_decision["credibility_audit"]["checks"]
    )
    relevant_claims = [
        claim for claim in result.evidence_bundle["claims"]
        if claim["claim_type"] in {"association", "optimization"}
    ]
    assert relevant_claims
    assert all(claim["grade"] != "refuted" for claim in relevant_claims)
    assert any(
        claim["claim_type"] == "association"
        and "成本加成率—销量关系" in claim["statement"]
        for claim in result.evidence_bundle["claims"]
    )
    assert not any(
        claim["claim_type"] == "mechanistic_specification"
        for claim in result.evidence_bundle["claims"]
    )
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "分组时间粒度预测 · category" in report
    assert "分组时间粒度预测 · item" in report
    assert "上层品类需求覆盖审计" in report
    assert "成本加成率—销量关系审计（观察性）" in report
    assert "预测区间情景与下行风险审计" in report
    assert "纯题面通用数学 IR" not in report
    assert {"hierarchical_sales", "grouped_forecast", "prescriptive_decision"} <= {
        chart["type"] for chart in result.charts
    }
    assert "mechanistic_operator_graph" not in {chart["type"] for chart in result.charts}


def test_arbitrary_region_station_dimensions_compile_without_domain_keywords(tmp_path):
    dates = pd.date_range("2025-01-01", periods=70, freq="D")
    stations = [f"S{index:02d}" for index in range(8)]
    master = pd.DataFrame({
        "站点编号": stations,
        "站点名称": [f"节点{index:02d}" for index in range(8)],
        "区域名称": np.repeat(["北区", "南区"], 4),
    })
    observations = pd.DataFrame([
        {
            "观测日期": date,
            "站点编号": station,
            "需求量": 10.0 + station_index + date.dayofweek * 0.5,
        }
        for date in dates
        for station_index, station in enumerate(stations)
    ])
    result = MathModelingAssistant(
        output_dir=str(tmp_path), feedback_optimization=False,
    ).run(
        "分析各区域和各站点需求量的分布与相互关系，并预测各区域和各站点未来7天的日需求总量",
        {"观测记录": observations, "层级主数据": master},
        run_modeling=True, generate_plots=False,
    )

    forecasts = result.specialized_results["grouped_forecasts"]
    assert {item["group_column"] for item in forecasts} == {"区域名称", "站点名称"}
    assert all(item["horizon_days"] == 7 for item in forecasts)
    assert all(item["target"] == "需求量" for item in forecasts)
    assert all(item["requested_grain"].startswith("dimension:") for item in forecasts)
    assert {item["groups_forecast"] for item in forecasts} == {2, 8}
    hierarchy = result.specialized_results["hierarchical_distribution"]
    assert hierarchy["parent_dimension"] == "区域名称"
    assert hierarchy["child_dimension"] == "站点名称"
    assert len(hierarchy["parent_summary"]) == 2
    assert len(hierarchy["child_summary"]) == 8


def test_point_in_time_join_never_uses_future_records(tmp_path):
    base = pd.DataFrame({
        "entity_id": [1, 1, 1],
        "date": pd.to_datetime(["2024-01-02", "2024-01-04", "2024-01-06"]),
        "target": [1.0, 2.0, 3.0],
    })
    events = pd.DataFrame({
        "entity_id": [1, 1, 1],
        "event_time": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-10"]),
        "amount": [10.0, 20.0, 10_000.0],
    })
    assistant = MathModelingAssistant(output_dir=str(tmp_path), feedback_optimization=False)
    assistant._datasets = {"base": base, "events": events}
    assistant.profile_datasets("预测 target")
    assistant._relationships_cache = [DatasetRelation(
        left_dataset="base", right_dataset="events",
        left_key="entity_id", right_key="entity_id",
        relationship="many_to_many", confidence=99.0,
        name_similarity=1.0, value_overlap=1.0,
        left_coverage=1.0, right_coverage=1.0,
        estimated_join_rows=9, safe_to_join=False,
        left_keys=["entity_id"], right_keys=["entity_id"],
    )]
    view = assistant._build_modeling_view(
        "base", "target", temporal=True, base_time_column="date"
    )
    assert view["events__amount__sum_asof"].tolist() == [10.0, 30.0, 30.0]
    assert view.attrs["feature_join_audit"][0]["strategy"] == "point_in_time"


def test_fdr_is_computed_over_all_tested_hypotheses():
    findings = [
        InteractionFinding("a", "b", "x", "y", "test", 0.5, 100, "positive", "", p_value=p)
        for p in (0.01, 0.02, 0.5)
    ]
    MathModelingAssistant._apply_fdr(findings)
    assert [finding.q_value for finding in findings] == [0.03, 0.03, 0.5]


def test_numeric_interaction_reports_conditional_strength_and_stability(tmp_path):
    rng = np.random.default_rng(61)
    n_rows = 300
    confounder = rng.normal(size=n_rows)
    left = pd.DataFrame({
        "entity_id": np.arange(n_rows),
        "x": confounder + rng.normal(0, 0.15, n_rows),
        "confounder": confounder,
    })
    right = pd.DataFrame({
        "entity_id": np.arange(n_rows),
        "y": confounder + rng.normal(0, 0.15, n_rows),
    })
    result = MathModelingAssistant(output_dir=str(tmp_path), feedback_optimization=False).run(
        "分析多个数据集之间变量的统计交互",
        {"left": left, "right": right},
        run_modeling=False,
        generate_plots=False,
    )
    finding = next(item for item in result.interactions if {item.left_variable, item.right_variable} == {"x", "y"})
    assert finding.strength > 0.8
    assert abs(finding.conditional_strength) < 0.3
    assert finding.stability_score > 0.7


def test_feedback_optimization_never_replaces_baseline_with_worse_cv_result(tmp_path):
    rng = np.random.default_rng(62)
    x = rng.uniform(-3, 3, 100)
    data = pd.DataFrame({"x": x, "target": x ** 2 + rng.normal(0, 0.3, len(x))})
    result = MathModelingAssistant(
        output_dir=str(tmp_path),
        feedback_optimization=True,
        feedback_trials=2,
    ).run(
        "根据 x 预测 target",
        {"data": data},
        target="data.target",
        run_modeling=True,
        generate_plots=False,
    )
    model = result.model_result
    feedback = model["feedback_optimization"]
    assert feedback["attempted"]
    if feedback["accepted"]:
        assert model["metrics"]["rmse"] == feedback["tuned_score"]
        assert feedback["tuned_score"] < feedback["baseline_score"]
    else:
        assert model["metrics"]["rmse"] == feedback["baseline_score"]


def test_feedback_for_forecast_uses_last_time_block_as_confirmation(tmp_path):
    rng = np.random.default_rng(63)
    n_rows = 100
    time_index = np.arange(n_rows, dtype=float)
    data = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n_rows, freq="D"),
        "driver": np.sin(time_index / 8),
        "target": 0.08 * time_index + np.sin(time_index / 8) + rng.normal(0, 0.1, n_rows),
    })
    result = MathModelingAssistant(
        output_dir=str(tmp_path),
        feedback_optimization=True,
        feedback_trials=2,
    ).run(
        "根据 date 和 driver 预测 target 的未来趋势",
        {"series": data},
        target="series.target",
        run_modeling=True,
        generate_plots=False,
    )

    model = result.model_result
    assert model["validation"] == "temporal_holdout_after_inner_cv"
    assert model["fit_samples"] == 80
    assert model["confirmation_samples"] == 20
    assert model["feedback_optimization"]["confirmation"] == "末段时间留出"
    assert "独立确认集" in model["note"]


def test_many_dataset_relation_candidates_are_cached_and_interactions_are_bounded(tmp_path):
    datasets = {
        f"table_{index}": pd.DataFrame({
            "entity_id": np.arange(40),
            f"value_{index}": np.arange(40, dtype=float) * (index + 1),
        })
        for index in range(5)
    }
    assistant = MathModelingAssistant(
        output_dir=str(tmp_path),
        max_interaction_pairs=2,
    )
    calls = {}
    original = assistant._relation_candidates

    def counted(dataset_name):
        calls[dataset_name] = calls.get(dataset_name, 0) + 1
        return original(dataset_name)

    assistant._relation_candidates = counted
    result = assistant.run(
        "分析多个数据集之间的变量交互性",
        datasets,
        run_modeling=False,
        generate_plots=False,
    )
    assistant.discover_relationships()
    assert calls == {name: 1 for name in datasets}
    assert any("避免数据集数量平方增长" in warning for warning in result.warnings)


def test_custom_analyzer_is_extensible_and_failure_is_isolated(tmp_path):
    data = pd.DataFrame({"x": [1.0, 2.0, 3.0], "cost": [3.0, 2.0, 1.0]})
    assistant = MathModelingAssistant(output_dir=str(tmp_path / "success"))
    assistant.register_analyzer(
        "optimization",
        lambda **context: {"solver": "domain_solver", "objective": float(context["datasets"]["data"]["cost"].min())},
    )
    result = assistant.run(
        "优化资源配置，使成本最小",
        {"data": data},
        run_modeling=False,
        generate_plots=False,
    )
    assert result.specialized_results["custom"]["optimization"]["objective"] == 1.0
    capability = next(item for item in result.capability_report["tasks"] if item["task_type"] == "optimization")
    assert capability["status"] == "executed"

    failing = MathModelingAssistant(output_dir=str(tmp_path / "failure"))

    def broken_analyzer(**_):
        raise RuntimeError("intentional failure")

    failing.register_analyzer("optimization", broken_analyzer)
    degraded = failing.run(
        "优化资源配置，使成本最小",
        {"data": data},
        run_modeling=False,
        generate_plots=False,
    )
    assert Path(degraded.report_path).is_file()
    assert any("扩展分析器" in warning and "已隔离" in warning for warning in degraded.warnings)


def test_credibility_audit_rejects_a_target_copy_even_when_score_is_high(tmp_path):
    rng = np.random.default_rng(71)
    target = rng.normal(size=120)
    data = pd.DataFrame({
        "ordinary_feature": rng.normal(size=120),
        "target_copy": target,
        "target": target,
    })
    result = MathModelingAssistant(
        output_dir=str(tmp_path),
        feedback_optimization=False,
        credibility_iterations=50,
    ).run(
        "根据变量预测 target",
        {"data": data},
        target="data.target",
        generate_plots=False,
    )

    audit = result.model_result["credibility_audit"]
    leakage = next(check for check in audit["checks"] if check["id"] == "target_leakage")
    assert result.model_result["metrics"]["r2"] > 0.9
    assert leakage["status"] == "fail"
    assert leakage["details"]["suspected_features"][0]["feature"] == "target_copy"
    assert audit["status"] == "fail"
    assert "结果可信度审计" in Path(result.report_path).read_text(encoding="utf-8")


def test_credibility_audit_rejects_predictions_that_do_not_beat_chance(tmp_path):
    rng = np.random.default_rng(72)
    actual = np.linspace(-1.0, 1.0, 100)
    prediction = np.zeros(100)
    X = pd.DataFrame({"feature": rng.normal(size=100)})
    assistant = MathModelingAssistant(
        output_dir=str(tmp_path), credibility_iterations=50
    )
    audit = assistant._audit_model_credibility(
        X_fit=X,
        y_fit=pd.Series(actual),
        X_evaluation=X.copy(),
        actual=actual,
        prediction=prediction,
        task="regression",
        target="target",
        validation="holdout_after_inner_cv",
        use_time_validation=False,
        group_column=None,
        diagnostics={"primary_metric": "rmse", "fold_relative_std": 0.05},
        join_audit=[],
        feature_importance=[],
        engine=None,
    )

    statuses = {check["id"]: check["status"] for check in audit["checks"]}
    assert statuses["naive_baseline"] == "fail"
    assert statuses["prediction_permutation"] == "fail"
    assert audit["status"] == "fail"


def test_repeated_entity_key_automatically_uses_group_validation(tmp_path):
    rng = np.random.default_rng(73)
    entity_id = np.repeat(np.arange(40), 4)
    feature = rng.normal(size=len(entity_id))
    data = pd.DataFrame({
        "entity_id": entity_id,
        "feature": feature,
        "target": 2.0 * feature + rng.normal(size=len(feature)),
    })
    result = MathModelingAssistant(
        output_dir=str(tmp_path),
        feedback_optimization=True,
        feedback_trials=2,
        credibility_iterations=50,
    ).run(
        "预测 target",
        {"panel": data},
        target="panel.target",
        generate_plots=False,
    )

    model = result.model_result
    assert model["validation"] == "group_holdout_after_inner_cv"
    assert model["validation_group"] == "entity_id"
    validation_check = next(
        check for check in model["credibility_audit"]["checks"]
        if check["id"] == "validation_protocol"
    )
    assert validation_check["status"] == "pass"
    assert validation_check["details"]["group_overlap"] == 0
