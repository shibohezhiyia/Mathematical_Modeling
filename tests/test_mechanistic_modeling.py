"""Regression tests for the question-independent, no-dataset modeling path."""

from pathlib import Path
import math

import pytest
import pandas as pd

from core.mechanistic_modeling import (
    MechanisticModelingEngine,
    MechanisticOperatorRegistry,
    OperatorDefinition,
)
from core.modeling_assistant import MathModelingAssistant
from core.problem_solver import analyze_problem


def test_cross_domain_statements_use_the_same_operator_compiler():
    engine = MechanisticModelingEngine()
    cases = {
        "dynamic": (
            "某系统的状态 x 满足微分方程 dx/dt=-k*x，初值 x(0)=100，求其随时间的变化。",
            "first_order_ode",
        ),
        "geometry": (
            "移动体初始位置为(0,0,10)，以 5 m/s 匀速运动；当它进入半径 3 m 的区域时事件生效，求持续时间。",
            "region_membership",
        ),
        "network": (
            "给定节点和有向边，每条边有容量；选择流量使总流量最大，且中间节点满足流量守恒。",
            "network_flow",
        ),
    }
    for problem, required_operator in cases.values():
        result = engine.analyze(problem)
        assert result["schema_version"] == "mathmodel.mechanistic-ir/v2"
        assert required_operator in {node["key"] for node in result["operator_graph"]}
        assert result["input_policy"]["observations_invented"] is False
        assert result["execution_status"] == "needs_model_completion"
        assert result["compiler_plan"]["executable"] is False


def test_registry_extends_by_mathematical_primitive_not_question_adapter():
    registry = MechanisticOperatorRegistry()
    registry.register(OperatorDefinition(
        key="variational_energy",
        category="optimization",
        description="minimize an energy functional",
        required_bindings=("decision_variables", "objective", "constraints"),
        produces=("stationary_solution",),
        solver_route="variational_solver",
        triggers=(r"变分|能量泛函|variational",),
    ))
    result = MechanisticModelingEngine(registry).analyze(
        "选择函数 u，在边界约束下最小化能量泛函，建立变分模型。"
    )
    assert "variational_energy" in {node["key"] for node in result["operator_graph"]}
    with pytest.raises(ValueError, match="already registered"):
        registry.register(registry.get("variational_energy"))


def test_verified_structured_ode_executes_and_rechecks_tolerance():
    structured_ir = {
        "relations": [{
            "id": "ode_1",
            "kind": "ode_system",
            "state_variables": ["x"],
            "rhs": {"x": "-k*x"},
            "initial_values": {"x": 100.0},
            "parameters": {"k": 0.2},
            "time_variable": "t",
            "time_span": [0.0, 10.0],
            "output_points": 101,
            "units": {"x": "人", "k": "1/s", "t": "s"},
        }]
    }
    result = MechanisticModelingEngine().analyze(
        "状态 x 满足一阶微分方程，初值已给定，计算状态轨迹。",
        ir_override=structured_ir,
    )
    relation = result["mathematical_ir"]["relations"][0]
    assert relation["parse_status"] == "machine_verified"
    assert result["execution_status"] == "executed"
    trajectory = result["numerical_results"][0]
    assert trajectory["convergence"]["status"] == "pass"
    assert trajectory["summary"]["x"]["final"] == pytest.approx(100.0 * math.exp(-2), rel=2e-5)
    assert len(trajectory["plot_data"]["time"]) == 101


def test_four_layer_contract_selects_solver_by_mathematical_structure():
    structured_ir = {
        "relations": [{
            "id": "population_decay",
            "kind": "ode_system",
            "state_variables": ["x"],
            "rhs": {"x": "-k*x"},
            "initial_values": {"x": 12.0},
            "parameters": {"k": 0.3},
            "time_variable": "t",
            "time_span": [0.0, 4.0],
            "output_points": 61,
            "units": {"x": "人", "k": "1/s", "t": "s"},
        }]
    }
    result = MechanisticModelingEngine().analyze(
        "状态满足一阶微分方程并给出初值，计算轨迹。",
        ir_override=structured_ir,
    )
    pipeline = result["four_layer_pipeline"]
    assert pipeline["schema_version"] == "mathmodel.four-layer-pipeline/v1"
    assert pipeline["semantic_contract"]["schema_version"] == "mathmodel.semantic-contract/v1"
    assert pipeline["mathematical_ir"]["schema_version"] == "mathmodel.unified-ir/v1"
    math_node = pipeline["mathematical_ir"]["nodes"][0]
    assert math_node["mathematical_form"] == "initial_value_problem"
    assert math_node["status"] == "executable"
    plan = pipeline["solver_plan"]
    assert plan["selection_rule"] == "mathematical_form_only"
    assert plan["nodes"][0]["executor_key"] == "adaptive_ode/v1"
    assert plan["nodes"][0]["resource_budget"]["max_evaluations"] == 250_000
    independent = pipeline["independent_audit"]
    assert independent["status"] == "pass"
    assert independent["coverage"]["complete"] is True
    assert result["numerical_results"][0]["independent_audit"]["grade"] == "supported"


def test_four_layer_execution_isolates_a_failing_math_node():
    structured_ir = {
        "relations": [
            {
                "id": "valid_ode", "kind": "ode_system",
                "state_variables": ["x"], "rhs": {"x": "-k*x"},
                "initial_values": {"x": 2.0}, "parameters": {"k": 0.1},
                "time_variable": "t", "time_span": [0.0, 1.0], "output_points": 40,
                "units": {"x": "1", "k": "1/s", "t": "s"},
            },
            {
                "id": "singular_program", "kind": "optimization_problem",
                "decision_variables": ["z"], "objective": "1/(z-a)",
                "direction": "minimize", "parameters": {"a": 0.0},
                "bounds": {"z": [-1.0, 1.0]}, "initial_values": {"z": 0.0},
                "constraints": [], "units": {"z": "1", "a": "1"},
                "multistart_trials": 4,
            },
        ]
    }
    result = MechanisticModelingEngine().analyze(
        "计算一个动力系统，并求解一个有界优化问题。", ir_override=structured_ir,
    )
    execution = result["solver_execution"]
    assert execution["status"] == "partially_executed"
    assert len(execution["results"]) == 1
    assert execution["results"][0]["relation_id"] == "valid_ode"
    assert len(execution["failures"]) == 1
    assert execution["failures"][0]["relation_id"] == "singular_program"
    assert execution["failure_policy"] == "isolate_node_and_continue"
    assert result["four_layer_pipeline"]["independent_audit"]["status"] == "warning"


def test_unsafe_structured_expression_is_rejected_without_execution():
    structured_ir = {
        "relations": [{
            "kind": "ode_system",
            "state_variables": ["x"],
            "rhs": {"x": "__import__('os').system('echo unsafe')"},
            "initial_values": {"x": 1.0},
            "parameters": {},
            "time_variable": "t",
            "time_span": [0.0, 1.0],
            "units": {"x": "1", "t": "s"},
        }]
    }
    result = MechanisticModelingEngine().analyze(
        "状态 x 满足微分方程。", ir_override=structured_ir,
    )
    relation = result["mathematical_ir"]["relations"][0]
    assert relation["parse_status"] != "machine_verified"
    assert any("unsafe_rhs" in error for error in relation["validation_errors"])
    assert result["execution_status"] == "needs_model_completion"
    assert result["numerical_results"] == []


def test_bounded_nonlinear_program_uses_generic_multistart_solver():
    structured_ir = {
        "relations": [{
            "kind": "optimization_problem",
            "decision_variables": ["x", "y"],
            "objective": "(x-a)**2 + (y-b)**2",
            "direction": "minimize",
            "parameters": {"a": 2.0, "b": -1.0},
            "bounds": {"x": [-5.0, 5.0], "y": [-5.0, 5.0]},
            "initial_values": {"x": 0.0, "y": 0.0},
            "constraints": [{"lhs": "x+y", "sense": ">=", "rhs": "0"}],
            "units": {"x": "1", "y": "1", "a": "1", "b": "1"},
            "multistart_trials": 8,
        }]
    }
    result = MechanisticModelingEngine().analyze(
        "选择决策变量，在约束下最小化目标函数。", ir_override=structured_ir,
    )
    assert result["execution_status"] == "executed"
    numerical = result["numerical_results"][0]
    assert numerical["kind"] == "optimization_solution"
    assert numerical["solution"]["x"] == pytest.approx(2.0, abs=1e-6)
    assert numerical["solution"]["y"] == pytest.approx(-1.0, abs=1e-6)
    assert numerical["maximum_constraint_violation"] <= 1e-7
    assert numerical["successful_starts"] >= 1
    assert numerical["credibility_audit"]["status"] == "warning"
    assert "全局" in numerical["credibility_audit"]["label"]


def test_no_dataset_assistant_writes_ordered_artifacts_and_generic_chart(tmp_path):
    result = MathModelingAssistant(
        output_dir=str(tmp_path), feedback_optimization=False,
    ).run(
        "某移动对象以恒定速度运动，进入给定区域时事件生效，求有效持续时间。",
        datasets=None, run_modeling=False, generate_plots=True,
    )
    assert result.input_mode == "mechanistic_no_dataset"
    assert result.dataset_profiles == []
    assert "mechanistic_model" in result.specialized_results
    assert any(chart["type"] == "mechanistic_operator_graph" for chart in result.charts)
    assert Path(result.report_path).is_file()
    assert Path(result.artifact_manifest_path).is_file()
    assert (tmp_path / "evidence" / "mechanistic_model.json").is_file()
    assert (tmp_path / "evidence" / "01_semantic_contract.json").is_file()
    assert (tmp_path / "evidence" / "02_unified_mathematical_ir.json").is_file()
    assert (tmp_path / "evidence" / "03_solver_plan.json").is_file()
    assert (tmp_path / "evidence" / "04_independent_audit.json").is_file()
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "四层数学建模流水线" in report
    assert "通用数学结构能力矩阵" in report
    assert "纯题面通用数学 IR" in report
    assert "规范方程草案" in report
    assert "已完成阶段" in report
    assert "未通过安全门时不生成数值答案" in report


def test_verified_ode_flows_through_chart_report_and_evidence(tmp_path):
    structured_ir = {
        "relations": [{
            "kind": "ode_system", "state_variables": ["x"],
            "rhs": {"x": "-k*x"}, "initial_values": {"x": 10.0},
            "parameters": {"k": 0.1}, "time_variable": "t",
            "time_span": [0.0, 5.0], "output_points": 81,
            "units": {"x": "人", "k": "1/s", "t": "s"},
        }]
    }
    result = MathModelingAssistant(
        output_dir=str(tmp_path), feedback_optimization=False,
    ).run(
        "状态 x 满足微分方程，初值已给定，计算轨迹。",
        datasets=None, run_modeling=False, generate_plots=True,
        mechanistic_ir=structured_ir,
    )
    mechanism = result.specialized_results["mechanistic_model"]
    assert mechanism["execution_status"] == "executed"
    assert any(chart["type"] == "mechanistic_ode_trajectory" for chart in result.charts)
    numerical_claims = [
        claim for claim in result.evidence_bundle["claims"]
        if claim["claim_type"] == "mechanistic_execution"
    ]
    assert len(numerical_claims) == 1
    assert numerical_claims[0]["grade"] == "conditionally_supported"
    assert numerical_claims[0]["numerical_certificate"]["status"] == "pass"


def test_core_has_no_benchmark_specific_solver_branch():
    root = Path(__file__).resolve().parents[1]
    source = "\n".join(
        (root / relative).read_text(encoding="utf-8")
        for relative in (
            "core/mechanistic_modeling.py",
            "core/modeling_assistant.py",
            "core/mathematical_reasoning.py",
        )
    )
    forbidden = ("FY1", "fixed_strategy_result", "_is_smoke_screen_problem")
    assert all(token not in source for token in forbidden)


def test_repeated_unresolved_nodes_are_consolidated_with_concrete_next_actions(tmp_path):
    problem = "\n".join(
        f"问题{index}：选择方案并优化成本，使总费用最小，同时分析影响因素。"
        for index in range(1, 6)
    )
    result = MathModelingAssistant(
        output_dir=str(tmp_path), feedback_optimization=False,
    ).run(problem, datasets=None, run_modeling=False, generate_plots=False)
    pending = [
        claim for claim in result.evidence_bundle["claims"]
        if claim["disposition"] == "unresolved"
    ]
    assert len([claim for claim in pending if claim["claim_type"] == "optimization"]) == 1
    assert len([claim for claim in pending if claim["claim_type"] == "statistical_inference"]) == 1
    optimization_gap = next(claim for claim in pending if claim["claim_type"] == "optimization")
    assert optimization_gap["label"] == "待完成"
    assert "5 个待求解节点" in optimization_gap["statement"]
    assert "约束与变量边界" in optimization_gap["statement"]
    assert "缺少 可执行证据" not in optimization_gap["statement"]
    specification = next(
        claim for claim in result.evidence_bundle["claims"]
        if claim["claim_type"] == "mechanistic_specification"
    )
    assert "规范方程草案" in specification["statement"]


def test_problem_ir_runs_even_when_a_dataset_is_present(tmp_path):
    result = MathModelingAssistant(
        output_dir=str(tmp_path), feedback_optimization=False,
    ).run(
        "选择决策变量优化运输成本，满足容量约束并使总成本最小。",
        datasets={"context": pd.DataFrame({"capacity": [10.0, 12.0], "cost": [3.0, 4.0]})},
        run_modeling=False, generate_plots=False,
    )
    mechanism = result.specialized_results["mechanistic_model"]
    assert "constrained_optimization" in {
        node["key"] for node in mechanism["operator_graph"]
    }
    assert any(
        claim["claim_type"] == "mechanistic_specification"
        for claim in result.evidence_bundle["claims"]
    )


def test_coordinate_rich_no_data_problem_has_five_real_tasks_and_separate_readiness(tmp_path):
    problem = """
以假目标为原点，真目标圆心为(0,200,0)。导弹M1、M2、M3分别位于
(20000,0,2000)、(19000,600,2100)、(18000,−600,1900)；无人机位置信息分别为
FY1(17800,0,1800)、FY2(12000,1400,1400)、FY3(6000,−3000,700)、
FY4(11000,2000,1800)、FY5(13000,−2000,1300)。导弹以300 m/s匀速飞向原点，
无人机可在70~140 m/s内匀速飞行，干扰弹在重力作用下运动；云团以3 m/s下沉，
10 m范围在20 s内有效。
问题1：FY1以120 m/s飞行，受领任务1.5 s后投放，3.6 s后起爆，求有效遮蔽时长。
问题2：确定飞行方向、速度、投放点和起爆点，使遮蔽时间尽可能长。
问题3：投放3枚，给出优化策略，并将结果保存到文件。
问题4：3架无人机各投放1枚，给出优化策略，并将结果保存到文件。
问题5：5架无人机至多各投放3枚，给出优化策略，并将结果保存到文件。
"""
    analysis = analyze_problem(problem)
    assert len(analysis["subproblems"]) == 5
    assert [node["task_type"] for node in analysis["task_graph"]] == [
        "simulation", "optimization", "optimization", "optimization", "optimization",
    ]
    assert {item["task_type"] for item in analysis["task_candidates"]}.isdisjoint({
        "prediction_forecast", "statistical_inference",
    })

    mechanism = MechanisticModelingEngine().analyze(problem)
    assert len(mechanism["mathematical_ir"]["entities"]) == 9
    labels = {item["label"] for item in mechanism["mathematical_ir"]["entities"]}
    assert {"M1", "M2", "M3", "FY1", "FY2", "FY3", "FY4", "FY5"} <= labels
    assert all(item["unit"] == "m" for item in mechanism["mathematical_ir"]["entities"])
    speed_range = next(
        item for item in mechanism["mathematical_ir"]["quantities"]
        if item.get("value_kind") == "closed_range"
    )
    assert speed_range["value"] == [70.0, 140.0]

    result = MathModelingAssistant(
        output_dir=str(tmp_path), feedback_optimization=False,
    ).run(problem, datasets=None, run_modeling=False, generate_plots=False)
    spec = result.mathematical_model_spec
    assert spec["readiness"] == "model_draft_ready"
    assert spec["readiness_by_track"] == {
        "observational_modeling": "not_applicable",
        "mechanistic_structure": "ready",
        "numerical_execution": "needs_confirmation",
    }
    assert spec["missing_requirements"] == ["verified_symbol_and_unit_bindings"]
    assert not {
        "目标变量", "足够样本", "有序时间变量", "至少50个时间点", "至少两个变量",
    } & set(spec["missing_requirements"])
    assert result.evidence_bundle["overall_label"] == "数学结构已形成，数值结论待验证"
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "机理数学结构：**已就绪**" in report
    assert "观测数据建模：**不适用**" in report
    assert "至少50个时间点" not in report
    assert "needs_input" not in report


def test_bound_kinematic_visibility_subproblem_executes_with_semantic_audit(tmp_path):
    problem = """
某载体释放作用装置后，装置在重力作用下运动；激活后形成球状影响区，影响区中心以
3 m/s 匀速下沉，中心10 m范围内在激活20 s内有效。移动源以300 m/s匀速飞向
参考点，参考点为坐标原点。被保护对象是半径7 m、高10 m的圆柱形固定目标，其下底面
圆心为(0,200,0)。移动源A1位于A1(20000,0,2000)，载体C1位于C1(17800,0,1800)。
载体受领任务后等高度匀速直线飞行，速度可在70~140 m/s内选择。
问题1：利用载体C1投放1枚装置实施对A1的干扰，C1以120 m/s朝向参考点方向飞行，
受领任务1.5 s后释放装置，间隔3.6 s后激活，求有效遮蔽时长。
问题2：利用载体C1投放1枚装置实施对A1的干扰，确定载体方向、速度、
释放点和激活点，使有效时间尽可能长。
问题3：利用载体C1投放3枚装置实施对A1的干扰，给出联合投放策略。
"""
    result = MathModelingAssistant(
        output_dir=str(tmp_path), feedback_optimization=False,
    ).run(problem, datasets=None, run_modeling=False, generate_plots=True)
    mechanism = result.specialized_results["mechanistic_model"]
    compiled = [
        relation for relation in mechanism["mathematical_ir"]["relations"]
        if relation.get("kind") == "kinematic_visibility_event"
    ]
    assert len(compiled) == 1
    assert compiled[0]["parse_status"] == "machine_compiled"
    assert mechanism["execution_status"] == "partially_executed"
    assert mechanism["subproblems"][0]["status"] == "executed"
    assert mechanism["subproblems"][1]["status"] == "executed"
    assert mechanism["subproblems"][2]["status"] == "partial"
    numerical = mechanism["numerical_results"][0]
    assert numerical["duration"] == pytest.approx(1.41019739938, abs=1e-9)
    assert numerical["effective_intervals"][0] == pytest.approx(
        [8.037890759321, 9.448088158702], abs=1e-9
    )
    assert numerical["activation_point"] == pytest.approx(
        [17188.0, 0.0, 1736.496], abs=1e-9
    )
    assert numerical["semantic_duration_range"] == pytest.approx(
        [1.41019739938, 1.460641087763], abs=1e-9
    )
    assert numerical["convergence"]["status"] == "pass"
    assert numerical["credibility_audit"]["status"] == "warning"
    optimized = next(
        item for item in mechanism["numerical_results"]
        if item["kind"] == "kinematic_visibility_optimization_solution"
    )
    assert optimized["duration"] > 4.0
    assert optimized["duration"] > numerical["duration"]
    assert optimized["successful_starts"] >= 2
    assert optimized["convergence"]["status"] == "pass"
    assert optimized["credibility_audit"]["status"] == "warning"
    assert optimized["independent_audit"]["grade"] == "conditionally_supported"
    assert "simulation_global_certificate_boundary" in optimized["independent_audit"][
        "false_confidence_flags"
    ]
    assert result.mathematical_model_spec["readiness_by_track"]["numerical_execution"] == "partial"
    assert result.problem_analysis["task_graph"][0]["status"] == "executed"
    assert result.problem_analysis["task_graph"][1]["status"] == "executed"
    assert result.problem_analysis["task_graph"][2]["status"] == "partial"
    optimization_capability = next(
        item for item in result.capability_report["tasks"]
        if item["task_type"] == "optimization"
    )
    assert optimization_capability["status"] == "partial"
    assert result.evidence_bundle["overall_label"] == "数学结构已形成，部分数值结论已验证"
    execution_claim = next(
        claim for claim in result.evidence_bundle["claims"]
        if claim["claim_type"] == "mechanistic_execution"
    )
    assert execution_claim["disposition"] == "restricted"
    assert execution_claim["numerical_certificate"]["status"] == "pass"
    optimization_gap = next(
        claim for claim in result.evidence_bundle["claims"]
        if claim["claim_type"] == "optimization" and claim["disposition"] == "unresolved"
    )
    assert "task_3" in optimization_gap["statement"]
    assert "task_2" not in optimization_gap["statement"]
    assert any(chart["type"] == "mechanistic_visibility_event" for chart in result.charts)
    assert any(
        chart["type"] == "mechanistic_visibility_optimization" for chart in result.charts
    )
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "主语义有效时长：**1.410197 s**" in report
    assert "## 数据集画像" not in report
    assert "## 数据关系" not in report
    assert "## 跨数据集交互" not in report
