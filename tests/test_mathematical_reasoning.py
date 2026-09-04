from pathlib import Path
import json

import pandas as pd

from core.mathematical_reasoning import (
    MathematicalReasoningEngine,
    classify_expression_structure,
    check_equation_dimensions,
    check_expression_dimensions,
    extract_column_unit,
    parse_unit,
)
from core.modeling_assistant import InteractionFinding, MathModelingAssistant
from core.problem_solver import analyze_problem


def test_unit_algebra_accepts_physical_identity_and_rejects_invalid_addition():
    units = {
        "distance_km": "km",
        "speed_km_h": "km/h",
        "time_h": "h",
        "unused_unknown": "mystery-unit",
    }
    valid = check_equation_dimensions(
        "distance_km", "speed_km_h * time_h", units
    )
    invalid = check_expression_dimensions("distance_km + time_h", units)

    assert valid["status"] == "pass"
    assert valid["left_dimension"]["powers"] == {"length": 1.0}
    assert invalid["status"] == "fail"
    assert "量纲不一致" in invalid["evidence"]
    assert parse_unit("km/h").mapping == {"length": 1.0, "time": -1.0}
    assert extract_column_unit("speed_km_h") == "km/h"


def test_fdr_nonsignificant_association_is_not_promoted_to_supported_claim():
    engine = MathematicalReasoningEngine()
    datasets = {
        "left": pd.DataFrame({"key": range(20), "x": range(20)}),
        "right": pd.DataFrame({"key": range(20), "y": range(20)}),
    }
    problem = "分析两个数据集变量之间的统计关系"
    analysis = analyze_problem(problem)
    spec = engine.build_spec(problem, datasets, analysis)
    finding = InteractionFinding(
        "left", "right", "x", "y", "correlation_ratio_eta_squared",
        0.18, 20, "group_effect", "存在一个探索性组间效应。",
        p_value=0.06, q_value=0.12, significant=False,
    )
    bundle = engine.build_evidence_bundle(
        spec, datasets, relationships=[], interactions=[finding],
        model_results=[], ranking_result=None, specialized_results={},
        task_graph=analysis["task_graph"],
    ).to_dict()

    claim = next(item for item in bundle["claims"] if item["claim_type"] == "association")
    assert claim["grade"] == "undetermined"
    assert claim["disposition"] == "unresolved"
    assert "未通过全局 FDR 校正" in claim["statement"]


def test_untested_nonlinear_association_remains_undetermined():
    engine = MathematicalReasoningEngine()
    datasets = {
        "left": pd.DataFrame({"key": range(20), "x": range(20)}),
        "right": pd.DataFrame({"key": range(20), "y": range(20)}),
    }
    problem = "分析两个数据集变量之间的非线性关系"
    analysis = analyze_problem(problem)
    spec = engine.build_spec(problem, datasets, analysis)
    finding = InteractionFinding(
        "left", "right", "x", "y", "binned_normalized_mutual_information",
        0.2, 20, "nonlinear_dependence", "存在未经显著性检验的非线性依赖。",
        p_value=None, q_value=None, significant=None,
    )
    bundle = engine.build_evidence_bundle(
        spec, datasets, relationships=[], interactions=[finding],
        model_results=[], ranking_result=None, specialized_results={},
        task_graph=analysis["task_graph"],
    ).to_dict()

    claim = next(item for item in bundle["claims"] if item["claim_type"] == "association")
    assert claim["grade"] == "undetermined"
    assert claim["disposition"] == "unresolved"


def test_model_spec_extracts_roles_assumptions_candidates_and_formula_checks():
    data = pd.DataFrame({
        "distance_km": [10.0, 20.0, 30.0],
        "speed_km_h": [5.0, 10.0, 10.0],
        "time_h": [2.0, 2.0, 3.0],
        "treatment": [0, 1, 1],
        "outcome": [1.0, 3.0, 4.0],
        "baseline": [0.2, 0.5, 0.7],
    })
    problem = (
        "研究因果效应，处理变量=treatment，结果变量=outcome，控制变量均为处理前基线协变量；"
        "并满足 distance_km = speed_km_h * time_h"
    )
    engine = MathematicalReasoningEngine()
    spec = engine.build_spec(problem, {"travel": data}, analyze_problem(problem))

    assert spec.role_bindings == {
        "treatment": "travel.treatment", "outcome": "travel.outcome"
    }
    assert spec.unit_checks and spec.unit_checks[0]["status"] == "pass"
    assert not spec.contradictions
    causal_candidates = [
        item for item in spec.candidate_models if item.task_type == "causal_inference"
    ]
    assert len(causal_candidates) >= 2
    assert any(item.solver == "cross_fitted_dml" for item in causal_candidates)
    assumption = next(item for item in spec.assumptions if item.id == "assumption_exchangeability")
    assert assumption.status == "not_assessed"
    assert assumption.critical
    assert spec.output_policy["narrative_generation"] == "disabled_by_default"


def test_model_spec_turns_dimensional_contradiction_into_invalid_readiness():
    data = pd.DataFrame({"distance_km": [1.0, 2.0], "time_h": [1.0, 2.0]})
    problem = "建立关系并满足 distance_km = time_h"
    engine = MathematicalReasoningEngine()
    spec = engine.build_spec(problem, {"travel": data}, analyze_problem(problem))

    assert spec.unit_checks[0]["status"] == "fail"
    assert spec.readiness == "invalid"
    assert any(item["id"].startswith("unit_") for item in spec.contradictions)

    inequality = engine.build_spec(
        "约束 distance_km <= time_h", {"travel": data},
        analyze_problem("约束 distance_km <= time_h"),
    )
    assert inequality.unit_checks[0]["status"] == "fail"


def test_explicit_linear_optimization_compiles_only_after_complete_formulation():
    problem = (
        "建立优化模型；决策变量=x,y；最小化 3*x + 2*y；"
        "约束 x + y >= 10；x >= 0；y >= 0"
    )
    engine = MathematicalReasoningEngine()
    spec = engine.build_spec(
        problem, {"parameters": pd.DataFrame({"demand": [10.0]})},
        analyze_problem(problem),
    )

    decisions = {symbol.name for symbol in spec.symbols if symbol.role == "decision"}
    compiler = next(item for item in spec.compiler_plan if item["task_type"] == "optimization")
    assert decisions == {"x", "y"}
    assert spec.objectives[0]["direction"] == "minimize"
    assert all(item["executable"] for item in spec.constraints)
    assert compiler["status"] == "ready_to_compile"
    assert compiler["formulation_class"] == "linear_program"
    assert "highs" in compiler["solver"]
    solved = engine.solve_explicit_optimization(
        spec, {"parameters": pd.DataFrame({"demand": [10.0]})}
    )
    assert solved["solver_success"] is True
    assert abs(solved["solution"]["x"]) < 1e-8
    assert abs(solved["solution"]["y"] - 10.0) < 1e-8
    assert abs(solved["objective_value"] - 20.0) < 1e-8
    assert solved["maximum_constraint_violation"] <= 1e-8
    assert solved["optimality_certificate"]["status"] == "pass"
    assert solved["optimality_certificate"]["maximum_kkt_residual"] <= 1e-8
    assert solved["robust_feedback"]["attempted"] is True
    assert solved["robust_feedback"]["accepted_as_primary"] is False
    assert solved["robust_feedback"]["candidate_worst_normalized_regret"] <= (
        solved["robust_feedback"]["nominal_worst_normalized_regret"] + 1e-12
    )
    assert {item["id"] for item in solved["credibility_audit"]["checks"]} >= {
        "solver_termination", "primal_feasibility", "near_optimal_identifiability",
        "objective_sensitivity", "kkt_optimality",
    }

    incomplete = engine.build_spec(
        "优化资源配置使成本最低", {"parameters": pd.DataFrame({"cost": [3.0]})},
        analyze_problem("优化资源配置使成本最低"),
    )
    incomplete_plan = next(
        item for item in incomplete.compiler_plan if item["task_type"] == "optimization"
    )
    assert incomplete_plan["status"] == "needs_input"
    assert "显式决策变量" in incomplete_plan["missing_requirements"]

    nonlinear = classify_expression_structure("x**2 + y", {"x", "y"}, {"x", "y"})
    assert nonlinear["structure"] == "quadratic"

    integer_problem = (
        "建立优化模型；0-1决策变量=x,y；最大化 4*x+3*y；"
        "约束 2*x+y<=2"
    )
    integer_spec = engine.build_spec(
        integer_problem, {"parameters": pd.DataFrame({"dummy": [1.0]})},
        analyze_problem(integer_problem),
    )
    integer_plan = next(
        item for item in integer_spec.compiler_plan if item["task_type"] == "optimization"
    )
    assert integer_plan["formulation_class"] == "mixed_integer_linear_program"
    assert integer_plan["discrete_variables"] == ["x", "y"]
    assert integer_plan["executable"] is False
    assert engine.solve_explicit_optimization(
        integer_spec, {"parameters": pd.DataFrame({"dummy": [1.0]})}
    ) is None

    parameter_problem = (
        "优化配置；决策变量=x；参数=demand；最小化 3*x；约束 x>=demand"
    )
    parameter_data = {"parameters": pd.DataFrame({"demand": [7.0]})}
    parameter_spec = engine.build_spec(
        parameter_problem, parameter_data, analyze_problem(parameter_problem)
    )
    parameter_solution = engine.solve_explicit_optimization(parameter_spec, parameter_data)
    assert parameter_solution["solution"]["x"] == 7.0


def test_failed_falsification_blocks_claim_from_writing_contract():
    data = pd.DataFrame({"x": range(50), "target": range(50)})
    problem = "预测 target"
    engine = MathematicalReasoningEngine()
    analysis = analyze_problem(problem)
    spec = engine.build_spec(problem, {"sample": data}, analysis, "sample.target")
    model = {
        "dataset": "sample", "target": "target", "task_type": "regression",
        "best_model": "apparently_good", "validation": "random_cv",
        "metrics": {"r2": 0.999},
        "credibility_audit": {
            "status": "fail",
            "checks": [{
                "id": "target_leakage", "name": "目标泄漏", "status": "fail",
                "evidence": "x 是目标的逐行复制", "recommendation": "删除泄漏字段",
            }],
        },
    }
    bundle = engine.build_evidence_bundle(
        spec, {"sample": data}, [], [], [model], None, {},
        [{"id": "task_1", "task_type": "prediction_forecast", "status": "executed"}],
    )

    claim = next(item for item in bundle.claims if item.claim_type == "predictive")
    assert claim.grade == "refuted"
    assert claim.disposition == "rejected"
    assert claim.id in bundle.writing_contract["prohibited_claim_ids"]
    assert claim.id not in bundle.writing_contract["allowed_claim_ids"]
    assert bundle.overall_status == "contains_rejected_claims"
    assert bundle.argument_integrity["status"] == "pass"
    assert claim.challenges


def test_assistant_executes_complete_lp_and_marks_optimum_as_conditional(tmp_path):
    problem = (
        "建立优化模型；决策变量=x,y；最小化 3*x + 2*y；"
        "约束 x + y >= 10；x >= 0；y >= 0"
    )
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        problem, {"parameters": pd.DataFrame({"demand": [10.0]})},
        run_modeling=False, generate_plots=True,
    )

    optimization = result.specialized_results["optimization"]
    assert optimization["solver_success"] is True
    assert optimization["solution"] == {"x": 0.0, "y": 10.0}
    task = next(
        item for item in result.problem_analysis["task_graph"]
        if item["task_type"] == "optimization"
    )
    assert task["status"] == "executed"
    claim = next(
        item for item in result.evidence_bundle["claims"]
        if item["claim_type"] == "optimization"
    )
    assert claim["grade"] == "conditionally_supported"
    assert claim["disposition"] == "restricted"
    assert claim["numerical_certificate"]["status"] == "pass"
    assert claim["id"] in result.evidence_bundle["writing_contract"]["allowed_claim_ids"]
    certificate_claim = next(
        item for item in result.evidence_bundle["claims"]
        if item["claim_type"] == "optimization_certificate"
    )
    assert certificate_claim["grade"] == "deductively_verified"
    assert certificate_claim["label"] == "数学上已验证"
    chart = next(item for item in result.charts if item["type"] == "optimization_solution")
    assert Path(chart["path"]).is_file()


def test_assistant_rejects_simpson_reversal_and_persists_compilation_evidence(tmp_path):
    rows = []
    for group in range(5):
        for within in range(20):
            rows.append({
                "组别": f"G{group}",
                "解释变量": group * 10 + within,
                "目标变量": (4 - group) * 100 + within * 2,
            })
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "分析各组解释变量与目标变量的关系",
        {"样本": pd.DataFrame(rows)},
        target="样本.目标变量",
        run_modeling=False,
        generate_plots=True,
    )

    compilation = result.specialized_results["mathematical_data_compilation"]
    assert compilation["status"] == "contradicted"
    rejected = [
        claim for claim in result.evidence_bundle["claims"]
        if claim["disposition"] == "rejected"
        and "稳定同向关系" in claim["statement"]
    ]
    assert rejected
    assert rejected[0]["id"] in result.evidence_bundle["writing_contract"]["prohibited_claim_ids"]
    assert any(
        chart["type"] == "mathematical_data_view_stability"
        and Path(chart["path"]).is_file()
        for chart in result.charts
    )
    assert (tmp_path / "evidence" / "mathematical_data_compilation.json").is_file()
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "数学数据编译与多视图反证" in report


def test_assistant_rejects_raw_many_to_many_join_as_estimand_violation(tmp_path):
    left = pd.DataFrame({
        "组别": ["A"] * 5 + ["B"] * 5 + ["C"] * 5,
        "指标A": [float(index) for index in range(15)],
    })
    right = pd.DataFrame({
        "组别": ["A"] * 7 + ["B"] * 7 + ["C"] * 7,
        "指标B": [float(index) for index in range(21)],
    })
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "按组分析两个表的指标关系",
        {"left": left, "right": right},
        target="left.指标A",
        run_modeling=False,
        generate_plots=False,
    )

    cross = result.specialized_results["mathematical_data_compilation"][
        "cross_dataset_contracts"
    ][0]
    assert cross["status"] == "blocked"
    claim = next(
        item for item in result.evidence_bundle["claims"]
        if "总量和样本单位保持不变" in item["statement"]
    )
    assert claim["grade"] == "refuted"
    assert claim["disposition"] == "rejected"
    assert claim["id"] in result.evidence_bundle["writing_contract"]["prohibited_claim_ids"]


def test_linear_optimizer_returns_infeasible_and_unbounded_as_failure_evidence():
    engine = MathematicalReasoningEngine()
    datasets = {"parameters": pd.DataFrame({"dummy": [1.0]})}
    infeasible_problem = (
        "优化配置；决策变量=x,y；最小化 x+y；"
        "约束 x+y>=10；x+y<=5；x>=0；y>=0"
    )
    infeasible_spec = engine.build_spec(
        infeasible_problem, datasets, analyze_problem(infeasible_problem)
    )
    infeasible = engine.solve_explicit_optimization(infeasible_spec, datasets)
    assert infeasible["solver_success"] is False
    assert infeasible["credibility_audit"]["status"] == "fail"
    assert "infeasible" in infeasible["message"].lower()

    unbounded_problem = "优化配置；决策变量=x；最小化 -x；约束 x>=0"
    unbounded_spec = engine.build_spec(
        unbounded_problem, datasets, analyze_problem(unbounded_problem)
    )
    unbounded = engine.solve_explicit_optimization(unbounded_spec, datasets)
    assert unbounded["solver_success"] is False
    assert unbounded["credibility_audit"]["status"] == "fail"
    assert "unbounded" in unbounded["message"].lower()


def test_unexecuted_task_is_an_explicit_undetermined_claim():
    data = pd.DataFrame({"cost": [3.0, 2.0, 4.0]})
    problem = "建立优化模型使成本最小"
    engine = MathematicalReasoningEngine()
    spec = engine.build_spec(problem, {"sample": data}, analyze_problem(problem))
    bundle = engine.build_evidence_bundle(
        spec, {"sample": data}, [], [], [], None, {},
        [{
            "id": "task_1", "task_type": "optimization", "status": "needs_input",
            "missing_requirements": ["决策变量", "可执行约束"],
        }],
    )

    claim = next(item for item in bundle.claims if item.claim_type == "optimization")
    assert claim.grade == "undetermined"
    assert "决策变量" in claim.statement
    assert claim.id in bundle.writing_contract["prohibited_claim_ids"]


def test_assistant_outputs_evidence_bundle_instead_of_claiming_to_write_paper(tmp_path):
    data = pd.DataFrame({
        "city_id": ["A", "B", "C", "D"],
        "benefit": [8.0, 9.0, 6.0, 7.0],
        "cost": [4.0, 7.0, 3.0, 5.0],
    })
    result = MathModelingAssistant(output_dir=str(tmp_path)).run(
        "综合评价城市，成本越低越好", {"cities": data},
        run_modeling=False, generate_plots=False,
    )

    assert result.mathematical_model_spec["output_policy"]["narrative_generation"] == "disabled_by_default"
    assert result.evidence_bundle["writing_contract"]["enabled"] is False
    assert result.evidence_bundle["argument_integrity"]["status"] == "pass"
    assert result.evidence_bundle["claims"]
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert report.startswith("# 数学建模论证证据包")
    assert "不是自动生成的竞赛论文" in report
    assert "假设账本" in report
    assert "论证结论分级" in report
    assert (tmp_path / "evidence" / "mathematical_model_spec.json").is_file()
    assert (tmp_path / "evidence" / "evidence_bundle.json").is_file()
    manifest = json.loads((tmp_path / "artifact_manifest.json").read_text("utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == "mathmodel.run-artifacts/v1"
