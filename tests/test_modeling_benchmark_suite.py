import numpy as np
import pytest

from core.modeling_benchmark_suite import (
    build_modeling_benchmark_suite,
    current_problem_compiler_adapter,
    legacy_typed_executor_adapter,
    isolated_baseline_adapter,
    run_three_arm_modeling_comparison,
    run_automatic_modeling_ablation,
    run_compositional_depth_ablation,
    run_compositional_beam_comparison,
    run_compositional_depth_three_comparison,
    run_compositional_search_strategy_comparison,
    run_modeling_scorer_attack_audit,
    score_modeling_benchmark_case,
    simple_tool_modeling_adapter,
)


def _contains_key(value, forbidden):
    if isinstance(value, dict):
        return bool(set(value) & forbidden) or any(_contains_key(item, forbidden) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, forbidden) for item in value)
    return False


def test_modeling_suite_starts_from_prose_and_raw_attachments_without_typed_answers():
    cases = build_modeling_benchmark_suite()
    assert len(cases) == 13
    assert len({case.structure_group for case in cases}) == 13
    assert {case.family for case in cases} == {
        "modeling_algebra", "modeling_ode", "modeling_optimization", "modeling_multi_table",
    }
    forbidden = {"candidate", "nodes", "coefficients", "A_ub", "b_ub", "joins",
                 "hidden_reference", "known_solution"}
    for case in cases:
        assert case.statement
        assert case.public_input.get("attachments")
        assert not _contains_key(case.public_input, forbidden)
        case.public_metadata()  # hidden references must also be content-addressable


def test_modeling_scorer_requires_model_even_when_numeric_answer_is_correct():
    case = build_modeling_benchmark_suite()[0]
    scored = score_modeling_benchmark_case(
        {"predictions": case.hidden_reference["predictions"]}, case.hidden_reference,
    )
    assert scored == {"valid": False, "score": 1.0, "reason": "model_description_missing"}


def test_modeling_optimization_scorer_rejects_forged_objective_and_missing_constraints():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.family == "modeling_optimization")
    reference = case.hidden_reference
    incomplete_model = {
        "model": {"family": case.family, "structure": reference["structure"],
                  "decision_variables": reference["decision_variables"], "constraint_ids": [],
                  "decision_units": reference["decision_units"], "objective_unit": reference["objective_unit"]},
        "solution": dict(zip(reference["decision_variables"], reference["known_solution"])),
        "objective": reference["objective"],
    }
    assert score_modeling_benchmark_case(incomplete_model, reference)["reason"] == "optimization_constraints_incomplete"

    forged = dict(incomplete_model)
    forged["model"] = {**incomplete_model["model"], "constraint_ids": reference["constraint_ids"]}
    forged["solution"] = {name: 1e6 for name in reference["decision_variables"]}
    forged["maximum_constraint_violation"] = 0.0
    scored = score_modeling_benchmark_case(forged, reference)
    assert scored["valid"] is False
    assert scored["reason"] in {"bound_violation", "constraint_violation"}

    wrong_units = dict(incomplete_model)
    wrong_units["model"] = {**forged["model"], "decision_units": {
        name: "kg" for name in reference["decision_variables"]}}
    wrong_units["solution"] = dict(zip(reference["decision_variables"], reference["known_solution"]))
    assert score_modeling_benchmark_case(wrong_units, reference)["reason"] == "optimization_units_incorrect"


def test_current_raw_input_compiler_solves_supported_suite_and_simple_baseline_exposes_gaps():
    cases = build_modeling_benchmark_suite()
    current_valid = 0
    simple_valid = 0
    for case in cases:
        task = {"family": case.family, "statement": case.statement, "input": case.public_input}
        current_valid += score_modeling_benchmark_case(
            current_problem_compiler_adapter(task, {}), case.hidden_reference,
        )["valid"]
        try:
            simple_output = simple_tool_modeling_adapter(task, {})
        except Exception:
            continue
        simple_valid += score_modeling_benchmark_case(simple_output, case.hidden_reference)["valid"]
    assert current_valid == 13
    assert simple_valid == 8


def test_multitable_scorer_rejects_future_semantics_and_extra_groups():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-multitable-point-in-time")
    task = {"family": case.family, "statement": case.statement, "input": case.public_input}
    output = simple_tool_modeling_adapter(task, {})
    wrong_time = {**output, "model": {**output["model"], "point_in_time": False}}
    assert score_modeling_benchmark_case(wrong_time, case.hidden_reference)["reason"] == "time_semantics_incorrect"
    extra_group = {**output, "group_totals": {**output["group_totals"], "future": 0.0}}
    assert score_modeling_benchmark_case(extra_group, case.hidden_reference)["reason"] == "join_groups_incorrect"


def test_symbolic_scorer_accepts_numerically_equivalent_operator_identity():
    inputs = [-1.7, -0.2, 0.9, 2.1]
    reference = {"family": "modeling_algebra", "structure": "compositional_symbolic",
                 "input_variables": ["x"], "operator_signature": ["tanh", "sin"],
                 "equivalence_inputs": inputs,
                 "equivalence_outputs": [float(np.tanh(np.sin(x))) for x in inputs],
                 "predictions": [float(np.tanh(np.sin(x))) for x in inputs], "tolerance": 1e-8}
    output = {"model": {"family": "modeling_algebra", "structure": "compositional_symbolic",
                         "input_variables": ["x"], "operator_signature": ["tanh", "cos"],
                         "parameters": [0.0, 1.0, 1.0, -np.pi / 2]},
              "predictions": reference["predictions"]}
    assert score_modeling_benchmark_case(output, reference)["valid"] is True


def test_compositional_depth_ablation_attributes_unseen_topology_success():
    report = run_compositional_depth_ablation()
    assert report["summary"]["depth_two"]["valid_count"] == 4
    assert report["summary"]["flat_only"]["valid_count"] < 4
    assert report["paired_effect"]["resampling_unit"] == "cluster"
    assert "not_search_algorithm_novelty" in report["policy"]


def test_compositional_search_comparison_uses_equal_topology_budget():
    report = run_compositional_search_strategy_comparison()
    budgets = {row["topology_budget"] for row in report["rows"]
               if row["topology_budget"] is not None}
    assert budgets == {21}
    assert report["summary"]["exhaustive_depth_two"]["valid_count"] == 4
    assert "parameter_optimizer_work_may_differ" in report["policy"]


def test_beam_search_uses_equal_budget_and_isolated_workers():
    from core.modeling_equivalence_holdout import build_equivalence_holdout
    report = run_compositional_beam_comparison(cases=build_equivalence_holdout()[:1])
    assert {row["topology_budget"] for row in report["rows"]} == {21}
    assert all(row["resource_enforced"] for row in report["rows"])
    assert report["summary"]["beam_equal_budget"]["case_count"] == 1
    assert "failures_in_denominator" in report["policy"]


def test_depth_three_comparison_records_intentionally_different_grammar_budgets():
    from core.modeling_beam_holdout import build_beam_holdout
    report = run_compositional_depth_three_comparison(cases=build_beam_holdout()[:1])
    budgets = {row["variant"]: row["topology_budget"] for row in report["rows"]}
    assert budgets == {"exhaustive_depth_three": 85, "exhaustive_depth_two": 21}
    assert all(row["resource_enforced"] for row in report["rows"])
    assert "attributes_grammar_reach_not_search_efficiency" in report["policy"]


def test_three_arm_comparison_uses_same_grid_budget_and_keeps_failures():
    report = run_three_arm_modeling_comparison({
        "frozen_old": legacy_typed_executor_adapter,
        "candidate_new": current_problem_compiler_adapter,
        "simple_tool_baseline": simple_tool_modeling_adapter,
    })
    assert report["status"] == "assessed"
    assert report["arms"] == ["frozen_old", "candidate_new", "simple_tool_baseline"]
    assert len(report["rows"]) == 39
    assert report["arm_summary"]["frozen_old"]["valid_count"] == 0
    assert report["arm_summary"]["candidate_new"]["valid_count"] == 13
    assert report["arm_summary"]["simple_tool_baseline"]["valid_count"] == 8
    assert all(summary["case_count"] == 13 for summary in report["arm_summary"].values())
    assert report["version_binding_status"] == "proxy_only"
    assert report["resource_comparison_eligible"] is False
    assert report["arm_summary"]["candidate_new"]["usage_attested_count"] == 13
    assert report["paired_valid_rate_effects"]["simple_tool_baseline"]["paired_effect"] == pytest.approx(5 / 13)
    assert report["interface_probe"]["comparison_eligible"] is False
    assert report["historical_baseline_status"] == "not_available"
    assert report["method_comparison_eligibility"]["candidate_new_vs_historical_system"] is False
    assert report["paired_valid_rate_effects"]["simple_tool_baseline"]["resampling_unit"] == "cluster"
    assert "same_budget_contract" in report["policy"]
    assert "failures_in_denominator" in report["policy"]


def test_development_ablation_attributes_successes_to_specific_mechanisms():
    report = run_automatic_modeling_ablation()
    assert report["summary"]["full"]["valid_count"] == 13
    assert report["summary"]["without_pairwise_interactions"]["valid_count"] == 12
    assert report["summary"]["without_ode_model_selection"]["valid_count"] == 10
    assert report["summary"]["without_optimization_direction_inference"]["valid_count"] == 12
    assert report["summary"]["without_ambiguity_gate"]["valid_count"] == 12
    assert report["lost_successes"]["without_pairwise_interactions"] == ["modeling-algebra-bilinear"]
    assert report["status"] == "descriptive_development_only"


def test_three_arm_report_requires_version_and_usage_evidence_for_resource_claims():
    versions = {name: {"version_id": name + "/v1", "source_digest": character * 64}
                for name, character in zip(
                    ("frozen_old", "candidate_new", "simple_tool_baseline"), ("a", "b", "c"))}
    report = run_three_arm_modeling_comparison({
        "frozen_old": isolated_baseline_adapter("frozen_old"),
        "candidate_new": current_problem_compiler_adapter,
        "simple_tool_baseline": isolated_baseline_adapter("simple_tool_baseline"),
    }, system_versions=versions, fixed_budget={
        "per_case_wall_seconds": 30.0, "memory_mb": 1024,
        "max_model_api_calls": 0, "max_numerical_solver_calls": 2,
        "max_manual_interventions": 1, "seed": 20260915,
    })
    assert report["version_binding_status"] == "bound"
    assert report["resource_comparison_eligible"] is True
    assert all(summary["usage_attested_count"] == 13 for summary in report["arm_summary"].values())
    assert all(summary["resource_enforced_count"] == 13 for summary in report["arm_summary"].values())


def test_scorer_attack_audit_rejects_every_injected_invalid_submission():
    report = run_modeling_scorer_attack_audit()
    assert report["status"] == "passed"
    assert report["control_count"] == 13
    assert report["control_failure_count"] == 0
    assert report["attack_count"] >= 40
    assert report["false_accept_count"] == 0
    assert report["attack_detection_rate"] == 1.0
    assert {row["attack_id"] for row in report["attacks"]} >= {
        "correct_numeric_answer_without_model",
        "correct_predictions_wrong_equation",
        "correct_trajectory_wrong_dynamics",
        "objective_only_without_decisions",
        "forged_feasibility_and_objective",
        "future_information_semantics",
        "invent_answer_for_ambiguous_input",
    }
