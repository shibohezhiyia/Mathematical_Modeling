import math

from core.automated_benchmark_suite import (
    build_automated_benchmark_suite, run_typed_execution_benchmark,
    score_automated_benchmark_case,
)


def test_suite_has_four_families_and_structure_grouped_variants():
    cases = build_automated_benchmark_suite(cases_per_family=8)
    assert len(cases) == 32
    assert {case.family for case in cases} == {"algebra", "ode", "optimization", "multi_table"}
    assert len({case.structure_group for case in cases}) == 7
    assert all("hidden_reference" not in case.public_input for case in cases)


def test_independent_scorer_accepts_known_references_and_rejects_bad_outputs():
    cases = build_automated_benchmark_suite(cases_per_family=1)
    for case in cases:
        reference = case.hidden_reference
        if case.family == "algebra":
            output = {"outputs": {"out": reference["output"]}}
        elif case.family == "ode":
            output = {"trajectory": reference["trajectory"]}
        elif case.family == "optimization":
            objective = reference["objective_coefficients"]
            capacity = reference["b_ub"][0]
            upper = [bound[1] for bound in reference["bounds"]]
            remaining = capacity
            decision = [0.0, 0.0]
            for position in sorted(range(2), key=lambda item: objective[item], reverse=True):
                decision[position] = min(upper[position], max(0.0, remaining))
                remaining -= decision[position]
            output = {"solution": {"x0": decision[0], "x1": decision[1]},
                      "objective": reference["objective"]}
        else:
            output = {"rows": reference["rows"], "group_totals": reference["group_totals"]}
        score = score_automated_benchmark_case(output, reference)
        assert score["valid"] is True, (case.case_id, score)
        bad = score_automated_benchmark_case({"value": math.nan}, reference)
        assert bad["valid"] is False


def test_optimization_scorer_rejects_scalar_only_infeasible_and_forged_answers():
    case = next(item for item in build_automated_benchmark_suite(cases_per_family=1)
                if item.family == "optimization")
    reference = case.hidden_reference
    assert score_automated_benchmark_case(
        {"objective": reference["objective"]}, reference,
    )["reason"] == "decision_solution_missing_or_invalid"

    infeasible = {"solution": {"x0": 1e9, "x1": 1e9},
                  "objective": reference["objective"]}
    scored = score_automated_benchmark_case(infeasible, reference)
    assert scored["valid"] is False
    assert scored["reason"] == "bound_violation"

    forged = {"solution": {"x0": 0.0, "x1": 0.0},
              "objective": reference["objective"],
              "maximum_constraint_violation": 0.0}
    scored = score_automated_benchmark_case(forged, reference)
    assert scored["valid"] is False
    assert scored["reason"] == "reported_objective_mismatch"


def test_typed_execution_baseline_runs_all_default_cases():
    report = run_typed_execution_benchmark(cases_per_family=8, wall_seconds=60)
    assert report["status"] == "completed"
    assert report["case_count"] == 32
    assert report["completed_count"] == 32
    assert report["valid_count"] == 32
    assert report["valid_rate"] == 1.0
    assert "not_real_world_accuracy" in report["policy"]
