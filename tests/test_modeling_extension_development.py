from core.automatic_modeling import induce_and_solve_modeling_task
from core.modeling_benchmark_suite import score_modeling_benchmark_case
from core.modeling_extension_development import (
    build_modeling_extension_development_suite,
    build_modeling_extension_confirmation_suite,
    run_modeling_extension_ablation,
)


def test_extension_development_suite_exercises_all_consumed_failure_structures():
    cases = build_modeling_extension_development_suite()
    assert len(cases) == 8
    assert len({case.structure_group for case in cases}) == 8
    assert {case.source_group for case in cases} == {"post-challenge-development-v1"}


def test_bounded_modeler_solves_extension_development_suite():
    failures = []
    for case in build_modeling_extension_development_suite():
        output = induce_and_solve_modeling_task(case.public_input)
        scored = score_modeling_benchmark_case(output, case.hidden_reference)
        if not scored["valid"]:
            failures.append((case.case_id, output, scored))
    assert failures == []


def test_extension_ablation_attributes_two_tasks_to_each_family_extension():
    report = run_modeling_extension_ablation()
    assert report["summary"]["full"]["valid_count"] == 8
    for variant in (
        "without_higher_order_algebra", "without_extended_ode",
        "without_extended_optimization", "without_three_table_reasoning",
    ):
        assert report["summary"][variant]["valid_count"] == 6
    assert report["status"] == "descriptive_post_challenge_development_only"


def test_integer_extension_scorer_rejects_better_continuous_relaxation():
    case = next(case for case in build_modeling_extension_development_suite()
                if case.structure_group == "challenge-optimization-integer")
    output = induce_and_solve_modeling_task(case.public_input)
    forged = {**output, "solution": {"A": 1.5, "B": 0.0}, "objective": 13.65}
    scored = score_modeling_benchmark_case(forged, case.hidden_reference)
    assert scored["valid"] is False
    assert scored["reason"] == "integrality_violation"
    assert scored["maximum_integrality_violation"] == 0.5


def test_extension_confirmation_metadata_is_deterministic_and_separate():
    development = build_modeling_extension_development_suite()
    first = build_modeling_extension_confirmation_suite()
    second = build_modeling_extension_confirmation_suite()
    assert len(first) == 16
    assert len({case.structure_group for case in first}) == 8
    assert not ({case.case_id for case in development} & {case.case_id for case in first})
    assert [case.public_metadata() for case in first] == [case.public_metadata() for case in second]
