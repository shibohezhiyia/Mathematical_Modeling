from core.modeling_benchmark_suite import build_modeling_benchmark_suite, current_problem_compiler_adapter, score_modeling_benchmark_case
from core.modeling_open_structure_challenge import build_modeling_open_structure_challenge


def test_open_structure_challenge_is_disjoint_and_current_library_does_not_pass_it():
    cases = build_modeling_open_structure_challenge()
    assert len(cases) == 4
    assert {case.structure_group for case in cases}.isdisjoint(
        {case.structure_group for case in build_modeling_benchmark_suite()}
    )
    outcomes = [score_modeling_benchmark_case(
        current_problem_compiler_adapter({"family": case.family, "statement": case.statement,
                                          "input": case.public_input}, {}),
        case.hidden_reference,
    )["valid"] for case in cases]
    assert outcomes == [False, False, False, False]
