from core.modeling_benchmark_suite import build_modeling_benchmark_suite
from core.modeling_structure_challenge import build_modeling_structure_challenge


def test_structure_challenge_is_disjoint_and_balanced_without_typed_public_models():
    development = build_modeling_benchmark_suite()
    challenge = build_modeling_structure_challenge()
    assert len(challenge) == 8
    assert len({case.structure_group for case in challenge}) == 8
    assert not ({case.structure_group for case in development} & {case.structure_group for case in challenge})
    assert {case.family for case in challenge} == {
        "modeling_algebra", "modeling_ode", "modeling_optimization", "modeling_multi_table",
    }
    for case in challenge:
        assert case.public_input["attachments"]
        assert "hidden_reference" not in case.public_input
        case.public_metadata()
