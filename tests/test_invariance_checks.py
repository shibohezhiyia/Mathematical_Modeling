import pytest

from core.invariance_checks import InvarianceCheckError, run_invariance_checks


def test_entity_permutation_and_unit_roundtrip_are_explicit_checks():
    result = run_invariance_checks(
        [1, 2, 3], 6, sum,
        [{"id": "entity_permutation", "apply": lambda values: list(reversed(values)),
          "compare": lambda before, after, tol: before == after}],
    )
    assert result["status"] == "tested_not_falsified"


def test_failed_transformation_is_a_counterexample_not_an_exception():
    result = run_invariance_checks(
        2, 4, lambda x: x * x,
        [{"id": "bad_scale", "apply": lambda x: x * 2,
          "compare": lambda before, after, tol: before == after}],
    )
    assert result["status"] == "counterexample"


def test_invariance_cases_have_a_bounded_unique_budget():
    case = {"id": "same", "apply": lambda value: value, "compare": lambda a, b, tol: True}
    with pytest.raises(InvarianceCheckError, match="invariance_case_id_must_be_unique"):
        run_invariance_checks(1, 1, lambda value: value, [case, dict(case)])
    with pytest.raises(InvarianceCheckError, match="invariance_case_budget_exceeded"):
        run_invariance_checks(
            1, 1, lambda value: value,
            ({"id": str(index), "apply": lambda value: value, "compare": lambda a, b, tol: True}
             for index in range(3)),
            max_cases=2,
        )
