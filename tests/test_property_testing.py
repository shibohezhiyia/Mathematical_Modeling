import pytest

from core.property_testing import PropertyTestBudget, PropertyTestError, check_scalar_property


def test_bounded_property_records_finite_evidence_without_claiming_proof():
    result = check_scalar_property(lambda point: point["x"] ** 2, domain={"x": [-2, 2]},
                                   kind="bounded", lower=0, upper=4,
                                   budget=PropertyTestBudget(probe_count=12, seed=7))
    assert result["status"] == "pass"
    assert result["proof_status"] == "tested_not_falsified"
    assert result["evaluations"] == 12


def test_monotonicity_and_symmetry_find_counterexamples():
    decreasing = check_scalar_property(lambda point: -point["x"], domain={"x": [-1, 1]},
                                       kind="monotone_increasing", budget=PropertyTestBudget(probe_count=8))
    assert decreasing["status"] == "fail"
    assert decreasing["proof_status"] == "counterexample_found"
    even = check_scalar_property(lambda point: point["x"], domain={"x": [-1, 1]},
                                 kind="symmetry_even", budget=PropertyTestBudget(probe_count=8))
    assert even["status"] == "fail"


def test_property_budget_and_input_contract_are_explicit():
    with pytest.raises(PropertyTestError, match="ordered_property"):
        check_scalar_property(lambda point: point["x"], domain={"x": [-1, 1]}, kind="monotone_increasing",
                              variable="missing")
    result = check_scalar_property(lambda point: point["x"], domain={"x": [-1, 1]}, kind="bounded",
                                   lower=-1, upper=1, budget=PropertyTestBudget(probe_count=8, max_evaluations=2))
    assert result["status"] == "not_assessed"
    assert result["reason"] == "property_evaluation_budget_exhausted"
