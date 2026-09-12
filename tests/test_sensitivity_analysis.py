import math

import pytest

from core.sensitivity_analysis import SensitivityAnalysisError, assess_global_sensitivity


def test_global_screen_ranks_material_parameter_higher():
    result = assess_global_sensitivity(
        lambda p: 10 * p["important"] + 0.1 * p["minor"],
        parameter_bounds={"important": [0, 1], "minor": [0, 1]},
        sample_count=24,
    )
    assert result["status"] == "assessed"
    by_name = {item["parameter"]: item for item in result["effects"]}
    assert by_name["important"]["rank"] == 1
    assert by_name["important"]["total_effect"] > by_name["minor"]["total_effect"]
    assert "not_causal" in by_name["important"]["interpretation"]


def test_constant_output_is_not_called_identifiable():
    result = assess_global_sensitivity(lambda p: 3.0, parameter_bounds={"a": [0, 1]}, sample_count=8)
    assert result["status"] == "no_output_variation"
    assert result["effects"][0]["rank"] is None


def test_budget_and_nonfinite_failures_are_structured():
    limited = assess_global_sensitivity(lambda p: p["a"], parameter_bounds={"a": [0, 1]},
                                        sample_count=8, max_evaluations=4)
    assert limited["status"] == "not_assessed"
    failed = assess_global_sensitivity(lambda p: float("nan") if p["a"] > 0.5 else p["a"],
                                       parameter_bounds={"a": [0, 1]}, sample_count=16)
    assert failed["status"] == "not_assessed"
    assert failed["failures"]


@pytest.mark.parametrize("kwargs", [
    {"sample_count": 3}, {"seed": -1}, {"max_evaluations": 0},
])
def test_invalid_sensitivity_contract(kwargs):
    with pytest.raises(SensitivityAnalysisError):
        assess_global_sensitivity(lambda p: 0.0, parameter_bounds={"a": [0, 1]}, **kwargs)


def test_sensitivity_rejects_boolean_bounds_and_whitespace_collisions():
    with pytest.raises(SensitivityAnalysisError, match="invalid_parameter_bounds"):
        assess_global_sensitivity(lambda p: 0.0, parameter_bounds={"a": [False, 1]})
    with pytest.raises(SensitivityAnalysisError, match="parameter_names_must_be_unique"):
        assess_global_sensitivity(lambda p: 0.0, parameter_bounds={"a": [0, 1], " a ": [0, 1]})
