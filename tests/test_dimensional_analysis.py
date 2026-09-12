import pytest

from core.dimensional_analysis import DimensionalAnalysisError, buckingham_pi_groups


def test_buckingham_pi_uses_exact_rational_nullspace():
    result = buckingham_pi_groups({
        "length": {"L": 1}, "time": {"T": 1}, "speed": {"L": 1, "T": -1},
    })
    assert result["group_count"] == 1
    assert result["groups"][0]["dimensionless"] is True
    assert result["groups"][0]["exponents"] == {"length": -1, "time": 1, "speed": 1}
    assert result["policy"] == "exact_nullspace_candidate_not_physical_validation"


def test_dimensionless_quantity_and_duplicate_dimensions_are_preserved():
    result = buckingham_pi_groups({"x": {"L": 1}, "y": {"L": 1}, "ratio": {}})
    assert result["group_count"] == 2
    assert result["rank"] == 1


def test_dimensional_analysis_rejects_malformed_or_nonfinite_exponents():
    with pytest.raises(DimensionalAnalysisError, match="invalid_dimensions"):
        buckingham_pi_groups({"x": [1, 2]})
    with pytest.raises(DimensionalAnalysisError, match="dimension_exponent"):
        buckingham_pi_groups({"x": {"L": 1e100}})


def test_dimensional_analysis_rejects_unstable_quantity_and_base_names():
    with pytest.raises(DimensionalAnalysisError, match="invalid_quantity_names"):
        buckingham_pi_groups({"": {"L": 1}})
    with pytest.raises(DimensionalAnalysisError, match="invalid_base_dimensions"):
        buckingham_pi_groups({"x": [1]}, base_dimensions=("L", ["bad"]))
