import math

import pytest

from core.domain_extrapolation import DomainExtrapolationError, assess_domain_extrapolation


def test_finite_model_can_be_checked_inside_and_outside_declared_domain():
    result = assess_domain_extrapolation(lambda point: 1.0 + point["x"], domain={"x": [0, 1]},
                                         probe_count=8, output_lower=-5, output_upper=5)
    assert result["status"] == "no_falsification_in_tested_extension"
    assert len(result["declared_domain"]["points"]) == 8
    assert result["policy"].startswith("finite_domain_stress")


def test_domain_extension_records_bound_counterexample_without_relabeling_training_fit():
    result = assess_domain_extrapolation(lambda point: point["x"] ** 2, domain={"x": [-1, 1]},
                                         expansion_factor=2, probe_count=10, output_upper=1)
    assert result["status"] == "counterexample_found"
    assert any(item["phase"] == "extended_domain" for item in result["counterexamples"])
    assert result["evidence_scope"] == "tested_points_only"


def test_nonfinite_outside_value_is_a_failure_witness():
    def model(point):
        return float("inf") if point["x"] > 1 else 1 / (point["x"] - 1)

    result = assess_domain_extrapolation(model, domain={"x": [0, 0.9]},
                                         expansion_factor=10, probe_count=12)
    assert result["status"] in {"counterexample_found", "domain_shift_signal"}
    assert result["counterexamples"]


@pytest.mark.parametrize("kwargs", [
    {"probe_count": 1}, {"expansion_factor": 1.0}, {"seed": -1},
    {"output_lower": 2, "output_upper": 1},
])
def test_domain_extrapolation_rejects_invalid_contract(kwargs):
    with pytest.raises(DomainExtrapolationError):
        assess_domain_extrapolation(lambda point: 0.0, domain={"x": [0, 1]}, **kwargs)


def test_evaluator_errors_are_structured_and_bounded():
    result = assess_domain_extrapolation(lambda point: math.sqrt(point["x"]),
                                         domain={"x": [-1, 0]}, probe_count=8)
    assert result["status"] == "counterexample_found"
    assert all("reason" in item for item in result["counterexamples"])
