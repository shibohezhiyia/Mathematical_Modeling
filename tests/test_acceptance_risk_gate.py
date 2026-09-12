import pytest

from core.acceptance_risk_gate import AcceptanceRiskGateError, assess_repair_acceptance_risk


def _rows(n=24):
    return [{"baseline": {"valid": False, "error_accept": False},
             "treatment": {"valid": True, "error_accept": False}} for _ in range(n)]


def test_acceptance_gate_requires_minimum_evidence_before_claiming_pass():
    result = assess_repair_acceptance_risk(_rows(10), min_cases=20)
    assert result["status"] == "not_assessed"
    assert result["valid_rate_gain_interval"] is None


def test_acceptance_gate_passes_only_when_gain_and_error_risk_both_clear():
    result = assess_repair_acceptance_risk(_rows(), min_cases=20, min_valid_rate_gain=0.5,
                                           max_treatment_error_rate=0.15, bootstrap_replicates=100)
    assert result["status"] == "pass"
    assert result["treatment_error_acceptance_interval"][1] <= 0.15


def test_acceptance_gate_rejects_error_prone_treatment():
    rows = _rows()
    for row in rows:
        row["treatment"]["error_accept"] = True
    result = assess_repair_acceptance_risk(rows, min_cases=20, min_valid_rate_gain=0.1,
                                           max_treatment_error_rate=0.05, bootstrap_replicates=100)
    assert result["status"] == "fail"


@pytest.mark.parametrize("kwargs", [{"min_cases": 0}, {"bootstrap_replicates": 99}, {"seed": -1}])
def test_acceptance_gate_bounds_resources(kwargs):
    with pytest.raises(AcceptanceRiskGateError):
        assess_repair_acceptance_risk(_rows(), **kwargs)
