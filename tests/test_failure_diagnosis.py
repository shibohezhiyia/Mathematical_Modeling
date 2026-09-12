import pytest

from core.failure_diagnosis import FailureDiagnosisError, diagnose_failure


def test_failure_diagnosis_orders_alternative_explanations():
    result = diagnose_failure({"observation_error": {"status": "fail", "evidence": "heteroscedastic"}}, residual=3.0)
    assert result["leading_explanations"] == ["observation_error"]
    assert result["latent_state_action"].startswith("not_automatic")


def test_failure_diagnosis_rejects_invalid_status():
    with pytest.raises(FailureDiagnosisError):
        diagnose_failure({"numerical_tolerance": {"status": "guess"}})


def test_failure_diagnosis_rejects_nonfinite_residual_and_signal_overflow():
    with pytest.raises(FailureDiagnosisError, match="invalid_residual"):
        diagnose_failure({}, residual=float("nan"))
    with pytest.raises(FailureDiagnosisError, match="signals_must_be_mapping"):
        diagnose_failure({str(index): {} for index in range(33)})


def test_failure_diagnosis_covers_six_root_cause_layers():
    result = diagnose_failure({
        "semantic": {"status": "fail"},
        "structural_misspecification": {"status": "fail"},
        "parameter_identifiability": {"status": "fail"},
        "data": {"status": "fail"},
        "numerical_tolerance": {"status": "fail"},
        "resource": {"status": "fail"},
    })
    assert {row["category"] for row in result["explanations"]} == {"semantic", "structural", "parameter", "data", "numerical", "resource"}
