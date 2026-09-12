import pytest

from core.conclusion_certificate import ConclusionCertificateError, build_conclusion_certificate


def test_certificate_keeps_boundary_residual_constraints_and_counterexamples():
    result = build_conclusion_certificate(
        applicability_boundary=["x in [0, 1]"],
        residuals=[{"id": "holdout", "rmse": 0.1}],
        constraint_checks=[{"id": "capacity", "max_violation": 0.0}],
        assumptions=["observations are aligned"],
    )
    assert result["status"] == "tested_not_falsified"
    assert result["applicability_boundary"]
    assert result["counterexamples"] == []
    with_counterexample = build_conclusion_certificate(
        applicability_boundary="x in [0, 1]", residuals=[{"value": 0.2}],
        constraint_checks=[{"violation": 1.0}], counterexamples=[{"id": "edge"}])
    assert with_counterexample["status"] == "counterexample_found"


def test_certificate_rejects_missing_numeric_evidence():
    with pytest.raises(ConclusionCertificateError, match="residual_value_required"):
        build_conclusion_certificate(applicability_boundary="declared", residuals=[{}], constraint_checks=[])
