import pytest

from core.diagnostic_comparison import DiagnosticComparisonError, compare_residual_explanations


def test_residual_explanations_rank_evidence_without_declaring_hidden_state():
    report = compare_residual_explanations({
        "numerical_tolerance": {"status": "does_not_support", "strength": .9},
        "observation_error": {"status": "supports", "strength": .7},
        "parameter_identifiability": {"status": "not_assessed"},
        "structural_missing": {"status": "supports", "strength": .4},
    })
    assert report["top_explanation"] == "observation_error"
    assert "hidden_state" in report["policy"]
    assert len(report["clarifying_questions"]) == 1


def test_residual_explanations_reject_nonfinite_strength():
    with pytest.raises(DiagnosticComparisonError, match="strength"):
        compare_residual_explanations({"numerical_tolerance": {"status": "supports", "strength": float("nan")}})
