import numpy as np
import pytest

from core.model_set_assessment import ModelSetAssessmentError, assess_model_set


def _candidate(identifier, loss, complexity, prediction, decision):
    return {"id": identifier, "predictions": prediction, "decision": decision,
            "metrics": {"validation_loss": loss, "complexity": complexity,
                         "constraint_violation": 0.0, "instability": 0.0}}


def test_model_set_assessment_keeps_pareto_disagreement_and_envelope():
    result = assess_model_set([
        _candidate("simple", 1.0, 3.0, [1, 2, 3], "accept"),
        _candidate("flexible", 0.8, 5.0, [1.2, 2.5, 3.1], "reject"),
        _candidate("dominated", 2.0, 6.0, [0, 0, 0], "reject"),
    ])
    assert result["status"] == "model_disagreement"
    assert set(result["pareto_candidate_ids"]) == {"simple", "flexible"}
    assert result["prediction_envelope"]["lower"] == [1.0, 2.0, 3.0]
    assert result["decision_consensus"] is False
    assert result["uncertainty_decomposition"]["parameter"] == "not_assessed"


def test_model_set_assessment_reports_consensus_without_posterior_claim():
    result = assess_model_set([_candidate("a", 1, 1, [2, 3], "same"),
                               _candidate("b", 1.2, 2, [2.1, 3.1], "same")])
    assert result["status"] == "decision_consensus"
    assert result["decision_consensus"] is True
    assert "posterior" in result["policy"]


def test_model_set_assessment_rejects_malformed_candidates():
    with pytest.raises(ModelSetAssessmentError, match="missing_or_invalid_metric"):
        assess_model_set([{"id": "bad", "predictions": [1], "metrics": {}}])


def test_compute_cost_is_used_only_when_all_candidates_report_it():
    left = _candidate("fast", 1.0, 2.0, [1, 2], "same")
    right = _candidate("slow", 1.0, 2.0, [1, 2], "same")
    left["metrics"]["compute_cost"] = 1.0
    right["metrics"]["compute_cost"] = 10.0
    result = assess_model_set([left, right])
    assert result["comparison_axes"][-1] == "compute_cost"
    assert result["pareto_candidate_ids"] == ["fast"]


def test_absent_decisions_are_not_reported_as_consensus():
    result = assess_model_set([
        _candidate("a", 1.0, 1.0, [1, 2], None),
        _candidate("b", 1.1, 2.0, [1, 2], None),
    ])
    assert result["status"] == "decision_not_assessed"
    assert result["decision_assessed"] is False
    assert result["decision_consensus"] is False


def test_model_set_assessment_validates_budgets_before_numeric_comparisons():
    with pytest.raises(ModelSetAssessmentError, match="candidate_budget"):
        assess_model_set([_candidate("a", 1, 1, [1], "same")], max_candidates="2")
    with pytest.raises(ModelSetAssessmentError, match="prediction_budget"):
        assess_model_set([_candidate("a", 1, 1, [1], "same")], max_prediction_points=True)
