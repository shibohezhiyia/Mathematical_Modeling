import pytest

from core.baseline_evaluation import BaselineEvaluationError, baseline_first_report


def test_baseline_is_reported_before_candidate_search():
    result = baseline_first_report("regression", 2.0, [{"id": "simple", "value": 1.5}, {"id": "bad", "value": 2.5}])
    assert result["status"] == "candidate_improves"
    assert result["candidate_better_ids"] == ["simple"]
    assert result["baseline"]["value"] == 2.0


def test_baseline_evaluation_keeps_invalid_candidates_unassessed():
    result = baseline_first_report("regression", 2.0, [{"id": "unknown", "value": float("nan")}])
    assert result["candidates"][0]["status"] == "not_assessed"
    with pytest.raises(BaselineEvaluationError):
        baseline_first_report("regression", 2.0, [{"id": "x", "value": 1}], max_candidates=0)
