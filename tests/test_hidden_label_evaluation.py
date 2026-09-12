import pytest

from core.hidden_label_evaluation import HiddenLabelEvaluationError, summarize_hidden_label_scores, validate_hidden_label_score


def _row(task="task-1", score=0.8, status="completed"):
    return {
        "schema_version": "mathmodel.hidden-label-score/v1",
        "source_id": "drivendata-hidden-labels",
        "benchmark_id": "demo-competition",
        "task_id": task,
        "submission_id": "submission-1",
        "metric": "rmse",
        "score_direction": "lower_is_better",
        "status": status,
        "score": score,
        "test_fingerprint": "a" * 64,
        "external_evaluator": "platform",
        "independent_evaluation_attested": True,
    }


def test_hidden_score_validates_and_summarizes_without_predictions():
    first = validate_hidden_label_score(_row())
    assert first["score"] == 0.8
    result = summarize_hidden_label_scores([_row(), _row("task-2", 0.6)])
    assert result["completed_count"] == 2
    assert result["scores"]["mean"] == pytest.approx(0.7)
    assert result["status"] == "descriptive"


def test_hidden_score_rejects_ground_truth_and_duplicate_tasks():
    with pytest.raises(HiddenLabelEvaluationError, match="ground_truth_or_prediction_forbidden"):
        validate_hidden_label_score({**_row(), "predictions": [1]})
    with pytest.raises(HiddenLabelEvaluationError, match="duplicate_task_id"):
        summarize_hidden_label_scores([_row(), _row()])
