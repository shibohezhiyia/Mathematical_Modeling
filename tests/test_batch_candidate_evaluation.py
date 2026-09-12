import pytest

from core.batch_candidate_evaluation import (
    BatchEvaluationConfig,
    BatchEvaluationError,
    evaluate_candidate_batch,
)


def test_batch_evaluation_shares_feature_views_and_projects_feedback():
    calls = []

    def build(key, candidates):
        calls.append((key, tuple(item["id"] for item in candidates)))
        return {"key": key, "count": len(candidates)}

    def evaluate(candidate, view):
        return {"status": "executed", "metrics": {"loss": view["count"]},
                "predictions": ["must_not_be_forwarded"], "diagnostic": "x" * 10_000}

    result = evaluate_candidate_batch([
        {"id": "a", "feature_key": "shared", "secret": "one"},
        {"id": "b", "feature_key": "shared", "secret": "two"},
        {"id": "c", "feature_key": "other"},
    ], build, evaluate)
    assert result["feature_view_count"] == 2
    assert result["shared_feature_hits"] == 1
    assert len(calls) == 2
    assert all("predictions" not in row["result"] for row in result["results"])
    assert len(result["results"][0]["result"]["diagnostic"]) == 2_000


def test_retryable_results_are_planned_but_not_hidden_reexecutions():
    evaluations = []

    def evaluate(candidate, _view):
        evaluations.append(candidate["id"])
        return {"status": "retryable", "failure_code": "worker_busy"}

    result = evaluate_candidate_batch(
        [{"id": "a", "feature_key": "f"}], lambda _key, _items: object(), evaluate,
        config=BatchEvaluationConfig(max_retries=2),
    )
    assert evaluations == ["a"]
    assert result["retry_plan"] == [{"id": "a", "attempts_allowed": 1, "backoff_seconds": [1]}]


def test_batch_contract_and_feature_budget_are_strict():
    with pytest.raises(BatchEvaluationError, match="duplicate_candidate_id"):
        evaluate_candidate_batch([
            {"id": "a", "feature_key": "f"}, {"id": "a", "feature_key": "g"}
        ], lambda _key, _items: None, lambda _candidate, _view: {})
    with pytest.raises(BatchEvaluationError, match="feature_view_budget_exceeded"):
        evaluate_candidate_batch(
            [{"id": str(i), "feature_key": str(i)} for i in range(3)],
            lambda _key, _items: None, lambda _candidate, _view: {},
            config=BatchEvaluationConfig(max_feature_views=2),
        )
