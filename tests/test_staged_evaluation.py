import pytest
import threading

from core.staged_evaluation import StageSpec, StagedEvaluationError, run_staged_evaluation


def test_staged_evaluation_prunes_hard_failures_and_keeps_confirmation_reserve():
    calls = []

    def evaluate(candidate, stage):
        calls.append((candidate["id"], stage.name))
        if candidate["id"] == "bad":
            return {"status": "counterexample", "score": 100, "evaluations_used": 1}
        if stage.confirmation:
            return {"status": "pass", "score": candidate["score"], "evaluations_used": 2}
        return {"status": "pass", "score": candidate["score"], "evaluations_used": 1}

    result = run_staged_evaluation(
        [{"id": "a", "score": 1}, {"id": "b", "score": 2}, {"id": "bad", "score": 0}],
        evaluate,
        [StageSpec("cheap", 1, keep_fraction=0.5, min_survivors=1),
         StageSpec("confirm", 2, min_survivors=1, confirmation=True)],
        max_total_budget=5,
    )
    assert result["status"] == "completed"
    assert result["accepted_ids"] == ["a"]
    assert "bad" not in result["survivor_ids"]
    assert result["budget"]["confirmation_reserved"] is True
    assert calls == [("a", "cheap"), ("b", "cheap"), ("bad", "cheap"), ("a", "confirm")]


def test_budget_exhaustion_is_not_recorded_as_counterexample():
    def evaluate(candidate, stage):
        return {"status": "pass", "score": candidate["score"], "evaluations_used": stage.per_candidate_budget}

    result = run_staged_evaluation(
        [{"id": "a", "score": 1}, {"id": "b", "score": 2}],
        evaluate,
        [StageSpec("cheap", 2, keep_fraction=1), StageSpec("confirm", 3, min_survivors=1, confirmation=True)],
        max_total_budget=5,
    )
    assert result["status"] == "budget_exhausted"
    assert any(item["status"] == "budget_exhausted" for item in result["reports"])
    assert result["policy"].startswith("budget_exhaustion")


def test_unresolved_evaluator_error_survives_non_confirmation_stage():
    def evaluate(candidate, stage):
        return {"status": "execution_error", "score": None}

    result = run_staged_evaluation(
        [{"id": "a"}], evaluate,
        [StageSpec("cheap", 1, keep_fraction=1), StageSpec("confirm", 1, confirmation=True)],
    )
    assert result["status"] == "completed"
    assert result["survivor_ids"] == ["a"]
    assert result["accepted_ids"] == []


def test_unresolved_candidate_is_not_dropped_by_fractional_pruning():
    def evaluate(candidate, stage):
        if candidate["id"] == "slow":
            return {"status": "execution_error"}
        return {"status": "pass", "score": candidate["score"]}

    result = run_staged_evaluation(
        [{"id": "slow", "score": 999}, {"id": "good", "score": 1}, {"id": "weak", "score": 5}],
        evaluate,
        [StageSpec("cheap", 1, keep_fraction=0.5, min_survivors=1)],
    )
    assert "slow" in result["survivor_ids"]
    slow_report = next(item for item in result["reports"] if item["candidate_id"] == "slow")
    assert slow_report["selection_status"] == "survives_stage"


def test_stage_contract_rejects_invalid_confirmation_and_budget():
    with pytest.raises(StagedEvaluationError, match="confirmation_stage"):
        run_staged_evaluation([{"id": "a"}], lambda *_: {"status": "pass"},
                              [StageSpec("confirm", 1, confirmation=True), StageSpec("later", 1)])
    with pytest.raises(StagedEvaluationError, match="confirmation_reserve"):
        run_staged_evaluation([{"id": "a"}], lambda *_: {"status": "pass"},
                              [StageSpec("confirm", 2, confirmation=True)], max_total_budget=1)


def test_staged_evaluation_rejects_coercion_and_preserves_invalid_usage_as_unresolved():
    with pytest.raises(StagedEvaluationError, match="invalid_total_wall"):
        run_staged_evaluation([{"id": "a"}], lambda *_: {"status": "pass"},
                              [StageSpec("cheap", 1)], max_wall_seconds="1")

    result = run_staged_evaluation(
        [{"id": "a"}], lambda *_: {"status": "pass", "score": 1, "evaluations_used": "1"},
        [StageSpec("cheap", 2)],
    )
    assert result["status"] == "completed"
    assert result["accepted_ids"] == []
    assert result["reports"][0]["status"] == "evaluation_error"


def test_staged_evaluation_cancellation_is_unresolved_not_rejection():
    cancel = threading.Event()
    cancel.set()
    result = run_staged_evaluation(
        [{"id": "a"}], lambda *_: pytest.fail("cancelled work must not start"),
        [StageSpec("cheap", 1)], cancel=cancel,
    )
    assert result["status"] == "cancelled"
    assert result["accepted_ids"] == []
    assert result["reports"][0]["selection_status"] == "unresolved_execution"
