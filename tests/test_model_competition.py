import pytest
import threading

from core.model_competition import ModelCompetitionError, compete_models, compete_models_staged
from core.staged_evaluation import StageSpec


def _candidate(identifier, loss, *, decision=None, status="candidate_evaluated"):
    return {
        "id": identifier,
        "status": status,
        "predictions": [0.0, float(loss), 1.0],
        "metrics": {
            "validation_loss": float(loss), "complexity": float(loss + 1),
            "constraint_violation": 0.0, "instability": float(loss / 10),
        },
        "decision": decision,
    }


def test_staged_competition_propagates_cancellation_without_comparison_approval():
    cancel = threading.Event()
    cancel.set()
    result = compete_models_staged(
        [{"id": "a"}], lambda *_: pytest.fail("cancelled evaluator must not run"),
        [StageSpec("confirm", 1, confirmation=True)], cancel=cancel,
    )
    assert result["status"] == "cancelled"
    assert result["competition"]["verdict"]["status"] != "approved"


def test_competition_preserves_pareto_and_does_not_auto_approve():
    result = compete_models([
        _candidate("simple", 1.0, decision="yes"),
        _candidate("stable", 2.0, decision="no"),
    ])
    assert result["pareto_candidate_ids"] == ["simple"]
    assert result["verdict"]["status"] == "conditional"
    assert result["verdict"]["recommended_candidate_id"] is None
    assert result["policy"]["scores_are_not_probabilities"] is True


def test_counterexample_blocks_explicit_approval_and_recomputes_status():
    candidate = _candidate("a", 1.0, decision="yes")
    candidate.update({
        "verdict_state": "approved", "evidence_refs": ["evidence/a"],
    })
    result = compete_models([candidate], counterexamples=[{"candidate_id": "a", "reason": "boundary"}])
    assert result["verdict"]["status"] == "unresolved"
    assert result["verdict"]["recommended_candidate_id"] is None
    assert result["verdict"]["approved_candidate_ids"] == []


def test_missing_comparison_fields_remain_unresolved():
    result = compete_models([{"id": "pending", "status": "pending"}])
    assert result["status"] == "candidate_set_inadequate"
    assert result["verdict"]["unresolved_candidate_ids"] == ["pending"]


def test_hard_failure_is_not_comparable_or_hidden():
    result = compete_models([_candidate("ok", 1.0), {"id": "bad", "status": "counterexample"}])
    assert "bad" in result["verdict"]["rejected_candidate_ids"]
    assert result["comparison"]["candidate_count"] == 1


def test_malformed_candidate_does_not_abort_other_candidate_comparison():
    result = compete_models([
        _candidate("ok", 1.0),
        {"id": "missing_axis", "status": "candidate_evaluated",
         "predictions": [0.0, 1.0, 2.0], "metrics": {"validation_loss": 0.1}},
    ])
    assert result["comparison"]["candidate_count"] == 1
    unresolved = next(item for item in result["verdict"]["candidates"] if item["id"] == "missing_axis")
    assert unresolved["status"] == "not_assessed"
    assert "missing_or_invalid_metric:complexity" in unresolved["failure_reasons"]


def test_prediction_length_mismatch_is_preserved_as_unresolved():
    result = compete_models([_candidate("a", 1.0), {**_candidate("b", 2.0), "predictions": [1.0]}])
    assert result["comparison"]["candidate_count"] == 1
    unresolved = next(item for item in result["verdict"]["candidates"] if item["id"] == "b")
    assert unresolved["status"] == "not_assessed"
    assert "prediction_lengths_must_match" in unresolved["failure_reasons"]


@pytest.mark.parametrize("candidates", [[], [{"id": "x"}, {"id": "x"}]])
def test_invalid_candidate_collection_is_rejected(candidates):
    with pytest.raises(ModelCompetitionError):
        compete_models(candidates)


def test_staged_competition_only_compares_finally_confirmed_candidates():
    candidates = [_candidate("good", 1.0), _candidate("bad", 2.0)]
    calls = []

    def evaluate(candidate, stage):
        calls.append((candidate["id"], stage.name))
        if candidate["id"] == "bad":
            return {"status": "counterexample", "score": 99, "violations": [{"reason": "boundary"}]}
        return {"status": "pass", "score": candidate["metrics"]["validation_loss"],
                "evaluations_used": stage.per_candidate_budget}

    result = compete_models_staged(
        candidates, evaluate,
        [StageSpec("cheap", 1, keep_fraction=1), StageSpec("confirm", 2, confirmation=True)],
        max_total_budget=5,
    )
    assert result["status"] == "completed"
    assert result["competition"]["comparison"]["candidate_count"] == 1
    assert result["competition"]["pareto_candidate_ids"] == ["good"]
    assert calls == [("good", "cheap"), ("bad", "cheap"), ("good", "confirm")]


def test_staged_competition_keeps_pruned_candidates_unresolved():
    result = compete_models_staged(
        [_candidate("a", 1.0), _candidate("b", 2.0)],
        lambda candidate, stage: {"status": "pass", "score": candidate["metrics"]["validation_loss"]},
        [StageSpec("cheap", 1, keep_fraction=0.5, min_survivors=1)],
    )
    assert result["competition"]["comparison"]["candidate_count"] == 1
    assert result["competition"]["verdict"]["unresolved_candidate_ids"] == ["b"]
