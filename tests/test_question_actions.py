import pytest

from core.question_actions import QuestionActionError, classify_question_action


def test_candidate_generated_data_cannot_be_independent_evidence():
    result = classify_question_action({"id": "q", "evidence_action": "trusted_simulation", "candidate_generated_data": True})
    assert result["status"] == "not_independent_evidence"
    assert result["independent_evidence"] is False


def test_unknown_action_is_rejected():
    with pytest.raises(QuestionActionError):
        classify_question_action({"id": "q", "evidence_action": "guess"})


@pytest.mark.parametrize("payload, code", [
    ({"id": "q", "evidence_action": "collect_data", "cost": 0}, "invalid_question_cost"),
    ({"id": "q", "evidence_action": "collect_data", "candidate_generated_data": "yes"},
     "candidate_generated_data_must_be_boolean"),
    ({"id": "q", "evidence_action": "collect_data", "changes_candidate_decision": 1},
     "changes_candidate_decision_must_be_boolean"),
    ({"id": 7, "evidence_action": "collect_data"}, "question_id_required"),
])
def test_question_action_rejects_ambiguous_control_types(payload, code):
    with pytest.raises(QuestionActionError, match=code):
        classify_question_action(payload)


def test_question_action_normalizes_id_and_cost():
    result = classify_question_action({"id": "  q  ", "evidence_action": "collect_data", "cost": 2})
    assert result["id"] == "q"
    assert result["cost"] == 2.0
