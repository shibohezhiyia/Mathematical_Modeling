"""Actionability contract for high-value clarification questions."""

from __future__ import annotations

import math
from typing import Any, Mapping


class QuestionActionError(ValueError):
    pass


def classify_question_action(question: Mapping[str, Any]) -> dict[str, Any]:
    if (not isinstance(question, Mapping) or type(question.get("id")) is not str or
            not question["id"].strip() or len(question["id"].strip()) > 128):
        raise QuestionActionError("question_id_required")
    action = question.get("evidence_action")
    if type(action) is not str:
        raise QuestionActionError("unsupported_evidence_action")
    action = action.strip()
    if action not in {"collect_data", "trusted_simulation", "fixed_attachment", "user_semantics"}:
        raise QuestionActionError("unsupported_evidence_action")
    produced_by_candidate = question.get("candidate_generated_data", False)
    if type(produced_by_candidate) is not bool:
        raise QuestionActionError("candidate_generated_data_must_be_boolean")
    changes_decision = question.get("changes_candidate_decision", False)
    if type(changes_decision) is not bool:
        raise QuestionActionError("changes_candidate_decision_must_be_boolean")
    raw_cost = question.get("cost")
    if raw_cost is not None and (type(raw_cost) not in (int, float) or
                                 isinstance(raw_cost, bool) or not math.isfinite(float(raw_cost)) or
                                 float(raw_cost) <= 0):
        raise QuestionActionError("invalid_question_cost")
    cost = None if raw_cost is None else float(raw_cost)
    return {"schema_version": "mathmodel.question-action/v1", "id": question["id"].strip(),
            "evidence_action": action, "cost": cost,
            "changes_candidate_decision": changes_decision,
            "candidate_generated_data": produced_by_candidate,
            "independent_evidence": action != "fixed_attachment" and not produced_by_candidate,
            "status": "not_independent_evidence" if produced_by_candidate else "actionable",
            "policy": "candidate_generated_data_can_test_internal_behavior_but_cannot_prove_reality"}


__all__ = ["QuestionActionError", "classify_question_action"]
