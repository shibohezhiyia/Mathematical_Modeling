"""Conditional information-value calculations for active clarification.

The calculation is exact for the supplied categorical hypothesis weights and
answer partitions.  Those weights are assumptions, not automatically learned
posteriors; the output therefore carries that boundary explicitly.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class InformationValueError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _entropy(weights: Mapping[str, float]) -> float:
    total = sum(weights.values())
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log(value / total, 2)
                for value in weights.values() if value > 0)


def estimate_question_information_gain(
    hypotheses: Mapping[str, float],
    questions: Sequence[Mapping[str, Any]],
    *,
    max_questions: int = 32,
) -> dict[str, Any]:
    """Estimate expected entropy reduction from declared answer partitions."""
    if not isinstance(hypotheses, Mapping) or not hypotheses or len(hypotheses) > 128:
        raise InformationValueError("invalid_hypotheses")
    weights = {}
    for identifier, value in hypotheses.items():
        if type(identifier) is not str or not identifier.strip():
            raise InformationValueError("invalid_hypothesis_weight")
        identifier = identifier.strip()
        if (identifier in weights or type(value) not in (int, float) or
                not math.isfinite(float(value)) or value <= 0):
            raise InformationValueError("invalid_hypothesis_weight")
        weights[identifier] = float(value)
    if type(max_questions) is not int or not 1 <= max_questions <= 256:
        raise InformationValueError("invalid_question_budget")
    if not isinstance(questions, Sequence) or isinstance(questions, (str, bytes)) or not 1 <= len(questions) <= max_questions:
        raise InformationValueError("invalid_question_set")
    total = sum(weights.values())
    prior_entropy = _entropy(weights)
    rows = []
    for index, question in enumerate(questions):
        if not isinstance(question, Mapping) or type(question.get("question")) is not str or not question["question"].strip():
            raise InformationValueError("invalid_question")
        # Cost is an explicit acquisition/interaction budget, never an
        # implicit probability.  Keeping it on each question lets the UI
        # prefer a slightly less informative question when it is materially
        # cheaper to answer.  Omitted costs retain the historical unit cost.
        raw_cost = question.get("cost", 1.0)
        if (type(raw_cost) not in (int, float) or isinstance(raw_cost, bool) or
                not math.isfinite(float(raw_cost)) or float(raw_cost) <= 0):
            raise InformationValueError("invalid_question_cost")
        cost = float(raw_cost)
        answer_map = question.get("answers")
        if not isinstance(answer_map, Mapping) or not 2 <= len(answer_map) <= 8:
            raise InformationValueError("invalid_answer_partition")
        assigned: set[str] = set()
        branches: list[dict[str, Any]] = []
        for answer, ids in answer_map.items():
            if (type(answer) is not str or not answer.strip() or not isinstance(ids, Sequence) or
                    isinstance(ids, (str, bytes)) or any(type(identifier) is not str for identifier in ids)):
                raise InformationValueError("invalid_answer_branch")
            branch_ids = list(dict.fromkeys(identifier.strip() for identifier in ids))
            if not branch_ids or any(identifier not in weights for identifier in branch_ids):
                raise InformationValueError("unknown_answer_hypothesis")
            if assigned.intersection(branch_ids):
                raise InformationValueError("overlapping_answer_partition")
            assigned.update(branch_ids)
            branch_weights = {identifier: weights[identifier] for identifier in branch_ids}
            probability = sum(branch_weights.values()) / total
            branches.append({"answer": answer, "probability": probability,
                             "entropy": _entropy(branch_weights),
                             "hypothesis_count": len(branch_weights)})
        uncovered = set(weights) - assigned
        if uncovered:
            branch_weights = {identifier: weights[identifier] for identifier in uncovered}
            branches.append({"answer": "__uncovered__", "probability": sum(branch_weights.values()) / total,
                             "entropy": _entropy(branch_weights), "hypothesis_count": len(branch_weights)})
        expected_after = sum(item["probability"] * item["entropy"] for item in branches)
        gain = max(0.0, prior_entropy - expected_after)
        rows.append({"index": index, "question": question["question"],
                     "prior_entropy_bits": prior_entropy,
                     "expected_posterior_entropy_bits": expected_after,
                     "expected_information_gain_bits": gain,
                     "cost": cost,
                     "gain_per_cost": gain / cost,
                     "branches": branches})
    # Utility is information gained per declared unit cost.  The raw gain is
    # retained so callers can choose a different policy without recomputing
    # the partitions.  Ties are deterministic and preserve input order.
    rows.sort(key=lambda item: (-item["gain_per_cost"], -item["expected_information_gain_bits"], item["index"]))
    return {
        "schema_version": "mathmodel.information-value/v1",
        "questions": rows,
        "hypothesis_count": len(weights),
        "prior_entropy_bits": prior_entropy,
        "method": "categorical_expected_entropy_reduction",
        "ranking": "expected_information_gain_bits_per_declared_cost",
        "policy": "conditional_on_declared_weights_and_answer_partition_not_posterior_truth",
    }


__all__ = ["InformationValueError", "estimate_question_information_gain"]
