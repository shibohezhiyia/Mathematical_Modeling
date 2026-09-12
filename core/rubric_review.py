"""Independent rubric review records for blind mathematical-model benchmarks."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class RubricReviewError(ValueError):
    pass


def build_rubric(criteria: Sequence[str], *, scale_max: int = 4, minimum_reviewers: int = 2) -> dict[str, Any]:
    if not isinstance(criteria, Sequence) or isinstance(criteria, (str, bytes)) or not criteria:
        raise RubricReviewError("criteria_required")
    if len(criteria) > 32 or any(not isinstance(item, str) or not item.strip() for item in criteria):
        raise RubricReviewError("criteria_invalid")
    if type(scale_max) is not int or not 1 <= scale_max <= 100:
        raise RubricReviewError("scale_max_invalid")
    if type(minimum_reviewers) is not int or not 2 <= minimum_reviewers <= 8:
        raise RubricReviewError("minimum_reviewers_invalid")
    names = list(dict.fromkeys(item.strip() for item in criteria))
    if len(names) != len(criteria):
        raise RubricReviewError("duplicate_criterion")
    return {"schema_version": "mathmodel.review-rubric/v1", "criteria": names,
            "scale": {"min": 0, "max": scale_max, "integer": True},
            "minimum_reviewers": minimum_reviewers,
            "policy": "independent_review_required; rubric_score_is_not_ground_truth_without_adjudication"}


def record_review(rubric: Mapping[str, Any], *, case_id: str, evaluator_id: str,
                  scores: Mapping[str, Any], notes: str = "") -> dict[str, Any]:
    if not isinstance(rubric, Mapping) or rubric.get("schema_version") != "mathmodel.review-rubric/v1":
        raise RubricReviewError("rubric_required")
    if not isinstance(case_id, str) or not case_id.strip() or not isinstance(evaluator_id, str) or not evaluator_id.strip():
        raise RubricReviewError("review_identity_required")
    criteria = rubric.get("criteria", [])
    if not isinstance(scores, Mapping) or set(scores) != set(criteria):
        raise RubricReviewError("score_criteria_mismatch")
    maximum = rubric.get("scale", {}).get("max")
    clean = {}
    for criterion in criteria:
        value = scores[criterion]
        if type(value) is not int or not 0 <= value <= maximum:
            raise RubricReviewError("score_out_of_range")
        clean[criterion] = value
    if not isinstance(notes, str) or len(notes) > 4000:
        raise RubricReviewError("notes_invalid")
    return {"schema_version": "mathmodel.review/v1", "case_id": case_id.strip(),
            "evaluator_id": evaluator_id.strip(), "scores": clean, "notes": notes}


def aggregate_reviews(rubric: Mapping[str, Any], reviews: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(rubric, Mapping) or rubric.get("schema_version") != "mathmodel.review-rubric/v1":
        raise RubricReviewError("rubric_required")
    if not isinstance(reviews, Sequence) or isinstance(reviews, (str, bytes)):
        raise RubricReviewError("reviews_required")
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    seen = set()
    for review in reviews:
        if not isinstance(review, Mapping) or review.get("schema_version") != "mathmodel.review/v1":
            raise RubricReviewError("invalid_review")
        key = (review.get("case_id"), review.get("evaluator_id"))
        if not all(isinstance(item, str) and item for item in key):
            raise RubricReviewError("review_identity_required")
        if key in seen:
            raise RubricReviewError("duplicate_review")
        seen.add(key)
        grouped.setdefault(key[0], []).append(review)
    minimum = rubric["minimum_reviewers"]
    rows = []
    for case_id, entries in sorted(grouped.items()):
        complete = len(entries) >= minimum
        means = {}
        disagreements = []
        for criterion in rubric["criteria"]:
            values = [entry["scores"][criterion] for entry in entries]
            means[criterion] = sum(values) / len(values) if values else None
            if len(values) >= 2:
                disagreements.append(max(values) - min(values))
        rows.append({"case_id": case_id, "reviewer_count": len(entries), "status": "ready" if complete else "needs_review",
                     "mean_scores": means, "max_criterion_disagreement": max(disagreements, default=None)})
    ready = bool(rows) and all(row["status"] == "ready" for row in rows)
    return {"schema_version": "mathmodel.review-aggregate/v1", "status": "ready_for_adjudication" if ready else "incomplete",
            "case_count": len(rows), "rows": rows,
            "policy": "means_are_descriptive; independent reviewer agreement and adjudication remain required"}


__all__ = ["RubricReviewError", "build_rubric", "record_review", "aggregate_reviews"]
