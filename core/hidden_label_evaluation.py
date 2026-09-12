"""Validate scores returned by external hidden-label evaluators.

The evaluator owns the labels and computes the metric.  This module only
imports a signed/hashed score record and produces descriptive summaries; it
never accepts predictions or ground-truth data as a shortcut.
"""

from __future__ import annotations

import math
from statistics import mean, median
from typing import Any, Iterable, Mapping


HIDDEN_SCORE_SCHEMA = "mathmodel.hidden-label-score/v1"
_STATUSES = frozenset({"completed", "failed", "rejected", "timeout"})
_DIRECTIONS = frozenset({"higher_is_better", "lower_is_better"})
_FORBIDDEN = frozenset({"labels", "ground_truth", "predictions", "prediction", "test_data", "answer"})


class HiddenLabelEvaluationError(ValueError):
    pass


def _text(value: Any, code: str, limit: int = 200) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > limit or any(ch.isspace() for ch in value.strip()):
        raise HiddenLabelEvaluationError(code)
    return value.strip()


def _digest(value: Any, code: str) -> str:
    value = _text(value, code, 64)
    if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise HiddenLabelEvaluationError(code)
    return value


def validate_hidden_label_score(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one score callback exported by a hidden-label evaluator."""
    if not isinstance(payload, Mapping):
        raise HiddenLabelEvaluationError("score_must_be_object")
    if set(payload) & _FORBIDDEN:
        raise HiddenLabelEvaluationError("ground_truth_or_prediction_forbidden")
    required = {"schema_version", "source_id", "benchmark_id", "task_id", "submission_id", "metric", "score_direction", "status", "test_fingerprint", "external_evaluator", "independent_evaluation_attested"}
    allowed = required | {"score"}
    if set(payload) - allowed or required - set(payload):
        raise HiddenLabelEvaluationError("score_fields_invalid")
    if payload.get("schema_version") != HIDDEN_SCORE_SCHEMA:
        raise HiddenLabelEvaluationError("score_schema_invalid")
    source_id = _text(payload["source_id"], "source_id_invalid")
    benchmark_id = _text(payload["benchmark_id"], "benchmark_id_invalid")
    task_id = _text(payload["task_id"], "task_id_invalid")
    submission_id = _text(payload["submission_id"], "submission_id_invalid")
    metric = _text(payload["metric"], "metric_invalid")
    direction = _text(payload["score_direction"], "score_direction_invalid")
    status = _text(payload["status"], "score_status_invalid")
    if direction not in _DIRECTIONS:
        raise HiddenLabelEvaluationError("score_direction_invalid")
    if status not in _STATUSES:
        raise HiddenLabelEvaluationError("score_status_invalid")
    fingerprint = _digest(payload["test_fingerprint"], "test_fingerprint_invalid")
    evaluator = _text(payload["external_evaluator"], "external_evaluator_invalid")
    if type(payload["independent_evaluation_attested"]) is not bool:
        raise HiddenLabelEvaluationError("independent_evaluation_attested_must_be_boolean")
    score = payload.get("score")
    if status == "completed":
        if type(score) not in (int, float) or isinstance(score, bool) or not math.isfinite(float(score)):
            raise HiddenLabelEvaluationError("completed_score_required")
        score = float(score)
    elif score is not None and (type(score) not in (int, float) or isinstance(score, bool) or not math.isfinite(float(score))):
        raise HiddenLabelEvaluationError("invalid_score")
    return {
        "schema_version": HIDDEN_SCORE_SCHEMA, "source_id": source_id, "benchmark_id": benchmark_id,
        "task_id": task_id, "submission_id": submission_id, "metric": metric,
        "score_direction": direction, "status": status, "score": score,
        "test_fingerprint": fingerprint, "external_evaluator": evaluator,
        "independent_evaluation_attested": payload["independent_evaluation_attested"],
    }


def summarize_hidden_label_scores(rows: Iterable[Mapping[str, Any]], *, min_tasks: int = 1) -> dict[str, Any]:
    """Return descriptive external-score statistics without claiming significance."""
    if type(min_tasks) is not int or not 1 <= min_tasks <= 100_000:
        raise HiddenLabelEvaluationError("invalid_min_tasks")
    clean = [validate_hidden_label_score(row) for row in rows]
    if not clean:
        raise HiddenLabelEvaluationError("scores_required")
    fingerprints = {row["test_fingerprint"] for row in clean}
    if len(fingerprints) != 1:
        raise HiddenLabelEvaluationError("test_fingerprint_mismatch")
    task_ids = [row["task_id"] for row in clean]
    if len(set(task_ids)) != len(task_ids):
        raise HiddenLabelEvaluationError("duplicate_task_id")
    completed = [float(row["score"]) for row in clean if row["status"] == "completed"]
    status_counts: dict[str, int] = {}
    for row in clean:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    result = {
        "schema_version": "mathmodel.hidden-label-summary/v1",
        "task_count": len(clean), "completed_count": len(completed), "status_counts": status_counts,
        "metric": clean[0]["metric"], "score_direction": clean[0]["score_direction"],
        "test_fingerprint": next(iter(fingerprints)),
        "scores": {"mean": mean(completed) if completed else None, "median": median(completed) if completed else None},
        "status": "descriptive" if len(completed) >= min_tasks else "insufficient_tasks",
        "policy": "external_hidden_label_score_only;_no_modeling_quality_or_significance_claim",
    }
    return result


__all__ = ["HIDDEN_SCORE_SCHEMA", "HiddenLabelEvaluationError", "validate_hidden_label_score", "summarize_hidden_label_scores"]
