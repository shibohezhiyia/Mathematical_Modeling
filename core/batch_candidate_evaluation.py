"""Bounded batch evaluation for structurally related model candidates.

This orchestration layer builds an expensive feature view once for candidates
that declare the same ``feature_key`` and keeps only a small diagnostic
projection for repair prompts. It is not a solver and never promotes a
candidate or bypasses validation. Untrusted generated code must be executed
through :mod:`core.solver_worker` before reaching this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


class BatchEvaluationError(ValueError):
    """Raised when a batch contract is malformed or exceeds its budget."""


@dataclass(frozen=True)
class BatchEvaluationConfig:
    max_candidates: int = 64
    max_feature_views: int = 32
    max_retries: int = 2
    max_feedback_chars: int = 2_000

    def validate(self) -> "BatchEvaluationConfig":
        if type(self.max_candidates) is not int or not 1 <= self.max_candidates <= 256:
            raise BatchEvaluationError("invalid_candidate_budget")
        if type(self.max_feature_views) is not int or not 1 <= self.max_feature_views <= 128:
            raise BatchEvaluationError("invalid_feature_view_budget")
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 8:
            raise BatchEvaluationError("invalid_retry_budget")
        if type(self.max_feedback_chars) is not int or not 128 <= self.max_feedback_chars <= 16_000:
            raise BatchEvaluationError("invalid_feedback_budget")
        return self


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), default=str).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise BatchEvaluationError("candidate_not_serializable") from exc
    if len(encoded) > 64_000:
        raise BatchEvaluationError("candidate_payload_too_large")
    return hashlib.sha256(encoded).hexdigest()[:24]


def _finite_metrics(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    metrics: dict[str, float] = {}
    for key, raw in value.items():
        if len(metrics) >= 32:
            break
        try:
            number = float(raw)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(number):
            metrics[str(key)[:80]] = number
    return metrics


def _diagnostic_projection(result: Mapping[str, Any], *, max_chars: int) -> dict[str, Any]:
    status = str(result.get("status", "not_assessed"))
    if status not in {"executed", "accepted", "rejected", "failed", "not_assessed", "retryable"}:
        status = "not_assessed"
    projection: dict[str, Any] = {
        "status": status,
        "failure_code": str(result.get("failure_code", ""))[:120] or None,
        "metrics": _finite_metrics(result.get("metrics")),
    }
    hint = result.get("diagnostic")
    if isinstance(hint, str) and hint:
        projection["diagnostic"] = hint[:max_chars]
    return projection


def evaluate_candidate_batch(
    candidates: Sequence[Mapping[str, Any]],
    feature_builder: Callable[[str, Sequence[Mapping[str, Any]]], Any],
    evaluator: Callable[[Mapping[str, Any], Any], Mapping[str, Any]],
    *,
    config: BatchEvaluationConfig | None = None,
) -> dict[str, Any]:
    """Evaluate candidates with shared feature views and bounded feedback.

    ``feature_builder`` runs once for each distinct feature key. A result
    marked ``retryable`` is recorded in a retry plan but is not automatically
    re-executed; the caller may dispatch a later worker retry under its own
    global budget. This prevents hidden sleeps or unbounded API loops.
    """
    cfg = (config or BatchEvaluationConfig()).validate()
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise BatchEvaluationError("candidates_must_be_a_sequence")
    if not 1 <= len(candidates) <= cfg.max_candidates:
        raise BatchEvaluationError("candidate_count_out_of_bounds")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    identifiers: set[str] = set()
    normalized: list[Mapping[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            raise BatchEvaluationError("candidate_must_be_a_mapping")
        identifier = item.get("id")
        feature_key = item.get("feature_key")
        if not isinstance(identifier, str) or not identifier.strip():
            raise BatchEvaluationError("candidate_id_required")
        if identifier in identifiers:
            raise BatchEvaluationError("duplicate_candidate_id")
        if not isinstance(feature_key, str) or not feature_key.strip():
            raise BatchEvaluationError("feature_key_required")
        identifiers.add(identifier)
        copied = dict(item)
        normalized.append(copied)
        groups.setdefault(feature_key, []).append(copied)
    if len(groups) > cfg.max_feature_views:
        raise BatchEvaluationError("feature_view_budget_exceeded")

    views: dict[str, Any] = {}
    feature_failures: dict[str, str] = {}
    for key, members in groups.items():
        try:
            views[key] = feature_builder(key, tuple(members))
        except Exception as exc:  # trusted callback boundary; do not leak traceback
            feature_failures[key] = type(exc).__name__[:80]

    rows: list[dict[str, Any]] = []
    retry_plan: list[dict[str, Any]] = []
    for candidate in normalized:
        identifier = str(candidate["id"])
        feature_key = str(candidate["feature_key"])
        if feature_key in feature_failures:
            projected = {"status": "not_assessed", "failure_code": "feature_build_failed",
                         "metrics": {}, "diagnostic": feature_failures[feature_key]}
        else:
            try:
                raw = evaluator(candidate, views[feature_key])
                if not isinstance(raw, Mapping):
                    raise BatchEvaluationError("evaluator_result_must_be_a_mapping")
                projected = _diagnostic_projection(raw, max_chars=cfg.max_feedback_chars)
            except Exception as exc:  # do not turn callback errors into a claim
                projected = {"status": "not_assessed", "failure_code": "evaluator_failed",
                             "metrics": {}, "diagnostic": type(exc).__name__[:80]}
        retryable = projected["status"] == "retryable"
        planned = min(cfg.max_retries, 1) if retryable else 0
        if retryable:
            retry_plan.append({"id": identifier, "attempts_allowed": planned,
                               "backoff_seconds": [1, 2][:planned]})
        rows.append({"id": identifier, "candidate_hash": _digest(candidate),
                     "feature_key": feature_key, "result": projected,
                     "retry_planned": planned > 0})

    return {
        "schema_version": "mathmodel.batch-candidate-evaluation/v1",
        "status": "completed",
        "candidate_count": len(rows),
        "feature_view_count": len(groups),
        "shared_feature_hits": len(rows) - len(groups),
        "feature_failures": dict(feature_failures),
        "results": rows,
        "retry_plan": retry_plan,
        "feedback_policy": "diagnostic_projection_only; raw_predictions_and_tracebacks_are_not_forwarded",
        "execution_policy": "batch_orchestration_does_not_promote_candidates_or_bypass_validation",
    }


__all__ = ["BatchEvaluationConfig", "BatchEvaluationError", "evaluate_candidate_batch"]
