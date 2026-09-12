"""Baseline-first comparison with an explicit search budget."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .baseline_registry import BaselineRegistry, default_baseline_registry


class BaselineEvaluationError(ValueError):
    pass


def baseline_first_report(task: str, baseline_value: Any, candidates: Sequence[Mapping[str, Any]], *,
                          registry: BaselineRegistry | None = None, max_candidates: int = 32) -> dict[str, Any]:
    if not isinstance(task, str) or not task.strip():
        raise BaselineEvaluationError("task_required")
    if type(max_candidates) is not int or not 0 <= max_candidates <= 64:
        raise BaselineEvaluationError("invalid_candidate_budget")
    if type(baseline_value) not in (int, float) or not math.isfinite(float(baseline_value)):
        raise BaselineEvaluationError("baseline_value_must_be_finite")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or len(candidates) > max_candidates:
        raise BaselineEvaluationError("candidate_budget_exceeded")
    registry = registry or default_baseline_registry()
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("id"), str) or not candidate["id"].strip():
            raise BaselineEvaluationError("candidate_id_required")
        value = candidate.get("value")
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            rows.append({"id": candidate["id"], "status": "not_assessed", "reason": "candidate_value_nonfinite"})
            continue
        rows.append({"id": candidate["id"], **registry.compare(task, float(value), float(baseline_value))})
    better = [row["id"] for row in rows if row.get("status") == "candidate_better"]
    return {
        "schema_version": "mathmodel.baseline-evaluation/v1", "task": task,
        "baseline": {"value": float(baseline_value), "registry_digest": registry.digest},
        "candidates": rows, "candidate_better_ids": better,
        "status": "baseline_only" if not rows else ("candidate_improves" if better else "baseline_not_beaten"),
        "policy": "baseline_is_displayed_before_search; candidate_improvement_does_not_bypass_hard_evidence_or_validation",
    }


__all__ = ["BaselineEvaluationError", "baseline_first_report"]
