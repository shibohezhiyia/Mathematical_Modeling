"""Paired arm comparison runner for pre-registered model experiments.

This module is deliberately evaluator-agnostic.  It gives the same task grid,
budget and seed to every arm, keeps failures in the denominator, and only
produces descriptive paired effects.  It does not claim that a score is a
truth label or that a confidence interval proves significance.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Callable, Mapping, Sequence

from .benchmark_statistics import paired_benchmark_effect


class ComparisonProtocolError(ValueError):
    pass


_STATUSES = frozenset({"completed", "timeout", "rejected", "incomplete", "error"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def run_preregistered_comparison(
    manifest: Mapping[str, Any],
    tasks: Sequence[Mapping[str, Any]],
    evaluate: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Run every arm on every task with a fixed paired grid.

    ``evaluate`` receives ``(arm, task, context)``.  The context exposes only
    the pre-registered budget/seed and a monotonic elapsed value; callers must
    not use the final-test payload as a training input.  A malformed evaluator
    response becomes an ``error`` row so later arms still run.
    """
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "mathmodel.ablation-manifest/v1":
        raise ComparisonProtocolError("manifest_required")
    arms = manifest.get("arms")
    task_ids = manifest.get("task_ids")
    if not isinstance(arms, Sequence) or not arms or not isinstance(task_ids, Sequence) or not task_ids:
        raise ComparisonProtocolError("manifest_grid_required")
    if (isinstance(task_ids, (str, bytes)) or any(type(item) is not str or not item.strip() for item in task_ids)
            or len(set(task_ids)) != len(task_ids)):
        raise ComparisonProtocolError("task_ids_must_be_unique_strings")
    if not callable(evaluate):
        raise ComparisonProtocolError("evaluator_required")
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)):
        raise ComparisonProtocolError("tasks_required")
    by_id = {}
    for task in tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("id"), str) or not task["id"].strip():
            raise ComparisonProtocolError("invalid_task")
        identifier = task["id"].strip()
        if identifier in by_id:
            raise ComparisonProtocolError("duplicate_task_id")
        by_id[identifier] = dict(task)
    if set(task_ids) != set(by_id):
        raise ComparisonProtocolError("task_grid_mismatch")
    if wall_seconds is not None and (type(wall_seconds) not in (int, float) or not math.isfinite(float(wall_seconds)) or float(wall_seconds) <= 0):
        raise ComparisonProtocolError("invalid_wall_budget")
    started = time.monotonic()
    budget = dict(manifest.get("fixed_budget", {}))
    seed = budget.get("seed")
    rows = []
    seen_arms = set()
    for arm in arms:
        if not isinstance(arm, Mapping) or not isinstance(arm.get("id"), str) or not arm["id"].strip():
            raise ComparisonProtocolError("invalid_arm")
        if arm["id"].strip() in seen_arms:
            raise ComparisonProtocolError("arm_ids_must_be_unique")
        seen_arms.add(arm["id"].strip())
        for task_id in task_ids:
            task = by_id[task_id]
            if wall_seconds is not None and time.monotonic() - started >= float(wall_seconds):
                rows.append({"arm_id": arm["id"], "task_id": task_id, "status": "timeout", "valid": False, "score": None})
                continue
            tick = time.monotonic()
            context = {"budget": dict(budget), "seed": seed, "elapsed_seconds": max(0.0, tick - started),
                       "final_test_fingerprint": manifest.get("final_test_fingerprint"),
                       "policy": "paired_grid_context; evaluator_must_not_train_on_final_test"}
            try:
                result = evaluate(dict(arm), dict(task), context)
                if not isinstance(result, Mapping):
                    raise ComparisonProtocolError("evaluator_must_return_mapping")
                status = str(result.get("status", "completed"))
                if status not in _STATUSES:
                    raise ComparisonProtocolError("result_status_invalid")
                score = result.get("score") if status == "completed" else None
                if score is not None and (type(score) not in (int, float) or not math.isfinite(float(score))):
                    raise ComparisonProtocolError("score_invalid")
                valid = result.get("valid", False)
                if type(valid) is not bool:
                    raise ComparisonProtocolError("valid_must_be_boolean")
                rows.append({"arm_id": arm["id"], "task_id": task_id, "status": status,
                             "valid": valid if status == "completed" else False,
                             "score": float(score) if score is not None else None,
                             "duration_seconds": float(time.monotonic() - tick),
                             "error_code": result.get("error_code")})
            except ComparisonProtocolError as exc:
                rows.append({"arm_id": arm["id"], "task_id": task_id, "status": "error", "valid": False,
                             "score": None, "duration_seconds": float(time.monotonic() - tick), "error_code": str(exc)})
            except Exception as exc:  # evaluator failures must not erase denominator rows
                rows.append({"arm_id": arm["id"], "task_id": task_id, "status": "error", "valid": False,
                             "score": None, "duration_seconds": float(time.monotonic() - tick),
                             "error_code": type(exc).__name__})
    counts = {status: sum(row["status"] == status for row in rows) for status in sorted(_STATUSES)}
    return {"schema_version": "mathmodel.comparison-results/v1", "status": "assessed" if not any(row["status"] != "completed" for row in rows) else "incomplete",
            "manifest_digest": _digest(manifest), "rows": rows, "status_counts": counts,
            "expected_rows": len(arms) * len(task_ids), "observed_rows": len(rows),
            "policy": "same_task_arm_grid; failures_remain_in_denominator; descriptive_effects_only"}


def summarize_paired_comparison(results: Mapping[str, Any], *, baseline_arm: str,
                                treatment_arm: str, score_direction: str = "higher_is_better",
                                min_samples: int = 5) -> dict[str, Any]:
    """Summarize completed paired scores without turning them into a claim."""
    if not isinstance(results, Mapping) or results.get("schema_version") != "mathmodel.comparison-results/v1":
        raise ComparisonProtocolError("results_required")
    if score_direction not in {"higher_is_better", "lower_is_better"}:
        raise ComparisonProtocolError("score_direction_invalid")
    rows = results.get("rows", [])
    if not isinstance(baseline_arm, str) or not isinstance(treatment_arm, str) or baseline_arm == treatment_arm:
        raise ComparisonProtocolError("arm_names_invalid")
    base = {row["task_id"]: row["score"] for row in rows if row.get("arm_id") == baseline_arm and row.get("status") == "completed" and row.get("score") is not None}
    treatment = {row["task_id"]: row["score"] for row in rows if row.get("arm_id") == treatment_arm and row.get("status") == "completed" and row.get("score") is not None}
    common = sorted(set(base) & set(treatment))
    samples = [{"baseline": base[key], "treatment": treatment[key]} for key in common]
    effect = paired_benchmark_effect(samples, min_samples=min_samples) if samples else {"sample_count": 0, "status": "not_assessed", "confidence_interval": None}
    raw_delta = effect.get("paired_effect")
    adjusted = (-float(raw_delta) if raw_delta is not None and score_direction == "lower_is_better" else raw_delta)
    return {"schema_version": "mathmodel.comparison-summary/v1", "baseline_arm": baseline_arm,
            "treatment_arm": treatment_arm, "paired_task_count": len(common),
            "paired_effect": adjusted, "raw_paired_effect": raw_delta,
            "score_direction": score_direction, "statistics": effect,
            "status": "descriptive_only" if effect.get("status") != "assessed" else "assessed_not_significant",
            "policy": "no_claim_of_generalization_or_significance_without_preregistered_truth_labels_and_independent_test"}


__all__ = ["ComparisonProtocolError", "run_preregistered_comparison", "summarize_paired_comparison"]
