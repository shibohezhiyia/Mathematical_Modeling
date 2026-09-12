"""Promotion gates for low/high-fidelity candidate evaluation."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class ProgressiveBudgetError(ValueError):
    pass


def plan_progressive_promotion(
    low_results: Sequence[Mapping[str, Any]],
    high_results: Sequence[Mapping[str, Any]],
    *,
    exploration_quota: int = 1,
    minimum_evaluations: int = 1,
    final_confirmation_budget: int = 1,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Promote candidates while retaining exploration and unresolved states."""
    if not isinstance(low_results, Sequence) or not isinstance(high_results, Sequence) or not low_results:
        raise ProgressiveBudgetError("results_required")
    for name, value in (("exploration_quota", exploration_quota), ("minimum_evaluations", minimum_evaluations), ("final_confirmation_budget", final_confirmation_budget)):
        if type(value) is not int or value < 1:
            raise ProgressiveBudgetError(f"invalid_{name}")
    if not isinstance(lower_is_better, bool):
        raise ProgressiveBudgetError("lower_is_better_must_be_boolean")
    low: dict[str, Mapping[str, Any]] = {}
    for row in low_results:
        if not isinstance(row, Mapping) or not isinstance(row.get("id"), str) or row["id"] in low:
            raise ProgressiveBudgetError("low_result_id_invalid")
        low[row["id"]] = row
    high = {row.get("id"): row for row in high_results if isinstance(row, Mapping) and isinstance(row.get("id"), str)}
    scored = []
    unresolved = []
    for identifier, row in low.items():
        score = row.get("score")
        evaluations = row.get("evaluations", 0)
        valid_score = isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(float(score))
        if type(evaluations) is not int or evaluations < minimum_evaluations:
            unresolved.append(identifier)
        elif valid_score and row.get("status") in {"pass", "eligible", "completed"}:
            scored.append((float(score), identifier))
        else:
            unresolved.append(identifier)
    scored.sort(key=lambda item: (item[0], item[1]), reverse=not lower_is_better)
    ranked_ids = [identifier for _, identifier in scored]
    exploration = ranked_ids[-exploration_quota:] if len(ranked_ids) > exploration_quota else list(ranked_ids)
    promoted = list(dict.fromkeys(ranked_ids[:max(exploration_quota, 1)] + exploration + unresolved))
    final_ready = [identifier for identifier in promoted if identifier in high and high[identifier].get("status") in {"pass", "eligible", "completed"}]
    paired = []
    for identifier in set(low) & set(high):
        low_score, high_score = low[identifier].get("score"), high[identifier].get("score")
        if all(isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) for value in (low_score, high_score)):
            paired.append((identifier, float(low_score), float(high_score)))
    rank_correlation = None
    if len(paired) >= 2:
        low_order = {item[0]: rank for rank, item in enumerate(sorted(paired, key=lambda x: (x[1], x[0])))}
        high_order = {item[0]: rank for rank, item in enumerate(sorted(paired, key=lambda x: (x[2], x[0])))}
        diffs = [low_order[item[0]] - high_order[item[0]] for item in paired]
        n = len(paired)
        rank_correlation = 1.0 - 6.0 * sum(diff * diff for diff in diffs) / (n * (n * n - 1))
    return {
        "schema_version": "mathmodel.progressive-budget/v1",
        "status": "assessed",
        "promoted_ids": promoted,
        "exploration_ids": exploration,
        "unresolved_ids": unresolved,
        "final_ready_ids": final_ready,
        "low_high_overlap": len(set(low) & set(high)),
        "rank_correlation": rank_correlation,
        "rank_correlation_status": "assessed" if rank_correlation is not None else "not_assessed",
        "ranked_ids": ranked_ids,
        "final_confirmation_budget": final_confirmation_budget,
        "policy": "low_fidelity_only_guides_search; exploration_and_unresolved_candidates_are_retained; final_confirmation_required",
    }


__all__ = ["ProgressiveBudgetError", "plan_progressive_promotion"]
