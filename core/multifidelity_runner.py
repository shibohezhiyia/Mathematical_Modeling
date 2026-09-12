"""Budget-preserving low/high-fidelity candidate orchestration.

The runner is intentionally evaluator-agnostic: low-fidelity scores may only
rank candidates; every candidate that reaches a final decision still consumes
reserved high-fidelity and counterexample budget.  Cache metadata can be
attached for measurement, but never bypasses validation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .confirmation_budget import reserve_confirmation_budgets
from .progressive_budget import plan_progressive_promotion


class MultifidelityRunnerError(ValueError):
    pass


@dataclass(frozen=True)
class MultifidelityBudget:
    total_evaluations: int = 64
    confirmation_candidates: int = 2
    high_per_candidate: int = 1
    counterexample_per_candidate: int = 1
    exploration_quota: int = 1

    def validate(self, candidate_count: int) -> "MultifidelityBudget":
        values = (self.total_evaluations, self.confirmation_candidates, self.high_per_candidate,
                  self.counterexample_per_candidate, self.exploration_quota)
        if any(type(value) is not int or value < 1 for value in values):
            raise MultifidelityRunnerError("budget_values_must_be_positive_integers")
        if self.confirmation_candidates > candidate_count:
            raise MultifidelityRunnerError("confirmation_candidates_exceed_candidates")
        return self


def _row(identifier: str, result: Mapping[str, Any] | None, *, stage: str) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        return {"id": identifier, "stage": stage, "status": "error", "score": None, "evaluations": 0,
                "error_code": "evaluator_must_return_mapping"}
    status = result.get("status", "completed")
    score = result.get("score")
    if not isinstance(status, str) or status not in {"completed", "pass", "eligible", "timeout", "rejected", "incomplete", "error"}:
        status, score = "error", None
    if score is not None and (type(score) not in (int, float) or not math.isfinite(float(score))):
        status, score = "error", None
    evaluations = result.get("evaluations", 1 if status in {"completed", "pass", "eligible"} else 0)
    if type(evaluations) is not int or evaluations < 0:
        status, score, evaluations = "error", None, 0
    return {"id": identifier, "stage": stage, "status": status, "score": None if score is None else float(score),
            "evaluations": evaluations, "valid": bool(result.get("valid", False)) if status not in {"timeout", "error"} else False,
            "error_code": result.get("error_code")}


def run_multifidelity_search(
    candidates: Sequence[Mapping[str, Any]],
    low_evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    high_evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    counterexample_evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *, budget: MultifidelityBudget | None = None,
    cache_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run low fidelity, promote with exploration, then verify and attack."""
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates or len(candidates) > 256:
        raise MultifidelityRunnerError("candidates_out_of_bounds")
    if not all(callable(item) for item in (low_evaluate, high_evaluate, counterexample_evaluate)):
        raise MultifidelityRunnerError("evaluators_required")
    identifiers = []
    normalized = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("id"), str) or not candidate["id"].strip():
            raise MultifidelityRunnerError("candidate_id_required")
        identifier = candidate["id"].strip()
        if identifier in identifiers:
            raise MultifidelityRunnerError("duplicate_candidate_id")
        identifiers.append(identifier)
        normalized.append({**dict(candidate), "id": identifier})
    config = (budget or MultifidelityBudget()).validate(len(normalized))
    reserve = reserve_confirmation_budgets(
        len(normalized), confirmation_candidates=config.confirmation_candidates,
        confirmation_per_candidate=config.high_per_candidate,
        counterexample_per_candidate=config.counterexample_per_candidate,
        total_budget=config.total_evaluations,
    )
    if reserve["status"] != "admitted":
        return {"schema_version": "mathmodel.multifidelity-run/v1", "status": "insufficient_budget",
                "reservation": reserve, "low_results": [], "high_results": [], "counterexamples": [],
                "cache_observation": dict(cache_observation or {}),
                "policy": "no_low_fidelity_run_when_final_reserve_is_not_admitted"}
    low_results = []
    for candidate in normalized:
        try:
            low_results.append(_row(candidate["id"], low_evaluate(candidate), stage="low"))
        except Exception as exc:
            low_results.append({"id": candidate["id"], "stage": "low", "status": "error", "score": None,
                                "evaluations": 0, "valid": False, "error_code": type(exc).__name__})
    plan = plan_progressive_promotion(low_results, [], exploration_quota=config.exploration_quota,
                                      minimum_evaluations=1,
                                      final_confirmation_budget=reserve["reserved_budget"])
    promoted_ids = plan["promoted_ids"][: config.confirmation_candidates]
    by_id = {candidate["id"]: candidate for candidate in normalized}
    high_results = []
    counterexamples = []
    for identifier in promoted_ids:
        candidate = by_id[identifier]
        try:
            high = _row(identifier, high_evaluate(candidate), stage="high")
        except Exception as exc:
            high = {"id": identifier, "stage": "high", "status": "error", "score": None,
                    "evaluations": 0, "valid": False, "error_code": type(exc).__name__}
        high_results.append(high)
        if high["status"] in {"completed", "pass", "eligible"}:
            try:
                attack = counterexample_evaluate(candidate)
                if not isinstance(attack, Mapping):
                    raise MultifidelityRunnerError("counterexample_evaluator_must_return_mapping")
                counterexamples.append({"id": identifier, "status": str(attack.get("status", "not_assessed")),
                                        "found": bool(attack.get("found", False)),
                                        "reason": attack.get("reason")})
            except Exception as exc:
                counterexamples.append({"id": identifier, "status": "error", "found": False,
                                        "reason": type(exc).__name__})
    final = []
    attacks = {item["id"]: item for item in counterexamples}
    for high in high_results:
        attack = attacks.get(high["id"], {})
        final.append({"id": high["id"], "status": "accepted" if high["status"] in {"completed", "pass", "eligible"}
                      and attack.get("found") is False and attack.get("status") in {"tested_not_falsified", "not_found"}
                      else "unresolved", "high_status": high["status"], "counterexample": attack})
    return {"schema_version": "mathmodel.multifidelity-run/v1", "status": "assessed",
            "reservation": reserve, "low_results": low_results, "promotion": plan,
            "high_results": high_results, "counterexamples": counterexamples, "final": final,
            "cache_observation": dict(cache_observation or {}),
            "policy": "low_fidelity_guides_only; reserved_confirmation_and_counterexample_budget; cache_never_bypasses_validation"}


__all__ = ["MultifidelityRunnerError", "MultifidelityBudget", "run_multifidelity_search"]
