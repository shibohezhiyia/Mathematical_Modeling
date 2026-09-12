"""Reserve final confirmation and counterexample-search budgets."""

from __future__ import annotations

from typing import Any


class ConfirmationBudgetError(ValueError):
    pass


def reserve_confirmation_budgets(
    candidate_count: int,
    *,
    confirmation_candidates: int,
    confirmation_per_candidate: int,
    counterexample_per_candidate: int,
    total_budget: int,
) -> dict[str, Any]:
    values = (candidate_count, confirmation_candidates, confirmation_per_candidate,
              counterexample_per_candidate, total_budget)
    if any(type(value) is not int or value < 1 for value in values):
        raise ConfirmationBudgetError("budgets_must_be_positive_integers")
    if confirmation_candidates > candidate_count:
        raise ConfirmationBudgetError("confirmation_candidates_exceed_candidates")
    confirmation = confirmation_candidates * confirmation_per_candidate
    counterexample = confirmation_candidates * counterexample_per_candidate
    reserved = confirmation + counterexample
    return {
        "schema_version": "mathmodel.confirmation-budget/v1",
        "status": "admitted" if reserved <= total_budget else "insufficient_budget",
        "candidate_count": candidate_count,
        "confirmation_candidates": confirmation_candidates,
        "confirmation_budget": confirmation,
        "counterexample_budget": counterexample,
        "reserved_budget": reserved,
        "remaining_budget": max(0, total_budget - reserved),
        "total_budget": total_budget,
        "policy": "final_acceptance_requires_confirmation_and_counterexample_budget; low_fidelity_cannot_consume_reserve",
    }


__all__ = ["ConfirmationBudgetError", "reserve_confirmation_budgets"]
