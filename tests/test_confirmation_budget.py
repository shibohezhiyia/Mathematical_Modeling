import pytest

from core.confirmation_budget import ConfirmationBudgetError, reserve_confirmation_budgets


def test_confirmation_budget_reserves_both_final_checks():
    result = reserve_confirmation_budgets(10, confirmation_candidates=2, confirmation_per_candidate=3, counterexample_per_candidate=4, total_budget=20)
    assert result["status"] == "admitted"
    assert result["reserved_budget"] == 14
    assert result["counterexample_budget"] == 8


def test_confirmation_budget_rejects_underfunded_final_checks():
    result = reserve_confirmation_budgets(2, confirmation_candidates=2, confirmation_per_candidate=2, counterexample_per_candidate=2, total_budget=7)
    assert result["status"] == "insufficient_budget"
    with pytest.raises(ConfirmationBudgetError):
        reserve_confirmation_budgets(1, confirmation_candidates=2, confirmation_per_candidate=1, counterexample_per_candidate=1, total_budget=10)
