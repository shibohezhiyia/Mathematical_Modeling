import pytest

from core.progressive_budget import ProgressiveBudgetError, plan_progressive_promotion


def test_progressive_budget_retains_exploration_and_unresolved_candidates():
    result = plan_progressive_promotion([
        {"id": "a", "score": 1, "evaluations": 2, "status": "pass"},
        {"id": "b", "score": 2, "evaluations": 2, "status": "pass"},
        {"id": "slow", "score": None, "evaluations": 0, "status": "timeout"},
    ], [{"id": "a", "status": "completed"}], exploration_quota=1)
    assert "slow" in result["promoted_ids"]
    assert result["exploration_ids"]
    assert "final_confirmation" in result["policy"]


def test_progressive_budget_rejects_invalid_budget():
    with pytest.raises(ProgressiveBudgetError):
        plan_progressive_promotion([{"id": "a", "score": 1, "evaluations": 1, "status": "pass"}], [], exploration_quota=0)
