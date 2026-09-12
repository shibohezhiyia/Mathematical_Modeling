from core.multifidelity_runner import MultifidelityBudget, run_multifidelity_search


def test_multifidelity_preserves_final_reserve_and_attacks_promoted_candidates():
    calls = {"low": 0, "high": 0, "attack": 0}

    def low(candidate):
        calls["low"] += 1
        return {"status": "completed", "score": float(candidate["score"]), "valid": True}

    def high(candidate):
        calls["high"] += 1
        return {"status": "completed", "score": float(candidate["score"]), "valid": True}

    def attack(candidate):
        calls["attack"] += 1
        return {"status": "tested_not_falsified", "found": False}

    result = run_multifidelity_search(
        [{"id": "a", "score": 1}, {"id": "b", "score": 2}, {"id": "c", "score": 3}],
        low, high, attack,
        budget=MultifidelityBudget(total_evaluations=8, confirmation_candidates=2),
        cache_observation={"reused_nodes": ["x"], "validation_skipped": False},
    )
    assert result["status"] == "assessed"
    assert result["reservation"]["reserved_budget"] == 4
    assert calls == {"low": 3, "high": 2, "attack": 2}
    assert all(item["status"] == "accepted" for item in result["final"])
    assert result["cache_observation"]["validation_skipped"] is False


def test_multifidelity_refuses_to_spend_low_budget_when_reserve_is_missing():
    calls = []
    result = run_multifidelity_search(
        [{"id": "a"}],
        lambda candidate: calls.append("low") or {"score": 1},
        lambda candidate: {"score": 1},
        lambda candidate: {"status": "not_found", "found": False},
        budget=MultifidelityBudget(total_evaluations=1, confirmation_candidates=1,
                                   high_per_candidate=1, counterexample_per_candidate=1),
    )
    assert result["status"] == "insufficient_budget"
    assert calls == []
