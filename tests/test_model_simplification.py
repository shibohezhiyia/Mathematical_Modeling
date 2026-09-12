import pytest

from core.model_simplification import SimplificationConfig, SimplificationError, assess_simplifications


def test_ablation_keeps_only_stable_simplifications():
    model = {"terms": ["x", "y", "z", "w"]}

    def remove(current, identifier):
        return {"terms": [item for item in current["terms"] if item != identifier]}

    def evaluate(current):
        terms = set(current["terms"])
        return {
            "objective_loss": 1.0 if "x" in terms and "y" in terms else 1.3,
            "stress_objective_loss": 1.0 if "z" in terms else 1.8,
            "feasible": "x" in terms,
        }

    result = assess_simplifications(model, ["x", "y", "z", "w"], remove, evaluate)
    by_id = {item["id"]: item for item in result["candidates"]}
    assert by_id["z"]["status"] == "counterexample_found"
    assert by_id["w"]["status"] == "tested_not_falsified"
    assert "w" in result["recommended_ids"]


def test_ablation_preserves_not_assessed_and_budget_boundaries():
    result = assess_simplifications(
        {"terms": ["x"]}, ["x"], lambda model, identifier: model,
        lambda model: {"objective_loss": 1.0, "feasible": True},
    )
    assert result["candidates"][0]["status"] == "tested_not_falsified"
    with pytest.raises(SimplificationError, match="invalid_simplification_budget"):
        SimplificationConfig(max_candidates=0)
