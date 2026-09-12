from core.interactive_surface import build_constraint_state, build_pareto_front, build_sensitivity_surface


def test_constraint_state_and_surface_are_explicitly_bounded():
    result = build_constraint_state([
        {"id": "capacity", "value": 8, "bound": 10, "relation": "<="},
        {"id": "budget", "value": 12, "bound": 10, "relation": "<=", "tolerance": 0.5},
    ])
    assert result["all_pass"] is False
    assert result["constraints"][1]["violation"] == 1.5
    surface = build_sensitivity_surface(
        [{"alpha": 0.0, "score": 1.0}, {"alpha": 1.0, "score": 2.0}],
        axes=["alpha"], metric="score"
    )
    assert surface["points"][1]["metric"] == 2.0


def test_pareto_front_keeps_non_dominated_points_without_weighting():
    result = build_pareto_front([
        {"id": "a", "loss": 1, "cost": 3},
        {"id": "b", "loss": 2, "cost": 2},
        {"id": "c", "loss": 3, "cost": 4},
    ], objectives={"loss": "min", "cost": "min"})
    assert result["front_ids"] == ["a", "b"]
