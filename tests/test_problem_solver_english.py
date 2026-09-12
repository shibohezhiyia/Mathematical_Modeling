from core.problem_solver import analyze_problem


def test_english_optimization_is_not_reduced_to_generic_ranking():
    result = analyze_problem("Maximize total profit subject to budget constraints.")
    assert result["task_type"] == "optimization"


def test_english_forecast_and_network_terms_are_detected():
    forecast = analyze_problem("Forecast medal counts for the next Olympics.")
    assert "prediction_forecast" in {forecast["task_type"], *(item["task_type"] for item in forecast["task_candidates"])}

    network = analyze_problem("Find the shortest path in a weighted network graph.")
    assert "graph_network" in {network["task_type"], *(item["task_type"] for item in network["task_candidates"])}


def test_english_quantities_bounds_and_objectives_are_preserved():
    result = analyze_problem(
        "Maximize profit. The speed is 12 m/s and the budget is 100. "
        "Subject to at most 5 vehicles and at least 2 routes."
    )
    assert any("12 m/s" in item for item in result["variables"])
    assert any("英文" in item for item in result["constraints"])
    assert any(item.startswith("最大化:") for item in result["objectives"])


def test_question_and_part_markers_split_english_subproblems():
    result = analyze_problem("Question 1: Build a model.\nQuestion 2: Forecast demand.")
    assert len(result["subproblems"]) == 2
    assert result["task_graph"][1]["depends_on"] == []
