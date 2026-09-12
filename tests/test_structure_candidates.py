from core.structure_candidates import build_structure_candidates


def test_structure_candidates_are_bounded_and_not_executable():
    result = build_structure_candidates(
        {
            "task_type": "optimization",
            "task_candidates": [{"task_type": "optimization", "score": 2}],
        },
        has_observations=False,
        max_candidates=2,
    )
    assert result["status"] == "candidate_structures_only"
    assert len(result["candidates"]) == 2
    assert result["diversity_audit"]["status"] == "structurally_distinct"
    assert result["diversity_audit"]["unique_structures"] == 2
    assert all(item["status"] == "proposed_not_executed" for item in result["candidates"])
    assert all(item["numeric_execution"].startswith("blocked") for item in result["candidates"])
    assert all(item["primitive_graph"]["validation"]["status"] == "type_checked_not_executed"
               for item in result["candidates"])
    assert all(item["primitive_graph"]["validation"]["acyclic"] is True
               for item in result["candidates"])
    unknown_nodes = [node for item in result["candidates"] for node in item["primitive_graph"]["nodes"]
                     if node["op"] == "unknown_mechanism"]
    assert unknown_nodes and all(node["attributes"]["allowed_operators"] for node in unknown_nodes)
    assert all(node["attributes"]["search_budget"] == 8 for node in unknown_nodes)
    repeat = build_structure_candidates(
        {"task_type": "optimization", "task_candidates": [{"task_type": "optimization", "score": 2}]},
        has_observations=False,
        max_candidates=2,
    )
    assert result["candidate_digest"] == repeat["candidate_digest"]


def test_structure_candidates_compose_ranked_task_families():
    result = build_structure_candidates(
        {
            "task_type": "prediction_forecast",
            "task_candidates": [
                {"task_type": "prediction_forecast", "score": 3},
                {"task_type": "optimization", "score": 1},
            ],
        },
        has_observations=True,
        max_candidates=3,
    )
    assert "prediction_forecast" in {item["task_type"] for item in result["candidates"]}
    assert len(result["candidates"]) <= 3
    assert all(item["numeric_execution"] == "available_after_contract_binding" for item in result["candidates"])
    assert all(item["primitive_graph"]["nodes"] for item in result["candidates"])
