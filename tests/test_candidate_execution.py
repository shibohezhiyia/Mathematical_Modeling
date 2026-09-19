import pytest

from core.candidate_execution import (
    evaluate_structure_candidate,
    execute_structure_candidate,
    propose_arithmetic_repairs,
    run_arithmetic_candidate_cegis,
)
from core.cegis_controller import CEGISConfig


def _candidate():
    return {
        "id": "linear_candidate",
        "status": "proposed_not_executed",
        "primitive_graph": {
            "output_ids": ["out"],
            "nodes": [
                {"id": "out", "op": "add", "inputs": ["x", "one"], "kind": "quantity",
                 "dimensions": {"Q": 1}, "attributes": {}},
                {"id": "x", "op": "variable", "inputs": [], "kind": "quantity",
                 "dimensions": {"Q": 1}, "attributes": {}},
                {"id": "one", "op": "constant", "inputs": [], "kind": "quantity",
                 "dimensions": {"Q": 1}, "attributes": {"value": 1}},
            ],
        },
    }


def test_completed_candidate_executes_but_keeps_proposal_boundary():
    result = execute_structure_candidate(_candidate(), {"x": [2, 3]})
    assert result["status"] == "executed"
    assert result["proposal_status"] == "proposed_not_executed"
    assert result["result"]["outputs"]["out"] == pytest.approx([3, 4])
    assert result["evidence"]["independent_validation"] == "not_assessed"


def test_incomplete_candidate_is_rejected_without_guessing():
    candidate = _candidate()
    candidate["primitive_graph"]["nodes"][0]["op"] = "unknown_mechanism"
    candidate["primitive_graph"]["nodes"][0]["inputs"] = []
    candidate["primitive_graph"]["nodes"][0]["attributes"] = {
        "search_budget": 2, "candidate_language": "bounded_basis",
    }
    result = execute_structure_candidate(candidate, {"x": [2, 3]})
    assert result["status"] == "rejected"
    assert "unresolved" in result["reason"]


def test_candidate_evaluator_returns_cegis_witnesses_without_mutating_graph():
    candidate = _candidate()
    cases = [
        {"id": "ok", "bindings": {"x": 2}, "expected": 3},
        {"id": "bad", "bindings": {"x": 3}, "expected": 9},
    ]
    result = evaluate_structure_candidate(candidate, cases)
    assert result["status"] == "fail"
    assert result["violations"][0]["witness_id"] == "bad"
    assert result["cost_units"] == 2
    assert candidate["status"] == "proposed_not_executed"


def test_arithmetic_candidate_cegis_repairs_constant_with_bounded_mutations():
    candidate = _candidate()
    candidate["primitive_graph"]["nodes"][2]["attributes"]["value"] = 0.8
    result = run_arithmetic_candidate_cegis(
        candidate,
        [{"id": "case", "bindings": {"x": 2}, "expected": 3.2}],
        config=CEGISConfig(max_rounds=64, max_candidates=128, max_repairs=64),
        tolerance=1e-6,
    )
    assert result["status"] == "accepted_candidates"
    assert result["accepted_candidate_hashes"]
    assert result["repair_count"] > 0


def test_arithmetic_repairs_can_mutate_operator_after_constant_budget():
    candidate = _candidate()
    candidate["primitive_graph"]["nodes"][0]["op"] = "subtract"
    candidate["primitive_graph"]["nodes"][2]["attributes"]["value"] = 1.0
    proposals = propose_arithmetic_repairs(candidate, {"violations": [{"reason": "witness"}], "allow_structural": True})
    assert any(item["primitive_graph"]["nodes"][0]["op"] == "add" for item in proposals)
    assert all(item["primitive_graph"]["nodes"][0]["inputs"] == ["x", "one"] for item in proposals)
