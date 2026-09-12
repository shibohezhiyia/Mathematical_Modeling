import numpy as np
import pytest

from core.primitive_graph_runtime import PrimitiveGraphRuntimeError, execute_primitive_graph


def _graph():
    # Deliberately place the output before its dependencies to verify that
    # execution uses the validated dependency graph rather than list order.
    return [
        {"id": "out", "op": "add", "inputs": ["scaled", "one"], "kind": "quantity",
         "dimensions": {"L": 1}, "attributes": {}},
        {"id": "scaled", "op": "multiply", "inputs": ["x", "two"], "kind": "quantity",
         "dimensions": {"L": 1}, "attributes": {}},
        {"id": "x", "op": "variable", "inputs": [], "kind": "quantity",
         "dimensions": {"L": 1}, "attributes": {}},
        {"id": "two", "op": "constant", "inputs": [], "kind": "quantity",
         "dimensions": {}, "attributes": {"value": 2}},
        {"id": "one", "op": "constant", "inputs": [], "kind": "quantity",
         "dimensions": {"L": 1}, "attributes": {"value": 1}},
    ]


def test_typed_graph_runtime_executes_without_source_evaluation():
    result = execute_primitive_graph(_graph(), {"x": np.array([1.0, 3.0])}, output_ids=["out"])
    assert result["status"] == "executed"
    assert result["outputs"]["out"] == pytest.approx([3.0, 7.0])
    assert result["checks"]["source_execution"] is False


def test_typed_graph_runtime_rejects_unresolved_mechanism_and_missing_binding():
    unresolved = [{"id": "m", "op": "unknown_mechanism", "inputs": [], "kind": "quantity",
                   "dimensions": {"Q": 1}, "attributes": {"search_budget": 2, "candidate_language": "bounded_basis"}}]
    with pytest.raises(PrimitiveGraphRuntimeError, match="unresolved"):
        execute_primitive_graph(unresolved, {})
    with pytest.raises(PrimitiveGraphRuntimeError, match="missing_binding"):
        execute_primitive_graph(_graph(), {})


def test_typed_graph_dimension_checks_reject_invalid_addition():
    nodes = _graph()
    nodes[0]["dimensions"] = {"T": 1}
    with pytest.raises(PrimitiveGraphRuntimeError, match="algebra_output_dimension_mismatch"):
        execute_primitive_graph(nodes, {"x": 1.0}, output_ids=["out"])
