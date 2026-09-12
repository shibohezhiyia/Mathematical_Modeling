from core.primitive_graph import propagate_obligations


def test_obligations_are_propagated_to_output_without_being_called_proof():
    nodes = [
        {"id": "x", "op": "variable", "kind": "quantity", "dimensions": {"L": 1},
         "inputs": [], "attributes": {"domain": [0, 1]}},
        {"id": "t", "op": "coordinate", "kind": "coordinate", "dimensions": {"T": 1},
         "inputs": [], "attributes": {"domain": [0, 1]}},
        {"id": "integral", "op": "integral", "kind": "quantity", "dimensions": {"L": 1},
         "inputs": ["x", "t"], "attributes": {"domain": [0, 1]}},
    ]
    result = propagate_obligations(nodes)
    assert result["status"].endswith("not_proved")
    assert "integration_domain_required" in result["obligation_trace"]["integral"]
