import pytest

from core.math_primitives import PrimitiveValidationError
from core.primitive_graph import validate_primitive_graph


def _node(node_id, op, inputs=(), kind="quantity", dimensions=None, attributes=None):
    return {
        "id": node_id,
        "op": op,
        "inputs": list(inputs),
        "kind": kind,
        "dimensions": dimensions,
        "attributes": dict(attributes or {}),
    }


def test_graph_checks_references_obligations_and_outputs():
    nodes = [
        _node("x", "variable", dimensions={"L": 1}),
        _node("t", "coordinate", kind="coordinate", dimensions={"T": 1}),
        _node("dx", "derivative", ("x", "t"), dimensions={"L": 1, "T": -1},
              attributes={"initial_condition": "x0"}),
        _node("eq", "equation", ("dx", "dx"), kind="boolean", dimensions=None),
    ]
    result = validate_primitive_graph(nodes, output_ids=("eq",))
    assert result["node_count"] == 4
    assert result["edge_count"] == 4
    assert result["max_dependency_depth"] == 3
    assert result["estimated_cost_units"] > 0
    assert validate_primitive_graph(list(reversed(nodes)), output_ids=("eq",))["max_dependency_depth"] == 3
    assert result["output_ids"] == ["eq"]
    assert result["status"] == "type_checked_not_executed"


def test_graph_rejects_missing_reference_cycle_and_dimension_mismatch():
    with pytest.raises(PrimitiveValidationError, match="unknown_primitive_reference"):
        validate_primitive_graph([_node("x", "equation", ("missing", "missing"), kind="boolean")])
    cycle = [
        _node("a", "observation", ("b",), attributes={"identity": True}),
        _node("b", "observation", ("a",), attributes={"identity": True}),
    ]
    with pytest.raises(PrimitiveValidationError, match="primitive_graph_cycle"):
        validate_primitive_graph(cycle)
    bad_equation = [
        _node("x", "variable", dimensions={"L": 1}),
            _node("t", "variable", dimensions={"T": 1}),
        _node("eq", "equation", ("x", "t"), kind="boolean"),
    ]
    with pytest.raises(PrimitiveValidationError, match="equation_dimension_mismatch"):
        validate_primitive_graph(bad_equation)


def test_graph_rejects_missing_domain_obligation_and_unknown_output():
    integral = [_node("x", "variable"), _node("t", "coordinate", kind="coordinate"),
                _node("i", "integral", ("x", "t"), attributes={})]
    with pytest.raises(PrimitiveValidationError, match="missing_primitive_obligation"):
        validate_primitive_graph(integral)
    with pytest.raises(PrimitiveValidationError, match="invalid_primitive_outputs"):
        validate_primitive_graph([_node("x", "variable")], output_ids=("missing",))


def test_graph_checks_optional_shapes_without_requiring_shape_metadata():
    good = [
        _node("x", "variable", dimensions={"L": 1}, attributes={"shape": [3]}),
        _node("y", "variable", dimensions={"L": 1}, attributes={"shape": [3]}),
        _node("eq", "equation", ("x", "y"), kind="boolean"),
    ]
    assert validate_primitive_graph(good)["node_count"] == 3
    bad = [
        _node("x", "variable", dimensions={"L": 1}, attributes={"shape": [3]}),
        _node("y", "variable", dimensions={"L": 1}, attributes={"shape": [2]}),
        _node("eq", "equation", ("x", "y"), kind="boolean"),
    ]
    with pytest.raises(PrimitiveValidationError, match="equation_shape_mismatch"):
        validate_primitive_graph(bad)


def test_graph_can_represent_unresolved_mechanism_without_granting_execution():
    nodes = [
        _node("x", "variable", dimensions={"L": 1}),
        _node("g", "unknown_mechanism", ("x",), dimensions={"L": 1},
              attributes={"search_budget": 12, "candidate_language": "bounded_basis"}),
        _node("z", "latent_state", dimensions={"L": 1},
              attributes={"identification_strategy": "delay_embedding"}),
        _node("switch", "regime_switch", ("x",), dimensions={"L": 1},
              attributes={"boundary_semantics": "left_closed"}),
    ]
    result = validate_primitive_graph(nodes)
    assert result["status"] == "type_checked_not_executed"
    assert result["obligations_checked"] >= 3
