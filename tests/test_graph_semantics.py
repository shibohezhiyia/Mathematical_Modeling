import pytest

from core.graph_semantics import GraphSemanticsError, validate_graph_semantics


def test_causal_cycles_are_rejected_but_dynamic_coupling_is_explicit():
    edges = [{"source": "x", "target": "y"}, {"source": "y", "target": "x"}]
    with pytest.raises(GraphSemanticsError, match="cycles"):
        validate_graph_semantics(nodes=["x", "y"], edges=edges, graph_kind="causal")
    result = validate_graph_semantics(nodes=["x", "y"], edges=edges, graph_kind="mathematical_coupling")
    assert result["cyclic"] is True


def test_computational_dag_is_acyclic():
    result = validate_graph_semantics(nodes=["x", "y"], edges=[{"source": "x", "target": "y"}], graph_kind="computational")
    assert result["cyclic"] is False


def test_graph_semantics_rejects_stringified_booleans_duplicate_edges_and_overflow():
    with pytest.raises(GraphSemanticsError, match="edge_feedback_must_be_bool"):
        validate_graph_semantics(nodes=["x", "y"], edges=[{"source": "x", "target": "y", "feedback": "false"}], graph_kind="computational")
    with pytest.raises(GraphSemanticsError, match="duplicate_edge"):
        validate_graph_semantics(nodes=["x", "y"], edges=[{"source": "x", "target": "y"}, {"source": "x", "target": "y"}], graph_kind="computational")
    with pytest.raises(GraphSemanticsError, match="node_budget_exceeded"):
        validate_graph_semantics(nodes=["x", "y"], edges=[], graph_kind="computational", max_nodes=1)
