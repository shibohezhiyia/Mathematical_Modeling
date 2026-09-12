import pytest

from core.equivalence_graph import EquivalenceGraph, RewriteError


def test_rewrite_keeps_trace_and_domain_condition():
    graph = EquivalenceGraph({"op": "div", "args": [{"op": "symbol", "name": "x"}, {"op": "symbol", "name": "x"}]})
    proposal = graph.rewrite("cancel_self_division")
    assert proposal["edge"]["executable"] is False
    assert "nonzero_denominator_required" in proposal["edge"]["conditions"]
    approved = graph.rewrite("cancel_self_division", condition="nonzero_denominator")
    assert approved["edge"]["executable"] is True


def test_derivative_integral_rewrite_does_not_drop_constant():
    graph = EquivalenceGraph({"op": "derivative", "args": [{"op": "symbol", "name": "y"}, {"op": "symbol", "name": "t"}]})
    result = graph.rewrite("derivative_to_integral")
    assert result["edge"]["executable"] is False
    assert "integration_constant_required" in result["edge"]["conditions"]


def test_invalid_rule_is_rejected():
    with pytest.raises(RewriteError):
        EquivalenceGraph({"op": "symbol", "name": "x"}).rewrite("unknown")


def test_rewrite_chain_requires_explicit_source_and_only_verified_edges_merge_forms():
    graph = EquivalenceGraph({
        "op": "explicit_equation",
        "args": [{"op": "symbol", "name": "y"}, {"op": "symbol", "name": "x"}],
    })
    implicit = graph.rewrite("explicit_to_implicit")
    explicit = graph.rewrite(
        "implicit_to_explicit", source_hash=implicit["edge"]["target_hash"],
        condition="unique_solution_for_target",
    )
    assert explicit["edge"]["executable"] is True
    summary = graph.summary()
    assert any(len(group) == 2 for group in summary["verified_equivalence_classes"])
    with pytest.raises(RewriteError, match="unknown_source_form"):
        graph.rewrite("commute_add", source_hash="missing")


def test_unproved_rewrite_does_not_collapse_equivalence_classes():
    graph = EquivalenceGraph({
        "op": "div",
        "args": [{"op": "symbol", "name": "x"}, {"op": "symbol", "name": "x"}],
    })
    proposal = graph.rewrite("cancel_self_division")
    classes = graph.summary()["verified_equivalence_classes"]
    assert proposal["edge"]["executable"] is False
    assert all(len(group) == 1 for group in classes)


def test_expression_metadata_is_finite_canonical_and_depth_bounded():
    graph = EquivalenceGraph({"op": "constant", "value": "1"})
    assert graph.summary()["forms"] == 1
    with pytest.raises(RewriteError, match="metadata"):
        EquivalenceGraph({"op": "symbol", "name": "x", "sense": float("nan")})
    nested = {"op": "symbol", "name": "x"}
    for _ in range(34):
        nested = {"op": "add", "args": [nested, {"op": "constant", "value": 1}]}
    with pytest.raises(RewriteError, match="depth"):
        EquivalenceGraph(nested)
