import pytest

from core.equivalence_graph import EquivalenceGraph
from core.rewrite_certificate import RewriteCertificateError, certify_rewrite


def test_registered_commutativity_rewrite_gets_traceable_certificate():
    graph = EquivalenceGraph({"op": "add", "args": [
        {"op": "symbol", "name": "x"}, {"op": "constant", "value": 1},
    ]})
    proposal = graph.rewrite("commute_add")
    certificate = certify_rewrite(graph, proposal["edge"])
    assert certificate["status"] == "proved_by_registered_rule"
    assert certificate["theorem"] == "commutativity_of_addition"


def test_domain_missing_rewrite_stays_unproved():
    graph = EquivalenceGraph({"op": "div", "args": [
        {"op": "symbol", "name": "x"}, {"op": "symbol", "name": "x"},
    ]})
    proposal = graph.rewrite("cancel_self_division")
    certificate = certify_rewrite(graph, proposal["edge"])
    assert certificate["status"] == "not_proved"
    assert "nonzero_denominator_required" in certificate["obligations"]


def test_certificate_rejects_unknown_forms():
    graph = EquivalenceGraph({"op": "symbol", "name": "x"})
    with pytest.raises(RewriteCertificateError, match="unknown"):
        certify_rewrite(graph, {"source_hash": "x", "target_hash": "y", "rule": "commute_add"})
