"""Small exact certificates for the guarded equivalence rewrite graph."""

from __future__ import annotations

from typing import Any, Mapping

from .equivalence_graph import EquivalenceGraph, RewriteError


class RewriteCertificateError(ValueError):
    pass


def certify_rewrite(graph: EquivalenceGraph, edge: Mapping[str, Any]) -> dict[str, Any]:
    """Certify only syntactic rewrite rules with explicit conditions.

    Unsupported rules remain ``not_proved``; a finite numerical test is never
    upgraded to an algebraic proof. The returned digest binds the exact source
    and target forms already stored in the graph.
    """
    if not isinstance(graph, EquivalenceGraph) or not isinstance(edge, Mapping):
        raise RewriteCertificateError("graph_and_edge_required")
    source_hash, target_hash = edge.get("source_hash"), edge.get("target_hash")
    if source_hash not in graph.forms or target_hash not in graph.forms:
        raise RewriteCertificateError("unknown_rewrite_form")
    rule = edge.get("rule")
    conditions = tuple(edge.get("conditions", ()))
    executable = edge.get("executable") is True
    source, target = graph.forms[source_hash], graph.forms[target_hash]
    proved = False
    theorem = None
    obligations: list[str] = []
    if rule in {"commute_add", "commute_mul"}:
        proved = (source["op"] == ("add" if rule == "commute_add" else "mul") and
                  target["op"] == source["op"] and target["args"] == list(reversed(source["args"])))
        theorem = "commutativity_of_addition" if rule == "commute_add" else "commutativity_of_multiplication"
    elif rule == "explicit_to_implicit":
        proved = source["op"] == "explicit_equation" and target["op"] == "implicit_equation" and source["args"] == target["args"]
        theorem = "equation_zero_set_representation"
    elif rule == "cancel_self_division":
        proved = (executable and source["op"] == "div" and source["args"][0] == source["args"][1]
                  and target == {"op": "constant", "args": [], "value": 1.0})
        theorem = "self_division_on_nonzero_domain"
        if not executable:
            obligations.append("nonzero_denominator_required")
    elif rule == "minimize_to_maximize_negated":
        proved = executable and source["op"] == "minimize" and target["op"] == "maximize"
        theorem = "negation_reverses_order"
    else:
        obligations.append("formal_rule_not_registered")
    if not executable:
        proved = False
    return {"schema_version": "mathmodel.rewrite-certificate/v1",
            "status": "proved_by_registered_rule" if proved else "not_proved",
            "source_hash": source_hash, "target_hash": target_hash, "rule": rule,
            "conditions": list(conditions), "theorem": theorem,
            "obligations": sorted(set(obligations)),
            "policy": "registered_symbolic_certificate_only;finite_tests_are_not_algebraic_proof"}


__all__ = ["RewriteCertificateError", "certify_rewrite"]
