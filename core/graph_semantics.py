"""Separate computational, mathematical-coupling and causal graph contracts."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping, Sequence


class GraphSemanticsError(ValueError):
    pass


def validate_graph_semantics(*, nodes: Sequence[str], edges: Sequence[Mapping[str, Any]], graph_kind: str,
                             max_nodes: int = 1024, max_edges: int = 4096) -> dict[str, Any]:
    if not isinstance(graph_kind, str) or graph_kind not in {"computational", "mathematical_coupling", "causal"}:
        raise GraphSemanticsError("unsupported_graph_kind")
    if (type(max_nodes) is not int or not 1 <= max_nodes <= 4096 or
            type(max_edges) is not int or not 0 <= max_edges <= 16384):
        raise GraphSemanticsError("invalid_graph_budget")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)) or len(nodes) > max_nodes:
        raise GraphSemanticsError("node_budget_exceeded")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)) or len(edges) > max_edges:
        raise GraphSemanticsError("edge_budget_exceeded")
    labels = tuple(item.strip() for item in nodes if isinstance(item, str))
    if len(labels) != len(nodes) or not labels or len(labels) != len(set(labels)) or any(not item for item in labels):
        raise GraphSemanticsError("nodes_must_be_unique_nonempty")
    adjacency: dict[str, list[str]] = defaultdict(list)
    normalized: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str]] = set()
    for edge in edges:
        if not isinstance(edge, Mapping):
            raise GraphSemanticsError("edge_must_be_mapping")
        source, target = edge.get("source"), edge.get("target")
        if not isinstance(source, str) or not isinstance(target, str):
            raise GraphSemanticsError("edge_endpoint_invalid")
        if source not in labels or target not in labels or source == target:
            raise GraphSemanticsError("edge_endpoint_invalid")
        if (source, target) in seen_edges:
            raise GraphSemanticsError("duplicate_edge")
        seen_edges.add((source, target))
        if graph_kind == "causal" and edge.get("feedback") is True:
            raise GraphSemanticsError("causal_graph_cannot_mark_feedback_edge")
        feedback = edge.get("feedback", False)
        if type(feedback) is not bool:
            raise GraphSemanticsError("edge_feedback_must_be_bool")
        relation = edge.get("relation", "unspecified")
        if not isinstance(relation, str) or not relation.strip() or len(relation) > 256:
            raise GraphSemanticsError("edge_relation_invalid")
        adjacency[source].append(target)
        normalized.append({"source": source, "target": target,
                           "relation": relation.strip(), "feedback": feedback})
    indegree = {label: 0 for label in labels}
    for source in adjacency:
        for target in adjacency[source]:
            indegree[target] += 1
    queue = deque(label for label, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        source = queue.popleft(); visited += 1
        for target in adjacency[source]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    cyclic = visited != len(labels)
    if cyclic and graph_kind != "mathematical_coupling":
        raise GraphSemanticsError("cycles_require_dynamic_feedback_or_implicit_block")
    return {
        "schema_version": "mathmodel.graph-semantics/v1", "kind": graph_kind,
        "nodes": list(labels), "edges": normalized, "cyclic": cyclic,
        "status": "typed_graph_not_causal_proof",
        "policy": "computational_and_causal_dag_cycles_are_rejected; mathematical_coupling_cycles_are_explicit",
    }


__all__ = ["GraphSemanticsError", "validate_graph_semantics"]
