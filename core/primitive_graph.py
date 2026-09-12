"""Static validation for compositions of the minimal math primitives.

This is intentionally a pre-execution checker.  It does not evaluate a
formula, solve an equation, or prove that a model describes reality.  Its job
is to reject graph-level mistakes before a backend is selected.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

from .math_primitives import PrimitiveValidationError, validate_primitive_node
from .ir_schema import canonical_json


_OBLIGATION_FIELDS = {
    "initial_or_boundary_condition_required": ("initial_condition", "boundary_condition"),
    "integration_domain_required": ("domain",),
    "support_or_domain_required": ("support", "domain"),
    "event_boundary_semantics_required": ("boundary", "boundary_semantics"),
    "node_and_edge_semantics_required": ("node_semantics", "edge_semantics"),
    "optimization_sense_required": ("sense",),
    "feasible_domain_required": ("feasible_domain",),
    "control_bounds_required": ("bounds",),
    "observation_noise_or_identity_required": ("noise_model", "identity"),
    "mechanism_search_budget_required": ("search_budget", "candidate_language"),
    "latent_state_identification_required": ("identification_strategy", "proxy_or_prior"),
}


def _dimension(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    return {str(key): float(exponent) for key, exponent in value.items() if float(exponent) != 0.0}


def _same_dimension(left: Any, right: Any) -> bool:
    return _dimension(left) == _dimension(right)


def _shape(node: Mapping[str, Any]) -> tuple[int, ...] | None:
    raw = node["attributes"].get("shape")
    if raw is None:
        return None
    if type(raw) is not list or len(raw) > 4 or any(type(size) is not int or not 1 <= size <= 100_000 for size in raw):
        raise PrimitiveValidationError("invalid_primitive_shape", node["id"])
    size = 1
    for item in raw:
        size *= item
        if size > 1_000_000:
            raise PrimitiveValidationError("primitive_shape_size_limit", node["id"])
    return tuple(raw)


def _check_acyclic(node_ids: Sequence[str], edges: Mapping[str, Sequence[str]]) -> None:
    indegree = {node_id: 0 for node_id in node_ids}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for node_id, inputs in edges.items():
        for input_id in inputs:
            outgoing[input_id].append(node_id)
            indegree[node_id] += 1
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for child in outgoing[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(node_ids):
        raise PrimitiveValidationError("primitive_graph_cycle")


def _graph_depth(node_ids: Sequence[str], edges: Mapping[str, Sequence[str]]) -> int | None:
    """Return longest dependency depth for an acyclic graph."""
    depth: dict[str, int] = {}

    def visit(node_id: str) -> int:
        if node_id in depth:
            return depth[node_id]
        depth[node_id] = 1 + max((visit(ref) for ref in edges[node_id]), default=0)
        return depth[node_id]

    return max((visit(node_id) for node_id in node_ids), default=0)


def validate_primitive_graph(
    nodes: Sequence[Mapping[str, Any]],
    *,
    output_ids: Sequence[str] = (),
    allow_cycles: bool = False,
) -> dict[str, Any]:
    """Validate a finite, typed primitive graph and return a redacted summary.

    ``allow_cycles`` is reserved for explicit dynamic/implicit solver blocks;
    ordinary compositions reject cycles because they otherwise cannot be
    topologically compiled.  All attributes are checked for bounded JSON
    shape, but are not interpreted as executable code.
    """
    if not isinstance(nodes, (list, tuple)) or not nodes or len(nodes) > 5000:
        raise PrimitiveValidationError("invalid_primitive_graph_size")
    normalized: dict[str, dict[str, Any]] = {}
    for node in nodes:
        summary = validate_primitive_node(node)
        node_id = summary["id"]
        if node_id in normalized:
            raise PrimitiveValidationError("duplicate_primitive_node", node_id)
        # canonical_json enforces finite JSON values and bounded nesting while
        # keeping the original node out of the returned summary.
        canonical_json(dict(node))
        for obligation in summary["obligations"]:
            fields = _OBLIGATION_FIELDS.get(obligation, ())
            attributes = node["attributes"]
            if fields and not any(field in attributes and attributes[field] is not None for field in fields):
                raise PrimitiveValidationError("missing_primitive_obligation", node_id)
        normalized[node_id] = dict(node)

    edges: dict[str, list[str]] = {}
    for node_id, node in normalized.items():
        refs = list(node["inputs"])
        for ref in refs:
            if ref not in normalized:
                raise PrimitiveValidationError("unknown_primitive_reference", node_id)
        validate_primitive_node(node, input_kinds=tuple(normalized[ref]["kind"] for ref in refs))
        edges[node_id] = refs
        op = node["op"]
        output_shape = _shape(node)
        if op in {"equation", "objective", "constraint"} and output_shape not in (None, ()):
            raise PrimitiveValidationError("scalar_primitive_required", node_id)
        if op in {"derivative", "integral", "observation"} and refs:
            input_shape = _shape(normalized[refs[0]])
            if output_shape is not None and input_shape is not None and output_shape != input_shape:
                raise PrimitiveValidationError("primitive_shape_mismatch", node_id)
        if op == "equation" and not _same_dimension(
            normalized[refs[0]]["dimensions"], normalized[refs[1]]["dimensions"]
        ):
            raise PrimitiveValidationError("equation_dimension_mismatch", node_id)
        if op == "equation":
            left_shape, right_shape = _shape(normalized[refs[0]]), _shape(normalized[refs[1]])
            if left_shape is not None and right_shape is not None and left_shape != right_shape:
                raise PrimitiveValidationError("equation_shape_mismatch", node_id)
        if op == "derivative" and node["dimensions"] is None:
            raise PrimitiveValidationError("derivative_dimensions_required", node_id)
        if op in {"add", "subtract"}:
            if not _same_dimension(normalized[refs[0]]["dimensions"], normalized[refs[1]]["dimensions"]):
                raise PrimitiveValidationError("algebra_dimension_mismatch", node_id)
            if not _same_dimension(node["dimensions"], normalized[refs[0]]["dimensions"]):
                raise PrimitiveValidationError("algebra_output_dimension_mismatch", node_id)
        if op == "multiply":
            left, right = _dimension(normalized[refs[0]]["dimensions"]), _dimension(normalized[refs[1]]["dimensions"])
            expected = dict(left or {})
            for key, exponent in (right or {}).items():
                expected[key] = expected.get(key, 0.0) + exponent
                if expected[key] == 0.0:
                    expected.pop(key)
            if not _same_dimension(node["dimensions"], expected):
                raise PrimitiveValidationError("algebra_output_dimension_mismatch", node_id)
        if op == "divide":
            left, right = _dimension(normalized[refs[0]]["dimensions"]), _dimension(normalized[refs[1]]["dimensions"])
            expected = dict(left or {})
            for key, exponent in (right or {}).items():
                expected[key] = expected.get(key, 0.0) - exponent
                if expected[key] == 0.0:
                    expected.pop(key)
            if not _same_dimension(node["dimensions"], expected):
                raise PrimitiveValidationError("algebra_output_dimension_mismatch", node_id)
        if op == "power":
            if not _same_dimension(normalized[refs[1]]["dimensions"], {}):
                raise PrimitiveValidationError("power_exponent_dimensionful", node_id)
        if op in {"exp", "log", "sin", "cos"}:
            if not _same_dimension(normalized[refs[0]]["dimensions"], {}) or not _same_dimension(node["dimensions"], {}):
                raise PrimitiveValidationError("transcendental_requires_dimensionless", node_id)
        if op in {"negate", "absolute"}:
            if not _same_dimension(node["dimensions"], normalized[refs[0]]["dimensions"]):
                raise PrimitiveValidationError("algebra_output_dimension_mismatch", node_id)
    edge_count = sum(len(refs) for refs in edges.values())
    if edge_count > 20_000:
        raise PrimitiveValidationError("primitive_graph_edge_limit")
    if not allow_cycles:
        _check_acyclic(tuple(normalized), edges)
    requested = list(output_ids)
    if requested:
        if len(set(requested)) != len(requested) or any(item not in normalized for item in requested):
            raise PrimitiveValidationError("invalid_primitive_outputs")
    else:
        referenced = {ref for refs in edges.values() for ref in refs}
        requested = [node_id for node_id in normalized if node_id not in referenced]
    weights = {"derivative": 6, "integral": 6, "distribution": 5, "graph": 4,
               "event": 3, "objective": 2, "constraint": 2}
    estimated_cost = sum(weights.get(node["op"], 1) for node in normalized.values())
    return {
        "node_count": len(normalized),
        "edge_count": edge_count,
        "max_dependency_depth": None if allow_cycles else _graph_depth(tuple(normalized), edges),
        "estimated_cost_units": estimated_cost,
        "output_ids": requested,
        "acyclic": not allow_cycles,
        "obligations_checked": sum(len(validate_primitive_node(node)["obligations"]) for node in normalized.values()),
        "status": "type_checked_not_executed",
    }


def propagate_obligations(nodes: Sequence[Mapping[str, Any]], *, output_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Propagate declared obligations through a validated acyclic graph.

    This is bookkeeping for the next verification stage.  It does not claim
    existence, uniqueness, stability or real-world correctness merely because
    a type and a local obligation are present.
    """
    summary = validate_primitive_graph(nodes, output_ids=output_ids)
    by_id = {str(node["id"]): node for node in nodes}
    inherited: dict[str, set[str]] = {}
    for node_id in [str(item["id"]) for item in nodes]:
        node = by_id[node_id]
        obligations = set(validate_primitive_node(node)["obligations"])
        for ref in node.get("inputs", []):
            obligations.update(inherited[str(ref)])
        inherited[node_id] = obligations
    output_obligations = {node_id: sorted(inherited[node_id]) for node_id in summary["output_ids"]}
    return {**summary, "obligation_trace": output_obligations,
            "status": "type_checked_obligations_propagated_not_proved",
            "policy": "obligation_presence_is_not_property_proof"}


__all__ = ["validate_primitive_graph", "propagate_obligations"]
