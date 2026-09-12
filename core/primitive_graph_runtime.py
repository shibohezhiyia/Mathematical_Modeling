"""Bounded execution for closed arithmetic primitive graphs.

The runtime deliberately accepts a graph, not source code.  It validates the
graph first and only evaluates a small arithmetic vocabulary with NumPy.  Nodes
which still describe a derivative, event, optimization, latent state or
unknown mechanism are rejected as incomplete rather than silently guessed.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .primitive_graph import validate_primitive_graph
from .math_primitives import PrimitiveValidationError


class PrimitiveGraphRuntimeError(ValueError):
    pass


_EXECUTABLE = {"variable", "constant", "observation", "add", "subtract", "multiply",
               "divide", "power", "negate", "absolute", "exp", "log", "sin", "cos"}
_MAX_ELEMENTS = 200_000


def _numeric(value: Any, node_id: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise PrimitiveGraphRuntimeError(f"non_numeric_node_value:{node_id}") from exc
    if array.ndim > 2 or array.size < 1 or array.size > _MAX_ELEMENTS or not np.isfinite(array).all():
        raise PrimitiveGraphRuntimeError(f"node_value_limit:{node_id}")
    return array


def execute_primitive_graph(
    nodes: Sequence[Mapping[str, Any]],
    bindings: Mapping[str, Any],
    *,
    output_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Evaluate a completed arithmetic graph with explicit variable bindings."""
    try:
        checked = validate_primitive_graph(nodes, output_ids=output_ids)
    except PrimitiveValidationError as exc:
        raise PrimitiveGraphRuntimeError(str(exc)) from exc
    by_id = {str(node["id"]): node for node in nodes}
    requested = list(output_ids) if output_ids else list(checked["output_ids"])
    if any(by_id[node_id]["op"] not in _EXECUTABLE for node_id in by_id):
        raise PrimitiveGraphRuntimeError("graph_contains_unresolved_or_non_arithmetic_node")
    if not isinstance(bindings, Mapping):
        raise PrimitiveGraphRuntimeError("bindings_must_be_mapping")
    values: dict[str, np.ndarray] = {}
    pending = set(by_id)
    evaluation_order: list[str] = []
    while pending:
        ready = [node_id for node_id in pending if all(ref in evaluation_order for ref in by_id[node_id]["inputs"])]
        if not ready:
            raise PrimitiveGraphRuntimeError("graph_dependency_order_invalid")
        evaluation_order.extend(sorted(ready))
        pending.difference_update(ready)
    for node_id in evaluation_order:
        node = by_id[node_id]
        op = node["op"]
        inputs = [values.get(ref) for ref in node["inputs"]]
        if op == "variable":
            if node_id not in bindings:
                raise PrimitiveGraphRuntimeError(f"missing_binding:{node_id}")
            values[node_id] = _numeric(bindings[node_id], node_id)
        elif op == "constant":
            if "value" not in node["attributes"]:
                raise PrimitiveGraphRuntimeError(f"constant_value_missing:{node_id}")
            values[node_id] = _numeric(node["attributes"]["value"], node_id)
        elif op == "observation":
            if not inputs or inputs[0] is None:
                raise PrimitiveGraphRuntimeError(f"input_not_ready:{node_id}")
            values[node_id] = inputs[0]
        else:
            if any(value is None for value in inputs):
                raise PrimitiveGraphRuntimeError(f"input_not_ready:{node_id}")
            try:
                if op == "add": value = inputs[0] + inputs[1]
                elif op == "subtract": value = inputs[0] - inputs[1]
                elif op == "multiply": value = inputs[0] * inputs[1]
                elif op == "divide": value = inputs[0] / inputs[1]
                elif op == "power": value = np.power(inputs[0], inputs[1])
                elif op == "negate": value = -inputs[0]
                elif op == "absolute": value = np.abs(inputs[0])
                elif op == "exp": value = np.exp(inputs[0])
                elif op == "log": value = np.log(inputs[0])
                elif op == "sin": value = np.sin(inputs[0])
                elif op == "cos": value = np.cos(inputs[0])
                else: raise PrimitiveGraphRuntimeError(f"unsupported_graph_operator:{op}")
            except (FloatingPointError, ValueError, ZeroDivisionError) as exc:
                raise PrimitiveGraphRuntimeError(f"arithmetic_failure:{node_id}") from exc
            values[node_id] = _numeric(value, node_id)
    outputs = {node_id: values[node_id].tolist() for node_id in requested}
    return {
        "schema_version": "mathmodel.primitive-graph-runtime/v1",
        "status": "executed",
        "outputs": outputs,
        "node_count": len(by_id),
        "checks": {"finite": True, "bounded": True, "source_execution": False},
        "policy": "typed_arithmetic_graph_only; unresolved_mechanisms_rejected",
    }


__all__ = ["PrimitiveGraphRuntimeError", "execute_primitive_graph"]
