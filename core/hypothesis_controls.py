"""Typed assumption-control schema and affected-node recomputation planner."""

from __future__ import annotations

import math
import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


class HypothesisControlError(ValueError):
    pass


def build_hypothesis_controls(
    controls: Sequence[Mapping[str, Any]],
    *,
    node_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(controls, Sequence) or isinstance(controls, (str, bytes)) or len(controls) > 128:
        raise HypothesisControlError("controls_out_of_bounds")
    nodes = {str(item).strip() for item in node_ids if isinstance(item, str) and item.strip()}
    if not nodes:
        raise HypothesisControlError("node_ids_required")
    result = []
    seen = set()
    for control in controls:
        if not isinstance(control, Mapping) or not isinstance(control.get("id"), str) or not control["id"].strip():
            raise HypothesisControlError("control_id_required")
        identifier = control["id"].strip()
        if identifier in seen:
            raise HypothesisControlError("duplicate_control_id")
        seen.add(identifier)
        minimum, maximum, default, step = (control.get(key) for key in ("min", "max", "default", "step"))
        values = (minimum, maximum, default, step)
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) for value in values):
            raise HypothesisControlError("control_bounds_must_be_finite_numeric")
        if not float(minimum) < float(maximum) or not 0 < float(step) <= float(maximum - minimum) or not float(minimum) <= float(default) <= float(maximum):
            raise HypothesisControlError("control_bounds_invalid")
        affected = control.get("affected_nodes", [])
        if not isinstance(affected, Sequence) or isinstance(affected, (str, bytes)) or not affected or any(str(item) not in nodes for item in affected):
            raise HypothesisControlError("affected_nodes_invalid")
        parameter_id = control.get("parameter_id", identifier)
        if not isinstance(parameter_id, str) or not parameter_id.strip() or len(parameter_id) > 120:
            raise HypothesisControlError("parameter_id_invalid")
        version = control.get("version", 1)
        if type(version) is not int or version < 1:
            raise HypothesisControlError("control_version_invalid")
        result.append({"id": identifier, "parameter_id": parameter_id.strip(), "version": version,
                       "label": str(control.get("label", identifier))[:160],
                       "min": float(minimum), "max": float(maximum), "default": float(default),
                       "step": float(step), "affected_nodes": list(dict.fromkeys(str(item) for item in affected)),
                       "unit": str(control.get("unit", ""))[:64]})
    return {"schema_version": "mathmodel.hypothesis-controls/v1", "status": "validated",
            "controls": result, "node_count": len(nodes),
            "policy": "controls_are_finite_what_if_inputs; values_do_not_authorize_execution_or_causal_claims"}


def apply_hypothesis_controls(
    validated: Mapping[str, Any], values: Mapping[str, Any], bindings: Mapping[str, Any],
    *, graph: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate slider values and return a versioned binding preview."""
    if not isinstance(validated, Mapping) or validated.get("schema_version") != "mathmodel.hypothesis-controls/v1":
        raise HypothesisControlError("validated_controls_required")
    if not isinstance(values, Mapping) or not isinstance(bindings, Mapping):
        raise HypothesisControlError("preview_values_and_bindings_required")
    updated = dict(bindings)
    changes = []
    controls = validated.get("controls", [])
    for control in controls:
        identifier = str(control.get("id"))
        parameter_id = str(control.get("parameter_id"))
        raw = values.get(identifier, control.get("default"))
        if isinstance(raw, bool) or not isinstance(raw, (int, float)) or not math.isfinite(float(raw)):
            raise HypothesisControlError("control_value_must_be_finite")
        value = float(raw)
        if value < float(control["min"]) or value > float(control["max"]):
            raise HypothesisControlError("control_value_out_of_range")
        if parameter_id not in updated:
            raise HypothesisControlError("preview_parameter_binding_missing")
        previous = updated[parameter_id]
        updated[parameter_id] = value
        changes.append({"control_id": identifier, "parameter_id": parameter_id,
                        "version": int(control["version"]), "previous": previous, "value": value})
    plan = None
    if graph is not None:
        plan = affected_nodes_for_control(graph, affected_nodes=[str(item.get("parameter_id")) for item in controls])
    digest = hashlib.sha256(json.dumps(updated, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    return {"schema_version": "mathmodel.hypothesis-preview/v1", "status": "validated",
            "bindings": updated, "changes": changes, "binding_digest": digest,
            "recompute_plan": plan,
            "policy": "preview_inputs_are_untrusted; downstream_solver_must_recompute_and_revalidate"}


def affected_nodes_for_control(
    graph: Mapping[str, Any],
    *,
    affected_nodes: Sequence[str],
) -> dict[str, Any]:
    """Compute downstream closure for incremental recomputation."""
    if not isinstance(graph, Mapping) or not isinstance(graph.get("nodes"), Sequence):
        raise HypothesisControlError("graph_nodes_required")
    nodes = graph["nodes"]
    identifiers = {str(node.get("id")) for node in nodes if isinstance(node, Mapping) and node.get("id") is not None}
    seeds = {str(item) for item in affected_nodes}
    if not seeds or not seeds <= identifiers:
        raise HypothesisControlError("affected_node_unknown")
    downstream = {str(node.get("id")): {str(item) for item in node.get("inputs", [])}
                  for node in nodes if isinstance(node, Mapping) and node.get("id") is not None}
    changed = set(seeds)
    updated = True
    while updated:
        updated = False
        for identifier, inputs in downstream.items():
            if identifier not in changed and inputs & changed:
                changed.add(identifier)
                updated = True
    return {"schema_version": "mathmodel.hypothesis-controls/v1", "status": "planned",
            "seed_nodes": sorted(seeds), "recompute_nodes": sorted(changed),
            "skipped_nodes": sorted(identifiers - changed),
            "policy": "downstream_only_recompute_plan; caller_must_validate_graph_and_compare_full_recompute"}


__all__ = ["HypothesisControlError", "build_hypothesis_controls", "apply_hypothesis_controls", "affected_nodes_for_control"]
