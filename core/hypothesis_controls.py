"""Typed assumption-control schema and affected-node recomputation planner."""

from __future__ import annotations

import math
import hashlib
import json
from copy import deepcopy
from typing import Any, Mapping, Sequence


class HypothesisControlError(ValueError):
    pass


# These are the only numeric fields that the dynamic compiler treats as
# model parameters.  Deliberately excluding ``times``, observations, bounds,
# and row data prevents a slider from silently changing the validation set or
# feasible region.  The list is shared by every typed family, so the UI does
# not need family-specific code.
_DYNAMIC_PARAMETER_FIELDS = frozenset({
    "coefficients", "objective_coefficients", "linear_coefficients",
    "quadratic_matrix",
    # Bounded GNN training knobs are safe what-if parameters; data rows,
    # targets and graph edges remain outside the slider surface.
    "epochs", "hidden_dim", "restarts", "message_layers", "dynamic_windows",
    "max_variables", "max_rows", "edge_threshold", "validation_fraction",
})

_INTEGER_PARAMETER_FIELDS = frozenset({
    "epochs", "hidden_dim", "restarts", "message_layers", "dynamic_windows",
    "max_variables", "max_rows",
})
_PARAMETER_LIMITS: dict[str, tuple[float, float]] = {
    "epochs": (10.0, 500.0), "hidden_dim": (4.0, 64.0),
    "restarts": (1.0, 3.0), "message_layers": (1.0, 3.0),
    "dynamic_windows": (0.0, 5.0), "max_variables": (2.0, 32.0),
    "max_rows": (60.0, 10_000.0), "edge_threshold": (0.0, 1.0),
    "validation_fraction": (0.1, 0.4),
}


def _numeric_leaves(value: Any, path: list[str], *, depth: int = 0):
    """Yield ``(path, value)`` for bounded numeric leaves in a coefficient field."""
    if depth > 8:
        return
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            yield ".".join(path), number
        return
    if isinstance(value, list):
        if len(value) > 128:
            return
        for index, item in enumerate(value):
            yield from _numeric_leaves(item, path + [str(index)], depth=depth + 1)


def build_dynamic_hypothesis_controls(
    dynamic_contract: Mapping[str, Any], *, max_controls: int = 24,
) -> dict[str, Any]:
    """Create safe slider bindings from a typed dynamic competition contract.

    This is intentionally a *parameter* discovery helper, not a generic JSON
    editor.  Only coefficient-like fields are exposed and every generated
    control has a deterministic JSON path, finite bounds, and a corresponding
    binding.  The preview endpoint still recompiles and validates the complete
    contract after every change.
    """
    if not isinstance(dynamic_contract, Mapping):
        raise HypothesisControlError("dynamic_contract_required")
    if type(max_controls) is not int or not 1 <= max_controls <= 64:
        raise HypothesisControlError("dynamic_control_budget_invalid")
    families = dynamic_contract.get("families")
    if not isinstance(families, list) or len(families) > 8:
        raise HypothesisControlError("dynamic_contract_families_invalid")

    controls: list[dict[str, Any]] = []
    bindings: dict[str, float] = {}
    paths: dict[str, str] = {}
    for family_index, family in enumerate(families):
        if not isinstance(family, Mapping):
            continue
        candidates = family.get("candidates")
        if not isinstance(candidates, list):
            continue
        for candidate_index, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                continue
            for field in _DYNAMIC_PARAMETER_FIELDS:
                raw = candidate.get(field)
                if not isinstance(raw, (list, int, float)) or isinstance(raw, bool):
                    continue
                leaves = _numeric_leaves(raw, ["families", str(family_index), "candidates", str(candidate_index), field])
                for suffix, number in leaves:
                    if len(controls) >= max_controls:
                        break
                    parameter_id = "dynamic_" + suffix.replace(".", "_")
                    if parameter_id in bindings:
                        continue
                    # A symmetric, finite local neighbourhood is useful for
                    # what-if analysis while avoiding unbounded search.
                    span = max(1.0, abs(number) * 0.5)
                    minimum, maximum = number - span, number + span
                    if field in _PARAMETER_LIMITS:
                        lower, upper = _PARAMETER_LIMITS[field]
                        # GNN validation uses strict inequalities for these
                        # two probabilities; keep slider endpoints executable.
                        if field in {"edge_threshold", "validation_fraction"}:
                            lower += 1e-6
                            upper -= 1e-6
                        minimum, maximum = max(minimum, lower), min(maximum, upper)
                    integer_value = field in _INTEGER_PARAMETER_FIELDS
                    if integer_value:
                        minimum, maximum, number = math.ceil(minimum), math.floor(maximum), int(round(number))
                        if minimum >= maximum or not minimum <= number <= maximum:
                            continue
                        step = 1.0
                    else:
                        step = max((maximum - minimum) / 20.0, 1e-6)
                    controls.append({
                        "id": parameter_id, "parameter_id": parameter_id,
                        "label": f"{family.get('family', 'model')} · {field} [{suffix.rsplit('.', 1)[-1]}]",
                        "min": minimum, "max": maximum, "default": number,
                        "step": step, "affected_nodes": [parameter_id],
                        "value_type": "int" if integer_value else "float",
                        "unit": "",
                    })
                    bindings[parameter_id] = number
                    paths[parameter_id] = suffix
                if len(controls) >= max_controls:
                    break
            if len(controls) >= max_controls:
                break
        if len(controls) >= max_controls:
            break
    validated = build_hypothesis_controls(controls, node_ids=list(bindings)) if controls else {
        "schema_version": "mathmodel.hypothesis-controls/v1", "status": "validated",
        "controls": [], "node_count": 0,
        "policy": "controls_are_finite_what_if_inputs; values_do_not_authorize_execution_or_causal_claims",
    }
    return {
        "controls": validated["controls"], "bindings": bindings,
        "dynamic_paths": paths, "node_ids": list(bindings),
        "truncated": bool(len(controls) >= max_controls),
        "policy": "auto_generated_from_typed_coefficient_fields; bounded_preview_only",
    }


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
        value_type = control.get("value_type", "float")
        if value_type not in {"float", "int"}:
            raise HypothesisControlError("control_value_type_invalid")
        if value_type == "int" and any(float(value) != int(float(value)) for value in (minimum, maximum, default, step)):
            raise HypothesisControlError("integer_control_bounds_invalid")
        version = control.get("version", 1)
        if type(version) is not int or version < 1:
            raise HypothesisControlError("control_version_invalid")
        result.append({"id": identifier, "parameter_id": parameter_id.strip(), "version": version,
                       "label": str(control.get("label", identifier))[:160],
                       "min": float(minimum), "max": float(maximum), "default": float(default),
                       "step": float(step), "affected_nodes": list(dict.fromkeys(str(item) for item in affected)),
                       "unit": str(control.get("unit", ""))[:64], "value_type": value_type})
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
        value = int(round(float(raw))) if control.get("value_type", "float") == "int" else float(raw)
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


__all__ = [
    "HypothesisControlError", "build_hypothesis_controls",
    "build_dynamic_hypothesis_controls", "apply_hypothesis_controls",
    "affected_nodes_for_control",
]
