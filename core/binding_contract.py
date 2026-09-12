"""Versioned semantic binding contract for dynamic mathematical compilation."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mathmodel.binding-contract/v1"


class BindingContractError(ValueError):
    pass


def _bounded_json(value: Any) -> Any:
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BindingContractError("binding_contract_not_finite_json") from exc
    if len(encoded) > 2_000_000:
        raise BindingContractError("binding_contract_size_limit")
    return json.loads(encoded)


def _dimension(value: Any) -> dict[str, float] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or len(value) > 16:
        raise BindingContractError("binding_dimension_invalid")
    result = {}
    for key, exponent in value.items():
        if not isinstance(key, str) or not key or type(exponent) not in (int, float) or not math.isfinite(float(exponent)):
            raise BindingContractError("binding_dimension_invalid")
        if float(exponent) != 0:
            result[key] = float(exponent)
    return result


def build_binding_contract(
    *,
    input_tables: Sequence[Mapping[str, Any]] = (),
    variables: Sequence[Mapping[str, Any]] = (),
    target: Mapping[str, Any] | None = None,
    constraints: Sequence[Mapping[str, Any]] = (),
    initial_conditions: Sequence[Mapping[str, Any]] = (),
    valid_domain: Mapping[str, Sequence[float]] | None = None,
    sources: Sequence[Mapping[str, Any]] = (),
    unresolved: Sequence[str] = (),
    data_version: str | None = None,
) -> dict[str, Any]:
    """Normalize roles and return a compile gate; missing units stay unresolved."""
    if not isinstance(input_tables, Sequence) or isinstance(input_tables, (str, bytes)):
        raise BindingContractError("input_tables_must_be_sequence")
    if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes)):
        raise BindingContractError("variables_must_be_sequence")
    seen = set()
    normalized_variables = []
    role_set = {"decision", "state", "parameter", "input", "target", "time", "entity", "latent", "constraint"}
    pending = [str(item) for item in unresolved if isinstance(item, str) and item]
    for item in variables:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str) or not item["id"]:
            raise BindingContractError("variable_binding_invalid")
        identifier = str(item["id"])
        if identifier in seen:
            raise BindingContractError("duplicate_variable_binding")
        seen.add(identifier)
        role = str(item.get("role", "input"))
        if role not in role_set:
            raise BindingContractError("variable_role_invalid")
        dimensions = _dimension(item.get("dimensions"))
        if dimensions is None:
            pending.append(f"unit:{identifier}")
        normalized_variables.append({"id": identifier, "role": role, "dimensions": dimensions,
                                     "shape": item.get("shape", []), "source": item.get("source")})
    tables = []
    for table in input_tables:
        if not isinstance(table, Mapping) or not isinstance(table.get("name"), str) or not isinstance(table.get("columns", []), list):
            raise BindingContractError("input_table_binding_invalid")
        tables.append({"name": table["name"], "columns": [str(col) for col in table.get("columns", [])],
                       "sampling_unit": table.get("sampling_unit"), "available_at": table.get("available_at")})
    domain = {}
    for key, bounds in (valid_domain or {}).items():
        if not isinstance(key, str) or not isinstance(bounds, (list, tuple)) or len(bounds) != 2 or any(type(v) not in (int, float) or not math.isfinite(float(v)) for v in bounds) or float(bounds[0]) >= float(bounds[1]):
            raise BindingContractError("valid_domain_invalid")
        domain[key] = [float(bounds[0]), float(bounds[1])]
    if target is not None:
        if not isinstance(target, Mapping) or not isinstance(target.get("id"), str):
            raise BindingContractError("target_binding_invalid")
        if target["id"] not in seen:
            pending.append("target_variable_not_bound")
    if not isinstance(data_version, (str, type(None))):
        raise BindingContractError("data_version_invalid")
    payload = {"schema_version": SCHEMA_VERSION, "input_tables": tables,
               "variables": normalized_variables, "target": dict(target) if target is not None else None,
               "constraints": [dict(item) for item in constraints if isinstance(item, Mapping)],
               "initial_conditions": [dict(item) for item in initial_conditions if isinstance(item, Mapping)],
               "valid_domain": domain, "sources": [dict(item) for item in sources if isinstance(item, Mapping)],
               "unresolved": sorted(set(pending)), "data_version": data_version}
    payload = _bounded_json(payload)
    binding_status = "ready" if not payload["unresolved"] and bool(payload["variables"]) else "needs_input"
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {**payload, "binding_status": binding_status, "binding_digest": digest,
            "compile_gate": "open" if binding_status == "ready" else "blocked",
            "policy": "missing_units_and_roles_remain_unresolved; no_implicit_dimensionless_binding"}


def plan_bound_subgraphs(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Select conservative backend families from explicit roles, never from names alone."""
    if not isinstance(contract, Mapping) or contract.get("schema_version") != SCHEMA_VERSION:
        raise BindingContractError("binding_contract_schema_mismatch")
    if contract.get("compile_gate") != "open":
        return {"status": "blocked", "reason": "binding_contract_not_ready", "missing": list(contract.get("unresolved", []))}
    roles = {item.get("role") for item in contract.get("variables", []) if isinstance(item, Mapping)}
    families = []
    if "state" in roles or "time" in roles:
        families.append("ode_or_dynamics")
    if "decision" in roles or contract.get("constraints"):
        families.append("constrained_optimization")
    if not families:
        families.append("typed_algebra_or_statistical_relation")
    return {"status": "planned", "binding_digest": contract.get("binding_digest"),
            "backend_families": families, "execution": "requires_backend_specific_contract",
            "policy": "role_bound_planning_not_numerical_solution"}


__all__ = ["SCHEMA_VERSION", "BindingContractError", "build_binding_contract", "plan_bound_subgraphs"]
