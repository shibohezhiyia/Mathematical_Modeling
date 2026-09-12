"""Typed contracts for the minimal, domain-independent math primitive layer.

This module deliberately describes *interfaces*, not complete solvers.  A
candidate graph may be composed from these primitives only after the consumer
checks the declared arity, value kind, dimensions and domain obligations.
Unknown or domain-specific operations are rejected instead of being silently
treated as executable Python.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Mapping


SCHEMA_VERSION = "mathmodel.math-primitives/v1"


class PrimitiveValidationError(ValueError):
    def __init__(self, code: str, node_id: str | None = None):
        self.code = code
        self.node_id = node_id
        super().__init__(code if node_id is None else f"{code}:{node_id}")


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    category: str
    min_inputs: int
    max_inputs: int
    input_kinds: tuple[str, ...]
    output_kind: str
    requires_dimensions: bool
    obligations: tuple[str, ...] = ()

    def public(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "min_inputs": self.min_inputs,
            "max_inputs": self.max_inputs,
            "input_kinds": list(self.input_kinds),
            "output_kind": self.output_kind,
            "requires_dimensions": self.requires_dimensions,
            "obligations": list(self.obligations),
        }


PRIMITIVES: dict[str, PrimitiveSpec] = {
    "variable": PrimitiveSpec("variable", "symbol", 0, 0, (), "quantity", True),
    "coordinate": PrimitiveSpec("coordinate", "symbol", 0, 0, (), "coordinate", True),
    "constant": PrimitiveSpec("constant", "symbol", 0, 0, (), "quantity", True),
    "equation": PrimitiveSpec("equation", "relation", 2, 2, ("quantity", "quantity"), "boolean", True),
    "derivative": PrimitiveSpec("derivative", "calculus", 2, 2, ("quantity", "coordinate"), "quantity", True,
                                ("initial_or_boundary_condition_required",)),
    "integral": PrimitiveSpec("integral", "calculus", 2, 2, ("quantity", "coordinate"), "quantity", True,
                               ("integration_domain_required",)),
    "distribution": PrimitiveSpec("distribution", "probability", 0, 16, (), "distribution", True,
                                   ("support_or_domain_required",)),
    "event": PrimitiveSpec("event", "event", 1, 16, ("boolean",), "event", True,
                            ("event_boundary_semantics_required",)),
    "graph": PrimitiveSpec("graph", "network", 0, 256, (), "graph", False,
                            ("node_and_edge_semantics_required",)),
    "objective": PrimitiveSpec("objective", "optimization", 1, 1, ("quantity",), "objective", True,
                                ("optimization_sense_required",)),
    "constraint": PrimitiveSpec("constraint", "optimization", 1, 1, ("boolean",), "constraint", True,
                                 ("feasible_domain_required",)),
    "control": PrimitiveSpec("control", "optimization", 1, 16, ("quantity",), "quantity", True,
                              ("control_bounds_required",)),
    "observation": PrimitiveSpec("observation", "observation", 1, 1, ("quantity",), "quantity", True,
                                  ("observation_noise_or_identity_required",)),
    # Closed arithmetic primitives are intentionally small and data-free.  A
    # graph using them can be evaluated by the typed runtime without parsing
    # generated source; dimensions are checked by ``primitive_graph``.
    "add": PrimitiveSpec("add", "algebra", 2, 2, ("quantity", "quantity"), "quantity", True),
    "subtract": PrimitiveSpec("subtract", "algebra", 2, 2, ("quantity", "quantity"), "quantity", True),
    "multiply": PrimitiveSpec("multiply", "algebra", 2, 2, ("quantity", "quantity"), "quantity", True),
    "divide": PrimitiveSpec("divide", "algebra", 2, 2, ("quantity", "quantity"), "quantity", True),
    "power": PrimitiveSpec("power", "algebra", 2, 2, ("quantity", "quantity"), "quantity", True),
    "negate": PrimitiveSpec("negate", "algebra", 1, 1, ("quantity",), "quantity", True),
    "absolute": PrimitiveSpec("absolute", "algebra", 1, 1, ("quantity",), "quantity", True),
    "exp": PrimitiveSpec("exp", "algebra", 1, 1, ("quantity",), "quantity", True),
    "log": PrimitiveSpec("log", "algebra", 1, 1, ("quantity",), "quantity", True),
    "sin": PrimitiveSpec("sin", "algebra", 1, 1, ("quantity",), "quantity", True),
    "cos": PrimitiveSpec("cos", "algebra", 1, 1, ("quantity",), "quantity", True),
    # These three nodes intentionally describe an unresolved modelling choice.
    # They are type-checked graph objects, never executable callbacks.
    "unknown_mechanism": PrimitiveSpec("unknown_mechanism", "discovery", 0, 16, (), "quantity", True,
                                        ("mechanism_search_budget_required",)),
    "latent_state": PrimitiveSpec("latent_state", "discovery", 0, 16, (), "quantity", True,
                                   ("latent_state_identification_required",)),
    "regime_switch": PrimitiveSpec("regime_switch", "dynamics", 1, 16, (), "quantity", True,
                                    ("event_boundary_semantics_required",)),
}


def primitive_schema() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "node": {
            "required": ["id", "op", "inputs", "kind", "dimensions", "attributes"],
            "properties": {
                "id": {"type": "string", "maxLength": 64},
                "op": {"enum": sorted(PRIMITIVES)},
                "inputs": {"type": "array", "maxItems": 256},
                "kind": {"enum": ["quantity", "coordinate", "boolean", "distribution", "event", "graph", "objective", "constraint"]},
                "dimensions": {"type": ["object", "null"]},
                "attributes": {"type": "object"},
            },
        },
        "policy": "schema_validation_is_not_numerical_or_reality_correctness",
    }


def _identifier(value: Any, code: str) -> str:
    if type(value) is not str or not value or len(value) > 64 or any(ch.isspace() for ch in value):
        raise PrimitiveValidationError(code)
    return value


def _dimensions(value: Any) -> None:
    if value is None:
        return
    if type(value) is not dict or len(value) > 16:
        raise PrimitiveValidationError("invalid_dimensions")
    for key, exponent in value.items():
        if type(key) is not str or len(key) > 32 or type(exponent) not in (int, float):
            raise PrimitiveValidationError("invalid_dimensions")
        if not math.isfinite(float(exponent)) or abs(float(exponent)) > 1e6:
            raise PrimitiveValidationError("invalid_dimensions")


def validate_primitive_node(node: Mapping[str, Any], *, input_kinds: tuple[str, ...] = ()) -> dict[str, Any]:
    """Validate one non-executable primitive node and return a safe summary."""
    if type(node) is not dict:
        raise PrimitiveValidationError("node_must_be_object")
    required = {"id", "op", "inputs", "kind", "dimensions", "attributes"}
    if set(node) != required:
        raise PrimitiveValidationError("invalid_node_fields")
    node_id = _identifier(node["id"], "invalid_node_id")
    op = node["op"]
    if type(op) is not str or op not in PRIMITIVES:
        raise PrimitiveValidationError("unknown_primitive", node_id)
    spec = PRIMITIVES[op]
    inputs = node["inputs"]
    if type(inputs) is not list or not spec.min_inputs <= len(inputs) <= spec.max_inputs:
        raise PrimitiveValidationError("invalid_input_arity", node_id)
    if any(type(item) is not str for item in inputs):
        raise PrimitiveValidationError("invalid_input_reference", node_id)
    kind = node["kind"]
    if kind != spec.output_kind:
        raise PrimitiveValidationError("output_kind_mismatch", node_id)
    _dimensions(node["dimensions"])
    if type(node["attributes"]) is not dict or len(node["attributes"]) > 32:
        raise PrimitiveValidationError("invalid_attributes", node_id)
    if input_kinds:
        if len(input_kinds) != len(inputs):
            raise PrimitiveValidationError("input_kind_count_mismatch", node_id)
        for actual, expected in zip(input_kinds, spec.input_kinds or ("quantity",) * len(inputs)):
            if expected != actual:
                raise PrimitiveValidationError("input_kind_mismatch", node_id)
    return {
        "id": node_id, "op": op, "category": spec.category,
        "kind": kind, "input_count": len(inputs),
        "obligations": list(spec.obligations),
        "executable": op in {"variable", "constant", "observation"},
    }


def primitive_catalog() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "primitives": {name: spec.public() for name, spec in sorted(PRIMITIVES.items())},
        "schema": primitive_schema(),
    }


__all__ = [
    "SCHEMA_VERSION", "PrimitiveValidationError", "PrimitiveSpec", "PRIMITIVES",
    "primitive_schema", "primitive_catalog", "validate_primitive_node",
]
