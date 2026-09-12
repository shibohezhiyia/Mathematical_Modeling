import pytest

from core.math_primitives import (
    PrimitiveValidationError,
    primitive_catalog,
    validate_primitive_node,
)


def node(op, kind, inputs=None, dimensions=None, attributes=None):
    return {
        "id": "n1", "op": op, "inputs": list(inputs or []), "kind": kind,
        "dimensions": dimensions, "attributes": attributes or {},
    }


def test_catalog_contains_domain_independent_minimal_families():
    catalog = primitive_catalog()
    names = set(catalog["primitives"])
    assert {"variable", "equation", "derivative", "integral", "distribution",
            "event", "graph", "objective", "constraint", "control", "observation",
            "unknown_mechanism", "latent_state", "regime_switch"} <= names
    assert catalog["schema_version"] == "mathmodel.math-primitives/v1"


def test_validate_primitive_reports_obligations_without_granting_execution():
    result = validate_primitive_node(node("derivative", "quantity", ["state", "time"], {"L": 1}))
    assert result["category"] == "calculus"
    assert "initial_or_boundary_condition_required" in result["obligations"]
    assert result["executable"] is False


def test_validate_primitive_rejects_unknown_or_mismatched_nodes():
    with pytest.raises(PrimitiveValidationError, match="unknown_primitive"):
        validate_primitive_node(node("custom_solver", "quantity"))
    with pytest.raises(PrimitiveValidationError, match="output_kind_mismatch"):
        validate_primitive_node(node("objective", "quantity", ["x"]))
    with pytest.raises(PrimitiveValidationError, match="input_kind_mismatch"):
        validate_primitive_node(node("derivative", "quantity", ["state", "state"], {"L": 1}),
                                input_kinds=("quantity", "quantity"))


def test_unresolved_discovery_primitives_are_non_executable_and_carry_obligations():
    unknown = validate_primitive_node(node("unknown_mechanism", "quantity", ["x"], {"L": 1}))
    assert unknown["executable"] is False
    assert "mechanism_search_budget_required" in unknown["obligations"]
    latent = validate_primitive_node(node("latent_state", "quantity", [], {"L": 1}))
    assert "latent_state_identification_required" in latent["obligations"]
