import pytest

from core.binding_contract import build_binding_contract, plan_bound_subgraphs


def test_binding_contract_does_not_assume_missing_units_are_dimensionless():
    contract = build_binding_contract(
        input_tables=[{"name": "observations", "columns": ["x", "y"], "sampling_unit": "row"}],
        variables=[{"id": "x", "role": "input", "dimensions": {"L": 1}}, {"id": "y", "role": "target"}],
        target={"id": "y", "role": "target"},
    )
    assert contract["binding_status"] == "needs_input"
    assert "unit:y" in contract["unresolved"]
    assert contract["compile_gate"] == "blocked"
    assert plan_bound_subgraphs(contract)["status"] == "blocked"


def test_ready_binding_contract_plans_roles_without_solving():
    contract = build_binding_contract(
        variables=[
            {"id": "t", "role": "time", "dimensions": {"T": 1}},
            {"id": "x", "role": "state", "dimensions": {"Q": 1}},
        ],
        target={"id": "x", "role": "target"},
        initial_conditions=[{"variable": "x", "value": 1.0}],
        valid_domain={"t": [0, 10]},
    )
    assert contract["binding_status"] == "ready"
    plan = plan_bound_subgraphs(contract)
    assert plan["status"] == "planned"
    assert "ode_or_dynamics" in plan["backend_families"]


def test_binding_contract_rejects_invalid_domain():
    with pytest.raises(ValueError, match="valid_domain_invalid"):
        build_binding_contract(variables=[{"id": "x", "dimensions": {"Q": 1}}], valid_domain={"x": [1, 1]})
