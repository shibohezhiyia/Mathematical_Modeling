import pytest

from core.hypothesis_controls import (
    HypothesisControlError, apply_hypothesis_controls, affected_nodes_for_control,
    build_dynamic_hypothesis_controls, build_hypothesis_controls,
)


def test_hypothesis_controls_validate_slider_and_downstream_closure():
    controls = build_hypothesis_controls([{"id": "k", "label": "rate", "min": 0, "max": 2, "default": 1, "step": 0.1, "affected_nodes": ["a"]}], node_ids=["a", "b", "c"])
    assert controls["status"] == "validated"
    plan = affected_nodes_for_control({"nodes": [{"id": "a", "inputs": []}, {"id": "b", "inputs": ["a"]}, {"id": "c", "inputs": ["b"]}]}, affected_nodes=["a"])
    assert plan["recompute_nodes"] == ["a", "b", "c"]


def test_hypothesis_controls_reject_unknown_nodes_and_bad_bounds():
    with pytest.raises(HypothesisControlError):
        build_hypothesis_controls([{"id": "k", "min": 1, "max": 1, "default": 1, "step": 1, "affected_nodes": ["a"]}], node_ids=["a"])
    with pytest.raises(HypothesisControlError):
        affected_nodes_for_control({"nodes": [{"id": "a", "inputs": []}]}, affected_nodes=["unknown"])


def test_apply_controls_updates_explicit_binding_and_records_version():
    controls = build_hypothesis_controls([{"id": "gain", "parameter_id": "k", "version": 2,
                                           "min": 0, "max": 2, "default": 1, "step": .1,
                                           "affected_nodes": ["k"]}], node_ids=["k"])
    preview = apply_hypothesis_controls(controls, {"gain": 1.5}, {"k": 1.0})
    assert preview["bindings"]["k"] == 1.5
    assert preview["changes"][0]["version"] == 2


def test_dynamic_contract_generates_bounded_coefficient_controls_and_paths():
    contract = {
        "families": [{
            "family": "ode",
            "candidates": [{"coefficients": [[-0.5, 0.2], [1.0, 0.0]]}],
            "cases": [],
        }]
    }
    generated = build_dynamic_hypothesis_controls(contract)
    assert len(generated["controls"]) == 4
    assert generated["bindings"]["dynamic_families_0_candidates_0_coefficients_0_0"] == -0.5
    assert generated["dynamic_paths"]["dynamic_families_0_candidates_0_coefficients_1_0"] == (
        "families.0.candidates.0.coefficients.1.0"
    )
    validated = build_hypothesis_controls(generated["controls"], node_ids=generated["node_ids"])
    assert validated["status"] == "validated"


def test_dynamic_control_generation_does_not_expose_observations_or_bounds():
    contract = {
        "families": [{"family": "ode", "candidates": [{
            "coefficients": [[-1.0]], "bounds": [[0.0, 10.0]],
        }], "cases": [{"times": [0.0, 1.0], "observations": [[1.0], [0.0]]}]}]
    }
    generated = build_dynamic_hypothesis_controls(contract)
    assert len(generated["controls"]) == 1
    assert all("observations" not in path and "bounds" not in path for path in generated["dynamic_paths"].values())


def test_dynamic_gnn_hyperparameter_controls_preserve_integer_contract_types():
    contract = {
        "families": [{"family": "gnn", "candidates": [{
            "epochs": 20, "hidden_dim": 16, "edge_threshold": 0.5,
        }], "cases": []}],
    }
    generated = build_dynamic_hypothesis_controls(contract)
    by_path = {path: generated["controls"][index] for index, path in enumerate(generated["dynamic_paths"].values())}
    assert any(control["value_type"] == "int" for control in by_path.values())
    epochs_id = next(key for key, path in generated["dynamic_paths"].items() if path.endswith(".epochs"))
    assert generated["bindings"][epochs_id] == 20
    assert next(control for control in generated["controls"] if control["id"] == epochs_id)["step"] == 1.0
