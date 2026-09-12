import pytest

from core.hypothesis_controls import HypothesisControlError, apply_hypothesis_controls, affected_nodes_for_control, build_hypothesis_controls


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
