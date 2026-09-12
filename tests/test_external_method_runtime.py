import numpy as np
import pytest

from core.external_method_runtime import ExternalMethodRuntimeError, execute_external_method
from core.execution_readiness import assess_execution_readiness
from core.ude_neural import predict_neural_ude


def _trajectory(rows=80):
    times = np.linspace(0.0, 4.0, rows)
    states = np.column_stack([np.exp(-times), 2.0 * np.exp(-0.5 * times)])
    return times.tolist(), states.tolist()


def test_runtime_executes_local_sindy_adapter():
    times, states = _trajectory()
    result = execute_external_method("sindy", {
        "times": times, "states": states, "state_names": ["x", "y"],
        "polynomial_degree": 1, "residual_tolerance": 10.0,
    })
    assert result["status"] == "executed"
    assert result["result"]["schema_version"].startswith("mathmodel.sindy-discovery/")
    assert result["execution_readiness"]["unit"] == "not_assessed"
    assert result["execution_readiness"]["source"] == "not_assessed"
    assert assess_execution_readiness(result["execution_readiness"])["status"] == "not_assessed"
    assert result["conclusion_certificate"]["status"] == "not_assessed"


def test_runtime_executes_weak_form_adapter():
    times, states = _trajectory(96)
    result = execute_external_method("weak_sindy", {
        "times": times, "states": states, "state_names": ["x", "y"],
        "polynomial_degree": 1, "residual_tolerance": 10.0,
        "window_count": 8,
    })
    assert result["status"] == "executed"
    assert result["result"]["window_count"] == 8


def test_runtime_keeps_pde_as_typed_plan_until_fitting_backend_is_added():
    result = execute_external_method("pde_find", {
        "field_shape": [8, 9], "coordinate_names": ["x", "t"],
        "derivative_orders": [1, 2],
    })
    assert result["status"] == "planned"
    assert result["result"]["status"] == "typed_not_fitted"


def test_runtime_executes_bounded_regular_grid_pde_candidate():
    times = np.linspace(0.0, 3.0, 24)
    coordinates = np.linspace(0.0, 2.0 * np.pi, 16)
    field = np.array([[np.exp(-0.2 * t) * np.sin(x - 0.3 * t) for x in coordinates] for t in times])
    result = execute_external_method("pde_find", {
        "times": times.tolist(), "coordinates": coordinates.tolist(), "field": field.tolist(),
        "residual_tolerance": 1.0, "boundary_tolerance": 1.0,
    })
    assert result["status"] == "executed"
    assert result["result"]["schema_version"].startswith("mathmodel.pde-discovery/")


def test_runtime_routes_2d_pde_boundaries_to_executor():
    times = np.linspace(0.0, 0.8, 20)
    x = np.linspace(0.0, 1.0, 9)
    y = np.linspace(0.0, 1.0, 9)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = np.asarray([
        np.exp(-0.4 * t) * np.sin(np.pi * xx) * np.sin(np.pi * yy)
        for t in times
    ])
    result = execute_external_method("pde_find", {
        "times": times.tolist(), "x_coordinates": x.tolist(), "y_coordinates": y.tolist(),
        "field": field.tolist(), "include_advection": False,
        "boundary_values": {
            "left": field[:, 0, :].tolist(), "right": field[:, -1, :].tolist(),
            "bottom": field[:, :, 0].tolist(), "top": field[:, :, -1].tolist(),
        },
        "residual_tolerance": 0.2, "boundary_tolerance": 1e-12,
    })
    assert result["status"] == "executed"
    assert result["result"]["boundary_status"] == "declared_applied"
    assert result["result"]["boundary_condition_rmse"] == 0.0


def test_runtime_routes_3d_pde_to_executor():
    times = np.linspace(0.0, 0.8, 20)
    x = np.linspace(0.0, 1.0, 5); y = np.linspace(0.0, 1.0, 5); z = np.linspace(0.0, 1.0, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    field = np.asarray([
        np.exp(-0.4 * t) * np.sin(np.pi * xx) * np.sin(np.pi * yy) * np.sin(np.pi * zz)
        for t in times
    ])
    result = execute_external_method("pde_find", {
        "times": times.tolist(), "x_coordinates": x.tolist(), "y_coordinates": y.tolist(),
        "z_coordinates": z.tolist(), "field": field.tolist(), "include_advection": False,
        "residual_tolerance": 0.5,
    })
    assert result["status"] == "executed"
    assert result["result"]["grid"]["field_shape"] == [20, 5, 5, 5]


def test_runtime_executes_bounded_ude_linear_adapter():
    target = np.linspace(1.0, 2.0, 32)
    known = target - 0.5
    features = np.column_stack([np.ones(32), np.linspace(0.0, 1.0, 32)])
    result = execute_external_method("ude", {
        "target_rhs": target.tolist(), "known_rhs": known.tolist(),
        "correction_features": features.tolist(), "max_condition_number": 1e12,
    })
    assert result["status"] == "executed"
    assert result["result"]["stability_status"] == "assessed"
    assert result["execution_readiness"]["resource"] == "ready"
    assert result["execution_readiness"]["security"] == "ready"


def test_runtime_executes_optional_neural_ude_with_holdout():
    rng = np.random.default_rng(5)
    features = rng.normal(size=(40, 2))
    known = np.linspace(0.0, 1.0, 40)
    target = known + 0.2 * features[:, 0]
    result = execute_external_method("ude_neural", {
        "target_rhs": target.tolist(), "known_rhs": known.tolist(),
        "correction_features": features.tolist(), "epochs": 20, "restarts": 1,
    })
    assert result["status"] == "executed"
    assert result["result"]["schema_version"].startswith("mathmodel.ude-neural-fit/")
    assert result["result"]["holdout_rmse"] < result["result"]["baseline_holdout_rmse"]
    assert result["result"]["network_state"]
    prediction = predict_neural_ude(features[:5].tolist(), result["result"])
    assert len(prediction) == 5
    assert np.allclose(prediction, predict_neural_ude(features[:5].tolist(), result["result"]))


def test_runtime_executes_joint_neural_ude_trajectory_backend():
    pytest.importorskip("torch")
    times = np.linspace(0.0, 1.0, 40)
    observations = np.exp(-0.5 * times).reshape(-1, 1)
    result = execute_external_method("ude_joint", {
        "times": times.tolist(), "observations": observations.tolist(),
        "known_matrix": [[-0.2]], "epochs": 20, "hidden_dim": 4, "restarts": 1,
    })
    if result["result"].get("status") == "unavailable":
        pytest.skip("torch unavailable")
    assert result["status"] == "executed"
    assert result["result"]["status"] == "fitted_joint_ude"


def test_runtime_executes_stiff_aware_ude_backend():
    pytest.importorskip("torch")
    from core.ude_neural import fit_neural_ude
    features = np.linspace(-1.0, 1.0, 48).reshape(-1, 1)
    fit = fit_neural_ude(np.zeros(48), np.zeros(48), features, epochs=20, restarts=1)
    if fit.get("status") == "unavailable":
        pytest.skip("torch unavailable")
    result = execute_external_method("ude_stiff", {
        "times": np.linspace(0.0, 0.2, 5).tolist(), "initial_state": [0.1],
        "linear_matrix": [[-20.0]], "fit_results": [fit], "method": "BDF",
    })
    assert result["status"] == "executed"
    assert result["result"]["integrator"] == "BDF"


def test_runtime_neural_ude_checks_declared_domain_and_rhs_dimensions():
    rng = np.random.default_rng(7)
    features = rng.normal(size=(40, 2))
    known = np.linspace(0.0, 1.0, 40)
    target = known + 0.1 * features[:, 0]
    result = execute_external_method("ude_neural", {
        "target_rhs": target.tolist(), "known_rhs": known.tolist(),
        "correction_features": features.tolist(), "epochs": 20, "restarts": 1,
        "feature_domains": [[-4, 4], [-4, 4]],
        "target_dimensions": {"L": 1}, "known_rhs_dimensions": {"L": 1},
    })
    assert result["status"] == "executed"
    assert result["result"]["unit_status"] == "known_and_target_rhs_dimensions_match"

    rejected = execute_external_method("ude_neural", {
        "target_rhs": target.tolist(), "known_rhs": known.tolist(),
        "correction_features": features.tolist(), "epochs": 20, "restarts": 1,
        "feature_domains": [[-0.1, 0.1], [-4, 4]],
    })
    assert rejected["status"] == "rejected"


def test_runtime_never_executes_llm_generated_code():
    result = execute_external_method("llm_sr", {"source": "__import__('os').system('x')"})
    assert result["status"] == "proposal_only"
    assert "not_an_execution_input" in result["reason"]


def test_runtime_executes_only_typed_llm_sr_tree_with_holdout():
    x = np.linspace(-2.0, 2.0, 32)
    result = execute_external_method("llm_sr", {
        "target": (2.0 * x + 1.0).tolist(), "features": {"x": x.tolist()},
        "expression": {"op": "add", "left": {"op": "multiply", "left": {"op": "param", "name": "a"}, "right": {"op": "var", "name": "x"}}, "right": {"op": "param", "name": "b"}},
        "parameter_bounds": {"a": [-5, 5], "b": [-5, 5]}, "max_nfev": 80,
    })
    assert result["status"] == "executed"
    assert result["result"]["status"] == "fitted"
    assert result["result"]["holdout_rmse"] < 1e-6


def test_runtime_rejects_untyped_llm_sr_expression_fields():
    with pytest.raises(ExternalMethodRuntimeError, match="llm_sr_payload_contains_unknown_fields"):
        execute_external_method("llm_sr", {"expression": {"source": "x"}, "source": "x"})


def test_runtime_enforces_typed_llm_sr_basis_contract():
    x = np.linspace(-2.0, 2.0, 32)
    result = execute_external_method("llm_sr", {
        "target": x.tolist(), "features": {"x": x.tolist()},
        "expression": {"op": "sin", "arg": {"op": "var", "name": "x"}},
        "parameter_bounds": {"a": [-1, 1]}, "allowed_operators": ["add"],
    })
    assert result["status"] == "rejected"


def test_runtime_evaluates_typed_llm_sr_candidate_pool_without_auto_approval():
    x = np.linspace(-2.0, 2.0, 32)
    linear = {"op": "add", "left": {"op": "multiply", "left": {"op": "param", "name": "a"}, "right": {"op": "var", "name": "x"}}, "right": {"op": "param", "name": "b"}}
    result = execute_external_method("llm_sr", {
        "target": (2.0 * x + 1.0).tolist(), "features": {"x": x.tolist()},
        "candidates": [
            {"expression": linear, "parameter_bounds": {"a": [-5, 5], "b": [-5, 5]}},
            {"expression": {"op": "const", "value": 0}, "parameter_bounds": {}},
        ],
        "max_nfev": 80,
    })
    assert result["status"] == "executed"
    assert result["result"]["status"] == "candidates_need_confirmation"
    assert result["result"]["best_id"] == "candidate_0"


def test_runtime_runs_bounded_typed_llm_sr_search_and_keeps_lineage():
    x = np.linspace(-2.0, 2.0, 32)
    expression = {"op": "add", "left": {"op": "multiply", "left": {"op": "param", "name": "a"}, "right": {"op": "var", "name": "x"}}, "right": {"op": "const", "value": 0.9}}
    result = execute_external_method("llm_sr", {
        "target": (2.0 * x + 1.0).tolist(), "features": {"x": x.tolist()},
        "candidates": [{"expression": expression, "parameter_bounds": {"a": [-5, 5]}}],
        "search": {"generations": 2, "max_candidates": 8, "beam_width": 1}, "max_nfev": 80,
    })
    assert result["status"] == "executed"
    assert result["result"]["generations"]
    assert len(result["result"]["lineage"]) >= 2


def test_runtime_blocks_typed_tree_with_dimensionally_invalid_addition():
    x = np.linspace(1.0, 2.0, 32)
    result = execute_external_method("llm_sr", {
            "target": x.tolist(), "features": {"x": x.tolist()},
            "expression": {"op": "add", "left": {"op": "param", "name": "a"}, "right": {"op": "var", "name": "x"}},
            "parameter_bounds": {"a": [-5, 5]},
            "feature_dimensions": {"x": {"L": 1}}, "target_dimensions": {"L": 1},
            "parameter_dimensions": {"a": {"T": 1}},
        })
    assert result["status"] == "rejected"


def test_runtime_builds_finite_certificate_only_with_explicit_boundary():
    times, states = _trajectory()
    result = execute_external_method("sindy", {
        "times": times, "states": states, "state_names": ["x", "y"],
        "polynomial_degree": 1, "residual_tolerance": 10.0,
        "applicability_boundary": "t in [0, 4] and observed state domain",
        "assumptions": ["time grid is ordered"],
    })
    certificate = result["conclusion_certificate"]
    assert certificate["status"] == "tested_not_falsified"
    assert certificate["applicability_boundary"]
    assert certificate["residuals"]


def test_runtime_blocks_malformed_certificate_metadata_without_losing_result():
    times, states = _trajectory()
    result = execute_external_method("sindy", {
        "times": times, "states": states,
        "applicability_boundary": "t in [0, 4]",
        "assumptions": "not-a-list",
    })
    assert result["status"] == "executed"
    assert result["conclusion_certificate"]["status"] == "blocked"


def test_runtime_rejects_unknown_payload_fields_and_methods():
    with pytest.raises(ExternalMethodRuntimeError, match="sindy_payload_contains_unknown_fields"):
        execute_external_method("sindy", {"times": [0, 1], "states": [[0], [1]], "eval": "bad"})
    with pytest.raises(ExternalMethodRuntimeError, match="method_invalid"):
        execute_external_method("made_up", {})
