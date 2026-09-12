import numpy as np
import pytest

from core.sindy_discovery import SINDyDiscoveryError, discover_sparse_dynamics, discover_weak_form_dynamics


def test_sindy_recovers_simple_polynomial_derivative_candidate():
    times = np.linspace(0, 10, 80)
    states = np.column_stack((times, times**2))
    result = discover_sparse_dynamics(times, states, ["x", "y"], polynomial_degree=2,
                                      sparsity_threshold=1e-4, residual_tolerance=1e-3)

    assert result["status"] == "candidate_found"
    assert result["proof_status"] == "tested_not_falsified"
    equations = {equation["lhs"]: equation["terms"] for equation in result["equations"]}
    assert any(term["term"] == "1" and abs(term["coefficient"] - 1) < 1e-3
               for term in equations["dx/dt"])
    assert any(term["term"] == "x" and abs(term["coefficient"] - 2) < 1e-3
               for term in equations["dy/dt"])


def test_sindy_dimension_filter_removes_incompatible_library_terms():
    times = np.linspace(0, 10, 80)
    states = np.column_stack((times, times**2))
    result = discover_sparse_dynamics(
        times, states, ["x", "y"], polynomial_degree=2,
        state_dimensions=[{"T": 1}, {"T": 2}],
    )
    assert result["dimension_filter"]["enabled"] is True
    assert result["equations"][0]["allowed_library_terms"] == ["1"]
    assert result["equations"][1]["allowed_library_terms"] == ["x"]


def test_sindy_rejects_invalid_library_and_time_contract():
    times = np.arange(16, dtype=float)
    states = np.column_stack((times, times**2))
    with pytest.raises(SINDyDiscoveryError, match="strictly_increasing"):
        discover_sparse_dynamics([0, 1, 1] + list(range(3, 16)), states)
    with pytest.raises(SINDyDiscoveryError, match="polynomial_library"):
        discover_sparse_dynamics(times, np.column_stack([times] * 8), polynomial_degree=3)
    with pytest.raises(SINDyDiscoveryError, match="time_dimension_must_be_mapping"):
        discover_sparse_dynamics(times, states, time_dimension=[{"T": 1}])


def test_weak_form_sindy_produces_bounded_candidate_and_holdout_windows():
    times = np.linspace(0, 10, 80)
    states = np.column_stack((times, times**2))
    result = discover_weak_form_dynamics(times, states, ["x", "y"], polynomial_degree=2,
                                          window_count=12)
    assert result["status"] == "candidate_generated"
    assert result["train_windows"] + result["validation_windows"] == result["window_count"]
    assert result["policy"].endswith("ode_proof")


def test_sindy_exposes_bounded_heldout_counterexamples_for_cegis():
    times = np.linspace(0, 12, 80)
    states = np.column_stack((times, np.sin(times)))
    result = discover_sparse_dynamics(times, states, ["x", "y"], polynomial_degree=1,
                                      sparsity_threshold=1e-3, residual_tolerance=1e-5)
    assert result["status"] == "candidate_rejected_by_validation"
    assert result["proof_status"] == "counterexample_found"
    assert result["counterexamples"]
    assert len(result["counterexamples"]) <= 32
