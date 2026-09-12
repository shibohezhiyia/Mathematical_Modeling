import numpy as np
import pytest

from core.ude_neural import fit_joint_neural_ude, fit_neural_ude, simulate_neural_ude_linear, simulate_neural_ude_stiff


def test_neural_ude_linear_simulation_executes_closed_loop_when_torch_available():
    pytest.importorskip("torch")
    features = np.linspace(-1.0, 1.0, 48).reshape(-1, 1)
    fit = fit_neural_ude(np.zeros(48), np.zeros(48), features, epochs=20, restarts=1)
    if fit.get("status") == "unavailable":
        pytest.skip("torch unavailable")
    result = simulate_neural_ude_linear(
        np.linspace(0.0, 0.2, 5), [0.1], [[-0.1]], [fit], state_bounds=[[-2.0, 2.0]])
    assert result["status"] == "executed"
    assert len(result["trajectory"]) == 5


def test_joint_neural_ude_fits_trajectory_and_reports_baseline():
    pytest.importorskip("torch")
    times = np.linspace(0.0, 1.0, 40)
    observations = np.exp(-0.5 * times).reshape(-1, 1)
    result = fit_joint_neural_ude(
        times, observations, [[-0.2]], epochs=20, hidden_dim=4, restarts=1,
    )
    if result.get("status") == "unavailable":
        pytest.skip("torch unavailable")
    assert result["status"] == "fitted_joint_ude"
    assert result["holdout_rows"] >= 8
    assert result["baseline_holdout_rmse"] >= 0.0
    assert result["learned_matrix"]
    assert result["network_state"]


def test_stiff_aware_neural_ude_backend_executes_bdf():
    pytest.importorskip("torch")
    features = np.linspace(-1.0, 1.0, 48).reshape(-1, 1)
    fit = fit_neural_ude(np.zeros(48), np.zeros(48), features, epochs=20, restarts=1)
    if fit.get("status") == "unavailable":
        pytest.skip("torch unavailable")
    result = simulate_neural_ude_stiff(
        np.linspace(0.0, 0.2, 5), [0.1], [[-20.0]], [fit], method="BDF")
    assert result["status"] == "executed"
    assert result["integrator"] == "BDF"
    assert len(result["trajectory"]) == 5
