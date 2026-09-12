"""Optional neural residual correction for Universal Differential Equations."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.ude-neural-fit/v1"


class NeuralUDEError(ValueError):
    pass


def fit_neural_ude(
    target_rhs: Sequence[float], known_rhs: Sequence[float], correction_features: Sequence[Sequence[float]],
    *, validation_fraction: float = 0.25, epochs: int = 200, hidden_dim: int = 16,
    restarts: int = 2, learning_rate: float = 0.01, max_rows: int = 20_000,
    random_state: int = 0, feature_domains: Sequence[Sequence[float]] | None = None,
    target_dimensions: Mapping[str, float] | None = None,
    known_rhs_dimensions: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Fit ``known_rhs + g(features)`` with a tiny bounded MLP."""
    target = np.asarray(target_rhs, dtype=np.float32).reshape(-1)
    known = np.asarray(known_rhs, dtype=np.float32).reshape(-1)
    features = np.asarray(correction_features, dtype=np.float32)
    if target.size < 32 or target.size > max_rows or known.shape != target.shape or features.ndim != 2 or features.shape[0] != target.size:
        raise NeuralUDEError("ude_neural_shapes_or_size_invalid")
    if features.shape[1] < 1 or features.shape[1] > 128 or not np.isfinite(target).all() or not np.isfinite(known).all() or not np.isfinite(features).all():
        raise NeuralUDEError("ude_neural_values_invalid")
    unit_status = "not_assessed"
    if feature_domains is not None:
        if not isinstance(feature_domains, Sequence) or len(feature_domains) != features.shape[1]:
            raise NeuralUDEError("ude_neural_feature_domains_invalid")
        for index, bounds in enumerate(feature_domains):
            if (not isinstance(bounds, Sequence) or len(bounds) != 2 or
                    not all(type(value) in (int, float) and math.isfinite(float(value)) for value in bounds) or
                    float(bounds[0]) >= float(bounds[1])):
                raise NeuralUDEError("ude_neural_feature_domain_invalid")
            if np.min(features[:, index]) < float(bounds[0]) or np.max(features[:, index]) > float(bounds[1]):
                raise NeuralUDEError("ude_neural_feature_domain_violation")
    if target_dimensions is not None or known_rhs_dimensions is not None:
        if not isinstance(target_dimensions, Mapping) or not isinstance(known_rhs_dimensions, Mapping) or dict(target_dimensions) != dict(known_rhs_dimensions):
            raise NeuralUDEError("ude_neural_rhs_dimension_mismatch")
        unit_status = "known_and_target_rhs_dimensions_match"
    if not 0.1 <= float(validation_fraction) <= 0.4:
        raise NeuralUDEError("ude_neural_validation_fraction_invalid")
    if type(epochs) is not int or not 20 <= epochs <= 1_000 or type(hidden_dim) is not int or not 4 <= hidden_dim <= 64:
        raise NeuralUDEError("ude_neural_budget_invalid")
    if type(restarts) is not int or not 1 <= restarts <= 3 or not 0 < float(learning_rate) <= 1:
        raise NeuralUDEError("ude_neural_options_invalid")
    try:
        import torch
        from torch import nn
    except ImportError:
        return {"schema_version": SCHEMA_VERSION, "status": "unavailable", "reason": "torch_not_installed"}
    split = max(16, min(target.size - 8, int(math.floor(target.size * (1.0 - float(validation_fraction))))))
    if split < 16 or target.size - split < 8:
        raise NeuralUDEError("ude_neural_holdout_too_small")
    x_train, x_valid = features[:split], features[split:]
    y_valid = target[split:]
    f_mean, f_scale = x_train.mean(0), np.maximum(x_train.std(0), 1e-6)
    x_train = (x_train - f_mean) / f_scale
    x_valid = (x_valid - f_mean) / f_scale
    tx, vx = torch.tensor(x_train), torch.tensor(x_valid)
    tknown = torch.tensor(known[:split])
    target_train = torch.tensor(target[:split])
    vknown = torch.tensor(known[split:])
    baseline_rmse = float(np.sqrt(np.mean((known[split:] - target[split:]) ** 2)))
    holdout_scores, train_scores = [], []
    selected_state, selected_score, selected_index = None, float("inf"), None
    for restart in range(restarts):
        torch.manual_seed(int(random_state + restart))
        network = nn.Sequential(nn.Linear(features.shape[1], hidden_dim), nn.Tanh(),
                                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        optimizer = torch.optim.Adam(network.parameters(), lr=float(learning_rate), weight_decay=1e-4)
        best, best_state, stale = float("inf"), None, 0
        for _ in range(epochs):
            network.train(); optimizer.zero_grad(set_to_none=True)
            correction = network(tx).squeeze(-1)
            # The network predicts the residual, but the objective is defined
            # on the full RHS.  Comparing against ``ty`` (the residual target)
            # would subtract the known mechanism twice and silently train the
            # wrong model.
            loss = torch.mean((tknown + correction - target_train) ** 2)
            loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(), 5.0); optimizer.step()
            network.eval()
            with torch.no_grad():
                valid_loss = float(torch.mean((vknown + network(vx).squeeze(-1) - torch.tensor(target[split:])) ** 2))
            if valid_loss + 1e-7 < best:
                best, best_state, stale = valid_loss, {k: v.detach().clone() for k, v in network.state_dict().items()}, 0
            else:
                stale += 1
                if stale >= 25:
                    break
        if best_state is not None:
            network.load_state_dict(best_state)
        network.eval()
        with torch.no_grad():
            train_prediction = tknown + network(tx).squeeze(-1)
            valid_prediction = vknown + network(vx).squeeze(-1)
        train_scores.append(float(torch.sqrt(torch.mean((train_prediction - target_train) ** 2))))
        holdout_scores.append(float(torch.sqrt(torch.mean((valid_prediction - torch.tensor(target[split:])) ** 2))))
        if holdout_scores[-1] < selected_score and best_state is not None:
            selected_score = holdout_scores[-1]
            selected_state = {key: value.detach().cpu().numpy().tolist() for key, value in best_state.items()}
            selected_index = restart
    holdout_rmse = float(np.mean(holdout_scores))
    status = "fitted_neural" if math.isfinite(holdout_rmse) else "rejected_nonfinite"
    return {
        "schema_version": SCHEMA_VERSION, "status": status, "train_rows": int(split),
        "holdout_rows": int(target.size - split), "train_rmse": float(np.mean(train_scores)),
        "holdout_rmse": holdout_rmse, "baseline_holdout_rmse": baseline_rmse,
        "relative_holdout_change": (holdout_rmse / baseline_rmse - 1.0) if baseline_rmse else None,
        "holdout_rmse_by_restart": holdout_scores, "restarts": restarts, "epochs_budget": epochs,
        "selected_restart": selected_index, "input_dim": int(features.shape[1]), "hidden_dim": int(hidden_dim),
        "network_state": selected_state, "feature_center": f_mean.astype(float).tolist(),
        "feature_scale": f_scale.astype(float).tolist(),
        "unit_status": unit_status,
        "policy": "bounded_neural_residual_with_temporal_holdout; domain_checked_if_declared; no_physical_interpretation_or_ode_proof",
    }


def predict_neural_ude(correction_features: Sequence[Sequence[float]], fit_result: Mapping[str, Any]) -> list[float]:
    """Apply a previously fitted residual network without retraining."""
    if not isinstance(fit_result, Mapping) or fit_result.get("status") != "fitted_neural":
        raise NeuralUDEError("ude_neural_fit_result_not_reusable")
    state = fit_result.get("network_state")
    center = np.asarray(fit_result.get("feature_center"), dtype=np.float32)
    scale = np.asarray(fit_result.get("feature_scale"), dtype=np.float32)
    features = np.asarray(correction_features, dtype=np.float32)
    if features.ndim != 2 or center.ndim != 1 or scale.shape != center.shape or features.shape[1] != center.size:
        raise NeuralUDEError("ude_neural_prediction_shape_invalid")
    if not np.isfinite(features).all() or not isinstance(state, Mapping):
        raise NeuralUDEError("ude_neural_prediction_values_invalid")
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise NeuralUDEError("torch_not_installed") from exc
    hidden_dim = int(fit_result.get("hidden_dim", 16))
    network = nn.Sequential(nn.Linear(center.size, hidden_dim), nn.Tanh(),
                            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
    try:
        network.load_state_dict({key: torch.tensor(value) for key, value in state.items()})
    except (RuntimeError, TypeError, ValueError) as exc:
        raise NeuralUDEError("ude_neural_saved_state_invalid") from exc
    network.eval()
    with torch.no_grad():
        prediction = network(torch.tensor((features - center) / np.maximum(scale, 1e-6))).squeeze(-1)
    values = prediction.detach().cpu().numpy()
    if not np.isfinite(values).all():
        raise NeuralUDEError("ude_neural_prediction_nonfinite")
    return values.astype(float).tolist()


def simulate_neural_ude_linear(
    times: Sequence[float], initial_state: Sequence[float], linear_matrix: Sequence[Sequence[float]],
    fit_results: Sequence[Mapping[str, Any]], *, state_bounds: Sequence[Sequence[float]] | None = None,
    max_abs_state: float = 1e6,
) -> dict[str, Any]:
    """Integrate a small closed-loop UDE with a linear known mechanism.

    Each fitted residual network supplies one component of
    ``dx/dt = A x + g_i(x)``.  RK4 is used on a regular time grid and every
    stage is domain checked.  The function is intentionally explicit about the
    supported known mechanism; it does not pretend that an arbitrary callable
    or an unvalidated neural RHS is safe to integrate.
    """
    t = np.asarray(times, dtype=float)
    x0 = np.asarray(initial_state, dtype=float)
    matrix = np.asarray(linear_matrix, dtype=float)
    if t.ndim != 1 or t.size < 3 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise NeuralUDEError("ude_simulation_times_invalid")
    if x0.ndim != 1 or not np.isfinite(x0).all() or matrix.shape != (x0.size, x0.size) or not np.isfinite(matrix).all():
        raise NeuralUDEError("ude_simulation_state_or_matrix_invalid")
    if x0.size > 8 or t.size > 2048:
        raise NeuralUDEError("ude_simulation_budget_exceeded")
    if not isinstance(fit_results, Sequence) or len(fit_results) != x0.size:
        raise NeuralUDEError("ude_simulation_fit_count_invalid")
    if type(max_abs_state) not in (int, float) or not math.isfinite(float(max_abs_state)) or max_abs_state <= 0:
        raise NeuralUDEError("ude_simulation_state_bound_invalid")
    bounds = None
    if state_bounds is not None:
        if not isinstance(state_bounds, Sequence) or len(state_bounds) != x0.size:
            raise NeuralUDEError("ude_simulation_state_bounds_invalid")
        bounds = []
        for pair in state_bounds:
            if not isinstance(pair, Sequence) or len(pair) != 2 or not all(math.isfinite(float(v)) for v in pair) or float(pair[0]) >= float(pair[1]):
                raise NeuralUDEError("ude_simulation_state_bounds_invalid")
            bounds.append((float(pair[0]), float(pair[1])))

    def rhs(state: np.ndarray) -> np.ndarray:
        if not np.isfinite(state).all() or np.max(np.abs(state)) > float(max_abs_state):
            raise NeuralUDEError("ude_simulation_state_domain_exceeded")
        if bounds is not None and any(value < low or value > high for value, (low, high) in zip(state, bounds)):
            raise NeuralUDEError("ude_simulation_state_bounds_exceeded")
        features = state.reshape(1, -1).astype(np.float32).tolist()
        correction = np.asarray([float(predict_neural_ude(features, fit)[0]) for fit in fit_results], dtype=float)
        result = matrix @ state + correction
        if not np.isfinite(result).all():
            raise NeuralUDEError("ude_simulation_rhs_nonfinite")
        return result

    trajectory = np.zeros((t.size, x0.size), dtype=float)
    trajectory[0] = x0
    for index, dt in enumerate(np.diff(t), start=1):
        previous = trajectory[index - 1]
        h = float(dt)
        k1 = rhs(previous)
        k2 = rhs(previous + 0.5 * h * k1)
        k3 = rhs(previous + 0.5 * h * k2)
        k4 = rhs(previous + h * k3)
        trajectory[index] = previous + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return {"schema_version": "mathmodel.ude-neural-simulation/v1", "status": "executed",
            "times": t.tolist(), "trajectory": trajectory.tolist(), "state_dimension": int(x0.size),
            "integrator": "rk4", "known_mechanism": "linear_Ax",
            "policy": "bounded_closed_loop_neural_residual;_not_a_physical_identifiability_proof"}


def fit_joint_neural_ude(
    times: Sequence[float], observations: Sequence[Sequence[float]],
    known_matrix: Sequence[Sequence[float]], *, validation_fraction: float = 0.25,
    epochs: int = 120, hidden_dim: int = 12, restarts: int = 2,
    learning_rate: float = 0.01, max_rows: int = 512,
    max_abs_state: float = 1e6, random_state: int = 0,
) -> dict[str, Any]:
    """Jointly fit a bounded linear mechanism and neural residual on trajectories.

    The train segment is rolled out from the observed initial state.  The
    learned matrix and residual network are then rolled from that same initial
    state across the untouched suffix, so the reported validation error is a
    trajectory error rather than a row-wise RHS fit.  This remains a small
    Euler UDE backend: no arbitrary callable, controls, stiffness handling or
    physical conservation law is inferred.
    """
    t = np.asarray(times, dtype=np.float32)
    y = np.asarray(observations, dtype=np.float32)
    matrix0 = np.asarray(known_matrix, dtype=np.float32)
    if (t.ndim != 1 or y.ndim != 2 or t.size < 32 or t.size > max_rows
            or y.shape[0] != t.size or y.shape[1] < 1 or y.shape[1] > 8
            or matrix0.shape != (y.shape[1], y.shape[1])
            or not np.isfinite(t).all() or not np.isfinite(y).all() or not np.isfinite(matrix0).all()
            or np.any(np.diff(t) <= 0)):
        raise NeuralUDEError("ude_joint_shapes_or_values_invalid")
    if not 0.1 <= float(validation_fraction) <= 0.4:
        raise NeuralUDEError("ude_joint_validation_fraction_invalid")
    if type(epochs) is not int or not 20 <= epochs <= 500 or type(hidden_dim) is not int or not 4 <= hidden_dim <= 32:
        raise NeuralUDEError("ude_joint_budget_invalid")
    if type(restarts) is not int or not 1 <= restarts <= 3 or not 0 < float(learning_rate) <= 0.2:
        raise NeuralUDEError("ude_joint_options_invalid")
    if type(max_abs_state) not in (int, float) or not math.isfinite(float(max_abs_state)) or max_abs_state <= 0:
        raise NeuralUDEError("ude_joint_state_bound_invalid")
    split = max(16, min(t.size - 8, int(math.floor(t.size * (1.0 - float(validation_fraction))))))
    if split < 16 or t.size - split < 8:
        raise NeuralUDEError("ude_joint_holdout_too_small")
    try:
        import torch
        from torch import nn
    except ImportError:
        return {"schema_version": "mathmodel.ude-joint-fit/v1", "status": "unavailable", "reason": "torch_not_installed"}
    tx, ty, ttorch = torch.tensor(t), torch.tensor(y), torch.tensor(t)
    x0 = ty[0]
    baseline = np.zeros_like(y, dtype=np.float32); baseline[0] = y[0]
    for i in range(1, t.size):
        baseline[i] = baseline[i - 1] + float(t[i] - t[i - 1]) * (matrix0 @ baseline[i - 1])
        if not np.isfinite(baseline[i]).all() or np.max(np.abs(baseline[i])) > max_abs_state:
            return {"schema_version": "mathmodel.ude-joint-fit/v1", "status": "rejected_nonfinite_baseline"}
    baseline_rmse = float(np.sqrt(np.mean((baseline[split:] - y[split:]) ** 2)))
    selected = None; selected_score = float("inf"); train_scores: list[float] = []; valid_scores: list[float] = []

    def rollout(matrix, network, stop: int):
        state = x0
        out = [state]
        for i in range(1, stop):
            dt = ttorch[i] - ttorch[i - 1]
            rhs = matrix @ state + network(state.unsqueeze(0)).squeeze(0)
            state = state + dt * rhs
            out.append(state)
        return torch.stack(out)

    for restart in range(restarts):
        torch.manual_seed(int(random_state + restart))
        matrix = nn.Parameter(torch.tensor(matrix0.copy()))
        network = nn.Sequential(nn.Linear(y.shape[1], hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, y.shape[1]))
        optimizer = torch.optim.Adam([matrix, *network.parameters()], lr=float(learning_rate), weight_decay=1e-4)
        best_state = None; best_valid = float("inf"); stale = 0
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            predicted_train = rollout(matrix, network, split)
            loss = torch.mean((predicted_train - ty[:split]) ** 2)
            if not torch.isfinite(loss):
                break
            loss.backward(); torch.nn.utils.clip_grad_norm_([matrix, *network.parameters()], 5.0); optimizer.step()
            with torch.no_grad():
                predicted_full = rollout(matrix, network, t.size)
                valid_loss = float(torch.mean((predicted_full[split:] - ty[split:]) ** 2))
            if math.isfinite(valid_loss) and valid_loss + 1e-8 < best_valid:
                best_valid = valid_loss
                best_state = {"matrix": matrix.detach().clone(), "network": {k: v.detach().clone() for k, v in network.state_dict().items()}}
                stale = 0
            else:
                stale += 1
                if stale >= 20:
                    break
        if best_state is None:
            continue
        network.load_state_dict(best_state["network"])
        matrix_value = best_state["matrix"]
        # Re-evaluate with the saved matrix without mutating optimizer state.
        with torch.no_grad():
            pred_train = rollout(matrix_value, network, split)
            pred_full = rollout(matrix_value, network, t.size)
            train_rmse = float(torch.sqrt(torch.mean((pred_train - ty[:split]) ** 2)))
            valid_rmse = float(torch.sqrt(torch.mean((pred_full[split:] - ty[split:]) ** 2)))
        train_scores.append(train_rmse); valid_scores.append(valid_rmse)
        if valid_rmse < selected_score:
            selected_score = valid_rmse
            selected = {"matrix": matrix_value.detach().cpu().numpy().tolist(),
                        "network": {k: v.detach().cpu().numpy().tolist() for k, v in best_state["network"].items()}}
    if selected is None or not math.isfinite(selected_score):
        return {"schema_version": "mathmodel.ude-joint-fit/v1", "status": "rejected_training_failed",
                "baseline_holdout_rmse": baseline_rmse}
    return {
        "schema_version": "mathmodel.ude-joint-fit/v1", "status": "fitted_joint_ude",
        "train_rows": int(split), "holdout_rows": int(t.size - split),
        "train_rmse": float(min(train_scores)), "holdout_rmse": float(selected_score),
        "baseline_holdout_rmse": baseline_rmse,
        "relative_holdout_change": (selected_score / baseline_rmse - 1.0) if baseline_rmse else None,
        "restarts": int(restarts), "epochs_budget": int(epochs), "state_dimension": int(y.shape[1]),
        "learned_matrix": selected["matrix"], "network_state": selected["network"],
        "policy": "bounded_joint_linear_parameter_and_neural_residual_trajectory_fit;_euler_only;_not_a_physical_proof",
    }


def simulate_neural_ude_stiff(
    times: Sequence[float], initial_state: Sequence[float], linear_matrix: Sequence[Sequence[float]],
    fit_results: Sequence[Mapping[str, Any]], *, method: str = "BDF",
    rtol: float = 1e-5, atol: float = 1e-7, max_step: float | None = None,
    max_abs_state: float = 1e6,
) -> dict[str, Any]:
    """Integrate the bounded neural UDE with a stiff-aware SciPy method.

    This is an execution backend for the same linear-plus-neural RHS used by
    ``simulate_neural_ude_linear``.  BDF/Radau are selected explicitly by the
    caller; tolerances and state magnitude are bounded to keep a stiff solve
    from becoming an unbounded resource sink.
    """
    if method not in {"BDF", "Radau"}:
        raise NeuralUDEError("ude_stiff_method_invalid")
    if type(rtol) not in (int, float) or not 1e-8 <= float(rtol) <= 1e-2:
        raise NeuralUDEError("ude_stiff_rtol_invalid")
    if type(atol) not in (int, float) or not 1e-10 <= float(atol) <= 1e-2:
        raise NeuralUDEError("ude_stiff_atol_invalid")
    t = np.asarray(times, dtype=float); x0 = np.asarray(initial_state, dtype=float)
    matrix = np.asarray(linear_matrix, dtype=float)
    if t.ndim != 1 or t.size < 3 or t.size > 2048 or np.any(np.diff(t) <= 0) or not np.isfinite(t).all():
        raise NeuralUDEError("ude_stiff_times_invalid")
    if x0.ndim != 1 or x0.size > 8 or not np.isfinite(x0).all() or matrix.shape != (x0.size, x0.size) or not np.isfinite(matrix).all():
        raise NeuralUDEError("ude_stiff_state_or_matrix_invalid")
    if not isinstance(fit_results, Sequence) or len(fit_results) != x0.size:
        raise NeuralUDEError("ude_stiff_fit_count_invalid")
    if type(max_abs_state) not in (int, float) or not math.isfinite(float(max_abs_state)) or max_abs_state <= 0:
        raise NeuralUDEError("ude_stiff_state_bound_invalid")
    if max_step is not None and (type(max_step) not in (int, float) or not math.isfinite(float(max_step)) or max_step <= 0):
        raise NeuralUDEError("ude_stiff_max_step_invalid")
    try:
        from scipy.integrate import solve_ivp
    except ImportError as exc:
        raise NeuralUDEError("scipy_not_installed") from exc
    evaluations = 0
    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        if evaluations > 50_000 or not np.isfinite(state).all() or np.max(np.abs(state)) > float(max_abs_state):
            raise NeuralUDEError("ude_stiff_resource_or_state_guard")
        features = state.reshape(1, -1).tolist()
        correction = np.asarray([float(predict_neural_ude(features, fit)[0]) for fit in fit_results], dtype=float)
        value = matrix @ state + correction
        if not np.isfinite(value).all():
            raise NeuralUDEError("ude_stiff_rhs_nonfinite")
        return value
    kwargs: dict[str, Any] = {"method": method, "rtol": float(rtol), "atol": float(atol)}
    if max_step is not None: kwargs["max_step"] = float(max_step)
    try:
        result = solve_ivp(rhs, (float(t[0]), float(t[-1])), x0, t_eval=t, **kwargs)
    except NeuralUDEError:
        raise
    except Exception as exc:
        raise NeuralUDEError("ude_stiff_integration_failed") from exc
    if not result.success or result.y.shape != (x0.size, t.size) or not np.isfinite(result.y).all():
        raise NeuralUDEError("ude_stiff_integration_failed")
    return {"schema_version": "mathmodel.ude-neural-stiff-simulation/v1", "status": "executed",
            "times": t.tolist(), "trajectory": result.y.T.tolist(), "state_dimension": int(x0.size),
            "integrator": method, "function_evaluations": int(evaluations),
            "solver_tolerances": {"rtol": float(rtol), "atol": float(atol)},
            "policy": "bounded_stiff_aware_neural_ude_execution;_not_a_stiffness_or_physical_proof"}


__all__ = ["SCHEMA_VERSION", "NeuralUDEError", "fit_neural_ude", "predict_neural_ude", "simulate_neural_ude_linear", "fit_joint_neural_ude", "simulate_neural_ude_stiff"]
