"""Bounded sparse ODE discovery with train/search/test temporal isolation.

Integral consistency uses observed trajectories; autonomous rollout does not.
Neither test proves that an observationally fitted equation is the real mechanism.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from time import monotonic
from typing import Any, Optional, Sequence
import warnings

import numpy as np
import pandas as pd


@dataclass
class PreparedDynamics:
    state_columns: list[str]
    center: np.ndarray
    scale: np.ndarray
    states: np.ndarray
    observed_states: np.ndarray
    elapsed: np.ndarray
    timestamps: pd.Series
    train_end: int
    search_end: int
    window: int
    starts: np.ndarray
    design: np.ndarray
    response: np.ndarray
    train_mask: np.ndarray
    search_mask: np.ndarray
    test_mask: np.ndarray
    term_names: list[str]
    training_fingerprint: str


def _library(states: np.ndarray) -> np.ndarray:
    columns = [np.ones(states.shape[:-1])]
    columns.extend(states[..., index] for index in range(states.shape[-1]))
    columns.extend(states[..., index] ** 2 for index in range(states.shape[-1]))
    columns.extend(states[..., left] * states[..., right]
                   for left, right in combinations(range(states.shape[-1]), 2))
    return np.stack(columns, axis=-1)


def prepare_dynamics(
    source: pd.DataFrame, *, time_column: str, target_column: str,
    candidate_columns: Optional[Sequence[str]] = None,
) -> Optional[PreparedDynamics]:
    """Split the timestamp axis before any value-based selection or imputation."""
    if time_column == target_column or target_column not in source or time_column not in source:
        return None
    # Column order is a schema choice, not a ranking learned from the test period.
    candidates = list(candidate_columns) if candidate_columns is not None else [
        column for column in source if column != time_column
        and pd.api.types.is_numeric_dtype(source[column])
        and not pd.api.types.is_bool_dtype(source[column])
    ]
    columns = list(dict.fromkeys([target_column, *candidates]))[:8]
    columns = [column for column in columns if column in source and column != time_column]
    frame = source[[time_column, *columns]].copy()
    frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Missing targets must NOT remove timestamps or shift the train/test boundary.
    frame = (frame.dropna(subset=[time_column])
             .groupby(time_column, as_index=False)[columns].mean()
             .sort_values(time_column).reset_index(drop=True))
    if len(frame) > 5000:
        frame = frame.iloc[np.linspace(0, len(frame) - 1, 5000, dtype=int)].reset_index(drop=True)
    if len(frame) < 50:
        return None
    train_end, search_end = int(len(frame) * 0.6), int(len(frame) * 0.8)
    training = frame.iloc[:train_end]
    available = [column for column in columns
                 if training[column].notna().sum() >= max(12, int(np.ceil(train_end * 0.8)))
                 and training[column].nunique(dropna=True) > 2]
    if target_column not in available:
        return None
    correlations = training[available].corr(method="spearman")[target_column].abs().fillna(0)
    ranked = correlations.sort_values(ascending=False, kind="stable").index
    state_columns = [target_column, *[column for column in ranked if column != target_column][:3]]
    values = frame[state_columns].to_numpy(dtype=float)
    center = np.nanmedian(values[:train_end], axis=0)
    mad = np.nanmedian(np.abs(values[:train_end] - center), axis=0)
    standard_deviation = np.nanstd(values[:train_end], axis=0)
    scale = np.where(1.4826 * mad > 1e-12, 1.4826 * mad,
                     np.where(standard_deviation > 1e-12, standard_deviation, 1.0))
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        return None
    # Forward fill is causal. Initial missing values remain missing; observed labels
    # below never use the filled values as if they were measurements.
    with np.errstate(over="ignore", invalid="ignore"):
        states = (frame[state_columns].ffill().to_numpy(dtype=float) - center) / scale
        observed_states = (values - center) / scale
        library = _library(states)
    elapsed = (frame[time_column] - frame[time_column].iloc[0]).dt.total_seconds().to_numpy() / 86400
    if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) <= 0):
        return None
    window = max(3, min(12, len(frame) // 30))
    starts = np.arange(len(frame) - window)
    with np.errstate(over="ignore", invalid="ignore"):
        increments = 0.5 * (library[:-1] + library[1:]) * np.diff(elapsed)[:, None]
        finite_edges = np.isfinite(increments).all(axis=1)
        # Cumulative quadrature is O(rows * terms), rather than recomputing every
        # overlapping window. Invalid edges are counted separately, never accepted.
        prefix = np.vstack([np.zeros(library.shape[1]),
                            np.cumsum(np.where(np.isfinite(increments), increments, 0), axis=0)])
        design = prefix[starts + window] - prefix[starts]
        response = observed_states[starts + window] - observed_states[starts]
    invalid_prefix = np.r_[0, np.cumsum(~finite_edges)]
    valid = ((invalid_prefix[starts + window] == invalid_prefix[starts])
             & np.isfinite(design).all(axis=1) & np.isfinite(response).all(axis=1))
    train_mask = valid & (starts + window < train_end)
    search_mask = valid & (starts >= train_end) & (starts + window < search_end)
    test_mask = valid & (starts >= search_end)
    if train_mask.sum() < 20 or search_mask.sum() < 6:
        return None
    term_names = ["1", *[f"z({column})" for column in state_columns],
                  *[f"z({column})²" for column in state_columns],
                  *[f"z({left})·z({right})" for left, right in combinations(state_columns, 2)]]
    fingerprint = sha256(pd.util.hash_pandas_object(training, index=False).values.tobytes())
    fingerprint.update("\0".join(map(str, training.columns)).encode("utf-8"))
    return PreparedDynamics(
        state_columns, center, scale, states, observed_states, elapsed, frame[time_column],
        train_end, search_end, window, starts, design, response, train_mask, search_mask,
        test_mask, term_names, fingerprint.hexdigest(),
    )


def _metrics(actual: np.ndarray, predicted: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    valid = np.isfinite(actual) & np.isfinite(predicted) & np.isfinite(baseline)
    count = int(valid.sum())
    result = {"n": count, "rmse": None, "baseline_rmse": None, "r2": None,
              "invalid_predictions": int((np.isfinite(actual) & ~np.isfinite(predicted)).sum())}
    if count < 2:
        return result
    actual, predicted, baseline = actual[valid], predicted[valid], baseline[valid]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mse = float(np.mean((actual - predicted) ** 2))
        baseline_mse = float(np.mean((actual - baseline) ** 2))
        variance = float(np.var(actual))
        result.update(rmse=float(np.sqrt(mse)), baseline_rmse=float(np.sqrt(baseline_mse)),
                      r2=1 - mse / variance if variance > 1e-24 else None)
    return {key: (None if isinstance(value, float) and not np.isfinite(value) else value)
            for key, value in result.items()}


def _metric_status(metrics: dict[str, Any]) -> str:
    if metrics.get("invalid_predictions", 0):
        return "fail"
    if metrics["n"] < 6:
        return "not_assessed"
    if metrics["rmse"] is None or metrics["baseline_rmse"] is None:
        return "fail"
    if metrics["baseline_rmse"] <= 1e-12 or metrics["r2"] is None:
        return "warning"
    if metrics["rmse"] < 0.9 * metrics["baseline_rmse"] and metrics["r2"] >= 0.25:
        return "pass"
    return "warning" if metrics["rmse"] < metrics["baseline_rmse"] else "fail"


def _fit_system(
    design_train: np.ndarray, response_train: np.ndarray,
    design_search: np.ndarray, response_search: np.ndarray, random_state: int,
) -> Optional[dict[str, Any]]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso
    from threadpoolctl import threadpool_limits

    feature_scale = np.std(design_train, axis=0)
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    # The fitting interface does not receive any final-test arrays or metadata.
    x_train = design_train / feature_scale
    x_search = design_search / feature_scale
    coefficients, searches, selected_alphas = [], [], []
    with threadpool_limits(limits=1):
        for state_index in range(response_train.shape[1]):
            y_train = response_train[:, state_index]
            y_search = response_search[:, state_index]
            response_scale = max(float(np.std(y_train)), 1e-12)
            candidates, best = [], None
            for alpha in response_scale * np.logspace(-4, -0.7, 16):
                model = Lasso(alpha=float(alpha), fit_intercept=False, max_iter=20000,
                              random_state=random_state)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    try:
                        model.fit(x_train, y_train)
                    except (ConvergenceWarning, ValueError, FloatingPointError):
                        continue
                prediction = model.predict(x_search)
                rmse = float(np.sqrt(np.mean((y_search - prediction) ** 2)))
                nonzero = int(np.count_nonzero(np.abs(model.coef_) > 1e-8))
                objective = rmse / response_scale + 0.005 * nonzero
                if not np.isfinite(objective) or not np.isfinite(model.coef_).all():
                    continue
                candidates.append({"alpha": float(alpha), "validation_rmse": rmse,
                                   "nonzero_terms": nonzero, "selection_objective": objective})
                if best is None or objective < best[0]:
                    best = objective, model.coef_.copy() / feature_scale, float(alpha)
            if best is None:
                return None
            coefficients.append(best[1])
            selected_alphas.append(best[2])
            searches.append(candidates)
    return {"coefficients": np.asarray(coefficients), "searches": searches,
            "selected_alphas": selected_alphas, "feature_scale": feature_scale}


def _rollout(
    times: np.ndarray, initial: np.ndarray, coefficients: np.ndarray, *,
    max_evaluations: int, timeout_seconds: float,
) -> tuple[Optional[np.ndarray], dict[str, Any]]:
    """This evaluator has no access to future observed states."""
    from scipy.integrate import solve_ivp

    started, evaluations = monotonic(), 0

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        if evaluations > max_evaluations:
            raise RuntimeError("evaluation_budget_exhausted")
        if monotonic() - started > timeout_seconds:
            raise RuntimeError("rollout_deadline_exceeded")
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e6:
            raise RuntimeError("numerical_guard_exceeded")
        with np.errstate(over="raise", invalid="raise"):
            return coefficients @ _library(state)

    try:
        solution = solve_ivp(rhs, (float(times[0]), float(times[-1])), initial,
                             t_eval=times, rtol=1e-6, atol=1e-8)
        if not solution.success or solution.y.shape[1] != len(times) or not np.isfinite(solution.y).all():
            raise RuntimeError("integration_failed")
        return solution.y.T, {"solver_success": True, "nfev": evaluations,
                              "solver": "RK45", "rtol": 1e-6, "atol": 1e-8}
    except (RuntimeError, FloatingPointError, ValueError, OverflowError) as exc:
        return None, {"solver_success": False, "nfev": evaluations,
                      "reason": str(exc) if isinstance(exc, RuntimeError) else "numerical_failure"}


def _support_stability(prepared: PreparedDynamics, fitted: dict[str, Any], random_state: int) -> Optional[float]:
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import Lasso

    midpoint = prepared.train_end // 2
    masks = [prepared.train_mask & (prepared.starts + prepared.window < midpoint),
             prepared.train_mask & (prepared.starts >= midpoint)]
    supports = []
    for mask in masks:
        if mask.sum() < 15:
            return None
        model = Lasso(alpha=fitted["selected_alphas"][0], fit_intercept=False,
                      max_iter=20000, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            try:
                model.fit(prepared.design[mask] / fitted["feature_scale"], prepared.response[mask, 0])
            except (ConvergenceWarning, ValueError, FloatingPointError):
                return None
        supports.append(set(np.flatnonzero(np.abs(model.coef_) > 1e-8)))
    union = supports[0] | supports[1]
    return len(supports[0] & supports[1]) / len(union) if union else 1.0


def discover_integral_dynamics(
    source: pd.DataFrame, *, time_column: str, target_column: str,
    candidate_columns: Optional[Sequence[str]] = None, random_state: int = 42,
    max_rollout_evaluations: int = 5000, rollout_timeout_seconds: float = 3.0,
) -> Optional[dict[str, Any]]:
    prepared = prepare_dynamics(source, time_column=time_column, target_column=target_column,
                                candidate_columns=candidate_columns)
    if prepared is None:
        return None
    fitted = _fit_system(
        prepared.design[prepared.train_mask], prepared.response[prepared.train_mask],
        prepared.design[prepared.search_mask], prepared.response[prepared.search_mask], random_state,
    )
    if fitted is None:
        return None
    coefficients = fitted["coefficients"]
    support = _support_stability(prepared, fitted, random_state)
    # Feedback is compiled from development arrays before the final audit.
    # Its interface cannot inspect test labels, residuals or rollout outcomes.
    from .model_diagnostics import DEVELOPMENT_VERSION, diagnose_development
    try:
        development_diagnostics = diagnose_development(
            design_train=prepared.design[prepared.train_mask], response_train=prepared.response[prepared.train_mask],
            design_search=prepared.design[prepared.search_mask], response_search=prepared.response[prepared.search_mask],
            coefficients=coefficients, search_starts=prepared.starts[prepared.search_mask],
            window_points=prepared.window, training_fingerprint=prepared.training_fingerprint,
            term_names=prepared.term_names, state_names=prepared.state_columns,
            seed=random_state, support_jaccard=support,
        )
    except Exception:
        # Diagnosis is optional; its failure must not erase a computed model.
        development_diagnostics = {"schema_version": DEVELOPMENT_VERSION, "status": "failed_safe",
                                   "error_code": "development_diagnostics_unavailable", "records": [], "checks": {}}
    actual = prepared.response[prepared.test_mask, 0]
    prediction = prepared.design[prepared.test_mask] @ coefficients[0]
    integral_metrics = _metrics(actual, prediction, np.zeros_like(actual))
    integral_status = _metric_status(integral_metrics)
    # Use disjoint windows for residual-memory diagnostics: overlap itself causes
    # serial correlation even when measurement errors are independent.
    test_starts = prepared.starts[prepared.test_mask]
    disjoint = (test_starts - prepared.search_end) % (prepared.window + 1) == 0
    residual = actual[disjoint] - prediction[disjoint]
    autocorrelation = None
    if len(residual) >= 6 and np.std(residual[:-1]) > 1e-12 and np.std(residual[1:]) > 1e-12:
        value = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
        autocorrelation = value if np.isfinite(value) else None
    initial_row = prepared.search_end - 1
    initial = prepared.observed_states[initial_row]
    trajectory: dict[str, Any] = {
        "status": "not_assessed", "solver_success": False, "initial_row": initial_row,
        "uses_future_observations": False, "kind": "autonomous_joint_state_rollout",
        "metrics": {}, "limits": {"max_evaluations": max_rollout_evaluations,
                                    "timeout_seconds": rollout_timeout_seconds},
    }
    rollout_actual = prepared.observed_states[prepared.search_end:, 0] * prepared.scale[0] + prepared.center[0]
    rollout_prediction = np.full(len(rollout_actual), np.nan)
    if not np.isfinite(initial).all():
        trajectory["reason"] = "initial_state_not_observed"
    else:
        simulated, solver = _rollout(prepared.elapsed[initial_row:], initial, coefficients,
                                     max_evaluations=max_rollout_evaluations,
                                     timeout_seconds=rollout_timeout_seconds)
        trajectory.update(solver)
        if simulated is None:
            trajectory["status"] = "fail"
        else:
            simulated = simulated[1:]
            rollout_prediction = simulated[:, 0] * prepared.scale[0] + prepared.center[0]
            baseline = np.full(len(rollout_actual), initial[0] * prepared.scale[0] + prepared.center[0])
            trajectory["metrics"] = _metrics(rollout_actual, rollout_prediction, baseline)
            trajectory["status"] = _metric_status(trajectory["metrics"])
            per_state = {}
            for index, column in enumerate(prepared.state_columns):
                per_state[column] = _metrics(prepared.observed_states[prepared.search_end:, index],
                                             simulated[:, index], np.full(len(simulated), initial[index]))
            trajectory["state_metrics_standardized"] = per_state
            statuses = [_metric_status(item) for item in per_state.values()]
            if trajectory["status"] != "not_assessed":
                if "fail" in statuses:
                    trajectory["status"] = "fail"
                elif any(status != "pass" for status in statuses):
                    trajectory["status"] = "warning"
            if trajectory["metrics"]["n"] < 6:
                trajectory["reason"] = "insufficient_observed_test_targets"

    def check(key: str, name: str, status: str, evidence: str, recommendation: str = "") -> dict[str, Any]:
        return dict(id=key, name=name, status=status, evidence=evidence,
                    recommendation=recommendation, details={})

    checks = [
        check("dynamics_integral_test", "锁定测试段积分一致性", integral_status,
              f"使用观测轨迹构造积分项，指标={integral_metrics}；不是独立轨迹预测。"),
        check("dynamics_autonomous_test", "锁定测试段独立轨迹预测", trajectory["status"],
              f"联合状态方程从测试前最后一个观测状态出发，不读取未来状态；结果={trajectory.get('reason', trajectory['metrics'])}。",
              "观测拟合不能保证状态闭合或真实机理；检查外部输入、观测误差、结构与数值条件。"),
        check("equation_support_stability", "训练段项集敏感性",
              "not_assessed" if support is None else ("pass" if support >= 0.7 else "warning"),
              f"固定训练预处理，前后不重叠子段重拟合的项集 Jaccard={support}；不是独立测试。"),
        check("dynamics_residual_memory", "非重叠窗口残差记忆",
              "not_assessed" if autocorrelation is None else ("pass" if abs(autocorrelation) < 0.3 else "warning"),
              f"非重叠测试窗口数={len(residual)}，一阶自相关={autocorrelation}。",
              "残差相关可能来自噪声、数值误差、观测过程或机制不足，不能直接判定存在隐变量。"),
    ]
    status = "fail" if any(item["status"] == "fail" for item in checks) else (
        "pass" if all(item["status"] == "pass" for item in checks) else "warning")
    equations, active_terms = [], []
    for column, row in zip(prepared.state_columns, coefficients):
        active = [{"term": term, "coefficient": float(value), "absolute_coefficient": float(abs(value))}
                  for term, value in zip(prepared.term_names, row) if abs(value) > 1e-8]
        active.sort(key=lambda item: item["absolute_coefficient"], reverse=True)
        expression = " + ".join(f"{item['coefficient']:.5g}·{item['term']}" for item in active) or "0"
        equations.append({"state": column, "equation": f"d z({column}) / d day = {expression}",
                          "active_terms": active})
        active_terms.append(active)
    split = {}
    for name, start, stop in (("train", 0, prepared.train_end),
                              ("search", prepared.train_end, prepared.search_end),
                              ("test", prepared.search_end, len(prepared.states))):
        split[name] = {"start": start, "stop_exclusive": stop,
                       "start_time": prepared.timestamps.iloc[start].isoformat(),
                       "end_time": prepared.timestamps.iloc[stop - 1].isoformat()}
    return {
        "schema_version": "mathmodel.integral-dynamics/v2",
        "method": "derivative_free_integral_sparse_dynamics",
        "evaluation_protocol": "train_search_locked_test", "split": split,
        "time_column": time_column, "target": target_column,
        "state_columns": prepared.state_columns, "system_equations": equations,
        "equation": equations[0]["equation"], "active_terms": active_terms[0],
        "standardization": {column: {"center": float(center), "scale": float(scale)}
                            for column, center, scale in zip(prepared.state_columns, prepared.center, prepared.scale)},
        "preprocessing": {"fit_partition": "train", "feature_imputation": "causal_forward_fill",
                          "labels": "observed_only", "max_candidate_columns": 8, "max_states": 4},
        "training_fingerprint": prepared.training_fingerprint,
        "development_diagnostics": development_diagnostics,
        "window_points": prepared.window, "n_time_points": len(prepared.states),
        "training_windows": int(prepared.train_mask.sum()),
        "selection_windows": int(prepared.search_mask.sum()),
        "test_windows": int(prepared.test_mask.sum()),
        "validation_windows": int(prepared.test_mask.sum()),
        "selected_alpha": fitted["selected_alphas"][0],
        "candidate_search": fitted["searches"][0],
        "system_selection": {column: {"alpha": alpha, "candidates": candidates}
                             for column, alpha, candidates in zip(prepared.state_columns, fitted["selected_alphas"], fitted["searches"])},
        "test_integral_metrics": integral_metrics, "trajectory_test": trajectory,
        # Compatibility aliases now refer ONLY to the locked test, not search scores.
        "metrics": {"validation_rmse": integral_metrics["rmse"], "baseline_rmse": integral_metrics["baseline_rmse"],
                    "validation_r2": integral_metrics["r2"], "support_jaccard": support,
                    "residual_autocorrelation": autocorrelation},
        "validation_actual": actual, "validation_prediction": prediction,
        "test_rollout_actual": rollout_actual, "test_rollout_prediction": rollout_prediction,
        "credibility_audit": {
            "status": status, "label": {"pass": "经验检查通过", "warning": "谨慎候选", "fail": "未通过"}[status],
            "checks": checks, "decision": "结果仅是当前时间段内的可反证候选；积分一致性与独立轨迹检查分开报告，不证明真实机理。",
        },
        "literature_basis": {"idea": "weak/integral sparse identification avoids pointwise derivative estimation",
                             "doi": "10.1137/20M1343166"},
        "note": "前60%训练、中20%选参、末20%锁定测试；仅训练段拟合选列与尺度。积分指标使用测试观测，独立轨迹只用测试前初值。"
                "方程基于标准化状态及天尺度；预测改善不证明因果、状态闭合或物理单位正确。",
    }
