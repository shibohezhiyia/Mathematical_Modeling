"""受控低维 ODE 候选与 CEGIS 适配器。

这是统一模型族 CEGIS 的第一个动力学适配器：候选只允许 JSON 多项式 RHS，
数值积分使用 SciPy ``solve_ivp``，不解析生成源码。它能真实执行、记录轨迹
反例并做有界系数修复，但只覆盖低维自治 ODE，不是任意微分方程发现器。
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .cegis_controller import CEGISConfig
from .model_family_adapters import ModelFamilyAdapter, run_model_family_cegis


class ODECEGISError(ValueError):
    pass


_BASIS = ("constant", "linear", "quadratic")


def _validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ODECEGISError("ode_candidate_must_be_object")
    state_dim = candidate.get("state_dim")
    if type(state_dim) is not int or not 1 <= state_dim <= 3:
        raise ODECEGISError("ode_state_dim_invalid")
    basis = candidate.get("basis")
    if type(basis) is not list or not basis or len(basis) > 3 or any(item not in _BASIS for item in basis):
        raise ODECEGISError("ode_basis_invalid")
    if len(set(basis)) != len(basis):
        raise ODECEGISError("ode_basis_duplicate")
    coefficients = np.asarray(candidate.get("coefficients"), dtype=float)
    if coefficients.shape != (state_dim, len(basis)) or not np.isfinite(coefficients).all():
        raise ODECEGISError("ode_coefficients_shape_invalid")
    if np.max(np.abs(coefficients)) > 1e6:
        raise ODECEGISError("ode_coefficients_out_of_bounds")
    return {
        "id": str(candidate.get("id", "ode_candidate"))[:120],
        "state_dim": state_dim, "basis": list(basis),
        "coefficients": coefficients.tolist(),
    }


def _basis_values(state: np.ndarray, basis: Sequence[str]) -> np.ndarray:
    values = []
    for item in basis:
        if item == "constant":
            values.append(1.0)
        elif item == "linear":
            values.append(state)
        elif item == "quadratic":
            values.append(state * state)
    return np.asarray(values, dtype=float)


def compile_ode_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a polynomial RHS candidate."""
    return _validate_candidate(candidate)


def _integrate(compiled: Mapping[str, Any], case: Mapping[str, Any]) -> np.ndarray:
    from scipy.integrate import solve_ivp

    times = np.asarray(case.get("times"), dtype=float)
    initial = np.asarray(case.get("initial"), dtype=float)
    if times.ndim != 1 or times.size < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ODECEGISError("ode_times_invalid")
    if initial.shape != (compiled["state_dim"],) or not np.isfinite(initial).all():
        raise ODECEGISError("ode_initial_invalid")
    coefficients = np.asarray(compiled["coefficients"], dtype=float)
    basis = compiled["basis"]
    evaluations = 0

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        if evaluations > 10_000 or not np.isfinite(state).all() or np.max(np.abs(state)) > 1e6:
            raise RuntimeError("ode_numerical_guard_exceeded")
        with np.errstate(over="raise", invalid="raise"):
            return coefficients @ _basis_values(state, basis)

    result = solve_ivp(rhs, (float(times[0]), float(times[-1])), initial,
                       t_eval=times, method="RK45", rtol=1e-6, atol=1e-8)
    if not result.success or result.y.shape != (compiled["state_dim"], times.size):
        raise ODECEGISError("ode_integration_failed")
    trajectory = result.y.T
    if not np.isfinite(trajectory).all():
        raise ODECEGISError("ode_non_finite_trajectory")
    return trajectory


def evaluate_ode_candidate(compiled: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-2) -> dict[str, Any]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases or len(cases) > 128:
        raise ODECEGISError("ode_cases_invalid")
    if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ODECEGISError("ode_tolerance_invalid")
    violations = []
    errors = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or "observations" not in case:
            return {"status": "not_assessed", "failure_code": "ode_case_contract_invalid", "violations": [], "cost_units": index + 1}
        expected = np.asarray(case["observations"], dtype=float)
        if expected.ndim != 2 or expected.shape[1] != compiled["state_dim"] or not np.isfinite(expected).all():
            return {"status": "not_assessed", "failure_code": "ode_observations_invalid", "violations": [], "cost_units": index + 1}
        try:
            predicted = _integrate(compiled, case)
        except Exception as exc:
            return {"status": "not_assessed", "failure_code": str(exc)[:100], "violations": [], "cost_units": index + 1}
        if predicted.shape != expected.shape:
            return {"status": "not_assessed", "failure_code": "ode_trajectory_shape_mismatch", "violations": [], "cost_units": index + 1}
        absolute = np.abs(predicted - expected)
        errors.extend(absolute.reshape(-1).tolist())
        worst = float(np.max(absolute))
        if worst > float(tolerance):
            violations.append({"reason": "ode_trajectory_error_exceeds_tolerance",
                               "witness_id": str(case.get("id", f"case_{index}"))[:80],
                               "absolute_error": worst})
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    return {"status": "pass" if not violations else "fail", "score": rmse,
            "violations": violations[:16], "cost_units": len(cases),
            "policy": "bounded_ode_trajectory_evaluation;_not_a_dynamics_proof"}


def diagnose_ode_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(feedback, Mapping):
        return {"step": 0.0}
    raw_count = feedback.get("counterexample_archive_count", 0)
    count = int(raw_count) if type(raw_count) is int and raw_count >= 0 else 0
    return {"step": 0.1, "failure_code": str(feedback.get("failure_code", ""))[:100],
            "counterexample_archive_count": count}


def patch_ode_coefficients(candidate: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    normalized = _validate_candidate(candidate)
    step = float(diagnostic.get("step", 0.1)) if isinstance(diagnostic, Mapping) else 0.1
    step = min(max(abs(step), 1e-4), 1.0)
    coefficients = np.asarray(normalized["coefficients"], dtype=float)
    # Explore one coefficient at a time to keep the mutation interpretable.
    for row in range(coefficients.shape[0]):
        for col in range(coefficients.shape[1]):
            for delta, label in ((-step, "down"), (step, "up")):
                revised = dict(normalized)
                revised_coefficients = coefficients.copy()
                revised_coefficients[row, col] += delta
                revised["coefficients"] = revised_coefficients.tolist()
                revised["id"] = f"{normalized['id']}_r{row}_{col}_{label}"
                yield revised
    # Structure mutations are deliberately sparse and typed: add one missing
    # basis with zero coefficients, then let the same bounded coefficient
    # search tune it.  This lets CEGIS leave a bad initial function family
    # without allowing arbitrary source/code generation.  They are yielded
    # after local coefficient repairs so existing low-cost searches retain
    # their convergence behavior.
    # Keep local coefficient repair the first-line search.  A structural jump
    # is opened only after repeated witnesses; otherwise a broad basis branch
    # can crowd a nearby coefficient solution out of a small CEGIS budget.
    witness_count = int(diagnostic.get("counterexample_archive_count", 0)) if isinstance(diagnostic, Mapping) else 0
    if witness_count < 8:
        return
    for basis in _BASIS:
        if basis in normalized["basis"] or len(normalized["basis"]) >= 3:
            continue
        revised = dict(normalized)
        revised_basis = list(normalized["basis"]) + [basis]
        revised["basis"] = revised_basis
        revised["coefficients"] = np.column_stack(
            [coefficients, np.zeros(normalized["state_dim"], dtype=float)]
        ).tolist()
        revised["id"] = f"{normalized['id']}_add_{basis}"
        yield revised


def build_ode_cegis_adapter(*, tolerance: float = 1e-2) -> ModelFamilyAdapter:
    def evaluate(compiled: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        return evaluate_ode_candidate(compiled, cases, tolerance=tolerance)

    return ModelFamilyAdapter(
        family="ode",
        compile=compile_ode_candidate,
        evaluate=evaluate,
        diagnose=diagnose_ode_feedback,
        patch=patch_ode_coefficients,
        replay=lambda compiled, witnesses: evaluate_ode_candidate(compiled, witnesses, tolerance=tolerance),
    )


def run_ode_cegis(initial_candidates: Iterable[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], *, config: CEGISConfig | None = None, tolerance: float = 1e-2) -> dict[str, Any]:
    return run_model_family_cegis(build_ode_cegis_adapter(tolerance=tolerance), initial_candidates, cases, config=config)


__all__ = ["ODECEGISError", "build_ode_cegis_adapter", "compile_ode_candidate",
           "evaluate_ode_candidate", "run_ode_cegis"]
