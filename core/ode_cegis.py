"""受控低维 ODE 候选与 CEGIS 适配器。

这是统一模型族 CEGIS 的第一个动力学适配器：候选只允许 JSON 多项式 RHS，
数值积分使用 SciPy ``solve_ivp``，不解析生成源码。它能真实执行、记录轨迹
反例并做有界系数修复；目前支持低维跨状态乘积和显式时间驱动，但仍不是
任意微分方程发现器。
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .cegis_controller import CEGISConfig
from .model_family_adapters import ModelFamilyAdapter, run_model_family_cegis


class ODECEGISError(ValueError):
    pass


_BASIS = ("constant", "linear", "quadratic", "cross", "time")
_DRIVER_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
_SOLVERS = ("auto", "RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")
_UNIT_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,15}$")


def _normalize_dimensions(raw: Any, *, code: str) -> dict[str, float]:
    if not isinstance(raw, Mapping) or len(raw) > 16:
        raise ODECEGISError(code)
    result: dict[str, float] = {}
    for name, exponent in raw.items():
        if not isinstance(name, str) or not _UNIT_NAME.fullmatch(name):
            raise ODECEGISError(code)
        if type(exponent) not in (int, float) or not math.isfinite(float(exponent)) or abs(float(exponent)) > 32:
            raise ODECEGISError(code)
        value = float(exponent)
        if abs(value) > 1e-12:
            result[name] = value
    return result


def _add_dimensions(*values: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for value in values:
        for name, exponent in value.items():
            result[name] = result.get(name, 0.0) + float(exponent)
    return {name: exponent for name, exponent in result.items() if abs(exponent) > 1e-12}


def _same_dimensions(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    names = set(left) | set(right)
    return all(abs(float(left.get(name, 0.0)) - float(right.get(name, 0.0))) <= 1e-12 for name in names)


def _validate_unit_contract(
    raw: Any,
    *,
    state_dim: int,
    basis: Sequence[str],
    drivers: Sequence[str],
    delays: Sequence[Mapping[str, Any]],
    coefficients: np.ndarray,
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, Mapping) or set(raw) - {"states", "time", "drivers", "coefficients"}:
        raise ODECEGISError("ode_unit_contract_invalid")
    states_raw = raw.get("states")
    if not isinstance(states_raw, list) or len(states_raw) != state_dim:
        raise ODECEGISError("ode_state_units_invalid")
    states = [_normalize_dimensions(item, code="ode_state_units_invalid") for item in states_raw]
    time = _normalize_dimensions(raw.get("time"), code="ode_time_units_invalid")
    drivers_raw = raw.get("drivers", {})
    if not isinstance(drivers_raw, Mapping) or set(drivers_raw) != set(drivers):
        raise ODECEGISError("ode_driver_units_invalid")
    driver_units = {name: _normalize_dimensions(drivers_raw[name], code="ode_driver_units_invalid") for name in drivers}
    coefficient_units_raw = raw.get("coefficients")
    auto_propagated = coefficient_units_raw is None
    if (not auto_propagated and
            (not isinstance(coefficient_units_raw, list) or len(coefficient_units_raw) != state_dim
             or any(not isinstance(row, list) or len(row) != len(basis) for row in coefficient_units_raw))):
        raise ODECEGISError("ode_coefficient_units_invalid")
    coefficient_units = None if auto_propagated else [
        [_normalize_dimensions(item, code="ode_coefficient_units_invalid") for item in row]
        for row in coefficient_units_raw
    ]
    derivative_units = [_add_dimensions(states[index], {name: -value for name, value in time.items()})
                        for index in range(state_dim)]
    expected: list[list[dict[str, float]]] = []
    for equation in range(state_dim):
        terms = []
        for term in basis:
            if term == "constant":
                feature = {}
            elif term == "linear":
                feature = states[equation]
            elif term == "quadratic":
                feature = _add_dimensions(states[equation], states[equation])
            elif term == "cross":
                pair_units = [_add_dimensions(states[i], states[j])
                              for i in range(state_dim) for j in range(i + 1, state_dim)]
                if not pair_units or any(not _same_dimensions(pair_units[0], item) for item in pair_units[1:]):
                    raise ODECEGISError("ode_cross_units_ambiguous")
                feature = pair_units[0]
            elif term == "time":
                feature = time
            elif term.startswith("driver:"):
                feature = driver_units[term.split(":", 1)[1]]
            elif term.startswith("delay:"):
                delay_index = int(term.split(":", 1)[1])
                feature = states[int(delays[delay_index]["state_index"])]
            else:
                raise ODECEGISError("ode_unit_basis_unknown")
            terms.append(_add_dimensions(derivative_units[equation], {name: -value for name, value in feature.items()}))
        expected.append(terms)
    mismatches = []
    if auto_propagated:
        coefficient_units = expected
    else:
        for equation in range(state_dim):
            for term_index, (actual, wanted) in enumerate(zip(coefficient_units[equation], expected[equation])):
                if not _same_dimensions(actual, wanted):
                    mismatches.append({"equation": equation, "basis": basis[term_index],
                                       "provided": actual, "expected": wanted})
    if mismatches:
        raise ODECEGISError("ode_coefficient_units_mismatch")
    return {"status": "checked", "state_units": states, "time_units": time,
            "driver_units": driver_units, "coefficient_units": coefficient_units,
            "expected_coefficient_units": expected, "mismatches": mismatches,
            "auto_propagated": auto_propagated}


def _validate_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ODECEGISError("ode_candidate_must_be_object")
    state_dim = candidate.get("state_dim")
    if type(state_dim) is not int or not 1 <= state_dim <= 3:
        raise ODECEGISError("ode_state_dim_invalid")
    basis = candidate.get("basis")
    solver = candidate.get("solver", "RK45")
    if not isinstance(solver, str) or solver not in _SOLVERS:
        raise ODECEGISError("ode_solver_invalid")
    drivers = candidate.get("drivers", [])
    if type(drivers) is not list or len(drivers) > 4 or any(
        not isinstance(item, str) or not _DRIVER_NAME.fullmatch(item) for item in drivers
    ) or len(set(drivers)) != len(drivers):
        raise ODECEGISError("ode_drivers_invalid")
    valid_basis = set(_BASIS) | {f"driver:{name}" for name in drivers}
    raw_delays = candidate.get("delays", [])
    if type(raw_delays) is not list or len(raw_delays) > 4:
        raise ODECEGISError("ode_delays_invalid")
    delays = []
    for delay in raw_delays:
        if not isinstance(delay, Mapping):
            raise ODECEGISError("ode_delay_invalid")
        state_index, tau = delay.get("state_index"), delay.get("tau")
        if type(state_index) is not int or not 0 <= state_index < state_dim:
            raise ODECEGISError("ode_delay_state_index_invalid")
        if type(tau) not in (int, float) or not math.isfinite(float(tau)) or not 0 < float(tau) <= 1e4:
            raise ODECEGISError("ode_delay_tau_invalid")
        delays.append({"state_index": state_index, "tau": float(tau)})
    delay_basis = {f"delay:{index}" for index in range(len(delays))}
    valid_basis |= delay_basis
    if (type(basis) is not list or not basis or len(basis) > 5
            or any(not isinstance(item, str) or item not in valid_basis for item in basis)):
        raise ODECEGISError("ode_basis_invalid")
    if len(set(basis)) != len(basis):
        raise ODECEGISError("ode_basis_duplicate")
    if "cross" in basis and state_dim < 2:
        raise ODECEGISError("ode_cross_basis_requires_multiple_states")
    raw_events = candidate.get("events", [])
    if type(raw_events) is not list or len(raw_events) > 4:
        raise ODECEGISError("ode_events_invalid")
    events = []
    for event in raw_events:
        if not isinstance(event, Mapping):
            raise ODECEGISError("ode_event_invalid")
        state_index = event.get("state_index")
        direction = event.get("direction", 0)
        priority = event.get("priority", 0)
        max_occurrences = event.get("max_occurrences", 32)
        requires = event.get("requires", [])
        threshold = event.get("threshold")
        reset_delta = event.get("reset_delta")
        reset_vector = event.get("reset")
        if type(state_index) is not int or not 0 <= state_index < state_dim:
            raise ODECEGISError("ode_event_state_index_invalid")
        if type(direction) is not int or direction not in (-1, 0, 1):
            raise ODECEGISError("ode_event_direction_invalid")
        if type(priority) is not int or not -32 <= priority <= 32:
            raise ODECEGISError("ode_event_priority_invalid")
        if type(max_occurrences) is not int or not 1 <= max_occurrences <= 32:
            raise ODECEGISError("ode_event_max_occurrences_invalid")
        if (not isinstance(requires, list) or len(requires) > 4
                or any(type(item) is not int or not 0 <= item < len(raw_events) or item >= len(events)
                       for item in requires)
                or len(set(requires)) != len(requires)):
            # Dependencies are restricted to earlier events.  This gives the
            # event list a small acyclic network semantics without exposing a
            # general callback or arbitrary state-machine language.
            raise ODECEGISError("ode_event_dependencies_invalid")
        if (type(threshold) not in (int, float) or not math.isfinite(float(threshold))
                or abs(float(threshold)) > 1e6):
            raise ODECEGISError("ode_event_threshold_invalid")
        if reset_vector is not None:
            if (not isinstance(reset_vector, list) or len(reset_vector) != state_dim
                    or any(type(value) not in (int, float) or not math.isfinite(float(value))
                           or abs(float(value)) > 1e6 for value in reset_vector)
                    or not any(float(value) != 0.0 for value in reset_vector)):
                raise ODECEGISError("ode_event_reset_invalid")
            reset_vector = [float(value) for value in reset_vector]
        else:
            if (type(reset_delta) not in (int, float) or not math.isfinite(float(reset_delta))
                    or reset_delta == 0 or abs(float(reset_delta)) > 1e6):
                raise ODECEGISError("ode_event_reset_invalid")
            reset_delta = float(reset_delta)
        events.append({"state_index": state_index, "direction": direction,
                       "priority": priority, "max_occurrences": max_occurrences,
                       "threshold": float(threshold), "reset_delta": reset_delta,
                       **({"reset": reset_vector} if reset_vector is not None else {}),
                       "requires": list(requires)})
    coefficients = np.asarray(candidate.get("coefficients"), dtype=float)
    if coefficients.shape != (state_dim, len(basis)) or not np.isfinite(coefficients).all():
        raise ODECEGISError("ode_coefficients_shape_invalid")
    if np.max(np.abs(coefficients)) > 1e6:
        raise ODECEGISError("ode_coefficients_out_of_bounds")
    unit_audit = _validate_unit_contract(
        candidate.get("units"), state_dim=state_dim, basis=basis,
        drivers=drivers, delays=delays, coefficients=coefficients,
    )
    return {
        "id": str(candidate.get("id", "ode_candidate"))[:120],
        "state_dim": state_dim, "basis": list(basis), "solver": solver,
        "drivers": list(drivers),
        "delays": delays,
        "events": events,
        "coefficients": coefficients.tolist(),
        **({"units": unit_audit} if unit_audit is not None else {}),
    }


def _basis_values(
    state: np.ndarray,
    basis: Sequence[str],
    time: float = 0.0,
    driver_values: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return a bounded feature row for every RHS basis.

    ``linear`` and ``quadratic`` retain their historical per-state semantics.
    The new ``cross`` and ``time`` terms are scalar mechanisms broadcast to all
    state equations; separate coefficients let each equation decide whether to
    use the mechanism.  This keeps the matrix contract stable while making the
    coupling and forcing assumptions explicit and auditable.
    """
    values = []
    for item in basis:
        if item == "constant":
            values.append(np.ones(state.shape, dtype=float))
        elif item == "linear":
            values.append(state)
        elif item == "quadratic":
            values.append(state * state)
        elif item == "cross":
            # Sum of pairwise products is the smallest typed cross-state basis;
            # an individual pair can be added later without changing the
            # execution boundary.  Broadcast so the coefficient row controls
            # which equations receive the coupling.
            cross_value = float(sum(state[i] * state[j]
                                    for i in range(state.size)
                                    for j in range(i + 1, state.size)))
            values.append(np.full(state.shape, cross_value, dtype=float))
        elif item == "time":
            values.append(np.full(state.shape, float(time), dtype=float))
        elif item.startswith("driver:"):
            name = item.split(":", 1)[1]
            if driver_values is None or name not in driver_values:
                raise ODECEGISError("ode_driver_value_missing")
            values.append(np.full(state.shape, float(driver_values[name]), dtype=float))
        elif item.startswith("delay:"):
            index = int(item.split(":", 1)[1])
            if driver_values is None or f"delay:{index}" not in driver_values:
                raise ODECEGISError("ode_delay_value_missing")
            values.append(np.full(state.shape, float(driver_values[f"delay:{index}"]), dtype=float))
    return np.asarray(values, dtype=float)


def compile_ode_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize a polynomial RHS candidate."""
    return _validate_candidate(candidate)


def _integrate_delay(
    compiled: Mapping[str, Any],
    times: np.ndarray,
    initial: np.ndarray,
    drivers: Mapping[str, np.ndarray],
    case: Mapping[str, Any],
    *,
    event_log: list[dict[str, Any]] | None = None,
) -> np.ndarray:
    if times.size < 4:
        raise ODECEGISError("ode_delay_times_too_short")
    history = np.asarray(case.get("history"), dtype=float)
    if history.shape != initial.shape or not np.isfinite(history).all():
        raise ODECEGISError("ode_delay_history_invalid")
    trajectory = np.empty((times.size, compiled["state_dim"]), dtype=float)
    trajectory[0] = initial
    coefficients = np.asarray(compiled["coefficients"], dtype=float)
    basis = compiled["basis"]
    delays = list(compiled.get("delays", []))
    event_specs = list(compiled.get("events", []))
    event_counts = [0 for _ in event_specs]
    event_count = 0
    for index in range(times.size - 1):
        state = trajectory[index]
        delayed: dict[str, float] = {}
        for delay_index, spec in enumerate(delays):
            query = float(times[index] - spec["tau"])
            state_index = int(spec["state_index"])
            if query <= float(times[0]):
                value = float(history[state_index])
            else:
                value = float(np.interp(query, times[:index + 1], trajectory[:index + 1, state_index]))
            delayed[f"delay:{delay_index}"] = value
        driver_values = {name: float(np.interp(times[index], times, values))
                         for name, values in drivers.items()}
        driver_values.update(delayed)
        features = _basis_values(state, basis, float(times[index]), driver_values)
        derivative = np.sum(coefficients.T * features, axis=0)
        step = float(times[index + 1] - times[index])
        next_state = state + step * derivative
        if not np.isfinite(next_state).all() or np.max(np.abs(next_state)) > 1e6:
            raise ODECEGISError("ode_numerical_guard_exceeded")
        # Delay integration is an explicit method-of-steps approximation.  A
        # sampled event layer is still useful for hybrid delay systems: detect
        # bounded threshold crossings on each step, resolve simultaneous rules
        # by priority, and apply one atomic reset at the sample endpoint.
        eligible = []
        for event_index, spec in enumerate(event_specs):
            if event_counts[event_index] >= int(spec.get("max_occurrences", 32)):
                continue
            if any(event_counts[int(dep)] <= 0 for dep in spec.get("requires", [])):
                continue
            state_index = int(spec["state_index"])
            before = float(state[state_index] - float(spec["threshold"]))
            after = float(next_state[state_index] - float(spec["threshold"]))
            direction = int(spec.get("direction", 0))
            crossed = ((direction < 0 and before > 0.0 and after <= 0.0)
                       or (direction > 0 and before < 0.0 and after >= 0.0)
                       or (direction == 0 and before != after and before * after <= 0.0))
            if crossed:
                eligible.append((-int(spec.get("priority", 0)), event_index))
        if eligible:
            _, event_index = min(eligible)
            spec = event_specs[event_index]
            if isinstance(spec.get("reset"), list):
                next_state = next_state + np.asarray(spec["reset"], dtype=float)
            else:
                next_state[int(spec["state_index"])] += float(spec["reset_delta"])
            event_counts[event_index] += 1
            event_count += 1
            if event_log is not None:
                event_log.append({"event_index": event_index, "time": float(times[index + 1]),
                                  "state_index": int(spec["state_index"]),
                                  "reset_delta": spec.get("reset_delta"),
                                  **({"reset": list(spec["reset"])} if isinstance(spec.get("reset"), list) else {}),
                                  "priority": int(spec.get("priority", 0)),
                                  "requires": list(spec.get("requires", [])),
                                  "occurrence": event_counts[event_index],
                                  "detection": "sampled_method_of_steps"})
            if event_count > 32:
                raise ODECEGISError("ode_event_budget_exceeded")
        trajectory[index + 1] = next_state
    return trajectory


def _estimate_stiffness(
    compiled: Mapping[str, Any], initial: np.ndarray, time: float,
    driver_values: Mapping[str, float],
) -> dict[str, Any]:
    """Estimate local stiffness for routing, not for a stability proof."""
    try:
        coefficients = np.asarray(compiled["coefficients"], dtype=float)
        basis = compiled["basis"]

        def value(state: np.ndarray) -> np.ndarray:
            features = _basis_values(state, basis, time, driver_values)
            return np.sum(coefficients.T * features, axis=0)

        dimension = int(initial.size)
        jacobian = np.empty((dimension, dimension), dtype=float)
        for column in range(dimension):
            step = 1e-6 * max(1.0, abs(float(initial[column])))
            plus = initial.copy(); plus[column] += step
            minus = initial.copy(); minus[column] -= step
            jacobian[:, column] = (value(plus) - value(minus)) / (2.0 * step)
        eigenvalues = np.linalg.eigvals(jacobian)
        magnitudes = np.abs(eigenvalues)
        nonzero = magnitudes[magnitudes > 1e-10]
        spread = float(np.max(nonzero) / np.min(nonzero)) if nonzero.size else 1.0
        spectral_radius = float(np.max(magnitudes)) if magnitudes.size else 0.0
        return {
            "status": "estimated", "stiff": bool(spread >= 1e3 or spectral_radius >= 1e3),
            "eigenvalue_spread": spread, "spectral_radius": spectral_radius,
            "policy": "local_jacobian_routing_only_not_global_stability_proof",
        }
    except (FloatingPointError, ValueError, TypeError, np.linalg.LinAlgError):
        return {"status": "not_assessed", "stiff": None,
                "policy": "stiffness_estimate_failed_no_solver_claim"}


def _select_solver(
    compiled: Mapping[str, Any], initial: np.ndarray, time: float,
    driver_values: Mapping[str, float],
) -> tuple[str, dict[str, Any]]:
    requested = str(compiled.get("solver", "RK45"))
    if requested != "auto":
        return requested, {"status": "not_needed", "requested": requested,
                           "selected": requested, "policy": "user_selected_solver"}
    estimate = _estimate_stiffness(compiled, initial, time, driver_values)
    selected = "Radau" if estimate.get("stiff") is True else "RK45"
    return selected, {"requested": "auto", "selected": selected, **estimate}


def _integrate(
    compiled: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    event_log: list[dict[str, Any]] | None = None,
    integration_evidence: dict[str, Any] | None = None,
) -> np.ndarray:
    from scipy.integrate import solve_ivp

    times = np.asarray(case.get("times"), dtype=float)
    initial = np.asarray(case.get("initial"), dtype=float)
    if times.ndim != 1 or times.size < 3 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ODECEGISError("ode_times_invalid")
    if initial.shape != (compiled["state_dim"],) or not np.isfinite(initial).all():
        raise ODECEGISError("ode_initial_invalid")
    coefficients = np.asarray(compiled["coefficients"], dtype=float)
    basis = compiled["basis"]
    driver_names = list(compiled.get("drivers", []))
    drivers: dict[str, np.ndarray] = {}
    if driver_names:
        supplied_drivers = case.get("drivers")
        if not isinstance(supplied_drivers, Mapping):
            raise ODECEGISError("ode_driver_data_missing")
        for name in driver_names:
            values = np.asarray(supplied_drivers.get(name), dtype=float)
            if values.shape != times.shape or not np.isfinite(values).all():
                raise ODECEGISError("ode_driver_data_invalid")
            drivers[name] = values
    event_specs = list(compiled.get("events", []))
    if compiled.get("delays"):
        event_start = len(event_log) if event_log is not None else 0
        trajectory = _integrate_delay(compiled, times, initial, drivers, case, event_log=event_log)
        if integration_evidence is not None:
            integration_evidence.update({
                "solver": "explicit_euler_method_of_steps",
                "solver_requested": str(compiled.get("solver", "RK45")),
                "stiffness": {"status": "not_assessed", "reason": "delay_method_of_steps"},
                "rhs_evaluations": int(times.size - 1),
                "event_count": (len(event_log) - event_start) if event_log is not None else 0,
                "delay_count": len(compiled.get("delays", [])),
            })
        return trajectory
    evaluations = 0
    event_count = 0

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        evaluations += 1
        if evaluations > 10_000 or not np.isfinite(state).all() or np.max(np.abs(state)) > 1e6:
            raise RuntimeError("ode_numerical_guard_exceeded")
        with np.errstate(over="raise", invalid="raise"):
            driver_values = {
                name: float(np.interp(_time, times, values))
                for name, values in drivers.items()
            }
            features = _basis_values(state, basis, _time, driver_values)
            # Each equation owns one coefficient per basis term.  Elementwise
            # reduction preserves the historical scalar behavior while also
            # supporting vector states without returning a matrix to solve_ivp.
            return np.sum(coefficients.T * features, axis=0)

    event_counts = [0 for _ in event_specs]

    def make_event(event_index: int, spec: Mapping[str, Any]):
        def crossing(_time: float, state: np.ndarray) -> float:
            if event_counts[event_index] >= int(spec.get("max_occurrences", 32)):
                return 1.0
            if any(event_counts[int(dependency)] <= 0 for dependency in spec.get("requires", [])):
                return 1.0
            return float(state[int(spec["state_index"])] - float(spec["threshold"]))
        crossing.direction = int(spec["direction"])
        crossing.terminal = True
        return crossing

    driver_values_at_start = {
        name: float(np.interp(float(times[0]), times, values))
        for name, values in drivers.items()
    }
    selected_solver, solver_evidence = _select_solver(
        compiled, initial, float(times[0]), driver_values_at_start,
    )
    event_functions = [make_event(index, spec) for index, spec in enumerate(event_specs)]
    current_time = float(times[0])
    current_state = initial.copy()
    trajectory_rows = [current_state.copy()]
    for end_time in times[1:]:
        end_time = float(end_time)
        while current_time < end_time - 1e-12:
            result = solve_ivp(
                rhs, (current_time, end_time), current_state, t_eval=[end_time],
                events=event_functions or None, method=selected_solver, rtol=1e-6, atol=1e-8,
            )
            if not result.success:
                raise ODECEGISError("ode_integration_failed")
            hits = []
            for event_index, event_times in enumerate(result.t_events or []):
                if len(event_times):
                    hits.append((float(event_times[0]), -int(event_specs[event_index].get("priority", 0)), event_index))
            if not hits:
                if result.y.shape != (compiled["state_dim"], 1):
                    raise ODECEGISError("ode_trajectory_shape_mismatch")
                current_state = result.y[:, -1]
                current_time = end_time
                break
            hit_time, _priority, event_index = min(hits)
            if not result.y_events or len(result.y_events[event_index]) == 0:
                raise ODECEGISError("ode_event_state_missing")
            # ``solve_ivp`` may report a crossing that starts exactly on the
            # left endpoint of the current segment.  In that case the sample
            # at ``current_time`` was already emitted by the previous outer
            # iteration, so update that row to the post-reset state.  Without
            # this correction the event log is right but the trajectory is
            # one sample behind the reset (an especially subtle error for
            # sampled hybrid systems).
            started_on_boundary = abs(float(result.t_events[event_index][0]) - current_time) <= 1e-10
            current_state = np.asarray(result.y_events[event_index][0], dtype=float)
            spec = event_specs[event_index]
            if isinstance(spec.get("reset"), list):
                current_state = current_state + np.asarray(spec["reset"], dtype=float)
            else:
                current_state[int(spec["state_index"])] += float(spec["reset_delta"])
            current_time = hit_time
            event_counts[event_index] += 1
            event_count += 1
            if event_log is not None:
                event_log.append({"event_index": event_index, "time": hit_time,
                                  "state_index": int(spec["state_index"]),
                                  "reset_delta": spec.get("reset_delta"),
                                  **({"reset": list(spec["reset"])} if isinstance(spec.get("reset"), list) else {}),
                                  "priority": int(spec.get("priority", 0)),
                                  "requires": list(spec.get("requires", [])),
                                  "occurrence": event_counts[event_index]})
            if started_on_boundary and trajectory_rows:
                trajectory_rows[-1] = current_state.copy()
            if event_count > 32:
                raise ODECEGISError("ode_event_budget_exceeded")
        trajectory_rows.append(current_state.copy())
    trajectory = np.asarray(trajectory_rows, dtype=float)
    if not np.isfinite(trajectory).all():
        raise ODECEGISError("ode_non_finite_trajectory")
    if integration_evidence is not None:
        integration_evidence.update({
            "solver": selected_solver,
            "solver_requested": str(compiled.get("solver", "RK45")),
            "stiffness": solver_evidence,
            "rhs_evaluations": int(evaluations),
            "event_count": int(event_count),
            "delay_count": 0,
        })
    return trajectory


def evaluate_ode_candidate(compiled: Mapping[str, Any], cases: Sequence[Mapping[str, Any]], *, tolerance: float = 1e-2) -> dict[str, Any]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases or len(cases) > 128:
        raise ODECEGISError("ode_cases_invalid")
    if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) or tolerance <= 0:
        raise ODECEGISError("ode_tolerance_invalid")
    violations = []
    errors = []
    flattened_predictions: list[float] = []
    max_abs_state = 0.0
    event_records: list[dict[str, Any]] = []
    integration_records: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping) or "observations" not in case:
            return {"status": "not_assessed", "failure_code": "ode_case_contract_invalid", "violations": [], "cost_units": index + 1}
        expected = np.asarray(case["observations"], dtype=float)
        if expected.ndim != 2 or expected.shape[1] != compiled["state_dim"] or not np.isfinite(expected).all():
            return {"status": "not_assessed", "failure_code": "ode_observations_invalid", "violations": [], "cost_units": index + 1}
        try:
            integration_evidence: dict[str, Any] = {}
            predicted = _integrate(compiled, case, event_log=event_records,
                                   integration_evidence=integration_evidence)
            integration_records.append({"case": index, **integration_evidence})
        except Exception as exc:
            return {"status": "not_assessed", "failure_code": str(exc)[:100], "violations": [], "cost_units": index + 1}
        if predicted.shape != expected.shape:
            return {"status": "not_assessed", "failure_code": "ode_trajectory_shape_mismatch", "violations": [], "cost_units": index + 1}
        flattened_predictions.extend(float(value) for value in predicted.reshape(-1))
        max_abs_state = max(max_abs_state, float(np.max(np.abs(predicted))))
        absolute = np.abs(predicted - expected)
        errors.extend(absolute.reshape(-1).tolist())
        worst = float(np.max(absolute))
        if worst > float(tolerance):
            violations.append({"reason": "ode_trajectory_error_exceeds_tolerance",
                               "witness_id": str(case.get("id", f"case_{index}"))[:80],
                               "absolute_error": worst})
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    identifiability = None
    # A local response-rank diagnostic helps distinguish a good-looking fit
    # from parameters that cannot be separated on the supplied trajectory. It
    # is intentionally advisory: finite Jacobian rank is not global
    # identifiability and never changes pass/fail by itself.
    try:
        from .identifiability import assess_local_identifiability
        coefficient_array = np.asarray(compiled.get("coefficients", []), dtype=float)
        parameter_names = [
            f"c_{row}_{column}"
            for row in range(coefficient_array.shape[0])
            for column in range(coefficient_array.shape[1])
        ]
        if parameter_names and len(parameter_names) <= 4 and cases:
            base_case = dict(cases[0])

            def predict(parameters: Mapping[str, float]) -> Sequence[float]:
                revised = dict(compiled)
                values = coefficient_array.copy()
                for name, value in parameters.items():
                    _, row, column = name.split("_")
                    values[int(row), int(column)] = float(value)
                revised["coefficients"] = values.tolist()
                evidence: dict[str, Any] = {}
                trajectory = _integrate(revised, base_case, integration_evidence=evidence)
                return trajectory.reshape(-1)

            identifiability = assess_local_identifiability(
                predict,
                {name: float(value) for name, value in zip(parameter_names, coefficient_array.reshape(-1))},
                max_evaluations=min(128, 1 + 2 * len(parameter_names)),
            )
        elif parameter_names:
            identifiability = {
                "status": "not_assessed", "reason": "parameter_budget_exceeded",
                "parameter_count": len(parameter_names),
            }
    except Exception as exc:
        identifiability = {"status": "not_assessed", "reason": type(exc).__name__}
    complexity = float(compiled["state_dim"] * len(compiled["basis"])
                       + len(compiled.get("events", []))
                       + len(compiled.get("delays", [])))
    return {"status": "pass" if not violations else "fail", "score": rmse,
            "predictions": flattened_predictions,
            "metrics": {
                "validation_loss": float(rmse) if rmse is not None else 0.0,
                "complexity": complexity,
                "constraint_violation": 0.0,
                # This is only a numerical guard-margin proxy, not a Lyapunov
                # stability certificate.  A separate stability analysis is
                # required before making a dynamical-stability claim.
                "instability": float(max_abs_state / 1e6),
            },
            "violations": violations[:16], "cost_units": len(cases),
            "event_count": len(event_records), "events": event_records[:32],
            "integration_evidence": integration_records[:128],
            "identifiability": identifiability,
            "policy": "bounded_ode_trajectory_evaluation;_local_identifiability_diagnostic_only;_not_a_dynamics_proof;instability_is_guard_margin"}


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
        if basis in normalized["basis"] or len(normalized["basis"]) >= 5:
            continue
        if basis == "cross" and normalized["state_dim"] < 2:
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
