"""Bounded deterministic restarts using training observations only.

The caller supplies a trusted numeric evaluator, not generated code. This is
local least squares, not a global solver or an identifiability certificate.
"""
from __future__ import annotations

import numpy as np


class _AttemptLimit(Exception):
    """A local allocation expired; this is not a mathematical counterexample."""


def _starts(bounds):
    initial = np.asarray([b[2] for b in bounds], dtype=float)
    lower = np.asarray([b[0] for b in bounds], dtype=float)
    upper = np.asarray([b[1] for b in bounds], dtype=float)
    step = np.minimum(0.1 * (upper - lower), np.maximum(np.abs(initial), 1.0))
    starts = [initial]
    for sign in (1, -1):
        candidate = np.clip(initial + sign * step, lower, upper)
        if not any(np.array_equal(candidate, old) for old in starts):
            starts.append(candidate)
    return starts, lower, upper


def _affine_subset(program, selected):
    """Prove affine dependence with ``selected`` as the inner parameters.

    This deliberately declines products of parameter-dependent expressions,
    parameterized denominators and nonlinear functions of parameters.
    """
    dependent = {}
    affine = {}
    for node in program.order:
        key, op = node["id"], node["op"]
        args = node["inputs"]
        if op == "parameter":
            dependent[key], affine[key] = node["id"] in selected, True
        elif op in ("variable", "constant"):
            dependent[key], affine[key] = False, True
        elif op in ("observation", "negate"):
            dependent[key], affine[key] = dependent[args[0]], affine[args[0]]
        elif op in ("add", "subtract"):
            dependent[key] = dependent[args[0]] or dependent[args[1]]
            affine[key] = affine[args[0]] and affine[args[1]]
        elif op == "multiply":
            left, right = dependent[args[0]], dependent[args[1]]
            dependent[key] = left or right
            affine[key] = affine[args[0]] and affine[args[1]] and not (left and right)
        elif op == "divide":
            dependent[key] = dependent[args[0]] or dependent[args[1]]
            affine[key] = affine[args[0]] and not dependent[args[1]]
        elif op in ("exp", "log"):
            dependent[key] = dependent[args[0]]
            affine[key] = not dependent[key]
        else:  # The trusted interpreter should already have rejected this.
            return False
    return all(affine[key] for key in program.outputs)


def _all_parameters_affine(program):
    """Prove affine dependence from the allow-listed graph syntax."""
    return _affine_subset(program, {n["id"] for n in program.order if n["op"] == "parameter"})


def _select_affine_subset(program, names):
    """Conservatively retain a deterministic subset for variable projection.

    Testing each addition prevents selecting both sides of a parameter product;
    rejected parameters remain outer nonlinear variables.
    """
    selected = []
    for name in names:
        candidate = selected + [name]
        if _affine_subset(program, set(candidate)):
            selected = candidate
    return selected


def _linear_fit(program, points, target, names, bounds, scale, fit_budget):
    """Solve a structurally proven all-parameter affine block."""
    from scipy.optimize import lsq_linear

    initial = np.asarray([b[2] for b in bounds], dtype=float)
    lower = np.asarray([b[0] for b in bounds], dtype=float)
    upper = np.asarray([b[1] for b in bounds], dtype=float)
    required = len(names) + 1
    attempt = {"backend": "scipy.optimize.lsq_linear", "start": dict(zip(names, map(float, initial))),
               "evaluation_quota": required, "structurally_affine": True}
    if fit_budget < required:
        attempt.update(status="fit_budget_exhausted", evaluations_used=0)
        return None, attempt

    before = program.counter.used
    try:
        anchor = program.evaluate(points, dict(zip(names, initial)))
        columns = []
        for index, (lo, hi, _) in enumerate(bounds):
            step = min(0.1 * (hi - lo), max(abs(initial[index]), 1.0))
            candidate = initial[index] + step if initial[index] + step <= hi else initial[index] - step
            delta = candidate - initial[index]
            if delta == 0:  # Extremely narrow representable bounds near a large value.
                candidate = hi if abs(hi - initial[index]) >= abs(initial[index] - lo) else lo
                delta = candidate - initial[index]
            if delta == 0:
                attempt.update(status="not_converged", failure_reason="unresolvable_parameter_step",
                               evaluations_used=program.counter.used - before)
                return None, attempt
            probe = initial.copy()
            probe[index] = candidate
            value = program.evaluate(points, dict(zip(names, probe)))
            columns.append(((value - anchor) / delta).ravel())
        design = np.column_stack(columns)
        row_scale = np.tile(scale, len(points))
        weighted_design = design / row_scale[:, None]
        weighted_target = (target.ravel() - anchor.ravel() + design @ initial) / row_scale
        fitted = lsq_linear(weighted_design, weighted_target, bounds=(lower, upper),
                            method="trf", tol=1e-10, lsmr_tol="auto", max_iter=200)
        valid = fitted.success and np.isfinite(fitted.x).all() and np.isfinite(fitted.fun).all()
        attempt.update(status="converged" if valid else "not_converged",
                       evaluations_used=program.counter.used - before,
                       design_rank=int(np.linalg.matrix_rank(weighted_design)),
                       design_columns=len(names))
        if not valid:
            return None, attempt
        magnitude = float(np.max(np.abs(fitted.fun), initial=0.0))
        attempt["normalized_train_rmse"] = (float(magnitude * np.sqrt(np.mean((fitted.fun / magnitude)**2)))
                                               if magnitude else 0.0)
        return {k: float(v) for k, v in zip(names, fitted.x)}, attempt
    except ArithmeticError as exc:
        attempt.update(status="numeric_failure", failure_node=getattr(exc, "node_id", None),
                       evaluations_used=program.counter.used - before)
        return None, attempt
    except (ValueError, np.linalg.LinAlgError):
        attempt.update(status="not_converged", failure_reason="linear_solver_numeric_failure",
                       evaluations_used=program.counter.used - before)
        return None, attempt


def _variable_projection_fit(program, points, target, names, bounds, scale, fit_budget,
                             linear_names, outer_names):
    """Fit a proven affine inner block while searching outer parameters."""
    from scipy.optimize import least_squares, lsq_linear

    all_bounds = {name: bound for name, bound in zip(names, bounds)}
    linear_bounds = [all_bounds[name] for name in linear_names]
    outer_bounds = [all_bounds[name] for name in outer_names]
    inner_initial = np.asarray([b[2] for b in linear_bounds], dtype=float)
    outer_initial = np.asarray([b[2] for b in outer_bounds], dtype=float)
    inner_lower = np.asarray([b[0] for b in linear_bounds], dtype=float)
    inner_upper = np.asarray([b[1] for b in linear_bounds], dtype=float)
    starts, outer_lower, outer_upper = _starts(outer_bounds)
    per_projection = len(linear_names) + 1
    report = {"status": "not_converged", "policy": "variable_projection/v1",
              "linear_parameter_ids": list(linear_names), "outer_parameter_ids": list(outer_names),
              "selection_data": "training_only", "attempts": [], "selected_attempt": None,
              "global_optimum_proven": False, "projection_evaluations_per_call": per_projection,
              "fit_evaluation_budget": fit_budget}
    fit_end = program.counter.used + fit_budget
    best_error, best_params = float("inf"), None

    def project(outer_values, quota_end):
        if program.counter.used + per_projection > quota_end:
            raise _AttemptLimit()
        outer = dict(zip(outer_names, map(float, outer_values)))
        inner = dict(zip(linear_names, map(float, inner_initial)))
        base_params = {**outer, **inner}
        anchor = program.evaluate(points, base_params)
        columns = []
        for index, (lo, hi, _) in enumerate(linear_bounds):
            step = min(0.1 * (hi - lo), max(abs(inner_initial[index]), 1.0))
            candidate = inner_initial[index] + step if inner_initial[index] + step <= hi else inner_initial[index] - step
            delta = candidate - inner_initial[index]
            if delta == 0:
                candidate = hi if abs(hi - inner_initial[index]) >= abs(inner_initial[index] - lo) else lo
                delta = candidate - inner_initial[index]
            if delta == 0:
                raise _AttemptLimit()
            probe = dict(base_params)
            probe[linear_names[index]] = candidate
            columns.append(((program.evaluate(points, probe) - anchor) / delta).ravel())
        design = np.column_stack(columns)
        rowscale = np.tile(scale, len(points))
        weighted_design = design / rowscale[:, None]
        weighted_target = (target.ravel() - anchor.ravel() + design @ inner_initial) / rowscale
        fitted = lsq_linear(weighted_design, weighted_target, bounds=(inner_lower, inner_upper),
                            method="trf", tol=1e-10, lsmr_tol="auto", max_iter=200)
        if not fitted.success or not np.isfinite(fitted.x).all() or not np.isfinite(fitted.fun).all():
            raise ValueError("variable_projection_linear_failure")
        values = {**outer, **dict(zip(linear_names, map(float, fitted.x)))}
        magnitude = float(np.max(np.abs(fitted.fun), initial=0.0))
        error = (float(magnitude * np.sqrt(np.mean((fitted.fun / magnitude)**2)))
                 if magnitude else 0.0)
        return values, error, int(np.linalg.matrix_rank(weighted_design)), fitted.fun

    for index, start in enumerate(starts):
        remaining = fit_end - program.counter.used
        if remaining < per_projection:
            break
        quota = max(per_projection, int(0.6 * remaining) if index == 0 and len(starts) > 1
                    else remaining // (len(starts) - index))
        before = program.counter.used
        attempt = {"backend": "least_squares+lsq_linear", "start": dict(zip(outer_names, map(float, start))),
                   "evaluation_quota": quota, "linear_parameter_ids": list(linear_names)}
        local_best = {"values": None, "error": float("inf"), "rank": None}

        def residual(values):
            nonlocal local_best
            projected, error, rank, residual_values = project(values, min(fit_end, before + quota))
            if error < local_best["error"]:
                local_best = {"values": projected, "error": error, "rank": rank}
            return residual_values

        try:
            fitted = least_squares(residual, start, bounds=(outer_lower, outer_upper),
                                   method="trf", jac="2-point", x_scale=1.0, max_nfev=100,
                                   ftol=1e-9, xtol=1e-9, gtol=1e-9)
            valid = fitted.success and local_best["values"] is not None
            attempt.update(status="converged" if valid else "not_converged",
                           evaluations_used=program.counter.used - before,
                           normalized_train_rmse=local_best["error"] if valid else None,
                           projection_design_rank=local_best["rank"])
            if valid and local_best["error"] < best_error:
                best_error, best_params = local_best["error"], local_best["values"]
                report["selected_attempt"] = len(report["attempts"])
        except _AttemptLimit:
            attempt.update(status="attempt_budget_exhausted", evaluations_used=program.counter.used - before)
        except ArithmeticError as exc:
            attempt.update(status="numeric_failure", failure_node=getattr(exc, "node_id", None),
                           evaluations_used=program.counter.used - before)
        except (ValueError, np.linalg.LinAlgError) as exc:
            attempt.update(status="not_converged", failure_reason=str(exc),
                           evaluations_used=program.counter.used - before)
        report["attempts"].append(attempt)
        if best_error <= 1e-8:
            report["early_stop_reason"] = "training_residual_threshold"
            break
    if best_params is None:
        statuses = {a["status"] for a in report["attempts"]}
        report["status"] = ("fit_budget_exhausted" if not statuses or "attempt_budget_exhausted" in statuses
                             else "numeric_failure" if statuses == {"numeric_failure"} else "not_converged")
        return None, report
    report.update(status="variable_projection_completed", normalized_train_rmse=best_error,
                  parameter_count=len(names), global_optimum_proven=False)
    return best_params, report


def fit_training_parameters(program, points, target, parameter_bounds, *, reserved_evaluations):
    """Select a converged fit solely by normalized training squared error.

    No search labels, check tolerances, properties or final data enter this
    function. Count finite differences and failed domain evaluations through
    the same program counter. Leave the caller's verification allocation intact.
    """
    names = list(parameter_bounds)
    starts, lower, upper = _starts([parameter_bounds[k] for k in names])
    scale = np.maximum(1.0, np.max(np.abs(target), axis=0))
    counter = program.counter
    fit_budget = max(0, counter.maximum - counter.used - reserved_evaluations)
    fit_end = counter.used + fit_budget
    report = {
        "status": "not_converged", "training_cases": len(points),
        "policy": "structural_linear_then_deterministic_restarts/v1", "selection_data": "training_only",
        "parameter_count": len(names), "jacobian_rank": None,
        "global_optimum_proven": False, "attempts": [], "selected_attempt": None,
        "fit_evaluation_budget": fit_budget, "reserved_check_evaluations": reserved_evaluations,
        "early_stop_normalized_train_rmse": 1e-8,
    }
    if _all_parameters_affine(program):
        linear, attempt = _linear_fit(program, points, target, names,
                                      [parameter_bounds[k] for k in names], scale, fit_budget)
        report["attempts"].append(attempt)
        if linear is not None:
            report.update(status="bounded_linear_fit_completed", selected_attempt=0,
                          jacobian_rank=attempt["design_rank"], convex_subproblem=True,
                          eliminated_nonlinear_search_dimensions=len(names))
            return linear, report
        if attempt["status"] == "fit_budget_exhausted":
            report["status"] = "fit_budget_exhausted"
            return None, report

    affine_names = _select_affine_subset(program, names)
    outer_names = [name for name in names if name not in affine_names]
    if affine_names and outer_names:
        projected, projected_report = _variable_projection_fit(
            program, points, target, names, [parameter_bounds[k] for k in names],
            scale, fit_budget, affine_names, outer_names)
        report["attempts"].extend(projected_report["attempts"])
        for key, value in projected_report.items():
            if key != "attempts":
                report[key] = value
        if projected is not None:
            report["selected_attempt"] = len(report["attempts"]) - len(projected_report["attempts"]) + projected_report["selected_attempt"]
            return projected, report
        if projected_report["status"] == "fit_budget_exhausted":
            return None, report

    from scipy.optimize import least_squares

    best = None
    best_error = float("inf")
    for index, start in enumerate(starts):
        remaining = fit_end - counter.used
        if remaining <= 0:
            break
        # The initial start gets 60%; later starts share remaining evaluations.
        # Unused allocations carry forward, but no verification funds are spent.
        quota = (max(1, int(0.6 * remaining)) if index == 0 and len(starts) > 1
                 else max(1, remaining // (len(starts) - index)))
        before = counter.used
        attempt = {"start": dict(zip(names, map(float, start))), "evaluation_quota": quota}

        def residual(values):
            if counter.used - before >= quota:
                raise _AttemptLimit()
            prediction = program.evaluate(points, dict(zip(names, values)))
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                return ((prediction - target) / scale).ravel()

        try:
            fitted = least_squares(
                residual, start, bounds=(lower, upper), method="trf", jac="2-point",
                x_scale=1.0, max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            valid = (fitted.success and np.isfinite(fitted.x).all()
                     and np.isfinite(fitted.fun).all() and np.isfinite(fitted.jac).all())
            attempt["status"] = "converged" if valid else "not_converged"
            if valid:
                # Stable RMS avoids squaring a large residual before scaling it.
                magnitude = float(np.max(np.abs(fitted.fun), initial=0.0))
                error = (float(magnitude * np.sqrt(np.mean((fitted.fun / magnitude)**2)))
                         if magnitude else 0.0)
                attempt["normalized_train_rmse"] = error
                if error < best_error:
                    best, best_error = fitted, error
                    report["selected_attempt"] = len(report["attempts"])
        except _AttemptLimit:
            attempt["status"] = "attempt_budget_exhausted"
        except ArithmeticError as exc:
            attempt.update(status="numeric_failure", failure_node=getattr(exc, "node_id", None))
        attempt["evaluations_used"] = counter.used - before
        report["attempts"].append(attempt)
        if best_error <= report["early_stop_normalized_train_rmse"]:
            report["early_stop_reason"] = "training_residual_threshold"
            break

    if best is None:
        statuses = {a["status"] for a in report["attempts"]}
        report["status"] = ("numeric_failure" if statuses == {"numeric_failure"}
                            else "fit_budget_exhausted" if not statuses or "attempt_budget_exhausted" in statuses
                            else "not_converged")
        return None, report
    report.update(status="local_fit_completed", jacobian_rank=int(np.linalg.matrix_rank(best.jac)))
    return {k: float(v) for k, v in zip(names, best.x)}, report
