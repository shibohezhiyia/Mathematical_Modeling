"""Finite solver-tolerance stability checks for mathematical conclusions."""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "mathmodel.numerical-stability/v1"


class NumericalStabilityError(ValueError):
    """Raised when the tolerance-stability contract is invalid."""


def assess_numerical_error_budget(
    runs: Sequence[Mapping[str, Any]], *, max_runs: int = 16,
) -> dict[str, Any]:
    """Record declared discretization/conditioning/solver error metadata.

    Estimates must come from an independently checked solver. This function
    validates and aggregates the declarations; it never turns a condition
    number into an error theorem or combines values with different units.
    """
    if not isinstance(runs, Sequence) or isinstance(runs, (str, bytes)):
        raise NumericalStabilityError("numerical_error_runs_must_be_a_sequence")
    if type(max_runs) is not int or not 1 <= max_runs <= 64 or not 1 <= len(runs) <= max_runs:
        raise NumericalStabilityError("numerical_error_run_budget_invalid")
    rows: list[dict[str, Any]] = []
    units: set[str] = set()
    for index, raw in enumerate(runs):
        if not isinstance(raw, Mapping):
            raise NumericalStabilityError(f"numerical_error_run_{index}_must_be_a_mapping")
        tolerance = _finite(raw.get("tolerance"))
        if tolerance <= 0:
            raise NumericalStabilityError("numerical_error_tolerance_must_be_positive")
        row: dict[str, Any] = {"tolerance": tolerance}
        for field in ("discretization_error", "solver_error_bound", "condition_number"):
            if field not in raw:
                row[field] = None
                continue
            value = _finite(raw[field])
            if value < 0:
                raise NumericalStabilityError(f"{field}_must_be_nonnegative")
            row[field] = value
        unit = raw.get("unit_signature")
        if unit is not None:
            if not isinstance(unit, str) or not unit.strip():
                raise NumericalStabilityError("numerical_error_unit_invalid")
            units.add(unit.strip())
            row["unit_signature"] = unit.strip()
        rows.append(row)
    if len(units) > 1:
        raise NumericalStabilityError("numerical_error_units_must_match")
    finite_bounds = [
        float(row["discretization_error"] or 0.0) + float(row["solver_error_bound"] or 0.0)
        for row in rows
    ]
    return {
        "schema_version": "mathmodel.numerical-error-budget/v1",
        "status": "assessed" if any(row["discretization_error"] is not None
                                     or row["solver_error_bound"] is not None
                                     or row["condition_number"] is not None for row in rows)
                   else "not_assessed",
        "runs": rows,
        "unit_signature": next(iter(units), None),
        "declared_additive_bound": max(finite_bounds) if finite_bounds else None,
        "condition_number_max": max((row["condition_number"] for row in rows
                                      if row["condition_number"] is not None), default=None),
        "policy": "declared_solver_metadata_only; no universal discretization_or_conditioning_error_theorem",
    }


def _finite(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise NumericalStabilityError("solver_outputs_must_be_finite_scalars") from exc
    if not math.isfinite(number):
        raise NumericalStabilityError("solver_outputs_must_be_finite_scalars")
    return number


def assess_solver_tolerance_stability(
    solve: Callable[[float], Mapping[str, Any]],
    *,
    tolerances: Sequence[float] = (1e-4, 1e-6, 1e-8),
    absolute_tolerance: float = 1e-6,
    relative_tolerance: float = 1e-4,
    max_failures: int = 0,
    uncertainty_field: str | None = None,
) -> dict[str, Any]:
    """Run one trusted/sandboxed solver at fixed tolerances and compare outputs."""
    if not isinstance(tolerances, (list, tuple)) or not 2 <= len(tolerances) <= 8:
        raise NumericalStabilityError("tolerances_must_contain_2_to_8_values")
    normalized = []
    for value in tolerances:
        if type(value) not in (int, float) or not math.isfinite(float(value)) or not 0 < float(value) < 1:
            raise NumericalStabilityError("tolerances_must_be_finite_between_0_and_1")
        normalized.append(float(value))
    if len(set(normalized)) != len(normalized):
        raise NumericalStabilityError("tolerances_must_be_unique")
    if (type(absolute_tolerance) not in (int, float) or type(relative_tolerance) not in (int, float)
            or not math.isfinite(float(absolute_tolerance)) or not math.isfinite(float(relative_tolerance))
            or absolute_tolerance < 0 or relative_tolerance < 0):
        raise NumericalStabilityError("invalid_stability_threshold")
    if type(max_failures) is not int or not 0 <= max_failures <= len(normalized):
        raise NumericalStabilityError("invalid_stability_failure_budget")
    if uncertainty_field is not None and (type(uncertainty_field) is not str or not uncertainty_field.strip()):
        raise NumericalStabilityError("uncertainty_field_must_be_a_nonempty_string")
    runs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    names: tuple[str, ...] | None = None
    for index, tolerance in enumerate(normalized):
        try:
            raw = solve(tolerance)
            if not isinstance(raw, Mapping) or not raw:
                raise NumericalStabilityError("solver_must_return_nonempty_mapping")
            current_names = tuple(sorted(str(name) for name in raw))
            if names is None:
                names = current_names
            if current_names != names:
                raise NumericalStabilityError("solver_output_schema_changed")
            outputs = {name: _finite(raw[name]) for name in names}
            runs.append({"tolerance": tolerance, "outputs": outputs})
        except NumericalStabilityError as exc:
            failures.append({"tolerance": tolerance, "index": index, "reason": str(exc)})
        except Exception as exc:  # solver internals are not a mathematical verdict
            failures.append({"tolerance": tolerance, "index": index,
                             "reason": f"solver_error:{type(exc).__name__}"})
        if len(failures) > max_failures:
            break
    if failures:
        status = "not_assessed" if len(failures) > max_failures or len(runs) < 2 else "partial"
        return _result(status, normalized, runs, failures, None, absolute_tolerance, relative_tolerance)
    if len(runs) < 2 or names is None:
        return _result("not_assessed", normalized, runs, failures, None,
                       absolute_tolerance, relative_tolerance)
    comparisons = []
    unstable = []
    for left, right in zip(runs, runs[1:]):
        differences = {}
        for name in names:
            first, second = left["outputs"][name], right["outputs"][name]
            difference = abs(first - second)
            allowed = float(absolute_tolerance) + float(relative_tolerance) * max(abs(first), abs(second), 1.0)
            differences[name] = {"absolute_difference": difference, "allowed": allowed,
                                 "stable": difference <= allowed}
            if difference > allowed:
                unstable.append({"from_tolerance": left["tolerance"],
                                 "to_tolerance": right["tolerance"], "output": name,
                                 **differences[name]})
        comparisons.append({"from_tolerance": left["tolerance"], "to_tolerance": right["tolerance"],
                            "outputs": differences})
    direction_checks = None
    if uncertainty_field is not None:
        ordered = sorted(runs, key=lambda item: item["tolerance"])
        direction_checks = []
        for tighter, looser in zip(ordered, ordered[1:]):
            if uncertainty_field not in tighter["outputs"] or uncertainty_field not in looser["outputs"]:
                direction_checks.append({"from_tolerance": looser["tolerance"], "to_tolerance": tighter["tolerance"], "status": "not_assessed", "reason": "uncertainty_field_missing"})
                continue
            tighter_value = tighter["outputs"][uncertainty_field]
            looser_value = looser["outputs"][uncertainty_field]
            allowed = float(absolute_tolerance) + float(relative_tolerance) * max(abs(looser_value), 1.0)
            direction_checks.append({
                "from_tolerance": looser["tolerance"], "to_tolerance": tighter["tolerance"],
                "looser_uncertainty": looser_value, "tighter_uncertainty": tighter_value,
                "allowed_increase": allowed, "status": "pass" if tighter_value <= looser_value + allowed else "fail",
            })
    comparison_payload = {"comparisons": comparisons, "counterexamples": unstable}
    if direction_checks is not None:
        comparison_payload["uncertainty_direction"] = {
            "field": uncertainty_field, "checks": direction_checks,
            "status": "not_assessed" if any(item["status"] == "not_assessed" for item in direction_checks)
            else ("pass" if all(item["status"] == "pass" for item in direction_checks) else "fail"),
            "policy": "tighter_tolerance_should_not_increase_reported_numerical_uncertainty",
        }
    return _result("unstable" if unstable else "stable_on_tested_tolerances", normalized,
                   runs, failures, comparison_payload,
                   absolute_tolerance, relative_tolerance)


def _result(status, tolerances, runs, failures, comparison, absolute, relative):
    payload = {"tolerances": tolerances, "runs": runs, "failures": failures,
               "comparison": comparison, "absolute_tolerance": absolute,
               "relative_tolerance": relative}
    return {"schema_version": SCHEMA_VERSION, "status": status,
            "tolerance_count": len(tolerances), "successful_runs": len(runs),
            "failed_runs": len(failures), **payload,
            "fingerprint": sha256(json.dumps(payload, sort_keys=True,
                                              separators=(",", ":")).encode()).hexdigest(),
            "policy": "finite_tolerance_comparison_not_numerical_error_proof"}


__all__ = ["SCHEMA_VERSION", "NumericalStabilityError", "assess_numerical_error_budget",
           "assess_solver_tolerance_stability"]
