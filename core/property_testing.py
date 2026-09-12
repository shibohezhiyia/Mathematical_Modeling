"""Bounded, reusable property checks for executable mathematical primitives.

These checks are deliberately finite experiments.  A passing result means
``tested_not_falsified`` over the recorded probe set; it is never promoted to
a symbolic proof or a global monotonicity claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Mapping


PROPERTY_TEST_SCHEMA = "mathmodel.property-test/v1"
PROPERTY_KINDS = frozenset({
    "bounded", "monotone_increasing", "monotone_decreasing", "symmetry_even", "symmetry_odd",
})
TRANSITION_PROPERTY_KINDS = frozenset({"conservation", "reversible"})


class PropertyTestError(ValueError):
    """Structured input error before an evaluator is called."""


@dataclass(frozen=True)
class PropertyTestBudget:
    probe_count: int = 32
    max_evaluations: int = 128
    seed: int = 42

    def validate(self) -> None:
        if type(self.probe_count) is not int or not 1 <= self.probe_count <= 256:
            raise PropertyTestError("probe_count_must_be_between_1_and_256")
        if type(self.max_evaluations) is not int or not 1 <= self.max_evaluations <= 1024:
            raise PropertyTestError("max_evaluations_must_be_between_1_and_1024")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise PropertyTestError("seed_must_be_an_unsigned_32_bit_integer")


def _domain(domain: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    if not isinstance(domain, Mapping) or not 1 <= len(domain) <= 16:
        raise PropertyTestError("domain_must_bind_1_to_16_variables")
    normalized = {}
    for name, interval in domain.items():
        if not isinstance(name, str) or not name:
            raise PropertyTestError("domain_variable_name_must_be_nonempty")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise PropertyTestError(f"invalid_domain:{name}")
        lower, upper = float(interval[0]), float(interval[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise PropertyTestError(f"invalid_domain:{name}")
        normalized[name] = (lower, upper)
    return normalized


def _number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PropertyTestError("evaluator_must_return_a_finite_scalar") from exc
    if not math.isfinite(result):
        raise PropertyTestError("evaluator_returned_non_finite_value")
    return result


def _evaluate(evaluate: Callable[[Mapping[str, float]], Any], points: list[dict[str, float]], budget: PropertyTestBudget):
    if len(points) > budget.max_evaluations:
        return None, {"status": "not_assessed", "reason": "property_evaluation_budget_exhausted",
                      "evaluations": 0, "violations": []}
    values = []
    try:
        for point in points:
            values.append(_number(evaluate(dict(point))))
    except PropertyTestError as exc:
        return None, {"status": "not_assessed", "reason": str(exc),
                      "evaluations": len(values) + 1, "violations": []}
    except Exception as exc:  # do not leak arbitrary evaluator traces to callers
        return None, {"status": "not_assessed", "reason": f"evaluator_error:{type(exc).__name__}",
                      "evaluations": len(values) + 1, "violations": []}
    return values, None


def _base(kind: str, budget: PropertyTestBudget) -> dict[str, Any]:
    return {
        "schema_version": PROPERTY_TEST_SCHEMA,
        "kind": kind,
        "status": "not_assessed",
        "proof_status": "not_assessed",
        "probe_count": budget.probe_count,
        "evaluations": 0,
        "violations": [],
        "note": "有限探针未发现反例不等于全域数学证明。",
    }


def check_scalar_property(
    evaluate: Callable[[Mapping[str, float]], Any],
    *,
    domain: Mapping[str, Any],
    kind: str,
    variable: str | None = None,
    lower: float | None = None,
    upper: float | None = None,
    budget: PropertyTestBudget | None = None,
    absolute_tolerance: float = 1e-8,
    relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Check one scalar property and return an auditable finite result."""
    if kind not in PROPERTY_KINDS:
        raise PropertyTestError("unsupported_property_kind")
    budget = budget or PropertyTestBudget()
    budget.validate()
    bounds = _domain(domain)
    if variable is None and len(bounds) == 1 and kind != "bounded":
        variable = next(iter(bounds))
    if kind == "bounded":
        if lower is None and upper is None:
            raise PropertyTestError("bounded_property_requires_lower_or_upper")
    else:
        if variable not in bounds:
            raise PropertyTestError("ordered_property_requires_a_domain_variable")
        if kind.startswith("symmetry") and not (bounds[variable][0] <= 0 <= bounds[variable][1]):
            raise PropertyTestError("symmetry_property_requires_zero_in_domain")
    if (not isinstance(absolute_tolerance, (int, float)) or not isinstance(relative_tolerance, (int, float))
            or absolute_tolerance < 0 or relative_tolerance < 0
            or not math.isfinite(float(absolute_tolerance)) or not math.isfinite(float(relative_tolerance))):
        raise PropertyTestError("invalid_property_tolerance")
    output = _base(kind, budget)
    rng = random.Random(budget.seed)
    midpoint = {name: (lo + hi) / 2.0 for name, (lo, hi) in bounds.items()}
    points: list[dict[str, float]] = []
    pairs: list[tuple[dict[str, float], dict[str, float]]] = []
    if kind == "bounded":
        for _ in range(budget.probe_count):
            points.append({name: rng.uniform(lo, hi) for name, (lo, hi) in bounds.items()})
    elif kind.startswith("monotone"):
        lo, hi = bounds[variable]
        grid = sorted({lo, hi, *[rng.uniform(lo, hi) for _ in range(budget.probe_count)]})
        for left, right in zip(grid, grid[1:]):
            first, second = dict(midpoint), dict(midpoint)
            first[variable], second[variable] = left, right
            pairs.append((first, second))
        points = [point for pair in pairs for point in pair]
    else:
        lo, hi = bounds[variable]
        radius = min(abs(lo), abs(hi))
        for _ in range(budget.probe_count):
            magnitude = rng.uniform(0.0, radius)
            positive, negative = dict(midpoint), dict(midpoint)
            positive[variable], negative[variable] = magnitude, -magnitude
            pairs.append((negative, positive))
        points = [point for pair in pairs for point in pair]
    values, error = _evaluate(evaluate, points, budget)
    if error:
        output.update(error)
        return output
    output["evaluations"] = len(values)
    tolerance = lambda scale: float(absolute_tolerance) + float(relative_tolerance) * abs(float(scale))
    violations = []
    if kind == "bounded":
        for index, value in enumerate(values):
            if lower is not None and value < float(lower) - tolerance(lower):
                violations.append({"probe": index, "value": value, "bound": "lower", "expected": float(lower)})
            if upper is not None and value > float(upper) + tolerance(upper):
                violations.append({"probe": index, "value": value, "bound": "upper", "expected": float(upper)})
    else:
        for index, (left, right) in enumerate(zip(values[::2], values[1::2])):
            point_left, point_right = pairs[index]
            scale = max(abs(left), abs(right))
            if kind == "monotone_increasing" and right < left - tolerance(scale):
                violations.append({"pair": index, "left": left, "right": right, "expected": "nondecreasing"})
            elif kind == "monotone_decreasing" and right > left + tolerance(scale):
                violations.append({"pair": index, "left": left, "right": right, "expected": "nonincreasing"})
            elif kind == "symmetry_even" and abs(right - left) > tolerance(scale):
                violations.append({"pair": index, "left": left, "right": right, "expected": "f(-x)=f(x)"})
            elif kind == "symmetry_odd" and abs(right + left) > tolerance(scale):
                violations.append({"pair": index, "left": left, "right": right, "expected": "f(-x)=-f(x)"})
            if len(violations) >= 32:
                break
    output["violations"] = violations
    output["status"] = "fail" if violations else "pass"
    output["proof_status"] = "counterexample_found" if violations else "tested_not_falsified"
    return output


def check_transition_property(
    transition: Callable[[Mapping[str, float]], Mapping[str, float]],
    states: list[Mapping[str, float]], *, kind: str,
    invariant: Callable[[Mapping[str, float]], float] | None = None,
    inverse: Callable[[Mapping[str, float]], Mapping[str, float]] | None = None,
    absolute_tolerance: float = 1e-8, relative_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Finite conservation/reversibility check for a trusted transition map."""
    if kind not in TRANSITION_PROPERTY_KINDS:
        raise PropertyTestError("unsupported_transition_property_kind")
    if not isinstance(states, list) or not 1 <= len(states) <= 256 or any(not isinstance(item, Mapping) for item in states):
        raise PropertyTestError("states_must_be_a_bounded_mapping_list")
    if kind == "conservation" and not callable(invariant):
        raise PropertyTestError("conservation_requires_invariant")
    if kind == "reversible" and not callable(inverse):
        raise PropertyTestError("reversible_requires_inverse")
    if (type(absolute_tolerance) not in (int, float) or type(relative_tolerance) not in (int, float)
            or not math.isfinite(float(absolute_tolerance)) or not math.isfinite(float(relative_tolerance))
            or absolute_tolerance < 0 or relative_tolerance < 0):
        raise PropertyTestError("invalid_transition_tolerance")
    output = {"schema_version": PROPERTY_TEST_SCHEMA, "kind": kind, "status": "not_assessed",
              "proof_status": "not_assessed", "evaluations": 0, "violations": [],
              "note": "有限状态探针未发现反例不等于全域数学证明。"}
    for index, state in enumerate(states):
        try:
            next_state = transition(dict(state))
            if not isinstance(next_state, Mapping):
                raise PropertyTestError("transition_must_return_mapping")
            if kind == "conservation":
                before, after = _number(invariant(dict(state))), _number(invariant(dict(next_state)))
                allowed = float(absolute_tolerance) + float(relative_tolerance) * max(abs(before), abs(after), 1.0)
                if abs(after - before) > allowed:
                    output["violations"].append({"probe": index, "before": before, "after": after, "allowed": allowed})
            else:
                restored = inverse(dict(next_state))
                if not isinstance(restored, Mapping) or set(restored) != set(state):
                    output["violations"].append({"probe": index, "reason": "inverse_schema_mismatch"})
                else:
                    diffs = {str(key): abs(_number(restored[key]) - _number(state[key])) for key in state}
                    allowed = float(absolute_tolerance) + float(relative_tolerance) * max((abs(_number(state[key])) for key in state), default=1.0)
                    if max(diffs.values(), default=0.0) > allowed:
                        output["violations"].append({"probe": index, "maximum_difference": max(diffs.values()), "allowed": allowed})
            output["evaluations"] += 1
        except PropertyTestError as exc:
            output["status"] = "not_assessed"
            output["reason"] = str(exc)
            return output
        except Exception as exc:
            output["status"] = "not_assessed"
            output["reason"] = f"evaluator_error:{type(exc).__name__}"
            return output
        if len(output["violations"]) >= 32:
            break
    output["status"] = "fail" if output["violations"] else "pass"
    output["proof_status"] = "counterexample_found" if output["violations"] else "tested_not_falsified"
    return output


__all__ = ["PROPERTY_KINDS", "TRANSITION_PROPERTY_KINDS", "PROPERTY_TEST_SCHEMA", "PropertyTestBudget", "PropertyTestError", "check_scalar_property", "check_transition_property"]
