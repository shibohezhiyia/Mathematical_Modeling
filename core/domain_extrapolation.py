"""Bounded out-of-domain stress checks for executable scalar models.

The check intentionally separates evidence inside the declared domain from a
small, deterministic extension.  A failure outside the domain is not proof of
real-world invalidity, while passing a finite probe is not a global guarantee.
The evaluator should be a trusted/sandboxed callable; this module does not
pretend to provide process isolation.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import random
from typing import Any, Callable, Mapping


SCHEMA_VERSION = "mathmodel.domain-extrapolation/v1"


class DomainExtrapolationError(ValueError):
    """Raised when the finite stress-check contract is invalid."""


def _domain(domain: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    if not isinstance(domain, Mapping) or not 1 <= len(domain) <= 8:
        raise DomainExtrapolationError("domain_must_bind_1_to_8_variables")
    result: dict[str, tuple[float, float]] = {}
    for name, interval in domain.items():
        if not isinstance(name, str) or not name:
            raise DomainExtrapolationError("domain_variable_name_must_be_nonempty")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise DomainExtrapolationError(f"invalid_domain:{name}")
        lower, upper = float(interval[0]), float(interval[1])
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise DomainExtrapolationError(f"invalid_domain:{name}")
        result[name] = (lower, upper)
    return result


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DomainExtrapolationError("evaluator_must_return_finite_scalar") from exc
    if not math.isfinite(number):
        raise DomainExtrapolationError("evaluator_must_return_finite_scalar")
    return number


def _probe_points(bounds: dict[str, tuple[float, float]], *, count: int, seed: int,
                  outside: bool, expansion_factor: float) -> list[dict[str, float]]:
    rng = random.Random(seed + (1 if outside else 0))
    names = list(bounds)
    points = []
    for index in range(count):
        point = {}
        for name in names:
            lower, upper = bounds[name]
            span = upper - lower
            point[name] = rng.uniform(lower, upper)
        if outside:
            name = names[index % len(names)]
            lower, upper = bounds[name]
            span = upper - lower
            if index % 2:
                point[name] = lower - rng.uniform(0.05, expansion_factor) * span
            else:
                point[name] = upper + rng.uniform(0.05, expansion_factor) * span
        points.append(point)
    return points


def _evaluate(evaluate: Callable[[Mapping[str, float]], Any], points: list[dict[str, float]],
              *, lower: float | None, upper: float | None, phase: str) -> dict[str, Any]:
    values: list[float] = []
    violations: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, point in enumerate(points):
        try:
            value = _number(evaluate(dict(point)))
        except DomainExtrapolationError as exc:
            failures.append({"phase": phase, "probe": index, "point": point, "reason": str(exc)})
            if len(failures) >= 32:
                break
            continue
        except Exception as exc:  # evaluator implementation detail is not an evidence claim
            failures.append({"phase": phase, "probe": index, "point": point,
                             "reason": f"evaluator_error:{type(exc).__name__}"})
            if len(failures) >= 32:
                break
            continue
        values.append(value)
        if lower is not None and value < lower:
            violations.append({"phase": phase, "probe": index, "point": point,
                               "value": value, "bound": "lower", "expected": lower})
        if upper is not None and value > upper:
            violations.append({"phase": phase, "probe": index, "point": point,
                               "value": value, "bound": "upper", "expected": upper})
        if len(violations) >= 32:
            break
    return {"phase": phase, "points": points, "values": values,
            "failures": failures, "bound_violations": violations}


def assess_domain_extrapolation(
    evaluate: Callable[[Mapping[str, float]], Any],
    *,
    domain: Mapping[str, Any],
    expansion_factor: float = 1.5,
    probe_count: int = 16,
    seed: int = 42,
    output_lower: float | None = None,
    output_upper: float | None = None,
) -> dict[str, Any]:
    """Compare finite probes in the declared domain with a small extension."""
    bounds = _domain(domain)
    if type(probe_count) is not int or not 2 <= probe_count <= 128:
        raise DomainExtrapolationError("probe_count_must_be_between_2_and_128")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise DomainExtrapolationError("seed_must_be_an_unsigned_32_bit_integer")
    if (type(expansion_factor) not in (int, float) or not math.isfinite(float(expansion_factor))
            or not 1.01 <= float(expansion_factor) <= 10):
        raise DomainExtrapolationError("expansion_factor_must_be_between_1.01_and_10")
    for bound in (output_lower, output_upper):
        if bound is not None and (type(bound) not in (int, float) or not math.isfinite(float(bound))):
            raise DomainExtrapolationError("output_bounds_must_be_finite")
    if output_lower is not None and output_upper is not None and output_lower > output_upper:
        raise DomainExtrapolationError("output_lower_must_not_exceed_output_upper")
    in_domain = _evaluate(evaluate, _probe_points(bounds, count=probe_count, seed=seed,
                                                  outside=False, expansion_factor=float(expansion_factor)),
                          lower=output_lower, upper=output_upper, phase="declared_domain")
    out_domain = _evaluate(evaluate, _probe_points(bounds, count=probe_count, seed=seed,
                                                   outside=True, expansion_factor=float(expansion_factor)),
                           lower=output_lower, upper=output_upper, phase="extended_domain")
    all_failures = [*in_domain["failures"], *out_domain["failures"]]
    all_violations = [*in_domain["bound_violations"], *out_domain["bound_violations"]]
    in_values, out_values = in_domain["values"], out_domain["values"]
    in_scale = max((abs(value) for value in in_values), default=0.0)
    out_scale = max((abs(value) for value in out_values), default=0.0)
    ratio = float(out_scale / max(in_scale, 1e-12)) if out_values else None
    if all_failures or all_violations:
        status = "counterexample_found"
    elif ratio is not None and ratio > 10:
        status = "domain_shift_signal"
    else:
        status = "no_falsification_in_tested_extension"
    fingerprint = sha256(json.dumps({"domain": bounds, "in": in_domain["points"],
                                     "out": out_domain["points"], "seed": seed},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "probe_count": probe_count,
        "expansion_factor": float(expansion_factor),
        "seed": seed,
        "domain_fingerprint": fingerprint,
        "declared_domain": in_domain,
        "extended_domain": out_domain,
        "max_absolute_scale_ratio": ratio,
        "counterexamples": [*all_failures, *all_violations][:32],
        "policy": "finite_domain_stress_not_global_extrapolation_proof",
        "evidence_scope": "tested_points_only",
    }


__all__ = ["SCHEMA_VERSION", "DomainExtrapolationError", "assess_domain_extrapolation"]
