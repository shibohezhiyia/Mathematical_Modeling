"""Bounded, model-agnostic global sensitivity screening.

The implementation uses deterministic permutation total effects over a finite
parameter design.  It is useful for deciding which parameters or assumptions
deserve more computation, but it is not a causal effect, a posterior
probability, or a proof of global identifiability.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import random
from typing import Any, Callable, Mapping


SCHEMA_VERSION = "mathmodel.global-sensitivity/v1"


class SensitivityAnalysisError(ValueError):
    """Raised when a sensitivity-screening contract is invalid."""


def _bounds(value: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    if not isinstance(value, Mapping) or not 1 <= len(value) <= 16:
        raise SensitivityAnalysisError("parameter_bounds_must_bind_1_to_16_parameters")
    result: dict[str, tuple[float, float]] = {}
    for name, interval in value.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise SensitivityAnalysisError("parameter_name_must_be_nonempty")
        name = name.strip()
        if name in result:
            raise SensitivityAnalysisError("parameter_names_must_be_unique")
        if not isinstance(interval, (list, tuple)) or len(interval) != 2:
            raise SensitivityAnalysisError(f"invalid_parameter_bounds:{name}")
        if any(type(item) is bool for item in interval):
            raise SensitivityAnalysisError(f"invalid_parameter_bounds:{name}")
        low, high = float(interval[0]), float(interval[1])
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise SensitivityAnalysisError(f"invalid_parameter_bounds:{name}")
        result[name] = (low, high)
    return result


def _finite_scalar(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SensitivityAnalysisError("evaluator_must_return_finite_scalar") from exc
    if not math.isfinite(result):
        raise SensitivityAnalysisError("evaluator_must_return_finite_scalar")
    return result


def assess_global_sensitivity(
    evaluate: Callable[[Mapping[str, float]], Any],
    *,
    parameter_bounds: Mapping[str, Any],
    sample_count: int = 32,
    seed: int = 42,
    max_evaluations: int | None = None,
) -> dict[str, Any]:
    """Estimate finite-design permutation total effects for scalar output."""
    bounds = _bounds(parameter_bounds)
    names = list(bounds)
    if type(sample_count) is not int or not 4 <= sample_count <= 128:
        raise SensitivityAnalysisError("sample_count_must_be_between_4_and_128")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise SensitivityAnalysisError("seed_must_be_an_unsigned_32_bit_integer")
    required_evaluations = sample_count * (len(names) + 1)
    if max_evaluations is None:
        max_evaluations = required_evaluations
    if type(max_evaluations) is not int or not 1 <= max_evaluations <= 4096:
        raise SensitivityAnalysisError("invalid_sensitivity_evaluation_budget")
    if max_evaluations < required_evaluations:
        return {
            "schema_version": SCHEMA_VERSION, "status": "not_assessed",
            "reason": "sensitivity_evaluation_budget_exhausted",
            "required_evaluations": required_evaluations, "max_evaluations": max_evaluations,
            "parameter_names": names, "policy": "finite_permutation_screen_not_global_proof",
        }
    rng = random.Random(seed)
    design = [{name: rng.uniform(*bounds[name]) for name in names} for _ in range(sample_count)]
    values: list[float] = []
    failures: list[dict[str, Any]] = []

    def evaluate_point(point: Mapping[str, float], phase: str, index: int) -> float | None:
        try:
            return _finite_scalar(evaluate(dict(point)))
        except SensitivityAnalysisError as exc:
            failures.append({"phase": phase, "index": index, "reason": str(exc)})
            return None
        except Exception as exc:  # evaluator internals are not evidence claims
            failures.append({"phase": phase, "index": index, "reason": f"evaluator_error:{type(exc).__name__}"})
            return None

    for index, point in enumerate(design):
        value = evaluate_point(point, "base", index)
        if value is None:
            return _not_assessed(names, required_evaluations, max_evaluations, failures, seed)
        values.append(value)
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 1e-24:
        return {
            "schema_version": SCHEMA_VERSION, "status": "no_output_variation",
            "parameter_names": names, "sample_count": sample_count,
            "evaluations": required_evaluations, "output_variance": variance,
            "effects": [{"parameter": name, "total_effect": 0.0, "rank": None,
                          "interpretation": "no_finite_output_variation"} for name in names],
            "failures": failures, "seed": seed,
            "design_fingerprint": _fingerprint(design, seed),
            "policy": "finite_permutation_screen_not_global_proof",
        }
    effects: list[dict[str, Any]] = []
    for parameter_index, name in enumerate(names):
        permuted = list(range(sample_count))
        random.Random(seed + 1009 * (parameter_index + 1)).shuffle(permuted)
        squared_differences = []
        signs = []
        for index, point in enumerate(design):
            candidate = dict(point)
            candidate[name] = design[permuted[index]][name]
            value = evaluate_point(candidate, f"permutation:{name}", index)
            if value is None:
                return _not_assessed(names, required_evaluations, max_evaluations, failures, seed)
            squared_differences.append((values[index] - value) ** 2)
            signs.append(values[index] - value)
        effect = float(sum(squared_differences) / (2.0 * variance))
        positive = sum(sign > 0 for sign in signs)
        negative = sum(sign < 0 for sign in signs)
        effects.append({"parameter": name, "total_effect": effect,
                        "direction_consistency": max(positive, negative) / len(signs),
                        "nonzero_direction_fraction": (positive + negative) / len(signs),
                        "rank": None,
                        "interpretation": "finite_permutation_effect_not_causal"})
    order = sorted(range(len(effects)), key=lambda index: (-effects[index]["total_effect"], names[index]))
    for rank, index in enumerate(order, start=1):
        effects[index]["rank"] = rank
    return {
        "schema_version": SCHEMA_VERSION, "status": "assessed",
        "parameter_names": names, "sample_count": sample_count,
        "evaluations": required_evaluations, "output_variance": variance,
        "effects": effects, "failures": failures, "seed": seed,
        "design_fingerprint": _fingerprint(design, seed),
        "policy": "finite_permutation_screen_not_global_proof",
    }


def _not_assessed(names, required, maximum, failures, seed):
    return {"schema_version": SCHEMA_VERSION, "status": "not_assessed",
            "reason": "nonfinite_or_failed_evaluation", "parameter_names": names,
            "required_evaluations": required, "max_evaluations": maximum,
            "failures": failures[:32], "seed": seed,
            "policy": "finite_permutation_screen_not_global_proof"}


def _fingerprint(design: list[dict[str, float]], seed: int) -> str:
    return sha256(json.dumps({"design": design, "seed": seed}, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")).hexdigest()


__all__ = ["SCHEMA_VERSION", "SensitivityAnalysisError", "assess_global_sensitivity"]
