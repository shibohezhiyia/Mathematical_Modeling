"""Bounded bootstrap diagnostics for parameter uncertainty.

The caller supplies a fit function that receives resampled observation indices.
The output is an empirical stability report conditional on the current data,
fit routine, and resampling scheme; it is not a posterior distribution or a
proof that the parameters are globally identifiable.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
import random
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "mathmodel.parameter-bootstrap/v1"


class ParameterUncertaintyError(ValueError):
    """Raised when the bootstrap contract is invalid."""


def profile_likelihood_uncertainty(
    objective: Callable[[float], Any], parameter_grid: Sequence[float], *,
    delta_threshold: float = 1.92, max_points: int = 256,
) -> dict[str, Any]:
    """Evaluate a bounded one-parameter profile-likelihood curve.

    ``objective`` should be a negative log-likelihood (or another explicitly
    documented loss). The returned interval is an empirical level-set of the
    supplied grid; it is not a confidence or posterior interval unless the
    caller separately supplies the required statistical assumptions.
    """
    if not isinstance(parameter_grid, Sequence) or isinstance(parameter_grid, (str, bytes)):
        raise ParameterUncertaintyError("parameter_grid_must_be_a_sequence")
    if type(max_points) is not int or not 3 <= max_points <= 2_048:
        raise ParameterUncertaintyError("invalid_profile_grid_budget")
    if not 3 <= len(parameter_grid) <= max_points:
        raise ParameterUncertaintyError("profile_grid_size_out_of_bounds")
    if type(delta_threshold) not in (int, float) or not math.isfinite(float(delta_threshold)) or delta_threshold <= 0:
        raise ParameterUncertaintyError("invalid_profile_delta_threshold")
    points: list[tuple[float, float]] = []
    for value in parameter_grid:
        parameter = _number(value)
        try:
            score = _number(objective(parameter))
        except ParameterUncertaintyError:
            raise
        except Exception as exc:
            return {"schema_version": "mathmodel.profile-likelihood/v1", "status": "not_assessed",
                    "reason": f"objective_error:{type(exc).__name__}", "evaluated_points": len(points),
                    "policy": "bounded_empirical_profile_not_confidence_proof"}
        points.append((parameter, score))
    minimum = min(score for _, score in points)
    delta = [(parameter, score - minimum) for parameter, score in points]
    accepted = [parameter for parameter, difference in delta if difference <= float(delta_threshold)]
    ordered = sorted(points)
    return {
        "schema_version": "mathmodel.profile-likelihood/v1",
        "status": "assessed" if accepted else "not_assessed",
        "evaluated_points": len(points),
        "minimum": {"parameter": min(points, key=lambda item: item[1])[0], "objective": minimum},
        "delta_threshold": float(delta_threshold),
        "profile": [{"parameter": parameter, "objective": score, "delta": score - minimum}
                     for parameter, score in ordered],
        "interval": {"lower": min(accepted), "upper": max(accepted)} if accepted else None,
        "policy": "bounded_empirical_profile_not_confidence_proof",
    }


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ParameterUncertaintyError("fit_must_return_finite_parameters") from exc
    if not math.isfinite(number):
        raise ParameterUncertaintyError("fit_must_return_finite_parameters")
    return number


def bootstrap_parameter_uncertainty(
    fit: Callable[[Sequence[int]], Mapping[str, Any]],
    *,
    sample_count: int,
    bootstrap_replicates: int = 64,
    seed: int = 42,
    max_failures: int = 16,
) -> dict[str, Any]:
    """Resample observation indices and summarize stable parameter estimates."""
    if type(sample_count) is not int or not 4 <= sample_count <= 100_000:
        raise ParameterUncertaintyError("sample_count_must_be_between_4_and_100000")
    if type(bootstrap_replicates) is not int or not 8 <= bootstrap_replicates <= 512:
        raise ParameterUncertaintyError("bootstrap_replicates_must_be_between_8_and_512")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ParameterUncertaintyError("seed_must_be_an_unsigned_32_bit_integer")
    if type(max_failures) is not int or not 0 <= max_failures <= 512:
        raise ParameterUncertaintyError("invalid_bootstrap_failure_budget")
    rng = random.Random(seed)
    estimates: list[dict[str, float]] = []
    failures: list[dict[str, Any]] = []
    expected_names: tuple[str, ...] | None = None
    for replicate in range(bootstrap_replicates):
        indices = tuple(rng.randrange(sample_count) for _ in range(sample_count))
        try:
            raw = fit(indices)
            if not isinstance(raw, Mapping) or not raw:
                raise ParameterUncertaintyError("fit_must_return_nonempty_mapping")
            normalized_raw: dict[str, Any] = {}
            for name, value in raw.items():
                if type(name) is not str or not name.strip() or len(name.strip()) > 128:
                    raise ParameterUncertaintyError("parameter_name_must_be_nonempty")
                normalized_name = name.strip()
                if normalized_name in normalized_raw:
                    raise ParameterUncertaintyError("parameter_names_must_be_unique")
                normalized_raw[normalized_name] = value
            current_names = tuple(sorted(normalized_raw))
            if not current_names:
                raise ParameterUncertaintyError("parameter_name_must_be_nonempty")
            if expected_names is None:
                expected_names = current_names
            if current_names != expected_names:
                raise ParameterUncertaintyError("fit_parameter_schema_changed_across_replicates")
            estimates.append({name: _number(normalized_raw[name]) for name in expected_names})
        except ParameterUncertaintyError as exc:
            failures.append({"replicate": replicate, "reason": str(exc)})
        except Exception as exc:  # fitting internals are not evidence claims
            failures.append({"replicate": replicate, "reason": f"fit_error:{type(exc).__name__}"})
        if len(failures) > max_failures:
            break
    if expected_names is None or not estimates:
        return {
            "schema_version": SCHEMA_VERSION, "status": "not_assessed",
            "sample_count": sample_count, "requested_replicates": bootstrap_replicates,
            "successful_replicates": 0, "failures": failures[:32], "seed": seed,
            "policy": "conditional_empirical_bootstrap_not_posterior",
        }
    if len(failures) > max_failures:
        status = "fit_failure_budget_exhausted"
    elif len(estimates) < max(8, bootstrap_replicates // 2):
        status = "few_successful_replicates"
    else:
        status = "assessed"
    summary: list[dict[str, Any]] = []
    for name in expected_names:
        values = sorted(item[name] for item in estimates)
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / len(values)
        summary.append({
            "parameter": name,
            "mean": mean,
            "std": math.sqrt(max(variance, 0.0)),
            "quantiles": {"q05": _quantile(values, 0.05), "q50": _quantile(values, 0.5),
                          "q95": _quantile(values, 0.95)},
            "min": values[0], "max": values[-1],
            "interpretation": "empirical_resampling_variability_not_confidence_proof",
        })
    return {
        "schema_version": SCHEMA_VERSION, "status": status,
        "sample_count": sample_count, "requested_replicates": bootstrap_replicates,
        "successful_replicates": len(estimates), "failed_replicates": len(failures),
        "parameters": summary, "failures": failures[:32], "seed": seed,
        "resampling": "observation_indices_with_replacement",
        "design_fingerprint": sha256(json.dumps({"n": sample_count, "seed": seed},
                                                  sort_keys=True).encode()).hexdigest(),
        "policy": "conditional_empirical_bootstrap_not_posterior",
    }


def _quantile(values: list[float], probability: float) -> float:
    if len(values) == 1:
        return values[0]
    position = probability * (len(values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + fraction * (values[upper] - values[lower])


__all__ = ["SCHEMA_VERSION", "ParameterUncertaintyError", "bootstrap_parameter_uncertainty",
           "profile_likelihood_uncertainty"]
