"""Finite-sample calibration checks for prediction intervals and probabilities.

Calibration is deliberately separate from model fit.  A narrow interval can
fit the development data while missing future observations; these helpers make
that failure visible on a supplied evaluation sample.  They do not estimate a
posterior and do not guarantee coverage outside the checked distribution.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.calibration/v1"


class CalibrationError(ValueError):
    """Raised when calibration inputs are malformed or unsafe."""


def _finite(value: Any, code: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CalibrationError(code) from exc
    if not math.isfinite(result):
        raise CalibrationError(code)
    return result


def _vector(value: Any, name: str, *, maximum: int) -> list[float]:
    if isinstance(value, (str, bytes, Mapping)):
        raise CalibrationError(f"{name}_must_be_a_sequence")
    if not isinstance(value, Sequence):
        try:
            value = value.tolist() if hasattr(value, "tolist") else list(value)
        except (TypeError, ValueError) as exc:
            raise CalibrationError(f"{name}_must_be_a_sequence") from exc
    if not 1 <= len(value) <= maximum:
        raise CalibrationError(f"{name}_length_out_of_bounds")
    return [_finite(item, f"{name}_must_be_finite") for item in value]


def assess_interval_calibration(
    actual: Sequence[Any], lower: Sequence[Any], upper: Sequence[Any], *,
    nominal_coverage: float = 0.9, coverage_tolerance: float | None = None,
    max_points: int = 200_000,
) -> dict[str, Any]:
    """Assess empirical coverage and interval score on a locked evaluation set."""
    if type(max_points) is not int or not 1 <= max_points <= 1_000_000:
        raise CalibrationError("invalid_calibration_budget")
    nominal = _finite(nominal_coverage, "invalid_nominal_coverage")
    if not 0 < nominal < 1:
        raise CalibrationError("nominal_coverage_must_be_between_zero_and_one")
    alpha = 1.0 - nominal
    tolerance = (None if coverage_tolerance is None else _finite(coverage_tolerance, "invalid_coverage_tolerance"))
    if tolerance is not None and tolerance < 0:
        raise CalibrationError("coverage_tolerance_must_be_nonnegative")
    y = _vector(actual, "actual", maximum=max_points)
    lo = _vector(lower, "lower", maximum=max_points)
    hi = _vector(upper, "upper", maximum=max_points)
    if len(y) != len(lo) or len(y) != len(hi):
        raise CalibrationError("calibration_lengths_must_match")
    if any(left > right for left, right in zip(lo, hi)):
        raise CalibrationError("interval_lower_must_not_exceed_upper")
    covered = [left <= value <= right for value, left, right in zip(y, lo, hi)]
    lower_misses = [max(0.0, left - value) for value, left in zip(y, lo)]
    upper_misses = [max(0.0, value - right) for value, right in zip(y, hi)]
    widths = [right - left for left, right in zip(lo, hi)]
    score = [width + (2.0 / alpha) * low + (2.0 / alpha) * high
             for width, low, high in zip(widths, lower_misses, upper_misses)]
    ordered_widths = sorted(widths)
    middle = len(ordered_widths) // 2
    median_width = (ordered_widths[middle] if len(ordered_widths) % 2 else
                    (ordered_widths[middle - 1] + ordered_widths[middle]) / 2.0)
    empirical = sum(covered) / len(y)
    gap = empirical - nominal
    status = "not_assessed" if len(y) < 10 else "assessed"
    if status == "assessed" and tolerance is not None:
        status = "within_declared_tolerance" if abs(gap) <= tolerance else "outside_declared_tolerance"
    return {
        "schema_version": SCHEMA_VERSION, "status": status,
        "sample_count": len(y), "nominal_coverage": nominal,
        "empirical_coverage": float(empirical), "coverage_gap": float(gap),
        "covered_count": int(sum(covered)),
        "lower_miss_rate": float(sum(item > 0 for item in lower_misses) / len(y)),
        "upper_miss_rate": float(sum(item > 0 for item in upper_misses) / len(y)),
        "mean_width": float(sum(widths) / len(widths)),
        "median_width": float(median_width),
        "interval_score": float(sum(score) / len(score)),
        "coverage_tolerance": tolerance,
        "policy": "finite_sample_calibration_not_distribution_free_coverage_proof",
    }


def assess_probability_calibration(
    outcomes: Sequence[Any], probabilities: Sequence[Any], *, bins: int = 10,
    max_points: int = 200_000,
) -> dict[str, Any]:
    """Report Brier score and bounded reliability bins for binary outcomes."""
    if type(bins) is not int or not 2 <= bins <= 100:
        raise CalibrationError("invalid_calibration_bins")
    y = _vector(outcomes, "outcomes", maximum=max_points)
    p = _vector(probabilities, "probabilities", maximum=max_points)
    if len(y) != len(p):
        raise CalibrationError("calibration_lengths_must_match")
    if any(value not in (0.0, 1.0) for value in y):
        raise CalibrationError("outcomes_must_be_binary")
    if any(value < 0.0 or value > 1.0 for value in p):
        raise CalibrationError("probabilities_must_be_between_zero_and_one")
    rows = []
    weighted_error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [i for i, value in enumerate(p)
                   if (lower <= value < upper) or (index == bins - 1 and value == upper)]
        if not members:
            continue
        mean_probability = sum(p[i] for i in members) / len(members)
        observed_rate = sum(y[i] for i in members) / len(members)
        weighted_error += len(members) / len(y) * abs(mean_probability - observed_rate)
        rows.append({"bin": index, "lower": lower, "upper": upper,
                     "count": len(members), "mean_probability": mean_probability,
                     "observed_rate": observed_rate,
                     "absolute_gap": abs(mean_probability - observed_rate)})
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "not_assessed" if len(y) < 10 else "assessed",
        "sample_count": len(y), "brier_score": float(sum((a - q) ** 2 for a, q in zip(y, p)) / len(y)),
        "expected_calibration_error": float(weighted_error), "bins": rows,
        "bin_count": bins,
        "policy": "finite_sample_calibration_not_probability_truth_or_causal_effect",
    }


def assess_four_layer_uncertainty(
    scenarios: Mapping[str, Sequence[Any]], *, nominal_coverage: float | None = None,
    seed: int = 0, bootstrap_replicates: int = 2000,
) -> dict[str, Any]:
    """Decompose scenario variance without pretending scores are probabilities.

    ``scenarios`` maps each uncertainty layer to a vector of predictions under
    the same rows.  The returned contributions are conditional variance
    components (between-layer means plus within-layer variance), not a Bayesian
    posterior.  A small bootstrap interval makes sampling noise explicit.
    """
    names = ("semantic", "structure", "parameter", "numerical")
    if not isinstance(scenarios, Mapping):
        raise CalibrationError("scenarios_must_be_mapping")
    arrays = {}
    for name in names:
        if name not in scenarios:
            raise CalibrationError(f"missing_uncertainty_layer:{name}")
        arrays[name] = np.asarray(_vector(scenarios[name], name, maximum=200_000), dtype=float)
    lengths = {len(item) for item in arrays.values()}
    if len(lengths) != 1:
        raise CalibrationError("uncertainty_layer_lengths_must_match")
    n = lengths.pop()
    if n < 10:
        status = "not_assessed"
    else:
        status = "assessed"
    layer_means = {name: float(np.mean(values)) for name, values in arrays.items()}
    grand = float(np.mean(list(layer_means.values())))
    between = {name: float((layer_means[name] - grand) ** 2) for name in names}
    within = {name: float(np.var(values, ddof=1)) if n > 1 else 0.0 for name, values in arrays.items()}
    total = float(np.var(np.concatenate(list(arrays.values())), ddof=1)) if n > 1 else 0.0
    raw = {name: between[name] + within[name] / max(1, n) for name in names}
    denom = float(sum(raw.values()))
    shares = {name: (raw[name] / denom if denom > 0 else 0.0) for name in names}
    rng = np.random.default_rng(int(seed))
    reps = max(200, min(int(bootstrap_replicates), 10_000))
    boot = np.empty((reps, len(names)), dtype=float)
    for i in range(reps):
        idx = rng.integers(0, n, size=n)
        means = {name: float(np.mean(arrays[name][idx])) for name in names}
        center = float(np.mean(list(means.values())))
        vals = np.array([(means[name] - center) ** 2 + float(np.var(arrays[name][idx], ddof=1)) / max(1, n) for name in names])
        boot[i] = vals / vals.sum() if vals.sum() > 0 else 0.0
    intervals = {name: [float(np.quantile(boot[:, j], 0.025)), float(np.quantile(boot[:, j], 0.975))]
                 for j, name in enumerate(names)}
    result = {"schema_version": "mathmodel.four-layer-calibration/v1", "status": status,
              "sample_count": int(n), "layer_means": layer_means,
              "variance_components": raw, "variance_share": shares,
              "variance_total": total, "bootstrap_share_ci_95": intervals,
              "bootstrap_replicates": reps, "seed": int(seed),
              "policy": "conditional_scenario_variance_not_posterior_probability"}
    if nominal_coverage is not None:
        nominal = _finite(nominal_coverage, "invalid_nominal_coverage")
        if not 0 < nominal < 1:
            raise CalibrationError("nominal_coverage_must_be_between_zero_and_one")
        result["coverage_note"] = "提供 actual/lower/upper 后调用 assess_interval_calibration 单独验证覆盖率"
        result["nominal_coverage"] = nominal
    return result


__all__ = ["SCHEMA_VERSION", "CalibrationError", "assess_interval_calibration", "assess_probability_calibration", "assess_four_layer_uncertainty"]
