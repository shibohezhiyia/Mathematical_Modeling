"""Synthetic-truth calibration protocol.

This module evaluates interval or probability outputs on data generated from a
known process.  It is deliberately a *calibration diagnostic*, not a proof of
coverage on real data: the generator defines the distribution being checked.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from typing import Any

from .calibration import CalibrationError, assess_interval_calibration


class SyntheticCalibrationError(ValueError):
    """Raised when a synthetic calibration protocol is malformed."""


IntervalBuilder = Callable[[random.Random, float, int], tuple[Sequence[Any], Sequence[Any], Sequence[Any]]]


def assess_synthetic_interval_calibration(
    builder: IntervalBuilder,
    *,
    nominal_coverages: Sequence[float] = (0.8, 0.9, 0.95),
    replicates: int = 100,
    seed: int = 0,
    max_points: int = 200_000,
) -> dict[str, Any]:
    """Run a bounded coverage check against a known synthetic generator.

    ``builder(rng, nominal, replicate)`` must return ``(actual, lower, upper)``
    for one independent replicate.  Individual replicates are scored first;
    aggregate coverage is computed by pooling the covered indicators.  The
    result records the generator protocol and finite-sample uncertainty, while
    explicitly avoiding posterior or distribution-free claims.
    """
    if not callable(builder):
        raise SyntheticCalibrationError("builder_must_be_callable")
    if type(replicates) is not int or not 2 <= replicates <= 10_000:
        raise SyntheticCalibrationError("replicates_out_of_bounds")
    if type(seed) is not int:
        raise SyntheticCalibrationError("seed_must_be_integer")
    try:
        levels = [float(level) for level in nominal_coverages]
    except (TypeError, ValueError) as exc:
        raise SyntheticCalibrationError("nominal_coverages_must_be_a_sequence") from exc
    if not levels or any(not math.isfinite(level) or not 0 < level < 1 for level in levels):
        raise SyntheticCalibrationError("nominal_coverage_out_of_bounds")
    if len(set(levels)) != len(levels):
        raise SyntheticCalibrationError("nominal_coverages_must_be_unique")
    if type(max_points) is not int or not 1 <= max_points <= 1_000_000:
        raise SyntheticCalibrationError("invalid_calibration_budget")

    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for nominal in levels:
        covered_total = 0
        point_total = 0
        widths: list[float] = []
        failures = 0
        for replicate in range(replicates):
            try:
                actual, lower, upper = builder(rng, nominal, replicate)
                report = assess_interval_calibration(
                    actual, lower, upper, nominal_coverage=nominal,
                    max_points=max_points,
                )
            except (CalibrationError, TypeError, ValueError) as exc:
                failures += 1
                continue
            covered_total += int(report["covered_count"])
            point_total += int(report["sample_count"])
            widths.append(float(report["mean_width"]))
        if point_total == 0:
            raise SyntheticCalibrationError(f"builder_returned_no_valid_points:{nominal:g}")
        empirical = covered_total / point_total
        gap = empirical - nominal
        standard_error = math.sqrt(max(empirical * (1.0 - empirical), 0.0) / point_total)
        rows.append({
            "nominal_coverage": nominal,
            "sample_count": point_total,
            "replicates": replicates,
            "failed_replicates": failures,
            "empirical_coverage": empirical,
            "coverage_gap": gap,
            "coverage_standard_error": standard_error,
            "mean_interval_width": sum(widths) / len(widths),
            "finite_sample_warning": point_total < 100,
        })
    return {
        "schema_version": "mathmodel.synthetic_calibration/v1",
        "status": "assessed" if all(row["failed_replicates"] == 0 for row in rows) else "partial",
        "seed": seed,
        "rows": rows,
        "max_absolute_coverage_gap": max(abs(row["coverage_gap"]) for row in rows),
        "policy": "synthetic_generator_calibration_not_real_distribution_coverage_proof",
    }


__all__ = ["SyntheticCalibrationError", "assess_synthetic_interval_calibration"]
