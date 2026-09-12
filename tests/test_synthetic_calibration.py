import math

import pytest

from core.synthetic_calibration import (
    SyntheticCalibrationError,
    assess_synthetic_interval_calibration,
)


def _normal_interval(rng, nominal, replicate):
    del replicate
    # The protocol intentionally uses a deterministic known generator.  The
    # interval is a simple normal approximation around zero.
    z = {0.8: 1.2816, 0.9: 1.6449, 0.95: 1.96}[round(nominal, 2)]
    actual = [rng.gauss(0.0, 1.0) for _ in range(80)]
    return actual, [-z] * len(actual), [z] * len(actual)


def test_synthetic_calibration_reports_finite_coverage_diagnostics():
    report = assess_synthetic_interval_calibration(
        _normal_interval, nominal_coverages=(0.8, 0.9, 0.95),
        replicates=12, seed=7,
    )
    assert report["status"] == "assessed"
    assert len(report["rows"]) == 3
    assert all(row["sample_count"] == 960 for row in report["rows"])
    assert all(math.isfinite(row["coverage_standard_error"]) for row in report["rows"])
    assert "not_real_distribution" in report["policy"]


def test_synthetic_calibration_rejects_bad_protocol_and_surfaces_partial_runs():
    with pytest.raises(SyntheticCalibrationError, match="replicates"):
        assess_synthetic_interval_calibration(_normal_interval, replicates=1)

    def broken(rng, nominal, replicate):
        del rng, nominal
        if replicate == 0:
            return [0.0], [0.0], [0.0]
        return [float("nan")], [0.0], [1.0]

    report = assess_synthetic_interval_calibration(broken, nominal_coverages=(0.9,), replicates=2)
    assert report["status"] == "partial"
    assert report["rows"][0]["failed_replicates"] == 1
