import pytest

from core.calibration import (
    CalibrationError,
    assess_interval_calibration,
    assess_probability_calibration,
)


def test_interval_calibration_reports_coverage_and_score():
    actual = list(range(10))
    report = assess_interval_calibration(actual, [-1] * 10, [10] * 10,
                                         nominal_coverage=0.9, coverage_tolerance=0.11)
    assert report["status"] == "within_declared_tolerance"
    assert report["empirical_coverage"] == pytest.approx(1.0)
    assert report["covered_count"] == 10
    assert report["interval_score"] == pytest.approx(11.0)


def test_interval_calibration_exposes_undercoverage_and_small_sample_status():
    report = assess_interval_calibration([0] * 10, [1] * 10, [2] * 10,
                                         nominal_coverage=0.9, coverage_tolerance=0.05)
    assert report["status"] == "outside_declared_tolerance"
    assert report["lower_miss_rate"] == pytest.approx(1.0)
    small = assess_interval_calibration([0], [-1], [1])
    assert small["status"] == "not_assessed"


def test_probability_calibration_reports_brier_and_reliability_bins():
    outcomes = [0, 0, 1, 1] * 3
    probabilities = [0.1, 0.2, 0.8, 0.9] * 3
    report = assess_probability_calibration(outcomes, probabilities, bins=4)
    assert report["status"] == "assessed"
    assert report["brier_score"] == pytest.approx(0.025)
    assert report["expected_calibration_error"] > 0
    assert sum(row["count"] for row in report["bins"]) == 12


def test_calibration_rejects_invalid_bounds_probability_and_lengths():
    with pytest.raises(CalibrationError, match="interval_lower"):
        assess_interval_calibration([1], [2], [0])
    with pytest.raises(CalibrationError, match="probabilities"):
        assess_probability_calibration([0, 1], [-0.1, 1.1])
    with pytest.raises(CalibrationError, match="lengths"):
        assess_probability_calibration([0, 1], [0.5])
