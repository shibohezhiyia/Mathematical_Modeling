"""Unified uncertainty and calibration audit.

The model search layer can produce four different kinds of variation.  This
adapter keeps those layers separate from finite-sample calibration checks so
the UI/reporting code cannot accidentally turn a fit score into a calibrated
confidence claim.  Missing evaluation data is represented explicitly.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .calibration import assess_interval_calibration, assess_probability_calibration
from .uncertainty_propagation import propagate_layered_uncertainty


class UncertaintyAuditError(ValueError):
    pass


def build_uncertainty_audit(
    evaluations: Sequence[Mapping[str, Any]],
    *,
    unit_signature: str | None = None,
    interval: Mapping[str, Sequence[Any]] | None = None,
    probability: Mapping[str, Sequence[Any]] | None = None,
    calibration_min_points: int = 10,
) -> dict[str, Any]:
    """Combine layered spread with optional locked evaluation calibration.

    ``evaluations`` is never reused as a calibration sample.  The caller must
    provide independent ``actual/lower/upper`` or ``outcomes/probabilities``
    arrays when calibration is desired.  The returned status is ``partial``
    whenever a layer or requested calibration is unavailable.
    """
    if type(calibration_min_points) is not int or not 1 <= calibration_min_points <= 200_000:
        raise UncertaintyAuditError("invalid_calibration_min_points")
    try:
        propagation = propagate_layered_uncertainty(evaluations, unit_signature=unit_signature)
    except Exception as exc:
        if isinstance(exc, (TypeError, ValueError)):
            raise UncertaintyAuditError(f"propagation_failed:{type(exc).__name__}") from exc
        raise

    if interval is not None and not isinstance(interval, Mapping):
        raise UncertaintyAuditError("interval_must_be_mapping")
    if probability is not None and not isinstance(probability, Mapping):
        raise UncertaintyAuditError("probability_must_be_mapping")
    if interval is not None and probability is not None:
        raise UncertaintyAuditError("choose_one_calibration_kind")

    calibration: dict[str, Any]
    if interval is None and probability is None:
        calibration = {"status": "not_assessed", "reason": "independent_calibration_sample_required"}
    elif interval is not None:
        required = ("actual", "lower", "upper")
        if any(key not in interval for key in required):
            raise UncertaintyAuditError("interval_fields_required")
        actual = interval["actual"]
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            raise UncertaintyAuditError("calibration_sample_must_be_sequence")
        calibration = assess_interval_calibration(actual, interval["lower"], interval["upper"])
        if len(actual) < calibration_min_points and calibration["status"] == "assessed":
            calibration["status"] = "not_assessed"
            calibration["reason"] = "below_declared_calibration_min_points"
    else:
        required = ("outcomes", "probabilities")
        if any(key not in probability for key in required):
            raise UncertaintyAuditError("probability_fields_required")
        outcomes = probability["outcomes"]
        if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes)):
            raise UncertaintyAuditError("calibration_sample_must_be_sequence")
        calibration = assess_probability_calibration(outcomes, probability["probabilities"])
        if len(outcomes) < calibration_min_points and calibration["status"] == "assessed":
            calibration["status"] = "not_assessed"
            calibration["reason"] = "below_declared_calibration_min_points"

    layer_ready = propagation.get("status") == "ok"
    calibration_ready = calibration.get("status") in {"assessed", "within_declared_tolerance", "outside_declared_tolerance"}
    status = "assessed" if layer_ready and calibration_ready else "partial"
    return {
        "schema_version": "mathmodel.uncertainty-audit/v1",
        "status": status,
        "propagation": propagation,
        "calibration": calibration,
        "policy": "four_layer_empirical_spread_and_independent_finite_sample_calibration; neither is posterior_or_distribution_free_proof",
    }


__all__ = ["UncertaintyAuditError", "build_uncertainty_audit"]
