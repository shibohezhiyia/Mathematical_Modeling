"""Finite evidence certificate for a model conclusion.

This is a report-level contract: it makes boundaries, residuals, constraints
and counterexamples impossible to omit.  It deliberately never labels finite
checks as a theorem or as real-world correctness.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class ConclusionCertificateError(ValueError):
    pass


def _bounded_rows(value: Any, name: str, maximum: int = 2048) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) > maximum:
        raise ConclusionCertificateError(f"{name}_must_be_bounded_sequence")
    rows = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ConclusionCertificateError(f"{name}_row_must_be_mapping")
        rows.append(dict(item))
    return rows


def build_conclusion_certificate(*, applicability_boundary: Sequence[str] | str,
                                 residuals: Sequence[Mapping[str, Any]],
                                 constraint_checks: Sequence[Mapping[str, Any]],
                                 counterexamples: Sequence[Mapping[str, Any]] = (),
                                 assumptions: Sequence[str] = ()) -> dict[str, Any]:
    """Build an auditable, finite certificate with conservative status labels."""
    if isinstance(applicability_boundary, str):
        boundary = [applicability_boundary.strip()] if applicability_boundary.strip() else []
    elif isinstance(applicability_boundary, Sequence) and not isinstance(applicability_boundary, (bytes, str)):
        boundary = [str(item).strip() for item in applicability_boundary if str(item).strip()]
    else:
        raise ConclusionCertificateError("applicability_boundary_required")
    if not boundary or len(boundary) > 64:
        raise ConclusionCertificateError("applicability_boundary_required")
    residual_rows = _bounded_rows(residuals, "residuals")
    constraint_rows = _bounded_rows(constraint_checks, "constraint_checks")
    counter_rows = _bounded_rows(counterexamples, "counterexamples")
    assumption_rows = [str(item).strip() for item in assumptions if str(item).strip()]
    if len(assumption_rows) > 128:
        raise ConclusionCertificateError("assumptions_too_large")
    for row in residual_rows:
        value = row.get("value", row.get("rmse"))
        try:
            finite = value is not None and math.isfinite(float(value))
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            raise ConclusionCertificateError("residual_value_required")
    for row in constraint_rows:
        violation = row.get("violation", row.get("max_violation"))
        try:
            finite = violation is not None and math.isfinite(float(violation)) and float(violation) >= 0
        except (TypeError, ValueError, OverflowError):
            finite = False
        if not finite:
            raise ConclusionCertificateError("constraint_violation_required")
    status = "counterexample_found" if counter_rows else "tested_not_falsified"
    return {
        "schema_version": "mathmodel.conclusion-certificate/v1",
        "status": status,
        "applicability_boundary": boundary,
        "assumptions": assumption_rows,
        "residuals": residual_rows,
        "constraint_checks": constraint_rows,
        "counterexamples": counter_rows,
        "policy": "finite_residual_constraint_and_counterexample_record; not_a_global_proof_or_real_world_correctness_claim",
    }


__all__ = ["ConclusionCertificateError", "build_conclusion_certificate"]
