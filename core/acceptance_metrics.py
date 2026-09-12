"""Explicit denominators for accepted/error-prone benchmark outcomes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class AcceptanceMetricsError(ValueError):
    pass


def assess_acceptance_rates(results: Sequence[Mapping[str, Any]], ground_truth: Mapping[str, bool] | None = None) -> dict[str, Any]:
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise AcceptanceMetricsError("results_required")
    rows = [item for item in results if isinstance(item, Mapping)]
    malformed_result_count = len(results) - len(rows)
    if ground_truth is None:
        return {"schema_version": "mathmodel.acceptance-metrics/v1", "status": "not_assessed",
                "result_count": len(rows), "malformed_result_count": malformed_result_count, "denominators": {},
                "policy": "no_ground_truth_no_error_acceptance_rate"}
    if not isinstance(ground_truth, Mapping):
        raise AcceptanceMetricsError("ground_truth_must_be_mapping")
    normalized_truth = {str(case_id).strip(): value for case_id, value in ground_truth.items()
                        if isinstance(case_id, str) and case_id.strip() and type(value) is bool}
    invalid_truth_count = len(ground_truth) - len(normalized_truth)
    accepted = []
    missing_case_id_count = 0
    for item in rows:
        if item.get("status") not in {"approved", "accepted", "pass"}:
            continue
        case_id = item.get("case_id")
        if not isinstance(case_id, str) or not case_id.strip():
            missing_case_id_count += 1
            continue
        case_id = case_id.strip()
        if case_id not in accepted:
            accepted.append(case_id)
    known = [case_id for case_id in accepted if case_id in normalized_truth]
    wrong_accepted = [case_id for case_id in known if normalized_truth[case_id] is False]
    total_truth = list(normalized_truth)
    return {"schema_version": "mathmodel.acceptance-metrics/v1", "status": "assessed",
            "result_count": len(rows), "malformed_result_count": malformed_result_count,
            "accepted_count": len(accepted), "accepted_with_truth_count": len(known),
            "wrong_accepted_count": len(wrong_accepted),
            "error_acceptance_rate_among_accepted": (len(wrong_accepted) / len(known)) if known else None,
            "error_acceptance_rate_over_truth_cases": (len(wrong_accepted) / len(total_truth)) if total_truth else None,
            "denominators": {"accepted_with_truth": len(known), "truth_cases": len(total_truth)},
            "wrong_accepted_case_ids": wrong_accepted,
            "missing_case_id_count": missing_case_id_count,
            "invalid_ground_truth_count": invalid_truth_count,
            "policy": "rates_are_conditional_on_declared_ground_truth_not_missing_cases"}


__all__ = ["AcceptanceMetricsError", "assess_acceptance_rates"]
