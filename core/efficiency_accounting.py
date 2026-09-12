"""Auditable accounting for search-time savings.

This module records *why* a candidate did not reach an expensive evaluator and
checks that acceleration did not silently remove independent, rare-event, or
slow-convergence checks.  It never turns fewer evaluations into a quality claim;
the returned gate only says whether a speed comparison is eligible to report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any


class EfficiencyAccountingError(ValueError):
    pass


def _ids(values: Sequence[Mapping[str, Any]], label: str) -> list[str]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise EfficiencyAccountingError(f"{label}_must_be_a_sequence")
    out: list[str] = []
    for item in values:
        if not isinstance(item, Mapping):
            raise EfficiencyAccountingError(f"{label}_items_must_be_objects")
        identifier = item.get("id")
        if not isinstance(identifier, (str, int)) or isinstance(identifier, bool) or not str(identifier).strip():
            raise EfficiencyAccountingError(f"{label}_id_required")
        out.append(str(identifier).strip())
    if len(set(out)) != len(out):
        raise EfficiencyAccountingError(f"{label}_ids_must_be_unique")
    return out


def build_efficiency_account(
    cases: Sequence[Mapping[str, Any]],
    *,
    expensive_evaluations: Sequence[Mapping[str, Any]],
    independent_checks: Sequence[Mapping[str, Any]],
    rare_event_ids: Sequence[str] = (),
    slow_convergence_ids: Sequence[str] = (),
    minimum_independent_rate: float = 1.0,
) -> dict[str, Any]:
    """Create a conservative, case-level acceleration account.

    ``cases`` is the pre-registered denominator.  Every case must be present in
    both evaluation and independent-check records; a case may be marked
    ``skipped`` only with a bounded reason (e.g. static rejection or budget).
    Rare and slow cases are never inferred from a label: their IDs must be
    explicitly registered in the denominator.
    """
    case_ids = _ids(cases, "cases")
    eval_ids = _ids(expensive_evaluations, "expensive_evaluations")
    check_ids = _ids(independent_checks, "independent_checks")
    if set(eval_ids) != set(case_ids):
        raise EfficiencyAccountingError("expensive_evaluations_must_cover_all_cases")
    if set(check_ids) != set(case_ids):
        raise EfficiencyAccountingError("independent_checks_must_cover_all_cases")
    if not isinstance(rare_event_ids, Sequence) or isinstance(rare_event_ids, (str, bytes)):
        raise EfficiencyAccountingError("rare_event_ids_must_be_a_sequence")
    if not isinstance(slow_convergence_ids, Sequence) or isinstance(slow_convergence_ids, (str, bytes)):
        raise EfficiencyAccountingError("slow_convergence_ids_must_be_a_sequence")
    rare = {str(item).strip() for item in rare_event_ids if str(item).strip()}
    slow = {str(item).strip() for item in slow_convergence_ids if str(item).strip()}
    unknown_special = (rare | slow) - set(case_ids)
    if unknown_special:
        raise EfficiencyAccountingError("special_case_id_not_in_denominator")
    if not isinstance(minimum_independent_rate, (int, float)) or isinstance(minimum_independent_rate, bool):
        raise EfficiencyAccountingError("minimum_independent_rate_must_be_numeric")
    minimum_independent_rate = float(minimum_independent_rate)
    if not math.isfinite(minimum_independent_rate) or not 0 <= minimum_independent_rate <= 1:
        raise EfficiencyAccountingError("minimum_independent_rate_out_of_range")

    eval_by_id = {str(item["id"]): item for item in expensive_evaluations}
    checks_by_id = {str(item["id"]): item for item in independent_checks}
    bounded_skip_reasons = {"static_rejection", "low_fidelity_pruned", "budget_exhausted", "not_run"}
    skipped = []
    invalid = []
    for identifier in case_ids:
        evaluation = eval_by_id[identifier]
        if evaluation.get("status") == "skipped":
            reason = str(evaluation.get("skip_reason", ""))
            if reason not in bounded_skip_reasons:
                invalid.append({"id": identifier, "reason": "unbounded_skip_reason"})
            skipped.append(identifier)
        check = checks_by_id[identifier]
        if check.get("status") not in {"checked", "not_applicable"}:
            invalid.append({"id": identifier, "reason": "independent_check_missing"})
        if identifier in rare | slow and check.get("status") != "checked":
            invalid.append({"id": identifier, "reason": "special_case_not_checked"})
    checked_count = sum(checks_by_id[item].get("status") == "checked" for item in case_ids)
    independent_rate = checked_count / len(case_ids) if case_ids else 0.0
    saved_count = sum(eval_by_id[item].get("status") == "skipped" for item in case_ids)
    eligible = not invalid and independent_rate >= minimum_independent_rate
    return {
        "schema_version": "mathmodel.efficiency-account/v1",
        "case_count": len(case_ids),
        "expensive_evaluated_count": len(case_ids) - saved_count,
        "saved_evaluation_count": saved_count,
        "independent_checked_count": checked_count,
        "independent_check_rate": round(independent_rate, 8),
        "rare_event_case_ids": sorted(rare),
        "slow_convergence_case_ids": sorted(slow),
        "special_cases_checked": not any(
            item.get("reason") == "special_case_not_checked" for item in invalid
        ),
        "invalid_records": invalid,
        "eligible_for_speed_report": eligible,
        "policy": "savings_are_descriptive;_independent_and_special_case_checks_cannot_be_dropped",
    }


__all__ = ["EfficiencyAccountingError", "build_efficiency_account"]
