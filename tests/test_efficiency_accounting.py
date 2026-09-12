import pytest

from core.efficiency_accounting import EfficiencyAccountingError, build_efficiency_account


def _rows(ids, statuses):
    return [{"id": identifier, "status": status, **(
        {"skip_reason": "static_rejection"} if status == "skipped" else {}
    )} for identifier, status in zip(ids, statuses)]


def test_efficiency_account_preserves_special_cases_and_reports_savings():
    ids = ["normal", "rare", "slow"]
    account = build_efficiency_account(
        _rows(ids, ["registered"] * 3),
        expensive_evaluations=_rows(ids, ["skipped", "evaluated", "evaluated"]),
        independent_checks=_rows(ids, ["checked"] * 3),
        rare_event_ids=["rare"],
        slow_convergence_ids=["slow"],
    )
    assert account["saved_evaluation_count"] == 1
    assert account["special_cases_checked"] is True
    assert account["eligible_for_speed_report"] is True


def test_efficiency_account_rejects_missing_special_case_check():
    ids = ["rare", "ordinary"]
    with pytest.raises(EfficiencyAccountingError, match="independent_checks_must_cover"):
        build_efficiency_account(
            _rows(ids, ["registered"] * 2),
            expensive_evaluations=_rows(ids, ["evaluated"] * 2),
            independent_checks=[{"id": "ordinary", "status": "checked"}],
            rare_event_ids=["rare"],
        )
