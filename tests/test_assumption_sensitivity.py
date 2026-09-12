import pytest

from core.assumption_sensitivity import AssumptionSensitivityError, assess_assumption_sensitivity


def test_assumption_sensitivity_reports_decision_changes():
    result = assess_assumption_sensitivity(
        {"threshold": 1},
        [{"id": "low", "overrides": {"threshold": 0}}, {"id": "same", "overrides": {"threshold": 2}}],
        lambda a: {"status": "assessed", "decision": "yes" if a["threshold"] >= 1 else "no", "objective": a["threshold"]},
    )
    assert result["decision_change_count"] == 1
    assert result["variants"][0]["changed_decision"] is True


def test_malformed_evaluator_result_is_unresolved_not_a_crash():
    result = assess_assumption_sensitivity(
        {"threshold": 1}, [{"id": "bad", "overrides": {"threshold": 2}}],
        lambda _a: ["not", "a", "mapping"],
    )
    assert result["status"] == "not_assessed"
    assert result["reason"] == "reference_evaluation_failed"


def test_duplicate_variant_ids_and_invalid_override_keys_are_rejected():
    with pytest.raises(AssumptionSensitivityError, match="variant_id_must_be_unique"):
        assess_assumption_sensitivity(
            {}, [{"id": "same", "overrides": {}}, {"id": "same", "overrides": {}}],
            lambda _a: {"decision": "ok"},
        )
    with pytest.raises(AssumptionSensitivityError, match="variant_overrides_invalid"):
        assess_assumption_sensitivity(
            {}, [{"id": "bad", "overrides": {"": 1}}],
            lambda _a: {"decision": "ok"},
        )
