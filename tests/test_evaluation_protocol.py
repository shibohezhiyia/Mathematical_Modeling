import pytest

from core.evaluation_protocol import (
    EvaluationProtocolError,
    ValidationTrialLedger,
    assess_dynamics_channels,
    validate_multi_table_manifest,
)


FP1 = "a" * 64
FP2 = "b" * 64


def test_validation_trials_are_counted_and_final_test_is_locked():
    ledger = ValidationTrialLedger().record_validation(run_id="r1", data_fingerprint=FP1)
    ledger = ledger.record_validation(run_id="r2", data_fingerprint=FP1)
    assert len(ledger.records) == 2
    ledger = ledger.freeze_final_test(data_fingerprint=FP2)
    ledger = ledger.consume_final_test(run_id="r3")
    assert ledger.final_test_consumed is True
    with pytest.raises(EvaluationProtocolError, match="final_test_already_consumed"):
        ledger.consume_final_test(run_id="r4")


def test_final_test_cannot_drive_adaptation_and_dynamics_channels_stay_separate():
    ledger = ValidationTrialLedger().freeze_final_test(data_fingerprint=FP1)
    with pytest.raises(EvaluationProtocolError, match="final_test_cannot_drive_adaptation"):
        ledger.consume_final_test(run_id="r1", adapted_after_test=True)
    result = assess_dynamics_channels({
        "derivative_consistency": {"status": "pass", "residual": 0.1},
        "integral_consistency": {"status": "pass", "residual": 0.2},
        "conditional_prediction": {"status": "not_assessed"},
        "independent_trajectory": {"status": "fail", "residual": 2.0},
    })
    assert result["status"] == "partial"
    assert result["channels"]["independent_trajectory"]["status"] == "fail"


def test_multi_table_manifest_requires_explicit_keys_units_and_provenance():
    result = validate_multi_table_manifest([{
        "left_table": "orders", "right_table": "customers",
        "left_keys": ["customer_id"], "right_keys": ["id"],
        "cardinality": "many_to_one",
        "entity_key_evidence": {"status": "verified"},
        "time_key_evidence": {"status": "not_available"},
        "unit_bindings": {"amount": "CNY"},
        "source_columns": {"orders.customer_id": "user_file"},
    }])
    assert result["status"] == "assessed"
    with pytest.raises(EvaluationProtocolError, match="relation_evidence_incomplete"):
        validate_multi_table_manifest([{"left_table": "a", "right_table": "b"}])
