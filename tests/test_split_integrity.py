import pytest

from core.split_integrity import SplitIntegrityError, audit_final_test_isolation, audit_point_in_time_features, audit_split_manifest


def test_split_integrity_detects_final_test_influence():
    base = {"feature_selection_digest": "f", "preprocessing_fit_digest": "p", "model_fit_digest": "m", "training_row_count": 10}
    assert audit_final_test_isolation(base, dict(base), final_test_fingerprint="final", training_fingerprint="train")["status"] == "pass"
    changed = dict(base); changed["preprocessing_fit_digest"] = "final:p"
    assert audit_final_test_isolation(base, changed, final_test_fingerprint="final", training_fingerprint="train")["status"] == "fail"


def test_split_manifest_requires_distinct_partitions_and_training_fit():
    payload = {"training_fingerprint": "train", "validation_fingerprint": "valid", "final_test_fingerprint": "test", "preprocessing_fit_split": "training"}
    assert audit_split_manifest(payload)["status"] == "pass"
    with pytest.raises(SplitIntegrityError):
        audit_final_test_isolation(payload, payload, final_test_fingerprint="same", training_fingerprint="same")


def test_point_in_time_feature_audit_rejects_future_values():
    assert audit_point_in_time_features([{"prediction_time": 2, "available_at": 1}])["status"] == "pass"
    result = audit_point_in_time_features([{"prediction_time": 2, "available_at": 3}])
    assert result["status"] == "fail"
