"""Audits for train/validation/final-test split and preprocessing isolation."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class SplitIntegrityError(ValueError):
    pass


def audit_final_test_isolation(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    final_test_fingerprint: str,
    training_fingerprint: str,
) -> dict[str, Any]:
    """Verify that reading final-test values cannot alter development artifacts."""
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        raise SplitIntegrityError("snapshots_must_be_mappings")
    for name, value in (("final_test_fingerprint", final_test_fingerprint),
                        ("training_fingerprint", training_fingerprint)):
        if not isinstance(value, str) or not value.strip():
            raise SplitIntegrityError(f"{name}_required")
    if final_test_fingerprint == training_fingerprint:
        raise SplitIntegrityError("training_and_final_fingerprints_must_differ")
    dimensions = ("feature_selection_digest", "preprocessing_fit_digest", "model_fit_digest", "training_row_count")
    changed = [key for key in dimensions if before.get(key) != after.get(key)]
    final_referenced = [key for key in dimensions if final_test_fingerprint in str(after.get(key, ""))]
    return {
        "schema_version": "mathmodel.split-integrity/v1",
        "status": "pass" if not changed and not final_referenced else "fail",
        "training_fingerprint": training_fingerprint,
        "final_test_fingerprint": final_test_fingerprint,
        "changed_training_artifacts": changed,
        "final_test_references": final_referenced,
        "policy": "final_test_is_locked_and_cannot_change_training_preprocessing_or_fit_artifacts",
    }


def audit_split_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(manifest, Mapping):
        raise SplitIntegrityError("manifest_must_be_mapping")
    required = ("training_fingerprint", "validation_fingerprint", "final_test_fingerprint", "preprocessing_fit_split")
    missing = [key for key in required if not isinstance(manifest.get(key), str) or not manifest[key].strip()]
    if missing:
        return {"schema_version": "mathmodel.split-integrity/v1", "status": "not_assessed", "missing": missing}
    distinct = len({manifest[key] for key in required[:3]}) == 3
    fit_ok = manifest["preprocessing_fit_split"] in {"training", "training_only"}
    return {
        "schema_version": "mathmodel.split-integrity/v1",
        "status": "pass" if distinct and fit_ok else "fail",
        "missing": [],
        "distinct_split_fingerprints": distinct,
        "preprocessing_fit_split": manifest["preprocessing_fit_split"],
        "policy": "fit_statistics_must_be_estimated_inside_training_partition",
    }


def audit_point_in_time_features(
    records: Sequence[Mapping[str, Any]],
    *,
    prediction_time_key: str = "prediction_time",
    available_at_key: str = "available_at",
) -> dict[str, Any]:
    """Check that every feature is available no later than its prediction time."""
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise SplitIntegrityError("records_must_be_sequence")
    violations = []
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or prediction_time_key not in row or available_at_key not in row:
            raise SplitIntegrityError("time_keys_required")
        try:
            prediction = float(row[prediction_time_key])
            available = float(row[available_at_key])
        except (TypeError, ValueError, OverflowError) as exc:
            raise SplitIntegrityError("time_values_must_be_numeric") from exc
        if available > prediction:
            violations.append({"index": index, "prediction_time": prediction, "available_at": available})
    return {"schema_version": "mathmodel.split-integrity/v1", "status": "pass" if not violations else "fail",
            "checked_rows": len(records), "violation_count": len(violations), "violations": violations[:128],
            "policy": "point_in_time_features_cannot_read_future_values"}


__all__ = ["SplitIntegrityError", "audit_final_test_isolation", "audit_split_manifest", "audit_point_in_time_features"]
