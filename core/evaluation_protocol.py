"""Protocols preventing validation/final-test reuse and mixing dynamics metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence


class EvaluationProtocolError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fingerprint(value: Any) -> str:
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise EvaluationProtocolError("invalid_data_fingerprint")
    return value


@dataclass(frozen=True)
class ValidationTrialLedger:
    """Append-only trial ledger with an explicit final-test freeze boundary."""

    records: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    final_test_fingerprint: str | None = None
    final_test_consumed: bool = False

    def record_validation(self, *, run_id: str, data_fingerprint: str, purpose: str = "model_selection") -> "ValidationTrialLedger":
        if type(run_id) is not str or not run_id or len(run_id) > 128:
            raise EvaluationProtocolError("invalid_run_id")
        _fingerprint(data_fingerprint)
        if type(purpose) is not str or not purpose or len(purpose) > 128:
            raise EvaluationProtocolError("invalid_trial_purpose")
        item = {"run_id": run_id, "split": "validation", "data_fingerprint": data_fingerprint,
                "purpose": purpose, "trial_index": len(self.records) + 1}
        return ValidationTrialLedger((*self.records, item), self.final_test_fingerprint, self.final_test_consumed)

    def freeze_final_test(self, *, data_fingerprint: str) -> "ValidationTrialLedger":
        fingerprint = _fingerprint(data_fingerprint)
        if self.final_test_fingerprint is not None and self.final_test_fingerprint != fingerprint:
            raise EvaluationProtocolError("final_test_fingerprint_already_frozen")
        if self.final_test_consumed:
            raise EvaluationProtocolError("final_test_already_consumed")
        return ValidationTrialLedger(self.records, fingerprint, False)

    def consume_final_test(self, *, run_id: str, adapted_after_test: bool = False) -> "ValidationTrialLedger":
        if self.final_test_fingerprint is None:
            raise EvaluationProtocolError("final_test_not_frozen")
        if self.final_test_consumed:
            raise EvaluationProtocolError("final_test_already_consumed")
        if type(run_id) is not str or not run_id:
            raise EvaluationProtocolError("invalid_run_id")
        if adapted_after_test:
            raise EvaluationProtocolError("final_test_cannot_drive_adaptation")
        item = {"run_id": run_id, "split": "final_test", "data_fingerprint": self.final_test_fingerprint,
                "purpose": "locked_evaluation", "trial_index": len(self.records) + 1}
        return ValidationTrialLedger((*self.records, item), self.final_test_fingerprint, True)

    def public(self) -> dict[str, Any]:
        return {"schema_version": "mathmodel.evaluation-ledger/v1", "records": [dict(item) for item in self.records],
                "final_test_fingerprint": self.final_test_fingerprint,
                "final_test_consumed": self.final_test_consumed,
                "policy": "validation_reuse_counted_final_test_locked"}


def assess_dynamics_channels(channels: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Keep derivative, conditional-history and independent-trajectory evidence separate."""
    required = ("derivative_consistency", "integral_consistency", "conditional_prediction", "independent_trajectory")
    if not isinstance(channels, Mapping):
        raise EvaluationProtocolError("channels_must_be_mapping")
    output: dict[str, Any] = {}
    for name in required:
        raw = channels.get(name)
        if not isinstance(raw, Mapping):
            output[name] = {"status": "not_assessed", "reason": "channel_missing"}
            continue
        status = raw.get("status", "not_assessed")
        if status not in {"pass", "fail", "not_assessed"}:
            raise EvaluationProtocolError("invalid_dynamics_channel_status")
        residual = raw.get("residual")
        if residual is not None and (type(residual) not in (int, float) or not math.isfinite(float(residual)) or residual < 0):
            raise EvaluationProtocolError("invalid_dynamics_residual")
        output[name] = {"status": status, "residual": float(residual) if residual is not None else None,
                        "scope": str(raw.get("scope", ""))[:160]}
    assessed = [item for item in output.values() if item["status"] != "not_assessed"]
    return {"schema_version": "mathmodel.dynamics-evaluation/v1", "channels": output,
            "assessed_count": len(assessed), "status": "partial" if len(assessed) < len(required) else "assessed",
            "policy": "channels_are_not_interchangeable"}


def validate_multi_table_manifest(relations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate explicit multi-table join evidence without performing a join."""
    if not isinstance(relations, Sequence) or isinstance(relations, (str, bytes)) or len(relations) > 256:
        raise EvaluationProtocolError("invalid_relation_manifest")
    allowed_cardinality = {"one_to_one", "one_to_many", "many_to_one", "many_to_many"}
    normalized = []
    for index, relation in enumerate(relations):
        if not isinstance(relation, Mapping):
            raise EvaluationProtocolError("relation_must_be_object")
        required = {"left_table", "right_table", "left_keys", "right_keys", "cardinality",
                    "entity_key_evidence", "time_key_evidence", "unit_bindings", "source_columns"}
        if not required <= set(relation):
            raise EvaluationProtocolError("relation_evidence_incomplete")
        if any(type(relation[key]) is not str or not relation[key].strip() for key in ("left_table", "right_table")):
            raise EvaluationProtocolError("relation_table_name_required")
        left_keys, right_keys = relation["left_keys"], relation["right_keys"]
        if (not isinstance(left_keys, list) or not left_keys or len(left_keys) > 16
                or not isinstance(right_keys, list) or len(left_keys) != len(right_keys)
                or any(type(key) is not str or not key for key in (*left_keys, *right_keys))):
            raise EvaluationProtocolError("relation_keys_invalid")
        if relation["cardinality"] not in allowed_cardinality:
            raise EvaluationProtocolError("relation_cardinality_invalid")
        for evidence_key in ("entity_key_evidence", "time_key_evidence"):
            evidence = relation[evidence_key]
            if type(evidence) is not dict or evidence.get("status") not in {"verified", "not_available", "not_assessed"}:
                raise EvaluationProtocolError("relation_key_evidence_invalid")
        if type(relation["unit_bindings"]) is not dict or type(relation["source_columns"]) is not dict:
            raise EvaluationProtocolError("relation_source_or_unit_evidence_invalid")
        normalized.append({"index": index, "left_table": relation["left_table"],
                           "right_table": relation["right_table"], "left_keys": list(left_keys),
                           "right_keys": list(right_keys), "cardinality": relation["cardinality"],
                           "entity_key_status": relation["entity_key_evidence"]["status"],
                           "time_key_status": relation["time_key_evidence"]["status"],
                           "source_column_count": len(relation["source_columns"]),
                           "unit_binding_count": len(relation["unit_bindings"])})
    return {"schema_version": "mathmodel.multi-table-evidence/v1", "relation_count": len(normalized),
            "relations": normalized, "status": "assessed" if normalized else "not_assessed",
            "policy": "correlation_alone_cannot_authorize_join"}


__all__ = ["EvaluationProtocolError", "ValidationTrialLedger", "assess_dynamics_channels",
           "validate_multi_table_manifest"]
