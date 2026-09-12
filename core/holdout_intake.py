"""Pre-sealing intake checks for an independent modeling holdout.

This is intentionally metadata-only.  It verifies that a proposed holdout
has enough stratified cases and role separation before the existing
``blind_benchmark`` sealing command is run; it never reads or stores a problem
statement, attachment, or reference solution.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping


HOLDOUT_INTAKE_SCHEMA = "mathmodel.holdout-intake/v1"


class HoldoutIntakeError(ValueError):
    pass


def _digest(value: Any, code: str, *, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise HoldoutIntakeError(code)
    return value


def _positive_int(value: Any, code: str, upper: int = 100_000_000) -> int:
    if type(value) is not int or not 1 <= value <= upper:
        raise HoldoutIntakeError(code)
    return value


def _case_id(value: Any) -> str:
    if type(value) is not str or not value or len(value) > 128 or any(ch.isspace() for ch in value):
        raise HoldoutIntakeError("invalid_case_id")
    return value


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HoldoutIntakeError("non_json_payload") from exc


def validate_holdout_intake(
    payload: Mapping[str, Any], *, min_cases: int = 20, min_families: int = 4,
) -> dict[str, Any]:
    """Validate independent-author/evaluator metadata before sealing.

    The returned digest commits to metadata only.  A ``ready_for_sealing``
    status still requires the caller to run ``build_sealed_case`` for each
    case and pass the resulting manifest through the blind-statistics gates.
    """
    if not isinstance(payload, Mapping) or payload.get("schema_version") != HOLDOUT_INTAKE_SCHEMA:
        raise HoldoutIntakeError("intake_schema_invalid")
    required = {"schema_version", "protocol_id", "development_freeze_digest", "independent_author_attested",
                "independent_evaluator_attested", "cases", "rubric_digest"}
    if set(payload) - required or not required <= set(payload):
        raise HoldoutIntakeError("intake_fields_invalid")
    protocol_id = payload["protocol_id"]
    if type(protocol_id) is not str or not protocol_id.strip() or len(protocol_id) > 120 or any(ch.isspace() for ch in protocol_id):
        raise HoldoutIntakeError("protocol_id_invalid")
    _digest(payload["development_freeze_digest"], "development_freeze_digest_invalid")
    _digest(payload["rubric_digest"], "rubric_digest_invalid")
    if type(payload["independent_author_attested"]) is not bool or type(payload["independent_evaluator_attested"]) is not bool:
        raise HoldoutIntakeError("independence_attestation_must_be_boolean")
    if type(min_cases) is not int or not 1 <= min_cases <= 100_000:
        raise HoldoutIntakeError("min_cases_invalid")
    if type(min_families) is not int or not 1 <= min_families <= 100:
        raise HoldoutIntakeError("min_families_invalid")
    cases = payload["cases"]
    if not isinstance(cases, list) or len(cases) > 100_000:
        raise HoldoutIntakeError("cases_invalid")
    normalized = []
    ids = set()
    families = set()
    for row in cases:
        if not isinstance(row, Mapping):
            raise HoldoutIntakeError("case_must_be_object")
        allowed = {"id", "family", "statement_sha256", "statement_bytes", "attachment_sha256",
                   "answer_sha256", "answer_bytes", "author_commitment", "evaluator_commitment"}
        required_case = {"id", "family", "statement_sha256", "statement_bytes", "attachment_sha256",
                         "answer_sha256", "answer_bytes", "author_commitment", "evaluator_commitment"}
        if set(row) - allowed or not required_case <= set(row):
            raise HoldoutIntakeError("case_fields_invalid")
        case_id = _case_id(row["id"])
        if case_id in ids:
            raise HoldoutIntakeError("duplicate_case_id")
        ids.add(case_id)
        family = row["family"]
        if type(family) is not str or not family.strip() or len(family) > 80 or any(ch.isspace() for ch in family):
            raise HoldoutIntakeError("case_family_invalid")
        families.add(family)
        _digest(row["statement_sha256"], "statement_digest_invalid")
        _digest(row["answer_sha256"], "answer_digest_required")
        _positive_int(row["statement_bytes"], "statement_bytes_invalid")
        _positive_int(row["answer_bytes"], "answer_bytes_invalid", upper=500_000_000)
        attachments = row["attachment_sha256"]
        if not isinstance(attachments, list) or len(attachments) > 128:
            raise HoldoutIntakeError("attachment_digests_invalid")
        for digest in attachments:
            _digest(digest, "attachment_digest_invalid")
        author_commitment = _digest(row["author_commitment"], "role_commitment_invalid")
        evaluator_commitment = _digest(row["evaluator_commitment"], "role_commitment_invalid")
        if author_commitment == evaluator_commitment:
            raise HoldoutIntakeError("role_commitments_must_differ")
        normalized.append({"id": case_id, "family": family, "has_attachment": bool(attachments)})
    missing = []
    if len(normalized) < min_cases:
        missing.append("minimum_case_count")
    if len(families) < min_families:
        missing.append("minimum_family_count")
    if payload["independent_author_attested"] is not True:
        missing.append("independent_author_attestation")
    if payload["independent_evaluator_attested"] is not True:
        missing.append("independent_evaluator_attestation")
    status = "ready_for_sealing" if not missing else "not_ready"
    metadata = {
        "schema_version": HOLDOUT_INTAKE_SCHEMA,
        "protocol_id": protocol_id.strip(),
        "status": status,
        "missing": missing,
        "case_count": len(normalized),
        "family_count": len(families),
        "families": sorted(families),
        "attachment_case_count": sum(1 for row in normalized if row["has_attachment"]),
        "metadata_digest": sha256(_canonical({"protocol_id": protocol_id.strip(), "cases": normalized}).encode("utf-8")).hexdigest(),
        "policy": "metadata_only;seal_cases_outside_repository;independent_scores_unlocked_after_runs",
    }
    return metadata


__all__ = ["HOLDOUT_INTAKE_SCHEMA", "HoldoutIntakeError", "validate_holdout_intake"]
