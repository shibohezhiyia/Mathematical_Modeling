import hashlib

import pytest

from core.holdout_intake import HoldoutIntakeError, validate_holdout_intake


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _payload(count=20, author=True, evaluator=True):
    cases = []
    for index in range(count):
        cases.append({
            "id": f"u-{index}", "family": ["optimization", "dynamics", "statistics", "network"][index % 4],
            "statement_sha256": _digest(f"statement-{index}"), "statement_bytes": 100,
            "attachment_sha256": [], "answer_sha256": _digest(f"answer-{index}"), "answer_bytes": 200,
            "author_commitment": _digest(f"author-{index}"), "evaluator_commitment": _digest(f"evaluator-{index}"),
        })
    return {
        "schema_version": "mathmodel.holdout-intake/v1", "protocol_id": "p1",
        "development_freeze_digest": _digest("freeze"), "rubric_digest": _digest("rubric"),
        "independent_author_attested": author, "independent_evaluator_attested": evaluator,
        "cases": cases,
    }


def test_intake_ready_when_stratified_and_roles_are_attested():
    result = validate_holdout_intake(_payload())
    assert result["status"] == "ready_for_sealing"
    assert result["case_count"] == 20
    assert result["family_count"] == 4


def test_intake_blocks_small_or_unattested_holdout():
    result = validate_holdout_intake(_payload(count=3, evaluator=False))
    assert result["status"] == "not_ready"
    assert {"minimum_case_count", "minimum_family_count", "independent_evaluator_attestation"} <= set(result["missing"])


def test_intake_rejects_missing_answer_commitment():
    payload = _payload()
    payload["cases"][0]["answer_sha256"] = None
    with pytest.raises(HoldoutIntakeError, match="answer_digest_required"):
        validate_holdout_intake(payload)


def test_intake_rejects_same_author_and_evaluator_commitment():
    payload = _payload()
    payload["cases"][0]["evaluator_commitment"] = payload["cases"][0]["author_commitment"]
    with pytest.raises(HoldoutIntakeError, match="role_commitments_must_differ"):
        validate_holdout_intake(payload)
