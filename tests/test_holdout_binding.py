import hashlib

import pytest

from core.blind_benchmark import BlindManifest, build_sealed_case
from core.holdout_binding import HoldoutBindingError, bind_holdout_intake


def _digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _intake(tmp_path, count=20):
    rows = []
    for index in range(count):
        statement = tmp_path / f"s{index}.txt"
        answer = tmp_path / f"a{index}.json"
        statement.write_text(f"question-{index}", encoding="utf-8")
        answer.write_text(f"answer-{index}", encoding="utf-8")
        rows.append({"id": f"u-{index}", "family": ["optimization", "dynamics", "statistics", "network"][index % 4],
                     "statement_sha256": _digest(f"question-{index}"), "statement_bytes": len(f"question-{index}".encode()),
                     "attachment_sha256": [], "answer_sha256": _digest(f"answer-{index}"), "answer_bytes": len(f"answer-{index}".encode()),
                     "author_commitment": _digest(f"author-{index}"), "evaluator_commitment": _digest(f"evaluator-{index}"),
                     "statement": statement, "answer": answer})
    return rows


def test_binding_matches_manifest_hashes_and_families(tmp_path):
    rows = _intake(tmp_path)
    sealed = [build_sealed_case(case_id=row["id"], family=row["family"], statement_path=row["statement"], answer_path=row["answer"], split="unseen", provenance="external_real", unlock_token="t" * 32)[0].public() for row in rows]
    intake = {"schema_version": "mathmodel.holdout-intake/v1", "protocol_id": "p1",
              "development_freeze_digest": _digest("freeze"), "rubric_digest": _digest("rubric"),
              "independent_author_attested": True, "independent_evaluator_attested": True,
              "cases": [{key: row[key] for key in ("id", "family", "statement_sha256", "statement_bytes", "attachment_sha256", "answer_sha256", "answer_bytes", "author_commitment", "evaluator_commitment")} for row in rows]}
    manifest = BlindManifest.create(sealed, budget={"max_seconds": 10, "max_memory_mb": 32, "max_api_calls": 0, "max_candidates": 1, "seed": 1})
    result = bind_holdout_intake(intake, manifest)
    assert result["status"] == "bound"
    assert result["case_count"] == 20


def test_binding_rejects_changed_manifest_digest(tmp_path):
    rows = _intake(tmp_path)
    sealed = [build_sealed_case(case_id=row["id"], family=row["family"], statement_path=row["statement"], answer_path=row["answer"], split="unseen", provenance="external_real", unlock_token="t" * 32)[0].public() for row in rows]
    intake = {"schema_version": "mathmodel.holdout-intake/v1", "protocol_id": "p1",
              "development_freeze_digest": _digest("freeze"), "rubric_digest": _digest("rubric"),
              "independent_author_attested": True, "independent_evaluator_attested": True,
              "cases": [{key: row[key] for key in ("id", "family", "statement_sha256", "statement_bytes", "attachment_sha256", "answer_sha256", "answer_bytes", "author_commitment", "evaluator_commitment")} for row in rows]}
    sealed[0]["answer_sha256"] = _digest("tampered")
    manifest = BlindManifest.create(sealed, budget={"max_seconds": 10, "max_memory_mb": 32, "max_api_calls": 0, "max_candidates": 1, "seed": 1})
    with pytest.raises(HoldoutBindingError, match="answer_sha256_mismatch"):
        bind_holdout_intake(intake, manifest)
