import hashlib
import json

import pytest

from core.blind_benchmark import (
    BlindBenchmarkError,
    BlindManifest,
    build_blind_report,
    build_sealed_case,
    record_blind_run,
    scan_leakage,
    unlock_and_score,
)


def _budget():
    return {
        "max_seconds": 30,
        "max_memory_mb": 256,
        "max_api_calls": 4,
        "max_candidates": 12,
        "seed": 17,
    }


def _sealed(tmp_path):
    statement = tmp_path / "question.txt"
    answer = tmp_path / "reference.txt"
    statement.write_text("仅用于封存流程的外部题面，不是公开题目。", encoding="utf-8")
    answer.write_text("外部评测者的参考解。", encoding="utf-8")
    case, token = build_sealed_case(
        case_id="real-u-001", family="optimization", statement_path=statement,
        answer_path=answer, provenance="external_real", unlock_token="secret-token-for-test-001",
    )
    return statement, answer, case, token


def test_manifest_contains_only_commitments_and_is_deterministic(tmp_path):
    statement, answer, case, _ = _sealed(tmp_path)
    manifest = BlindManifest.create([case], budget=_budget())
    payload = manifest.public()
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "外部题面" not in encoded
    assert "参考解" not in encoded
    assert payload["cases"][0]["statement_sha256"] == hashlib.sha256(statement.read_bytes()).hexdigest()
    assert payload["cases"][0]["answer_sha256"] == hashlib.sha256(answer.read_bytes()).hexdigest()
    assert BlindManifest.from_payload(payload).digest == manifest.digest


def test_leakage_scan_passes_clean_tree_and_detects_case_id_or_exact_copy(tmp_path):
    statement, answer, case, _ = _sealed(tmp_path)
    manifest = BlindManifest.create([case], budget=_budget())
    public = tmp_path / "public_manifest.json"
    public.write_text(json.dumps(manifest.public(), ensure_ascii=False), encoding="utf-8")
    clean = scan_leakage(tmp_path, manifest, exclude_paths=[public, statement, answer])
    assert clean["status"] == "pass"
    leaked = tmp_path / "docs.md"
    leaked.write_text("实现 real-u-001 的题面。", encoding="utf-8")
    result = scan_leakage(tmp_path, manifest, exclude_paths=[public, statement, answer])
    assert result["status"] == "fail"
    assert any(item["reason"] == "case_id_literal" for item in result["findings"])


def test_leakage_scan_detects_partial_protected_text_overlap(tmp_path):
    statement, answer, case, _ = _sealed(tmp_path)
    manifest = BlindManifest.create([case], budget=_budget())
    protected = tmp_path / "protected.txt"
    protected.write_text("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda", encoding="utf-8")
    leaked = tmp_path / "partial.md"
    leaked.write_text("前言 alpha beta gamma delta epsilon zeta eta theta iota 后续说明", encoding="utf-8")
    result = scan_leakage(tmp_path, manifest, exclude_paths=[statement, answer, protected], protected_files=[protected])
    assert result["status"] == "fail"
    assert any(item["reason"] == "protected_file_shingle_overlap" for item in result["findings"])


def test_fixed_budget_run_is_required_and_pre_unlock_report_has_no_scores(tmp_path):
    _statement, _answer, case, token = _sealed(tmp_path)
    manifest = BlindManifest.create([case], budget=_budget())
    run = record_blind_run(
        manifest, case_id=case.case_id, run_id="run-1", status="completed", budget=_budget(),
        system_version="candidate-1", api_calls=2, candidate_count=4,
    )
    with pytest.raises(BlindBenchmarkError, match="fixed_budget_mismatch"):
        record_blind_run(manifest, case_id=case.case_id, run_id="run-2", status="completed",
                         budget={**_budget(), "seed": 18}, system_version="candidate-1")
    before = build_blind_report(manifest, [run])
    assert before["evaluation_status"] == "not_scored"
    assert before["score_means"] == {}
    assert before["real_unseen_declared"] is True
    score = unlock_and_score(
        manifest, run,
        reference={
            "schema_version": "mathmodel.blind-score/v1", "manifest_digest": manifest.digest,
            "case_id": case.case_id, "answer_sha256": case.answer_sha256,
            "scores": {"contract_correct": True, "numerically_correct": 0.75},
            "evaluator": "independent-evaluator",
        },
        unlock_token=token,
    )
    after = build_blind_report(manifest, [run], [score])
    assert after["evaluation_status"] == "scored"
    assert after["score_means"]["numerically_correct"] == pytest.approx(0.75)


def test_wrong_token_or_reference_cannot_unlock(tmp_path):
    _statement, _answer, case, _token = _sealed(tmp_path)
    manifest = BlindManifest.create([case], budget=_budget())
    run = record_blind_run(manifest, case_id=case.case_id, run_id="run-1", status="completed",
                           budget=_budget(), system_version="candidate-1")
    ref = {
        "schema_version": "mathmodel.blind-score/v1", "manifest_digest": manifest.digest,
        "case_id": case.case_id, "answer_sha256": case.answer_sha256,
        "scores": {"numerically_correct": 1.0}, "evaluator": "evaluator",
    }
    with pytest.raises(BlindBenchmarkError, match="unlock_token_mismatch"):
        unlock_and_score(manifest, run, reference=ref, unlock_token="wrong-token-for-test")
    with pytest.raises(BlindBenchmarkError, match="reference_answer_commitment_mismatch"):
        unlock_and_score(manifest, run, reference={**ref, "answer_sha256": "0" * 64},
                         unlock_token="secret-token-for-test-001")


def test_synthetic_fixture_is_explicitly_not_real_evidence(tmp_path):
    statement = tmp_path / "question.txt"
    statement.write_text("fixture", encoding="utf-8")
    case, _ = build_sealed_case(case_id="fixture-1", family="data", statement_path=statement,
                                provenance="synthetic_fixture", unlock_token="fixture-token-00000001")
    manifest = BlindManifest.create([case], budget=_budget())
    assert build_blind_report(manifest, [])["real_unseen_declared"] is False
