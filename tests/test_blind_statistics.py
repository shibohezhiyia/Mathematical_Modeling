import pytest

from core.blind_benchmark import BlindManifest, build_sealed_case, record_blind_run, unlock_and_score
from core.blind_statistics import assess_blind_accuracy


def _budget():
    return {
        "max_seconds": 30, "max_memory_mb": 256, "max_api_calls": 4,
        "max_candidates": 12, "seed": 17,
    }


def _fixture(tmp_path, count=20, provenance="external_real"):
    cases, tokens = [], {}
    for index in range(count):
        statement = tmp_path / f"question-{index}.txt"
        answer = tmp_path / f"answer-{index}.txt"
        statement.write_text(f"sealed question {index}", encoding="utf-8")
        answer.write_text(f"independent reference {index}", encoding="utf-8")
        case, token = build_sealed_case(
            case_id=f"unseen-{index:03d}", family="generic", statement_path=statement,
            answer_path=answer, provenance=provenance,
            unlock_token=f"fixture-token-{index:03d}-long-enough",
        )
        cases.append(case)
        tokens[case.case_id] = token
    return BlindManifest.create(cases, budget=_budget()), tokens


def _runs_and_scores(manifest, tokens, *, treatment_score=1.0):
    runs, scores = [], []
    for index, case in enumerate(manifest.cases()):
        for system, score in (("baseline-v1", 0.0), ("treatment-v1", treatment_score)):
            run = record_blind_run(
                manifest, case_id=case.case_id, run_id=f"{system}-{index}",
                status="completed", budget=_budget(), system_version=system,
            )
            runs.append(run)
            scores.append(unlock_and_score(
                manifest, run,
                reference={
                    "schema_version": "mathmodel.blind-score/v1",
                    "manifest_digest": manifest.digest, "case_id": case.case_id,
                    "answer_sha256": case.answer_sha256,
                    "scores": {"numerically_correct": score},
                    "evaluator": "independent-rater",
                },
                unlock_token=tokens[case.case_id],
            ))
    return runs, scores


def test_real_unseen_accuracy_and_paired_significance_are_claimable(tmp_path):
    manifest, tokens = _fixture(tmp_path)
    runs, scores = _runs_and_scores(manifest, tokens)
    report = assess_blind_accuracy(
        manifest, runs, scores, baseline_system="baseline-v1", treatment_system="treatment-v1",
        min_cases=20, bootstrap_replicates=200, independent_evaluation_attested=True,
    )
    assert report["status"] == "statistically_significant"
    assert report["accuracy_claim"] == "conditional_real_unseen"
    assert report["baseline_accuracy"] == pytest.approx(0.0)
    assert report["treatment_accuracy"] == pytest.approx(1.0)
    assert report["mcnemar"]["p_value"] < 0.05
    assert report["paired_difference_interval"][0] > 0


def test_accuracy_report_requires_independent_attestation(tmp_path):
    manifest, tokens = _fixture(tmp_path)
    runs, scores = _runs_and_scores(manifest, tokens)
    report = assess_blind_accuracy(
        manifest, runs, scores, baseline_system="baseline-v1", treatment_system="treatment-v1",
        min_cases=20, bootstrap_replicates=200,
    )
    assert report["status"] == "not_assessed"
    assert report["reason"] == "independent_evaluation_attestation_required"


def test_synthetic_or_mixed_cases_never_produce_real_accuracy_claim(tmp_path):
    manifest, tokens = _fixture(tmp_path, provenance="synthetic_fixture")
    runs, scores = _runs_and_scores(manifest, tokens)
    report = assess_blind_accuracy(
        manifest, runs, scores, baseline_system="baseline-v1", treatment_system="treatment-v1",
        min_cases=20, bootstrap_replicates=200, independent_evaluation_attested=True,
    )
    assert report["status"] == "not_assessed"
    assert report["accuracy_claim"] == "not_assessed"
    assert report["reason"] == "cases_are_not_all_external_real"


def test_failure_policy_must_be_explicit_for_incomplete_cells(tmp_path):
    manifest, tokens = _fixture(tmp_path)
    runs, scores = _runs_and_scores(manifest, tokens)
    runs = [row for row in runs if row["system_version"] != "treatment-v1" or row["case_id"] != "unseen-000"]
    scores = [row for row in scores if row["run_id"] != "treatment-v1-0"]
    report = assess_blind_accuracy(
        manifest, runs, scores, baseline_system="baseline-v1", treatment_system="treatment-v1",
        min_cases=20, bootstrap_replicates=200, independent_evaluation_attested=True,
        failure_as_incorrect=False,
    )
    assert report["status"] == "not_assessed"
    assert report["reason"] == "incomplete_cells_and_failure_policy_disallows_imputation"
