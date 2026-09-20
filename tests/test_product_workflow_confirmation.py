import pytest

from core.product_workflow_confirmation import (score_product_workflow_output,
                                                summarize_product_workflow_scores)


def test_product_workflow_scorer_checks_prediction_not_completion_alone():
    reference = {"expected_status": "completed", "expected_prediction": 4.0, "tolerance": 0.01}
    assert score_product_workflow_output(
        {"status": "completed", "predictions": [4.0]}, reference)["valid"] is True
    wrong = score_product_workflow_output(
        {"status": "completed", "result_grade": "validated_candidate", "predictions": [9.0]}, reference)
    assert wrong["valid"] is False
    assert wrong["covered"] is True
    assert wrong["validated_accepted"] is True
    assert wrong["incorrect_prediction_accept"] is True
    assert wrong["out_of_scope_accept"] is False
    assert wrong["unsafe_accept"] is True


def test_product_workflow_scorer_counts_wrong_refusal_separately():
    reference = {"expected_status": "completed", "expected_prediction": 4.0, "tolerance": 0.01}
    scored = score_product_workflow_output(
        {"status": "needs_input", "result_grade": "abstain"}, reference)
    assert scored["false_abstain"] is True
    assert scored["unsafe_accept"] is False
    assert scored["covered"] is False
    ungraded = score_product_workflow_output({"status": "needs_input"}, reference)
    assert ungraded["false_abstain"] is True
    assert ungraded["abstained"] is True


def test_product_workflow_risk_summary_keeps_acceptance_errors_and_accuracy_distinct():
    completed = {"expected_status": "completed", "expected_prediction": 4.0, "tolerance": 0.01}
    abstain = {"expected_status": "abstain", "expected_prediction": None, "tolerance": 0.0}
    rows = [
        score_product_workflow_output({"status": "completed", "predictions": [4.0]}, completed),
        score_product_workflow_output({"status": "completed", "predictions": [9.0]}, completed),
        score_product_workflow_output({"status": "completed", "predictions": [1.0]}, abstain),
        score_product_workflow_output({"status": "needs_input", "result_grade": "abstain"}, completed),
    ]
    summary = summarize_product_workflow_scores(rows)
    assert summary["accepted_count"] == 3
    assert summary["accepted_correct_count"] == 1
    assert summary["accepted_accuracy"] == pytest.approx(1 / 3)
    assert summary["incorrect_prediction_accept_count"] == 1
    assert summary["out_of_scope_accept_count"] == 1
    assert summary["false_abstain_count"] == 1
    assert summary["unsafe_accept_count"] == 2


def test_product_workflow_scorer_requires_explicit_abstention_for_out_of_scope_case():
    reference = {"expected_status": "abstain", "expected_prediction": None, "tolerance": 0.0}
    accepted = score_product_workflow_output(
        {"status": "completed", "predictions": [1.0]}, reference)
    assert accepted["valid"] is False
    assert accepted["unsafe_accept"] is True
    abstained = score_product_workflow_output(
        {"status": "needs_input", "result_grade": "abstain"}, reference)
    assert abstained["valid"] is True
    assert abstained["unsafe_accept"] is False


def test_product_workflow_scorer_accepts_missing_query_request():
    reference = {"expected_status": "needs_input", "expected_prediction": None, "tolerance": 0.0}
    scored = score_product_workflow_output(
        {"status": "needs_input", "reason": "model_identified_but_prediction_query_missing"}, reference)
    assert scored["valid"] is True
from core.product_fallback_confirmation import build_product_fallback_confirmation
from core.product_rescue_confirmation import build_product_rescue_confirmation


def test_fallback_confirmation_cases_have_raw_queries_and_distinct_controls():
    cases = build_product_fallback_confirmation()
    assert len(cases) == 4
    assert len({case.case_id for case in cases}) == 4
    assert {case.structure_group for case in cases} == {
        "four-way-product-sine", "affine-control",
    }
    assert all(case.expected_prediction is not None and len(case.records) >= 64 for case in cases)
    assert all(case.target_column in case.records[0] for case in cases)


def test_rescue_confirmation_cases_use_fresh_domains_and_keep_controls():
    cases = build_product_rescue_confirmation()
    assert len(cases) == 4
    assert {case.structure_group for case in cases} == {
        "sqrt-absolute-transition", "affine-control",
    }
    assert all(len(case.records) >= 64 and case.expected_prediction is not None for case in cases)
    assert cases[0].records != cases[1].records
