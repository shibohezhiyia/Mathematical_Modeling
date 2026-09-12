from core.acceptance_metrics import assess_acceptance_rates


def test_acceptance_metrics_exposes_both_denominators():
    result = assess_acceptance_rates(
        [{"case_id": "a", "status": "approved"}, {"case_id": "b", "status": "approved"}],
        {"a": True, "b": False, "c": True},
    )
    assert result["denominators"] == {"accepted_with_truth": 2, "truth_cases": 3}
    assert result["wrong_accepted_count"] == 1


def test_acceptance_metrics_deduplicates_ids_and_reports_malformed_rows():
    result = assess_acceptance_rates([
        {"case_id": " a ", "status": "approved"},
        {"case_id": "a", "status": "approved"},
        {"case_id": " ", "status": "approved"},
        None,
    ], {"a": False, "b": "unknown"})
    assert result["accepted_count"] == 1
    assert result["accepted_with_truth_count"] == 1
    assert result["wrong_accepted_count"] == 1
    assert result["missing_case_id_count"] == 1
    assert result["malformed_result_count"] == 1
    assert result["invalid_ground_truth_count"] == 1
