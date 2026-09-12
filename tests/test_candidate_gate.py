import pytest

from core.candidate_gate import CandidateGateError, classify_failure, gate_candidates


def test_hard_failures_cannot_be_offset_by_better_fit():
    result = gate_candidates([
        {"id": "bad", "hard_checks": {"units": "fail"}, "metrics": {"loss": 0.1}},
        {"id": "good", "hard_checks": {"units": "pass"}, "metrics": {"loss": 1.0}},
    ])
    assert result["candidates"][0]["status"] == "hard_failure"
    assert result["pareto_ids"] == ["good"]


def test_failure_taxonomy_is_explicit():
    assert classify_failure("resource")["mathematical_verdict"] == "not_assessed"
    with pytest.raises(CandidateGateError):
        classify_failure("unknown")


def test_invalid_metrics_remain_unassessed_instead_of_crashing_or_being_ranked():
    result = gate_candidates([
        {"id": "nan_candidate", "hard_checks": {"units": "pass"}, "metrics": {"loss": float("nan")}},
        {"id": "valid_candidate", "hard_checks": {"units": "pass"}, "metrics": {"loss": 1.0}},
    ])
    row = next(item for item in result["candidates"] if item["id"] == "nan_candidate")
    assert row["status"] == "not_assessed"
    assert "metric_invalid" in row["pending_hard_checks"]
    assert "nan_candidate" not in result["pareto_ids"]


def test_incomplete_metric_schema_is_not_pareto_ranked_against_a_different_schema():
    result = gate_candidates([
        {"id": "loss_only", "hard_checks": {"units": "pass"}, "metrics": {"loss": 0.1}},
        {"id": "loss_and_coverage", "hard_checks": {"units": "pass"},
         "metrics": {"loss": 0.2, "coverage": 0.9}},
    ])
    assert result["pareto_ids"] == []
    assert result["eligible_ids"] == []
    assert result["incomparable_metric_schema_ids"] == ["loss_and_coverage", "loss_only"]
    assert all("metric_schema_incomplete" in row["pending_hard_checks"]
               for row in result["candidates"])
    assert set(result["metric_schema_groups"]) == {"loss", "coverage|loss"}


def test_homogeneous_metric_schema_still_supports_pareto_comparison():
    result = gate_candidates([
        {"id": "a", "hard_checks": {"units": "pass"}, "metrics": {"loss": 0.1, "complexity": 2}},
        {"id": "b", "hard_checks": {"units": "pass"}, "metrics": {"loss": 0.2, "complexity": 3}},
    ])
    assert result["pareto_ids"] == ["a"]
    assert result["incomparable_metric_schema_ids"] == []


def test_duplicate_candidate_ids_are_rejected_before_comparison():
    with pytest.raises(CandidateGateError, match="candidate_id_must_be_unique"):
        gate_candidates([
            {"id": "same", "hard_checks": {}, "metrics": {"loss": 1}},
            {"id": "same", "hard_checks": {}, "metrics": {"loss": 2}},
        ])


def test_explicitly_executable_candidate_requires_readiness_evidence():
    result = gate_candidates([{
        "id": "ran_without_evidence",
        "execution_status": "executed",
        "hard_checks": {"type": "pass", "unit": "pass", "source": "pass",
                         "resource": "pass", "security": "pass"},
        "metrics": {"loss": 0.1},
    }])
    row = result["candidates"][0]
    assert row["status"] == "not_assessed"
    assert "execution_readiness" in row["pending_hard_checks"]


def test_executable_candidate_with_complete_readiness_can_be_compared():
    result = gate_candidates([{
        "id": "audited_run",
        "execution_status": "executed",
        "hard_checks": {"type": "pass", "unit": "pass", "source": "pass",
                         "resource": "pass", "security": "pass"},
        "execution_readiness": {
            "type": "ready", "unit": "ready", "source": "ready",
            "resource": "ready", "security": "ready",
        },
        "metrics": {"loss": 0.1},
    }])
    assert result["eligible_ids"] == ["audited_run"]
