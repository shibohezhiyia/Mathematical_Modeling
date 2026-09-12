import pytest

from core.model_verdict import ModelVerdictError, build_model_verdict


def test_verdict_never_upgrades_conditional_score_to_approval():
    result = build_model_verdict([
        {"id": "a", "status": "pass", "decision": "use-a", "evidence_refs": ["run/a"]},
        {"id": "b", "status": "execution_error", "decision": "use-b"},
    ], uncertainty={"structural": {"status": "partial", "evidence": ["diag/1"]}})
    assert result["status"] == "conditional"
    assert result["approved_candidate_ids"] == []
    assert result["recommended_candidate_id"] is None
    assert result["unresolved_candidate_ids"] == ["b"]
    assert result["uncertainty"]["structural"]["status"] == "partial"
    assert result["policy"]["scores_are_not_probabilities"] is True


def test_explicit_approval_and_shared_decision_are_traceable():
    result = build_model_verdict([
        {"id": "a", "verdict_state": "approved", "status": "confirmed", "decision": 3,
         "evidence_refs": ["confirmation/a"]},
        {"id": "b", "status": "pass", "decision": 3},
    ], assumptions=["unit is SI"], run_id="run-1", evidence_refs=["bundle"])
    assert result["status"] == "approved"
    assert result["recommended_candidate_id"] == "a"
    assert result["minimum_common_conclusion"]["status"] == "supported_by_current_candidates"
    assert "bundle" in result["evidence_refs"]
    assert result["assumptions"] == ["unit is SI"]


def test_rejected_candidates_are_not_part_of_common_conclusion():
    result = build_model_verdict([
        {"id": "bad", "status": "counterexample", "decision": "bad"},
        {"id": "unknown", "status": "needs_input", "decision": "unknown"},
    ])
    assert result["status"] == "unresolved"
    assert result["rejected_candidate_ids"] == ["bad"]
    assert result["minimum_common_conclusion"]["status"] == "not_established"
    assert result["policy"]["unresolved_is_not_rejected"] is True


def test_verdict_input_contract_is_strict():
    with pytest.raises(ModelVerdictError, match="duplicate"):
        build_model_verdict([{"id": "a"}, {"id": "a"}])
    with pytest.raises(ModelVerdictError, match="verdict_state"):
        build_model_verdict([{"id": "a", "verdict_state": "certain"}])
    with pytest.raises(ModelVerdictError, match="evidence"):
        build_model_verdict([{"id": "a", "verdict_state": "approved", "status": "confirmed"}])


def test_counterexample_for_approved_candidate_revokes_recommendation():
    result = build_model_verdict(
        [{"id": "a", "verdict_state": "approved", "status": "confirmed",
          "evidence_refs": ["confirmation/a"]}],
        counterexamples=[{"candidate_id": "a", "reason": "boundary_violation"}],
    )
    assert result["approved_candidate_ids"] == []
    assert result["recommended_candidate_id"] is None
    assert result["status"] == "unresolved"


def test_numerical_stability_is_a_layered_status_not_an_approval():
    result = build_model_verdict(
        [{"id": "a", "status": "pass", "decision": "use-a"}],
        numerical_stability={
            "status": "unstable", "successful_runs": 3, "failed_runs": 0,
            "fingerprint": "f" * 64, "runs": [{"outputs": {"secret": 1.0}}],
        },
    )
    assert result["status"] == "conditional"
    assert result["uncertainty"]["numerical"]["status"] == "counterexample_found"
    assert result["numerical_stability"]["status"] == "unstable"
    assert result["numerical_stability"]["successful_runs"] == 3
    assert "runs" not in result["numerical_stability"]
    assert any("数值容差" in warning for warning in result["warnings"])


def test_stable_tolerance_check_is_not_a_numerical_proof():
    result = build_model_verdict(
        [],
        numerical_stability={"status": "stable_on_tested_tolerances", "successful_runs": 3},
    )
    assert result["uncertainty"]["numerical"]["status"] == "tested_not_falsified"
    assert result["numerical_stability"]["policy"].endswith("not_numerical_error_proof")


def test_layered_variance_summary_is_sanitized_and_keeps_units():
    result = build_model_verdict(
        [],
        uncertainty_propagation={
            "status": "ok", "unit_signature": "m/s", "record_count": 4,
            "reconstructed_variance": 0.25,
            "variance_components": {
                "semantic": {"status": "assessed"},
                "structure": {"status": "assessed"},
                "parameter": {"status": "partial"},
                "numerical": {"status": "assessed"},
            },
            "evaluations": [{"secret": "must_not_copy"}],
        },
    )
    assert result["uncertainty"]["semantic"]["status"] == "tested_not_falsified"
    assert result["uncertainty"]["structural"]["status"] == "tested_not_falsified"
    assert result["uncertainty"]["parameter"]["status"] == "not_assessed"
    assert result["uncertainty_propagation"]["unit_signature"] == "m/s"
    assert "evaluations" not in result["uncertainty_propagation"]
