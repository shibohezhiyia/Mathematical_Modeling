import pytest

from core.ablation_protocol import AblationProtocolError, build_ablation_manifest, record_ablation_results


def _manifest():
    return build_ablation_manifest([
        {"id": "baseline", "baseline": True, "components": {"search": False}},
        {"id": "treatment", "components": {"search": True}},
    ], task_ids=["t1", "t2"], fixed_budget={"seconds": 10, "api_calls": 2},
        final_test_fingerprint="locked", scoring_criteria=["valid_rate", "score"])


def test_ablation_protocol_keeps_failures_in_complete_grid_denominator():
    manifest = _manifest()
    result = record_ablation_results(manifest, [
        {"arm_id": "baseline", "task_id": "t1", "status": "completed", "valid": True, "score": 1},
        {"arm_id": "baseline", "task_id": "t2", "status": "timeout"},
        {"arm_id": "treatment", "task_id": "t1", "status": "completed", "valid": True, "score": 2},
        {"arm_id": "treatment", "task_id": "t2", "status": "rejected"},
    ])
    assert result["status"] == "assessed"
    assert result["status_counts"]["timeout"] == 1
    assert "denominator" in result["policy"]


def test_ablation_protocol_rejects_missing_baseline_or_duplicate_result():
    with pytest.raises(AblationProtocolError, match="baseline"):
        build_ablation_manifest([{"id": "a", "components": {}}], task_ids=["t"], fixed_budget={"x": 1}, final_test_fingerprint="f", scoring_criteria=["s"])
    manifest = _manifest()
    with pytest.raises(AblationProtocolError, match="status"):
        record_ablation_results(manifest, [{"arm_id": "baseline", "task_id": "t1", "status": "unknown"}])


def test_ablation_protocol_rejects_ambiguous_grid_and_result_types():
    with pytest.raises(AblationProtocolError, match="task_ids_required"):
        build_ablation_manifest([{"id": "a", "baseline": True}], task_ids=["t", "t"],
                                fixed_budget={"x": 1}, final_test_fingerprint="f", scoring_criteria=["s"])
    with pytest.raises(AblationProtocolError, match="baseline_must_be_boolean"):
        build_ablation_manifest([{"id": "a", "baseline": "yes"}], task_ids=["t"],
                                fixed_budget={"x": 1}, final_test_fingerprint="f", scoring_criteria=["s"])
    manifest = _manifest()
    with pytest.raises(AblationProtocolError, match="valid_must_be_boolean"):
        record_ablation_results(manifest, [{"arm_id": "baseline", "task_id": "t1",
                                            "status": "completed", "valid": "yes", "score": 1}])
    with pytest.raises(AblationProtocolError, match="score_invalid"):
        record_ablation_results(manifest, [{"arm_id": "baseline", "task_id": "t1",
                                            "status": "completed", "valid": True, "score": float("nan")}])
