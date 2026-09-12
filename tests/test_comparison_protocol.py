from core.ablation_protocol import build_ablation_manifest
import pytest

from core.comparison_protocol import ComparisonProtocolError, run_preregistered_comparison, summarize_paired_comparison


def _manifest():
    return build_ablation_manifest(
        [{"id": "baseline", "baseline": True}, {"id": "diagnostic_search"}],
        task_ids=["t1", "t2", "t3"], fixed_budget={"seed": 7, "calls": 3},
        final_test_fingerprint="locked-digest", scoring_criteria=["score", "valid"],
    )


def test_comparison_keeps_failures_and_uses_same_context():
    seen = []

    def evaluate(arm, task, context):
        seen.append((arm["id"], task["id"], context["seed"], context["final_test_fingerprint"]))
        if arm["id"] == "diagnostic_search" and task["id"] == "t2":
            return {"status": "rejected", "error_code": "counterexample"}
        return {"status": "completed", "valid": True, "score": 1.0 if arm["id"] == "baseline" else 1.2}

    result = run_preregistered_comparison(_manifest(), [{"id": f"t{i}"} for i in range(1, 4)], evaluate)
    assert result["expected_rows"] == result["observed_rows"] == 6
    assert result["status_counts"]["rejected"] == 1
    assert len(seen) == 6
    assert all(item[2:] == (7, "locked-digest") for item in seen)


def test_summary_never_upgrades_to_significance_and_handles_direction():
    def evaluate(arm, task, context):
        return {"score": 2.0 if arm["id"] == "baseline" else 1.0, "valid": True}

    result = run_preregistered_comparison(_manifest(), [{"id": f"t{i}"} for i in range(1, 4)], evaluate)
    summary = summarize_paired_comparison(result, baseline_arm="baseline", treatment_arm="diagnostic_search",
                                          score_direction="lower_is_better", min_samples=5)
    assert summary["paired_task_count"] == 3
    assert summary["paired_effect"] == 1.0
    assert summary["status"] == "descriptive_only"
    assert "significance" in summary["policy"]


def test_comparison_rejects_duplicate_grid_ids_and_non_boolean_validity():
    manifest = _manifest()
    manifest["task_ids"] = ["t1", "t1"]
    with pytest.raises(ComparisonProtocolError, match="task_ids_must_be_unique_strings"):
        run_preregistered_comparison(manifest, [{"id": "t1"}], lambda *_: {"score": 1.0})

    result = run_preregistered_comparison(
        _manifest(), [{"id": "t1"}, {"id": "t2"}, {"id": "t3"}],
        lambda *_: {"score": 1.0, "valid": "yes"},
    )
    assert result["status_counts"]["error"] == 6
    assert all(row["error_code"] == "valid_must_be_boolean" for row in result["rows"])


def test_comparison_rejects_duplicate_arm_ids():
    manifest = _manifest()
    manifest["arms"] = [manifest["arms"][0], dict(manifest["arms"][0])]
    with pytest.raises(ComparisonProtocolError, match="arm_ids_must_be_unique"):
        run_preregistered_comparison(manifest, [{"id": f"t{i}"} for i in range(1, 4)], lambda *_: {})
