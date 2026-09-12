from core.search_ablation import COMPONENTS, build_search_ablation_manifest, run_search_ablation


def _manifest():
    return build_search_ablation_manifest(task_ids=["a", "b"], fixed_budget={"seed": 3, "steps": 4},
                                          final_test_fingerprint="f" * 64, scoring_criteria=["valid"])


def test_search_ablation_registers_each_control_as_one_change():
    manifest = _manifest()
    assert len(manifest["arms"]) == len(COMPONENTS) + 1
    full = manifest["arms"][0]["components"]
    for arm in manifest["arms"][1:]:
        changed = [name for name in COMPONENTS if arm["components"][name] != full[name]]
        assert len(changed) == 1


def test_search_ablation_keeps_failures_in_paired_grid():
    manifest = _manifest()

    def evaluate(arm, task, context):
        if arm["id"] == "without_counterexample_replay" and task["id"] == "b":
            return {"status": "error", "error_code": "controlled_failure"}
        return {"status": "completed", "valid": True, "score": 1.0 if arm["id"] == "full_search" else 0.9}

    result = run_search_ablation(manifest, [{"id": "a"}, {"id": "b"}], evaluate, baseline_arm="full_search",
                                 treatment_arm="without_counterexample_replay")
    assert result["comparison"]["status_counts"]["error"] == 1
    assert result["summary"]["paired_task_count"] == 1
