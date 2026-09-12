from core.reuse_plan import build_reuse_plan


def test_reuse_plan_requires_matching_signatures_and_cold_comparison():
    snapshot = {"data_view_digest": "d", "training_features_digest": "f", "sparse_structure_digest": "s", "key_partition_digest": "k"}
    result = build_reuse_plan(snapshot, dict(snapshot), parameter_mapping={"old": "new"})
    assert result["status"] == "reusable_intermediates"
    assert result["warm_start"] == "allowed"
    changed = dict(snapshot); changed["data_view_digest"] = "new"
    assert build_reuse_plan(snapshot, changed, parameter_mapping={"old": "new"})["warm_start"] == "denied"
