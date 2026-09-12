from core.experience_store import ExperienceStore


def test_verified_recipe_becomes_stale_when_solver_scope_changes(tmp_path):
    store = ExperienceStore(tmp_path / "store")
    item = store.put({"ir": {"id": "x"}}, graph_signature=["x"])
    store.promote_verified(item["experience_id"], {
        "replay_passed": True, "counterexamples_checked": True, "security_checked": True,
        "validation_scope": {"solver_version": "1", "schema": "a"},
    })
    assert store.assess_revalidation(item["experience_id"], {"solver_version": "2", "schema": "a"})["status"] == "stale"
