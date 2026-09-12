from core.modeling_policy import audit_modeling_choice


def test_modeling_policy_preserves_baselines_and_does_not_force_axioms():
    result = audit_modeling_choice({"law_forced": False, "black_box_baseline_disabled": False})
    assert result["status"] == "pass"


def test_modeling_policy_rejects_unjustified_forcing_and_pruning():
    result = audit_modeling_choice({"law_forced": True, "fixed_drop_fraction": 0.3, "black_box_baseline_disabled": True})
    assert result["status"] == "fail"
    assert len(result["failed_checks"]) == 3
