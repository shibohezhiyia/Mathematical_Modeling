import json

import pytest

from core.experience_store import ExperienceStore, ExperienceStoreError, redact_payload


def _recipe(secret="sk-test-secret-value"):
    return {
        "problem_contract": {"target": "y", "assumption": "closed system"},
        "ir": {"nodes": ["x", "exp", "y"], "constraints": ["y>=0"]},
        "execution_plan": {"solver": "bounded_ode", "timeout_seconds": 2},
        "certificates": {"residual_rmse": 0.12},
        "failure_trace": [],
        "api_key": secret,
        "raw_data": [{"x": 1, "y": 2}],
        "source_path": r"C:\Users\Alice\secret.xlsx",
    }


def test_store_is_content_addressed_and_redacts_sensitive_payload(tmp_path):
    store = ExperienceStore(tmp_path / "experiences")
    first = store.put(_recipe(), graph_signature=["exp", "coupling"],
                      unit_signature=["L", "T^-1"], objective_signature=["rmse"],
                      run_id="run-1", topic_key="cooling")
    second = store.put(_recipe(), graph_signature=["exp", "coupling"],
                       unit_signature=["L", "T^-1"], objective_signature=["rmse"],
                       run_id="run-1", topic_key="cooling")
    assert first["experience_id"] == second["experience_id"]
    payload = store.get(first["experience_id"])["payload"]
    encoded = json.dumps(payload, ensure_ascii=False)
    assert "sk-test-secret-value" not in encoded
    assert "Alice" not in encoded
    assert payload["recipe"]["raw_data"]["redacted"] is True
    assert payload["recipe"]["api_key"] == "<redacted-secret>"


def test_verified_requires_replay_counterexample_and_security_evidence(tmp_path):
    store = ExperienceStore(tmp_path / "experiences")
    record = store.put(_recipe(), graph_signature=["ode"])
    with pytest.raises(ExperienceStoreError, match="复现、反例和安全"):
        store.promote_verified(record["experience_id"], {"replay_passed": True})
    verified = store.promote_verified(record["experience_id"], {
        "replay_passed": True,
        "counterexamples_checked": True,
        "security_checked": True,
        "validation_scope": {"dataset_digest": "abc", "solver_version": "v1"},
        "api_key": "sk-another-secret",
    })
    assert verified["status"] == "verified"
    assert verified["verification"]["api_key"] == "<redacted-secret>"
    assert verified["seed_only"] is True


def test_search_uses_graph_units_and_objective_signatures_and_marks_seed_only(tmp_path):
    store = ExperienceStore(tmp_path / "experiences")
    near = store.put(_recipe(), graph_signature=["ode", "coupling"],
                     unit_signature=["L", "T^-1"], objective_signature=["rmse"])
    store.put(_recipe("sk-other"), graph_signature=["classification"],
              unit_signature=["count"], objective_signature=["accuracy"])
    results = store.search(graph_signature=["ode", "coupling"],
                           unit_signature=["L"], objective_signature=["rmse"])
    assert results[0]["experience_id"] == near["experience_id"]
    assert results[0]["seed_only"] is True
    assert results[0]["requires_current_validation"] is True


def test_purge_defaults_to_provisional_and_preserves_verified(tmp_path):
    store = ExperienceStore(tmp_path / "experiences")
    provisional = store.put(_recipe(), run_id="run-a", topic_key="topic-a")
    verified = store.put(_recipe("sk-verified"), graph_signature=["verified"],
                         run_id="run-b", topic_key="topic-b")
    store.promote_verified(verified["experience_id"], {
        "replay_passed": True, "counterexamples_checked": True,
        "security_checked": True, "validation_scope": {"split": "holdout"},
    })
    preview = store.purge(run_id="run-a", dry_run=True)
    assert preview["matched"] == 1
    assert store.get(provisional["experience_id"])["status"] == "provisional"
    store.purge(run_id="run-a", dry_run=False)
    with pytest.raises(ExperienceStoreError, match="经验不存在"):
        store.get(provisional["experience_id"])
    assert store.get(verified["experience_id"])["status"] == "verified"


def test_verified_deletion_requires_explicit_scope(tmp_path):
    store = ExperienceStore(tmp_path / "experiences")
    record = store.put(_recipe(), run_id="run-c")
    store.promote_verified(record["experience_id"], {
        "replay_passed": True, "counterexamples_checked": True,
        "security_checked": True, "validation_scope": {"split": "holdout"},
    })
    with pytest.raises(ExperienceStoreError, match="明确范围"):
        store.purge(statuses=("verified",), dry_run=True)


def test_redaction_rejects_cycles_and_excessive_nesting_before_persistence():
    cyclic = {}
    cyclic["self"] = cyclic
    with pytest.raises(ExperienceStoreError, match="循环引用"):
        redact_payload(cyclic)
    nested = value = {}
    for index in range(34):
        value["next"] = {}
        value = value["next"]
    with pytest.raises(ExperienceStoreError, match="超过脱敏预算"):
        redact_payload(nested)
