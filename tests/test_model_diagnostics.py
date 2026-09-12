"""N06: bounded diagnostics, competing explanations and holdout-safe feedback."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.integral_dynamics import discover_integral_dynamics
from core.model_diagnostics import build_model_diagnostics, diagnose_development


def design_case(n=120):
    rng = np.random.default_rng(41)
    train = np.column_stack([np.ones(n * 2), rng.normal(size=n * 2)])
    search = np.column_stack([np.ones(n), rng.normal(size=n)])
    coefficients = np.array([[0.3, 1.2]])
    return dict(design_train=train, response_train=train @ coefficients.T,
                design_search=search, response_search=search @ coefficients.T,
                coefficients=coefficients, search_starts=np.arange(n) * 2, window_points=1,
                training_fingerprint="a" * 64, term_names=["1", "x"], state_names=["x"],
                support_jaccard=1.0)


def codes(result):
    return {item["code"] for item in result["records"]}


def oscillator(n=300):
    time = np.arange(n) * 0.2
    return pd.DataFrame({"date": pd.Timestamp("2024-01-01") + pd.to_timedelta(time, unit="D"),
                         "state": np.sin(0.2 * time), "driver": np.cos(0.2 * time)})


def discover(frame):
    return discover_integral_dynamics(frame, time_column="date", target_column="state")


def test_well_specified_design_does_not_manufacture_a_hidden_state():
    result = diagnose_development(**design_case())
    assert result["records"] == []
    assert result["checks"]["candidate_library"]["rank"] == 2
    assert result["checks"]["residuals"][0]["lag1_correlation"] is None
    assert result["may_execute"] is False


@pytest.mark.parametrize("epsilon", [0.0, 1e-10])
def test_rank_or_condition_identifies_active_parameter_problem(epsilon):
    case = design_case()
    rng = np.random.default_rng(90)
    for key in ("design_train", "design_search"):
        matrix = case[key]
        case[key] = np.column_stack([matrix, matrix[:, 1] + rng.normal(size=len(matrix)) * epsilon])
    case.update(coefficients=np.array([[0.3, 0.6, 0.6]]), term_names=["1", "x", "near_x"])
    result = diagnose_development(**case)
    assert "active_parameter_instability" in codes(result)
    if epsilon == 0:
        assert result["checks"]["active_designs"][0]["condition_scaled"] is None
        assert "redundant_candidate_basis" in codes(result)
    assert all(item["mathematical_verdict"] == "not_assessed" for item in result["records"])


def test_unused_redundant_basis_does_not_prove_active_parameters_unidentifiable():
    case = design_case()
    for key in ("design_train", "design_search"):
        case[key] = np.column_stack([case[key], case[key][:, 1]])
    case.update(coefficients=np.array([[0.3, 1.2, 0]]), term_names=["1", "x", "duplicate"])
    result = diagnose_development(**case)
    assert "redundant_candidate_basis" in codes(result)
    assert "active_parameter_instability" not in codes(result)


def test_weak_support_has_parameter_route_not_a_claim_of_new_physics():
    case = design_case()
    case["support_jaccard"] = 0.2
    result = diagnose_development(**case)
    record = next(item for item in result["records"] if item["code"] == "unstable_training_support")
    assert record["category"] == "parameter"
    assert "inspect_parameterization" in record["action_ids"]


def test_biased_measurements_keep_noise_and_mechanism_explanations_competing():
    case = design_case()
    case["response_search"] += 3
    result = diagnose_development(**case)
    assert "search_residual_bias" in codes(result)
    record = next(item for item in result["records"] if item["code"] == "search_residual_bias")
    assert "测量偏置" in record["alternative_explanations"]
    assert "遗漏驱动或机制" in record["alternative_explanations"]
    assert record["state"] == "suspected"


def test_correlated_observation_noise_does_not_become_proven_latent_state():
    case = design_case()
    # The generating process is unchanged; only observation noise has memory.
    noise = np.sin(np.arange(120) * 0.15)
    case["response_search"][:, 0] += noise
    result = diagnose_development(**case)
    record = next(item for item in result["records"] if item["code"] == "search_residual_memory")
    assert "相关观测噪声" in record["alternative_explanations"]
    assert record["state"] == "suspected"
    assert record["authority"] == "diagnostic_only"
    assert all(action["may_execute"] is False for action in result["actions"])


def test_overlapping_or_gapped_windows_are_not_stitched_into_fake_neighbors():
    case = design_case(n=36)
    case["window_points"] = 12
    case["search_starts"] = np.arange(36)
    case["response_search"][:, 0] += np.sin(np.arange(36) * 0.1)
    result = diagnose_development(**case)
    assert result["checks"]["residuals"][0]["disjoint_windows"] == 3
    assert "search_residual_memory" not in codes(result)
    assert result["checks"]["residuals"][0]["pattern_status"] == "insufficient_disjoint_windows"
    case["window_points"], case["search_starts"] = 1, np.arange(36) * 3
    result = diagnose_development(**case)
    assert result["checks"]["residuals"][0]["adjacent_pairs"] == 0
    assert result["checks"]["residuals"][0]["lag1_correlation"] is None


def test_prediction_scale_pattern_only_proposes_observation_hypothesis():
    case = design_case()
    predictions = case["response_search"][:, 0].copy()
    case["response_search"][:, 0] += np.abs(predictions) * np.where(np.arange(120) % 2, 1, -1)
    result = diagnose_development(**case)
    record = next(item for item in result["records"] if item["code"] == "search_error_scale_pattern")
    assert record["state"] == "suspected"
    assert "异方差观测" in record["alternative_explanations"]


def test_nonfinite_computation_routes_to_numerics_not_structure():
    case = design_case()
    case["design_search"][:] = 1e308
    case["coefficients"][:] = 1e308
    result = diagnose_development(**case)
    assert "development_nonfinite_prediction" in codes(result)
    assert "search_baseline_not_improved" not in codes(result)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("mutation", [
    lambda case: case.update(design_train=np.zeros((5001, 2))),
    lambda case: case.update(design_search=np.zeros((2, 65))),
    lambda case: case.update(coefficients=np.ones((1, 3))),
    lambda case: case.update(window_points=True),
    lambda case: case.update(search_starts=np.arange(120)[::-1]),
    lambda case: case.update(response_search=np.full((120, 1), np.nan)),
    lambda case: case.update(support_jaccard=2),
])
def test_diagnostic_budget_and_alignment_guards(mutation):
    case = design_case()
    mutation(case)
    with pytest.raises(ValueError):
        diagnose_development(**case)


@pytest.mark.parametrize("field", ["response_search", "response_train", "coefficients"])
def test_development_context_hash_changes_with_evidence(field):
    case = design_case()
    first = diagnose_development(**case)
    case[field][0, 0] += 0.1
    second = diagnose_development(**case)
    assert first["context"]["input_hash"] != second["context"]["input_hash"]


def test_development_diagnostics_expose_competing_unclosed_state_explanations():
    result = diagnose_development(**design_case())
    competition = result["checks"]["unclosed_state_competition"]
    assert competition["status"] == "screening_only"
    assert competition["policy"]["latent_state_not_proven"] is True
    assert any(item["id"] == "heteroscedastic_observation"
               for item in competition["candidate_explanations"])


@pytest.mark.parametrize("code,category,action", [
    ("timeout", "resource", "inspect_resources"), ("memory_limit", "resource", "inspect_resources"),
    ("numeric_domain", "numerical", "replay_numerics"), ("evaluation_limit", "resource", "replay_numerics"),
    ("invalid_contract", "semantic", "confirm_semantics"), ("upstream_failed", "dependency", "resolve_upstream"),
    ("worker_exit", "execution", "replay_numerics"),
])
def test_structured_execution_diagnosis_does_not_read_raw_tracebacks(code, category, action):
    mechanism = {"problem_fingerprint": "a" * 64, "solver_execution": {"failures": [{
        "relation_id": "r", "failure_code": code, "message": "SECRET_FULL_STACK",
    }]}}
    result = build_model_diagnostics(mechanistic=mechanism)
    assert result["records"][0]["category"] == category
    assert result["records"][0]["action_ids"] == [action]
    assert "SECRET_FULL_STACK" not in json.dumps(result)


def test_invalid_unit_and_hypothesis_graph_get_different_routes():
    result = build_model_diagnostics(
        mechanistic={"mathematical_ir": {"relations": [{"id": "bad", "parse_status": "pending",
                    "validation_errors": ["rhs_dimension_mismatch:x"]}]}},
        proposals={"contract_hash": "a" * 64, "rejected_proposals": [{"index": 0, "code": "shape_mismatch", "node_id": "v"}]},
    )
    assert "unit_binding_unresolved" in codes(result)
    assert "hypothesis_type_rejected" in codes(result)
    assert {item["target"] for item in result["proposed_actions"]} == {"semantic", "structure"}


def test_final_test_mutation_cannot_change_development_feedback_or_model():
    frame = oscillator()
    changed = frame.copy()
    changed.loc[240:, ["state", "driver"]] += 40
    original, altered = discover(frame), discover(changed)
    assert original["development_diagnostics"] == altered["development_diagnostics"]
    assert original["development_diagnostics"].get("status") != "failed_safe"
    assert original["system_equations"] == altered["system_equations"]
    first, second = build_model_diagnostics(dynamics=original), build_model_diagnostics(dynamics=altered)
    assert first["search_feedback"] == second["search_feedback"]
    assert second["records"][-1]["context"]["phase"] == "final_test"
    assert second["records"][-1]["context"]["may_feed_search"] is False
    assert second["policy"]["latent_state_discovery_proven"] is False


def test_final_initial_value_failure_stays_in_reporting_channel():
    result = build_model_diagnostics(dynamics={
        "evaluation_protocol": "train_search_locked_test", "trajectory_test": {
            "status": "not_assessed", "reason": "initial_state_not_observed", "solver_success": False,
        },
    })
    assert result["records"][0]["category"] == "data"
    assert result["search_feedback"]["records"] == []
    assert result["search_feedback"]["actions"] == []


def test_passing_rollout_cannot_hide_a_failed_integral_audit():
    result = build_model_diagnostics(dynamics={
        "evaluation_protocol": "train_search_locked_test", "trajectory_test": {"status": "pass", "solver_success": True},
        "credibility_audit": {"checks": [{"id": "dynamics_integral_test", "status": "fail"}]},
    })
    assert "locked_test_not_confirmed" in codes(result)
    assert result["search_feedback"]["records"] == []


def test_diagnosis_failure_does_not_drop_an_existing_fitted_model(monkeypatch):
    from core import model_diagnostics

    baseline = discover(oscillator())

    def fail(**kwargs):
        raise RuntimeError("SECRET_DIAGNOSIS_ERROR")

    monkeypatch.setattr(model_diagnostics, "diagnose_development", fail)
    result = discover(oscillator())
    assert result["system_equations"] == baseline["system_equations"]
    assert result["trajectory_test"]["metrics"] == baseline["trajectory_test"]["metrics"]
    assert "development_diagnostics_unavailable" in codes(build_model_diagnostics(dynamics=result))
    assert "SECRET_DIAGNOSIS_ERROR" not in str(result)


def test_no_data_no_failure_does_not_mean_all_models_are_correct():
    result = build_model_diagnostics()
    assert result["status"] == "no_findings_in_checked_scope"
    assert result["coverage"]["all_model_families_covered"] is False
    assert result["policy"]["diagnostics_are_proof"] is False


def test_prediction_diagnostics_join_common_feedback_without_raw_prediction_payload():
    secret = ["private-label", 1, 2, 3]
    result = build_model_diagnostics(prediction_results=[{
        "dataset": "sales", "target": "amount", "task_type": "regression",
        "actual": secret, "oof_prediction": secret,
        "feedback_optimization": {"diagnostics": {
            "primary_metric": "rmse", "fold_relative_std": 0.31,
            "normalized_bias": 0.22, "residual_prediction_correlation": 0.3,
            "heteroscedasticity_signal": 0.4, "recommendations": ["SECRET"],
        }, "_confirmation_actual": secret},
    }])
    assert result["coverage"]["prediction_models_checked"] == 1
    assert result["coverage"]["prediction_models_with_diagnostics"] == 1
    assert {item["code"] for item in result["records"]} >= {
        "prediction_cv_instability", "prediction_residual_bias",
        "prediction_residual_structure", "prediction_heteroscedasticity_signal",
    }
    serialized = json.dumps(result, ensure_ascii=False)
    assert "private-label" not in serialized
    assert "SECRET" not in serialized
    assert "oof_prediction" not in serialized
    assert all(item["context"]["partition"] == "oof_and_inner_cv_only"
               for item in result["records"] if item["code"].startswith("prediction_"))


def test_prediction_diagnostics_ignores_untrusted_non_mapping_and_locked_arrays():
    result = build_model_diagnostics(prediction_results=[None, {
        "dataset": "x", "target": "y", "task_type": "classification",
        "actual": ["should-not-copy"],
        "feedback_optimization": {"diagnostics": {"oof_error_rate": 0.4}},
    }])
    assert result["coverage"]["prediction_models_checked"] == 1
    assert "classification_oof_error" in codes(result)
    assert "should-not-copy" not in json.dumps(result)


def test_clustering_audit_warnings_join_common_feedback_without_cluster_labels():
    result = build_model_diagnostics(model_results=[{
        "dataset": "customers", "task_type": "clustering", "best_k": 3,
        "cluster_labels": [0, 1, 0],
        "credibility_audit": {"checks": [
            {"id": "cluster_separation", "status": "warning", "message": "弱分离"},
            {"id": "cluster_seed_stability", "status": "fail", "message": "SECRET"},
            {"id": "cluster_size_balance", "status": "pass", "message": "ok"},
        ]},
    }])
    assert result["coverage"]["clustering_models_checked"] == 1
    assert {item["code"] for item in result["records"]} == {
        "clustering_separation_weak", "clustering_seed_instability",
    }
    assert "SECRET" not in json.dumps(result)
    assert "cluster_labels" not in json.dumps(result)


def test_temporal_structure_signals_are_search_hints_not_mechanism_claims():
    result = build_model_diagnostics(structure_results=[{
        "dataset": "series", "temporal_structure": {
            "schema_version": "mathmodel.structure-diagnostics/v1",
            "diagnostics": {"temperature": {
                "status": "signal", "signals": ["periodic_candidate", "change_point_candidate"],
                "periodicity": {"autocorrelation": 0.8},
                "monotone": {"score": 0.1}, "change_point": {"score": 1.4},
            }},
        },
    }])
    assert result["coverage"]["temporal_structure_sets_checked"] == 1
    assert {item["code"] for item in result["records"]} == {
        "periodic_structure_signal", "change_point_signal",
    }
    assert all(item["mathematical_verdict"] == "not_assessed" for item in result["records"])


def test_specialized_audit_warnings_are_read_only_and_bounded():
    result = build_model_diagnostics(specialized_results={
        "optimization": {"dataset": "plan", "credibility_audit": {"status": "fail", "label": "不可信"}},
        "causal_effect": {"dataset": "study", "credibility_audit": {"status": "warning", "label": "条件性"}},
        "ranking_result": {"dataset": "cities", "credibility_audit": {"status": "pass"}},
    })
    assert result["coverage"]["specialized_audits_checked"] == 3
    assert {item["code"] for item in result["records"]} == {
        "optimization_audit_warning", "causal_audit_warning",
    }
    assert all(item["context"]["may_feed_search"] is True for item in result["records"])


def test_input_payload_is_immutable_and_diagnostics_budget_is_explicit():
    mechanism = {"solver_execution": {"failures": [
        {"failure_code": "timeout", "relation_id": str(index)} for index in range(70)
    ]}}
    original = copy.deepcopy(mechanism)
    result = build_model_diagnostics(mechanistic=mechanism)
    assert mechanism == original
    assert len(result["records"]) == 64
    assert result["dropped_diagnostics"] == 6


def test_research_writes_separate_diagnostic_artifact_and_does_not_promote_claims(tmp_path):
    from core.modeling_assistant import MathModelingAssistant
    from core.artifact_manager import RunArtifactManager

    result = MathModelingAssistant(output_dir=str(tmp_path), feedback_optimization=False).run(
        "研究状态的微分方程与动力学", {"series": oscillator()}, target="series.state",
        run_modeling=False, generate_plots=False,
    )
    panel = result.specialized_results["model_diagnostics"]
    assert panel["policy"]["may_execute_repairs"] is False
    artifact = tmp_path / "evidence" / "model_diagnostics.json"
    assert json.loads(artifact.read_text(encoding="utf-8")) == panel
    manifest = json.loads(Path(result.artifact_manifest_path).read_text(encoding="utf-8"))
    item = next(item for item in manifest["artifacts"] if item["id"] == "diagnostics.model")
    assert item["metadata"]["role"] == "diagnostic_not_proof"
    report = Path(result.report_path).read_text(encoding="utf-8")
    assert "诊断与修复方向" in report
    assert "锁定测试不回传搜索" in report
    assert not any(item.get("claim_type") == "model_diagnostics" for item in result.evidence_bundle["claims"])
    manager = RunArtifactManager.open_existing(tmp_path)
    assert artifact.exists()  # A durable, scoped diagnosis; not an executable recipe.
