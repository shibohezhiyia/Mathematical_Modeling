import numpy as np

from core.external_method_comparison import (
    EXTERNAL_METHOD_SCORING,
    build_external_method_manifest,
    external_method_arms,
    run_external_method_comparison,
)


def test_external_method_arms_are_distinct_and_have_one_baseline():
    arms = external_method_arms()
    assert len(arms) == 5
    assert sum(item["baseline"] for item in arms) == 1
    assert len({item["id"] for item in arms}) == len(arms)
    assert {item["components"]["method"] for item in arms} == {
        "sindy", "weak_sindy", "pde_find", "ude", "llm_sr",
    }


def test_external_method_manifest_freezes_budget_and_final_test():
    manifest = build_external_method_manifest(
        task_ids=["case-a", "case-b"],
        fixed_budget={"seconds": 30, "api_calls": 0, "seed": 7},
        final_test_fingerprint="locked-final-v1",
    )
    assert manifest["status"] == "preregistered"
    assert manifest["scoring_criteria"] == list(EXTERNAL_METHOD_SCORING)
    assert manifest["final_test_fingerprint"] == "locked-final-v1"
    assert manifest["policy"].startswith("same_tasks_budget_backend")


def _comparison_task(task_id="case-a"):
    times = np.linspace(0.0, 8.0, 64).tolist()
    states = np.column_stack([np.exp(-0.1 * np.asarray(times)),
                              0.5 * np.exp(-0.2 * np.asarray(times))]).tolist()
    target = np.linspace(1.0, 2.0, 16)
    known = np.linspace(0.8, 1.8, 16)
    features = np.column_stack([np.ones(16), np.linspace(0.0, 1.0, 16)]).tolist()
    return {
        "id": task_id,
        "payloads": {
            "sindy": {
                "times": times, "states": states, "state_names": ["x", "y"],
                "polynomial_degree": 1, "residual_tolerance": 10.0,
            },
            "weak_sindy": {
                "times": times, "states": states, "state_names": ["x", "y"],
                "polynomial_degree": 1, "residual_tolerance": 10.0,
                "window_count": 8,
            },
            "pde_find": {
                "field_shape": [8, 9], "coordinate_names": ["x", "t"],
            },
            "ude": {
                "target_rhs": target.tolist(), "known_rhs": known.tolist(),
                "correction_features": features,
            },
            "llm_sr": {"proposal": "not executable"},
        },
    }


def test_external_method_comparison_runs_local_arms_and_keeps_failures():
    manifest = build_external_method_manifest(
        task_ids=["case-a"], fixed_budget={"seconds": 30, "seed": 3},
        final_test_fingerprint="locked-final-v1",
    )
    result = run_external_method_comparison(manifest, [_comparison_task()])
    assert result["expected_rows"] == result["observed_rows"] == 5
    assert result["status"] == "incomplete"
    assert result["primary_metric"] == "validation_residual_rmse"
    assert result["score_direction"] == "lower_is_better"
    assert result["status_counts"]["completed"] == 3
    assert result["status_counts"]["rejected"] == 2
    assert {row["arm_id"] for row in result["rows"] if row["status"] == "completed"} == {
        "local_sindy", "local_weak_sindy", "local_ude",
    }
    assert all(row["score"] is not None for row in result["rows"] if row["status"] == "completed")


def test_external_method_comparison_missing_payload_is_a_rejected_row():
    manifest = build_external_method_manifest(
        task_ids=["case-a"], fixed_budget={"seconds": 30, "seed": 3},
        final_test_fingerprint="locked-final-v1",
    )
    result = run_external_method_comparison(manifest, [{"id": "case-a", "payloads": {}}])
    assert result["observed_rows"] == 5
    assert result["status_counts"]["rejected"] == 5
    assert {row["error_code"] for row in result["rows"]} == {"method_payload_missing"}
