from core.standard_comparison_plan import (
    STANDARD_SCORING,
    build_standard_comparison_manifest,
    standard_comparison_arms,
)


def test_standard_comparison_plan_has_one_baseline_and_distinct_methods():
    arms = standard_comparison_arms()
    assert len(arms) == 6
    assert sum(bool(item["baseline"]) for item in arms) == 1
    assert len({item["id"] for item in arms}) == len(arms)
    assert {item["id"] for item in arms} >= {
        "registry_baseline", "single_llm", "automl", "symbolic_regression",
        "mm_agent_style", "diagnostic_search",
    }


def test_standard_comparison_manifest_is_preregistered_and_fixed():
    manifest = build_standard_comparison_manifest(
        task_ids=["case-a", "case-b"],
        fixed_budget={"seconds": 30, "api_calls": 4, "seed": 7},
        final_test_fingerprint="locked-final-v1",
    )
    assert manifest["status"] == "preregistered"
    assert manifest["scoring_criteria"] == list(STANDARD_SCORING)
    assert manifest["policy"].startswith("same_tasks_budget_backend")
