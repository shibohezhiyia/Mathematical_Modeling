import pytest
from pathlib import Path

from core.benchmark_material import (
    BenchmarkMaterialError,
    build_material_plan,
    validate_material_plan,
)
from core.benchmark_sources import BenchmarkSourceRegistry


def _registry():
    return BenchmarkSourceRegistry.load(Path(__file__).resolve().parents[1] / "examples" / "benchmark_sources.json")


def test_material_plan_binds_each_role_and_is_digest_protected():
    plan = build_material_plan(
        _registry(), case_id="external-2025-mcm-c",
        input_source_ids=["comap-2025-mcm-c-problem"],
        rubric_source_ids=["comap-2025-rules", "comap-umap-46-4-commentary"],
        comparison_source_ids=["mm-agent-mm-bench"],
        reference_source_ids=["community-2025-mcm-c"],
    )
    assert validate_material_plan(_registry(), plan)["status"] == "pass"
    tampered = dict(plan)
    tampered["case_id"] = "other"
    with pytest.raises(BenchmarkMaterialError, match="material_plan_digest_mismatch"):
        validate_material_plan(_registry(), tampered)


def test_material_plan_rejects_public_solution_in_input_or_rubric_role():
    with pytest.raises(BenchmarkMaterialError, match="source_role_mismatch"):
        build_material_plan(
            _registry(), case_id="case-a",
            input_source_ids=["community-2025-mcm-c"],
            rubric_source_ids=["comap-2025-rules"],
        )


def test_material_plan_rejects_reusing_source_across_roles():
    with pytest.raises(BenchmarkMaterialError, match="source_reused_across_material_roles"):
        build_material_plan(
            _registry(), case_id="case-a",
            input_source_ids=["comap-2025-rules"],
            rubric_source_ids=["comap-2025-rules"],
        )
