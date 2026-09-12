import json
from pathlib import Path

import pytest

from core.benchmark_sources import (
    BenchmarkSourceError,
    BenchmarkSourceRegistry,
)
from core.rubric_review import build_rubric


def _registry():
    path = Path(__file__).resolve().parents[1] / "examples" / "benchmark_sources.json"
    return BenchmarkSourceRegistry.load(path)


def test_public_registry_contains_usable_roles_and_never_gold_truth():
    registry = _registry()
    assert len(registry.sources) >= 10
    assert registry.by_role("task_input")
    assert registry.by_role("rubric_material")
    assert registry.by_role("method_comparison")
    assert registry.by_role("reference_only")
    assert all(item.role != "gold_truth" for item in registry.sources)
    assert all(
        item.role == "reference_only"
        for item in registry.sources
        if item.kind == "reference_solution"
    )


def test_registry_includes_external_method_and_unseen_problem_sources():
    registry = _registry()
    ids = {item.source_id for item in registry.sources}
    assert {
        "comap-2023-problem-index",
        "comap-2024-problem-index",
        "pysindy-official-docs",
        "pysindy-weak-pde-example",
        "sciml-ude-docs",
        "ude-original-paper",
        "llm-sr-official-code",
        "llm-srbench-paper",
    } <= ids
    assert {item.source_id for item in registry.by_role("method_comparison")} >= {
        "pysindy-official-docs", "sciml-ude-docs", "llm-sr-official-code",
    }


def test_registry_serialization_is_metadata_only():
    payload = _registry().public_metadata()
    assert payload["gold_truth_policy"].startswith("no_public_source_is_gold_truth")
    assert all("statement" not in row and "solution_text" not in row for row in payload["sources"])


def test_registry_rejects_reference_solution_as_gold_truth():
    with pytest.raises(BenchmarkSourceError, match="reference_solution_cannot_be_gold_truth"):
        BenchmarkSourceRegistry.from_payload({
            "schema_version": "mathmodel.benchmark-source-registry/v1",
            "sources": [{
                "id": "bad", "kind": "reference_solution", "title": "bad",
                "url": "https://example.com/bad", "authority": "community",
                "role": "task_input", "access": "public",
                "license_status": "metadata_only",
            }],
        })


def test_officially_informed_rubric_template_is_compatible_with_review_protocol():
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "examples" / "benchmark_rubric_mcm_icm.json").read_text(encoding="utf-8"))
    rubric = build_rubric(payload["criteria"], scale_max=payload["scale"]["max"],
                          minimum_reviewers=payload["minimum_reviewers"])
    assert rubric["criteria"] == payload["criteria"]
    assert payload["policy"].endswith("not_gold_truth")


def test_repository_external_source_registry_is_loadable():
    path = Path(__file__).resolve().parents[1] / "data" / "benchmarks" / "source_registry.json"
    registry = BenchmarkSourceRegistry.load(path)
    assert len(registry.public_metadata()["sources"]) == 7
    assert {item.source_id for item in registry.by_role("rubric_material")} >= {
        "immc-official-review-process", "immc-judge-guide-2021",
    }
    assert any(item.role == "method_comparison" for item in registry.sources)
