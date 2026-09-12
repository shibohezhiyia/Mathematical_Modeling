"""Pre-registered comparison arms for the open-world modeling roadmap."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .ablation_protocol import build_ablation_manifest


STANDARD_SCORING = (
    "valid_rate", "contract_correct", "evidence_completeness",
    "stability", "latency_seconds", "peak_memory_mb", "api_cost",
)


def standard_comparison_arms() -> list[dict[str, Any]]:
    """Return fixed labels/components; components are routing declarations only."""
    return [
        {"id": "registry_baseline", "label": "当前注册表基线", "baseline": True,
         "components": {"structure": "registered_templates", "diagnostics": False,
                         "counterexamples": False, "active_questions": False}},
        {"id": "single_llm", "label": "单次 LLM 提议", "baseline": False, "components":
         {"structure": "one_external_proposal", "diagnostics": False,
          "counterexamples": False, "active_questions": False}},
        {"id": "automl", "label": "AutoML 基线", "baseline": False, "components":
         {"structure": "algorithm_selector", "diagnostics": False,
          "counterexamples": False, "active_questions": False}},
        {"id": "symbolic_regression", "label": "符号回归基线", "baseline": False, "components":
         {"structure": "bounded_symbolic_library", "diagnostics": False,
          "counterexamples": False, "active_questions": False}},
        {"id": "mm_agent_style", "label": "MM-Agent 风格四阶段", "baseline": False, "components":
         {"structure": "analysis_formulation_compute_report", "diagnostics": False,
          "counterexamples": False, "active_questions": False}},
        {"id": "diagnostic_search", "label": "诊断驱动开放组合", "baseline": False, "components":
         {"structure": "typed_primitive_graph", "diagnostics": True,
          "counterexamples": True, "active_questions": True}},
    ]


def build_standard_comparison_manifest(*, task_ids: Sequence[str], fixed_budget: Mapping[str, Any],
                                       final_test_fingerprint: str) -> dict[str, Any]:
    """Build a reproducible manifest; no evaluator is called here."""
    return build_ablation_manifest(
        standard_comparison_arms(), task_ids=task_ids, fixed_budget=fixed_budget,
        final_test_fingerprint=final_test_fingerprint, scoring_criteria=STANDARD_SCORING,
    )


__all__ = ["STANDARD_SCORING", "standard_comparison_arms", "build_standard_comparison_manifest"]
