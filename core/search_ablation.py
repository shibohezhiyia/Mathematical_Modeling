"""Pre-registered ablations for the search-control mechanisms.

The protocol compares the complete search pipeline with one disabled control
at a time.  It is intentionally evaluator-agnostic: a result is evidence only
after a real runner supplies every arm×task row under the frozen budget.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .ablation_protocol import build_ablation_manifest
from .comparison_protocol import run_preregistered_comparison, summarize_paired_comparison


class SearchAblationError(ValueError):
    pass


COMPONENTS = ("staged_elimination", "incremental_cache", "diagnostic_routing",
              "prior_constraints", "counterexample_replay")


def build_search_ablation_manifest(*, task_ids: Sequence[str], fixed_budget: Mapping[str, Any],
                                   final_test_fingerprint: str,
                                   scoring_criteria: Sequence[str]) -> dict[str, Any]:
    full = {name: True for name in COMPONENTS}
    arms = [{"id": "full_search", "label": "full search controls", "components": full, "baseline": True}]
    for component in COMPONENTS:
        disabled = dict(full)
        disabled[component] = False
        arms.append({"id": f"without_{component}", "label": f"without {component}",
                     "components": disabled, "baseline": False})
    return build_ablation_manifest(arms, task_ids=task_ids, fixed_budget=fixed_budget,
                                   final_test_fingerprint=final_test_fingerprint,
                                   scoring_criteria=scoring_criteria)


def run_search_ablation(manifest: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]],
                        evaluate: Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
                        *, wall_seconds: float | None = None, baseline_arm: str = "full_search",
                        treatment_arm: str = "without_counterexample_replay") -> dict[str, Any]:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "mathmodel.ablation-manifest/v1":
        raise SearchAblationError("manifest_required")
    ids = {str(arm.get("id")) for arm in manifest.get("arms", []) if isinstance(arm, Mapping)}
    if baseline_arm not in ids or treatment_arm not in ids or baseline_arm == treatment_arm:
        raise SearchAblationError("comparison_arms_required")
    result = run_preregistered_comparison(manifest, tasks, evaluate, wall_seconds=wall_seconds)
    summary = summarize_paired_comparison(result, baseline_arm=baseline_arm, treatment_arm=treatment_arm,
                                          score_direction="higher_is_better")
    return {"schema_version": "mathmodel.search-ablation-results/v1", "comparison": result,
            "summary": summary,
            "policy": "one_control_disabled_per_arm; descriptive_until_real_tasks_and_independent_final_test"}


__all__ = ["COMPONENTS", "SearchAblationError", "build_search_ablation_manifest", "run_search_ablation"]
