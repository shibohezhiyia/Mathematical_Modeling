"""Baseline-first orchestration for exploration and strict validation."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

from .baseline_evaluation import baseline_first_report
from .staged_evaluation import StageSpec, StagedEvaluationError, run_staged_evaluation


class BaselineStagedRunnerError(ValueError):
    pass


def run_baseline_then_search(
    task: str,
    baseline_value: float,
    candidates: Sequence[Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any], StageSpec], Mapping[str, Any]],
    stages: Sequence[StageSpec],
    *,
    mode: str = "exploration",
    max_candidates: int = 32,
    max_total_budget: int | None = None,
    max_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Return a report whose first section is always the baseline.

    Exploration may use cheap stages without confirmation. Strict validation
    requires one final confirmation stage; beating the baseline alone never
    promotes a candidate to a final claim.
    """
    if mode not in {"exploration", "strict_validation"}:
        raise BaselineStagedRunnerError("invalid_mode")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise BaselineStagedRunnerError("candidates_must_be_sequence")
    if type(max_candidates) is not int or not 0 <= max_candidates <= 64:
        raise BaselineStagedRunnerError("invalid_candidate_budget")
    if len(candidates) > max_candidates:
        raise BaselineStagedRunnerError("candidate_budget_exceeded")
    if not callable(evaluator):
        raise BaselineStagedRunnerError("evaluator_required")
    try:
        baseline = baseline_first_report(task, baseline_value, candidates, max_candidates=max_candidates)
    except Exception as exc:
        raise BaselineStagedRunnerError(f"baseline_invalid:{type(exc).__name__}") from exc
    confirmation_count = sum(1 for stage in stages if getattr(stage, "confirmation", False))
    if mode == "strict_validation" and confirmation_count != 1:
        raise BaselineStagedRunnerError("strict_validation_requires_one_confirmation_stage")
    if not candidates:
        return {"schema_version": "mathmodel.baseline-staged-runner/v1", "mode": mode,
                "baseline": baseline, "search": None, "status": "baseline_only",
                "policy": "baseline_is_first;empty_search_is_not_a_failed_candidate"}
    try:
        search = run_staged_evaluation(candidates, evaluator, stages,
                                       max_total_budget=max_total_budget,
                                       max_wall_seconds=max_wall_seconds)
    except StagedEvaluationError:
        raise
    return {"schema_version": "mathmodel.baseline-staged-runner/v1", "mode": mode,
            "baseline": baseline, "search": search,
            "status": "strict_validation_complete" if mode == "strict_validation" else "exploration_complete",
            "policy": "baseline_is_first;exploration_never_claims_final_evidence;strict_requires_confirmation"}


__all__ = ["BaselineStagedRunnerError", "run_baseline_then_search"]
