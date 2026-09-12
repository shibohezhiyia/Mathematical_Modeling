"""Generic competition wrapper for model candidates.

This module keeps numeric Pareto comparison and evidence verdicts together,
without selecting a winner from an uncalibrated score.  Candidates missing
predictions/metrics remain visible as unresolved instead of being silently
dropped from the denominator.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .model_set_assessment import ModelSetAssessmentError, assess_model_set
from .model_verdict import build_model_verdict
from .staged_evaluation import StageSpec, run_staged_evaluation


class ModelCompetitionError(ValueError):
    pass


_UNRESOLVED = frozenset({
    "execution_error", "evaluation_error", "budget_exhausted", "needs_input",
    "unidentifiable", "candidate_set_inadequate", "not_assessed", "pending",
})
_REQUIRED_METRICS = ("validation_loss", "complexity", "constraint_violation", "instability")


def _comparison_fields(candidate: Mapping[str, Any], *, expected_prediction_size: int | None) -> tuple[bool, str | None, int | None]:
    """Validate one candidate without allowing one bad row to abort a set."""
    if "predictions" not in candidate or not isinstance(candidate.get("metrics"), Mapping):
        return False, "missing_predictions_or_metrics", expected_prediction_size
    try:
        predictions = np.asarray(candidate["predictions"], dtype=float)
        if predictions.ndim != 1 or predictions.size == 0 or not np.isfinite(predictions).all():
            return False, "predictions_not_finite_1d", expected_prediction_size
    except (TypeError, ValueError, OverflowError):
        return False, "predictions_not_numeric", expected_prediction_size
    metrics = candidate["metrics"]
    for name in _REQUIRED_METRICS:
        try:
            value = float(metrics[name])
        except (KeyError, TypeError, ValueError, OverflowError):
            return False, f"missing_or_invalid_metric:{name}", expected_prediction_size
        if not math.isfinite(value) or value < 0:
            return False, f"invalid_metric:{name}", expected_prediction_size
    size = int(predictions.size)
    if expected_prediction_size is not None and size != expected_prediction_size:
        return False, "prediction_lengths_must_match", expected_prediction_size
    return True, None, size


def compete_models(
    candidates: Sequence[Mapping[str, Any]], *,
    counterexamples: Sequence[Mapping[str, Any]] = (),
    uncertainty: Mapping[str, Any] | None = None,
    assumptions: Sequence[str] = (),
    next_questions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Compare candidates while preserving failures and recommendation gates."""
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ModelCompetitionError("candidates_must_be_a_sequence")
    if not 1 <= len(candidates) <= 32:
        raise ModelCompetitionError("candidate_count_out_of_bounds")
    normalized: list[dict[str, Any]] = []
    comparable: list[dict[str, Any]] = []
    expected_prediction_size: int | None = None
    ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ModelCompetitionError("candidate_must_be_a_mapping")
        identifier = str(candidate.get("id", "")).strip()
        if not identifier or identifier in ids:
            raise ModelCompetitionError("candidate_id_required_or_duplicate")
        ids.add(identifier)
        status = str(candidate.get("status", "not_assessed"))
        item = dict(candidate)
        item["id"] = identifier
        item["status"] = status
        normalized.append(item)
        if status in _UNRESOLVED or status in {"counterexample", "constraint_violation", "invalid"}:
            continue
        valid, reason, size = _comparison_fields(candidate, expected_prediction_size=expected_prediction_size)
        if not valid:
            item["status"] = "not_assessed"
            item.setdefault("failure_reasons", []).append(reason)
            continue
        expected_prediction_size = size
        comparable.append({
            "id": identifier,
            "predictions": candidate["predictions"],
            "metrics": candidate["metrics"],
            "decision": candidate.get("decision"),
        })

    if comparable:
        try:
            comparison = assess_model_set(comparable)
        except ModelSetAssessmentError as exc:
            raise ModelCompetitionError(f"comparison_invalid:{exc}") from exc
    else:
        comparison = {
            "schema_version": "mathmodel.model-set-assessment/v1",
            "status": "candidate_set_inadequate", "candidate_count": 0,
            "pareto_candidate_ids": [], "pareto_count": 0,
            "prediction_points": 0, "decision_consensus": False,
            "policy": "no_comparable_candidate_is_not_a_winner",
        }

    pareto_ids = set(comparison.get("pareto_candidate_ids", []))
    verdict_inputs = []
    for item in normalized:
        verdict = dict(item)
        verdict["evidence_refs"] = list(item.get("evidence_refs", [])) if isinstance(item.get("evidence_refs", []), Sequence) and not isinstance(item.get("evidence_refs", []), (str, bytes)) else []
        # Pareto membership is evidence of non-dominance only; it never grants
        # approval. Explicit ``verdict_state`` and evidence remain required.
        verdict.setdefault("label", item["id"])
        if item["id"] in pareto_ids:
            verdict.setdefault("comparison_role", "pareto_candidate")
        verdict_inputs.append(verdict)
    verdict = build_model_verdict(
        verdict_inputs, uncertainty=uncertainty, assumptions=assumptions,
        counterexamples=counterexamples, next_questions=next_questions,
    )
    return {
        "schema_version": "mathmodel.model-competition/v1",
        "status": comparison.get("status", "candidate_set_inadequate"),
        "comparison": comparison,
        "verdict": verdict,
        "pareto_candidate_ids": sorted(pareto_ids),
        "policy": {
            "pareto_is_not_approval": True,
            "missing_metrics_remain_unresolved": True,
            "counterexamples_block_recommendation": True,
            "scores_are_not_probabilities": True,
        },
    }


def compete_models_staged(
    candidates: Sequence[Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any], StageSpec], Mapping[str, Any]],
    stages: Sequence[StageSpec], *,
    max_total_budget: int | None = None,
    max_wall_seconds: float | None = None,
    cancel: threading.Event | None = None,
    counterexamples: Sequence[Mapping[str, Any]] = (),
    uncertainty: Mapping[str, Any] | None = None,
    assumptions: Sequence[str] = (),
    next_questions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Run staged evaluation before Pareto comparison.

    The final stage is the only stage allowed to make a candidate comparable.
    Budget exhaustion, evaluator errors and candidates pruned before
    confirmation remain unresolved; they are never treated as poor scores.
    This is an orchestration adapter, not a replacement for an independent
    holdout or evidence gate.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ModelCompetitionError("candidates_must_be_a_sequence")
    staged = run_staged_evaluation(
        candidates, evaluator, stages, max_total_budget=max_total_budget,
        max_wall_seconds=max_wall_seconds, cancel=cancel,
    )
    final_stage = stages[-1].name if stages else ""
    final_reports = {
        str(item["candidate_id"]): item
        for item in staged.get("reports", [])
        if item.get("stage") == final_stage
    }
    enriched: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        identifier = str(item.get("id", ""))
        report = final_reports.get(identifier)
        item["staged_evidence"] = report
        if report is None or report.get("selection_status") == "pruned_by_stage":
            item["status"] = "not_assessed"
        else:
            status = str(report.get("status", "not_assessed"))
            if status in {"pass", "eligible", "eligible_for_confirmation"}:
                # Eligible is intentionally weaker than approval; the regular
                # competition/verdict gate still requires metrics and evidence.
                item["status"] = "eligible"
            elif status in {"hard_failure", "invalid", "counterexample", "constraint_violation", "type_error"}:
                item["status"] = "counterexample" if status == "counterexample" else "invalid"
            else:
                item["status"] = "not_assessed"
        enriched.append(item)
    comparison = compete_models(
        enriched, counterexamples=counterexamples, uncertainty=uncertainty,
        assumptions=assumptions, next_questions=next_questions,
    )
    return {
        "schema_version": "mathmodel.staged-model-competition/v1",
        "status": staged.get("status", "not_assessed"),
        "staged_evaluation": staged,
        "competition": comparison,
        "policy": {
            "final_stage_only_for_comparison": True,
            "budget_or_execution_failure_is_unresolved": True,
            "staged_pass_is_not_approval": True,
        },
    }


__all__ = ["ModelCompetitionError", "compete_models", "compete_models_staged"]
