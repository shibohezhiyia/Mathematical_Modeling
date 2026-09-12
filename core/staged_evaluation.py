"""Budget-aware staged candidate evaluation.

The scheduler is intentionally backend-agnostic: callers provide a trusted
evaluator (usually a supervised worker).  It implements the policy around the
evaluator, not a mathematical verdict.  Only explicitly declared hard-failure
statuses can remove a candidate.  Timeouts, resource errors and budget
exhaustion remain unresolved and are reported as such.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "mathmodel.staged-evaluation/v1"


class StagedEvaluationError(ValueError):
    """Raised when a staged evaluation plan violates its resource contract."""


@dataclass(frozen=True)
class StageSpec:
    name: str
    per_candidate_budget: int
    keep_fraction: float = 1.0
    min_survivors: int = 1
    confirmation: bool = False

    def validate(self) -> None:
        if not isinstance(self.name, str) or not 1 <= len(self.name.strip()) <= 80:
            raise StagedEvaluationError("stage_name_must_be_nonempty")
        if type(self.per_candidate_budget) is not int or not 1 <= self.per_candidate_budget <= 10_000_000:
            raise StagedEvaluationError("invalid_stage_budget")
        if type(self.keep_fraction) not in (int, float) or not math.isfinite(float(self.keep_fraction)):
            raise StagedEvaluationError("invalid_stage_keep_fraction")
        if not 0 < float(self.keep_fraction) <= 1:
            raise StagedEvaluationError("stage_keep_fraction_must_be_between_zero_and_one")
        if type(self.min_survivors) is not int or not 1 <= self.min_survivors <= 10_000:
            raise StagedEvaluationError("invalid_stage_min_survivors")
        if type(self.confirmation) is not bool:
            raise StagedEvaluationError("invalid_stage_confirmation_flag")


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    if not isinstance(candidate, Mapping):
        raise StagedEvaluationError("candidate_must_be_an_object")
    identifier = candidate.get("id")
    if not isinstance(identifier, (str, int)) or isinstance(identifier, bool) or not str(identifier).strip():
        raise StagedEvaluationError("candidate_id_required")
    return str(identifier).strip()


def _normalise_stages(stages: Sequence[StageSpec]) -> tuple[StageSpec, ...]:
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)) or not 1 <= len(stages) <= 16:
        raise StagedEvaluationError("stage_count_out_of_bounds")
    normalized = []
    seen = set()
    for stage in stages:
        if not isinstance(stage, StageSpec):
            raise StagedEvaluationError("stages_must_contain_stage_specs")
        stage.validate()
        if stage.name in seen:
            raise StagedEvaluationError("duplicate_stage_name")
        seen.add(stage.name)
        normalized.append(stage)
    confirmation_indices = [index for index, stage in enumerate(normalized) if stage.confirmation]
    if len(confirmation_indices) > 1 or (confirmation_indices and confirmation_indices[0] != len(normalized) - 1):
        raise StagedEvaluationError("confirmation_stage_must_be_last_and_unique")
    return tuple(normalized)


def _result_status(raw: Any) -> tuple[str, float | None, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        return "evaluation_error", None, {"reason": "evaluator_must_return_an_object"}
    status = raw.get("status")
    if not isinstance(status, str) or not 1 <= len(status) <= 80:
        return "evaluation_error", None, {"reason": "evaluator_status_required"}
    score = raw.get("score")
    normalized_score = None
    if score is not None:
        try:
            normalized_score = float(score)
        except (TypeError, ValueError, OverflowError):
            return "evaluation_error", None, {"reason": "evaluator_score_must_be_finite"}
        if not math.isfinite(normalized_score):
            return "evaluation_error", None, {"reason": "evaluator_score_must_be_finite"}
    public = {str(key): value for key, value in raw.items()
              if str(key) not in {"raw_output", "traceback", "observations", "code"}}
    return status, normalized_score, public


def run_staged_evaluation(
    candidates: Sequence[Mapping[str, Any]],
    evaluator: Callable[[Mapping[str, Any], StageSpec], Mapping[str, Any]],
    stages: Sequence[StageSpec], *,
    max_total_budget: int | None = None,
    max_wall_seconds: float | None = None,
    cancel: threading.Event | None = None,
    hard_failure_statuses: Sequence[str] = (
        "hard_failure", "invalid", "counterexample", "constraint_violation", "type_error",
    ),
    lower_is_better: bool = True,
) -> dict[str, Any]:
    """Run candidates through progressively more expensive stages.

    The scheduler reserves enough budget for the declared confirmation stage
    and never treats an uncalled candidate as rejected.  A stage may prune only
    explicit hard failures and score-ranked candidates after retaining its
    ``min_survivors`` and ``keep_fraction``.  This is a search policy; the
    caller still needs an independent confirmation/evidence gate.
    """
    if not callable(evaluator):
        raise StagedEvaluationError("evaluator_must_be_callable")
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise StagedEvaluationError("candidates_must_be_a_sequence")
    if not 1 <= len(candidates) <= 10_000:
        raise StagedEvaluationError("candidate_count_out_of_bounds")
    if type(lower_is_better) is not bool:
        raise StagedEvaluationError("lower_is_better_must_be_boolean")
    if max_wall_seconds is not None:
        if (type(max_wall_seconds) not in (int, float)
                or not math.isfinite(float(max_wall_seconds))
                or not 0.05 <= float(max_wall_seconds) <= 3600):
            raise StagedEvaluationError("invalid_total_wall_seconds")
    if cancel is not None and not isinstance(cancel, threading.Event):
        raise StagedEvaluationError("cancel_must_be_threading_event")
    normalized_stages = _normalise_stages(stages)
    identifiers = [_candidate_id(candidate) for candidate in candidates]
    if len(set(identifiers)) != len(identifiers):
        raise StagedEvaluationError("duplicate_candidate_id")
    hard_failures = {str(item) for item in hard_failure_statuses}
    if not hard_failures or any(not item or len(item) > 80 for item in hard_failures):
        raise StagedEvaluationError("invalid_hard_failure_statuses")
    if max_total_budget is not None:
        if type(max_total_budget) is not int or not 1 <= max_total_budget <= 100_000_000:
            raise StagedEvaluationError("invalid_total_budget")
    confirmation = next((stage for stage in normalized_stages if stage.confirmation), None)
    reserve = (confirmation.per_candidate_budget * min(confirmation.min_survivors, len(candidates))
               if confirmation else 0)
    if max_total_budget is not None and max_total_budget < reserve:
        raise StagedEvaluationError("total_budget_below_confirmation_reserve")
    active = list(identifiers)
    by_id = dict(zip(identifiers, candidates))
    reports: list[dict[str, Any]] = []
    stage_audits: list[dict[str, Any]] = []
    charged = 0
    unresolved_budget = False
    started = time.monotonic()
    deadline = started + float(max_wall_seconds) if max_wall_seconds is not None else None
    interrupted = None
    for stage in normalized_stages:
        before = charged
        stage_reports: list[dict[str, Any]] = []
        for identifier in active:
            if cancel is not None and cancel.is_set():
                interrupted = "cancelled"
            elif deadline is not None and time.monotonic() >= deadline:
                interrupted = "deadline_exhausted"
            if interrupted is not None:
                stage_reports.append({"candidate_id": identifier, "stage": stage.name,
                                      "status": interrupted, "score": None,
                                      "selection_status": "unresolved_execution",
                                      "evaluations_charged": 0,
                                      "evidence": {"reason": interrupted}})
                continue
            if max_total_budget is not None:
                available = max_total_budget - charged
                reserve_needed = 0 if stage.confirmation else reserve
                if available - reserve_needed < stage.per_candidate_budget:
                    unresolved_budget = True
                    stage_reports.append({"candidate_id": identifier, "stage": stage.name,
                                          "status": "budget_exhausted", "score": None,
                                          "selection_status": "unresolved_budget",
                                          "evaluations_charged": 0})
                    continue
            try:
                raw = evaluator(by_id[identifier], stage)
            except Exception as exc:  # evaluator details are not evidence of a math failure
                raw = {"status": "execution_error", "reason": f"evaluator_error:{type(exc).__name__}"}
            status, score, public = _result_status(raw)
            used = public.get("evaluations_used", stage.per_candidate_budget)
            if (type(used) is not int or used < 1 or used > stage.per_candidate_budget):
                status, score = "evaluation_error", None
                public = {**public, "reason": "evaluations_used_must_be_bounded_integer"}
                used = stage.per_candidate_budget
            if max_total_budget is not None and charged + used > max_total_budget:
                used = stage.per_candidate_budget
                status, score, public = "budget_exhausted", None, {"reason": "stage_result_exceeded_total_budget"}
            charged += used
            if cancel is not None and cancel.is_set():
                interrupted = "cancelled"
            elif deadline is not None and time.monotonic() >= deadline:
                interrupted = "deadline_exhausted"
            if interrupted is not None and status not in hard_failures:
                status, score = interrupted, None
                public = {"reason": interrupted}
            stage_reports.append({"candidate_id": identifier, "stage": stage.name,
                                  "status": status, "score": score,
                                  "selection_status": "hard_failed" if status in hard_failures else "assessed",
                                  "evaluations_charged": used, "evidence": public})
        reports.extend(stage_reports)
        hard_failed = {item["candidate_id"] for item in stage_reports if item["status"] in hard_failures}
        eligible = [item for item in stage_reports if item["candidate_id"] not in hard_failed]
        if stage.confirmation:
            active = [item["candidate_id"] for item in eligible]
        else:
            ranked = [item for item in eligible if item["score"] is not None]
            ranked.sort(key=lambda item: item["score"], reverse=not lower_is_better)
            target = min(len(eligible), max(stage.min_survivors, math.ceil(len(eligible) * stage.keep_fraction)))
            # Errors and budget exhaustion are unresolved, not poor scores.
            # Keep them alive for a later retry/confirmation instead of
            # silently turning a resource failure into a model rejection.
            unresolved_ids = {
                item["candidate_id"] for item in eligible
                if item["status"] in {"execution_error", "evaluation_error", "budget_exhausted",
                                       "cancelled", "deadline_exhausted"}
            }
            selected = set(unresolved_ids)
            selection_target = max(target, len(unresolved_ids))
            for item in ranked:
                if len(selected) >= selection_target:
                    break
                selected.add(item["candidate_id"])
            if len(selected) < min(stage.min_survivors, len(eligible)):
                for item in eligible:
                    if item["candidate_id"] not in selected:
                        selected.add(item["candidate_id"])
                    if len(selected) >= min(stage.min_survivors, len(eligible)):
                        break
            active = [item["candidate_id"] for item in eligible if item["candidate_id"] in selected]
        for item in stage_reports:
            if item["candidate_id"] in active and item["selection_status"] == "assessed":
                item["selection_status"] = "survives_stage"
            elif item["selection_status"] == "assessed":
                item["selection_status"] = "pruned_by_stage"
        stage_audits.append({"stage": stage.name, "candidate_count": len(stage_reports),
                             "survivor_ids": list(active), "charged_evaluations": charged - before,
                             "confirmation": stage.confirmation,
                             "budget_exhausted": any(item["status"] == "budget_exhausted" for item in stage_reports)})
        if not active:
            break
    final_stage = normalized_stages[-1].name
    final_reports = {item["candidate_id"]: item for item in reports if item["stage"] == final_stage}
    accepted = [identifier for identifier in active
                if identifier in final_reports and final_reports[identifier]["status"] in {"pass", "eligible", "eligible_for_confirmation"}]
    if interrupted is not None:
        overall_status = interrupted
    elif unresolved_budget:
        overall_status = "budget_exhausted"
    else:
        overall_status = "no_survivors" if not active else "completed"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": overall_status,
        "candidate_count": len(candidates), "survivor_ids": list(active),
        "accepted_ids": accepted, "reports": reports, "stages": stage_audits,
        "budget": {"charged": charged, "max_total": max_total_budget,
                   "confirmation_reserve": reserve, "confirmation_reserved": reserve > 0},
        "policy": "budget_exhaustion_execution_failure_timeout_and_cancellation_are_unresolved_not_counterexamples",
    }


__all__ = ["SCHEMA_VERSION", "StagedEvaluationError", "StageSpec", "run_staged_evaluation"]
