"""State machine for honest progressive research results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


class ProgressiveResultError(ValueError):
    pass


_PHASES = ("baseline", "preview", "development", "confirmation", "final")


@dataclass
class ProgressiveRun:
    run_id: str
    budget: Mapping[str, Any]
    phase: str = "baseline"
    execution_state: str = "pending"
    evidence_state: str = "not_assessed"
    semantic_state: str = "not_assessed"
    candidate_ids: list[str] = field(default_factory=list)
    selected_candidate: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_stage: str | None = None
    cancelled: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ProgressiveResultError("run_id_required")
        if self.phase not in _PHASES:
            raise ProgressiveResultError("invalid_phase")
        if not isinstance(self.budget, Mapping):
            raise ProgressiveResultError("budget_required")

    def advance(self, phase: str, *, execution_state: str | None = None,
                evidence_state: str | None = None, semantic_state: str | None = None,
                metrics: Mapping[str, Any] | None = None) -> "ProgressiveRun":
        if self.cancelled:
            raise ProgressiveResultError("run_cancelled")
        if phase not in _PHASES or _PHASES.index(phase) < _PHASES.index(self.phase):
            raise ProgressiveResultError("phase_must_be_monotonic")
        effective_evidence = self.evidence_state if evidence_state is None else str(evidence_state)
        if phase == "final" and (effective_evidence != "approved" or not self.selected_candidate):
            raise ProgressiveResultError("final_requires_approved_candidate_and_selection")
        self.phase = phase
        if execution_state is not None: self.execution_state = str(execution_state)
        if evidence_state is not None: self.evidence_state = str(evidence_state)
        if semantic_state is not None: self.semantic_state = str(semantic_state)
        if metrics is not None: self.metrics.update(dict(metrics))
        return self

    def add_candidate(self, candidate_id: str) -> None:
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ProgressiveResultError("candidate_id_required")
        candidate_id = candidate_id.strip()
        if candidate_id not in self.candidate_ids:
            self.candidate_ids.append(candidate_id)

    def select(self, candidate_id: str) -> None:
        if self.cancelled:
            raise ProgressiveResultError("run_cancelled")
        if not isinstance(candidate_id, str) or not candidate_id.strip():
            raise ProgressiveResultError("candidate_id_required")
        candidate_id = candidate_id.strip()
        if candidate_id not in self.candidate_ids:
            raise ProgressiveResultError("candidate_must_be_known")
        self.selected_candidate = candidate_id

    def cancel(self, stage: str) -> None:
        if not isinstance(stage, str) or not stage.strip():
            raise ProgressiveResultError("failure_stage_required")
        self.cancelled = True
        self.execution_state = "cancelled"
        self.failure_stage = stage.strip()

    def public(self) -> dict[str, Any]:
        return {"schema_version": "mathmodel.progressive-result/v1", "run_id": self.run_id,
                "phase": self.phase, "execution_state": self.execution_state,
                "evidence_state": self.evidence_state, "semantic_state": self.semantic_state,
                "candidate_ids": list(self.candidate_ids), "selected_candidate": self.selected_candidate,
                "metrics": dict(self.metrics), "failure_stage": self.failure_stage,
                "cancelled": self.cancelled,
                "is_final_claim": self.phase == "final" and self.evidence_state == "approved",
                "policy": "preview_and_development_are_not_final_evidence"}


__all__ = ["ProgressiveResultError", "ProgressiveRun"]
