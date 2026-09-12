"""Hard-evidence gate and failure taxonomy for candidate competition."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .execution_readiness import ExecutionReadinessError, assess_execution_readiness


class CandidateGateError(ValueError):
    pass


_FAILURE_CLASSES = {"semantic", "structure", "parameter", "data", "numerical", "resource", "counterexample"}
_EXECUTABLE_MARKERS = frozenset({"executable", "executed", "runnable", "implemented", "solver_ready"})


def gate_candidates(candidates: Sequence[Mapping[str, Any]], *, metric_directions: Mapping[str, str] | None = None) -> dict[str, Any]:
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise CandidateGateError("candidates_required")
    directions = dict(metric_directions or {})
    for name, direction in directions.items():
        if direction not in {"min", "max"}:
            raise CandidateGateError("metric_direction_invalid")
    normalized = []
    seen_ids: set[str] = set()
    comparable = []
    # Keep the metric schema explicit.  Comparing a loss-only candidate with a
    # candidate that also reports coverage/complexity is not a valid Pareto
    # comparison: the former is silently missing an objective.  We therefore
    # collect schemas first and only rank a homogeneous set below.
    metric_rows = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not str(candidate.get("id", "")).strip():
            raise CandidateGateError("candidate_id_required")
        candidate_id = str(candidate["id"]).strip()
        if candidate_id in seen_ids:
            raise CandidateGateError("candidate_id_must_be_unique")
        seen_ids.add(candidate_id)
        item = dict(candidate)
        checks = item.get("hard_checks", {})
        if not isinstance(checks, Mapping):
            raise CandidateGateError("hard_checks_must_be_mapping")
        failed = [str(key) for key, value in checks.items() if value == "fail" or value is False]
        pending = [str(key) for key, value in checks.items()
                   if value is None or (isinstance(value, str) and value in {"not_assessed", "pending"})]
        readiness = item.get("execution_readiness")
        readiness_report = None
        execution_marker = item.get("execution_status")
        if execution_marker in _EXECUTABLE_MARKERS and readiness is None:
            # A result claiming it ran cannot be ranked without the same
            # typed/unit/source/resource/security evidence required of every
            # other executable candidate.  Proposal-only candidates retain the
            # historical pending checks and are not affected by this rule.
            pending.append("execution_readiness")
        if readiness is not None:
            try:
                readiness_report = assess_execution_readiness(readiness)
            except ExecutionReadinessError as exc:
                failed.append("execution_readiness_contract")
                readiness_report = {"status": "blocked", "error": str(exc)}
            else:
                if readiness_report["status"] == "blocked":
                    failed.append("execution_readiness")
                elif readiness_report["status"] != "ready":
                    pending.append("execution_readiness")
        if failed:
            status = "hard_failure"
        elif pending:
            status = "not_assessed"
        else:
            status = "eligible"
        metrics = item.get("metrics", {})
        normalized_metrics = dict(metrics) if isinstance(metrics, Mapping) else {}
        if status == "eligible" and isinstance(metrics, Mapping) and metrics:
            clean_metrics = {}
            invalid_metric = False
            for key, value in metrics.items():
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError):
                    invalid_metric = True
                    break
                if not math.isfinite(number):
                    invalid_metric = True
                    break
                clean_metrics[str(key)] = number
            if invalid_metric:
                status = "not_assessed"
                pending.append("metric_invalid")
            else:
                metric_rows.append({"id": candidate_id, "metrics": clean_metrics})
        normalized.append({"id": candidate_id, "status": status,
                           "failed_hard_checks": failed, "pending_hard_checks": pending,
                           "failure_class": str(item.get("failure_class", "")) if item.get("failure_class") else None,
                           "metrics": normalized_metrics,
                           "metric_schema": sorted(str(key) for key in normalized_metrics),
                           "execution_readiness": readiness_report})

    schema_groups: dict[tuple[str, ...], list[str]] = {}
    for row in metric_rows:
        schema_groups.setdefault(tuple(sorted(row["metrics"])), []).append(row["id"])
    eligible_rows = [row for row in normalized if row["status"] == "eligible"]
    eligible_schemas = {tuple(row["metric_schema"]) for row in eligible_rows}
    # An empty metric mapping is also a schema.  Only trigger this guard when
    # there is something to compare; all-empty candidates remain eligible but
    # correctly produce candidate_set_inadequate below.
    schema_mismatch = len(eligible_schemas) > 1 and bool(metric_rows)
    incomparable_ids: list[str] = []
    if schema_mismatch:
        for row in normalized:
            if row["status"] == "eligible":
                row["status"] = "not_assessed"
                row["pending_hard_checks"].append("metric_schema_incomplete")
                incomparable_ids.append(row["id"])
    else:
        comparable = metric_rows
    pareto = []
    for row in comparable:
        dominated = False
        for other in comparable:
            if other is row:
                continue
            no_worse = True; strictly = False
            for name, value in row["metrics"].items():
                if name not in other["metrics"]:
                    no_worse = False; break
                direction = directions.get(name, "min")
                if direction == "min":
                    if other["metrics"][name] > value: no_worse = False
                    if other["metrics"][name] < value: strictly = True
                else:
                    if other["metrics"][name] < value: no_worse = False
                    if other["metrics"][name] > value: strictly = True
            if no_worse and strictly:
                dominated = True; break
        if not dominated:
            pareto.append(row["id"])
    return {"schema_version": "mathmodel.candidate-gate/v1", "candidates": normalized,
            "eligible_ids": [row["id"] for row in normalized if row["status"] == "eligible"],
            "pareto_ids": sorted(pareto), "status": "ready_for_comparison" if pareto else "candidate_set_inadequate",
            "metric_schema_groups": {"|".join(schema): ids for schema, ids in sorted(schema_groups.items())},
            "incomparable_metric_schema_ids": sorted(incomparable_ids),
            "policy": "hard_failures_or_unassessed_checks_cannot_be_offset_by_fit_metrics;_metric_schema_must_be_homogeneous"}


def classify_failure(category: str, *, evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    category = str(category)
    if category not in _FAILURE_CLASSES:
        raise CandidateGateError("unknown_failure_class")
    return {"class": category, "evidence": dict(evidence or {}),
            "mathematical_verdict": "not_assessed" if category != "counterexample" else "rejected",
            "policy": "failure_taxonomy_does_not_infer_root_cause_without_evidence"}


__all__ = ["CandidateGateError", "gate_candidates", "classify_failure"]
