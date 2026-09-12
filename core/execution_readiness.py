"""Unified execution-admission contract for heterogeneous model backends.

Backends can expose different validators, but publication must not depend on a
caller remembering one of them.  This module only aggregates explicit check
statuses; it never infers a pass from a score, a successful import, or a
missing field.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class ExecutionReadinessError(ValueError):
    pass


REQUIRED_CHECKS = ("type", "unit", "source", "resource", "security")
_PASS = frozenset({"pass", "passed", "verified", "assessed", "ready", "isolated", "ok"})
_PENDING = frozenset({"not_assessed", "pending", "unknown", "unresolved"})
_FAIL = frozenset({"fail", "failed", "blocked", "denied", "invalid", "unsafe"})


def assess_execution_readiness(checks: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate five mandatory checks without filling missing evidence."""
    if not isinstance(checks, Mapping):
        raise ExecutionReadinessError("checks_must_be_mapping")
    missing = [name for name in REQUIRED_CHECKS if name not in checks]
    if missing:
        raise ExecutionReadinessError("required_check_missing:" + ",".join(missing))
    rows: dict[str, dict[str, Any]] = {}
    failed, pending = [], []
    for name in REQUIRED_CHECKS:
        raw = checks[name]
        evidence: Sequence[Any] = ()
        if isinstance(raw, Mapping):
            status = raw.get("status", raw.get("state"))
            evidence = raw.get("evidence", raw.get("evidence_refs", ()))
        else:
            status = raw
        if type(status) is not str or not status.strip():
            raise ExecutionReadinessError("check_status_required:" + name)
        normalized = status.strip().lower()
        if normalized not in _PASS | _PENDING | _FAIL:
            raise ExecutionReadinessError("unknown_check_status:" + name)
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            raise ExecutionReadinessError("check_evidence_must_be_sequence:" + name)
        refs = [str(item).strip()[:240] for item in evidence[:16] if str(item).strip()]
        rows[name] = {"status": normalized, "evidence": refs}
        if normalized in _FAIL:
            failed.append(name)
        elif normalized in _PENDING:
            pending.append(name)
    state = "blocked" if failed else "not_assessed" if pending else "ready"
    return {
        "schema_version": "mathmodel.execution-readiness/v1",
        "status": state,
        "checks": rows,
        "failed_checks": failed,
        "pending_checks": pending,
        "policy": "all_five_explicit_checks_required;_missing_or_unresolved_never_passes",
    }


__all__ = ["ExecutionReadinessError", "REQUIRED_CHECKS", "assess_execution_readiness"]
