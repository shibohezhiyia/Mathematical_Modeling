"""Evidence-first routing for large residuals and solver failures."""
from __future__ import annotations

import math
from typing import Any, Mapping


class FailureDiagnosisError(ValueError):
    pass


_ORDER = (
    "semantic", "structural_misspecification", "parameter_identifiability",
    "data", "numerical_tolerance", "resource", "observation_error",
)


def diagnose_failure(signals: Mapping[str, Mapping[str, Any]], *, residual: float | None = None) -> dict[str, Any]:
    """Rank alternative explanations without automatically inventing a latent state."""
    if not isinstance(signals, Mapping) or len(signals) > 32:
        raise FailureDiagnosisError("signals_must_be_mapping")
    if residual is not None and (type(residual) not in (int, float) or not math.isfinite(float(residual)) or residual < 0):
        raise FailureDiagnosisError("invalid_residual")
    rows = []
    for rank, name in enumerate(_ORDER):
        raw = signals.get(name, {})
        if not isinstance(raw, Mapping):
            raise FailureDiagnosisError("diagnostic_signal_must_be_mapping")
        status = str(raw.get("status", "not_assessed"))
        if status not in {"pass", "fail", "not_assessed"}:
            raise FailureDiagnosisError("invalid_diagnostic_status")
        rows.append({"name": name, "category": {
                         "semantic": "semantic", "structural_misspecification": "structural",
                         "parameter_identifiability": "parameter", "data": "data",
                         "numerical_tolerance": "numerical", "resource": "resource",
                         "observation_error": "data",
                     }[name], "status": status, "priority": rank + 1,
                     "evidence": str(raw.get("evidence", ""))[:500],
                     "action": str(raw.get("action", "measure_or_test"))[:160]})
    assessed_failures = [row for row in rows if row["status"] == "fail"]
    return {"schema_version": "mathmodel.failure-diagnosis/v1", "status": "assessed" if any(row["status"] != "not_assessed" for row in rows) else "not_assessed",
            "residual": residual, "explanations": rows,
            "leading_explanations": [row["name"] for row in assessed_failures[:2]],
            "latent_state_action": "not_automatic; require competing evidence",
            "policy": "large_residual_is_not_proof_of_structural_or_latent_state_failure"}


__all__ = ["FailureDiagnosisError", "diagnose_failure"]
