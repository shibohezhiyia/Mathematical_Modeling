"""Compare competing residual explanations before proposing a new mechanism."""

from __future__ import annotations

import math
from typing import Any, Mapping


class DiagnosticComparisonError(ValueError):
    pass


_EXPLANATIONS = {
    "numerical_tolerance": ("先检查数值容差/步长是否改变残差", "tighten_tolerance"),
    "observation_error": ("观测误差或噪声尺度是否解释残差", "noise_model_check"),
    "parameter_identifiability": ("参数是否在重复划分中可辨识", "identifiability_check"),
    "structural_missing": ("剩余结构是否稳定地支持机制缺失", "structure_expansion"),
}


def compare_residual_explanations(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Return evidence-weighted, reversible explanations for a large residual.

    Each evidence field is an object with ``status`` (`supports`, `does_not_support`,
    or `not_assessed`) and optional finite ``strength`` in [0, 1]. The function
    only ranks hypotheses; it never turns a residual into a latent-state claim.
    """
    if not isinstance(evidence, Mapping):
        raise DiagnosticComparisonError("evidence_must_be_mapping")
    rows = []
    for key, (question, route) in _EXPLANATIONS.items():
        item = evidence.get(key, {})
        if not isinstance(item, Mapping):
            raise DiagnosticComparisonError(f"{key}_evidence_must_be_mapping")
        status = item.get("status", "not_assessed")
        if status not in {"supports", "does_not_support", "not_assessed"}:
            raise DiagnosticComparisonError(f"{key}_status_invalid")
        strength = item.get("strength", 0.0)
        try:
            strength = float(strength)
        except (TypeError, ValueError, OverflowError) as exc:
            raise DiagnosticComparisonError(f"{key}_strength_invalid") from exc
        if not math.isfinite(strength) or not 0 <= strength <= 1:
            raise DiagnosticComparisonError(f"{key}_strength_invalid")
        score = strength if status == "supports" else 0.0
        rows.append({"explanation": key, "status": status, "strength": strength,
                     "evidence_score": score, "question": question, "route": route,
                     "not_claimed": "causal_mechanism_or_hidden_state"})
    rows.sort(key=lambda row: (-row["evidence_score"], row["explanation"]))
    supported = [row for row in rows if row["status"] == "supports"]
    return {"schema_version": "mathmodel.diagnostic-comparison/v1",
            "status": "ranked" if supported else "not_assessed",
            "explanations": rows, "top_explanation": supported[0]["explanation"] if supported else None,
            "clarifying_questions": [row["question"] for row in rows if row["status"] == "not_assessed"][:3],
            "policy": "residual_does_not_auto_trigger_hidden_state_or_structural_expansion"}


__all__ = ["DiagnosticComparisonError", "compare_residual_explanations"]
