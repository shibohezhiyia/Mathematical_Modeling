"""Evidence-weighted semantic hypothesis uncertainty (not Bayesian posterior)."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


class SemanticUncertaintyError(ValueError):
    pass


def assess_semantic_hypotheses(hypotheses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(hypotheses, Sequence) or isinstance(hypotheses, (str, bytes)) or not hypotheses or len(hypotheses) > 64:
        raise SemanticUncertaintyError("hypotheses_required")
    rows = []
    total = 0.0
    seen_ids: set[str] = set()
    for item in hypotheses:
        if not isinstance(item, Mapping) or not str(item.get("id", "")).strip():
            raise SemanticUncertaintyError("hypothesis_id_required")
        identifier = str(item["id"]).strip()
        if identifier in seen_ids:
            raise SemanticUncertaintyError("hypothesis_id_must_be_unique")
        seen_ids.add(identifier)
        weight = item.get("evidence_weight", item.get("weight"))
        if not isinstance(weight, (int, float)) or isinstance(weight, bool) or not math.isfinite(float(weight)) or weight < 0:
            raise SemanticUncertaintyError("evidence_weight_invalid")
        refs = item.get("evidence_refs", [])
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            raise SemanticUncertaintyError("evidence_refs_must_be_a_sequence")
        normalized_refs = [str(ref).strip() for ref in refs[:32] if str(ref).strip()]
        rows.append({"id": identifier, "weight": float(weight),
                     "evidence_refs": normalized_refs})
        total += float(weight)
    if total <= 0:
        return {"schema_version": "mathmodel.semantic-uncertainty/v1", "status": "not_assessed",
                "hypotheses": rows, "entropy": None,
                "policy": "weights_are_declared_evidence_scores_not_posterior_probabilities"}
    for row in rows:
        row["normalized_weight"] = row["weight"] / total
    entropy = -sum(row["normalized_weight"] * math.log(row["normalized_weight"])
                   for row in rows if row["normalized_weight"] > 0)
    return {"schema_version": "mathmodel.semantic-uncertainty/v1", "status": "assessed",
            "hypotheses": rows, "entropy": entropy, "effective_hypothesis_count": math.exp(entropy),
            "policy": "declared_evidence_weights_are_not_posterior_probabilities"}


__all__ = ["SemanticUncertaintyError", "assess_semantic_hypotheses"]
