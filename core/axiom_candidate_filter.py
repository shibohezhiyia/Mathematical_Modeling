"""Filter model candidates using explicit data evidence and background axioms."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


class AxiomFilterError(ValueError):
    pass


def filter_candidates_by_axioms(
    candidates: Sequence[Mapping[str, Any]], *,
    data_evidence: Mapping[str, Mapping[str, Any]],
    background_axioms: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply declared evidence gates before numerical candidate ranking.

    A candidate must explicitly name data and axiom IDs. ``supports`` is not
    inferred from a score; missing/contradictory evidence remains unresolved.
    Domain extrapolation records with ``counterexample_found`` reject a
    candidate for that tested extension only.
    """
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)) or not candidates:
        raise AxiomFilterError("candidates_required")
    if not isinstance(data_evidence, Mapping) or not isinstance(background_axioms, Mapping):
        raise AxiomFilterError("evidence_mappings_required")
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or not isinstance(candidate.get("id"), str) or not candidate["id"].strip():
            raise AxiomFilterError("candidate_id_required")
        identifier = candidate["id"].strip()
        data_ids, axiom_ids = candidate.get("data_evidence_ids", ()), candidate.get("axiom_ids", ())
        if not isinstance(data_ids, (list, tuple)) or not isinstance(axiom_ids, (list, tuple)):
            raise AxiomFilterError("evidence_ids_must_be_sequences")
        missing = [item for item in (*data_ids, *axiom_ids)
                   if not isinstance(item, str) or not item.strip()]
        if missing:
            raise AxiomFilterError("evidence_id_must_be_nonempty")
        unknown_data = sorted(set(data_ids) - set(data_evidence))
        unknown_axioms = sorted(set(axiom_ids) - set(background_axioms))
        unsupported = []
        for item in data_ids:
            if data_evidence[item].get("status") != "supports":
                unsupported.append(f"data:{item}")
        for item in axiom_ids:
            if background_axioms[item].get("status") != "supports":
                unsupported.append(f"axiom:{item}")
        extrapolation = candidate.get("domain_extrapolation", {})
        if isinstance(extrapolation, Mapping) and extrapolation.get("status") == "counterexample_found":
            unsupported.append("domain_extrapolation")
        if unknown_data or unknown_axioms:
            status = "not_assessed"
            reason = "unknown_evidence_reference"
        elif unsupported:
            status = "rejected" if "domain_extrapolation" in unsupported else "not_assessed"
            reason = "axiom_or_data_not_supported"
        else:
            status, reason = "eligible", "declared_data_and_axioms_support_candidate"
        rows.append({"id": identifier, "status": status, "reason": reason,
                     "unknown_data_ids": unknown_data, "unknown_axiom_ids": unknown_axioms,
                     "unsupported": sorted(set(unsupported))})
    return {"schema_version": "mathmodel.axiom-candidate-filter/v1",
            "status": "ready" if any(row["status"] == "eligible" for row in rows) else "candidate_set_inadequate",
            "candidates": rows,
            "eligible_ids": [row["id"] for row in rows if row["status"] == "eligible"],
            "policy": "explicit_data_and_axiom_evidence_before_fit;finite_extrapolation_is_not_global_proof"}


__all__ = ["AxiomFilterError", "filter_candidates_by_axioms"]
