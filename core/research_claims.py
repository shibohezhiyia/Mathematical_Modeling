"""Operational definitions for research and release claims."""

from __future__ import annotations

from typing import Any, Mapping


class ResearchClaimError(ValueError):
    pass


_CLAIMS = {
    "open_world": "tested on declared unseen/structure-transformed cases without task-specific solver code",
    "automatic_repair": "a bounded candidate mutation was executed and passed declared checks after repair",
    "trustworthy": "the output has explicit evidence, scope, counterexample status and independent confirmation",
}


def operationalize_claim(claim: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    claim = str(claim)
    if claim not in _CLAIMS:
        raise ResearchClaimError("unknown_claim")
    if not isinstance(evidence, Mapping):
        raise ResearchClaimError("evidence_required")
    required = {"cases_tested", "budget", "failure_policy", "independent_confirmation"}
    missing = sorted(required - set(evidence))
    invalid = []
    cases = evidence.get("cases_tested")
    if not (type(cases) is int and 1 <= cases <= 1_000_000):
        invalid.append("cases_tested")
    budget = evidence.get("budget")
    if not isinstance(budget, Mapping) or not budget:
        invalid.append("budget")
    policy = evidence.get("failure_policy")
    if not isinstance(policy, str) or not policy.strip() or len(policy) > 2_000:
        invalid.append("failure_policy")
    if type(evidence.get("independent_confirmation")) is not bool:
        invalid.append("independent_confirmation")
    return {"schema_version": "mathmodel.research-claim/v1", "claim": claim,
            "definition": _CLAIMS[claim], "status": "operationalized" if not missing and not invalid else "not_assessed",
            "missing": missing, "invalid": sorted(set(invalid)),
            "evidence": {key: evidence.get(key) for key in sorted(set(evidence) & required)},
            "policy": "operational_definition_does_not_guarantee_external_validity"}


__all__ = ["ResearchClaimError", "operationalize_claim"]
