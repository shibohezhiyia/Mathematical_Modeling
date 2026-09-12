"""Counterexample-guided, typed repair directives for primitive-graph search.

The directive builder only translates known failure codes into a closed set of
primitive preferences.  It never edits an IR, relaxes a constraint, or marks a
candidate as repaired; the normal graph validator and independent checks still
decide whether a generated patch is admissible.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping


SCHEMA_VERSION = "mathmodel.cegis-repair/v1"
_BINARY = {
    "numeric_domain_at_bound_input": ("minimum", "maximum", "multiply"),
    "output_out_of_bounds": ("minimum", "maximum"),
    "bound_violation": ("minimum", "maximum"),
    "property_bound_violation": ("minimum", "maximum"),
    "residual_bias": ("add", "subtract"),
    "missing_offset": ("add", "subtract"),
    "interaction_missing": ("multiply",),
}
_UNARY = {
    "numeric_domain_at_bound_input": ("abs", "sqrt"),
    "nonfinite_prediction": ("abs", "sqrt"),
    "log_domain_error": ("abs", "sqrt"),
}


class CEGISRepairError(ValueError):
    """Raised when counterexample feedback is not an auditable mapping."""


def build_repair_directives(feedback: Mapping[str, Any], *, max_directives: int = 16) -> dict[str, Any]:
    """Convert bounded violation records into non-executable repair directives."""
    if not isinstance(feedback, Mapping):
        raise CEGISRepairError("feedback_must_be_mapping")
    if type(max_directives) is not int or not 1 <= max_directives <= 32:
        raise CEGISRepairError("invalid_repair_directive_budget")
    violations = feedback.get("violations", [])
    if not isinstance(violations, list) or len(violations) > 128:
        raise CEGISRepairError("invalid_counterexample_records")
    directives: list[dict[str, Any]] = []
    for index, violation in enumerate(violations[:max_directives]):
        if not isinstance(violation, Mapping):
            raise CEGISRepairError("invalid_counterexample_record")
        reason = str(violation.get("reason", "unknown"))
        binary = list(_BINARY.get(reason, ()))
        unary = list(_UNARY.get(reason, ()))
        directives.append({
            "id": "repair_" + sha256(json.dumps({"index": index, "record": dict(violation)},
                                                   sort_keys=True, default=str).encode()).hexdigest()[:20],
            "witness_id": violation.get("witness_id"),
            "reason": reason,
            "candidate_binary_primitives": binary,
            "candidate_unary_primitives": unary,
            "status": "proposal_not_executed",
            "known_reason": bool(binary or unary),
            "may_modify_ir": True,
            "may_change_hard_constraints": False,
            "requires_revalidation": True,
            "evidence": {key: violation[key] for key in ("witness_id", "reason") if key in violation},
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "directives": directives,
        "directive_count": len(directives),
        "counterexample_count": len(violations),
        "unknown_reason_count": sum(not item["known_reason"] for item in directives),
        "policy": "counterexample_guided_preference_only; validator_remains_authoritative",
    }


def prioritized_operators(feedback: Mapping[str, Any], *, unary: bool = False) -> tuple[str, ...]:
    """Return a deterministic closed operator order derived from known reasons."""
    directives = build_repair_directives(feedback)["directives"]
    key = "candidate_unary_primitives" if unary else "candidate_binary_primitives"
    result: list[str] = []
    for directive in directives:
        result.extend(directive[key])
    return tuple(dict.fromkeys(result))


__all__ = ["SCHEMA_VERSION", "CEGISRepairError", "build_repair_directives", "prioritized_operators"]
