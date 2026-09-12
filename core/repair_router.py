"""Choose deterministic repairs before asking a semantic model for help."""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping


SCHEMA_VERSION = "mathmodel.repair-router/v1"
_RULE_CODES = frozenset({
    "type_error", "unit_mismatch", "shape_mismatch", "nonfinite_input",
    "numeric_tolerance", "bound_violation", "solver_iteration_limit",
})
_SEMANTIC_CODES = frozenset({
    "unknown_mechanism", "missing_variable_role", "causal_ambiguity",
    "missing_observation_process", "candidate_set_inadequate",
})


class RepairRouterError(ValueError):
    pass


def route_repair(issue_code: Any, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a non-authorizing repair route with a stable bounded context hash."""
    if not isinstance(issue_code, str) or not issue_code.strip() or len(issue_code) > 120:
        raise RepairRouterError("issue_code_invalid")
    context = {} if context is None else context
    if not isinstance(context, Mapping):
        raise RepairRouterError("repair_context_must_be_an_object")
    try:
        encoded = json.dumps(dict(context), ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RepairRouterError("repair_context_must_be_finite_json") from exc
    if len(encoded) > 16_384:
        raise RepairRouterError("repair_context_too_large")
    code = issue_code.strip()
    if code in _RULE_CODES:
        route, model_allowed, reason = "rule_solver", False, "deterministic_type_unit_numeric_or_resource_repair"
    elif code in _SEMANTIC_CODES:
        route, model_allowed, reason = "semantic_model", True, "new_mechanism_or_semantic_branch_requires_grounded_proposal"
    else:
        route, model_allowed, reason = "needs_input", False, "unclassified_issue_requires_evidence_or_user_clarification"
    return {
        "schema_version": SCHEMA_VERSION,
        "issue_code": code,
        "route": route,
        "model_calls_allowed": model_allowed,
        "reason": reason,
        "context_sha256": sha256(encoded).hexdigest(),
        "execution_authorized": False,
        "policy": "routing_only; every repair must re-enter contract_validation_and_evidence_gates",
    }


__all__ = ["SCHEMA_VERSION", "RepairRouterError", "route_repair"]
