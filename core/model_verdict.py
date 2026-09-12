"""Conservative, machine-readable verdict assembly for model comparison.

This module is a reporting boundary. It never turns a score, ranking or finite
test into a proof. Callers must explicitly mark an ``approved`` candidate;
otherwise the strongest state emitted is ``conditional`` or ``unresolved``.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mathmodel.model-verdict/v1"
_STATES = frozenset({"approved", "conditional", "rejected", "unresolved"})
_UNRESOLVED_STATUSES = frozenset({
    "execution_error", "evaluation_error", "budget_exhausted", "needs_input",
    "unidentifiable", "candidate_set_inadequate", "not_assessed", "pending",
})
_REJECTED_STATUSES = frozenset({
    "hard_failure", "invalid", "counterexample", "constraint_violation", "type_error",
})


class ModelVerdictError(ValueError):
    """Raised for malformed verdict inputs."""


def _json_key(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _safe_count(value: Any) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        value = 0
    return max(0, min(value, 1_000_000))


def _state(candidate: Mapping[str, Any]) -> tuple[str, str]:
    explicit = candidate.get("verdict_state")
    if explicit is not None:
        if explicit not in _STATES:
            raise ModelVerdictError("invalid_candidate_verdict_state")
        if explicit == "approved" and str(candidate.get("status", "")) in _REJECTED_STATUSES:
            raise ModelVerdictError("approved_candidate_cannot_have_hard_failure_status")
        if explicit == "approved" and not candidate.get("evidence_refs") and not candidate.get("approval_basis"):
            raise ModelVerdictError("approved_candidate_requires_evidence_reference")
        return str(explicit), "explicit_verdict_state"
    status = str(candidate.get("status", "unresolved"))
    if status in _REJECTED_STATUSES:
        return "rejected", f"status:{status}"
    if status in _UNRESOLVED_STATUSES:
        return "unresolved", f"status:{status}"
    return "conditional", f"status:{status}"


def _normalise_layered_uncertainty(value: Any) -> dict[str, Any]:
    default = {layer: {"status": "not_assessed", "evidence": []}
               for layer in ("semantic", "structural", "parameter", "numerical")}
    if not isinstance(value, Mapping):
        return default
    for layer in default:
        item = value.get(layer)
        if isinstance(item, Mapping):
            status = str(item.get("status", item.get("state", "not_assessed")))
            evidence = item.get("evidence", item.get("evidence_refs", []))
            if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
                evidence = []
            default[layer] = {"status": status[:80], "evidence": [str(ref)[:240] for ref in evidence[:32]]}
        elif item is not None:
            default[layer] = {"status": str(item)[:80], "evidence": []}
    return default


def build_model_verdict(
    candidates: Sequence[Mapping[str, Any]], *,
    uncertainty: Mapping[str, Any] | None = None,
    numerical_stability: Mapping[str, Any] | None = None,
    uncertainty_propagation: Mapping[str, Any] | None = None,
    assumptions: Sequence[str] | None = None,
    counterexamples: Sequence[Mapping[str, Any]] | None = None,
    next_questions: Sequence[Mapping[str, Any]] | None = None,
    run_id: str | None = None,
    evidence_refs: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a conservative verdict and traceability summary."""
    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise ModelVerdictError("candidates_must_be_a_sequence")
    if len(candidates) > 128:
        raise ModelVerdictError("candidate_count_out_of_bounds")
    normalized: list[dict[str, Any]] = []
    ids: set[str] = set()
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ModelVerdictError("candidate_must_be_an_object")
        identifier = candidate.get("id")
        if not isinstance(identifier, (str, int)) or isinstance(identifier, bool) or not str(identifier).strip():
            raise ModelVerdictError("candidate_id_required")
        identifier = str(identifier).strip()
        if identifier in ids:
            raise ModelVerdictError("duplicate_candidate_id")
        ids.add(identifier)
        state, basis = _state(candidate)
        refs = candidate.get("evidence_refs", [])
        if not isinstance(refs, Sequence) or isinstance(refs, (str, bytes)):
            refs = []
        reasons = candidate.get("failure_reasons", [])
        if not isinstance(reasons, Sequence) or isinstance(reasons, (str, bytes)):
            reasons = []
        normalized.append({
            "id": identifier,
            "label": str(candidate.get("label", identifier))[:240],
            "state": state,
            "state_basis": basis,
            "status": str(candidate.get("status", "not_assessed"))[:80],
            "decision": candidate.get("decision"),
            "evidence_refs": [str(ref)[:240] for ref in refs[:32]],
            "failure_reasons": [str(item)[:240] for item in reasons[:16]],
        })

    approved = [item["id"] for item in normalized if item["state"] == "approved"]
    assumptions_list = [] if assumptions is None else [str(item)[:500] for item in assumptions[:64]]
    counterexample_list = [] if counterexamples is None else [dict(item) for item in counterexamples[:128]
                                                               if isinstance(item, Mapping)]
    counterexample_ids = {
        str(item.get("candidate_id", item.get("id"))).strip()
        for item in counterexample_list
        if item.get("candidate_id", item.get("id")) is not None
    }
    global_counterexample = any(
        item.get("candidate_id", item.get("id")) is None for item in counterexample_list
    )
    blocked_approved = [identifier for identifier in approved
                        if global_counterexample or identifier in counterexample_ids]
    if blocked_approved:
        approved = [identifier for identifier in approved if identifier not in blocked_approved]
    usable = [item for item in normalized
              if item["state"] in {"approved", "conditional"} and item["id"] not in blocked_approved]
    decision_keys = [_json_key(item["decision"]) for item in usable if item.get("decision") is not None]
    consensus = bool(decision_keys) and len(set(decision_keys)) == 1
    shared_decision = next((item["decision"] for item in usable if item.get("decision") is not None), None) \
        if consensus else None
    unresolved = [item["id"] for item in normalized if item["state"] == "unresolved"]
    rejected = [item["id"] for item in normalized if item["state"] == "rejected"]
    overall = "approved" if len(approved) == 1 else ("conditional" if usable else "unresolved")
    question_list = [] if next_questions is None else [dict(item) for item in next_questions[:32]
                                                       if isinstance(item, Mapping)]
    all_refs = [str(item)[:240] for item in (evidence_refs or [])[:64]]
    all_refs.extend(ref for candidate in normalized for ref in candidate["evidence_refs"] if ref not in all_refs)
    layered_uncertainty = _normalise_layered_uncertainty(uncertainty)
    propagation_summary = None
    if isinstance(uncertainty_propagation, Mapping):
        components = uncertainty_propagation.get("variance_components", {})
        if isinstance(components, Mapping):
            for layer in ("semantic", "structural", "parameter", "numerical"):
                source_layer = "structure" if layer == "structural" else layer
                item = components.get(source_layer)
                if isinstance(item, Mapping):
                    raw_status = str(item.get("status", "not_assessed"))[:80]
                    layered_uncertainty[layer] = {
                        "status": "tested_not_falsified" if raw_status == "assessed" else "not_assessed",
                        "evidence": [f"uncertainty_propagation.{source_layer}"],
                    }
        propagation_summary = {
            "status": str(uncertainty_propagation.get("status", "not_assessed"))[:80],
            "unit_signature": str(uncertainty_propagation.get("unit_signature", ""))[:128],
            "record_count": _safe_count(uncertainty_propagation.get("record_count", 0)),
            "reconstructed_variance": uncertainty_propagation.get("reconstructed_variance")
            if isinstance(uncertainty_propagation.get("reconstructed_variance"), (int, float))
            and math.isfinite(float(uncertainty_propagation.get("reconstructed_variance"))) else None,
            "policy": "empirical_layered_variance_not_probability_or_confidence",
        }
    stability_summary = None
    if isinstance(numerical_stability, Mapping):
        # Numerical tolerance checks are evidence about the tested solver
        # settings, never a proof of numerical error bounds.  Keep only the
        # bounded status/counters/fingerprint so a raw solver payload cannot
        # silently enter a public verdict.
        stability_status = str(numerical_stability.get("status", "not_assessed"))[:80]
        status_map = {
            "stable_on_tested_tolerances": "tested_not_falsified",
            "unstable": "counterexample_found",
            "partial": "not_assessed",
            "not_assessed": "not_assessed",
        }
        layered_uncertainty["numerical"] = {
            "status": status_map.get(stability_status, "not_assessed"),
            "evidence": ["numerical_stability.tolerance_comparison"],
        }
        def _count(name: str) -> int:
            value = numerical_stability.get(name, 0)
            try:
                value = int(value)
            except (TypeError, ValueError, OverflowError):
                value = 0
            return max(0, min(value, 1_000_000))
        stability_summary = {
            "status": stability_status,
            "successful_runs": _count("successful_runs"),
            "failed_runs": _count("failed_runs"),
            "fingerprint": str(numerical_stability.get("fingerprint", ""))[:64],
            "policy": "finite_tolerance_comparison_not_numerical_error_proof",
        }
    warnings = []
    if unresolved:
        warnings.append("存在未决候选；资源不足、执行错误或缺少输入不等于反例淘汰。")
    if not approved:
        warnings.append("没有显式 approved 候选；当前输出不能作为最终获准结论。")
    if blocked_approved:
        warnings.append("显式获准候选存在对应反例记录，已撤销推荐并要求重新确认。")
    if stability_summary and stability_summary["status"] == "unstable":
        warnings.append("数值容差比较发现输出不稳定；当前数值结论不能脱离误差与求解设置使用。")
    if consensus:
        minimum = {"status": "supported_by_current_candidates", "decision": shared_decision,
                   "candidate_ids": [item["id"] for item in usable],
                   "interpretation": "候选在当前证据和假设下给出相同决策；不是置信区间或现实正确性证明。"}
    else:
        minimum = {"status": "not_established", "decision": None, "candidate_ids": [],
                   "interpretation": "候选决策不一致或没有可用候选，不能压缩为单一结论。"}
    return {
        "schema_version": SCHEMA_VERSION,
        "status": overall,
        "run_id": str(run_id)[:240] if run_id is not None else None,
        "candidate_count": len(normalized),
        "candidates": normalized,
        "approved_candidate_ids": approved,
        "unresolved_candidate_ids": unresolved,
        "rejected_candidate_ids": rejected,
        "recommended_candidate_id": approved[0] if len(approved) == 1 else None,
        "minimum_common_conclusion": minimum,
        "uncertainty": layered_uncertainty,
        "numerical_stability": stability_summary,
        "uncertainty_propagation": propagation_summary,
        "assumptions": assumptions_list,
        "counterexamples": counterexample_list,
        "next_questions": question_list,
        "evidence_refs": all_refs,
        "warnings": warnings,
        "policy": {
            "scores_are_not_probabilities": True,
            "conditional_is_not_approved": True,
            "unresolved_is_not_rejected": True,
            "counterexamples_block_recommendation": True,
            "requires_independent_confirmation": True,
        },
    }


__all__ = ["SCHEMA_VERSION", "ModelVerdictError", "build_model_verdict"]
