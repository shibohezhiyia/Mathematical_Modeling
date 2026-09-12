"""Pre-registered comparison arms for the external-method adapters."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .ablation_protocol import build_ablation_manifest
from .comparison_protocol import run_preregistered_comparison
from .external_method_runtime import ExternalMethodRuntimeError, execute_external_method


EXTERNAL_METHOD_SCORING = (
    "validation_residual_rmse",
    "valid_rate", "contract_correct", "evidence_completeness",
    "stability", "latency_seconds", "peak_memory_mb", "api_cost",
)


def external_method_arms() -> list[dict[str, Any]]:
    """Return fixed local/proposal arms; no evaluator is invoked."""
    return [
        {"id": "local_sindy", "label": "本地 SINDy", "baseline": True,
         "components": {"method": "sindy", "external_code": False}},
        {"id": "local_weak_sindy", "label": "本地 weak-form SINDy", "baseline": False,
         "components": {"method": "weak_sindy", "external_code": False}},
        {"id": "local_pde_library", "label": "本地 PDE 特征库", "baseline": False,
         "components": {"method": "pde_find", "external_code": False}},
        {"id": "local_ude", "label": "本地 UDE 线性修正", "baseline": False,
         "components": {"method": "ude", "external_code": False}},
        {"id": "llm_sr_proposal", "label": "LLM-SR 提议态", "baseline": False,
         "components": {"method": "llm_sr", "external_code": True,
                         "execution": "proposal_only"}},
    ]


def build_external_method_manifest(*, task_ids: Sequence[str], fixed_budget: Mapping[str, Any],
                                   final_test_fingerprint: str) -> dict[str, Any]:
    """Freeze an external-method comparison without running any arm."""
    return build_ablation_manifest(
        external_method_arms(), task_ids=task_ids, fixed_budget=fixed_budget,
        final_test_fingerprint=final_test_fingerprint,
        scoring_criteria=EXTERNAL_METHOD_SCORING,
    )


def run_external_method_comparison(
    manifest: Mapping[str, Any], tasks: Sequence[Mapping[str, Any]], *,
    wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Run the registered local adapters on a pre-registered task grid.

    Each task supplies ``payloads`` keyed by method name.  The payload is a
    development input only; final-test data must remain outside this function.
    A method that is proposal-only or lacks a suitable local score is retained
    as a rejected/incomplete row instead of being silently dropped.
    """
    def evaluate(arm: Mapping[str, Any], task: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        components = arm.get("components", {})
        method = components.get("method") if isinstance(components, Mapping) else None
        payloads = task.get("payloads", {})
        if not isinstance(method, str) or not isinstance(payloads, Mapping):
            return {"status": "rejected", "valid": False, "error_code": "task_payloads_invalid"}
        payload = payloads.get(method)
        if not isinstance(payload, Mapping):
            return {"status": "rejected", "valid": False, "error_code": "method_payload_missing"}
        try:
            execution = execute_external_method(method, payload)
        except ExternalMethodRuntimeError as exc:
            return {"status": "rejected", "valid": False,
                    "error_code": str(exc)}
        if execution.get("status") == "proposal_only":
            return {"status": "rejected", "valid": False, "error_code": "proposal_only"}
        if execution.get("status") != "executed":
            return {"status": "rejected", "valid": False, "error_code": "method_execution_rejected"}
        result = execution.get("result", {})
        if method == "sindy":
            values = result.get("validation_derivative_rmse", ())
            valid = result.get("status") == "candidate_found"
        elif method == "weak_sindy":
            values = result.get("validation_weak_form_rmse", ())
            valid = result.get("status") == "candidate_generated"
        elif method == "ude":
            values = [result.get("holdout_rmse")]
            valid = result.get("status") == "fitted"
        else:
            return {"status": "incomplete", "valid": False, "error_code": "no_numeric_score"}
        try:
            finite_values = [float(value) for value in values]
        except (TypeError, ValueError, OverflowError):
            return {"status": "error", "valid": False, "error_code": "score_not_numeric"}
        if not finite_values:
            return {"status": "error", "valid": False, "error_code": "score_missing"}
        if not all(math.isfinite(value) for value in finite_values):
            return {"status": "error", "valid": False, "error_code": "score_not_finite"}
        return {"status": "completed", "valid": bool(valid),
                "score": sum(finite_values) / len(finite_values)}

    result = run_preregistered_comparison(manifest, tasks, evaluate, wall_seconds=wall_seconds)
    result["primary_metric"] = "validation_residual_rmse"
    result["score_direction"] = "lower_is_better"
    return result


__all__ = ["EXTERNAL_METHOD_SCORING", "external_method_arms", "build_external_method_manifest",
           "run_external_method_comparison"]
