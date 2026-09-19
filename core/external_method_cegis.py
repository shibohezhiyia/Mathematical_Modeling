"""Bounded CEGIS adapters for typed external-method runtimes.

This bridge closes a practical gap between the standalone PDE/UDE/symbolic
executors and the common model-family loop.  A candidate is a typed method
payload, never source code.  A case must provide an independently chosen
metric bound; merely returning ``executed`` is not enough to accept a model.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .cegis_controller import CEGISConfig
from .external_method_runtime import execute_external_method
from .model_family_adapters import ModelFamilyAdapter, run_model_family_cegis


class ExternalMethodCEGISError(ValueError):
    pass


_SUPPORTED = frozenset({"pde_find", "ude", "ude_neural", "ude_joint", "ude_stiff", "llm_sr"})
_METRIC_BY_METHOD = {
    "pde_find": "validation_rmse",
    "ude": "holdout_rmse",
    "ude_neural": "holdout_rmse",
    "ude_joint": "holdout_rmse",
    "ude_stiff": "trajectory_rmse",
    "llm_sr": "holdout_rmse",
}


def compile_external_method_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise ExternalMethodCEGISError("external_candidate_must_be_object")
    method = candidate.get("method")
    payload = candidate.get("payload")
    if method not in _SUPPORTED:
        raise ExternalMethodCEGISError("external_method_not_supported_by_cegis")
    if not isinstance(payload, Mapping):
        raise ExternalMethodCEGISError("external_candidate_payload_must_be_object")
    # Explicitly reject code/path-shaped fields even though the downstream
    # runtime has its own allow-list.  This keeps the CEGIS contract narrow.
    forbidden = {"source", "code", "path", "repository", "module", "command"}
    if forbidden.intersection(payload):
        raise ExternalMethodCEGISError("external_candidate_source_fields_forbidden")
    return {"id": str(candidate.get("id", f"{method}_candidate"))[:120],
            "method": method, "payload": dict(payload)}


def _case_payload(compiled: Mapping[str, Any], case: Mapping[str, Any]) -> tuple[dict[str, Any], float, str]:
    if not isinstance(case, Mapping):
        raise ExternalMethodCEGISError("external_case_must_be_object")
    overrides = case.get("payload_overrides", {}) or {}
    if not isinstance(overrides, Mapping):
        raise ExternalMethodCEGISError("external_case_overrides_must_be_object")
    if set(overrides).intersection({"method", "source", "code", "path", "repository", "module", "command"}):
        raise ExternalMethodCEGISError("external_case_source_fields_forbidden")
    bound = case.get("max_metric")
    if type(bound) not in (int, float) or not math.isfinite(float(bound)) or float(bound) < 0:
        raise ExternalMethodCEGISError("external_case_max_metric_required")
    method = str(compiled["method"])
    metric = str(case.get("metric", _METRIC_BY_METHOD[method]))
    payload = dict(compiled["payload"])
    payload.update(dict(overrides))
    return payload, float(bound), metric


def evaluate_external_method_candidate(
    compiled: Mapping[str, Any], cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases or len(cases) > 32:
        raise ExternalMethodCEGISError("external_cases_invalid")
    violations: list[dict[str, Any]] = []
    scores: list[float] = []
    complexity = 0.0
    for index, case in enumerate(cases):
        try:
            payload, bound, metric = _case_payload(compiled, case)
            result = execute_external_method(str(compiled["method"]), payload)
        except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
            return {"status": "not_assessed", "failure_code": "external_method_failed",
                    "diagnostic": type(exc).__name__, "violations": [], "cost_units": index + 1}
        if result.get("status") != "executed" or not isinstance(result.get("result"), Mapping):
            return {"status": "not_assessed", "failure_code": "external_method_not_executed",
                    "diagnostic": result.get("reason", result.get("status")),
                    "violations": [], "cost_units": index + 1}
        value = result["result"].get(metric)
        if metric == "trajectory_rmse" and value is None:
            observations = case.get("observations") if isinstance(case, Mapping) else None
            trajectory = result["result"].get("trajectory")
            try:
                observed = np.asarray(observations, dtype=float)
                predicted = np.asarray(trajectory, dtype=float)
                if observed.shape != predicted.shape or observed.ndim != 2:
                    raise ValueError("trajectory_shape_mismatch")
                value = float(np.sqrt(np.mean((predicted - observed) ** 2)))
            except (TypeError, ValueError, OverflowError):
                value = None
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            return {"status": "not_assessed", "failure_code": "external_metric_missing",
                    "diagnostic": metric, "violations": [], "cost_units": index + 1}
        value = float(value)
        scores.append(value)
        complexity = max(complexity, float(len(compiled.get("payload", {}))))
        if value > bound:
            violations.append({"reason": "external_metric_bound_exceeded",
                               "witness_id": str(case.get("id", f"case_{index}"))[:80],
                               "metric": metric, "value": value, "bound": bound})
    mean_score = float(sum(scores) / len(scores)) if scores else None
    return {"status": "pass" if not violations else "fail",
            "score": -mean_score if mean_score is not None else None,
            "predictions": scores,
            "metrics": ({
                "validation_loss": mean_score,
                "complexity": complexity,
                "constraint_violation": 0.0,
                # Finite metric execution is only a numerical guard proxy;
                # it is not a PDE stability or physical-validity certificate.
                "instability": 0.0,
            } if mean_score is not None else {}),
            "violations": violations[:16], "cost_units": len(cases),
            "policy": "typed_external_method_replay;_independent_metric_bound_required;instability_not_physical_stability"}


def diagnose_external_method_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    return {"step": 0.05, "failure_code": str(feedback.get("failure_code", ""))[:100]}


def patch_external_method_candidate(
    candidate: Mapping[str, Any], diagnostic: Mapping[str, Any],
) -> Iterable[Mapping[str, Any]]:
    normalized = compile_external_method_candidate(candidate)
    method = normalized["method"]
    payload = normalized["payload"]
    proposals: list[dict[str, Any]] = []
    if method == "pde_find":
        current = float(payload.get("sparsity_threshold", 0.0))
        for value in sorted({0.0, min(1.0, current + 0.1)}):
            revised = dict(normalized); revised["payload"] = {**payload, "sparsity_threshold": value}
            revised["id"] = f"{normalized['id']}_sparsity_{value:g}"; proposals.append(revised)
        if "include_advection" in payload:
            revised = dict(normalized); revised["payload"] = {**payload, "include_advection": not bool(payload["include_advection"])}
            revised["id"] = f"{normalized['id']}_advection_flip"; proposals.append(revised)
    elif method == "ude":
        current = float(payload.get("ridge", 1e-8))
        for value in sorted({max(1e-10, current * 0.1), min(1e3, current * 10.0)}):
            revised = dict(normalized); revised["payload"] = {**payload, "ridge": value}
            revised["id"] = f"{normalized['id']}_ridge_{value:g}"; proposals.append(revised)
    elif method in {"ude_neural", "ude_joint"}:
        current = int(payload.get("hidden_dim", 16))
        upper = 32 if method == "ude_joint" else 64
        for value in sorted({max(4, current // 2), min(upper, current * 2)}):
            revised = dict(normalized); revised["payload"] = {**payload, "hidden_dim": value}
            revised["id"] = f"{normalized['id']}_hidden_{value}"; proposals.append(revised)
    elif method == "llm_sr" and "search" in payload:
        search = dict(payload["search"])
        current = int(search.get("generations", 1))
        if current < 8:
            revised = dict(normalized); revised["payload"] = {**payload, "search": {**search, "generations": current + 1}}
            revised["id"] = f"{normalized['id']}_generation_{current + 1}"; proposals.append(revised)
    return proposals


def build_external_method_cegis_adapter(method: str) -> ModelFamilyAdapter:
    if method not in _SUPPORTED:
        raise ExternalMethodCEGISError("external_method_not_supported_by_cegis")
    return ModelFamilyAdapter(
        family=method,
        compile=compile_external_method_candidate,
        evaluate=evaluate_external_method_candidate,
        diagnose=diagnose_external_method_feedback,
        patch=patch_external_method_candidate,
        replay=evaluate_external_method_candidate,
    )


def run_external_method_cegis(
    method: str, initial_candidates: Iterable[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]], *, config: CEGISConfig | None = None,
) -> dict[str, Any]:
    return run_model_family_cegis(build_external_method_cegis_adapter(method), initial_candidates, cases, config=config)


__all__ = ["ExternalMethodCEGISError", "compile_external_method_candidate",
           "evaluate_external_method_candidate", "build_external_method_cegis_adapter",
           "run_external_method_cegis"]
