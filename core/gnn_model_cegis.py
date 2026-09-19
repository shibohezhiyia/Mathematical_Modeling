"""Typed GNN interaction family adapter for dynamic model competition.

The adapter deliberately treats a GNN as a bounded predictive model family.
It accepts rows through JSON-compatible case payloads, runs the existing local
interaction screen, and exposes a common prediction/metrics contract.  It
does not turn learned gates into causal edges or a mechanism proof.
"""
from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .gnn_interaction_screen import GNNInteractionError, discover_gnn_interactions
from .model_family_adapters import ModelFamilyAdapter, run_model_family_cegis


class GNNModelCEGISError(ValueError):
    pass


_CANDIDATE_FIELDS = {
    "id", "target", "columns", "max_variables", "max_rows", "epochs",
    "hidden_dim", "restarts", "message_layers", "group_column", "time_column",
    "dynamic_windows", "edge_threshold", "validation_fraction", "random_state",
}


def compile_gnn_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise GNNModelCEGISError("gnn_candidate_must_be_object")
    if set(candidate) - _CANDIDATE_FIELDS:
        raise GNNModelCEGISError("gnn_candidate_contains_unknown_fields")
    target = candidate.get("target")
    if target is not None and (not isinstance(target, str) or not target.strip()):
        raise GNNModelCEGISError("gnn_target_invalid")
    columns = candidate.get("columns")
    if columns is not None and (not isinstance(columns, list) or len(columns) > 32
                                or any(not isinstance(item, str) for item in columns)):
        raise GNNModelCEGISError("gnn_columns_invalid")
    integer_limits = {
        "max_variables": (2, 32), "max_rows": (60, 10_000), "epochs": (10, 500),
        "hidden_dim": (4, 64), "restarts": (1, 3), "message_layers": (1, 3),
        "dynamic_windows": (0, 5), "random_state": (0, 2**32 - 1),
    }
    normalized: dict[str, Any] = {"id": str(candidate.get("id", "gnn_candidate"))[:120]}
    if target is not None:
        normalized["target"] = target
    if columns is not None:
        normalized["columns"] = list(columns)
    for name, (lower, upper) in integer_limits.items():
        if name in candidate:
            value = candidate[name]
            if type(value) is not int or not lower <= value <= upper:
                raise GNNModelCEGISError(f"gnn_{name}_invalid")
            normalized[name] = value
    for name, lower, upper in (
        ("edge_threshold", 0.0, 1.0),
        ("validation_fraction", 0.1, 0.4),
    ):
        if name in candidate:
            value = candidate[name]
            if type(value) not in (int, float) or not math.isfinite(float(value)) or not lower < float(value) < upper:
                raise GNNModelCEGISError(f"gnn_{name}_invalid")
            normalized[name] = float(value)
    for name in ("group_column", "time_column"):
        if name in candidate:
            value = candidate[name]
            if value is not None and (not isinstance(value, str) or not value):
                raise GNNModelCEGISError(f"gnn_{name}_invalid")
            normalized[name] = value
    if normalized.get("group_column") and normalized.get("group_column") == normalized.get("time_column"):
        raise GNNModelCEGISError("gnn_group_and_time_columns_must_differ")
    return normalized


def _case_frame(case: Mapping[str, Any]) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    rows = case.get("rows", case.get("data"))
    if not isinstance(rows, list) or not rows or any(not isinstance(row, Mapping) for row in rows):
        raise GNNModelCEGISError("gnn_case_rows_invalid")
    frame = pd.DataFrame([dict(row) for row in rows])
    target = case.get("target")
    if not isinstance(target, str) or target not in frame.columns:
        raise GNNModelCEGISError("gnn_case_target_invalid")
    options = case.get("options", {})
    if not isinstance(options, Mapping):
        raise GNNModelCEGISError("gnn_case_options_invalid")
    return frame, target, dict(options)


def evaluate_gnn_candidate(compiled: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases or len(cases) > 32:
        raise GNNModelCEGISError("gnn_cases_invalid")
    scores: list[float] = []
    all_edges = []
    details = []
    violations = []
    for index, raw_case in enumerate(cases):
        if not isinstance(raw_case, Mapping):
            return {"status": "not_assessed", "failure_code": "gnn_case_invalid", "cost_units": index + 1}
        try:
            frame, case_target, options = _case_frame(raw_case)
            call = dict(options)
            call.update({key: compiled[key] for key in compiled if key not in {"id", "target"} and key not in call})
            target = str(compiled.get("target", case_target))
            if target not in frame.columns:
                raise GNNModelCEGISError("gnn_target_not_in_case")
            result = discover_gnn_interactions(frame, target, **call)
        except (GNNInteractionError, TypeError, ValueError, KeyError, OverflowError) as exc:
            return {"status": "not_assessed", "failure_code": str(exc)[:100], "cost_units": index + 1}
        if result.get("status") != "executed" or result.get("validation_rmse") is None:
            return {"status": "not_assessed", "failure_code": str(result.get("reason", "gnn_not_executed")),
                    "details": details, "cost_units": index + 1}
        score = float(result["validation_rmse"])
        if not math.isfinite(score):
            return {"status": "not_assessed", "failure_code": "gnn_rmse_not_finite", "cost_units": index + 1}
        scores.append(score)
        all_edges.extend(result.get("edges", []))
        details.append({"case": str(raw_case.get("id", f"case_{index}"))[:80],
                        "validation_rmse": score, "rows": result.get("rows"),
                        "split_policy": result.get("split_policy"), "edge_count": len(result.get("edges", []))})
        bound = raw_case.get("max_metric")
        if bound is not None and (type(bound) not in (int, float) or not math.isfinite(float(bound))):
            return {"status": "not_assessed", "failure_code": "gnn_case_metric_bound_invalid", "cost_units": index + 1}
        if bound is not None and score > float(bound):
            violations.append({"reason": "gnn_validation_rmse_exceeds_bound", "witness_id": str(raw_case.get("id", f"case_{index}"))[:80],
                               "validation_rmse": score, "max_metric": float(bound)})
    complexity = float((len(compiled.get("columns", [])) or 0) + len(all_edges)
                       + int(compiled.get("message_layers", 1)))
    instability = float(np.std(scores) if len(scores) > 1 else 0.0)
    return {
        "status": "pass" if not violations else "fail", "score": float(np.mean(scores)),
        "predictions": scores, "metrics": {
            "validation_loss": float(np.mean(scores)), "complexity": complexity,
            "constraint_violation": float(len(violations)), "instability": instability,
        }, "violations": violations[:16], "details": details,
        "edges": all_edges[:256], "cost_units": len(cases),
        "policy": "bounded_gnn_predictive_screen;_not_causal_discovery_or_mechanism_proof",
    }


def diagnose_gnn_feedback(feedback: Mapping[str, Any]) -> dict[str, Any]:
    return {"step": 0.0, "failure_code": str(feedback.get("failure_code", ""))[:100]} if isinstance(feedback, Mapping) else {"step": 0.0}


def patch_gnn_candidate(candidate: Mapping[str, Any], diagnostic: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    # GNN hyperparameter changes are not treated as mathematical repairs.  A
    # future search can add bounded mutations after an independent budget is
    # specified; for now a failed screen remains unresolved.
    return ()


def build_gnn_cegis_adapter() -> ModelFamilyAdapter:
    return ModelFamilyAdapter(
        family="gnn", compile=compile_gnn_candidate, evaluate=evaluate_gnn_candidate,
        diagnose=diagnose_gnn_feedback, patch=patch_gnn_candidate,
        replay=evaluate_gnn_candidate,
    )


def run_gnn_cegis(initial_candidates: Iterable[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], **kwargs: Any) -> dict[str, Any]:
    return run_model_family_cegis(build_gnn_cegis_adapter(), initial_candidates, cases, **kwargs)


__all__ = ["GNNModelCEGISError", "compile_gnn_candidate", "evaluate_gnn_candidate",
           "build_gnn_cegis_adapter", "run_gnn_cegis"]
