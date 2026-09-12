"""Turn bounded diagnostics into typed, reversible search hints.

This module is the narrow bridge between cheap evidence and expensive model
search.  It never edits an IR, changes a hard constraint, or labels a signal
as a causal mechanism.  A downstream searcher may use the returned primitive
IDs as *soft* proposals and must run its normal type/unit/holdout gates again.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mathmodel.diagnostic-router/v1"

_SIGNAL_ROUTES: dict[str, dict[str, Any]] = {
    "monotone_candidate": {
        "family": "monotone_response",
        "primitives": ("monotone_constraint", "saturating_response"),
        "questions": ("是否应在当前域内保持单调？",),
        "priority": 0.72,
    },
    "periodic_candidate": {
        "family": "periodic_or_oscillatory",
        "primitives": ("sinusoidal_basis", "oscillatory_state", "seasonal_lag"),
        "questions": ("周期是外部驱动还是系统自身振荡？",),
        "priority": 0.68,
    },
    "change_point_candidate": {
        "family": "piecewise_or_event",
        "primitives": ("piecewise_regime", "event_switch", "threshold_constraint"),
        "questions": ("变点是否对应题面明确事件或规则切换？",),
        "priority": 0.64,
    },
}

_RESIDUAL_ROUTES: dict[str, dict[str, Any]] = {
    "search_residual_memory": {
        "family": "memory_or_unobserved_input",
        "primitives": ("delay_embedding", "exogenous_input", "latent_state"),
        "questions": ("残差记忆是否来自滞后、外部输入或观测偏差？",),
        "priority": 0.66,
    },
    "search_error_scale_pattern": {
        "family": "heteroscedastic_observation",
        "primitives": ("variance_link", "multiplicative_noise", "robust_loss"),
        "questions": ("误差尺度变化是观测噪声还是机制失配？",),
        "priority": 0.61,
    },
    "search_residual_bias": {
        "family": "missing_offset_or_input",
        "primitives": ("intercept", "group_effect", "exogenous_input"),
        "questions": ("系统偏差是否来自遗漏输入、分组效应或基线偏移？",),
        "priority": 0.57,
    },
    "prediction_cv_instability": {
        "family": "stable_validation",
        "primitives": ("regularization", "robust_split", "simpler_model"),
        "questions": ("跨折波动来自样本划分、参数灵活性还是数据量不足？",),
        "priority": 0.63,
    },
    "prediction_residual_bias": {
        "family": "prediction_bias",
        "primitives": ("intercept", "calibration", "observation_bias"),
        "questions": ("预测偏差是目标变换、观测偏置还是遗漏机制？",),
        "priority": 0.60,
    },
    "prediction_residual_structure": {
        "family": "unexplained_prediction_structure",
        "primitives": ("nonlinear_basis", "interaction", "group_effect"),
        "questions": ("残差结构是否在独立分组或时间块上仍然存在？",),
        "priority": 0.62,
    },
    "prediction_heteroscedasticity_signal": {
        "family": "heteroscedastic_observation",
        "primitives": ("variance_link", "transform_target", "robust_loss"),
        "questions": ("误差尺度变化是观测噪声还是高值区机制变化？",),
        "priority": 0.58,
    },
    "classification_oof_error": {
        "family": "classification_boundary",
        "primitives": ("class_weight", "regularization", "label_audit"),
        "questions": ("错误主要来自类别不平衡、标签噪声还是特征缺失？",),
        "priority": 0.55,
    },
    "clustering_separation_weak": {
        "family": "cluster_structure",
        "primitives": ("distance_metric", "soft_clustering", "no_cluster_conclusion"),
        "questions": ("弱分离是距离度量问题还是数据本身没有离散群组？",),
        "priority": 0.52,
    },
    "clustering_seed_instability": {
        "family": "cluster_stability",
        "primitives": ("alternative_clustering", "regularization", "no_cluster_conclusion"),
        "questions": ("分组是否依赖初始化或簇数选择？",),
        "priority": 0.50,
    },
    "clustering_tiny_cluster": {
        "family": "cluster_outlier",
        "primitives": ("outlier_audit", "robust_clustering", "no_cluster_conclusion"),
        "questions": ("极小簇是异常点还是有业务意义的小群体？",),
        "priority": 0.48,
    },
    "periodic_structure_signal": {
        "family": "periodic_or_oscillatory",
        "primitives": ("seasonal_lag", "oscillatory_state", "external_periodic_input"),
        "questions": ("周期信号能否在未见时间段复现？",),
        "priority": 0.54,
    },
    "change_point_signal": {
        "family": "piecewise_or_event",
        "primitives": ("piecewise_regime", "event_switch", "change_point_test"),
        "questions": ("候选变点是否有独立事件或规则来源？",),
        "priority": 0.53,
    },
    "monotone_structure_signal": {
        "family": "monotone_response",
        "primitives": ("monotone_constraint", "saturating_response", "boundary_check"),
        "questions": ("单调性是否只在当前观测域内成立？",),
        "priority": 0.51,
    },
    "optimization_audit_warning": {
        "family": "optimization_validation",
        "primitives": ("constraint_recheck", "robust_scenario", "near_optimal_set"),
        "questions": ("约束违反、参数敏感性和近优方案是否已独立复核？",),
        "priority": 0.56,
    },
    "causal_audit_warning": {
        "family": "causal_identification",
        "primitives": ("confounder_audit", "placebo_test", "sensitivity_to_unobserved_confounding"),
        "questions": ("处理—结果方向和混杂控制是否有题面或设计证据？",),
        "priority": 0.59,
    },
    "ranking_audit_warning": {
        "family": "multiobjective_ranking",
        "primitives": ("weight_sensitivity", "pareto_front", "rank_stability"),
        "questions": ("权重扰动或无权重 Pareto 前沿是否改变决策？",),
        "priority": 0.50,
    },
    "simulation_audit_warning": {
        "family": "uncertainty_simulation",
        "primitives": ("bootstrap_recheck", "scenario_stress", "distribution_sensitivity"),
        "questions": ("区间覆盖与情景分布假设是否有独立证据？",),
        "priority": 0.49,
    },
}


class DiagnosticRouterError(ValueError):
    """Raised when a diagnostic payload is outside the routing contract."""


def _stable_hash(value: Any) -> str:
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DiagnosticRouterError("diagnostic_payload_must_be_finite_json") from exc
    return sha256(raw).hexdigest()


def _bounded_evidence(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DiagnosticRouterError("diagnostic_evidence_must_be_mapping")
    try:
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise DiagnosticRouterError("diagnostic_evidence_must_be_finite_json") from exc
    if len(encoded.encode("utf-8")) > 32_000:
        raise DiagnosticRouterError("diagnostic_evidence_too_large")
    return dict(value)


def _route(signal: str, *, source: str, subject: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
    config = (_SIGNAL_ROUTES if source == "structure" else _RESIDUAL_ROUTES).get(signal)
    if config is None:
        raise DiagnosticRouterError(f"unsupported_diagnostic_signal:{signal}")
    if not isinstance(subject, str) or not subject.strip() or len(subject) > 160:
        raise DiagnosticRouterError("diagnostic_subject_invalid")
    evidence = _bounded_evidence(evidence)
    return {
        "id": "hint_" + _stable_hash({"source": source, "subject": subject, "signal": signal})[:20],
        "source": source,
        "subject": subject.strip(),
        "signal": signal,
        "family": config["family"],
        "candidate_primitives": list(config["primitives"]),
        "priority": float(config["priority"]),
        "evidence": evidence,
        "clarifying_questions": list(config["questions"]),
        "status": "proposal_not_executed",
        "hard_constraint": False,
        "may_feed_search": True,
        "requires_current_validation": True,
        "not_claimed": ["causal_edge", "mechanism_proof", "hidden_state_existence"],
    }


def route_structure_diagnostics(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Route ``diagnose_series_structure`` output into reversible hints."""
    if not isinstance(payload, Mapping) or payload.get("schema_version") != "mathmodel.structure-diagnostics/v1":
        raise DiagnosticRouterError("invalid_structure_diagnostics_schema")
    diagnostics = payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping) or len(diagnostics) > 128:
        raise DiagnosticRouterError("invalid_structure_diagnostics_records")
    hints: list[dict[str, Any]] = []
    for subject, record in diagnostics.items():
        if not isinstance(subject, str) or not isinstance(record, Mapping):
            raise DiagnosticRouterError("invalid_structure_diagnostic_record")
        if record.get("status") != "signal":
            continue
        signals = record.get("signals", ())
        if not isinstance(signals, (list, tuple)) or len(signals) > 32 or any(type(item) is not str for item in signals):
            raise DiagnosticRouterError("invalid_structure_signals")
        for signal in sorted(set(item.strip() for item in signals if item.strip())):
            if signal in _SIGNAL_ROUTES:
                hints.append(_route(signal, source="structure", subject=subject, evidence={
                    "trajectory_sha256": payload.get("trajectory_sha256"),
                    "signal_record": dict(record),
                }))
    return _result(hints, payload_hash=payload.get("trajectory_sha256"))


def route_model_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Route development diagnostic records without routing locked-test failures."""
    if not isinstance(records, (list, tuple)) or len(records) > 256:
        raise DiagnosticRouterError("invalid_model_diagnostics_records")
    hints: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise DiagnosticRouterError("invalid_model_diagnostic_record")
        context = record.get("context", {})
        if not isinstance(context, Mapping):
            raise DiagnosticRouterError("invalid_model_diagnostic_context")
        if context.get("phase") not in {"development", "execution", "preflight"}:
            continue
        if context.get("may_feed_search") is not True:
            continue
        code = record.get("code", "")
        if not isinstance(code, str) or len(code) > 160:
            raise DiagnosticRouterError("invalid_model_diagnostic_code")
        if code in _RESIDUAL_ROUTES:
            subject = record.get("id", "unknown")
            if not isinstance(subject, str):
                raise DiagnosticRouterError("diagnostic_subject_invalid")
            evidence = record.get("evidence", {})
            _bounded_evidence(evidence)
            hints.append(_route(code, source="residual", subject=subject,
                                evidence={"diagnostic_id": record.get("id"),
                                          "diagnostic_state": record.get("state"),
                                          "evidence": evidence}))
    return _result(hints, payload_hash=_stable_hash(records))


def _result(hints: list[dict[str, Any]], *, payload_hash: Any) -> dict[str, Any]:
    unique = {hint["id"]: hint for hint in hints}
    ordered = sorted(unique.values(), key=lambda item: (-item["priority"], item["id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "hints": ordered,
        "hint_count": len(ordered),
        "input_fingerprint": str(payload_hash or ""),
        "policy": "diagnostic_signal_only; downstream_validation_required",
        "may_modify_ir": False,
        "may_change_hard_constraints": False,
    }


__all__ = ["SCHEMA_VERSION", "DiagnosticRouterError", "route_structure_diagnostics", "route_model_diagnostics"]
