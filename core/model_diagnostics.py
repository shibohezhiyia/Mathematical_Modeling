"""Evidence-scoped diagnostics and proposed repair routes, never a repair executor.

Development diagnostics receive train/search arrays only. Locked-test failures
are reporting-only and cannot enter the feedback selected for future search.
Thresholds below are descriptive screening rules, not significance tests or
proof of causality, hidden states, structural correctness or identifiability.
"""

from __future__ import annotations

import copy
from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

from .unclosed_state_competition import compete_unclosed_state_explanations


SCHEMA_VERSION = "mathmodel.model-diagnostics/v1"
DEVELOPMENT_VERSION = "mathmodel.development-diagnostics/v1"
MAX_DIAGNOSTICS = 64
_POLICY = {
    "minimum_pattern_windows": 12, "lag_correlation_warning": 0.4,
    "bias_to_baseline_warning": 0.25, "scale_correlation_warning": 0.5,
    "condition_warning": 1e8, "support_jaccard_warning": 0.7,
    "thresholds_are": "uncalibrated_screening_not_significance_tests",
}
# Server-owned routes. The model cannot supply commands, altered thresholds,
# replacement facts, or permission to execute a proposed action.
_ACTIONS = {
    "inspect_resources": ("execution", "检查规模、稀疏表示和资源预算", "不得删除现实约束或缩短最终验证来制造通过。"),
    "replay_numerics": ("numerical", "复核尺度、定义域、容差与求解算法", "保留验收阈值，复算后重新检查残差与约束。"),
    "check_observations": ("data", "核验观测、初值与对齐方式", "不得补造标签、使用未来值填充或改变锁定划分。"),
    "inspect_parameterization": ("parameter", "检查活跃基底的秩与参数化", "正则化给出的唯一解不等于数据识别了参数；改写后在原方程复核。"),
    "compare_noise_models": ("observation", "比较测量偏差或噪声模型假设", "噪声解释仍需独立检验，不能用放大噪声掩盖失配。"),
    "compare_memory_models": ("structure", "比较滞后、驱动或记忆机制分支", "须与观测噪声、数值误差及较简单模型对照，不给潜状态自动命名。"),
    "compare_mechanisms": ("structure", "比较简单基线与不同机制假设", "只修改候选结构；保持题面约束、评价指标及最终测试不变。"),
    "confirm_semantics": ("semantic", "澄清单位、角色或缺失条件", "涉及事实或题意契约的变更必须由用户确认并产生新版本。"),
    "repair_typed_graph": ("structure", "修正候选图的类型或依赖", "重新通过原类型与量纲检查，不能关闭检查器。"),
    "resolve_upstream": ("dependency", "先恢复上游可用证据", "不得将缺失上游结果替换为零或其他占位值。"),
    "new_confirmation": ("validation", "保留失败并另设未使用的确认数据", "若据此修改模型，原最终测试已被使用，不能再次宣称独立验证。"),
}


def _hash(value: Any) -> str:
    return sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, allow_nan=False,
                             separators=(",", ":")).encode("ascii")).hexdigest()


def _finite(value: Any) -> float | None:
    return float(value) if type(value) in (int, float) and math.isfinite(value) else None


def _record(code, category, summary, context, evidence, actions=(), alternatives=(), *, state="observed"):
    identifier = "diagnostic_" + _hash({"code": code, "context": context})[:20]
    return {
        "id": identifier, "code": code, "category": category, "state": state,
        "summary": summary, "context": copy.deepcopy(context), "evidence": evidence,
        "alternative_explanations": list(alternatives), "action_ids": list(actions),
        "authority": "diagnostic_only", "mathematical_verdict": "not_assessed",
    }


def _routes(records: list[dict]) -> list[dict]:
    routes = []
    for record in records:
        for key in record["action_ids"]:
            target, label, guard = _ACTIONS[key]
            routes.append({
                "diagnostic_id": record["id"], "action_id": key, "target": target,
                "label": label, "guard": guard, "status": "proposed_not_executed",
                "requires_user_confirmation": target == "semantic",
                "may_execute": False,
            })
    return routes


def _matrix(value: np.ndarray, *, columns=64) -> np.ndarray:
    if (not isinstance(value, np.ndarray) or value.ndim != 2 or value.dtype.kind not in "fi"
            or not 1 <= value.shape[0] <= 5000 or not 1 <= value.shape[1] <= columns):
        raise ValueError("diagnostic_matrix_budget_or_shape")
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("diagnostic_inputs_must_be_finite")
    return result


_PREDICTION_DIAGNOSTIC_KEYS = (
    "primary_metric", "fold_relative_std", "normalized_bias",
    "residual_prediction_correlation", "heteroscedasticity_signal",
    "residual_p90_absolute", "oof_error_rate", "minority_error_warning",
)


def _prediction_scalar(value: Any) -> float | bool | str | None:
    """Keep only bounded scalar diagnostics; never copy predictions or labels."""
    if type(value) is bool:
        return value
    if type(value) in (int, float):
        return _finite(value)
    if type(value) is str:
        return value[:120]
    return None


def _prediction_diagnostic_records(
    prediction_results: Sequence[Mapping[str, Any]] | None,
    add,
) -> tuple[list[dict], int, int]:
    """Translate OOF/development diagnostics into the common read-only channel.

    ``actual`` and ``oof_prediction`` are deliberately not inspected.  The
    caller may pass complete model results, but only the server-produced
    scalar diagnostic summary is admitted into this panel and its search
    feedback.  This keeps the diagnostic artifact bounded and prevents a
    prediction payload (or a locked confirmation array) from leaking into the
    repair channel.
    """
    checks: list[dict] = []
    checked, available = 0, 0
    for result in list(prediction_results or [])[:32]:
        if not isinstance(result, Mapping):
            continue
        feedback = result.get("feedback_optimization")
        diagnostics = feedback.get("diagnostics") if isinstance(feedback, Mapping) else None
        if not isinstance(diagnostics, Mapping):
            diagnostics = result.get("validation_diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        available += 1
        task = str(result.get("task_type", "unknown"))[:80]
        dataset = str(result.get("dataset", "unknown"))[:160]
        target = str(result.get("target", "unknown"))[:160]
        subject = f"{dataset}.{target}" if target != "unknown" else dataset
        evidence = {
            key: _prediction_scalar(diagnostics.get(key))
            for key in _PREDICTION_DIAGNOSTIC_KEYS
            if _prediction_scalar(diagnostics.get(key)) is not None
        }
        # Do not let a user/model-controlled name become an executable route
        # or an unbounded artifact.  The hash is only provenance, not proof.
        context = {
            "phase": "development", "partition": "oof_and_inner_cv_only",
            "evaluator_version": SCHEMA_VERSION, "may_feed_search": True,
            "subject": subject, "task_type": task,
            "input_hash": _hash({"dataset": dataset, "target": target,
                                  "task_type": task, "evidence": evidence}),
        }
        check = {"subject": subject, "task_type": task, "evidence": evidence}
        checks.append(check)
        checked += 1
        fold_std = diagnostics.get("fold_relative_std")
        if _finite(fold_std) is not None and float(fold_std) > 0.1:
            add(_record("prediction_cv_instability", "parameter",
                        "预测模型跨折波动偏高，不能只按平均分选择模型。",
                        context, {"fold_relative_std": float(fold_std),
                                  "primary_metric": evidence.get("primary_metric")},
                        ["inspect_parameterization", "compare_mechanisms"],
                        ["数据分层不稳定", "样本量不足", "模型参数过灵活"], state="suspected"))
        task_is_regression = task == "regression"
        bias = diagnostics.get("normalized_bias")
        if task_is_regression and _finite(bias) is not None and abs(float(bias)) > 0.1:
            add(_record("prediction_residual_bias", "observation",
                        "交叉验证残差存在系统偏差，需区分观测偏置与机制失配。",
                        context, {"normalized_bias": float(bias)},
                        ["check_observations", "compare_noise_models", "compare_mechanisms"],
                        ["测量偏置", "目标变换不合适", "遗漏非线性机制"], state="suspected"))
        residual_corr = diagnostics.get("residual_prediction_correlation")
        if task_is_regression and _finite(residual_corr) is not None and abs(float(residual_corr)) > 0.15:
            add(_record("prediction_residual_structure", "structure",
                        "残差仍与预测值相关，当前函数族可能没有解释完系统结构。",
                        context, {"residual_prediction_correlation": float(residual_corr)},
                        ["compare_mechanisms", "inspect_parameterization"],
                        ["非线性遗漏", "异方差", "数据分组或时间漂移"], state="suspected"))
        hetero = diagnostics.get("heteroscedasticity_signal")
        if task_is_regression and _finite(hetero) is not None and abs(float(hetero)) > 0.15:
            add(_record("prediction_heteroscedasticity_signal", "observation",
                        "残差幅度随预测规模变化，异方差只是候选解释而非统计结论。",
                        context, {"heteroscedasticity_signal": float(hetero)},
                        ["compare_noise_models", "compare_mechanisms"],
                        ["异方差观测", "尺度变换缺失", "高值区机制变化"], state="suspected"))
        error_rate = diagnostics.get("oof_error_rate")
        if task == "classification" and _finite(error_rate) is not None and float(error_rate) > 0.25:
            add(_record("classification_oof_error", "structure",
                        "分类 OOF 错误率偏高，类别边界或标签质量仍需检查。",
                        context, {"oof_error_rate": float(error_rate)},
                        ["check_observations", "compare_mechanisms", "inspect_parameterization"],
                        ["类别不平衡", "标签噪声", "特征缺失"], state="suspected"))
    return checks, checked, available


def _clustering_diagnostic_records(
    model_results: Sequence[Mapping[str, Any]] | None,
    add,
) -> tuple[list[dict], int]:
    """Route clustering credibility warnings without treating labels as truth."""
    checks: list[dict] = []
    checked = 0
    for result in list(model_results or [])[:32]:
        if not isinstance(result, Mapping) or result.get("task_type") != "clustering":
            continue
        audit = result.get("credibility_audit")
        if not isinstance(audit, Mapping) or not isinstance(audit.get("checks"), list):
            continue
        checked += 1
        dataset = str(result.get("dataset", "unknown"))[:160]
        subject = dataset
        context = {
            "phase": "development", "partition": "cluster_screening_only",
            "evaluator_version": SCHEMA_VERSION, "may_feed_search": True,
            "subject": subject,
            "input_hash": _hash({"dataset": dataset, "best_k": result.get("best_k"),
                                  "metrics": {key: _prediction_scalar(value)
                                              for key, value in (result.get("metrics") or {}).items()
                                              if _prediction_scalar(value) is not None}}),
        }
        for item in audit.get("checks", [])[:8]:
            if not isinstance(item, Mapping) or item.get("status") not in {"warning", "fail"}:
                continue
            check_id = str(item.get("id", "cluster_check"))[:80]
            evidence = {"status": str(item.get("status")),
                        "check_id": check_id}
            checks.append({"subject": subject, "check_id": check_id,
                           "status": evidence["status"], "evidence": evidence})
            if check_id == "cluster_separation":
                code, category, text, actions, alternatives = (
                    "clustering_separation_weak", "structure",
                    "聚类簇间分离度不足，簇标签不能直接当成自然类别。",
                    ["compare_mechanisms"], ["连续梯度被离散化", "特征尺度或距离度量不合适", "样本异质性"])
            elif check_id == "cluster_seed_stability":
                code, category, text, actions, alternatives = (
                    "clustering_seed_instability", "structure",
                    "聚类结果对随机种子敏感，当前分组结构不稳定。",
                    ["compare_mechanisms", "inspect_parameterization"], ["局部最优", "簇数不合适", "特征噪声"])
            elif check_id == "cluster_size_balance":
                code, category, text, actions, alternatives = (
                    "clustering_tiny_cluster", "data",
                    "存在极小簇，需区分稳定群体与异常点集合。",
                    ["check_observations", "compare_mechanisms"], ["异常点", "真实小群体", "缺失值插补影响"])
            else:
                code, category, text, actions, alternatives = (
                    "clustering_credibility_warning", "validation",
                    "聚类可信度审计出现警告，不能把分组结果升级为事实。",
                    ["compare_mechanisms"], ["距离假设不合适", "数据分布不稳定"])
            add(_record(code, category, text, context, evidence, actions, alternatives, state="suspected"))
    return checks, checked


def _structure_signal_records(
    structure_results: Sequence[Mapping[str, Any]] | None,
    add,
) -> tuple[list[dict], int]:
    """Admit bounded temporal structure signals as reversible search hints."""
    checks: list[dict] = []
    checked = 0
    routes = {
        "periodic_candidate": ("periodic_structure_signal", "structure",
                               "序列存在周期候选信号，需与季节性、采样间隔和噪声解释竞争。",
                               ["compare_mechanisms"]),
        "change_point_candidate": ("change_point_signal", "structure",
                                   "序列存在变点候选信号，不能直接视为机制切换。",
                                   ["compare_mechanisms", "check_observations"]),
        "monotone_candidate": ("monotone_structure_signal", "structure",
                                "序列存在单调性候选信号，需检查边界、截断和时间窗。",
                                ["compare_mechanisms"]),
    }
    for result in list(structure_results or [])[:16]:
        if not isinstance(result, Mapping):
            continue
        payload = result.get("temporal_structure")
        if not isinstance(payload, Mapping) or payload.get("schema_version") != "mathmodel.structure-diagnostics/v1":
            continue
        diagnostics = payload.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            continue
        checked += 1
        dataset = str(result.get("dataset", "unknown"))[:160]
        for subject, record in list(diagnostics.items())[:32]:
            if not isinstance(record, Mapping) or record.get("status") != "signal":
                continue
            subject_text = f"{dataset}.{str(subject)[:120]}"
            context = {"phase": "development", "partition": "ordered_series_screening_only",
                       "evaluator_version": SCHEMA_VERSION, "may_feed_search": True,
                       "subject": subject_text,
                       "input_hash": _hash({"dataset": dataset, "subject": str(subject)[:120],
                                             "signals": sorted(str(item) for item in record.get("signals", []))})}
            for signal in sorted(set(str(item) for item in record.get("signals", []))):
                route = routes.get(signal)
                if route is None:
                    continue
                code, category, summary, actions = route
                evidence = {"signal": signal,
                            "periodicity_score": _finite((record.get("periodicity") or {}).get("autocorrelation")),
                            "monotone_score": _finite((record.get("monotone") or {}).get("score")),
                            "change_point_score": _finite((record.get("change_point") or {}).get("score"))}
                checks.append({"subject": subject_text, "signal": signal, "evidence": evidence})
                add(_record(code, category, summary, context, evidence, actions,
                            ["观测噪声", "时间对齐错误", "分组或采样机制变化"], state="suspected"))
    return checks, checked


def _specialized_audit_records(
    specialized_results: Mapping[str, Any] | None,
    add,
) -> tuple[list[dict], int]:
    """Expose non-tabular credibility warnings through the same read-only panel."""
    checks: list[dict] = []
    checked = 0
    route_by_kind = {
        "optimization": ("optimization_audit_warning", "optimization",
                         "优化可信度审计出现警告，当前方案不能仅凭求解器成功宣称现实最优。",
                         ["replay_numerics", "compare_mechanisms"]),
        "causal_effect": ("causal_audit_warning", "semantic",
                           "因果效应审计未完全通过，相关性不能替代识别假设。",
                           ["confirm_semantics", "compare_mechanisms"]),
        "ranking_result": ("ranking_audit_warning", "validation",
                            "综合评价稳定性审计出现警告，排名不能被解释为唯一客观次序。",
                            ["compare_mechanisms", "replay_numerics"]),
        "simulation": ("simulation_audit_warning", "numerical",
                        "仿真/不确定性审计出现警告，区间和情景仍需独立复核。",
                        ["replay_numerics", "compare_mechanisms"]),
    }
    for kind, payload in dict(specialized_results or {}).items():
        route = route_by_kind.get(kind)
        if route is None or not isinstance(payload, Mapping):
            continue
        audit = payload.get("credibility_audit")
        if not isinstance(audit, Mapping):
            continue
        checked += 1
        status = str(audit.get("status", "not_assessed"))
        if status not in {"warning", "fail", "not_assessed"}:
            continue
        subject = str(payload.get("target") or payload.get("dataset") or kind)[:180]
        context = {"phase": "execution", "partition": "development_or_solver_audit",
                   "evaluator_version": SCHEMA_VERSION, "may_feed_search": True,
                   "subject": subject,
                   "input_hash": _hash({"kind": kind, "subject": subject, "status": status})}
        code, category, summary, actions = route
        evidence = {"kind": kind, "audit_status": status,
                    "label": str(audit.get("label", ""))[:120]}
        checks.append({"kind": kind, "subject": subject, "status": status})
        add(_record(code, category, summary, context, evidence, actions,
                    ["数据质量", "题意契约不完整", "数值误差或模型结构不足"], state="suspected"))
    return checks, checked


def _rms(values: np.ndarray) -> float:
    amplitude = float(np.max(np.abs(values), initial=0))
    return float(amplitude * np.sqrt(np.mean((values / amplitude) ** 2))) if amplitude else 0.0


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 3:
        return None
    left, right = left / max(_rms(left), 1e-100), right / max(_rms(right), 1e-100)
    if np.std(left) <= 1e-12 or np.std(right) <= 1e-12:
        return None
    return _finite(float(np.corrcoef(left, right)[0, 1]))


def _rank(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def _spectrum(matrix: np.ndarray) -> dict:
    scales = np.max(np.abs(matrix), axis=0)
    normalized = matrix / np.where(scales > 0, scales, 1)
    singular = np.linalg.svd(normalized, compute_uv=False)
    tolerance = float(max(matrix.shape) * np.finfo(float).eps * singular[0])
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if rank == matrix.shape[1] else None
    return {"rows": len(matrix), "columns": matrix.shape[1], "rank": rank,
            "condition_scaled": condition, "rank_tolerance": tolerance,
            "scaling": "column_max_absolute_value"}


def diagnose_development(
    *, design_train: np.ndarray, response_train: np.ndarray,
    design_search: np.ndarray, response_search: np.ndarray, coefficients: np.ndarray,
    search_starts: np.ndarray, window_points: int, training_fingerprint: str,
    term_names: list[str], state_names: list[str], seed: int = 42,
    support_jaccard: float | None = None,
) -> dict:
    """Generic integral/regression design diagnostics with no final-test input."""
    train, search = _matrix(design_train), _matrix(design_search)
    y_train, y_search = _matrix(response_train, columns=8), _matrix(response_search, columns=8)
    beta = _matrix(coefficients)
    if (train.shape[1] != search.shape[1] or beta.shape != (y_train.shape[1], train.shape[1])
            or y_train.shape[0] != len(train) or y_search.shape != (len(search), y_train.shape[1])):
        raise ValueError("diagnostic_arrays_are_not_aligned")
    if (type(window_points) is not int or not 1 <= window_points <= 5000
            or not isinstance(search_starts, np.ndarray) or search_starts.ndim != 1
            or search_starts.dtype.kind not in "iu" or len(search_starts) != len(search)
            or np.any(np.diff(search_starts.astype(float)) <= 0) or np.any(search_starts < 0)):
        raise ValueError("diagnostic_windows_are_not_ordered")
    if (len(term_names) != train.shape[1] or len(state_names) != y_train.shape[1]
            or any(type(name) is not str or len(name) > 1000 for name in [*term_names, *state_names])
            or type(seed) is not int or seed < 0 or type(training_fingerprint) is not str
            or len(training_fingerprint) != 64):
        raise ValueError("diagnostic_context_is_incomplete")
    if support_jaccard is not None and (_finite(support_jaccard) is None or not 0 <= support_jaccard <= 1):
        raise ValueError("invalid_support_stability")
    digest = sha256()
    for array in (train, y_train, search, y_search, beta, search_starts):
        digest.update(str(array.shape).encode("ascii"))
        digest.update(np.ascontiguousarray(array, dtype="<f8").tobytes())
    digest.update(_hash({"training": training_fingerprint, "terms": term_names, "states": state_names,
                         "window": window_points, "seed": seed, "support": support_jaccard,
                         "policy": _POLICY, "version": DEVELOPMENT_VERSION}).encode("ascii"))
    context = {"phase": "development", "partition": "train_and_search_only",
               "input_hash": digest.hexdigest(), "evaluator_version": DEVELOPMENT_VERSION,
               "seed": seed, "may_feed_search": True}
    records, checks = [], {}

    def add(code, category, summary, evidence, actions, alternatives=(), state="observed", subject=None):
        records.append(_record(code, category, summary, {**context, "subject": subject}, evidence,
                               actions, alternatives, state=state))

    full = _spectrum(train)
    checks["candidate_library"] = full
    if full["rank"] < full["columns"]:
        add("redundant_candidate_basis", "parameter", "候选库含线性依赖，不能把每个候选项系数都解释为独立机制。",
            full, ["inspect_parameterization"], ["代数恒等关系", "观测激励不足", "候选基底重复"])
    active_checks = []
    for index, name in enumerate(state_names):
        active = np.flatnonzero(np.abs(beta[index]) > 1e-8)
        active_check = _spectrum(train[:, active]) if len(active) else {"rank": 0, "columns": 0, "condition_scaled": None}
        active_checks.append({"state": name, **active_check})
        if active_check["rank"] < len(active) or (active_check["condition_scaled"] or 0) > _POLICY["condition_warning"]:
            add("active_parameter_instability", "parameter", "活跃项设计矩阵秩亏或病态，系数解释需要额外辨识检验。",
                active_check, ["inspect_parameterization"], ["基底共线性", "有限轨迹未充分激励各项"], subject=name)
    checks["active_designs"] = active_checks
    checks["training_support_jaccard"] = support_jaccard
    if support_jaccard is not None and support_jaccard < _POLICY["support_jaccard_warning"]:
        add("unstable_training_support", "parameter", "训练子段选择的方程项差异较大，结构解释不稳定。",
            {"support_jaccard": support_jaccard, "threshold": _POLICY["support_jaccard_warning"]},
            ["inspect_parameterization", "compare_mechanisms"], ["参数弱辨识", "子段覆盖不同状态", "噪声或机制变化"])
    # Greedy disjoint selection does not fabricate neighboring residuals over gaps.
    selected, previous_end = [], -1
    for index, start in enumerate(search_starts):
        if int(start) > previous_end:
            selected.append(index)
            previous_end = int(start) + window_points
    selected = np.asarray(selected, dtype=int)
    adjacent = np.diff(search_starts[selected]) == window_points + 1
    with np.errstate(over="ignore", invalid="ignore"):
        train_prediction, search_prediction = train @ beta.T, search @ beta.T
        train_residual, search_residual = y_train - train_prediction, y_search - search_prediction
    if not np.isfinite(train_residual).all() or not np.isfinite(search_residual).all():
        add("development_nonfinite_prediction", "numerical", "开发段预测或残差出现非有限值。", {},
            ["replay_numerics"])
    else:
        residual_checks = []
        for index, name in enumerate(state_names):
            residual = search_residual[selected, index]
            predicted = search_prediction[selected, index]
            error, baseline = _rms(search_residual[:, index]), _rms(y_search[:, index])
            train_error = _rms(train_residual[:, index])
            ratio = _finite(error / baseline) if baseline > 1e-12 else None
            lag = _correlation(residual[:-1][adjacent], residual[1:][adjacent]) if adjacent.sum() >= 12 else None
            scale_corr = _correlation(_rank(np.abs(predicted)), _rank(np.abs(residual))) if len(residual) >= 24 else None
            bias = _finite(float(np.mean(residual)) / baseline) if baseline > 1e-12 and len(residual) >= 12 else None
            metrics = {"state": name, "search_rmse": error, "train_rmse": train_error,
                       "zero_increment_baseline_rmse": baseline, "rmse_ratio": ratio,
                       "disjoint_windows": len(residual), "adjacent_pairs": int(adjacent.sum()),
                       "lag1_correlation": lag, "absolute_error_scale_rank_correlation": scale_corr,
                       "signed_bias_to_baseline": bias,
                       "pattern_status": "screened" if len(residual) >= 12 else "insufficient_disjoint_windows"}
            residual_checks.append(metrics)
            if baseline > 1e-12 and error >= baseline:
                add("search_baseline_not_improved", "structure", "选参段未优于零增量基线，尚不能确定是机制还是观测问题。",
                    metrics, ["check_observations", "compare_mechanisms"],
                    ["数据或单位对齐错误", "观测噪声", "参数估计不稳", "结构不充分"], state="suspected", subject=name)
            if bias is not None and abs(bias) >= _POLICY["bias_to_baseline_warning"]:
                add("search_residual_bias", "observation", "非重叠选参窗口存在明显平均偏差，偏差来源仍未确定。",
                    metrics, ["check_observations", "compare_noise_models", "compare_mechanisms"],
                    ["测量偏置", "遗漏驱动或机制", "数值近似误差"], state="suspected", subject=name)
            if lag is not None and abs(lag) >= _POLICY["lag_correlation_warning"]:
                add("search_residual_memory", "structure", "相邻非重叠选参窗口的残差有时间相关性，不据此判定隐变量。",
                    metrics, ["replay_numerics", "compare_noise_models", "compare_memory_models"],
                    ["相关观测噪声", "时间对齐或积分近似", "滞后、外部驱动或未观测状态"], state="suspected", subject=name)
            if scale_corr is not None and abs(scale_corr) >= _POLICY["scale_correlation_warning"]:
                add("search_error_scale_pattern", "observation", "残差幅度与预测幅度相关，异方差仅是候选解释之一。",
                    metrics, ["compare_noise_models", "compare_mechanisms"],
                    ["异方差观测", "非线性机制不足", "不同状态区域的偏差"], state="suspected", subject=name)
        checks["residuals"] = residual_checks
        if len(search_residual) >= 16:
            # This is a development-window screening signal only.  It keeps
            # residual memory from being promoted directly to a latent state
            # by exposing competing observation/input/nonstationarity routes.
            checks["unclosed_state_competition"] = compete_unclosed_state_explanations(
                search_residual, y_search, state_names=state_names)
        else:
            checks["unclosed_state_competition"] = {
                "status": "not_assessed", "reason": "development_window_too_short",
                "policy": {"latent_state_not_proven": True},
            }
    return {"schema_version": DEVELOPMENT_VERSION, "context": context, "policy": copy.deepcopy(_POLICY),
            "records": records, "checks": checks, "actions": _routes(records), "may_execute": False}


_FAILURES = {
    "timeout": ("resource", "求解超时，未获得结构错误的证据。", "inspect_resources"),
    "queue_timeout": ("resource", "等待资源超时，模型尚未得到本次评价。", "inspect_resources"),
    "memory_limit": ("resource", "内存申请达到限制。", "inspect_resources"),
    "output_limit": ("resource", "输出体积超过限制。", "inspect_resources"),
    "evaluation_limit": ("resource", "评估预算耗尽；刚性、尺度或搜索困难仍需区分。", "replay_numerics"),
    "isolation_unavailable": ("resource", "必需的资源限制未能建立。", "inspect_resources"),
    "cancelled": ("resource", "执行已取消，不解释为模型错误。", None),
    "numeric_domain": ("numerical", "运行出现定义域、除零或溢出错误。", "replay_numerics"),
    "nonfinite_result": ("numerical", "数值结果包含非有限值。", "replay_numerics"),
    "invalid_contract": ("semantic", "执行契约未通过复核。", "confirm_semantics"),
    "upstream_failed": ("dependency", "缺少上游可用结果，下游未执行。", "resolve_upstream"),
}


def build_model_diagnostics(*, mechanistic: dict | None = None, proposals: dict | None = None,
                            dynamics: dict | None = None,
                            prediction_results: Sequence[Mapping[str, Any]] | None = None,
                            model_results: Sequence[Mapping[str, Any]] | None = None,
                            structure_results: Sequence[Mapping[str, Any]] | None = None,
                            specialized_results: Mapping[str, Any] | None = None) -> dict:
    """Summarize trusted pipeline outputs, without changing them or executing repairs."""
    records, dropped = [], 0

    def add(record):
        nonlocal dropped
        if len(records) < MAX_DIAGNOSTICS:
            records.append(record)
        else:
            dropped += 1

    development = (dynamics or {}).get("development_diagnostics", {})
    prediction_checks: list[dict] = []
    prediction_checked = prediction_available = 0
    clustering_checks: list[dict] = []
    clustering_checked = 0
    structure_checks: list[dict] = []
    structure_checked = 0
    specialized_checks: list[dict] = []
    specialized_checked = 0
    if development.get("schema_version") == DEVELOPMENT_VERSION:
        for record in development.get("records", []):
            add(copy.deepcopy(record))
        if development.get("status") == "failed_safe":
            add(_record("development_diagnostics_unavailable", "execution", "开发段诊断未完成，已有求解结果未被修改。",
                        {"phase": "development", "may_feed_search": False}, {}, state="pending"))
    mechanism = mechanistic or {}
    # Provenance fingerprints, not reusable proof certificates. Allowing NaN in
    # this private hash also permits diagnosis of already rejected contracts.
    contract_hashes = {
        str(relation.get("id")): sha256(json.dumps(relation, ensure_ascii=True, sort_keys=True,
                                                   default=str).encode("ascii")).hexdigest()
        for relation in mechanism.get("mathematical_ir", {}).get("relations", [])[:100]
    }
    context = {"phase": "execution", "partition": "no_observational_holdout",
               "evaluator_version": SCHEMA_VERSION, "may_feed_search": True,
               "problem_hash": mechanism.get("problem_fingerprint")}
    for failure in mechanism.get("solver_execution", {}).get("failures", [])[:100]:
        code = failure.get("failure_code", "unclassified_failure")
        category, summary, action = _FAILURES.get(code, (
            "execution", "执行失败尚未完成归因，不能从异常退出推断数学原因。", "replay_numerics"))
        evidence = {"failure_code": code if code in _FAILURES else "unclassified_failure"}
        add(_record("execution_" + evidence["failure_code"], category, summary,
                    {**context, "subject": str(failure.get("relation_id", "unknown"))[:200],
                     "contract_hash": contract_hashes.get(str(failure.get("relation_id"))),
                     "input_hash": _hash({"failure_code": evidence["failure_code"],
                                          "limits": failure.get("execution_supervision", {}).get("limits", {})})},
                    evidence, [action] if action else []))
    for numerical in mechanism.get("numerical_results", [])[:100]:
        if numerical.get("convergence", {}).get("status") == "fail":
            add(_record("numerical_confirmation_failed", "numerical", "数值复核未通过，应先核查数值方案。",
                        {**context, "subject": str(numerical.get("relation_id", "unknown"))[:200],
                         "contract_hash": contract_hashes.get(str(numerical.get("relation_id")))},
                        {"convergence_status": "fail"}, ["replay_numerics"]))
    for relation in mechanism.get("mathematical_ir", {}).get("relations", [])[:100]:
        errors = relation.get("validation_errors", [])
        if errors and relation.get("parse_status") not in {"machine_verified", "machine_compiled"}:
            unit_error = any("unit" in str(error) or "dimension" in str(error) for error in errors)
            add(_record("unit_binding_unresolved" if unit_error else "contract_validation_failed", "semantic",
                        "单位或量纲未通过核验，不能靠修改验收门继续求解。" if unit_error else "契约条件不完整或不合法，需要先核验绑定。",
                        {**context, "phase": "preflight", "subject": str(relation.get("id", "unknown"))[:200],
                         "contract_hash": contract_hashes.get(str(relation.get("id")))},
                        {"validation_error_count": len(errors)}, ["confirm_semantics"]))
    for hypothesis in (proposals or {}).get("hypotheses", [])[:8]:
        unknown = hypothesis.get("unknown_mechanisms", [])
        obligations = hypothesis.get("obligations", [])
        if unknown or obligations:
            proposal_context = {"phase": "preflight", "partition": "no_observational_holdout",
                                "evaluator_version": SCHEMA_VERSION, "may_feed_search": True,
                                "contract_hash": hypothesis.get("contract_hash"),
                                "hypothesis_hash": hypothesis.get("hypothesis_hash"), "subject": hypothesis.get("id")}
            add(_record("hypothesis_requires_binding", "semantic", "候选假设尚有机制、角色或验证义务未完成。",
                        proposal_context, {"unknown_mechanisms": len(unknown), "obligations": len(obligations)},
                        ["confirm_semantics", "compare_mechanisms"] if unknown else ["confirm_semantics"], state="pending"))
    for rejection in (proposals or {}).get("rejected_proposals", [])[:8]:
        if rejection.get("code") in {
            "shape_mismatch", "output_shape_mismatch", "output_dtype_mismatch", "dimension_mismatch",
            "output_dimension_mismatch", "expression_dependency_cycle", "unknown_input_node",
            "operator_arity_mismatch", "transcendental_requires_dimensionless",
        }:
            add(_record("hypothesis_type_rejected", "structure", "候选图在编译期被拒绝，应修改候选而不是放松类型检查。",
                        {"phase": "preflight", "may_feed_search": True, "partition": "no_observational_holdout",
                         "contract_hash": (proposals or {}).get("contract_hash"),
                         "subject": rejection.get("index"), "node_id": rejection.get("node_id"),
                         "evaluator_version": SCHEMA_VERSION},
                        {"rejection_code": rejection["code"]}, ["repair_typed_graph"]))
    # Keep prediction findings in the same bounded channel, after primary
    # mechanistic/preflight findings have been admitted.  This prevents a
    # large batch of targets from crowding out upstream contract failures.
    prediction_checks, prediction_checked, prediction_available = _prediction_diagnostic_records(
        prediction_results, add
    )
    clustering_checks, clustering_checked = _clustering_diagnostic_records(model_results, add)
    structure_checks, structure_checked = _structure_signal_records(structure_results, add)
    specialized_checks, specialized_checked = _specialized_audit_records(specialized_results, add)
    # Freeze the search-facing channel BEFORE adding any final-test findings.
    eligible = [copy.deepcopy(item) for item in records if item["context"].get("may_feed_search")
                and item["context"].get("phase") in {"development", "preflight", "execution"}]
    search_feedback = {"schema_version": SCHEMA_VERSION, "records": eligible, "actions": _routes(eligible),
                       "locked_test_included": False, "may_execute": False}
    trajectory = (dynamics or {}).get("trajectory_test", {})
    integral = (dynamics or {}).get("test_integral_metrics", {})
    if dynamics and dynamics.get("evaluation_protocol") == "train_search_locked_test":
        final_checks = {item.get("id"): item.get("status") for item in dynamics.get("credibility_audit", {}).get("checks", [])
                        if item.get("id") in {"dynamics_integral_test", "dynamics_autonomous_test", "dynamics_residual_memory"}}
        final_context = {"phase": "final_test", "partition": "locked_test", "may_feed_search": False,
                         "evaluator_version": SCHEMA_VERSION, "subject": "equation_discovery",
                         "evaluation_hash": _hash({"status": trajectory.get("status"), "checks": final_checks,
                                                   "reason": trajectory.get("reason"),
                                                   "integral_rmse": _finite(integral.get("rmse"))})}
        reason = trajectory.get("reason")
        if reason in {"initial_state_not_observed", "insufficient_observed_test_targets"}:
            category, summary = "data", "最终轨迹缺少观测初值或足够测试标签，不能补造后宣称验证通过。"
        elif trajectory.get("solver_success") is False and trajectory.get("status") == "fail":
            category = "resource" if reason in {"rollout_deadline_exceeded", "evaluation_budget_exhausted"} else "numerical"
            summary = "最终轨迹求解未完成；运行失败不等于发现了机制反例。"
        else:
            category, summary = "validation", "锁定最终段未通过或未充分完成检验，原因仍待区分。"
        if (trajectory.get("status") != "pass" or (integral.get("invalid_predictions") or 0) > 0
                or any(status != "pass" for status in final_checks.values())):
            add(_record("locked_test_not_confirmed", category, summary, final_context,
                        {"trajectory_status": trajectory.get("status", "not_assessed"),
                         "solver_success": trajectory.get("solver_success"),
                         "audit_statuses": final_checks,
                         "integral_rmse": _finite(integral.get("rmse"))}, ["new_confirmation"]))
    return {
        "schema_version": SCHEMA_VERSION, "records": records, "search_feedback": search_feedback,
        "proposed_actions": _routes(records), "dropped_diagnostics": dropped,
        "development_checks": copy.deepcopy(development.get("checks", {})),
        "prediction_checks": prediction_checks,
        "clustering_checks": clustering_checks,
        "structure_checks": structure_checks,
        "specialized_checks": specialized_checks,
        "status": "attention_required" if records else "no_findings_in_checked_scope",
        "coverage": {"development_design_available": bool(development),
                     "mechanistic_execution_available": bool(mechanism.get("solver_execution")),
                     "hypothesis_proposals_available": bool(proposals),
                     "prediction_models_checked": prediction_checked,
                     "prediction_models_with_diagnostics": prediction_available,
                     "clustering_models_checked": clustering_checked,
                     "temporal_structure_sets_checked": structure_checked,
                     "specialized_audits_checked": specialized_checked,
                     "all_model_families_covered": False},
        "policy": {"may_execute_repairs": False, "may_modify_facts": False,
                   "may_change_acceptance_thresholds": False, "locked_test_feedback_allowed": False,
                   "latent_state_discovery_proven": False, "diagnostics_are_proof": False,
                   "raw_observations_in_feedback": False},
    }
