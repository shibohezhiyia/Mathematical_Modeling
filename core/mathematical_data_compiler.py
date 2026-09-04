"""题目驱动的数学数据编译器与多视图反证审计。

普通 ETL 只回答“如何变换表”。本模块进一步回答：一行代表什么、目标
量在哪个粒度上定义、哪些量允许相加，以及结论是否会因合理的数据表述
变化而翻转。它不把相关性升级为因果结论，也不会自动覆盖估计对象。
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from difflib import SequenceMatcher
from itertools import combinations
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .mathematical_reasoning import extract_column_unit
from .table_transformer import (
    TableTransformError,
    TableTransformationEngine,
    _deduplicate_labels,
)


_TIME_TOKENS = ("date", "time", "day", "month", "year", "日期", "时间", "年月", "月份", "季度")
_DIMENSION_TOKENS = (
    "entity", "region", "site", "station", "customer", "product", "category",
    "实体", "地区", "区域", "站点", "客户", "产品", "品类", "类别", "组别", "方案",
)
_ADDITIVE_TOKENS = (
    "amount", "total", "count", "quantity", "sales", "demand", "flow", "revenue",
    "volume", "数量", "销量", "需求", "流量", "金额", "总量", "收入", "产量", "人数", "次数",
)
_NON_ADDITIVE_TOKENS = (
    "rate", "ratio", "price", "unit", "average", "mean", "percent", "temperature",
    "score", "index", "率", "比例", "比重", "单价", "均价", "平均", "温度", "浓度", "指数", "评分",
)
_SEMANTIC_ROLES = {"time", "technical_id", "dimension", "measure", "attribute"}
_ADDITIVITY_VALUES = {"additive", "non_additive", "unknown", "not_applicable"}


def _normalized(text: Any) -> str:
    return re.sub(r"[\s_\-（）()\[\]]+", "", str(text).strip().lower())


def _looks_identifier(text: Any) -> bool:
    raw = str(text).strip().lower()
    if any(token in raw for token in ("编号", "编码", "序号", "索引")):
        return True
    parts = [part for part in re.split(r"[^a-z0-9]+", raw) if part]
    return any(part in {"id", "code", "identifier", "uuid", "key"} for part in parts)


def _json_scalar(value: Any) -> Any:
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    return value


def _bh_adjust(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    valid = [(index, float(value)) for index, value in enumerate(p_values) if value is not None and np.isfinite(value)]
    output: List[Optional[float]] = [None] * len(p_values)
    if not valid:
        return output
    ordered = sorted(valid, key=lambda item: item[1])
    running = 1.0
    total = len(ordered)
    for rank in range(total, 0, -1):
        index, value = ordered[rank - 1]
        running = min(running, value * total / rank)
        output[index] = min(1.0, running)
    return output


@dataclass
class CompiledDataView:
    view_id: str
    name: str
    purpose: str
    estimand: str
    pipeline: List[Dict[str, Any]]
    output_grain: List[str]
    row_relation: str


class MathematicalDataCompiler:
    """编译多种建模视图，并用不变量和结论压力测试筛除不可信视图。"""

    def __init__(self, *, max_analysis_rows: int = 50_000, random_state: int = 42) -> None:
        self.max_analysis_rows = max(1_000, int(max_analysis_rows))
        self.random_state = int(random_state)
        self.engine = TableTransformationEngine()

    def compile(
        self,
        frame: pd.DataFrame,
        *,
        problem: str = "",
        target: Optional[Any] = None,
        max_views: int = 8,
        semantic_hints: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] == 0:
            raise TableTransformError("数学数据编译需要至少一个字段的二维表")
        started_at = time.perf_counter()
        working = frame.copy(deep=False)
        working.columns = _deduplicate_labels(working.columns)
        audited_frame = self._coverage_sample(working, self.max_analysis_rows)
        sampled_execution = len(audited_frame) < len(working)
        contract_started = time.perf_counter()
        hints = self._validate_semantic_hints(semantic_hints)
        effective_target = hints.get("target", target)
        contract = self._build_contract(working, str(problem or ""), effective_target)
        if hints:
            contract = self._apply_semantic_hints(working, contract, hints)
        contract_elapsed = time.perf_counter() - contract_started
        contract["audit_scope"] = {
            "mode": "coverage_sample" if sampled_execution else "full_dataset",
            "source_rows": int(len(working)),
            "audited_rows": int(len(audited_frame)),
            "full_execution_reaudit_required": sampled_execution,
        }
        views_started = time.perf_counter()
        candidates = self._generate_views(
            audited_frame, contract, str(problem or "")
        )[:max(1, min(int(max_views), 12))]
        views = [
            self._execute_and_audit_view(audited_frame, contract, candidate)
            for candidate in candidates
        ]
        for view in views:
            view["audit_scope"] = contract["audit_scope"]
            view["source_shape"] = list(working.shape)
        views_elapsed = time.perf_counter() - views_started
        stress_started = time.perf_counter()
        stress = self._stress_conclusions(audited_frame, contract)
        stress_elapsed = time.perf_counter() - stress_started
        reversals = [item for item in stress.get("relationships", []) if item.get("status") == "contradicted"]
        supported_relationships = [
            item for item in stress.get("relationships", [])
            if item.get("status") in {"stable_empirical", "restricted"}
        ]
        inconclusive_relationships = [
            item for item in stress.get("relationships", [])
            if item.get("status") == "inconclusive"
        ]
        blocked_views = [item for item in views if not item.get("admissible")]
        findings: List[Dict[str, Any]] = []
        if reversals:
            findings.append({
                "level": "contradicted",
                "message": f"发现{len(reversals)}个关系在合理数据视图下发生方向翻转；相关结论不得写成稳定规律。",
                "action": "按分组内、分组间和时间粒度分别建模，并解释差异来源。",
            })
        if blocked_views:
            findings.append({
                "level": "blocked",
                "message": f"{len(blocked_views)}个候选视图违反粒度、守恒或有限性检查，已禁止进入求解。",
                "action": "核对目标粒度、可加性和主键后重新编译。",
            })
        if sampled_execution:
            findings.append({
                "level": "restricted",
                "message": (
                    f"为控制内存与运行时间，候选视图在覆盖样本"
                    f"{len(audited_frame):,}/{len(working):,}行上审计。"
                ),
                "action": "应用候选流水线时必须在完整数据上重新执行守恒、键唯一性和有限值检查。",
            })
        if not reversals and supported_relationships:
            findings.append({
                "level": "empirical_support",
                "message": "已检验关系未出现强方向翻转，但这仍是经验稳定性，不是因果证明。",
                "action": "继续保留外部有效性、遗漏变量和测量误差假设。",
            })
        elif not reversals and inconclusive_relationships:
            findings.append({
                "level": "not_supported",
                "message": (
                    f"{len(inconclusive_relationships)}个候选关系未同时通过效应量、"
                    "95%方向区间与FDR门；系统不生成肯定关系结论。"
                ),
                "action": "增加有效样本、改进测量或提出有理论依据的分层假设后再检验。",
            })
        unresolved = list(contract.get("unresolved", []))
        if unresolved:
            findings.append({
                "level": "needs_input",
                "message": "仍有未绑定的数据语义：" + "；".join(unresolved),
                "action": "在应用改变粒度的视图前确认这些条件。",
            })
        status = (
            "contradicted" if reversals else
            "needs_input" if unresolved else
            "restricted" if blocked_views or sampled_execution else
            "assessed"
        )
        credibility_audit = self._build_credibility_audit(
            contract=contract,
            views=views,
            stress=stress,
            reversals=reversals,
            blocked_views=blocked_views,
            sampled_execution=sampled_execution,
        )
        return {
            "schema_version": "mathmodel.data-compilation/v2",
            "status": status,
            "contract": contract,
            "views": views,
            "conclusion_stress": stress,
            "credibility_audit": credibility_audit,
            "findings": findings,
            "summary": {
                "candidate_views": len(views),
                "admissible_views": len(views) - len(blocked_views),
                "blocked_views": len(blocked_views),
                "relationships_tested": len(stress.get("relationships", [])),
                "direction_reversals": len(reversals),
                "supported_or_restricted_relationships": len(supported_relationships),
                "inconclusive_relationships": len(inconclusive_relationships),
                "source_rows": int(len(working)),
                "audited_rows": int(len(audited_frame)),
                "sampled_execution": sampled_execution,
                "timing_ms": {
                    "contract": round(contract_elapsed * 1_000, 3),
                    "view_compilation_and_audit": round(views_elapsed * 1_000, 3),
                    "conclusion_stress": round(stress_elapsed * 1_000, 3),
                    "total": round((time.perf_counter() - started_at) * 1_000, 3),
                },
            },
        }

    def compile_many(
        self,
        datasets: Mapping[str, pd.DataFrame],
        *,
        problem: str = "",
        target: Optional[Any] = None,
        primary_dataset: Optional[str] = None,
        max_views: int = 8,
        max_cross_datasets: int = 12,
        semantic_hints: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compile one primary estimand plus bounded cross-dataset grain contracts.

        Cross-table compilation never materializes a join.  It estimates key
        cardinality and expansion from bounded coverage samples, then states
        which direction is admissible for feature enrichment and whether totals
        may be combined without prior aggregation.
        """
        multi_started = time.perf_counter()
        valid = {
            str(name): frame for name, frame in datasets.items()
            if isinstance(frame, pd.DataFrame) and frame.shape[1] > 0
        }
        if not valid:
            raise TableTransformError("多表数学编译需要至少一个非空字段表")
        all_hints = self._validate_multi_dataset_hints(semantic_hints)
        dataset_hints = all_hints.get("datasets", {})
        unknown_hint_datasets = set(dataset_hints) - set(valid)
        if unknown_hint_datasets:
            raise TableTransformError(
                "semantic_hints引用不存在数据集："
                + "、".join(sorted(unknown_hint_datasets))
            )
        target_dataset: Optional[str] = None
        target_column: Optional[Any] = target
        if isinstance(target, str) and "." in target:
            prefix, suffix = target.split(".", 1)
            if prefix in valid:
                target_dataset, target_column = prefix, suffix
        if primary_dataset in valid:
            target_dataset = str(primary_dataset)
        if target_dataset is None:
            hinted_targets = [
                name for name, hint in dataset_hints.items()
                if hint.get("target") is not None
            ]
            if len(hinted_targets) > 1:
                raise TableTransformError(
                    "多个数据集声明了target；请用primary_dataset或“表名.字段名”唯一绑定主估计对象"
                )
            target_dataset = (
                hinted_targets[0] if hinted_targets else
                max(
                    valid,
                    key=lambda name: len(
                        valid[name].select_dtypes(include=np.number).columns
                    ),
                )
            )
        primary = self.compile(
            valid[target_dataset],
            problem=problem,
            target=target_column,
            max_views=max_views,
            semantic_hints=dataset_hints.get(target_dataset),
        )
        primary["dataset"] = target_dataset

        ordered_names = [
            target_dataset,
            *(
                name for name in sorted(
                    (item for item in valid if item != target_dataset),
                    key=lambda item: (valid[item].shape[1], len(valid[item])),
                    reverse=True,
                )
            ),
        ]
        selected_names = ordered_names[:max(2, min(int(max_cross_datasets), 24))]
        supporting_count = max(len(selected_names) - 1, 1)
        supporting_profile_limit = max(
            1_000, self.max_analysis_rows // min(supporting_count, 8)
        )
        contracts: Dict[str, Dict[str, Any]] = {target_dataset: primary["contract"]}
        for name in selected_names:
            if name == target_dataset:
                contracts[name]["dataset_role"] = "primary_estimand"
                continue
            prepared = valid[name].copy(deep=False)
            prepared.columns = _deduplicate_labels(prepared.columns)
            contract = self._build_contract(
                prepared, str(problem or ""), None,
                profile_limit=supporting_profile_limit,
            )
            hint = dataset_hints.get(name)
            if hint:
                contract = self._apply_semantic_hints(
                    prepared, contract, self._validate_semantic_hints(hint)
                )
            contract["dataset_role"] = "supporting_evidence"
            contract["unresolved"] = [
                item for item in contract.get("unresolved", [])
                if "目标字段" not in item
            ]
            contract["estimand"] = "支持表：只提供已验证键上的协变量或约束，不单独声明目标估计对象。"
            contracts[name] = contract

        cross_started = time.perf_counter()
        cross_contracts = self._compile_cross_dataset_contracts(
            {name: valid[name] for name in selected_names}, contracts
        )
        cross_elapsed = time.perf_counter() - cross_started
        blocked_cross = [item for item in cross_contracts if item.get("status") == "blocked"]
        if blocked_cross:
            primary["findings"].append({
                "level": "blocked",
                "message": f"发现{len(blocked_cross)}个多表原始连接会造成多对多膨胀或时间错配。",
                "action": "按估计对象粒度先聚合，再以验证过的复合键或point-in-time规则连接。",
            })
        skipped_names = ordered_names[len(selected_names):]
        if skipped_names:
            primary["findings"].append({
                "level": "restricted",
                "message": f"数据表数量超过跨表编译预算，{len(skipped_names)}张表仅保留输入清单。",
                "action": "缩小候选表范围或提高跨表预算后重新编译。",
            })
        primary["dataset_contracts"] = contracts
        primary["cross_dataset_contracts"] = cross_contracts
        primary["unprofiled_datasets"] = skipped_names
        primary["summary"].update({
            "datasets_received": len(valid),
            "datasets_compiled": len(selected_names),
            "cross_dataset_contracts": len(cross_contracts),
            "blocked_cross_dataset_contracts": len(blocked_cross),
        })
        timing = primary["summary"].setdefault("timing_ms", {})
        timing["cross_dataset_contracts"] = round(cross_elapsed * 1_000, 3)
        timing["multi_table_total"] = round(
            (time.perf_counter() - multi_started) * 1_000, 3
        )
        return primary

    @staticmethod
    def _validate_semantic_hints(
        hints: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if hints is None:
            return {}
        if not isinstance(hints, Mapping):
            raise TableTransformError("semantic_hints必须是对象")
        allowed = {"target", "grain", "columns"}
        unknown = set(hints) - allowed
        if unknown:
            raise TableTransformError(
                "semantic_hints包含未知字段：" + "、".join(sorted(map(str, unknown)))
            )
        columns = hints.get("columns", {})
        if columns is not None and not isinstance(columns, Mapping):
            raise TableTransformError("semantic_hints.columns必须是字段到语义的对象")
        grain = hints.get("grain")
        if grain is not None and (
            not isinstance(grain, Sequence) or isinstance(grain, (str, bytes))
        ):
            raise TableTransformError("semantic_hints.grain必须是字段数组")
        if grain is not None and not grain:
            raise TableTransformError("semantic_hints.grain不能为空数组")
        if grain is not None and len({str(item) for item in grain}) != len(grain):
            raise TableTransformError("semantic_hints.grain不能包含重复字段")
        target = hints.get("target")
        if target is not None and not isinstance(target, str):
            raise TableTransformError("semantic_hints.target必须是字段名")
        return {
            **({"target": target} if target is not None else {}),
            **({"grain": [str(item) for item in grain]} if grain is not None else {}),
            "columns": dict(columns or {}),
        }

    @classmethod
    def _validate_multi_dataset_hints(
        cls,
        hints: Optional[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        if hints is None:
            return {"datasets": {}}
        if not isinstance(hints, Mapping):
            raise TableTransformError("多表semantic_hints必须是对象")
        datasets = hints.get("datasets", hints)
        if not isinstance(datasets, Mapping):
            raise TableTransformError("semantic_hints.datasets必须是数据集语义对象")
        return {
            "datasets": {
                str(name): cls._validate_semantic_hints(value)
                for name, value in datasets.items()
            }
        }

    def _apply_semantic_hints(
        self,
        frame: pd.DataFrame,
        contract: Dict[str, Any],
        hints: Mapping[str, Any],
    ) -> Dict[str, Any]:
        output = dict(contract)
        semantics = [dict(item) for item in contract.get("columns_semantics", [])]
        by_column = {str(item.get("column")): item for item in semantics}
        applied: List[Dict[str, Any]] = []
        for raw_column, raw_spec in (hints.get("columns") or {}).items():
            column = str(raw_column)
            if column not in frame.columns or column not in by_column:
                raise TableTransformError(f"semantic_hints引用不存在字段：{column}")
            if not isinstance(raw_spec, Mapping):
                raise TableTransformError(f"字段{column}的语义提示必须是对象")
            allowed = {"role", "unit", "additivity", "semantic_id"}
            unknown = set(raw_spec) - allowed
            if unknown:
                raise TableTransformError(
                    f"字段{column}的语义提示包含未知项："
                    + "、".join(sorted(map(str, unknown)))
                )
            item = by_column[column]
            role = raw_spec.get("role")
            if role is not None:
                role = str(role)
                if role not in _SEMANTIC_ROLES:
                    raise TableTransformError(f"字段{column}的role不受支持：{role}")
                if role == "measure" and not pd.api.types.is_numeric_dtype(frame[column]):
                    raise TableTransformError(f"非数值字段{column}不能声明为measure")
                item["role"] = role
            additivity = raw_spec.get("additivity")
            if additivity is not None:
                additivity = str(additivity)
                if additivity not in _ADDITIVITY_VALUES:
                    raise TableTransformError(
                        f"字段{column}的additivity不受支持：{additivity}"
                    )
                if additivity in {"additive", "non_additive"} and not pd.api.types.is_numeric_dtype(frame[column]):
                    raise TableTransformError(f"非数值字段{column}不能声明可加性")
                item["additivity"] = additivity
            if "unit" in raw_spec:
                unit = raw_spec.get("unit")
                if unit is not None and (not isinstance(unit, str) or len(unit) > 100):
                    raise TableTransformError(f"字段{column}的unit必须是短字符串或null")
                item["unit"] = unit
            if "semantic_id" in raw_spec:
                semantic_id = raw_spec.get("semantic_id")
                if not isinstance(semantic_id, str) or not semantic_id.strip() or len(semantic_id) > 100:
                    raise TableTransformError(f"字段{column}的semantic_id必须是非空短字符串")
                item["semantic_id"] = semantic_id.strip()
            item["semantic_source"] = "explicit_hint"
            applied.append({"column": column, **dict(raw_spec)})

        semantic_ids = [
            str(item["semantic_id"]).strip().lower()
            for item in semantics if item.get("semantic_id")
        ]
        if len(set(semantic_ids)) != len(semantic_ids):
            raise TableTransformError("同一数据集内semantic_id必须唯一")

        output["columns_semantics"] = semantics
        output["time_columns"] = [
            item["column"] for item in semantics if item.get("role") == "time"
        ]
        output["technical_ids"] = [
            item["column"] for item in semantics if item.get("role") == "technical_id"
        ]
        output["dimensions"] = [
            item["column"] for item in semantics if item.get("role") == "dimension"
        ]
        output["numeric_measures"] = [
            item["column"] for item in semantics
            if item.get("role") == "measure"
            and pd.api.types.is_numeric_dtype(frame[item["column"]])
        ]
        target = hints.get("target", output.get("target"))
        if target is not None:
            if target not in frame.columns:
                raise TableTransformError(f"semantic_hints.target不存在：{target}")
            output["target"] = target
        target_semantic = by_column.get(str(output.get("target")))
        output["target_additivity"] = (
            target_semantic.get("additivity") if target_semantic else None
        )
        if "grain" in hints:
            grain = list(hints.get("grain") or [])
            missing = [column for column in grain if column not in frame.columns]
            if missing:
                raise TableTransformError("semantic_hints.grain字段不存在：" + "、".join(missing))
            valid = self._coverage_sample(frame[grain].dropna(), self.max_analysis_rows) if grain else frame.iloc[0:0]
            uniqueness = (
                valid.drop_duplicates().shape[0] / max(len(valid), 1) if grain else 0.0
            )
            output["observed_grain"] = grain
            output["grain_uniqueness"] = round(float(uniqueness), 6)
            output["grain_status"] = (
                "verified_unique" if grain and uniqueness >= 0.999999 else
                "declared_non_unique" if grain else "not_identified"
            )
        unresolved = list(output.get("unresolved") or [])
        if output.get("target"):
            unresolved = [item for item in unresolved if "目标字段" not in item]
        if output.get("observed_grain"):
            unresolved = [item for item in unresolved if "实体×时间粒度" not in item]
        if target_semantic and target_semantic.get("additivity") != "unknown":
            unresolved = [item for item in unresolved if "能否跨实体或时间相加" not in item]
        output["unresolved"] = unresolved
        output["estimand"] = self._estimand_text(
            output.get("target"), output.get("observed_grain") or []
        )
        output["semantic_hints_applied"] = applied
        output["semantic_contract_source"] = "hybrid_explicit_and_heuristic"
        return output

    def _compile_cross_dataset_contracts(
        self,
        datasets: Mapping[str, pd.DataFrame],
        contracts: Mapping[str, Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        names = list(datasets)
        pair_limit = max(1_000, min(10_000, self.max_analysis_rows // max(len(names), 1)))
        sampled: Dict[str, pd.DataFrame] = {}
        for name, frame in datasets.items():
            prepared = frame.copy(deep=False)
            prepared.columns = _deduplicate_labels(prepared.columns)
            sampled[name] = self._coverage_sample(prepared, pair_limit)
        output: List[Dict[str, Any]] = []
        for left_name, right_name in combinations(names, 2):
            left = sampled[left_name]
            right = sampled[right_name]
            left_semantics = {
                (
                    "semantic:" + _normalized(item.get("semantic_id"))
                    if item.get("semantic_id") else
                    "name:" + _normalized(item.get("column"))
                ): item
                for item in contracts[left_name].get("columns_semantics", [])
                if item.get("role") in {"technical_id", "dimension", "time"}
            }
            right_semantics = {
                (
                    "semantic:" + _normalized(item.get("semantic_id"))
                    if item.get("semantic_id") else
                    "name:" + _normalized(item.get("column"))
                ): item
                for item in contracts[right_name].get("columns_semantics", [])
                if item.get("role") in {"technical_id", "dimension", "time"}
            }
            key_match_meta: Dict[str, Dict[str, Any]] = {
                token: {
                    "source": (
                        "explicit_semantic_id" if token.startswith("semantic:")
                        else "normalized_name"
                    ),
                    "confidence": 1.0,
                }
                for token in set(left_semantics) & set(right_semantics)
            }
            shared_tokens = []
            for token in left_semantics:
                if not token or token not in right_semantics:
                    continue
                left_role = str(left_semantics[token].get("role"))
                right_role = str(right_semantics[token].get("role"))
                if (left_role == "time") != (right_role == "time"):
                    continue
                shared_tokens.append(token)
                if len(shared_tokens) >= 8:
                    break
            matched_left_columns = {
                str(left_semantics[token].get("column")) for token in shared_tokens
            }
            matched_right_columns = {
                str(right_semantics[token].get("column")) for token in shared_tokens
            }
            left_unmatched = [
                item for item in contracts[left_name].get("columns_semantics", [])
                if item.get("role") in {"technical_id", "dimension", "time"}
                and str(item.get("column")) not in matched_left_columns
            ][:8]
            right_unmatched = [
                item for item in contracts[right_name].get("columns_semantics", [])
                if item.get("role") in {"technical_id", "dimension", "time"}
                and str(item.get("column")) not in matched_right_columns
            ][:8]
            alias_candidates: List[Dict[str, Any]] = []
            comparisons = 0
            for left_item in left_unmatched:
                for right_item in right_unmatched:
                    left_role = str(left_item.get("role"))
                    right_role = str(right_item.get("role"))
                    compatible = (
                        left_role == right_role
                        and left_role in {"technical_id", "time", "dimension"}
                    )
                    if not compatible:
                        continue
                    left_column = str(left_item.get("column"))
                    right_column = str(right_item.get("column"))
                    lexical = SequenceMatcher(
                        None, _normalized(left_column), _normalized(right_column)
                    ).ratio()
                    if left_role == "dimension" and lexical < 0.45:
                        continue
                    comparisons += 1
                    if comparisons > 16:
                        break
                    left_value, right_value = self._canonicalize_join_pair(
                        left[left_column], right[right_column],
                        is_time=left_role == "time",
                    )
                    candidate = self._evaluate_canonical_join_key(
                        pd.DataFrame({"key_0": left_value}),
                        pd.DataFrame({"key_0": right_value}),
                        source_left_rows=len(datasets[left_name]),
                        source_right_rows=len(datasets[right_name]),
                    )
                    if candidate is None or float(candidate.get("overlap_coverage", 0)) < 0.7:
                        continue
                    relation_bonus = (
                        1.0 if candidate.get("relationship") != "many_to_many" else 0.85
                    )
                    confidence = min(
                        0.99,
                        float(candidate.get("overlap_coverage", 0))
                        * relation_bonus * (0.9 + 0.1 * lexical),
                    )
                    alias_candidates.append({
                        "left": left_item,
                        "right": right_item,
                        "confidence": confidence,
                    })
                if comparisons > 16:
                    break
            used_left: set[str] = set()
            used_right: set[str] = set()
            for alias_index, alias in enumerate(
                sorted(alias_candidates, key=lambda item: item["confidence"], reverse=True), 1
            ):
                left_column = str(alias["left"].get("column"))
                right_column = str(alias["right"].get("column"))
                if left_column in used_left or right_column in used_right:
                    continue
                token = f"inferred:{alias_index}:{_normalized(left_column)}:{_normalized(right_column)}"
                left_semantics[token] = alias["left"]
                right_semantics[token] = alias["right"]
                key_match_meta[token] = {
                    "source": "value_overlap_inferred_alias",
                    "confidence": round(float(alias["confidence"]), 6),
                }
                shared_tokens.append(token)
                used_left.add(left_column)
                used_right.add(right_column)
                if len(shared_tokens) >= 8:
                    break
            role_priority = {"technical_id": 0, "time": 1, "dimension": 2}
            shared_tokens.sort(key=lambda token: (
                role_priority.get(str(left_semantics[token].get("role")), 3),
                -min(
                    float(left_semantics[token].get("unique_rate", 0)),
                    float(right_semantics[token].get("unique_rate", 0)),
                ),
            ))
            if not shared_tokens:
                output.append({
                    "left_dataset": left_name,
                    "right_dataset": right_name,
                    "status": "needs_key",
                    "key_pairs": [],
                    "evidence": "没有发现名称与角色同时相容的跨表键；禁止按列位置拼接。",
                    "safe_feature_enrichment_direction": None,
                    "combined_additive_analysis": "not_admissible",
                })
                continue
            left_canonical: Dict[str, pd.Series] = {}
            right_canonical: Dict[str, pd.Series] = {}
            for token in shared_tokens:
                left_column = str(left_semantics[token]["column"])
                right_column = str(right_semantics[token]["column"])
                left_value, right_value = self._canonicalize_join_pair(
                    left[left_column], right[right_column],
                    is_time=str(left_semantics[token].get("role")) == "time",
                )
                left_canonical[token] = left_value
                right_canonical[token] = right_value
            evaluated: List[Dict[str, Any]] = []

            def evaluate_candidates(key_candidates: Iterable[Sequence[str]]) -> None:
                for raw_tokens in key_candidates:
                    tokens = list(raw_tokens)
                    left_keys = [str(left_semantics[token]["column"]) for token in tokens]
                    right_keys = [str(right_semantics[token]["column"]) for token in tokens]
                    canonical = [f"key_{index}" for index in range(len(tokens))]
                    left_values = pd.DataFrame({
                        column: left_canonical[token]
                        for column, token in zip(canonical, tokens)
                    })
                    right_values = pd.DataFrame({
                        column: right_canonical[token]
                        for column, token in zip(canonical, tokens)
                    })
                    candidate = self._evaluate_canonical_join_key(
                        left_values, right_values,
                        source_left_rows=len(datasets[left_name]),
                        source_right_rows=len(datasets[right_name]),
                    )
                    if candidate is not None:
                        candidate["key_pairs"] = [
                            {
                                "left": left_key,
                                "right": right_key,
                                **(
                                    {
                                        "match_source": key_match_meta[token]["source"],
                                        "match_confidence": key_match_meta[token]["confidence"],
                                    }
                                    if key_match_meta.get(token, {}).get("source")
                                    != "normalized_name" else {}
                                ),
                            }
                            for token, left_key, right_key in zip(
                                tokens, left_keys, right_keys
                            )
                        ]
                        evaluated.append(candidate)

            evaluate_candidates(([token] for token in shared_tokens))
            strong_single = any(
                item.get("relationship") != "many_to_many"
                and float(item.get("overlap_coverage", 0)) >= 0.5
                for item in evaluated
            )
            if not strong_single:
                evaluate_candidates(combinations(shared_tokens[:5], 2))
            strong_pair = any(
                len(item.get("key_pairs", [])) <= 2
                and item.get("relationship") != "many_to_many"
                and float(item.get("overlap_coverage", 0)) >= 0.5
                for item in evaluated
            )
            if not strong_single and not strong_pair:
                evaluate_candidates(combinations(shared_tokens[:5], 3))
            if not evaluated:
                output.append({
                    "left_dataset": left_name,
                    "right_dataset": right_name,
                    "status": "needs_key",
                    "key_pairs": [],
                    "evidence": "候选同名键没有足够非空重叠值。",
                    "safe_feature_enrichment_direction": None,
                    "combined_additive_analysis": "not_admissible",
                })
                continue
            relation_factor = {
                "one_to_one": 1.0, "one_to_many": 0.97,
                "many_to_one": 0.97, "many_to_many": 0.75,
            }
            for candidate in evaluated:
                width = len(candidate.get("key_pairs", []))
                candidate["selection_score"] = round(float(
                    float(candidate.get("overlap_coverage", 0))
                    * relation_factor.get(str(candidate.get("relationship")), 0.5)
                    - 0.01 * max(width - 1, 0)
                    - 0.005 * math.log1p(float(candidate.get("estimated_expansion", 0)))
                ), 6)
            best = max(
                evaluated,
                key=lambda item: (
                    float(item.get("selection_score", -1)),
                    -float(item.get("estimated_expansion", float("inf"))),
                    -len(item.get("key_pairs", [])),
                ),
            )
            relationship = str(best["relationship"])
            full_cardinality_reaudit_required = bool(
                len(left) < len(datasets[left_name])
                or len(right) < len(datasets[right_name])
            )
            semantic_alias_reaudit_required = any(
                item.get("match_source") == "value_overlap_inferred_alias"
                for item in best["key_pairs"]
            )
            left_time = set(contracts[left_name].get("time_columns", []))
            right_time = set(contracts[right_name].get("time_columns", []))
            selected_left_keys = {item["left"] for item in best["key_pairs"]}
            selected_right_keys = {item["right"] for item in best["key_pairs"]}
            point_in_time_required = bool(
                left_time and right_time
                and not (selected_left_keys & left_time and selected_right_keys & right_time)
            )
            status = (
                "blocked" if relationship == "many_to_many" else
                "restricted" if relationship in {"one_to_many", "many_to_one"}
                or point_in_time_required or full_cardinality_reaudit_required
                or semantic_alias_reaudit_required else
                "admissible"
            )
            direction = {
                "one_to_one": "bidirectional",
                "one_to_many": f"{left_name}_to_{right_name}",
                "many_to_one": f"{right_name}_to_{left_name}",
            }.get(relationship)
            output.append({
                "left_dataset": left_name,
                "right_dataset": right_name,
                "status": status,
                **best,
                "key_candidates_evaluated": len(evaluated),
                "point_in_time_required": point_in_time_required,
                "full_cardinality_reaudit_required": full_cardinality_reaudit_required,
                "semantic_alias_reaudit_required": semantic_alias_reaudit_required,
                "safe_feature_enrichment_direction": direction,
                "combined_additive_analysis": (
                    "admissible_with_unit_check" if relationship == "one_to_one"
                    and not point_in_time_required else
                    "requires_preaggregation_to_estimand_grain"
                ),
                "boundary": (
                    "唯一侧字段可以向多侧补充特征，但唯一侧可加量会被复制，"
                    "不得在连接后直接求总量。"
                    if relationship in {"one_to_many", "many_to_one"} else
                    "多对多原始连接被阻断，必须先对至少一侧按目标粒度聚合。"
                    if relationship == "many_to_many" else
                    "一对一只证明键基数兼容，仍需核对单位、产生时点和业务语义。"
                ),
            })
        return output

    @staticmethod
    def _canonicalize_join_pair(
        left: pd.Series,
        right: pd.Series,
        *,
        is_time: bool,
    ) -> Tuple[pd.Series, pd.Series]:
        if is_time:
            left_time = pd.to_datetime(left, errors="coerce", utc=True)
            right_time = pd.to_datetime(right, errors="coerce", utc=True)
            return left_time.astype("string"), right_time.astype("string")
        left_numeric = pd.to_numeric(left, errors="coerce")
        right_numeric = pd.to_numeric(right, errors="coerce")
        left_rate = float(left_numeric.notna().mean()) if len(left) else 0.0
        right_rate = float(right_numeric.notna().mean()) if len(right) else 0.0
        if left_rate >= 0.95 and right_rate >= 0.95:
            return (
                left_numeric.astype("Float64").astype("string"),
                right_numeric.astype("Float64").astype("string"),
            )

        def clean_text(series: pd.Series) -> pd.Series:
            cleaned = series.astype("string").str.strip().str.casefold()
            return cleaned.mask(cleaned.eq(""), pd.NA)

        return clean_text(left), clean_text(right)

    @staticmethod
    def _evaluate_canonical_join_key(
        left_values: pd.DataFrame,
        right_values: pd.DataFrame,
        *,
        source_left_rows: int,
        source_right_rows: int,
    ) -> Optional[Dict[str, Any]]:
        left_valid = left_values.dropna()
        right_valid = right_values.dropna()
        if left_valid.empty or right_valid.empty:
            return None
        left_counts = left_valid.value_counts(dropna=False)
        right_counts = right_valid.value_counts(dropna=False)
        aligned = pd.concat(
            [left_counts.rename("left"), right_counts.rename("right")],
            axis=1, join="inner",
        ).dropna()
        if aligned.empty:
            return None
        left_unique = bool(int(left_counts.max()) == 1)
        right_unique = bool(int(right_counts.max()) == 1)
        relationship = (
            "one_to_one" if left_unique and right_unique else
            "one_to_many" if left_unique else
            "many_to_one" if right_unique else
            "many_to_many"
        )
        joined_rows = int((aligned["left"] * aligned["right"]).sum())
        matched_left_rows = int(aligned["left"].sum())
        matched_right_rows = int(aligned["right"].sum())
        denominator = max(matched_left_rows, matched_right_rows, 1)
        sampled_overlap_coverage = min(
            matched_left_rows / max(len(left_valid), 1),
            matched_right_rows / max(len(right_valid), 1),
        )
        left_unique_rate = len(left_counts) / max(len(left_valid), 1)
        right_unique_rate = len(right_counts) / max(len(right_valid), 1)
        left_fraction = min(1.0, len(left_valid) / max(int(source_left_rows), 1))
        right_fraction = min(1.0, len(right_valid) / max(int(source_right_rows), 1))
        overlap_coverage = sampled_overlap_coverage
        capture_recapture_used = False
        if (
            len(aligned) >= 10
            and left_unique_rate >= 0.8 and right_unique_rate >= 0.8
            and left_fraction < 1.0 and right_fraction < 1.0
        ):
            estimated_shared = len(aligned) / max(left_fraction * right_fraction, 1e-12)
            estimated_left_unique = min(
                float(source_left_rows), len(left_counts) / max(left_fraction, 1e-12)
            )
            estimated_right_unique = min(
                float(source_right_rows), len(right_counts) / max(right_fraction, 1e-12)
            )
            overlap_coverage = min(
                1.0,
                estimated_shared / max(min(estimated_left_unique, estimated_right_unique), 1.0),
            )
            capture_recapture_used = True
        if len(aligned) < 2 or overlap_coverage < 0.05:
            return None
        return {
            "relationship": relationship,
            "overlap_keys": int(len(aligned)),
            "overlap_coverage": round(float(overlap_coverage), 6),
            "sampled_overlap_coverage": round(float(sampled_overlap_coverage), 6),
            "capture_recapture_used": capture_recapture_used,
            "left_sample_fraction": round(float(left_fraction), 6),
            "right_sample_fraction": round(float(right_fraction), 6),
            "estimated_join_rows": joined_rows,
            "estimated_expansion": round(float(joined_rows / denominator), 6),
            "profiled_left_rows": int(len(left_valid)),
            "profiled_right_rows": int(len(right_valid)),
        }

    @staticmethod
    def _build_credibility_audit(
        *,
        contract: Mapping[str, Any],
        views: Sequence[Mapping[str, Any]],
        stress: Mapping[str, Any],
        reversals: Sequence[Mapping[str, Any]],
        blocked_views: Sequence[Mapping[str, Any]],
        sampled_execution: bool,
    ) -> Dict[str, Any]:
        unresolved = list(contract.get("unresolved") or [])
        grain_warning = contract.get("grain_status") != "verified_unique"
        leakage_failures = [
            view for view in views
            if (view.get("leakage_audit") or {}).get("status") == "fail"
        ]
        conservation_failures = [
            item
            for view in views
            for item in (view.get("conservation_audit") or [])
            if item.get("status") == "fail"
        ]
        assessable_relationships = [
            item for item in stress.get("relationships", [])
            if item.get("status") in {"contradicted", "restricted", "stable_empirical"}
        ]
        checks = [
            {
                "id": "estimand_binding",
                "name": "估计对象与观测粒度绑定",
                "status": "warning" if unresolved or grain_warning else "pass",
                "evidence": (
                    "；".join(unresolved) if unresolved else
                    f"target={contract.get('target')}, grain={contract.get('observed_grain')}, "
                    f"grain_status={contract.get('grain_status')}"
                ),
                "recommendation": "明确一行的实体×时间含义、目标变量和跨粒度聚合语义。",
            },
            {
                "id": "view_invariants",
                "name": "多视图粒度与守恒不变量",
                "status": (
                    "fail" if conservation_failures else
                    "warning" if blocked_views or sampled_execution else
                    "pass"
                ),
                "evidence": (
                    f"admissible={len(views) - len(blocked_views)}/{len(views)}, "
                    f"conservation_failures={len(conservation_failures)}, "
                    f"sampled_execution={sampled_execution}"
                ),
                "recommendation": "仅允许通过键唯一性、总量守恒和有限性检查的视图进入求解。",
            },
            {
                "id": "target_leakage",
                "name": "目标产生时点与未来信息泄漏",
                "status": "fail" if leakage_failures else "pass",
                "evidence": (
                    f"leakage_failures={len(leakage_failures)}；"
                    "目标滚动特征必须使用严格历史shift/lag。"
                ),
                "recommendation": "按真实预测时点重建特征，并执行时间外验证。",
            },
            {
                "id": "conclusion_view_stability",
                "name": "合理数据视图下的结论稳定性",
                "status": (
                    "fail" if reversals else
                    "not_assessed" if stress.get("status") != "assessed" else
                    "pass" if assessable_relationships else
                    "not_assessed"
                ),
                "evidence": (
                    f"tested={len(stress.get('relationships') or [])}, "
                    f"assessable={len(assessable_relationships)}, "
                    f"direction_reversals={len(reversals)}"
                ),
                "recommendation": "发生方向翻转时分开报告总体、组内、组间和时间聚合估计对象。",
            },
        ]
        statuses = {str(check.get("status")) for check in checks}
        audit_status = (
            "fail" if "fail" in statuses else
            "warning" if statuses & {"warning", "not_assessed"} else
            "pass"
        )
        return {
            "status": audit_status,
            "checks": checks,
            "decision": (
                "reject_unstable_relationships" if reversals else
                "restrict_until_full_reaudit" if sampled_execution else
                "restrict_until_semantics_resolved" if unresolved or blocked_views else
                "admit_with_empirical_scope"
            ),
            "boundary": "通过多视图审计不等于因果识别或外部有效性成立。",
        }

    def _build_contract(
        self,
        frame: pd.DataFrame,
        problem: str,
        target: Optional[Any],
        *,
        profile_limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        sampled = self._coverage_sample(frame, profile_limit or self.max_analysis_rows)
        numeric = [str(column) for column in sampled.select_dtypes(include=np.number).columns]
        time_columns: List[str] = []
        technical_ids: List[str] = []
        dimensions: List[str] = []
        column_semantics: List[Dict[str, Any]] = []
        for column in sampled.columns:
            name = str(column)
            series = sampled[name]
            token = _normalized(name)
            unique = int(series.nunique(dropna=True))
            unique_rate = unique / max(int(series.notna().sum()), 1)
            is_time = pd.api.types.is_datetime64_any_dtype(series)
            if not is_time and any(item in token for item in _TIME_TOKENS):
                values = series.dropna().astype(str).head(500)
                is_time = bool(not values.empty and pd.to_datetime(values, errors="coerce").notna().mean() >= 0.8)
            is_numeric = name in numeric
            is_identifier = _looks_identifier(name)
            is_dimension = (
                any(item in token for item in _DIMENSION_TOKENS)
                or (not is_numeric and not is_time and 1 < unique <= max(100, int(math.sqrt(max(len(sampled), 1))) * 2))
            )
            if is_time:
                time_columns.append(name)
            if is_identifier:
                technical_ids.append(name)
            if is_dimension and name not in technical_ids:
                dimensions.append(name)
            additivity = "not_applicable"
            if is_numeric:
                additivity = self._infer_additivity(name)
            column_semantics.append({
                "column": name,
                "dtype": str(series.dtype),
                "role": (
                    "time" if is_time else "technical_id" if is_identifier else
                    "measure" if is_numeric else "dimension" if is_dimension else "attribute"
                ),
                "additivity": additivity,
                "unit": extract_column_unit(name),
                "unique_count": unique,
                "unique_rate": round(unique_rate, 6),
                "missing_rate": round(float(series.isna().mean()), 6),
                "semantic_id": None,
                "semantic_source": "heuristic",
            })

        resolved_target = self._resolve_target(frame, problem, target, numeric)
        grain, grain_uniqueness = self._infer_grain(sampled, time_columns, dimensions, technical_ids)
        grain_status = (
            "verified_unique" if grain and grain_uniqueness >= 0.999999 else
            "near_unique_candidate" if grain else
            "not_identified"
        )
        dependencies = self._infer_functional_dependencies(sampled, dimensions + technical_ids)
        target_semantic = next((item for item in column_semantics if item["column"] == resolved_target), None)
        unresolved: List[str] = []
        if resolved_target is None:
            unresolved.append("目标字段未唯一绑定")
        if not grain:
            unresolved.append("一行观测所代表的实体×时间粒度未验证")
        if target_semantic and target_semantic.get("additivity") == "unknown":
            unresolved.append(f"目标字段“{resolved_target}”能否跨实体或时间相加尚未确认")
        measure_columns = [
            str(item["column"]) for item in column_semantics
            if item.get("role") == "measure"
        ]
        return {
            "rows": int(len(frame)),
            "columns": int(frame.shape[1]),
            "profiled_rows": int(len(sampled)),
            "target": resolved_target,
            "target_additivity": target_semantic.get("additivity") if target_semantic else None,
            "time_columns": time_columns,
            "dimensions": list(dict.fromkeys(dimensions)),
            "technical_ids": list(dict.fromkeys(technical_ids)),
            "numeric_measures": measure_columns,
            "observed_grain": grain,
            "grain_uniqueness": round(grain_uniqueness, 6),
            "grain_status": grain_status,
            "functional_dependencies": dependencies,
            "columns_semantics": column_semantics,
            "unresolved": unresolved,
            "estimand": self._estimand_text(resolved_target, grain),
        }

    def _coverage_sample(self, frame: pd.DataFrame, limit: int) -> pd.DataFrame:
        if len(frame) <= limit:
            return frame
        size = len(frame)
        coverage_count = max(2, limit // 2)
        positions = set(
            int(item) for item in np.linspace(
                0, size - 1, num=coverage_count, dtype=np.int64
            )
        )
        # A coprime modular walk adds deterministic quasi-random coverage without
        # allocating an O(n) permutation for very large tables.
        step = max(1, int(size * 0.6180339887498949))
        while math.gcd(step, size) != 1:
            step += 1
        cursor = (self.random_state * 2_654_435_761 + size + frame.shape[1]) % size
        attempts = 0
        while len(positions) < limit and attempts < size:
            positions.add(int(cursor))
            cursor = (cursor + step) % size
            attempts += 1
        if len(positions) < limit:
            # This branch is only reachable for unusually collision-heavy small
            # margins above the limit and remains bounded by ``limit`` additions.
            cursor = 0
            while len(positions) < limit:
                positions.add(cursor)
                cursor += 1
        return frame.iloc[sorted(positions)]

    @staticmethod
    def _infer_additivity(column: str) -> str:
        token = _normalized(column)
        if any(item in token for item in _NON_ADDITIVE_TOKENS):
            return "non_additive"
        if any(item in token for item in _ADDITIVE_TOKENS):
            return "additive"
        return "unknown"

    @staticmethod
    def _resolve_target(frame: pd.DataFrame, problem: str, target: Optional[Any], numeric: Sequence[str]) -> Optional[str]:
        requested: List[str] = []
        if isinstance(target, str):
            requested = [target]
        elif isinstance(target, Sequence):
            requested = [str(item) for item in target]
        for candidate in requested:
            if candidate in frame.columns:
                return candidate
            suffix = candidate.rsplit(".", 1)[-1]
            if suffix in frame.columns:
                return suffix
        mentioned = [column for column in numeric if column in problem]
        if len(mentioned) == 1:
            return mentioned[0]
        target_tokens = ("目标", "预测", "解释", "因变量", "结果变量", "target")
        for column in mentioned:
            location = problem.find(column)
            window = problem[max(0, location - 15): location + len(column) + 15]
            if any(token in window for token in target_tokens):
                return column
        return None

    @staticmethod
    def _infer_grain(
        frame: pd.DataFrame,
        time_columns: Sequence[str],
        dimensions: Sequence[str],
        technical_ids: Sequence[str],
    ) -> Tuple[List[str], float]:
        candidates: List[List[str]] = []
        times = list(time_columns[:2])
        dims = list(dict.fromkeys(dimensions))[:6]
        ids = list(dict.fromkeys(technical_ids))[:4]
        for column in ids:
            candidates.append([column])
        for time_column in times:
            candidates.append([time_column])
        for dimension in dims:
            candidates.append([dimension])
        for time_column in times:
            for dimension in dims + ids:
                if dimension != time_column:
                    candidates.append([time_column, dimension])
        for pair in combinations(dims[:5], 2):
            candidates.append(list(pair))
        for identifier in ids:
            for dimension in dims[:4]:
                if identifier != dimension:
                    candidates.append([identifier, dimension])
        for time_column in times:
            for pair in combinations(dims[:4], 2):
                candidates.append([time_column, *pair])
        for triple in combinations(dims[:5], 3):
            candidates.append(list(triple))
        best: List[str] = []
        best_rate = 0.0
        exact: Optional[Tuple[List[str], float]] = None
        for keys in candidates:
            valid = frame[keys].dropna()
            if valid.empty:
                continue
            rate = valid.drop_duplicates().shape[0] / len(valid)
            if rate > best_rate:
                best, best_rate = keys, rate
            if exact is None and rate >= 0.999999:
                exact = (keys, rate)
        if exact is not None:
            return exact
        return (best, best_rate) if best_rate >= 0.9 else ([], best_rate)

    @staticmethod
    def _infer_functional_dependencies(frame: pd.DataFrame, columns: Sequence[str]) -> List[Dict[str, Any]]:
        candidates = [column for column in dict.fromkeys(columns) if column in frame.columns][:12]
        dependencies: List[Dict[str, Any]] = []
        for determinant, dependent in combinations(candidates, 2):
            for left, right in ((determinant, dependent), (dependent, determinant)):
                valid = frame[[left, right]].dropna()
                if len(valid) < 5 or valid[left].nunique() < 2:
                    continue
                ambiguity = valid.groupby(left, observed=True)[right].nunique(dropna=True)
                if not ambiguity.empty and int(ambiguity.max()) == 1:
                    dependencies.append({
                        "determinant": [left],
                        "dependent": right,
                        "support_rows": int(len(valid)),
                        "status": "empirical_functional_dependency",
                    })
                    if len(dependencies) >= 20:
                        return dependencies
        return dependencies

    @staticmethod
    def _estimand_text(target: Optional[str], grain: Sequence[str]) -> str:
        if not target:
            return "目标未绑定；当前只能审计数据结构，不能定义统计估计对象。"
        grain_text = " × ".join(grain) if grain else "未验证观测粒度"
        return f"{grain_text}层面的“{target}”条件分布/期望；不自动解释为更高层总量或个体因果效应。"

    def _generate_views(self, frame: pd.DataFrame, contract: Mapping[str, Any], problem: str) -> List[CompiledDataView]:
        target = contract.get("target")
        dimensions = list(contract.get("dimensions") or [])
        time_columns = list(contract.get("time_columns") or [])
        numeric = list(contract.get("numeric_measures") or [])
        semantics = {item["column"]: item for item in contract.get("columns_semantics", [])}
        views = [CompiledDataView(
            view_id="observed_baseline",
            name="原始观测基线",
            purpose="不改变粒度，作为所有处理视图的对照。",
            estimand=str(contract.get("estimand")),
            pipeline=[],
            output_grain=list(contract.get("observed_grain") or []),
            row_relation="baseline",
        )]

        missing_numeric = [column for column in numeric if column != target and frame[column].isna().any()]
        missing_dimensions = [column for column in dimensions if frame[column].isna().any()]
        if missing_numeric or missing_dimensions:
            pipeline: List[Dict[str, Any]] = []
            if missing_numeric:
                pipeline.append({"operation": "fill_missing", "params": {"columns": missing_numeric[:20], "strategy": "median"}})
            if missing_dimensions:
                pipeline.append({"operation": "fill_missing", "params": {"columns": missing_dimensions[:20], "strategy": "mode"}})
            views.append(CompiledDataView(
                view_id="missing_robustness",
                name="缺失机制敏感性视图",
                purpose="比较完整样本与稳健填补后结论，目标字段本身不填补。",
                estimand=str(contract.get("estimand")), pipeline=pipeline,
                output_grain=list(contract.get("observed_grain") or []), row_relation="row_preserving",
            ))

        if time_columns:
            time_column = time_columns[0]
            pipeline = [
                {"operation": "convert_types", "params": {"mapping": {time_column: "datetime"}, "errors": "coerce"}},
                {"operation": "sort_rows", "params": {"by": [*dimensions[:1], time_column], "ascending": [True] * (1 + min(len(dimensions), 1))}},
                {"operation": "time_features", "params": {"time_column": time_column, "features": ["year", "quarter", "month", "dayofweek", "is_weekend", "month_sin", "month_cos"]}},
            ]
            forecast_requested = any(token in problem for token in ("预测", "趋势", "时序", "增长率", "滞后", "forecast"))
            if forecast_requested and target in numeric:
                pipeline.append({
                    "operation": "window_features",
                    "params": {
                        "order_by": time_column,
                        "partition_by": dimensions[:1],
                        "value_columns": [target],
                        "features": [
                            {"kind": "lag", "periods": 1},
                            {"kind": "diff", "periods": 1},
                            {"kind": "rolling_mean", "window": 7, "shift": 1},
                        ],
                    },
                })
            views.append(CompiledDataView(
                view_id="time_available_features",
                name="仅历史可用的时序视图",
                purpose="构造日历与滞后特征；目标滚动量强制shift≥1，避免未来泄漏。",
                estimand=str(contract.get("estimand")), pipeline=pipeline,
                output_grain=list(contract.get("observed_grain") or []), row_relation="row_preserving",
            ))

        group_by = dimensions[:2]
        if group_by and numeric:
            aggregations = []
            for column in numeric[:12]:
                additivity = semantics.get(column, {}).get("additivity")
                function = "sum" if additivity == "additive" else "mean"
                aggregations.append({"column": column, "function": function, "output": f"{column}__{function}"})
            target_function = next((item["function"] for item in aggregations if item["column"] == target), None)
            views.append(CompiledDataView(
                view_id="entity_level_estimand",
                name="实体/分组层估计对象",
                purpose="显式改变估计粒度；可加量求和，率/价格类取均值，未知量不擅自求和。",
                estimand=(
                    f"{' × '.join(group_by)}层面的{target_function or '汇总'}“{target}”"
                    if target else f"{' × '.join(group_by)}层面的指标分布"
                ),
                pipeline=[{"operation": "aggregate", "params": {"group_by": group_by, "aggregations": aggregations}}],
                output_grain=group_by,
                row_relation="grain_changing",
            ))

        if time_columns and numeric:
            time_column = time_columns[0]
            frequency = self._frequency_from_problem(problem)
            aggregations = []
            for column in numeric[:12]:
                additivity = semantics.get(column, {}).get("additivity")
                function = "sum" if additivity == "additive" else "mean"
                aggregations.append({"column": column, "function": function, "output": f"{column}__{function}"})
            panel_group = dimensions[:1]
            views.append(CompiledDataView(
                view_id="time_grain_estimand",
                name="时间粒度重编译视图",
                purpose=f"按{frequency}周期重定义观测，检验结论是否依赖原始记录频率。",
                estimand=(f"{' × '.join(panel_group + [frequency])}粒度的“{target}”" if target else f"{frequency}周期指标"),
                pipeline=[{"operation": "resample_time", "params": {
                    "time_column": time_column,
                    "frequency": frequency,
                    "group_by": panel_group,
                    "aggregations": aggregations,
                }}],
                output_grain=panel_group + [time_column],
                row_relation="grain_changing",
            ))
        return views

    @staticmethod
    def _frequency_from_problem(problem: str) -> str:
        for words, frequency in (
            (("年度", "每年", "按年"), "Y"),
            (("季度", "每季", "按季"), "Q"),
            (("月度", "每月", "按月"), "M"),
            (("周度", "每周", "按周"), "W"),
            (("小时", "每小时", "按小时"), "h"),
        ):
            if any(word in problem for word in words):
                return frequency
        return "D"

    def _execute_and_audit_view(
        self,
        original: pd.DataFrame,
        contract: Mapping[str, Any],
        view: CompiledDataView,
    ) -> Dict[str, Any]:
        blocking: List[str] = []
        warnings: List[str] = []
        audit: List[Dict[str, Any]] = []
        try:
            if view.pipeline:
                execution = self.engine.execute(original, view.pipeline)
                output = execution.data
                audit = execution.audit
                warnings.extend(execution.warnings)
            else:
                output = original
            numeric = output.select_dtypes(include=np.number)
            infinite_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum()) if not numeric.empty else 0
            if infinite_count:
                blocking.append(f"结果含{infinite_count}个无穷值")
            grain_uniqueness = None
            if view.output_grain and all(column in output.columns for column in view.output_grain):
                valid = output[view.output_grain].dropna()
                grain_uniqueness = valid.drop_duplicates().shape[0] / max(len(valid), 1)
                if view.row_relation == "grain_changing" and grain_uniqueness < 0.999999:
                    blocking.append("改变粒度后输出键仍不唯一")
            conservation = self._conservation_audit(original, output, view.pipeline)
            for item in conservation:
                if item["status"] == "fail":
                    blocking.append(f"{item['column']}总量守恒失败")
            leakage = self._leakage_audit(contract.get("target"), view.pipeline)
            if leakage["status"] == "fail":
                blocking.extend(leakage["violations"])
            return {
                "view_id": view.view_id,
                "name": view.name,
                "purpose": view.purpose,
                "estimand": view.estimand,
                "pipeline": view.pipeline,
                "row_relation": view.row_relation,
                "output_grain": view.output_grain,
                "input_shape": list(original.shape),
                "output_shape": list(output.shape),
                "grain_uniqueness": round(grain_uniqueness, 6) if grain_uniqueness is not None else None,
                "conservation_audit": conservation,
                "leakage_audit": leakage,
                "execution_audit": audit,
                "warnings": warnings,
                "blocking_reasons": blocking,
                "admissible": not blocking,
            }
        except (TableTransformError, ValueError, TypeError) as exc:
            return {
                "view_id": view.view_id,
                "name": view.name,
                "purpose": view.purpose,
                "estimand": view.estimand,
                "pipeline": view.pipeline,
                "row_relation": view.row_relation,
                "output_grain": view.output_grain,
                "input_shape": list(original.shape),
                "output_shape": None,
                "grain_uniqueness": None,
                "conservation_audit": [],
                "leakage_audit": {"status": "not_assessed", "violations": []},
                "execution_audit": audit,
                "warnings": warnings,
                "blocking_reasons": [str(exc)],
                "admissible": False,
            }

    @staticmethod
    def _conservation_audit(
        original: pd.DataFrame,
        output: pd.DataFrame,
        pipeline: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        audits: List[Dict[str, Any]] = []
        for step in pipeline:
            if step.get("operation") not in {"aggregate", "resample_time"}:
                continue
            for spec in step.get("params", {}).get("aggregations", []) or []:
                if spec.get("function") != "sum":
                    continue
                source = str(spec.get("column"))
                destination = str(spec.get("output") or f"{source}_sum")
                if source not in original.columns or destination not in output.columns:
                    continue
                before = float(pd.to_numeric(original[source], errors="coerce").sum(min_count=1))
                after = float(pd.to_numeric(output[destination], errors="coerce").sum(min_count=1))
                if not np.isfinite(before) or not np.isfinite(after):
                    audits.append({
                        "column": source,
                        "output": destination,
                        "before_total": _json_scalar(before),
                        "after_total": _json_scalar(after),
                        "relative_error": None,
                        "status": "not_assessed",
                        "reason": "源列或输出列没有可比较的有限总量。",
                    })
                    continue
                scale = max(abs(before), 1.0)
                relative_error = abs(after - before) / scale
                audits.append({
                    "column": source,
                    "output": destination,
                    "before_total": _json_scalar(before),
                    "after_total": _json_scalar(after),
                    "relative_error": relative_error,
                    "status": "pass" if relative_error <= 1e-9 else "fail",
                })
        return audits

    @staticmethod
    def _leakage_audit(target: Optional[str], pipeline: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        violations: List[str] = []
        checks = 0
        for step in pipeline:
            if step.get("operation") != "window_features":
                continue
            params = step.get("params", {})
            if target not in (params.get("value_columns") or []):
                continue
            for feature in params.get("features", []) or []:
                checks += 1
                kind = str(feature.get("kind", ""))
                if kind.startswith("rolling_") and int(feature.get("shift", 1)) < 1:
                    violations.append(f"目标字段{target}的{kind}使用shift<1，会读取当前/未来答案")
                if kind == "lag" and int(feature.get("periods", 1)) < 1:
                    violations.append(f"目标字段{target}的lag不是严格历史值")
        return {
            "status": "fail" if violations else "pass" if checks else "not_applicable",
            "checks": checks,
            "violations": violations,
        }

    def _stress_conclusions(self, frame: pd.DataFrame, contract: Mapping[str, Any]) -> Dict[str, Any]:
        target = contract.get("target")
        numeric = [column for column in contract.get("numeric_measures", []) if column != target]
        technical_ids = set(contract.get("technical_ids", []))
        numeric = [column for column in numeric if column not in technical_ids][:12]
        if target not in frame.columns or target not in contract.get("numeric_measures", []):
            return {
                "status": "not_assessed",
                "reason": "需要唯一绑定的数值目标字段才能执行结论翻转审计。",
                "relationships": [],
            }
        sampled = self._coverage_sample(frame, self.max_analysis_rows).copy()
        groups = [
            column for column in contract.get("dimensions", [])
            if column in sampled.columns and 2 <= sampled[column].nunique(dropna=True) <= 100
        ][:3]
        time_column = next((column for column in contract.get("time_columns", []) if column in sampled.columns), None)
        semantics = {
            str(item.get("column")): str(item.get("additivity", "unknown"))
            for item in contract.get("columns_semantics", [])
        }
        relationships: List[Dict[str, Any]] = []
        global_p_values: List[Optional[float]] = []
        for predictor in numeric:
            contexts = self._relationship_contexts(
                sampled, predictor, target, groups, time_column,
                predictor_additivity=semantics.get(predictor, "unknown"),
                target_additivity=semantics.get(str(target), "unknown"),
            )
            global_item = next((item for item in contexts if item["view"] == "global_complete_case"), None)
            global_p_values.append(global_item.get("p_value") if global_item else None)
            rho_values = [item["rho"] for item in contexts if item.get("rho") is not None]
            spread = max(rho_values) - min(rho_values) if rho_values else None
            relationships.append({
                "target": target,
                "predictor": predictor,
                "contexts": contexts,
                "effect_spread": round(float(spread), 6) if spread is not None else None,
            })
        adjusted = _bh_adjust(global_p_values)
        for item, q_value in zip(relationships, adjusted):
            item["global_fdr_q"] = q_value
            contexts = list(item.get("contexts") or [])
            independently_estimated = [
                context for context in contexts if not context.get("reused_from")
            ]
            independent_q_values = _bh_adjust([
                context.get("p_value") for context in independently_estimated
            ])
            q_by_view: Dict[str, Optional[float]] = {}
            for context, context_q in zip(independently_estimated, independent_q_values):
                q_by_view[str(context.get("view"))] = context_q
            for context in contexts:
                context_q = q_by_view.get(str(context.get("view")))
                if context.get("reused_from"):
                    context_q = q_by_view.get(str(context.get("reused_from")))
                context["context_fdr_q"] = context_q
                context["direction_confirmed"] = bool(
                    context.get("direction_confident_95")
                    and context_q is not None and context_q <= 0.05
                    and context.get("rho") is not None and abs(float(context["rho"])) >= 0.1
                )
            global_item = next(
                (context for context in contexts if context.get("view") == "global_complete_case"),
                None,
            )
            global_confirmed = bool(
                global_item
                and global_item.get("direction_confident_95")
                and q_value is not None and q_value <= 0.05
                and global_item.get("rho") is not None
                and abs(float(global_item["rho"])) >= 0.1
            )
            item["global_significant_fdr_0_05"] = global_confirmed
            flips: List[Dict[str, Any]] = []
            if global_confirmed and global_item is not None:
                for context in contexts:
                    if context is global_item or not context.get("direction_confirmed"):
                        continue
                    if float(global_item["rho"]) * float(context["rho"]) < 0:
                        flips.append({
                            "against": context.get("view"),
                            "global_rho": global_item.get("rho"),
                            "global_ci_95": global_item.get("confidence_interval_95"),
                            "alternative_rho": context.get("rho"),
                            "alternative_ci_95": context.get("confidence_interval_95"),
                            "alternative_fdr_q": context.get("context_fdr_q"),
                        })
            spread = item.get("effect_spread")
            status = (
                "contradicted" if flips else
                "restricted" if spread is not None and float(spread) >= 0.3 else
                "stable_empirical" if global_confirmed else
                "inconclusive"
            )
            item["status"] = status
            item["direction_flips"] = flips
            item["simpson_risk"] = any(
                str(flip.get("against", "")).startswith(("within_group:", "between_group:"))
                for flip in flips
            )
            item["interpretation"] = (
                "总体与替代视图的方向均通过置信区间和FDR门且相反，稳定总体规律被反证。"
                if flips else
                "不同视图下效应大小明显变化，只能限定条件报告。"
                if status == "restricted" else
                "总体关系通过全局FDR且在已检验视图下未出现可信反向；这不是因果证明。"
                if status == "stable_empirical" else
                "总体方向未同时通过效应量、置信区间与全局FDR门，当前不形成关系结论。"
            )
        return {
            "status": "assessed",
            "sample_rows": int(len(sampled)),
            "target": target,
            "relationships": relationships,
            "multiple_testing": (
                "Benjamini-Hochberg FDR applied across global predictors and separately "
                "across alternative views within each relationship"
            ),
            "boundary": "视图稳定性只反证脆弱相关，不识别因果方向。",
        }

    def _relationship_contexts(
        self,
        frame: pd.DataFrame,
        predictor: str,
        target: str,
        groups: Sequence[str],
        time_column: Optional[str],
        *,
        predictor_additivity: str = "unknown",
        target_additivity: str = "unknown",
    ) -> List[Dict[str, Any]]:
        global_context = self._spearman_view(
            "global_complete_case", frame[predictor], frame[target]
        )
        contexts = [global_context]
        numeric = frame[[predictor, target]].copy()
        if numeric[predictor].isna().any():
            imputed = numeric.copy()
            imputed[predictor] = imputed[predictor].fillna(imputed[predictor].median())
            contexts.append(self._spearman_view(
                "median_imputed", imputed[predictor], imputed[target]
            ))
        else:
            contexts.append({
                **global_context,
                "view": "median_imputed",
                "reused_from": "global_complete_case",
            })
        winsorized = numeric.copy()
        for column in (predictor, target):
            low, high = winsorized[column].quantile([0.01, 0.99])
            winsorized[column] = winsorized[column].clip(low, high)
        contexts.append(self._spearman_view("winsorized_1pct", winsorized[predictor], winsorized[target]))
        for group in groups:
            work = frame[[group, predictor, target]].copy()
            grouped = work.groupby(group, observed=True)
            counts = grouped[target].transform("count")
            work = work[counts >= 5]
            grouped = work.groupby(group, observed=True)
            if len(work) >= 12:
                means = grouped[[predictor, target]].transform("mean")
                residual_x = work[predictor] - means[predictor]
                residual_y = work[target] - means[target]
                contexts.append(self._spearman_view(f"within_group:{group}", residual_x, residual_y))
            between = grouped[[predictor, target]].mean().dropna()
            if len(between) >= 5:
                contexts.append(self._spearman_view(f"between_group:{group}", between[predictor], between[target]))
        if time_column:
            dates = pd.to_datetime(frame[time_column], errors="coerce")
            valid = dates.notna()
            if int(valid.sum()) >= 12:
                span_days = (dates[valid].max() - dates[valid].min()).days
                period = "M" if span_days >= 180 else "D"
                temporary = numeric.loc[valid].copy()
                temporary["__period"] = dates.loc[valid].dt.to_period(period).dt.start_time
                temporal = temporary.groupby("__period", observed=True).agg({
                    predictor: "sum" if predictor_additivity == "additive" else "mean",
                    target: "sum" if target_additivity == "additive" else "mean",
                }).dropna()
                if len(temporal) >= 5:
                    contexts.append(self._spearman_view(
                        f"time_aggregate:{period}:"
                        f"{'sum' if predictor_additivity == 'additive' else 'mean'}_"
                        f"{'sum' if target_additivity == 'additive' else 'mean'}",
                        temporal[predictor], temporal[target],
                    ))
        return contexts

    @staticmethod
    def _spearman_view(name: str, predictor: pd.Series, target: pd.Series) -> Dict[str, Any]:
        valid = pd.DataFrame({"x": predictor, "y": target}).replace([np.inf, -np.inf], np.nan).dropna()
        if len(valid) < 8 or valid["x"].nunique() < 3 or valid["y"].nunique() < 3:
            return {"view": name, "n": int(len(valid)), "rho": None, "p_value": None, "status": "insufficient"}
        rho, p_value = spearmanr(valid["x"].to_numpy(), valid["y"].to_numpy())
        if not np.isfinite(rho) or not np.isfinite(p_value):
            return {"view": name, "n": int(len(valid)), "rho": None, "p_value": None, "status": "degenerate"}
        clipped = float(np.clip(rho, -0.999999, 0.999999))
        standard_error = 1.0 / math.sqrt(max(len(valid) - 3, 1))
        fisher_z = math.atanh(clipped)
        lower = math.tanh(fisher_z - 1.96 * standard_error)
        upper = math.tanh(fisher_z + 1.96 * standard_error)
        return {
            "view": name,
            "n": int(len(valid)),
            "rho": round(float(rho), 6),
            "p_value": float(p_value),
            "confidence_interval_95": [round(lower, 6), round(upper, 6)],
            "confidence_interval_method": "Fisher-z approximation for Spearman rank correlation",
            "direction_confident_95": bool(lower > 0 or upper < 0),
            "status": "estimated",
        }
