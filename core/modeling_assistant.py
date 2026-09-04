"""Problem-driven, multi-dataset mathematical modeling assistant.

This module connects the previously independent problem parser, tabular modeling
engine and plotting utilities into one research workflow.  It intentionally keeps
the planning layer deterministic: an LLM can explain the returned evidence, but it
is not allowed to invent joins, variables or numerical conclusions.
"""

from __future__ import annotations

import json
import math
import re
from itertools import combinations
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .problem_solver import analyze_problem
from .mathematical_reasoning import MathematicalReasoningEngine
from .mechanistic_modeling import MechanisticModelingEngine
from .semantic_model_compiler import SemanticModelCompiler
from .mathematical_data_compiler import MathematicalDataCompiler
from .artifact_manager import (
    ARTIFACT_SCHEMA_VERSION,
    RunArtifactManager,
    create_run_id,
)


_NON_NAME = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fff]+")
_ID_WORDS = (
    "id", "编号", "编码", "代码", "序号", "主键", "key", "code", "uuid",
    "客户", "用户", "产品", "商品", "地区", "城市", "企业", "设备",
)
_DATE_WORDS = ("日期", "时间", "年月", "date", "time", "day", "month", "year")
_NEGATIVE_INDICATORS = ("成本", "费用", "风险", "污染", "排放", "耗时", "损失", "cost", "risk", "loss", "time")
_TARGET_NAME_WORDS = (
    "target", "label", "目标", "结果", "销量", "销售", "需求", "得分", "评分", "产量",
    "价格", "利润", "收益", "收入", "满意度", "流失", "故障", "sales", "profit", "revenue",
    "income", "satisfaction", "churn", "failure", "outcome", "response",
)
_TARGET_CONTEXT_WORDS = (
    "预测", "预报", "估计", "拟合", "分类", "判别", "识别", "判断", "forecast", "predict",
    "estimate", "classify", "target", "因变量", "响应变量", "输出变量",
)
_MECHANISTIC_REQUIREMENT_LABELS = {
    "machine_readable_equations_or_algorithms": "可机器读取的方程或算法",
    "verified_symbol_and_unit_bindings": "题面符号、单位与方程核验",
    "decision_variables": "决策变量",
    "objective_function": "目标函数",
    "constraints_and_bounds": "约束与变量边界",
    "state_variables": "状态变量",
    "initial_conditions": "初始条件",
    "boundary_conditions": "边界条件",
    "dynamics_or_transition_rule": "动力学或状态转移规则",
    "geometry_definition": "几何对象与距离定义",
    "event_definition": "事件判定定义",
    "testable_hypothesis": "可检验假设",
    "variables_and_sampling_unit": "变量与样本单位",
    "model_contract_confirmation": "模型契约确认",
}
_MECHANISTIC_STAGE_LABELS = {
    "problem_decomposition": "问题分解",
    "quantity_and_entity_extraction": "实体与显式量抽取",
    "operator_composition": "通用算子组合",
    "canonical_equation_draft": "规范方程草案",
    "operator_selection": "算子选择",
    "binding": "角色绑定",
    "model_draft_ready": "数学草案已形成",
    "needs_confirmation": "待符号与单位确认",
    "not_applicable": "不适用",
    "not_applicable_without_observations": "无观测数据时不适用",
    "applicable": "适用",
    "not_assessed": "未评估",
    "needs_input": "需要补充输入",
    "partial": "部分完成",
    "partially_ready": "部分关系可执行",
    "partially_executed": "部分数值已执行",
    "complete": "完整",
    "complete_with_gaps": "完整但有待绑定项",
    "draft_only": "仅数学草案",
    "runnable": "可运行",
    "deferred": "已延后",
    "blocked": "受阻",
    "warning": "有条件通过",
    "fail": "未通过",
    "supported": "当前契约内支持",
    "conditionally_supported": "有条件支持",
    "rejected": "拒绝结论",
    "machine_compiled": "题面确定性编译",
    "accepted": "候选全部通过",
    "partially_accepted": "部分候选通过",
    "no_accepted_relations": "候选均未通过",
    "failed_safe": "失败并安全降级",
    "ready": "已就绪",
    "executed": "已执行",
    "solver_ready": "求解器已就绪",
    "mechanistic_structure": "机理数学结构",
    "numerical_execution": "数值执行",
    "observational_modeling": "观测数据建模",
}
_CANONICAL_PARTS = (
    (("customer", "client", "user", "客户", "用户"), "customer"),
    (("product", "item", "sku", "商品", "产品", "物料"), "product"),
    (("company", "firm", "enterprise", "企业", "公司"), "company"),
    (("region", "province", "city", "area", "地区", "区域", "省份", "城市"), "region"),
    (("order", "订单"), "order"),
    (("device", "equipment", "设备"), "device"),
    (("date", "datetime", "time", "日期", "时间", "年月"), "date"),
    (("sales", "sale", "销量", "销售量"), "sales"),
    (("demand", "需求", "需求量"), "demand"),
    (("revenue", "income", "收入", "营收"), "revenue"),
    (("profit", "利润", "收益"), "profit"),
    (("cost", "expense", "成本", "费用"), "cost"),
    (("score", "rating", "得分", "评分"), "score"),
    (("identifier", "number", "code", "编号", "编码", "代码", "序号"), "id"),
)


def _plain(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python objects."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_plain(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    if not isinstance(value, (str, bytes)):
        try:
            missing = pd.isna(value)
            if isinstance(missing, (bool, np.bool_)) and missing:
                return None
        except (TypeError, ValueError):
            pass
    return value


def _mechanistic_label(value: Any) -> str:
    text = str(value)
    return _MECHANISTIC_REQUIREMENT_LABELS.get(
        text, _MECHANISTIC_STAGE_LABELS.get(text, text)
    )


def _normalise_name(name: Any) -> str:
    text = _NON_NAME.sub("", str(name).strip().lower())
    for aliases, canonical in _CANONICAL_PARTS:
        for alias in aliases:
            text = text.replace(alias, canonical)
    for suffix in ("identifier", "number", "code", "编号", "编码", "代码"):
        text = text.replace(suffix, "id")
    return text


def _mentioned_as_target(problem: str, column: str) -> bool:
    """Conservatively detect a column in a local target-bearing phrase."""
    text = str(problem).lower()
    token = str(column).strip().lower()
    if not token:
        return False
    for match in re.finditer(re.escape(token), text):
        start = match.start()
        clause_start = max((text.rfind(mark, 0, start) for mark in "。！？!?；;，,\n"), default=-1) + 1
        prefix = text[max(clause_start, start - 48):start]
        suffix = text[start:start + len(token) + 20]
        if any(word in prefix for word in _TARGET_CONTEXT_WORDS):
            return True
        if any(marker in suffix for marker in ("为因变量", "作为因变量", "作为目标", "为目标", "作为输出")):
            return True
    return False


def _split_target_spec(target: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if isinstance(target, str):
        return [part.strip() for part in re.split(r"[,，;；]", target) if part.strip()]
    if isinstance(target, Sequence):
        return [str(part).strip() for part in target if str(part).strip()]
    return []


def _normalise_value(value: Any) -> str:
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if isinstance(value, (list, tuple, set, np.ndarray)):
        sequence = sorted(value, key=str) if isinstance(value, set) else list(value)
        return json.dumps(sequence, ensure_ascii=False, default=str)
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return f"{float(value):.12g}"
    return str(value).strip().casefold()


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except (TypeError, ValueError):
        return int(series.map(_normalise_value).replace("", np.nan).nunique(dropna=True))


def _safe_duplicate_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    try:
        return int(df.duplicated().sum())
    except (TypeError, ValueError):
        comparable = df.apply(lambda column: column.map(_normalise_value))
        return int(comparable.duplicated().sum())


def _make_unique_columns(columns: Iterable[Any]) -> Tuple[List[str], List[str]]:
    result: List[str] = []
    changed: List[str] = []
    counts: Dict[str, int] = {}
    for position, column in enumerate(columns):
        base = str(column).strip() or f"column_{position + 1}"
        counts[base] = counts.get(base, 0) + 1
        name = base if counts[base] == 1 else f"{base}__{counts[base]}"
        result.append(name)
        if name != column:
            changed.append(f"{column!s}→{name}")
    return result, changed


def _compose_key(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Create an unambiguous normalized scalar key from one or more columns."""
    if len(columns) == 1:
        return frame[columns[0]].map(_normalise_value)
    parts = [frame[column].map(_normalise_value) for column in columns]
    composed = parts[0].map(lambda value: f"{len(value)}:{value}")
    for part in parts[1:]:
        composed = composed + "|" + part.map(lambda value: f"{len(value)}:{value}")
    missing = np.zeros(len(frame), dtype=bool)
    for part in parts:
        missing |= part.eq("").to_numpy()
    return composed.mask(missing, "")


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff._-]+", "_", value).strip("._")
    return cleaned[:80] or "dataset"


def _sample_frame(df: pd.DataFrame, limit: int, random_state: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df
    return df.sample(n=limit, random_state=random_state)


def _column_kind(series: pd.Series, name: str = "") -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "categorical"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    normalised = _normalise_name(name)
    if any(token in normalised for token in _DATE_WORDS):
        probe = series.dropna().head(200)
        if len(probe) and pd.to_datetime(probe, errors="coerce", format="mixed").notna().mean() >= 0.8:
            return "datetime"
    unique = _safe_nunique(series)
    return "categorical" if unique <= max(50, int(len(series) * 0.2)) else "text"


def _is_explicit_identifier_name(column: Any) -> bool:
    """Recognise semantic keys independently of dtype and cardinality.

    Repeated entity codes are precisely the identifiers most likely to have a
    low unique ratio in transaction tables. Cardinality therefore cannot be a
    prerequisite for excluding them from metric geometry, trend estimation or
    supervised targets.
    """
    raw = str(column).strip().lower()
    if re.search(r"(?:^|[^0-9a-z])(id|key|code|uuid)(?:$|[^0-9a-z])", raw):
        return True
    if any(token in raw for token in ("编号", "编码", "序号", "主键")):
        return True
    canonical = _normalise_name(raw)
    return canonical in {
        "customer", "product", "company", "region", "order", "device",
        "customerid", "productid", "companyid", "regionid", "orderid", "deviceid",
    }


_TARGET_CONCEPTS: Tuple[Tuple[str, ...], ...] = (
    ("销量", "销售量", "销售总量", "日销售总量", "销售数量", "sales", "salevolume"),
    ("需求", "需求量", "demand"),
    ("价格", "售价", "销售价格", "price"),
    ("利润", "收益", "profit"),
    ("收入", "营收", "revenue", "income"),
    ("产量", "产出", "yield", "output"),
    ("成本", "费用", "cost", "expense"),
    ("损耗", "损失", "loss", "waste"),
    ("得分", "评分", "score", "rating"),
)


def _target_context_score(problem: str, column: str) -> Tuple[float, List[str]]:
    """Score whether a column concept occurs in an actual target-bearing clause."""
    text = str(problem).lower()
    column_norm = _normalise_name(column)
    aliases: List[str] = [str(column).strip().lower()]
    for concept in _TARGET_CONCEPTS:
        if any(_normalise_name(alias) in column_norm for alias in concept):
            aliases.extend(alias.lower() for alias in concept)
    aliases = list(dict.fromkeys(alias for alias in aliases if alias))
    direct_mention = False
    target_context = False
    for alias in aliases:
        for match in re.finditer(re.escape(alias), text):
            direct_mention = True
            start = match.start()
            clause_start = max(
                (text.rfind(mark, 0, start) for mark in "。！？!?；;，,\n"),
                default=-1,
            ) + 1
            prefix = text[max(clause_start, start - 56):start]
            suffix = text[match.end():match.end() + 24]
            if any(word in prefix for word in _TARGET_CONTEXT_WORDS) or any(
                marker in suffix
                for marker in ("为因变量", "作为因变量", "作为目标", "为目标", "作为输出")
            ):
                target_context = True
    score = 0.0
    reasons: List[str] = []
    if direct_mention:
        score += 0.15
        reasons.append("题目一般提及")
    if target_context:
        score += 0.55
        reasons.append("位于目标语境")
    return score, reasons


@dataclass
class DatasetProfile:
    name: str
    n_rows: int
    n_columns: int
    source_rows: int
    memory_mb: float
    numeric_columns: List[str]
    categorical_columns: List[str]
    datetime_columns: List[str]
    text_columns: List[str]
    id_candidates: List[str]
    target_candidates: List[Dict[str, Any]]
    missing_rate: float
    duplicate_rows: int
    role: str = "independent"


@dataclass
class DatasetRelation:
    left_dataset: str
    right_dataset: str
    left_key: str
    right_key: str
    relationship: str
    confidence: float
    name_similarity: float
    value_overlap: float
    left_coverage: float
    right_coverage: float
    estimated_join_rows: int
    safe_to_join: bool
    warning: Optional[str] = None
    left_keys: List[str] = field(default_factory=list)
    right_keys: List[str] = field(default_factory=list)


@dataclass
class InteractionFinding:
    left_dataset: str
    right_dataset: str
    left_variable: str
    right_variable: str
    method: str
    strength: float
    sample_size: int
    direction: str
    interpretation: str
    lag: Optional[int] = None
    p_value: Optional[float] = None
    q_value: Optional[float] = None
    confidence_interval: Optional[List[float]] = None
    significant: Optional[bool] = None
    conditional_strength: Optional[float] = None
    conditioning_variables: List[str] = field(default_factory=list)
    stability_score: Optional[float] = None


@dataclass
class ResearchResult:
    problem: str
    problem_analysis: Dict[str, Any]
    dataset_profiles: List[DatasetProfile]
    relationships: List[DatasetRelation]
    interactions: List[InteractionFinding]
    analysis_plan: List[Dict[str, Any]]
    capability_report: Dict[str, Any]
    mathematical_model_spec: Dict[str, Any] = field(default_factory=dict)
    evidence_bundle: Dict[str, Any] = field(default_factory=dict)
    charts: List[Dict[str, Any]] = field(default_factory=list)
    model_result: Optional[Dict[str, Any]] = None
    model_results: List[Dict[str, Any]] = field(default_factory=list)
    ranking_result: Optional[Dict[str, Any]] = None
    specialized_results: Dict[str, Any] = field(default_factory=dict)
    conclusions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    report_path: Optional[str] = None
    output_dir: Optional[str] = None
    run_id: Optional[str] = None
    artifact_manifest_path: Optional[str] = None
    artifact_schema_version: str = ARTIFACT_SCHEMA_VERSION
    input_mode: str = "data_assisted"

    def to_dict(self) -> Dict[str, Any]:
        return _plain(asdict(self))


class MathModelingAssistant:
    """Run a complete, evidence-backed study over multiple tabular datasets.

    Parameters are deliberately bounded so that relationship discovery and plots
    remain usable on competition-sized files.  Full source row counts may be
    supplied through ``DataFrame.attrs['source_rows']`` when the input is sampled.
    """

    def __init__(
        self,
        output_dir: Optional[str] = None,
        max_analysis_rows: int = 50_000,
        max_relation_values: int = 20_000,
        max_numeric_columns: int = 12,
        max_relationships_per_pair: int = 3,
        max_interaction_pairs: int = 30,
        max_modeling_relations: int = 12,
        feedback_optimization: bool = True,
        feedback_trials: int = 6,
        feedback_min_relative_gain: float = 0.001,
        credibility_audit: bool = True,
        credibility_iterations: int = 200,
        credibility_max_rows: int = 5_000,
        random_state: int = 42,
        semantic_compiler: Optional[SemanticModelCompiler] = None,
    ) -> None:
        if output_dir is None:
            self.run_id = create_run_id()
            self.output_dir = (
                Path("data/reports/modeling_research/runs") / self.run_id
            ).resolve()
        else:
            self.output_dir = Path(output_dir).resolve()
            self.run_id = self.output_dir.name or create_run_id()
        self.max_analysis_rows = max(1_000, int(max_analysis_rows))
        self.max_relation_values = max(1_000, int(max_relation_values))
        self.max_numeric_columns = max(2, int(max_numeric_columns))
        self.max_relationships_per_pair = max(1, int(max_relationships_per_pair))
        self.max_interaction_pairs = max(1, int(max_interaction_pairs))
        self.max_modeling_relations = max(1, int(max_modeling_relations))
        self.feedback_optimization = bool(feedback_optimization)
        self.feedback_trials = max(2, int(feedback_trials))
        self.feedback_min_relative_gain = max(0.0, float(feedback_min_relative_gain))
        self.credibility_audit = bool(credibility_audit)
        self.credibility_iterations = max(50, min(int(credibility_iterations), 1_000))
        self.credibility_max_rows = max(200, min(int(credibility_max_rows), 20_000))
        self.random_state = random_state
        self.semantic_compiler = semantic_compiler
        self._datasets: Dict[str, pd.DataFrame] = {}
        self._profiles: Dict[str, DatasetProfile] = {}
        self._relationships_cache: Optional[List[DatasetRelation]] = None
        self._relation_sketches: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._input_warnings: List[str] = []
        self._runtime_warnings: List[str] = []
        self._custom_analyzers: Dict[str, Callable[..., Optional[Dict[str, Any]]]] = {}
        self._artifact_manager: Optional[RunArtifactManager] = None

    def _artifact_path(self, category: str, relative_name: str) -> Path:
        if self._artifact_manager is None:
            raise RuntimeError("运行产物管理器尚未初始化")
        return self._artifact_manager.path(category, relative_name)

    def register_analyzer(
        self,
        task_type: str,
        analyzer: Callable[..., Optional[Dict[str, Any]]],
    ) -> "MathModelingAssistant":
        """Register a stage-isolated analyzer for a domain-specific task.

        The callable receives keyword arguments ``assistant``, ``problem``,
        ``datasets`` and ``target``.  A failing plugin is reported as a warning and
        cannot abort the built-in research workflow.
        """
        if not task_type or not callable(analyzer):
            raise ValueError("task_type 不能为空且 analyzer 必须可调用")
        self._custom_analyzers[str(task_type)] = analyzer
        return self

    def _run_custom_analyzers(
        self,
        candidate_tasks: Iterable[str],
        problem: str,
        target: Optional[str],
    ) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for task_type in candidate_tasks:
            analyzer = self._custom_analyzers.get(task_type)
            if analyzer is None:
                continue
            try:
                value = analyzer(
                    assistant=self,
                    problem=problem,
                    datasets=self._datasets,
                    target=target,
                )
                if value is not None:
                    results[task_type] = _plain(value)
            except Exception as exc:
                self._runtime_warnings.append(f"扩展分析器 {task_type!r} 失败，已隔离：{exc}")
        return results

    def run(
        self,
        problem: str,
        datasets: Optional[Mapping[str, pd.DataFrame]] = None,
        target: Optional[Union[str, Sequence[str]]] = None,
        run_modeling: bool = True,
        generate_plots: bool = True,
        mechanistic_ir: Optional[Mapping[str, Any]] = None,
        problem_images: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> ResearchResult:
        """Analyze the problem and datasets, execute safe analyses, and report."""
        if not problem or not str(problem).strip():
            raise ValueError("题目描述不能为空")

        self._input_warnings = []
        self._runtime_warnings = []
        self._datasets = self._validate_datasets(datasets or {})
        input_mode = "data_assisted" if self._datasets else "mechanistic_no_dataset"
        self._artifact_manager = RunArtifactManager(self.output_dir, run_id=self.run_id)
        self._relationships_cache = None
        self._relation_sketches = {}
        problem_analysis = analyze_problem(problem)
        profiles = self.profile_datasets(problem)
        spec_targets: Optional[Union[str, Sequence[str]]] = target
        spec_task_types = {
            str(item.get("task_type"))
            for item in problem_analysis.get("task_candidates", [])
            if item.get("task_type")
        } | {str(problem_analysis.get("task_type", ""))}
        if self._datasets and not target and spec_task_types & {
            "prediction_forecast", "classification", "causal_inference"
        }:
            inferred_for_spec = self._select_targets(None, problem=str(problem))
            if "prediction_forecast" in spec_task_types:
                additive_target = self._select_additive_time_target()
                if additive_target is not None:
                    inferred_for_spec = [
                        additive_target,
                        *(item for item in inferred_for_spec if item != additive_target),
                    ]
            spec_targets = [f"{name}.{column}" for name, column in inferred_for_spec]
        reasoning_engine = MathematicalReasoningEngine()
        mechanistic_result = MechanisticModelingEngine(
            semantic_compiler=self.semantic_compiler,
        ).analyze(
            str(problem), ir_override=mechanistic_ir, problem_images=problem_images,
        )
        observational_task_types = {
            "prediction_forecast", "classification", "statistical_inference",
            "causal_inference", "clustering", "dimension_reduction",
            "anomaly_detection", "evaluation_ranking", "data_requirements",
        }
        mechanistic_result["presentation_scope"] = (
            "primary"
            if not self._datasets or not (spec_task_types & observational_task_types)
            else "internal_semantic_support"
        )
        semantic_compilation = mechanistic_result.get("semantic_model_compilation", {})
        if semantic_compilation.get("status") == "failed_safe":
            self._runtime_warnings.append(
                "语义模型编译失败，已安全降级为确定性编译："
                f"{semantic_compilation.get('error', '未知错误')}"
            )
        mathematical_spec = reasoning_engine.build_spec(
            problem=str(problem), datasets=self._datasets,
            problem_analysis=problem_analysis, targets=spec_targets,
            mechanistic_result=mechanistic_result,
        )
        try:
            relationships = self.discover_relationships()
        except Exception as exc:
            relationships = []
            self._runtime_warnings.append(f"跨表关系发现已降级：{exc}")
        self._assign_dataset_roles(relationships)
        try:
            interactions = self.analyze_interactions(relationships)
        except Exception as exc:
            interactions = []
            self._runtime_warnings.append(f"跨表交互分析已降级：{exc}")
        plan = self.build_analysis_plan(problem_analysis, relationships, target)
        warnings: List[str] = list(self._input_warnings)

        target_specs = _split_target_spec(target)
        primary_target = target_specs[0] if target_specs else None
        ranking_result = None
        candidate_tasks = {item["task_type"] for item in problem_analysis.get("task_candidates", [])}
        candidate_tasks.add(problem_analysis["task_type"])
        if primary_target is None and "prediction_forecast" in candidate_tasks:
            additive_primary = self._select_additive_time_target()
            if additive_primary is not None:
                primary_target = f"{additive_primary[0]}.{additive_primary[1]}"
        if "evaluation_ranking" in candidate_tasks:
            ranking_result = self._run_entropy_topsis(primary_target)
            if ranking_result is None:
                warnings.append("评价排名任务缺少至少两个可用数值指标，未执行 TOPSIS。")

        specialized_results: Dict[str, Any] = {}
        specialized_results["mechanistic_model"] = mechanistic_result
        if self._datasets:
            try:
                selected_for_compiler = self._select_target(primary_target)
            except ValueError:
                selected_for_compiler = None
            if selected_for_compiler is None:
                compiler_dataset = max(
                    self._datasets,
                    key=lambda name: len(self._profiles[name].numeric_columns),
                )
                compiler_target = None
            else:
                compiler_dataset, compiler_target = selected_for_compiler
            try:
                data_compilation = MathematicalDataCompiler(
                    max_analysis_rows=self.max_analysis_rows,
                    random_state=self.random_state,
                ).compile_many(
                    self._datasets,
                    problem=str(problem),
                    target=(
                        f"{compiler_dataset}.{compiler_target}"
                        if compiler_target else None
                    ),
                    primary_dataset=compiler_dataset,
                )
                specialized_results["mathematical_data_compilation"] = data_compilation
                if data_compilation.get("summary", {}).get("direction_reversals", 0):
                    warnings.append(
                        "数学数据多视图审计发现结论方向翻转；相关关系已降级，"
                        "不得将全局相关直接写成稳定规律。"
                    )
            except Exception as exc:
                warnings.append(f"数学数据多视图编译未完成：{exc}")
        if self._datasets and "statistical_inference" in candidate_tasks:
            try:
                hierarchical_sales = self._run_hierarchical_additive_analysis(
                    str(problem), self._select_additive_time_target(), relationships
                )
                if hierarchical_sales:
                    specialized_results["hierarchical_distribution"] = hierarchical_sales
                    # Compatibility alias for older report/UI consumers.
                    specialized_results["hierarchical_sales"] = hierarchical_sales
            except Exception as exc:
                warnings.append(f"上下层维度的可加总量分析未完成：{exc}")
        grouped_requests = self._build_grouped_forecast_requests(
            problem_analysis, str(problem)
        ) if self._datasets else []
        if self._datasets and grouped_requests:
            inferred_grouped_target = self._select_additive_time_target(
                self._select_target(target_specs[0]) if target_specs else None
            )
            grouped_forecasts: List[Dict[str, Any]] = []
            for request in grouped_requests:
                try:
                    grouped_forecast = self._run_grouped_time_forecast(
                        request["text"], inferred_grouped_target, relationships,
                        requested_grain=request["grain"],
                        group_column_hint=request.get("group_column_hint"),
                        source_task_ids=request["task_ids"],
                    )
                    if grouped_forecast:
                        grouped_forecasts.append(grouped_forecast)
                except Exception as exc:
                    warnings.append(
                        f"{request['grain']}粒度预测编译未完成：{exc}"
                    )
            if grouped_forecasts:
                specialized_results["grouped_forecasts"] = grouped_forecasts
                # Backward-compatible primary view: prefer the category layer,
                # then preserve request order.  Consumers that understand
                # multiple grains must use grouped_forecasts.
                specialized_results["grouped_forecast"] = next(
                    (
                        item for item in grouped_forecasts
                        if item.get("requested_grain") == "category"
                    ),
                    grouped_forecasts[0],
                )
        if (
            self._datasets
            and "optimization" in candidate_tasks
            and specialized_results.get("grouped_forecasts")
        ):
            prescriptive_decisions: List[Dict[str, Any]] = []
            for grouped_forecast in specialized_results["grouped_forecasts"]:
                request_text = str(grouped_forecast.get("request_text") or problem)
                if not any(
                    token in request_text.lower()
                    for token in ("补货", "订购", "库存", "定价", "价格决策", "replenish", "pricing")
                ):
                    continue
                try:
                    prescriptive = self._run_prescriptive_replenishment_pricing(
                        request_text, grouped_forecast, relationships,
                        supporting_forecasts=specialized_results["grouped_forecasts"],
                    )
                    if prescriptive:
                        prescriptive_decisions.append(prescriptive)
                except Exception as exc:
                    warnings.append(
                        f"{grouped_forecast.get('requested_grain', '分组')}粒度预测到决策编译未完成：{exc}"
                    )
            if prescriptive_decisions:
                specialized_results["prescriptive_decisions"] = prescriptive_decisions
                specialized_results["prescriptive_decision"] = next(
                    (
                        item for item in prescriptive_decisions
                        if item.get("requested_grain") == "category"
                    ),
                    prescriptive_decisions[0],
                )
        if "data_requirements" in candidate_tasks:
            specialized_results["data_requirements"] = self._run_data_requirements_audit(
                problem_analysis, relationships
            )
        if not self._datasets:
            warnings.append(
                "当前为纯机理建模模式：题面数字只作为参数和边界，不会被伪装成观测数据。"
            )
        if "optimization" in candidate_tasks:
            try:
                optimization_result = reasoning_engine.solve_explicit_optimization(
                    mathematical_spec, self._datasets, random_state=self.random_state
                )
                if optimization_result:
                    specialized_results["optimization"] = optimization_result
                elif (
                    (self._datasets or not specialized_results.get("mechanistic_model"))
                    and not specialized_results.get("prescriptive_decisions")
                ):
                    compiler = next((
                        item for item in mathematical_spec.compiler_plan
                        if item.get("task_type") == "optimization"
                    ), {})
                    missing = compiler.get("missing_requirements", [])
                    warnings.append(
                        "优化求解未执行：" + (
                            "、".join(missing) if missing else
                            "仅自动执行完整的显式连续线性模型；其他模型需注册领域求解器。"
                        )
                    )
            except Exception as exc:
                warnings.append(f"显式优化模型编译或求解未完成：{exc}")
        if self._datasets and "graph_network" in candidate_tasks:
            try:
                graph_result = self._run_graph_analysis(problem)
                if graph_result:
                    specialized_results["graph_network"] = graph_result
                else:
                    warnings.append("网络任务未找到明确的起点列和终点列，未构造实体网络。")
            except Exception as exc:
                warnings.append(f"网络分析未完成：{exc}")
        if self._datasets and "simulation" in candidate_tasks:
            try:
                simulation_result = self._run_bootstrap_uncertainty(primary_target)
                if simulation_result:
                    specialized_results["simulation"] = simulation_result
            except Exception as exc:
                warnings.append(f"不确定性仿真未完成：{exc}")
        if self._datasets and (
            "differential_equations" in candidate_tasks
            or (
                "prediction_forecast" in candidate_tasks
                and not specialized_results.get("grouped_forecasts")
            )
        ):
            try:
                dynamics_result = self._run_time_dynamics(primary_target)
                if dynamics_result:
                    specialized_results["time_dynamics"] = dynamics_result
            except Exception as exc:
                warnings.append(f"时序动力特征分析未完成：{exc}")
        if self._datasets and "differential_equations" in candidate_tasks:
            try:
                equation_result = self._run_integral_equation_discovery(primary_target)
                if equation_result:
                    specialized_results["equation_discovery"] = equation_result
                else:
                    warnings.append("动力方程发现需要至少 50 个有效时间点和连续数值状态，当前未执行。")
            except Exception as exc:
                warnings.append(f"积分弱形式动力方程发现未完成：{exc}")
        if self._datasets:
            try:
                structure_results = self._run_data_structure_analysis()
                if structure_results:
                    specialized_results["data_structure"] = structure_results
            except Exception as exc:
                warnings.append(f"潜在结构与异常辅助分析未完成：{exc}")
        if "causal_inference" in candidate_tasks:
            try:
                causal_result = self._run_cross_fitted_causal_effect(problem, primary_target)
                if causal_result:
                    specialized_results["causal_effect"] = causal_result
                else:
                    warnings.append(
                        "因果估计未执行：请显式写明“处理变量=列名、结果变量=列名”；"
                        "系统不会根据相关性自行指定因果方向。"
                    )
            except Exception as exc:
                warnings.append(f"交叉拟合因果效应估计未完成：{exc}")

        model_results: List[Dict[str, Any]] = []
        if run_modeling and self._datasets:
            supervised_requested = bool(
                {"prediction_forecast", "classification", "statistical_inference", "causal_inference"}
                & candidate_tasks
            )
            grouped_forecast_complete = bool(specialized_results.get("grouped_forecast"))
            selected_targets = (
                [] if grouped_forecast_complete and "prediction_forecast" in candidate_tasks
                else self._select_targets(target, problem=str(problem))
                if (target or supervised_requested) else []
            )
            if selected_targets:
                for dataset_name, column in selected_targets:
                    try:
                        sibling_targets = [
                            sibling_column
                            for sibling_dataset, sibling_column in selected_targets
                            if sibling_dataset == dataset_name and sibling_column != column
                        ]
                        model_results.append(
                            self._run_supervised_model(
                                dataset_name, column, problem_analysis,
                                excluded_targets=sibling_targets,
                            )
                        )
                    except Exception as exc:  # one target must not block the others
                        warnings.append(f"目标 {dataset_name}.{column} 的自动预测未完成：{exc}")
            elif "clustering" in candidate_tasks:
                try:
                    model_results.append(self._run_clustering())
                except Exception as exc:
                    warnings.append(f"自动聚类未完成：{exc}")
            elif problem_analysis["task_type"] in {"prediction_forecast", "classification"}:
                warnings.append("题目包含预测/分类目标，但未能可靠识别目标列；请用“数据集.列名”指定 target。")

        custom_results = self._run_custom_analyzers(
            candidate_tasks, problem=str(problem), target=primary_target
        )
        if custom_results:
            specialized_results["custom"] = custom_results

        model_result = model_results[0] if model_results else None
        problem_analysis["task_graph"] = self._resolve_task_graph(
            problem_analysis.get("task_graph", []),
            model_results=model_results,
            relationships=relationships,
            interactions=interactions,
            ranking_result=ranking_result,
            specialized_results=specialized_results,
        )

        primary_task = problem_analysis["task_type"]
        primary_executed = (
            primary_task in {"prediction_forecast", "classification"}
            and model_result is not None
            and model_result.get("task_type") in {"regression", "classification"}
        ) or (
            primary_task == "clustering"
            and model_result is not None
            and model_result.get("task_type") == "clustering"
        ) or (primary_task == "evaluation_ranking" and ranking_result is not None) or (
            primary_task == "statistical_inference" and bool(interactions)
        ) or (
            primary_task == "causal_inference" and bool(specialized_results.get("causal_effect"))
        ) or (
            primary_task == "optimization" and bool(specialized_results.get("optimization"))
        )
        if primary_task in {"anomaly_detection", "dimension_reduction"}:
            primary_executed = bool(specialized_results.get("data_structure"))
        if primary_task == "data_requirements":
            primary_executed = bool(
                specialized_results.get("data_requirements", {}).get("recommendations")
            )
        if primary_task == "prediction_forecast" and specialized_results.get("grouped_forecast"):
            primary_executed = True
        if primary_executed:
            for step in plan:
                if step["phase"] == "primary_model":
                    step["status"] = "completed"
                if (
                    step["phase"] == "validation"
                    and (
                        (
                            model_result is not None
                            and model_result.get("credibility_audit", {}).get("enabled")
                        )
                        or (
                            ranking_result is not None
                            and ranking_result.get("credibility_audit")
                        )
                        or any(
                            item.get("credibility_audit")
                            for item in specialized_results.get("data_structure", [])
                        )
                        or bool(
                            specialized_results.get("causal_effect", {}).get("credibility_audit")
                        )
                        or bool(
                            specialized_results.get("equation_discovery", {}).get("credibility_audit")
                        )
                        or bool(
                            specialized_results.get("optimization", {}).get("credibility_audit")
                        )
                    )
                ):
                    step["status"] = "completed"

        charts: List[Dict[str, Any]] = []
        if generate_plots:
            try:
                charts = self._generate_charts(
                    relationships, interactions, model_results, ranking_result, specialized_results
                )
            except Exception as exc:
                warnings.append(f"部分图表生成失败：{exc}")
        for current_model in model_results:
            # Validation vectors are private plot inputs, not report/API payloads.
            current_model.pop("actual", None)
            current_model.pop("oof_prediction", None)
            current_model.pop("embedding", None)
            current_model.pop("cluster_labels", None)
        for structure in specialized_results.get("data_structure", []):
            # PCA coordinates are private plot inputs; the public result retains
            # only bounded summaries and auditable anomaly row references.
            structure.pop("projection", None)
        equation_result = specialized_results.get("equation_discovery", {})
        equation_result.pop("validation_actual", None)
        equation_result.pop("validation_prediction", None)
        warnings.extend(message for message in self._runtime_warnings if message not in warnings)
        capability_report = self._build_capability_report(
            problem_analysis, relationships, interactions, model_result, ranking_result,
            specialized_results,
        )
        if self._datasets:
            # These top-level tracks describe applicability and execution, not
            # whether every optional model family happened to be attempted.
            mathematical_spec.readiness_by_track["mechanistic_structure"] = "not_applicable"
            numerical_outputs = bool(
                model_results
                or interactions
                or ranking_result
                or any(
                    specialized_results.get(key)
                    for key in (
                        "optimization", "graph_network", "simulation",
                        "time_dynamics", "equation_discovery", "causal_effect",
                        "data_structure", "hierarchical_sales", "hierarchical_distribution",
                        "grouped_forecast", "prescriptive_decision",
                        "mathematical_data_compilation",
                    )
                )
            )
            mathematical_spec.readiness_by_track["numerical_execution"] = (
                "executed" if numerical_outputs else "not_applicable"
            )
        evidence_bundle = reasoning_engine.build_evidence_bundle(
            spec=mathematical_spec,
            datasets=self._datasets,
            relationships=relationships,
            interactions=interactions,
            model_results=model_results,
            ranking_result=ranking_result,
            specialized_results=specialized_results,
            task_graph=problem_analysis.get("task_graph", []),
        )
        conclusions = self._build_evidence_conclusions(
            evidence_bundle.to_dict(), relationships, model_results
        )
        if mathematical_spec.contradictions:
            warnings.extend(
                f"数学规范冲突：{item['message']}"
                for item in mathematical_spec.contradictions
            )
        result = ResearchResult(
            problem=str(problem).strip(),
            problem_analysis=problem_analysis,
            dataset_profiles=list(self._profiles.values()),
            relationships=relationships,
            interactions=interactions,
            analysis_plan=plan,
            capability_report=capability_report,
            mathematical_model_spec=mathematical_spec.to_dict(),
            evidence_bundle=evidence_bundle.to_dict(),
            charts=charts,
            model_result=model_result,
            model_results=model_results,
            ranking_result=ranking_result,
            specialized_results=specialized_results,
            conclusions=conclusions,
            warnings=warnings,
            output_dir=str(self.output_dir),
            run_id=self.run_id,
            artifact_manifest_path=str(self._artifact_manager.manifest_path),
            input_mode=input_mode,
        )
        result.report_path = str(self._write_report(result))
        self._artifact_manager.write_json(
            "evidence.model_spec", "evidence", "mathematical_model_spec.json",
            result.mathematical_model_spec, format_version="1.0", required=True,
        )
        self._artifact_manager.write_json(
            "evidence.bundle", "evidence", "evidence_bundle.json",
            result.evidence_bundle, format_version="1.0", required=True,
        )
        if specialized_results.get("mathematical_data_compilation"):
            compilation_payload = specialized_results["mathematical_data_compilation"]
            self._artifact_manager.write_json(
                "evidence.mathematical_data_compilation",
                "evidence",
                "mathematical_data_compilation.json",
                compilation_payload,
                format_version=str(compilation_payload.get("schema_version", "1.0")),
                required=False,
                metadata={
                    "role": "estimand_and_view_stability_audit",
                    "safe_to_delete_with_run": True,
                },
            )
        if specialized_results.get("mechanistic_model"):
            mechanistic_payload = specialized_results["mechanistic_model"]
            self._artifact_manager.write_json(
                "evidence.mechanistic_model", "evidence", "mechanistic_model.json",
                mechanistic_payload, format_version="1.0", required=True,
            )
            semantic_model_payload = mechanistic_payload.get("semantic_model_compilation")
            if semantic_model_payload and semantic_model_payload.get("status") != "not_configured":
                self._artifact_manager.write_json(
                    "evidence.layer.00.semantic_model",
                    "evidence", "00_semantic_model_compilation.json",
                    semantic_model_payload,
                    format_version=str(semantic_model_payload.get("schema_version", "1.0")),
                    required=False,
                    metadata={
                        "pipeline_layer": "semantic_model_proposals",
                        "model_has_execution_authority": False,
                        "ordered_layer_artifact": True,
                    },
                )
            four_layer = mechanistic_payload.get("four_layer_pipeline", {})
            layer_artifacts = (
                ("evidence.layer.01.semantic", "01_semantic_contract.json", "semantic_contract"),
                ("evidence.layer.02.math_ir", "02_unified_mathematical_ir.json", "mathematical_ir"),
                ("evidence.layer.03.solver_plan", "03_solver_plan.json", "solver_plan"),
                ("evidence.layer.04.audit", "04_independent_audit.json", "independent_audit"),
            )
            for artifact_id, filename, key in layer_artifacts:
                payload = four_layer.get(key)
                if payload:
                    self._artifact_manager.write_json(
                        artifact_id, "evidence", filename, payload,
                        format_version=str(payload.get("schema_version", "1.0")),
                        required=True,
                        metadata={"pipeline_layer": key, "ordered_layer_artifact": True},
                    )
        chart_by_path = {Path(item["path"]).resolve(): item for item in charts}
        for index, chart_path in enumerate(
            sorted((self.output_dir / "charts").glob("*.png")), start=1
        ):
            chart = chart_by_path.get(chart_path.resolve(), {})
            chart_type = _safe_filename(str(chart.get("type", "chart"))).lower()
            self._artifact_manager.register_existing(
                f"chart.{index:03d}.{chart_type}", "charts", chart_path,
                media_type="image/png", format_version="1.0", required=False,
                metadata={"title": chart.get("title", chart_path.stem)},
            )
        self._artifact_manager.write_json(
            "evidence.research_result", "evidence", "research_result.json",
            result.to_dict(), format_version="1.0", required=True,
        )
        self._artifact_manager.finalize("complete")
        return result

    def _validate_datasets(self, datasets: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        validated: Dict[str, pd.DataFrame] = {}
        for raw_name, df in datasets.items():
            if not isinstance(df, pd.DataFrame):
                try:
                    df = pd.DataFrame(df)
                    self._input_warnings.append(f"数据集 {raw_name!r} 已自动转换为 DataFrame。")
                except Exception as exc:
                    self._input_warnings.append(f"数据集 {raw_name!r} 无法转换，已跳过：{exc}")
                    continue
            name = str(raw_name).strip() or f"dataset_{len(validated) + 1}"
            if name in validated:
                suffix = 2
                while f"{name}_{suffix}" in validated:
                    suffix += 1
                name = f"{name}_{suffix}"
            unique_columns, changed = _make_unique_columns(df.columns)
            if changed:
                df = df.copy(deep=False)
                df.columns = unique_columns
                self._input_warnings.append(
                    f"数据集 {name!r} 的空白/重复/非字符串列名已规范化：{', '.join(changed[:8])}。"
                )
            if df.empty:
                self._input_warnings.append(f"数据集 {name!r} 为空，将保留画像但跳过关系与模型分析。")
            validated[name] = df
        return validated

    def profile_datasets(self, problem: str = "") -> List[DatasetProfile]:
        profiles: Dict[str, DatasetProfile] = {}
        problem_analysis = analyze_problem(problem) if problem else {}
        task_types = {
            str(item.get("task_type"))
            for item in problem_analysis.get("task_candidates", [])
            if item.get("task_type")
        }
        if problem_analysis.get("task_type"):
            task_types.add(str(problem_analysis["task_type"]))
        for name, df in self._datasets.items():
            numeric: List[str] = []
            categorical: List[str] = []
            datetimes: List[str] = []
            text: List[str] = []
            id_candidates: List[str] = []
            target_candidates: List[Dict[str, Any]] = []
            n_rows = len(df)
            profile_row_limit = min(
                self.max_analysis_rows,
                10_000,
                max(100, 2_000_000 // max(len(df.columns), 1)),
            )
            sampled = _sample_frame(df, profile_row_limit, self.random_state)
            column_metadata: List[Dict[str, Any]] = []

            for position, col in enumerate(df.columns):
                col_name = str(col)
                series = sampled[col]
                kind = _column_kind(series, col_name)
                semantic_identifier = _is_explicit_identifier_name(col_name)
                if semantic_identifier:
                    # The storage dtype remains untouched. This is a semantic
                    # role: codes are useful as keys/groups, never coordinates.
                    categorical.append(col_name)
                elif kind == "numeric":
                    numeric.append(col_name)
                elif kind == "datetime":
                    datetimes.append(col_name)
                elif kind == "categorical":
                    categorical.append(col_name)
                else:
                    text.append(col_name)

                non_null = series.dropna()
                unique_count = _safe_nunique(non_null)
                unique_ratio = unique_count / max(len(non_null), 1)
                if semantic_identifier or (
                    n_rows >= 20 and kind in {"categorical", "text"} and unique_ratio >= 0.98
                ):
                    id_candidates.append(col_name)
                column_metadata.append({
                    "position": position,
                    "column": col_name,
                    "kind": kind,
                    "unique_count": unique_count,
                    "semantic_identifier": semantic_identifier,
                })

            for metadata in column_metadata:
                position = int(metadata["position"])
                col_name = str(metadata["column"])
                kind = str(metadata["kind"])
                unique_count = int(metadata["unique_count"])
                col_norm = _normalise_name(col_name)
                score, reasons = _target_context_score(problem, col_name)
                if task_types & {"prediction_forecast", "classification", "statistical_inference"}:
                    if any(word in col_norm for word in _TARGET_NAME_WORDS):
                        score += 0.25
                        reasons.append("名称符合结果变量")
                    if position == len(df.columns) - 1:
                        score += 0.03
                        reasons.append("末列弱先验")
                if "prediction_forecast" in task_types and datetimes and kind == "numeric":
                    score += 0.12
                    reasons.append("所在数据集具有时间轴")
                if metadata["semantic_identifier"] or col_name in id_candidates:
                    score = -1.0
                    reasons.append("标识/编码列禁止作为目标")
                elif kind in {"datetime", "text"}:
                    score -= 0.55
                if kind == "numeric" or (kind == "categorical" and unique_count <= 50):
                    score += 0.05
                if score > 0:
                    target_candidates.append({
                        "column": col_name,
                        "score": round(min(score, 1.0), 3),
                        "reasons": reasons,
                    })

            target_candidates.sort(key=lambda item: (-item["score"], item["column"]))
            source_rows = int(df.attrs.get("source_rows", n_rows))
            profiles[name] = DatasetProfile(
                name=name,
                n_rows=n_rows,
                n_columns=len(df.columns),
                source_rows=max(source_rows, n_rows),
                memory_mb=round(float(df.memory_usage(deep=True).sum()) / 1024**2, 3),
                numeric_columns=numeric,
                categorical_columns=categorical,
                datetime_columns=datetimes,
                text_columns=text,
                id_candidates=id_candidates[:20],
                target_candidates=target_candidates[:5],
                missing_rate=round(
                    sum(int(df[col].isna().sum()) for col in df.columns) / df.size if df.size else 0.0,
                    5,
                ),
                duplicate_rows=_safe_duplicate_count(sampled),
            )
        self._profiles = profiles
        return list(profiles.values())

    def discover_relationships(self) -> List[DatasetRelation]:
        """Find evidence-backed join keys without performing an explosive full join."""
        if self._relationships_cache is not None:
            return list(self._relationships_cache)
        if not self._profiles:
            self.profile_datasets()
        names = list(self._datasets)
        candidate_columns = {name: self._relation_candidates(name) for name in names}
        # Normalising values and counting frequencies dominates relationship
        # discovery.  Build each column sketch once instead of once per dataset
        # pair, changing the expensive part from pairwise to linear in columns.
        for name, columns in candidate_columns.items():
            for column in columns:
                self._relation_sketch(name, column)
        relations: List[DatasetRelation] = []
        for left_idx, left_name in enumerate(names):
            for right_name in names[left_idx + 1:]:
                pair_relations: List[DatasetRelation] = []
                left_cols = candidate_columns[left_name]
                right_cols = candidate_columns[right_name]
                for left_col in left_cols:
                    for right_col in right_cols:
                        relation = None
                        try:
                            relation = self._score_relation(left_name, right_name, left_col, right_col)
                        except Exception as exc:
                            message = f"字段关系 {left_name}.{left_col}↔{right_name}.{right_col} 无法比较，已跳过：{exc}"
                            if message not in self._runtime_warnings and len(self._runtime_warnings) < 20:
                                self._runtime_warnings.append(message)
                        if relation is not None:
                            pair_relations.append(relation)
                # Composite discovery is useful only when no single key already
                # establishes a sufficiently unique relation.  Avoid rebuilding
                # pair keys for every pair in a large collection of fact tables.
                has_unique_single_key = any(
                    relation.relationship in {"one_to_one", "one_to_many", "many_to_one"}
                    and relation.confidence >= 75.0
                    for relation in pair_relations
                )
                if not has_unique_single_key:
                    pair_relations.extend(
                        self._discover_composite_relations(left_name, right_name, left_cols, right_cols)
                    )
                pair_relations.sort(
                    key=lambda rel: (rel.confidence, rel.value_overlap, -rel.estimated_join_rows),
                    reverse=True,
                )
                used: set = set()
                for relation in pair_relations:
                    signature = (_normalise_name(relation.left_key), _normalise_name(relation.right_key))
                    if signature in used:
                        continue
                    used.add(signature)
                    relations.append(relation)
                    if len(used) >= self.max_relationships_per_pair:
                        break
        self._relationships_cache = list(relations)
        return relations

    def _relation_candidates(self, dataset_name: str) -> List[str]:
        df = self._datasets[dataset_name]
        profile = self._profiles[dataset_name]
        sample = _sample_frame(df, min(self.max_analysis_rows, 10_000), self.random_state)
        prioritised = list(profile.id_candidates) + list(profile.datetime_columns)
        for col in df.columns:
            col = str(col)
            normalised = _normalise_name(col)
            if col not in prioritised and (
                any(token in normalised for token in _ID_WORDS)
                or _safe_nunique(sample[col]) <= min(20_000, max(100, int(len(sample) * 0.8)))
            ):
                prioritised.append(col)
            if len(prioritised) >= 20:
                break
        return prioritised

    def _score_relation(
        self, left_name: str, right_name: str, left_col: str, right_col: str
    ) -> Optional[DatasetRelation]:
        left_series = self._datasets[left_name][left_col]
        right_series = self._datasets[right_name][right_col]
        return self._score_relation_series(
            left_name=left_name,
            right_name=right_name,
            left_series=left_series,
            right_series=right_series,
            left_label=str(left_col),
            right_label=str(right_col),
            left_keys=[str(left_col)],
            right_keys=[str(right_col)],
            left_total=len(left_series),
            right_total=len(right_series),
            left_sketch=self._relation_sketch(left_name, left_col),
            right_sketch=self._relation_sketch(right_name, right_col),
        )

    def _relation_sketch(self, dataset_name: str, column: str) -> Dict[str, Any]:
        cache_key = (dataset_name, str(column))
        cached = self._relation_sketches.get(cache_key)
        if cached is not None:
            return cached
        source = self._datasets[dataset_name][column].dropna()
        if len(source) > self.max_relation_values:
            source = source.sample(self.max_relation_values, random_state=self.random_state)
        normalized = pd.Series(
            (_normalise_value(value) for value in source),
            dtype="object",
        )
        normalized = normalized[normalized.ne("")]
        counts = normalized.value_counts(sort=False)
        sketch = {
            "counts": counts,
            "values": frozenset(counts.index),
            "sample_size": int(len(source)),
            "unique_ratio": len(counts) / max(len(source), 1),
            "kind": _column_kind(source, str(column)),
        }
        self._relation_sketches[cache_key] = sketch
        return sketch

    def _score_relation_series(
        self,
        left_name: str,
        right_name: str,
        left_series: pd.Series,
        right_series: pd.Series,
        left_label: str,
        right_label: str,
        left_keys: Sequence[str],
        right_keys: Sequence[str],
        left_total: int,
        right_total: int,
        name_similarity: Optional[float] = None,
        left_sketch: Optional[Dict[str, Any]] = None,
        right_sketch: Optional[Dict[str, Any]] = None,
    ) -> Optional[DatasetRelation]:
        left_norm = _normalise_name(left_label)
        right_norm = _normalise_name(right_label)
        if name_similarity is None:
            name_similarity = SequenceMatcher(None, left_norm, right_norm).ratio()
        if left_norm == right_norm:
            name_similarity = 1.0
        id_like = any(token in left_norm + right_norm for token in _ID_WORDS)
        date_like = any(token in left_norm + right_norm for token in _DATE_WORDS)
        if name_similarity < 0.58 and not (id_like and name_similarity >= 0.42):
            return None

        if left_sketch is None:
            left_sample = left_series.dropna()
            if len(left_sample) > self.max_relation_values:
                left_sample = left_sample.sample(self.max_relation_values, random_state=self.random_state)
            left_normalized = pd.Series((_normalise_value(value) for value in left_sample), dtype="object")
            left_counts = left_normalized[left_normalized.ne("")].value_counts(sort=False)
            left_sketch = {
                "counts": left_counts,
                "values": frozenset(left_counts.index),
                "sample_size": int(len(left_sample)),
                "unique_ratio": len(left_counts) / max(len(left_sample), 1),
                "kind": _column_kind(left_sample, left_label),
            }
        if right_sketch is None:
            right_sample = right_series.dropna()
            if len(right_sample) > self.max_relation_values:
                right_sample = right_sample.sample(self.max_relation_values, random_state=self.random_state)
            right_normalized = pd.Series((_normalise_value(value) for value in right_sample), dtype="object")
            right_counts = right_normalized[right_normalized.ne("")].value_counts(sort=False)
            right_sketch = {
                "counts": right_counts,
                "values": frozenset(right_counts.index),
                "sample_size": int(len(right_sample)),
                "unique_ratio": len(right_counts) / max(len(right_sample), 1),
                "kind": _column_kind(right_sample, right_label),
            }
        left_values = left_sketch["values"]
        right_values = right_sketch["values"]
        if not left_values or not right_values:
            return None
        intersection = len(left_values & right_values)
        left_coverage = intersection / len(left_values)
        right_coverage = intersection / len(right_values)
        containment = intersection / min(len(left_values), len(right_values))
        minimum_overlap = 0.01 if name_similarity == 1.0 else 0.08
        if intersection == 0 or containment < minimum_overlap:
            return None

        left_unique = float(left_sketch["unique_ratio"])
        right_unique = float(right_sketch["unique_ratio"])
        kind_match = left_sketch["kind"] == right_sketch["kind"]
        confidence = (
            0.42 * name_similarity
            + 0.38 * containment
            + 0.10 * max(left_unique, right_unique)
            + 0.10 * float(kind_match)
        )
        if not (id_like or date_like) and left_unique < 0.15 and right_unique < 0.15:
            confidence *= 0.7
        if confidence < 0.55:
            return None

        left_is_unique = left_unique >= 0.97
        right_is_unique = right_unique >= 0.97
        if left_is_unique and right_is_unique:
            relationship = "one_to_one"
        elif left_is_unique:
            relationship = "one_to_many"
        elif right_is_unique:
            relationship = "many_to_one"
        else:
            relationship = "many_to_many"

        left_counts = left_sketch["counts"]
        right_counts = right_sketch["counts"]
        shared = left_counts.index.intersection(right_counts.index)
        sampled_join_rows = float((left_counts.loc[shared] * right_counts.loc[shared]).sum())
        scale = (
            left_total / max(int(left_sketch["sample_size"]), 1)
        ) * (
            right_total / max(int(right_sketch["sample_size"]), 1)
        )
        estimated_rows = int(min(sampled_join_rows * scale, np.iinfo(np.int64).max))
        maximum_safe = max(1_000_000, 10 * max(left_total, right_total))
        safe = relationship != "many_to_many" and estimated_rows <= maximum_safe
        warning = None
        if relationship == "many_to_many":
            warning = "多对多关联可能导致行数乘法膨胀；应先按关联键聚合。"
        elif not safe:
            warning = f"预计关联后约 {estimated_rows:,} 行；建议先聚合再关联。"

        return DatasetRelation(
            left_dataset=left_name,
            right_dataset=right_name,
            left_key=left_label,
            right_key=right_label,
            relationship=relationship,
            confidence=round(confidence * 100, 1),
            name_similarity=round(name_similarity, 4),
            value_overlap=round(containment, 4),
            left_coverage=round(left_coverage, 4),
            right_coverage=round(right_coverage, 4),
            estimated_join_rows=estimated_rows,
            safe_to_join=safe,
            warning=warning,
            left_keys=list(left_keys),
            right_keys=list(right_keys),
        )

    def _discover_composite_relations(
        self,
        left_name: str,
        right_name: str,
        left_candidates: Sequence[str],
        right_candidates: Sequence[str],
    ) -> List[DatasetRelation]:
        """Try two-column keys when individual fields are not sufficiently unique."""
        matches: List[Tuple[str, str, float]] = []
        used_right: set = set()
        for left_col in left_candidates:
            scored = []
            for right_col in right_candidates:
                if right_col in used_right:
                    continue
                similarity = SequenceMatcher(
                    None, _normalise_name(left_col), _normalise_name(right_col)
                ).ratio()
                if _normalise_name(left_col) == _normalise_name(right_col):
                    similarity = 1.0
                if similarity >= 0.82:
                    scored.append((similarity, right_col))
            if scored:
                similarity, right_col = max(scored)
                used_right.add(right_col)
                matches.append((left_col, right_col, similarity))
        if len(matches) < 2:
            return []

        left_source = self._datasets[left_name]
        right_source = self._datasets[right_name]
        left_sample = _sample_frame(left_source, self.max_relation_values, self.random_state)
        right_sample = _sample_frame(right_source, self.max_relation_values, self.random_state)
        relations: List[DatasetRelation] = []
        for first, second in combinations(matches[:6], 2):
            left_keys = [first[0], second[0]]
            right_keys = [first[1], second[1]]
            try:
                relation = self._score_relation_series(
                    left_name=left_name,
                    right_name=right_name,
                    left_series=_compose_key(left_sample, left_keys),
                    right_series=_compose_key(right_sample, right_keys),
                    left_label=" + ".join(left_keys),
                    right_label=" + ".join(right_keys),
                    left_keys=left_keys,
                    right_keys=right_keys,
                    left_total=len(left_source),
                    right_total=len(right_source),
                    name_similarity=(first[2] + second[2]) / 2,
                )
            except Exception as exc:
                message = f"复合键 {left_name}.{left_keys}↔{right_name}.{right_keys} 无法比较，已跳过：{exc}"
                if message not in self._runtime_warnings and len(self._runtime_warnings) < 20:
                    self._runtime_warnings.append(message)
                continue
            if relation is not None:
                relations.append(relation)
        return relations

    def _assign_dataset_roles(self, relations: Sequence[DatasetRelation]) -> None:
        scores = {name: {"fact": 0, "dimension": 0} for name in self._datasets}
        for rel in relations:
            if rel.relationship == "many_to_one":
                scores[rel.left_dataset]["fact"] += 1
                scores[rel.right_dataset]["dimension"] += 1
            elif rel.relationship == "one_to_many":
                scores[rel.left_dataset]["dimension"] += 1
                scores[rel.right_dataset]["fact"] += 1
        for name, score in scores.items():
            if score["fact"] > score["dimension"]:
                self._profiles[name].role = "fact"
            elif score["dimension"] > score["fact"]:
                self._profiles[name].role = "dimension"
            elif sum(score.values()):
                self._profiles[name].role = "bridge"

    def analyze_interactions(self, relationships: Sequence[DatasetRelation]) -> List[InteractionFinding]:
        """Measure numeric, categorical and nonlinear cross-table associations."""
        def variable_name(dataset: str, qualified: str) -> str:
            prefix = f"{dataset}::"
            return qualified[len(prefix):] if qualified.startswith(prefix) else qualified.split("::")[-1]

        def display_name(qualified: str) -> str:
            for dataset in self._datasets:
                prefix = f"{dataset}::"
                if qualified.startswith(prefix):
                    return f"{dataset}.{qualified[len(prefix):]}"
            return qualified.replace("::", ".")

        findings: List[InteractionFinding] = []
        tested_findings: List[InteractionFinding] = []
        best_per_pair: Dict[Tuple[str, str], DatasetRelation] = {}
        for relation in relationships:
            pair = (relation.left_dataset, relation.right_dataset)
            if pair not in best_per_pair or relation.confidence > best_per_pair[pair].confidence:
                best_per_pair[pair] = relation
        prioritized_relations = sorted(
            best_per_pair.values(),
            key=lambda item: (item.confidence, item.value_overlap),
            reverse=True,
        )
        if len(prioritized_relations) > self.max_interaction_pairs:
            self._runtime_warnings.append(
                f"跨表交互候选共有 {len(prioritized_relations)} 对；已按关系证据优先分析前 "
                f"{self.max_interaction_pairs} 对，避免数据集数量平方增长。"
            )
            prioritized_relations = prioritized_relations[:self.max_interaction_pairs]
        for relation in prioritized_relations:
            joined, left_columns, right_columns, left_categorical, right_categorical = (
                self._aggregated_mixed_relation_view(relation)
            )
            if len(joined) < 8:
                continue
            for left_col in left_columns:
                for right_col in right_columns:
                    valid = joined[[left_col, right_col]].replace([np.inf, -np.inf], np.nan).dropna()
                    if len(valid) < 8 or valid[left_col].nunique() < 2 or valid[right_col].nunique() < 2:
                        continue
                    try:
                        from scipy.stats import spearmanr
                        coefficient, p_value = spearmanr(valid[left_col], valid[right_col])
                    except Exception:
                        coefficient = valid[left_col].corr(valid[right_col], method="spearman")
                        p_value = np.nan
                    if not np.isfinite(coefficient):
                        continue
                    interval = self._correlation_interval(float(coefficient), len(valid))
                    is_reported = abs(coefficient) >= 0.15
                    conditional, conditioning_variables = (None, [])
                    stability_score = None
                    if is_reported:
                        conditional, conditioning_variables = self._conditional_spearman(
                            joined, left_col, right_col,
                            [column for column in left_columns + right_columns if column not in {left_col, right_col}],
                        )
                        stability_score = self._association_stability(valid, left_col, right_col)
                    tested = InteractionFinding(
                        left_dataset=relation.left_dataset,
                        right_dataset=relation.right_dataset,
                        left_variable=variable_name(relation.left_dataset, left_col),
                        right_variable=variable_name(relation.right_dataset, right_col),
                        method="aggregated_spearman",
                        strength=round(float(coefficient), 5),
                        sample_size=len(valid),
                        direction="positive" if coefficient > 0 else "negative",
                        p_value=round(float(p_value), 10) if np.isfinite(p_value) else None,
                        confidence_interval=interval,
                        conditional_strength=round(conditional, 5) if conditional is not None else None,
                        conditioning_variables=[display_name(column) for column in conditioning_variables],
                        stability_score=round(stability_score, 5) if stability_score is not None else None,
                        interpretation=(
                            f"按 {relation.left_key}/{relation.right_key} 聚合后，"
                            f"{relation.left_dataset}.{variable_name(relation.left_dataset, left_col)} 与 "
                            f"{relation.right_dataset}.{variable_name(relation.right_dataset, right_col)} "
                            f"呈{'正' if coefficient > 0 else '负'}向单调关系（ρ={coefficient:.3f}）。"
                        ),
                    )
                    tested_findings.append(tested)
                    if is_reported:
                        findings.append(tested)
                    elif len(valid) >= 20:
                        nonlinear = self._nonlinear_association(valid[left_col], valid[right_col])
                        if nonlinear >= 0.15:
                            findings.append(InteractionFinding(
                                left_dataset=relation.left_dataset,
                                right_dataset=relation.right_dataset,
                                left_variable=variable_name(relation.left_dataset, left_col),
                                right_variable=variable_name(relation.right_dataset, right_col),
                                method="binned_normalized_mutual_information",
                                strength=round(nonlinear, 5),
                                sample_size=len(valid),
                                direction="nonlinear",
                                interpretation=(
                                    f"{relation.left_dataset}.{variable_name(relation.left_dataset, left_col)} 与 "
                                    f"{relation.right_dataset}.{variable_name(relation.right_dataset, right_col)} "
                                    f"线性/单调关系较弱，但存在非线性依赖（NMI={nonlinear:.3f}）。"
                                ),
                            ))

            for categorical, numeric in (
                [(cat, num) for cat in left_categorical for num in right_columns]
                + [(cat, num) for cat in right_categorical for num in left_columns]
            ):
                effect, p_value, n_valid = self._categorical_numeric_effect(joined[categorical], joined[numeric])
                if n_valid < 8:
                    continue
                tested = InteractionFinding(
                    left_dataset=relation.left_dataset,
                    right_dataset=relation.right_dataset,
                    left_variable=display_name(categorical).rsplit(".", 1)[-1],
                    right_variable=display_name(numeric).rsplit(".", 1)[-1],
                    method="correlation_ratio_eta_squared",
                    strength=round(effect, 5),
                    sample_size=n_valid,
                    direction="group_effect",
                    p_value=round(p_value, 10) if p_value is not None else None,
                    confidence_interval=[0.0, 1.0],
                    interpretation=(
                        f"类别变量 {display_name(categorical)} 对 "
                        f"{display_name(numeric)} 的组间解释度 η²={effect:.3f}。"
                    ),
                )
                tested_findings.append(tested)
                if effect >= 0.02:
                    findings.append(tested)

            for left_cat in left_categorical:
                for right_cat in right_categorical:
                    effect, p_value, n_valid = self._categorical_association(joined[left_cat], joined[right_cat])
                    if n_valid < 8:
                        continue
                    tested = InteractionFinding(
                        left_dataset=relation.left_dataset,
                        right_dataset=relation.right_dataset,
                        left_variable=display_name(left_cat).rsplit(".", 1)[-1],
                        right_variable=display_name(right_cat).rsplit(".", 1)[-1],
                        method="bias_corrected_cramers_v",
                        strength=round(effect, 5),
                        sample_size=n_valid,
                        direction="categorical_association",
                        p_value=round(p_value, 10) if p_value is not None else None,
                        confidence_interval=[0.0, 1.0],
                        interpretation=(
                            f"类别变量 {display_name(left_cat)} 与 "
                            f"{display_name(right_cat)} "
                            f"存在关联（校正 Cramér's V={effect:.3f}）。"
                        ),
                    )
                    tested_findings.append(tested)
                    if effect >= 0.1:
                        findings.append(tested)
        self._apply_fdr(tested_findings)
        findings.sort(key=lambda item: abs(item.strength), reverse=True)
        return findings[:50]

    @staticmethod
    def _correlation_interval(coefficient: float, n_samples: int) -> List[float]:
        if n_samples <= 3:
            return [-1.0, 1.0]
        clipped = float(np.clip(coefficient, -0.999999, 0.999999))
        center = np.arctanh(clipped)
        radius = 1.96 / math.sqrt(n_samples - 3)
        return [round(float(np.tanh(center - radius)), 5), round(float(np.tanh(center + radius)), 5)]

    @staticmethod
    def _nonlinear_association(left: pd.Series, right: pd.Series) -> float:
        from sklearn.metrics import normalized_mutual_info_score
        n_bins = max(2, min(10, int(math.sqrt(len(left)))))
        try:
            left_bins = pd.qcut(left.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
            right_bins = pd.qcut(right.rank(method="first"), q=n_bins, labels=False, duplicates="drop")
            return float(normalized_mutual_info_score(left_bins, right_bins))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _association_stability(valid: pd.DataFrame, left_col: str, right_col: str) -> float:
        n_blocks = min(5, max(2, len(valid) // 20))
        coefficients: List[float] = []
        for positions in np.array_split(np.arange(len(valid)), n_blocks):
            block = valid.iloc[positions]
            if len(block) < 8:
                continue
            coefficient = block[left_col].corr(block[right_col], method="spearman")
            if np.isfinite(coefficient):
                coefficients.append(float(coefficient))
        if len(coefficients) < 2:
            return 0.0
        overall = valid[left_col].corr(valid[right_col], method="spearman")
        expected_sign = np.sign(overall)
        sign_agreement = float(np.mean([np.sign(value) == expected_sign for value in coefficients]))
        dispersion_penalty = max(0.0, 1.0 - float(np.std(coefficients, ddof=1)) / 0.5)
        return float(np.clip(sign_agreement * dispersion_penalty, 0, 1))

    @staticmethod
    def _conditional_spearman(
        joined: pd.DataFrame,
        left_col: str,
        right_col: str,
        candidate_controls: Sequence[str],
        max_controls: int = 3,
    ) -> Tuple[Optional[float], List[str]]:
        numeric_controls = [
            column for column in candidate_controls
            if column in joined.columns and pd.api.types.is_numeric_dtype(joined[column])
        ]
        if not numeric_controls:
            return None, []
        pair = joined[[left_col, right_col] + numeric_controls].replace([np.inf, -np.inf], np.nan)
        correlations = pair.corr(method="spearman")
        ranked = sorted(
            numeric_controls,
            key=lambda column: max(
                abs(float(correlations.at[left_col, column])) if np.isfinite(correlations.at[left_col, column]) else 0.0,
                abs(float(correlations.at[right_col, column])) if np.isfinite(correlations.at[right_col, column]) else 0.0,
            ),
            reverse=True,
        )[:max_controls]
        work = pair[[left_col, right_col] + ranked].dropna()
        if len(work) < max(20, len(ranked) * 5 + 5):
            return None, []
        controls = work[ranked].to_numpy(dtype=float)
        std = controls.std(axis=0)
        keep = std > 1e-12
        controls = controls[:, keep]
        ranked = [column for column, selected in zip(ranked, keep) if selected]
        if controls.shape[1] == 0:
            return None, []
        controls = (controls - controls.mean(axis=0)) / controls.std(axis=0)
        design = np.column_stack([np.ones(len(controls)), controls])
        left_values = work[left_col].to_numpy(dtype=float)
        right_values = work[right_col].to_numpy(dtype=float)
        left_residual = left_values - design @ np.linalg.lstsq(design, left_values, rcond=None)[0]
        right_residual = right_values - design @ np.linalg.lstsq(design, right_values, rcond=None)[0]
        conditional = pd.Series(left_residual).corr(pd.Series(right_residual), method="spearman")
        return (float(conditional), ranked) if np.isfinite(conditional) else (None, [])

    @staticmethod
    def _categorical_numeric_effect(category: pd.Series, numeric: pd.Series) -> Tuple[float, Optional[float], int]:
        valid = pd.DataFrame({"category": category.map(_normalise_value), "numeric": numeric}).replace("", np.nan).dropna()
        if len(valid) < 8:
            return 0.0, None, len(valid)
        top = valid["category"].value_counts().head(20).index
        valid = valid[valid["category"].isin(top)]
        groups = [group["numeric"].to_numpy(dtype=float) for _, group in valid.groupby("category") if len(group) >= 2]
        if len(groups) < 2:
            return 0.0, None, len(valid)
        overall = float(valid["numeric"].mean())
        total = float(((valid["numeric"] - overall) ** 2).sum())
        between = sum(len(group) * float((np.mean(group) - overall) ** 2) for group in groups)
        effect = between / total if total > 0 else 0.0
        try:
            from scipy.stats import f_oneway
            p_value = float(f_oneway(*groups).pvalue)
        except Exception:
            p_value = None
        return float(np.clip(effect, 0, 1)), p_value, len(valid)

    @staticmethod
    def _categorical_association(left: pd.Series, right: pd.Series) -> Tuple[float, Optional[float], int]:
        valid = pd.DataFrame({"left": left.map(_normalise_value), "right": right.map(_normalise_value)}).replace("", np.nan).dropna()
        if len(valid) < 8:
            return 0.0, None, len(valid)
        left_top = valid["left"].value_counts().head(20).index
        right_top = valid["right"].value_counts().head(20).index
        valid["left"] = valid["left"].where(valid["left"].isin(left_top), "__OTHER__")
        valid["right"] = valid["right"].where(valid["right"].isin(right_top), "__OTHER__")
        table = pd.crosstab(valid["left"], valid["right"])
        if min(table.shape) < 2:
            return 0.0, None, len(valid)
        try:
            from scipy.stats import chi2_contingency
            chi2, p_value, _, _ = chi2_contingency(table, correction=False)
        except Exception:
            return 0.0, None, len(valid)
        n = table.to_numpy().sum()
        phi2 = chi2 / n
        rows, cols = table.shape
        corrected_phi2 = max(0.0, phi2 - ((cols - 1) * (rows - 1)) / max(n - 1, 1))
        corrected_rows = rows - ((rows - 1) ** 2) / max(n - 1, 1)
        corrected_cols = cols - ((cols - 1) ** 2) / max(n - 1, 1)
        denominator = min(corrected_rows - 1, corrected_cols - 1)
        effect = math.sqrt(corrected_phi2 / denominator) if denominator > 0 else 0.0
        return float(np.clip(effect, 0, 1)), float(p_value), len(valid)

    @staticmethod
    def _apply_fdr(findings: Sequence[InteractionFinding]) -> None:
        indexed = [(index, item.p_value) for index, item in enumerate(findings) if item.p_value is not None]
        if not indexed:
            return
        indexed.sort(key=lambda pair: pair[1])
        adjusted: Dict[int, float] = {}
        running = 1.0
        for rank in range(len(indexed), 0, -1):
            original_index, p_value = indexed[rank - 1]
            candidate = min(1.0, float(p_value) * len(indexed) / rank)
            running = min(running, candidate)
            adjusted[original_index] = running
        for index, q_value in adjusted.items():
            findings[index].q_value = round(q_value, 10)
            findings[index].significant = q_value <= 0.05

    def _top_numeric(self, df: pd.DataFrame, exclude: Iterable[str] = ()) -> List[str]:
        excluded = {str(col) for col in exclude}
        numeric = [
            str(column) for column in df.select_dtypes(include=[np.number]).columns
            if str(column) not in excluded
            and not _is_explicit_identifier_name(str(column))
        ]
        if len(numeric) <= self.max_numeric_columns:
            return numeric
        sample = _sample_frame(df[numeric], min(len(df), 10_000), self.random_state)
        variances = sample.var(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(-1)
        return [str(c) for c in variances.nlargest(self.max_numeric_columns).index]

    def _top_categorical(self, df: pd.DataFrame, exclude: Iterable[str] = ()) -> List[str]:
        excluded = {str(column) for column in exclude}
        candidates: List[Tuple[float, str]] = []
        for column in df.columns:
            name = str(column)
            if name in excluded:
                continue
            kind = _column_kind(df[column], name)
            unique = _safe_nunique(df[column])
            if kind == "categorical" and 2 <= unique <= 30:
                missing = float(df[column].isna().mean()) if len(df) else 1.0
                candidates.append((missing + unique / 1000, name))
        candidates.sort()
        return [name for _, name in candidates[:6]]

    def _aggregated_mixed_relation_view(
        self, relation: DatasetRelation
    ) -> Tuple[pd.DataFrame, List[str], List[str], List[str], List[str]]:
        left = _sample_frame(self._datasets[relation.left_dataset], self.max_analysis_rows, self.random_state)
        right = _sample_frame(self._datasets[relation.right_dataset], self.max_analysis_rows, self.random_state)
        left_keys = relation.left_keys or [relation.left_key]
        right_keys = relation.right_keys or [relation.right_key]
        left_numeric = self._top_numeric(left, left_keys)
        right_numeric = self._top_numeric(right, right_keys)
        left_categorical = self._top_categorical(left, left_keys)
        right_categorical = self._top_categorical(right, right_keys)

        def aggregate_side(
            frame: pd.DataFrame,
            keys: Sequence[str],
            numeric: Sequence[str],
            categorical: Sequence[str],
            dataset_name: str,
        ) -> Tuple[pd.DataFrame, List[str], List[str]]:
            columns = list(dict.fromkeys(list(keys) + list(numeric) + list(categorical)))
            work = frame[columns].copy()
            work["__key"] = _compose_key(work, keys)
            work = work[work["__key"] != ""]
            pieces: List[pd.DataFrame] = []
            numeric_names = [f"{dataset_name}::{column}" for column in numeric]
            categorical_names = [f"{dataset_name}::{column}" for column in categorical]
            if numeric:
                numeric_agg = work.groupby("__key", sort=False)[list(numeric)].mean()
                numeric_agg.columns = numeric_names
                pieces.append(numeric_agg)
            for column, output_name in zip(categorical, categorical_names):
                values = work[["__key", column]].copy()
                values[column] = values[column].map(_normalise_value)
                values = values[values[column] != ""]
                if values.empty:
                    continue
                counts = values.groupby(["__key", column], sort=False).size().rename("__count").reset_index()
                modes = counts.sort_values("__count", ascending=False).drop_duplicates("__key")
                modes = modes.set_index("__key")[[column]].rename(columns={column: output_name})
                pieces.append(modes)
            if not pieces:
                return pd.DataFrame(), numeric_names, categorical_names
            aggregated = pd.concat(pieces, axis=1)
            return aggregated, numeric_names, [name for name in categorical_names if name in aggregated.columns]

        left_agg, left_numeric_names, left_categorical_names = aggregate_side(
            left, left_keys, left_numeric, left_categorical, relation.left_dataset
        )
        right_agg, right_numeric_names, right_categorical_names = aggregate_side(
            right, right_keys, right_numeric, right_categorical, relation.right_dataset
        )
        if left_agg.empty or right_agg.empty:
            return pd.DataFrame(), [], [], [], []
        return (
            left_agg.join(right_agg, how="inner"),
            left_numeric_names,
            right_numeric_names,
            left_categorical_names,
            right_categorical_names,
        )

    def _aggregated_relation_view(
        self, relation: DatasetRelation
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        left = _sample_frame(self._datasets[relation.left_dataset], self.max_analysis_rows, self.random_state)
        right = _sample_frame(self._datasets[relation.right_dataset], self.max_analysis_rows, self.random_state)
        left_keys = relation.left_keys or [relation.left_key]
        right_keys = relation.right_keys or [relation.right_key]
        left_numeric = self._top_numeric(left, left_keys)
        right_numeric = self._top_numeric(right, right_keys)
        if not left_numeric or not right_numeric:
            return pd.DataFrame(), [], []
        left_work = left[list(left_keys) + left_numeric].copy()
        right_work = right[list(right_keys) + right_numeric].copy()
        left_work["__key"] = _compose_key(left_work, left_keys)
        right_work["__key"] = _compose_key(right_work, right_keys)
        left_work = left_work[left_work["__key"] != ""]
        right_work = right_work[right_work["__key"] != ""]
        left_agg = left_work.groupby("__key", sort=False)[left_numeric].mean()
        right_agg = right_work.groupby("__key", sort=False)[right_numeric].mean()
        left_names = [f"{relation.left_dataset}::{col}" for col in left_numeric]
        right_names = [f"{relation.right_dataset}::{col}" for col in right_numeric]
        left_agg.columns = left_names
        right_agg.columns = right_names
        joined = left_agg.join(right_agg, how="inner")
        return joined, left_names, right_names

    def build_analysis_plan(
        self,
        problem_analysis: Dict[str, Any],
        relationships: Sequence[DatasetRelation],
        target: Optional[Union[str, Sequence[str]]] = None,
    ) -> List[Dict[str, Any]]:
        task = problem_analysis["task_type"]
        if self._datasets:
            plan: List[Dict[str, Any]] = [
                {"phase": "data_audit", "action": "逐表检查类型、缺失、重复和异常规模", "method": "schema profiling", "status": "completed"},
                {"phase": "data_graph", "action": "发现关联键、基数关系与联表膨胀风险", "method": "name + value-overlap evidence", "status": "completed"},
            ]
        else:
            plan = [
                {
                    "phase": "statement_audit",
                    "action": "分解子问题并追踪实体、数值、单位和原文证据",
                    "method": "provenance-aware mathematical IR",
                    "status": "completed",
                },
                {
                    "phase": "mechanism_compilation",
                    "action": "组合动力学、几何、事件与优化原语，按闭合关系独立放行",
                    "method": "deterministic primitive compiler",
                    "status": "completed",
                },
            ]
        if relationships:
            plan.append({
                "phase": "interaction", "action": "按键聚合后计算跨数据集变量关系",
                "method": "Spearman correlation on aggregated entities", "status": "completed",
            })
        task_actions = {
            "data_requirements": (
                "由任务契约与现有字段角色反推数据缺口、采集粒度和验收规则",
                "identifiability and value-of-information audit",
            ),
            "prediction_forecast": ("建立基线预测并使用时间/交叉验证", "regression or time-series validation"),
            "classification": ("建立分类模型并检查类别不平衡", "stratified cross-validation"),
            "clustering": ("标准化特征并比较群组结构", "K-Means / MiniBatchKMeans"),
            "optimization": ("从题目提取决策变量、目标函数与约束并进行敏感性分析", "mathematical programming"),
            "differential_equations": ("识别状态变量与时间列，拟合参数并验证动力学轨迹", "ODE calibration"),
            "simulation": ("拟合输入分布并执行蒙特卡洛不确定性分析", "Monte Carlo"),
            "graph_network": ("将实体与关联转换为图并分析路径/中心性", "graph analysis"),
            "statistical_inference": ("报告效应量、显著性与稳健性", "hypothesis tests / regression"),
            "causal_inference": ("显式定义因果角色并估计正交化处理效应", "cross-fitted double machine learning"),
            "evaluation_ranking": ("指标正向化、熵权赋权并计算综合排名", "entropy-weight TOPSIS"),
            "anomaly_detection": ("建立稳健多变量参照并识别结构偏离样本", "PCA reconstruction + robust MAD"),
            "dimension_reduction": ("提取潜在维度、载荷并验证子空间稳定性", "PCA + split-half stability"),
        }
        action, method = task_actions.get(task, task_actions["statistical_inference"])
        if not self._datasets:
            mechanism_actions = {
                "simulation": (
                    "执行连续轨迹、几何事件求根与区间并集计算",
                    "bounded event solver + root refinement",
                ),
                "optimization": (
                    "逐项符号化决策变量、目标、约束和不确定边界，再调用结构化优化器",
                    "validated mathematical programming",
                ),
                "differential_equations": (
                    "编译状态方程、初边值与单位并执行自适应积分",
                    "validated ODE/PDE solver",
                ),
            }
            action, method = mechanism_actions.get(
                task, ("完成题面数学契约并执行对应通用求解器", "verified operator dispatch")
            )
        plan.append({"phase": "primary_model", "action": action, "method": method, "status": "planned"})
        validation_action = (
            "执行步长/网格加密、替代语义、物理常数与参数敏感性复算"
            if not self._datasets else "进行残差、稳定性或敏感性检查，区分相关与因果"
        )
        validation_method = (
            "numerical convergence + semantic falsification"
            if not self._datasets else "diagnostics"
        )
        plan.extend([
            {"phase": "validation", "action": validation_action, "method": validation_method, "status": "planned"},
            {"phase": "communication", "action": "自动生成核心图、结论、限制和可复核报告", "method": "reproducible artifacts", "status": "completed"},
        ])
        return plan

    def _run_data_requirements_audit(
        self,
        problem_analysis: Mapping[str, Any],
        relationships: Sequence[DatasetRelation],
    ) -> Dict[str, Any]:
        """Derive collection recommendations from task contracts and schema gaps.

        This stage deliberately produces no synthetic observations. Each item
        states what to collect, at which grain, and which unresolved task it can
        make identifiable or executable.
        """
        task_types = {
            str(node.get("task_type"))
            for node in problem_analysis.get("task_graph", [])
            if node.get("task_type") and node.get("task_type") != "data_requirements"
        }
        task_types.update(
            str(item.get("task_type"))
            for item in problem_analysis.get("task_candidates", [])
            if item.get("task_type") and item.get("task_type") != "data_requirements"
        )
        all_columns = [
            str(column)
            for frame in self._datasets.values()
            for column in frame.columns
        ]
        normalized_columns = [_normalise_name(column) for column in all_columns]

        def observed(*aliases: str) -> bool:
            return any(
                _normalise_name(alias) in column
                for alias in aliases
                for column in normalized_columns
            )

        recommendations: List[Dict[str, Any]] = []

        def add(
            priority: str,
            role: str,
            reason: str,
            collection_design: str,
            supports: Sequence[str],
            gap_source: str,
        ) -> None:
            if any(item["data_role"] == role for item in recommendations):
                return
            recommendations.append({
                "priority": priority,
                "data_role": role,
                "reason": reason,
                "collection_design": collection_design,
                "supports_tasks": list(supports),
                "gap_source": gap_source,
            })

        has_time = any(profile.datetime_columns for profile in self._profiles.values())
        has_entity_key = any(profile.id_candidates for profile in self._profiles.values())
        if task_types & {"prediction_forecast", "differential_equations"} and not has_time:
            add(
                "P0", "有序时间戳与观测可用时点",
                "没有时间顺序就无法回测未来误差，也无法防止把未来信息泄漏进特征。",
                "逐条记录事件发生时间和数据可获得时间，统一时区与频率，并保留修订版本。",
                ["prediction_forecast", "differential_equations"], "缺少可验证时间轴",
            )
        if len(self._datasets) > 1 and not has_entity_key:
            add(
                "P0", "跨表稳定实体键",
                "没有稳定键时，多表关系只能停留在候选层，强行按行号拼接会制造伪关系。",
                "为同一实体使用不随时间复用的编码，记录键版本和映射生效区间。",
                ["statistical_inference", "prediction_forecast", "optimization"],
                "多数据集缺少稳定对齐身份",
            )
        if "prediction_forecast" in task_types:
            add(
                "P0", "真实需求/结果及删失标记",
                "成交或观测结果可能受缺货、容量上限和漏报截断；不记录删失会把受限结果误当真实需求。",
                "与目标同一实体和时间粒度记录结果、是否受限、受限原因及暴露量。",
                ["prediction_forecast", "optimization"], "预测目标的生成与删失机制未被证明完整",
            )
            add(
                "P1", "预测时可提前获得的外生驱动",
                "趋势模型无法解释日历、环境或市场冲击；同时必须区分预测时已知与事后才知道的变量。",
                "按目标粒度记录日历、环境、活动和市场状态，并单独保存发布时间/可用时点。",
                ["prediction_forecast"], "现有字段不能覆盖结构突变和外生冲击",
            )
        if "optimization" in task_types:
            if not observed("cost", "成本", "费用"):
                add(
                    "P0", "边际成本与全部决策代价",
                    "缺少边际代价时，利润或成本目标函数无法计算，优化器只能给方法建议。",
                    "按决策对象和生效时段记录采购、运输、处理、机会与违约成本及单位。",
                    ["optimization"], "目标函数参数缺口",
                )
            add(
                "P0", "实时资源状态、容量与服务约束",
                "历史结果不能证明未来可行域；库存、设备、人员、供给和服务下限会直接改变最优解。",
                "在每个决策时点做状态快照，记录上下界、不可用原因、提前期和约束来源。",
                ["optimization", "simulation"], "现实可行域尚未由观测字段闭合",
            )
            add(
                "P1", "决策实施记录与未实施备选",
                "只观察最终方案无法区分需求变化与决策本身的影响，也无法可靠估计反事实收益。",
                "记录候选方案、实际决策、决策时间、执行偏差和结果；条件允许时保留随机或准实验变化。",
                ["optimization", "causal_inference"], "缺少决策—执行—结果闭环",
            )
        if task_types & {"statistical_inference", "causal_inference"}:
            add(
                "P1", "处理前混杂因素与采样机制",
                "相关性可能由共同原因、选择偏差或聚合粒度制造；新增结果后的变量反而会引入偏差。",
                "在处理/决策前记录主体状态、进入样本概率、分组规则和缺失原因。",
                ["statistical_inference", "causal_inference"], "竞争解释尚未被可检验数据区分",
            )
        if observed("sales", "销量", "demand", "需求", "price", "价格"):
            add(
                "P1", "替代/互补关系与曝光机会",
                "对象之间的替代、互补和共同曝光会使单变量价格—需求关系失真。",
                "在同一门店/区域和时段记录可选集合、陈列/曝光、竞争对象价格、促销和缺货状态。",
                ["prediction_forecast", "optimization", "statistical_inference"],
                "跨对象交互缺少共同选择集证据",
            )
        add(
            "P1", "单位、口径、缺失原因与数据延迟元数据",
            "量纲、统计口径或缺失机制不一致会让联表、目标函数和误差比较在数值上成立但语义错误。",
            "建立字段字典，固定单位、聚合口径、有效期、缺失码、来源系统和到达延迟。",
            sorted(task_types) or ["all"], "现有数据不能自行证明语义和量纲一致",
        )
        priority_order = {"P0": 0, "P1": 1, "P2": 2}
        recommendations.sort(
            key=lambda item: (priority_order.get(item["priority"], 9), item["data_role"])
        )
        return _plain({
            "status": "executed",
            "method": "task_contract_minus_observed_roles",
            "observed_dataset_count": len(self._datasets),
            "observed_column_count": len(all_columns),
            "relationship_evidence_count": len(relationships),
            "recommendations": recommendations[:12],
            "note": (
                "这是由任务可执行性与可识别性缺口生成的采集设计，不是领域常识清单；"
                "新增数据仍须通过覆盖率、时点、单位和漂移验收。"
            ),
        })

    def _resolve_task_graph(
        self,
        task_graph: Sequence[Dict[str, Any]],
        *,
        model_results: Sequence[Dict[str, Any]],
        relationships: Sequence[DatasetRelation],
        interactions: Sequence[InteractionFinding],
        ranking_result: Optional[Dict[str, Any]],
        specialized_results: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        """Attach execution evidence and dependency state to each subproblem."""
        resolved: List[Dict[str, Any]] = []
        status_by_id: Dict[str, str] = {}
        custom = specialized_results.get("custom", {})
        structures = specialized_results.get("data_structure", [])
        mechanistic_candidate = specialized_results.get("mechanistic_model", {})
        mechanistic = (
            mechanistic_candidate
            if mechanistic_candidate.get("presentation_scope", "primary") == "primary"
            else {}
        )
        mechanistic_subproblems = mechanistic.get("subproblems", [])
        mechanistic_task_support = mechanistic.get("task_support", {})
        grouped_forecasts = list(specialized_results.get("grouped_forecasts") or [])
        if not grouped_forecasts and specialized_results.get("grouped_forecast"):
            grouped_forecasts = [specialized_results["grouped_forecast"]]
        prescriptive_decisions = list(specialized_results.get("prescriptive_decisions") or [])
        if not prescriptive_decisions and specialized_results.get("prescriptive_decision"):
            prescriptive_decisions = [specialized_results["prescriptive_decision"]]

        def matches_node(item: Mapping[str, Any], node: Mapping[str, Any]) -> bool:
            node_id = str(node.get("id") or "")
            if node_id and node_id in set(map(str, item.get("source_task_ids", []))):
                return True
            grains = self._requested_group_grains(str(node.get("text") or ""))
            return str(item.get("requested_grain")) in grains

        for raw_node in task_graph:
            node = dict(raw_node)
            task_type = node.get("task_type")
            status = "planned"
            evidence = "已生成可执行路线"
            missing: List[str] = []
            mechanistic_override = None
            if not self._datasets:
                for subproblem in mechanistic_subproblems:
                    number = str(subproblem.get("id", "")).replace("problem_", "")
                    if number and re.search(
                        rf"问题\s*{re.escape(number)}(?:\D|$)",
                        str(node.get("text", "")),
                    ):
                        mechanistic_override = subproblem
                        break
                if mechanistic_override is None:
                    mechanistic_override = mechanistic_task_support.get(str(task_type))
            if mechanistic_override:
                raw_status = str(mechanistic_override.get("status", "needs_input"))
                status = {
                    "executed": "executed", "partial": "partial",
                    "ready": "ready", "ready_to_execute": "ready",
                    "needs_input": "needs_input",
                }.get(raw_status, "ready")
                evidence = str(
                    mechanistic_override.get("evidence")
                    or mechanistic_override.get("description")
                    or "已建立题面驱动机理模型"
                )
                missing = list(mechanistic_override.get("missing_requirements", []))
            elif task_type in {"prediction_forecast", "classification"}:
                matching = [
                    item for item in model_results
                    if item.get("task_type") in {"regression", "classification"}
                ]
                grouped_forecast = next(
                    (item for item in grouped_forecasts if matches_node(item, node)), None
                )
                if task_type == "prediction_forecast" and grouped_forecast:
                    status = "executed"
                    evidence = (
                        f"已按题目粒度先聚合再预测 {grouped_forecast.get('groups_forecast', 0)} 个组、"
                        f"{grouped_forecast.get('horizon_days', 0)} 天，并完成末段回测与区间审计"
                    )
                elif matching:
                    status = "executed"
                    evidence = f"已完成 {len(matching)} 个目标的验证建模与可信度审计"
                else:
                    status = "needs_input"
                    missing = ["明确目标列", "足够的有效目标样本"]
            elif task_type == "clustering":
                matching = [item for item in model_results if item.get("task_type") == "clustering"]
                status = "executed" if matching else "ready"
                evidence = "已完成聚类和内部结构评估" if matching else "数据具备聚类候选，尚未执行"
            elif task_type == "evaluation_ranking":
                status = "executed" if ranking_result else "needs_input"
                evidence = "已完成熵权 TOPSIS" if ranking_result else "缺少至少两个有效评价指标"
                if not ranking_result:
                    missing = ["评价指标", "指标正负方向"]
            elif task_type in {"anomaly_detection", "dimension_reduction"}:
                status = "executed" if structures else "needs_input"
                evidence = (
                    f"已对 {len(structures)} 个数据集执行潜在结构、重构异常和稳定性分析"
                    if structures else "缺少至少两个非标识数值指标"
                )
                if not structures:
                    missing = ["至少两个非标识数值指标", "足够样本"]
            elif task_type == "graph_network":
                status = "executed" if specialized_results.get("graph_network") else (
                    "partial" if relationships else "needs_input"
                )
                evidence = (
                    "已构造实体网络并计算结构"
                    if specialized_results.get("graph_network") else
                    ("仅建立了数据表关系图" if relationships else "未识别节点和边")
                )
            elif task_type == "simulation":
                status = "partial" if specialized_results.get("simulation") else "needs_input"
                evidence = "已完成非参数 bootstrap；机理仿真仍需状态规则" if status == "partial" else "缺少随机输入与状态转移规则"
                if status != "partial":
                    missing = ["随机变量分布", "状态转移或结果函数"]
            elif task_type == "differential_equations":
                equation = specialized_results.get("equation_discovery")
                status = "partial" if equation or specialized_results.get("time_dynamics") else "needs_input"
                evidence = (
                    "已发现并外推验证积分弱形式稀疏候选方程；机理含义仍需领域验证"
                    if equation else
                    ("已计算经验动力特征；尚未虚构未知机理方程" if status == "partial" else "缺少状态变量和初边值条件")
                )
                if status != "partial":
                    missing = ["状态变量", "初边值条件", "机理关系"]
            elif task_type == "statistical_inference":
                hierarchical = specialized_results.get("hierarchical_distribution")
                status = "executed" if interactions or hierarchical else ("ready" if relationships else "needs_input")
                evidence = (
                    "已完成日×上层×下层分布、集中度、去趋势/星期效应联动和FDR审计"
                    if hierarchical else
                    ("已完成效应量、条件相关、FDR和稳定性" if interactions else "缺少可检验的对齐变量")
                )
            elif task_type == "causal_inference":
                causal = specialized_results.get("causal_effect")
                status = "executed" if causal else "needs_input"
                evidence = (
                    "已完成交叉拟合正交化效应、重叠性、跨折和安慰剂检查"
                    if causal else "系统不会从相关性自动指定处理和结果角色"
                )
                if not causal:
                    missing = ["显式处理变量", "显式结果变量", "处理前混杂变量"]
            elif task_type == "optimization":
                optimization = specialized_results.get("optimization")
                if optimization:
                    status = "executed"
                    evidence = (
                        "已安全编译连续线性模型并由 HiGHS 求解，完成可行性、近优解和系数扰动审计"
                        if optimization.get("solver_success") else
                        f"已尝试安全编译/求解并得到失败证据：{optimization.get('message', '-')}"
                    )
                else:
                    prescriptive = next(
                        (item for item in prescriptive_decisions if matches_node(item, node)), None
                    )
                if not optimization and prescriptive:
                    status = "partial"
                    evidence = (
                        f"已将分组预测编译为 {prescriptive.get('mathematical_form')}，"
                        f"输出 {prescriptive.get('decision_count', 0)} 条补货/价格候选；"
                        "观察性价格弹性和现实约束仍限制最优性"
                    )
                elif not optimization and task_type in custom:
                    status, evidence = "executed", "已由领域优化器返回可计算结果"
                elif not optimization and not prescriptive:
                    status, evidence = "needs_input", "不会从自然语言臆造目标函数和约束"
                    missing = ["决策变量", "可计算目标函数", "约束和边界"]
            elif task_type == "data_requirements":
                audit = specialized_results.get("data_requirements", {})
                recommendations = audit.get("recommendations", [])
                status = "executed" if recommendations else "needs_input"
                evidence = (
                    f"已由任务契约和现有字段反推 {len(recommendations)} 项分级数据需求、采集粒度与用途"
                    if recommendations else "缺少可审计的任务或字段角色"
                )
                if not recommendations:
                    missing = ["待解决任务", "现有字段角色"]
            dependency_states = {
                dependency: status_by_id.get(dependency, "missing")
                for dependency in node.get("depends_on", [])
            }
            blocked = [
                dependency for dependency, dependency_status in dependency_states.items()
                if dependency_status not in {"executed", "partial"}
            ]
            if blocked and status not in {"executed", "partial"}:
                status = "blocked"
                evidence = f"等待上游节点：{', '.join(blocked)}"
            node.update({
                "status": status,
                "evidence": evidence,
                "dependency_states": dependency_states,
                "missing_requirements": missing,
            })
            status_by_id[str(node.get("id"))] = status
            resolved.append(node)
        return resolved

    def _build_capability_report(
        self,
        problem_analysis: Dict[str, Any],
        relationships: Sequence[DatasetRelation],
        interactions: Sequence[InteractionFinding],
        model_result: Optional[Dict[str, Any]],
        ranking_result: Optional[Dict[str, Any]],
        specialized_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Describe what was executed and where mathematical assumptions are missing."""
        task_graph = problem_analysis.get("task_graph", [])
        candidates = [] if (not self._datasets and task_graph) else [
            item["task_type"] for item in problem_analysis.get("task_candidates", [])
        ]
        for node in task_graph:
            task_type = node.get("task_type")
            if task_type and task_type not in candidates:
                candidates.append(task_type)
        if not candidates and problem_analysis["task_type"] not in candidates:
            candidates.insert(0, problem_analysis["task_type"])
        capabilities: List[Dict[str, Any]] = []
        mechanistic = specialized_results.get("mechanistic_model", {})
        mechanistic_support = mechanistic.get("task_support", {})
        grouped_forecasts = list(specialized_results.get("grouped_forecasts") or [])
        if not grouped_forecasts and specialized_results.get("grouped_forecast"):
            grouped_forecasts = [specialized_results["grouped_forecast"]]
        prescriptive_decisions = list(specialized_results.get("prescriptive_decisions") or [])
        if not prescriptive_decisions and specialized_results.get("prescriptive_decision"):
            prescriptive_decisions = [specialized_results["prescriptive_decision"]]
        for task in candidates:
            status = "planned"
            evidence = "已生成方法路线"
            requirement = None
            if not self._datasets and task in mechanistic_support:
                support = mechanistic_support[task]
                status = str(support.get("status", "needs_input"))
                evidence = str(support.get("evidence", "已建立题面驱动机理模型"))
                missing = support.get("missing_requirements", [])
                requirement = "；".join(str(item) for item in missing) or None
                related_nodes = [
                    node for node in task_graph if node.get("task_type") == task
                ]
                executed_nodes = [
                    node for node in related_nodes if node.get("status") == "executed"
                ]
                pending_nodes = [
                    node for node in related_nodes if node.get("status") != "executed"
                ]
                if executed_nodes and pending_nodes:
                    status = "partial"
                    evidence = (
                        f"{len(executed_nodes)}/{len(related_nodes)} 个该类子问题已形成数值证据；"
                        f"其余 {len(pending_nodes)} 个仍按节点独立审计。"
                    )
                    pending_requirements = list(dict.fromkeys(
                        str(item)
                        for node in pending_nodes
                        for item in node.get("missing_requirements", [])
                    ))
                    requirement = "；".join(pending_requirements) or requirement
            elif task in specialized_results.get("custom", {}):
                status = "executed"
                evidence = "已由注册的扩展分析器执行"
            elif task == "data_requirements":
                audit = specialized_results.get("data_requirements", {})
                count = len(audit.get("recommendations", []))
                status = "executed" if count else "needs_input"
                evidence = (
                    f"已完成信息缺口与采集设计审计，共 {count} 项建议"
                    if count else "未形成可追溯的数据需求"
                )
                requirement = None if count else "待解决任务与现有字段角色"
            elif task in {"prediction_forecast", "classification"}:
                if task == "prediction_forecast" and grouped_forecasts:
                    status = "executed"
                    grains = "、".join(
                        str(item.get("group_column", "组")) for item in grouped_forecasts
                    )
                    evidence = (
                        f"已按 {len(grouped_forecasts)} 个子问题粒度先全量聚合再预测（{grains}），"
                        f"共输出 {sum(int(item.get('groups_forecast', 0)) for item in grouped_forecasts)} 个组层结果"
                    )
                elif model_result and model_result.get("task_type") in {"regression", "classification"}:
                    status = "executed"
                    evidence = f"已完成 {model_result.get('validation', 'cross_validation')}"
                else:
                    has_target_candidate = any(
                        candidate.get("score", 0) >= 0.45
                        for profile in self._profiles.values()
                        for candidate in profile.target_candidates
                    )
                    status = "ready" if has_target_candidate else "needs_input"
                    evidence = "已识别目标候选，尚未运行模型" if has_target_candidate else "未可靠识别目标列"
                    requirement = "确认目标列；时间预测还需要有效时间列"
            elif task == "clustering":
                if model_result and model_result.get("task_type") == "clustering":
                    status = "executed"
                    label = model_result.get("credibility_audit", {}).get("label", "未审计")
                    evidence = f"已自动选择簇数并完成稳定性复核（{label}）"
                else:
                    status = "ready"
                    requirement = "至少两个非标识数值特征"
            elif task == "evaluation_ranking":
                if ranking_result:
                    status = "executed"
                    label = ranking_result.get("credibility_audit", {}).get("label", "未审计")
                    evidence = f"已执行熵权 TOPSIS 与权重敏感性复核（{label}）"
                else:
                    status = "needs_input"
                    requirement = "至少两个指标，并确认正向/负向属性"
            elif task in {"anomaly_detection", "dimension_reduction"}:
                structures = specialized_results.get("data_structure", [])
                if structures:
                    status = "executed"
                    evidence = f"已对 {len(structures)} 个数据集执行稳健结构与异常分析"
                else:
                    status = "needs_input"
                    requirement = "至少两个非标识数值指标和足够样本"
            elif task == "statistical_inference":
                hierarchical = specialized_results.get("hierarchical_distribution")
                status = "executed" if interactions or hierarchical else "ready"
                evidence = (
                    "已执行数据绑定的上下层分布、残差联动与 FDR 校正"
                    if hierarchical else
                    ("已执行混合类型效应量与 FDR 校正" if interactions else "缺少可关联的跨表变量")
                )
            elif task == "causal_inference":
                causal = specialized_results.get("causal_effect")
                if causal:
                    status = "executed"
                    evidence = (
                        "已执行交叉拟合正交化处理效应；"
                        f"审计为{causal.get('credibility_audit', {}).get('label', '未审计')}"
                    )
                else:
                    status = "needs_input"
                    requirement = "显式写明处理变量、结果变量和处理前混杂变量"
            elif task == "graph_network":
                if specialized_results.get("graph_network"):
                    status = "executed"
                    evidence = "已构造实体网络并计算连通性与中心节点"
                    requirement = "指定起点/终点后可进一步求最短路或最大流"
                else:
                    status = "partial" if relationships else "needs_input"
                    evidence = "已建立数据集关系图" if relationships else "未识别到图的边关系"
                    requirement = "路径/流量求解需明确起点、终点、边和权重列"
            elif task == "optimization":
                optimization = specialized_results.get("optimization")
                if optimization:
                    status = "executed"
                    evidence = (
                        "已由 HiGHS 求解显式连续线性模型并完成近优解与参数敏感性审计"
                        if optimization.get("solver_success") else
                        "已执行安全编译/求解并保留不可行、无界或编译失败证据"
                    )
                    requirement = "现实有效性仍需确认约束完整性、参数单位和不确定范围"
                elif prescriptive_decisions:
                    status = "partial"
                    evidence = (
                        f"已把 {len(prescriptive_decisions)} 个粒度的预测情景编译为通用多选MILP并输出 "
                        f"{sum(int(item.get('decision_count', 0)) for item in prescriptive_decisions)} 条条件性决策候选"
                    )
                    requirement = "观察性价格弹性的因果验证；未观测库存、供给与容量约束"
                else:
                    status = "planning_only"
                    requirement = "需要显式决策变量、可解析目标函数和代数约束"
            elif task == "differential_equations":
                equation = specialized_results.get("equation_discovery")
                if equation:
                    status = "partial"
                    evidence = "已执行积分弱形式稀疏方程发现和末段外推验证"
                    requirement = "候选方程仍需单位、守恒律、初边值条件和领域机理确认"
                elif specialized_results.get("time_dynamics"):
                    status = "partial"
                    evidence = "已估计趋势、变化率和自相关动力特征"
                    requirement = "建立 ODE/PDE 仍需状态变量、初边值条件和机理关系"
                else:
                    status = "planning_only"
                    requirement = "需要状态变量、初边值条件和动力学关系"
            elif task == "simulation":
                if specialized_results.get("simulation"):
                    status = "partial"
                    evidence = "已执行非参数 bootstrap 不确定性仿真"
                    requirement = "机理蒙特卡洛仍需随机变量分布与状态转移规则"
                else:
                    status = "planning_only"
                    requirement = "需要随机变量分布、仿真规则和输出指标"
            capabilities.append({
                "task_type": task,
                "status": status,
                "evidence": evidence,
                "requirement": requirement,
            })
        data_guards = [
            "总读取样本预算与逐表行数上限",
            "关系候选缓存与跨表交互全局预算",
            "多对多和复合键先聚合后关联",
            "数值/类别/非线性效应量采用不同统计量",
            "全部已检验假设统一执行 Benjamini-Hochberg FDR 校正",
            "时序跨表特征执行 point-in-time 联接",
            "反馈调参只有超过验证噪声才会替换基线",
            "结果执行泄漏、基线、置乱、稳定性、漂移和敏感性反证",
            "重复实体自动隔离验证且实体键不进入模型",
            "近优模型集合比较不同算法假设的结论分歧",
            "潜在结构与异常名单执行分半和扰动稳定性复核",
            "动力方程使用积分弱形式候选库并通过末段外推反证",
            "因果效应只在显式角色下使用交叉拟合正交化估计",
            "回归预测输出保序区间并单独审计经验覆盖率",
            "综合评价同时保留无权重的 Pareto 非支配方案",
            "单阶段失败自动降级并保留其余结果",
        ]
        mechanism_guards = [
            "所有题面参数保留原文证据坐标",
            "只放行完成角色、单位与边界校验的数学关系",
            "连续事件边界执行网格加密与根精化复算",
            "重叠事件按区间并集计时，避免重复累计",
            "合理几何语义并行计算并报告结论范围",
            "常用物理常数与题面明示参数分开标记",
            "数值收敛、机理正确和现实适用性分别审计",
            "单个子问题可独立执行，未闭合节点不会拖死全局",
        ]
        return {
            "recognized_tasks": candidates,
            "tasks": capabilities,
            "robustness_guards": data_guards if self._datasets else mechanism_guards,
        }

    def _select_target(self, target: Optional[str]) -> Optional[Tuple[str, str]]:
        if target:
            if "." in target:
                dataset_name, column = target.rsplit(".", 1)
                if dataset_name in self._datasets and column in self._datasets[dataset_name].columns:
                    return dataset_name, column
            matches = [(name, target) for name, df in self._datasets.items() if target in df.columns]
            if len(matches) == 1:
                return matches[0]
            raise ValueError(f"目标列 {target!r} 不唯一或不存在，请使用“数据集.列名”")
        candidates: List[Tuple[float, str, str]] = []
        for name, profile in self._profiles.items():
            for candidate in profile.target_candidates:
                candidates.append((candidate["score"], name, candidate["column"]))
        candidates.sort(reverse=True)
        if candidates and candidates[0][0] >= 0.45:
            return candidates[0][1], candidates[0][2]
        return None

    def _select_targets(
        self,
        target: Optional[Union[str, Sequence[str]]],
        problem: Optional[str] = None,
        max_targets: int = 3,
    ) -> List[Tuple[str, str]]:
        explicit = _split_target_spec(target)
        selected: List[Tuple[str, str]] = []
        if explicit:
            for item in explicit[:max_targets]:
                resolved = self._select_target(item)
                if resolved not in selected:
                    selected.append(resolved)
            return selected

        candidates: List[Tuple[float, int, bool, str, str]] = []
        problem_text = str(problem or "").lower()
        for dataset_name, profile in self._profiles.items():
            for candidate in profile.target_candidates:
                if candidate.get("score", 0) >= 0.45:
                    column = candidate["column"]
                    position = problem_text.find(str(column).lower())
                    candidates.append((
                        candidate["score"],
                        position if position >= 0 else len(problem_text) + 1,
                        "位于目标语境" in candidate.get("reasons", []),
                        dataset_name,
                        column,
                    ))
        contextual = [item for item in candidates if item[2]]
        if contextual:
            candidates = contextual
        candidates.sort(key=lambda item: (-item[0], item[1], item[3], item[4]))
        for _, _, _, dataset_name, column in candidates:
            pair = (dataset_name, column)
            if pair not in selected:
                selected.append(pair)
            if len(selected) >= max_targets:
                break
        return selected

    def _select_additive_time_target(
        self,
        explicit: Optional[Tuple[str, str]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Select a flow/count outcome for aggregation, never a price or code.

        General target ranking is deliberately broad because a problem may ask
        for prices, costs, scores, or quantities.  A grouped total forecast has
        a narrower mathematical contract: its outcome must be additive over
        transaction rows.  Keeping that contract separate prevents a highly
        mentioned wholesale-price column from displacing sales quantity.
        """
        additive_tokens = (
            "销量", "销售量", "销售数量", "需求量", "需求", "数量", "产量",
            "sales", "salesvolume", "salevolume", "demand", "quantity", "volume",
        )
        non_additive_tokens = (
            "价格", "单价", "成本", "费率", "比率", "损耗率", "利润率",
            "price", "cost", "rate", "ratio",
        )

        def eligible(pair: Tuple[str, str]) -> bool:
            dataset_name, column = pair
            profile = self._profiles.get(dataset_name)
            if profile is None or not profile.datetime_columns:
                return False
            normalized = _normalise_name(column)
            return (
                column in profile.numeric_columns
                and column not in profile.id_candidates
                and not _is_explicit_identifier_name(column)
                and any(token in normalized for token in additive_tokens)
                and not any(token in normalized for token in non_additive_tokens)
            )

        if explicit is not None:
            return explicit if eligible(explicit) else None

        candidates: List[Tuple[float, int, str, str]] = []
        for dataset_name, profile in self._profiles.items():
            candidate_scores = {
                str(item.get("column")): float(item.get("score", 0.0))
                for item in profile.target_candidates
            }
            for column in profile.numeric_columns:
                pair = (dataset_name, column)
                if not eligible(pair):
                    continue
                normalized = _normalise_name(column)
                semantic = 3.0 if any(
                    token in normalized
                    for token in ("销量", "销售量", "销售数量", "salesvolume", "salevolume")
                ) else 2.0
                candidates.append((
                    semantic + candidate_scores.get(column, 0.0),
                    int(profile.source_rows), dataset_name, column,
                ))
        candidates.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
        return (candidates[0][2], candidates[0][3]) if candidates else None

    @staticmethod
    def _requested_group_grains(text: str) -> List[str]:
        normalized = str(text).lower()
        category = any(token in normalized for token in ("品类", "分类", "category"))
        # “商品” alone is too broad: category questions also contain this word.
        item = any(token in normalized for token in ("单品", "sku", "item-level", "item level"))
        item_decision = item and bool(re.search(
            r"(?:单品.{0,12}(?:补货|订购|定价)|(?:补货|订购|定价).{0,12}单品|可售单品|各单品)",
            normalized,
        ))
        category_decision = category and bool(re.search(
            r"(?:以(?:品类|分类)为单位|(?:各|按)(?:蔬菜)?(?:品类|分类).{0,16}(?:补货|预测|销售总量))",
            normalized,
        ))
        # A lower-level decision may mention parent-category demand only as a
        # coupling objective.  That does not request a second category output.
        if item_decision and not category_decision:
            return ["item"]
        if category_decision and not item_decision:
            return ["category"]
        if category or item:
            return [
                grain for grain, present in (("category", category), ("item", item))
                if present
            ]
        if any(token in normalized for token in ("各产品", "各商品", "每个产品", "每个商品")):
            return ["item"]
        return []

    def _discover_mentioned_group_bindings(
        self,
        text: str,
    ) -> List[Dict[str, Any]]:
        """Bind arbitrary mentioned dimensions to real dataset columns.

        This is the semantic layer of the grouped compiler.  It uses column
        evidence rather than a fixed domain taxonomy; category/item labels are
        retained only as backward-compatible aliases.
        """
        normalized_text = _normalise_name(text)
        requested_aliases = set(self._requested_group_grains(text))
        candidates: List[Tuple[float, int, str, str, str]] = []
        suffix_pattern = re.compile(
            r"(?:名称|编码|代码|编号|标识|id|name|code|identifier)$",
            re.IGNORECASE,
        )
        for dataset_name, profile in self._profiles.items():
            frame = self._datasets[dataset_name]
            for column in dict.fromkeys(
                profile.categorical_columns + profile.id_candidates
            ):
                column_name = str(column)
                normalized_column = _normalise_name(column_name)
                semantic_base = _normalise_name(suffix_pattern.sub("", column_name))
                direct = bool(normalized_column and normalized_column in normalized_text)
                base_mentioned = bool(
                    len(semantic_base) >= 2 and semantic_base in normalized_text
                )
                alias = "dimension"
                lowered = column_name.lower()
                if any(token in lowered for token in ("品类", "分类", "category")):
                    alias = "category"
                elif any(token in lowered for token in ("单品", "商品", "产品", "sku", "item")):
                    alias = "item"
                semantic_alias_match = alias in requested_aliases
                if not (direct or base_mentioned or semantic_alias_match):
                    continue
                unique_count = _safe_nunique(frame[column_name])
                if unique_count < 2 or unique_count > 500:
                    continue
                score = (
                    (4.0 if direct else 2.0)
                    + (1.5 if semantic_alias_match else 0.0)
                    + (1.0 if any(token in lowered for token in ("名称", "name")) else 0.0)
                    - (0.25 if _is_explicit_identifier_name(column_name) else 0.0)
                )
                candidates.append(
                    (score, unique_count, dataset_name, column_name, alias)
                )
        # Mentioning both a code and its name should create one mathematical
        # grain. Prefer the readable name, then the lower-cardinality binding.
        selected: Dict[str, Tuple[float, int, str, str, str]] = {}
        for candidate in candidates:
            _, _, _, column, alias = candidate
            base = _normalise_name(suffix_pattern.sub("", column)) or _normalise_name(column)
            key = alias if alias != "dimension" else base
            current = selected.get(key)
            if current is None or (-candidate[0], candidate[1], candidate[2], candidate[3]) < (
                -current[0], current[1], current[2], current[3]
            ):
                selected[key] = candidate
        output = [
            {
                "grain": (
                    alias if alias != "dimension"
                    else f"dimension:{dataset_name}.{column}"
                ),
                "group_column_hint": {
                    "dataset": dataset_name,
                    "column": column,
                    "unique_count": unique_count,
                },
            }
            for _, unique_count, dataset_name, column, alias in selected.values()
        ]
        output.sort(key=lambda item: (
            int(item["group_column_hint"]["unique_count"]), item["grain"]
        ))
        return output

    def _build_grouped_forecast_requests(
        self,
        problem_analysis: Mapping[str, Any],
        problem: str,
    ) -> List[Dict[str, Any]]:
        """Compile one forecast request per subproblem and mathematical grain."""
        requests: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for node in problem_analysis.get("task_graph", []):
            if node.get("task_type") not in {"prediction_forecast", "optimization"}:
                continue
            text = str(node.get("text") or "")
            lowered = text.lower()
            if not any(
                token in lowered
                for token in (
                    "预测", "未来", "补货", "订购", "库存", "销量", "销售总量",
                    "forecast", "replenish", "demand",
                )
            ):
                continue
            bindings = self._discover_mentioned_group_bindings(text)
            alias_grains = self._requested_group_grains(text)
            if alias_grains:
                by_alias = {item["grain"]: item for item in bindings}
                bindings = [
                    by_alias.get(grain, {"grain": grain, "group_column_hint": None})
                    for grain in alias_grains
                ]
            for binding in bindings:
                grain = str(binding["grain"])
                # Repeated prediction and optimization nodes for the same
                # numbered question share a text contract and are merged.
                key = (grain, re.sub(r"\s+", "", text))
                record = requests.setdefault(key, {
                    "grain": grain, "text": text, "task_ids": [],
                    "group_column_hint": binding.get("group_column_hint"),
                })
                task_id = str(node.get("id") or "")
                if task_id and task_id not in record["task_ids"]:
                    record["task_ids"].append(task_id)
        if not requests and "prediction_forecast" in {
            str(item.get("task_type"))
            for item in problem_analysis.get("task_candidates", [])
        }:
            bindings = self._discover_mentioned_group_bindings(problem)
            alias_grains = self._requested_group_grains(problem)
            if alias_grains:
                by_alias = {item["grain"]: item for item in bindings}
                bindings = [
                    by_alias.get(grain, {"grain": grain, "group_column_hint": None})
                    for grain in alias_grains
                ]
            for binding in bindings:
                grain = str(binding["grain"])
                requests[(grain, re.sub(r"\s+", "", problem))] = {
                    "grain": grain, "text": problem, "task_ids": [],
                    "group_column_hint": binding.get("group_column_hint"),
                }
        ordered = list(requests.values())
        ordered.sort(key=lambda item: (
            int((item.get("group_column_hint") or {}).get("unique_count", 10**9)),
            0 if item["grain"] == "category" else 1,
            item["text"],
        ))
        return ordered

    def _point_in_time_join(
        self,
        base: pd.DataFrame,
        other: pd.DataFrame,
        base_keys: Sequence[str],
        other_keys: Sequence[str],
        base_time_column: str,
        other_time_column: str,
        numeric: Sequence[str],
        other_name: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Attach cumulative features using only records available at each base time."""
        left = pd.DataFrame({
            "__pit_key": _compose_key(base, base_keys),
            "__pit_time": pd.to_datetime(base[base_time_column], errors="coerce"),
            "__pit_row": np.arange(len(base), dtype=np.int64),
        })
        right = other[list(dict.fromkeys(list(other_keys) + [other_time_column] + list(numeric)))].copy()
        right["__pit_key"] = _compose_key(right, other_keys)
        right["__pit_time"] = pd.to_datetime(right[other_time_column], errors="coerce")
        right = right[(right["__pit_key"] != "") & right["__pit_time"].notna()]
        valid_left = left[(left["__pit_key"] != "") & left["__pit_time"].notna()].copy()
        feature_columns: List[str] = []
        snapshots: List[pd.DataFrame] = []
        for column in numeric:
            values = pd.to_numeric(right[column], errors="coerce")
            moments = pd.DataFrame({
                "__pit_key": right["__pit_key"],
                "__pit_time": right["__pit_time"],
                "__value": values,
                "__square": values * values,
            }).dropna(subset=["__value"])
            if moments.empty:
                continue
            grouped = moments.groupby(["__pit_key", "__pit_time"], sort=False).agg(
                __sum=("__value", "sum"),
                __count=("__value", "count"),
                __sum_square=("__square", "sum"),
            ).reset_index().sort_values(["__pit_key", "__pit_time"])
            for moment in ("__sum", "__count", "__sum_square"):
                grouped[moment] = grouped.groupby("__pit_key", sort=False)[moment].cumsum()
            count = grouped["__count"].astype(float)
            total = grouped["__sum"].astype(float)
            variance = (grouped["__sum_square"] - total * total / count.clip(lower=1)) / (count - 1).clip(lower=1)
            prefix = f"{other_name}__{column}"
            grouped[f"{prefix}__mean_asof"] = total / count.clip(lower=1)
            grouped[f"{prefix}__std_asof"] = np.sqrt(variance.clip(lower=0)).where(count > 1)
            grouped[f"{prefix}__sum_asof"] = total
            grouped[f"{prefix}__count_asof"] = count
            current_features = [
                f"{prefix}__mean_asof", f"{prefix}__std_asof",
                f"{prefix}__sum_asof", f"{prefix}__count_asof",
            ]
            feature_columns.extend(current_features)
            snapshots.append(grouped[["__pit_key", "__pit_time"] + current_features].set_index(["__pit_key", "__pit_time"]))
        if not snapshots or valid_left.empty:
            return base, {
                "dataset": other_name,
                "strategy": "point_in_time",
                "features_added": 0,
                "reason": "没有有效的历史时间记录",
            }
        snapshot = pd.concat(snapshots, axis=1).reset_index()
        snapshot = snapshot.sort_values(["__pit_time", "__pit_key"])
        valid_left = valid_left.sort_values(["__pit_time", "__pit_key"])
        matched = pd.merge_asof(
            valid_left,
            snapshot,
            on="__pit_time",
            by="__pit_key",
            direction="backward",
            allow_exact_matches=True,
        )
        matched = matched.set_index("__pit_row")
        output = base.copy()
        for column in feature_columns:
            output[column] = pd.Series(matched[column], index=matched.index).reindex(range(len(output))).to_numpy()
        return output, {
            "dataset": other_name,
            "strategy": "point_in_time",
            "base_time_column": base_time_column,
            "source_time_column": other_time_column,
            "features_added": len(feature_columns),
            "matched_rows": int(matched[feature_columns].notna().any(axis=1).sum()),
        }

    def _build_modeling_view(
        self,
        dataset_name: str,
        target: str,
        temporal: bool = False,
        base_time_column: Optional[str] = None,
    ) -> pd.DataFrame:
        base = _sample_frame(self._datasets[dataset_name], self.max_analysis_rows, self.random_state).copy()
        # Add aggregate features from every directly connected table.  This keeps one
        # row per base observation regardless of relation cardinality.
        relations = self.discover_relationships()
        best_by_other: Dict[str, DatasetRelation] = {}

        def relation_priority(relation: DatasetRelation, other_name: str) -> Tuple[int, float, float]:
            if relation.left_dataset == dataset_name:
                base_relation_keys = relation.left_keys or [relation.left_key]
                other_relation_keys = relation.right_keys or [relation.right_key]
            else:
                base_relation_keys = relation.right_keys or [relation.right_key]
                other_relation_keys = relation.left_keys or [relation.left_key]
            temporal_compatible = True
            if temporal and base_time_column:
                consumes_base_time = base_time_column in base_relation_keys
                available_other_times = [
                    column for column in self._profiles[other_name].datetime_columns
                    if column not in other_relation_keys
                ]
                other_is_unique = (
                    relation.relationship == "one_to_one"
                    or (relation.left_dataset == other_name and relation.relationship == "one_to_many")
                    or (relation.right_dataset == other_name and relation.relationship == "many_to_one")
                )
                temporal_compatible = not consumes_base_time and bool(available_other_times or other_is_unique)
            return int(temporal_compatible), relation.confidence, relation.value_overlap

        for relation in relations:
            if dataset_name not in {relation.left_dataset, relation.right_dataset}:
                continue
            other = relation.right_dataset if relation.left_dataset == dataset_name else relation.left_dataset
            if (
                other not in best_by_other
                or relation_priority(relation, other) > relation_priority(best_by_other[other], other)
            ):
                best_by_other[other] = relation
        prioritized_relations = sorted(
            best_by_other.items(),
            key=lambda item: (item[1].confidence, item[1].value_overlap),
            reverse=True,
        )
        if len(prioritized_relations) > self.max_modeling_relations:
            self._runtime_warnings.append(
                f"{dataset_name} 可连接 {len(prioritized_relations)} 个数据集；建模视图仅纳入证据最强的 "
                f"{self.max_modeling_relations} 个，避免特征联接失控。"
            )
            prioritized_relations = prioritized_relations[:self.max_modeling_relations]
        join_audit: List[Dict[str, Any]] = []
        for other_name, relation in prioritized_relations:
            if relation.left_dataset == dataset_name:
                base_keys = relation.left_keys or [relation.left_key]
                other_keys = relation.right_keys or [relation.right_key]
            else:
                base_keys = relation.right_keys or [relation.right_key]
                other_keys = relation.left_keys or [relation.left_key]
            other = _sample_frame(self._datasets[other_name], self.max_analysis_rows, self.random_state)
            other_time_columns = [
                column for column in self._profiles[other_name].datetime_columns
                if column in other.columns and column not in other_keys
            ]
            numeric = self._top_numeric(other, list(other_keys) + other_time_columns)
            if not numeric or any(key not in base.columns for key in base_keys):
                continue
            if temporal and base_time_column and other_time_columns:
                base, audit = self._point_in_time_join(
                    base, other, base_keys, other_keys, base_time_column,
                    other_time_columns[0], numeric, other_name,
                )
                join_audit.append(audit)
                continue
            other_is_unique = (
                relation.relationship == "one_to_one"
                or (relation.left_dataset == other_name and relation.relationship == "one_to_many")
                or (relation.right_dataset == other_name and relation.relationship == "many_to_one")
            )
            if temporal and not other_is_unique:
                join_audit.append({
                    "dataset": other_name,
                    "strategy": "skipped",
                    "features_added": 0,
                    "reason": "时序任务中的明细表缺少可用时间列，静态全量聚合会泄漏未来信息",
                })
                self._runtime_warnings.append(
                    f"时序建模已跳过 {other_name}：该明细表没有可用时间列，无法执行 point-in-time 联接。"
                )
                continue
            work = other[list(other_keys) + numeric].copy()
            work["__join_key"] = _compose_key(work, other_keys)
            aggregations = work.groupby("__join_key", sort=False)[numeric].agg(["mean", "std", "sum", "count"])
            aggregations.columns = [f"{other_name}__{column}__{stat}" for column, stat in aggregations.columns]
            join_column = "__assistant_join_key"
            while join_column in base.columns:
                join_column = "_" + join_column
            base[join_column] = _compose_key(base, base_keys)
            base = base.join(aggregations, on=join_column)
            base = base.drop(columns=[join_column])
            join_audit.append({
                "dataset": other_name,
                "strategy": "static_dimension" if temporal else "aggregate",
                "features_added": len(aggregations.columns),
            })
        base.attrs["feature_join_audit"] = join_audit
        return base

    @staticmethod
    def _is_identifier_name(column: str) -> bool:
        """Return True only for explicit entity/key names, not broad substrings."""
        return _is_explicit_identifier_name(column)

    def _select_validation_group(self, frame: pd.DataFrame, target: str) -> Optional[str]:
        """Select a repeated entity key whose rows must stay in the same fold."""
        candidates: List[Tuple[float, str]] = []
        n_rows = len(frame)
        if n_rows < 30:
            return None
        for column in frame.columns:
            if column == target or not self._is_identifier_name(str(column)):
                continue
            series = frame[column].dropna()
            unique = _safe_nunique(series)
            if unique < 3 or unique > max(3, int(n_rows * 0.8)):
                continue
            counts = series.map(_normalise_value).value_counts()
            if counts.empty or int(counts.max()) < 2:
                continue
            coverage = len(series) / max(n_rows, 1)
            # Prefer a well-populated key with many independent entities.
            candidates.append((coverage + unique / max(n_rows, 1), str(column)))
        return max(candidates)[1] if candidates else None

    @staticmethod
    def _credibility_check(
        check_id: str,
        name: str,
        status: str,
        evidence: str,
        recommendation: str = "",
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "id": check_id,
            "name": name,
            "status": status,
            "evidence": evidence,
            "recommendation": recommendation,
            "details": details or {},
        }

    def _validation_protocol_check(
        self,
        validation: str,
        n_evaluation: int,
        time_ordered: bool,
        group_column: Optional[str],
        group_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        if time_ordered and "temporal" not in validation and "time_ordered" not in validation:
            return self._credibility_check(
                "validation_protocol", "验证隔离", "fail",
                "题目属于时序任务，但结果没有采用按时间向前的验证。",
                "按时间排序并使用未来留出或滚动验证，禁止随机打乱。",
            )
        if group_column and not time_ordered and "group" not in validation:
            return self._credibility_check(
                "validation_protocol", "验证隔离", "fail",
                f"检测到重复实体键 {group_column}，但验证没有隔离同一实体。",
                "使用 GroupKFold 或按实体留出，防止同一对象同时出现在训练和验证中。",
            )
        if group_column and group_overlap is not None and group_overlap > 0:
            return self._credibility_check(
                "validation_protocol", "验证隔离", "fail",
                f"开发集与确认集仍共享 {group_overlap} 个 {group_column} 实体。",
                "重新按实体键整体切分，任何实体都不能跨越训练和最终确认集。",
            )
        independent = "holdout_after_inner_cv" in validation
        if independent and n_evaluation >= 30:
            method = "未来末段留出" if time_ordered else ("实体隔离留出" if group_column else "独立随机/分层留出")
            overlap_text = f"，实体重叠数={group_overlap}" if group_column and group_overlap is not None else ""
            return self._credibility_check(
                "validation_protocol", "验证隔离", "pass",
                f"采用{method}，共有 {n_evaluation} 个未参与选参的确认样本{overlap_text}。",
                details={"validation": validation, "evaluation_samples": n_evaluation,
                         "group_overlap": group_overlap},
            )
        if independent:
            return self._credibility_check(
                "validation_protocol", "验证隔离", "warning",
                f"已使用独立确认集，但只有 {n_evaluation} 个样本，区间可能较宽。",
                "增加确认样本，正式结论建议至少覆盖 30 个且包含关键群体。",
                {"validation": validation, "evaluation_samples": n_evaluation},
            )
        return self._credibility_check(
            "validation_protocol", "验证隔离", "warning",
            f"指标来自 {validation}，尚无独立的最终确认集。",
            "保留一份完全不参与模型和参数选择的数据作最终确认。",
            {"validation": validation, "evaluation_samples": n_evaluation},
        )

    def _target_leakage_check(
        self,
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        target: str,
        use_time_validation: bool,
        join_audit: Sequence[Dict[str, Any]],
        group_column: Optional[str],
    ) -> Dict[str, Any]:
        if X_fit.empty:
            return self._credibility_check(
                "target_leakage", "目标泄漏", "not_assessed", "没有可审计的训练特征。"
            )
        limit = min(len(X_fit), self.credibility_max_rows)
        rng = np.random.default_rng(self.random_state + 101)
        positions = (
            np.arange(len(X_fit)) if len(X_fit) <= limit
            else np.sort(rng.choice(len(X_fit), size=limit, replace=False))
        )
        X_sample = X_fit.iloc[positions]
        y_sample = pd.Series(y_fit).iloc[positions].reset_index(drop=True)
        target_norm = _normalise_name(target)
        findings: List[Dict[str, Any]] = []
        columns = list(X_sample.columns)
        columns.sort(
            key=lambda column: (
                target_norm not in _normalise_name(column) if len(target_norm) >= 3 else True,
                not pd.api.types.is_numeric_dtype(X_sample[column]),
            )
        )
        for column in columns[:80]:
            if column == group_column:
                continue
            feature = X_sample[column].reset_index(drop=True)
            name_match = bool(len(target_norm) >= 3 and target_norm in _normalise_name(column))
            exact_rate = 0.0
            try:
                left = feature.map(_normalise_value)
                right = y_sample.map(_normalise_value)
                valid = left.ne("") & right.ne("")
                if int(valid.sum()) >= 20:
                    exact_rate = float((left[valid] == right[valid]).mean())
            except Exception:
                pass
            correlation = None
            if pd.api.types.is_numeric_dtype(feature):
                numeric = pd.DataFrame({
                    "feature": pd.to_numeric(feature, errors="coerce"),
                    "target": pd.to_numeric(y_sample, errors="coerce"),
                }).replace([np.inf, -np.inf], np.nan).dropna()
                if len(numeric) >= 20 and numeric["feature"].nunique() > 1 and numeric["target"].nunique() > 1:
                    value = numeric["feature"].corr(numeric["target"], method="spearman")
                    correlation = float(value) if np.isfinite(value) else None
            severity = None
            reason = None
            if exact_rate >= 0.995:
                severity, reason = "fail", "特征几乎逐行复制目标"
            elif name_match and correlation is not None and abs(correlation) >= 0.98:
                severity, reason = "fail", "字段名指向目标且与目标近乎完全相关"
            elif name_match:
                severity, reason = "warning", "字段名可能是目标的事后变量或代理变量"
            elif correlation is not None and abs(correlation) >= 0.995:
                severity, reason = "warning", "与目标近乎完全单调相关，需核查产生时间"
            if severity:
                findings.append({
                    "feature": str(column), "severity": severity, "reason": reason,
                    "exact_match_rate": exact_rate,
                    "absolute_spearman": abs(correlation) if correlation is not None else None,
                })
        unsafe_temporal = [
            item for item in join_audit
            if use_time_validation and item.get("strategy") not in {"point_in_time", "static_dimension", "skipped"}
        ]
        if unsafe_temporal:
            findings.append({
                "feature": "跨表特征", "severity": "fail",
                "reason": "时序任务使用了未通过时点审计的跨表特征",
            })
        status = "pass"
        if any(item["severity"] == "fail" for item in findings):
            status = "fail"
        elif findings:
            status = "warning"
        evidence = (
            "未发现目标复制、可疑目标代理或不安全的未来跨表特征。"
            if not findings else
            "；".join(f"{item['feature']}：{item['reason']}" for item in findings[:5])
        )
        recommendation = "" if status == "pass" else "逐项确认字段的生成时点；无法证明预测时可获得的字段必须删除后重跑。"
        return self._credibility_check(
            "target_leakage", "目标泄漏", status, evidence, recommendation,
            {"suspected_features": findings[:20], "audited_features": min(len(columns), 80)},
        )

    def _baseline_check(
        self,
        actual: np.ndarray,
        prediction: np.ndarray,
        y_fit: pd.Series,
        task: Any,
        use_time_validation: bool,
        independent_holdout: bool,
    ) -> Dict[str, Any]:
        task_value = getattr(task, "value", task)
        if len(actual) < 8 or len(actual) != len(prediction):
            return self._credibility_check(
                "naive_baseline", "简单基线", "not_assessed", "验证预测与真实值不足或长度不一致。"
            )
        if task_value == "regression":
            actual_numeric = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
            predicted_numeric = pd.to_numeric(pd.Series(prediction), errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(actual_numeric) & np.isfinite(predicted_numeric)
            actual_numeric, predicted_numeric = actual_numeric[valid], predicted_numeric[valid]
            training_numeric = pd.to_numeric(pd.Series(y_fit), errors="coerce").dropna().to_numpy(dtype=float)
            if len(actual_numeric) < 8 or not len(training_numeric):
                return self._credibility_check(
                    "naive_baseline", "简单基线", "not_assessed", "回归验证值不足。"
                )
            if use_time_validation and independent_holdout:
                baseline_value = float(training_numeric[-1])
                baseline_name = "训练末值持久性基线"
            else:
                # Validation median is an intentionally strong constant baseline.
                baseline_value = float(np.median(actual_numeric))
                baseline_name = "最优常数中位数基线"
            model_loss = float(np.sqrt(np.mean((actual_numeric - predicted_numeric) ** 2)))
            baseline_loss = float(np.sqrt(np.mean((actual_numeric - baseline_value) ** 2)))
            relative_gain = (baseline_loss - model_loss) / max(abs(baseline_loss), 1e-12)
            model_score, baseline_score, metric = model_loss, baseline_loss, "rmse"
        else:
            from sklearn.metrics import f1_score
            actual_values = np.asarray(actual)
            predicted_values = np.asarray(prediction)
            majority = pd.Series(actual_values).value_counts(dropna=False).index[0]
            baseline_prediction = np.repeat(majority, len(actual_values))
            model_score = float(f1_score(actual_values, predicted_values, average="weighted", zero_division=0))
            baseline_score = float(f1_score(actual_values, baseline_prediction, average="weighted", zero_division=0))
            relative_gain = (model_score - baseline_score) / max(abs(baseline_score), 1e-12)
            baseline_name, metric = "多数类基线", "f1_weighted"
        status = "pass" if relative_gain > 0.02 else ("warning" if relative_gain > 0 else "fail")
        evidence = (
            f"模型 {metric}={model_score:.4g}，{baseline_name}={baseline_score:.4g}，"
            f"相对改善 {relative_gain:.1%}。"
        )
        recommendation = "" if status == "pass" else "模型未稳定超过简单规则，不应将复杂模型结果作为主要结论。"
        return self._credibility_check(
            "naive_baseline", "简单基线", status, evidence, recommendation,
            {"metric": metric, "model": model_score, "baseline": baseline_score, "relative_gain": relative_gain},
        )

    def _prediction_permutation_check(
        self,
        actual: np.ndarray,
        prediction: np.ndarray,
        task: Any,
    ) -> Dict[str, Any]:
        if len(actual) < 20 or len(actual) != len(prediction):
            return self._credibility_check(
                "prediction_permutation", "验证结果置乱", "not_assessed",
                "至少需要 20 个对齐的验证样本。",
            )
        limit = min(len(actual), self.credibility_max_rows)
        rng = np.random.default_rng(self.random_state + 211)
        positions = (
            np.arange(len(actual)) if len(actual) <= limit
            else np.sort(rng.choice(len(actual), size=limit, replace=False))
        )
        actual_values = np.asarray(actual)[positions]
        predicted_values = np.asarray(prediction)[positions]
        task_value = getattr(task, "value", task)
        null_scores: List[float] = []
        if task_value == "regression":
            actual_values = pd.to_numeric(pd.Series(actual_values), errors="coerce").to_numpy(dtype=float)
            predicted_values = pd.to_numeric(pd.Series(predicted_values), errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
            actual_values, predicted_values = actual_values[valid], predicted_values[valid]
            if len(actual_values) < 20:
                return self._credibility_check(
                    "prediction_permutation", "验证结果置乱", "not_assessed", "有限回归验证值不足 20 个。"
                )
            observed = float(np.sqrt(np.mean((actual_values - predicted_values) ** 2)))
            for _ in range(self.credibility_iterations):
                shuffled = rng.permutation(actual_values)
                null_scores.append(float(np.sqrt(np.mean((shuffled - predicted_values) ** 2))))
            p_value = (1 + sum(score <= observed for score in null_scores)) / (len(null_scores) + 1)
            null_boundary = float(np.quantile(null_scores, 0.05))
            metric, direction = "rmse", "低于"
        else:
            from sklearn.metrics import f1_score
            observed = float(f1_score(actual_values, predicted_values, average="weighted", zero_division=0))
            for _ in range(self.credibility_iterations):
                shuffled = rng.permutation(actual_values)
                null_scores.append(float(f1_score(shuffled, predicted_values, average="weighted", zero_division=0)))
            p_value = (1 + sum(score >= observed for score in null_scores)) / (len(null_scores) + 1)
            null_boundary = float(np.quantile(null_scores, 0.95))
            metric, direction = "f1_weighted", "高于"
        status = "pass" if p_value <= 0.05 else ("warning" if p_value <= 0.1 else "fail")
        evidence = (
            f"真实对齐 {metric}={observed:.4g}，随机置乱边界={null_boundary:.4g}；"
            f"置乱 p={p_value:.4g}，真实预测应显著{direction}随机对齐。"
        )
        recommendation = "" if status == "pass" else "当前预测与真实结果的对齐未显著强于随机，应降级为探索性结果。"
        return self._credibility_check(
            "prediction_permutation", "验证结果置乱", status, evidence, recommendation,
            {"metric": metric, "observed": observed, "null_boundary": null_boundary,
             "p_value": p_value, "iterations": len(null_scores)},
        )

    def _uncertainty_check(
        self,
        actual: np.ndarray,
        prediction: np.ndarray,
        task: Any,
    ) -> Dict[str, Any]:
        if len(actual) < 20 or len(actual) != len(prediction):
            return self._credibility_check(
                "metric_uncertainty", "指标不确定性", "not_assessed", "验证样本不足 20 个。"
            )
        rng = np.random.default_rng(self.random_state + 307)
        actual_values = np.asarray(actual)
        predicted_values = np.asarray(prediction)
        if len(actual_values) > self.credibility_max_rows:
            positions = np.sort(rng.choice(len(actual_values), self.credibility_max_rows, replace=False))
            actual_values, predicted_values = actual_values[positions], predicted_values[positions]
        scores: List[float] = []
        task_value = getattr(task, "value", task)
        if task_value == "regression":
            actual_values = pd.to_numeric(pd.Series(actual_values), errors="coerce").to_numpy(dtype=float)
            predicted_values = pd.to_numeric(pd.Series(predicted_values), errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
            actual_values, predicted_values = actual_values[valid], predicted_values[valid]
            for _ in range(self.credibility_iterations):
                index = rng.integers(0, len(actual_values), len(actual_values))
                scores.append(float(np.sqrt(np.mean((actual_values[index] - predicted_values[index]) ** 2))))
            observed = float(np.sqrt(np.mean((actual_values - predicted_values) ** 2)))
            metric = "rmse"
        else:
            from sklearn.metrics import f1_score
            for _ in range(self.credibility_iterations):
                index = rng.integers(0, len(actual_values), len(actual_values))
                scores.append(float(f1_score(
                    actual_values[index], predicted_values[index], average="weighted", zero_division=0
                )))
            observed = float(f1_score(actual_values, predicted_values, average="weighted", zero_division=0))
            metric = "f1_weighted"
        lower, upper = (float(value) for value in np.quantile(scores, [0.025, 0.975]))
        relative_width = (upper - lower) / max(abs(observed), 1e-12)
        status = "pass" if relative_width <= 0.25 else ("warning" if relative_width <= 0.6 else "fail")
        evidence = f"{metric}={observed:.4g}，bootstrap 95% 区间 [{lower:.4g}, {upper:.4g}]。"
        return self._credibility_check(
            "metric_uncertainty", "指标不确定性", status, evidence,
            "扩大独立验证样本并报告区间，避免只引用单点分数。" if status != "pass" else "",
            {"metric": metric, "observed": observed, "confidence_interval_95": [lower, upper],
             "relative_width": relative_width, "iterations": len(scores)},
        )

    def _fold_stability_check(self, diagnostics: Mapping[str, Any]) -> Dict[str, Any]:
        relative_std = diagnostics.get("fold_relative_std")
        if relative_std is None or not np.isfinite(float(relative_std)):
            return self._credibility_check(
                "fold_stability", "跨折稳定性", "not_assessed", "没有可用的跨折波动数据。"
            )
        relative_std = float(relative_std)
        status = "pass" if relative_std <= 0.1 else ("warning" if relative_std <= 0.2 else "fail")
        return self._credibility_check(
            "fold_stability", "跨折稳定性", status,
            f"主指标的折间标准差/均值为 {relative_std:.1%}。",
            "检查分布漂移、异常折和分组方式，优先选择跨折更稳定的模型。" if status != "pass" else "",
            {"relative_std": relative_std, "primary_metric": diagnostics.get("primary_metric")},
        )

    def _subgroup_error_check(
        self,
        X_evaluation: pd.DataFrame,
        actual: np.ndarray,
        prediction: np.ndarray,
        task: Any,
        group_column: Optional[str],
    ) -> Dict[str, Any]:
        if X_evaluation is None or len(X_evaluation) != len(actual) or len(actual) < 30:
            return self._credibility_check(
                "subgroup_error", "分群误差", "not_assessed", "没有足够且逐行对齐的验证特征。"
            )
        frame = X_evaluation.reset_index(drop=True)
        actual_values = np.asarray(actual)
        predicted_values = np.asarray(prediction)
        task_value = getattr(task, "value", task)
        if task_value == "regression":
            actual_numeric = pd.to_numeric(pd.Series(actual_values), errors="coerce").to_numpy(dtype=float)
            predicted_numeric = pd.to_numeric(pd.Series(predicted_values), errors="coerce").to_numpy(dtype=float)
            row_error = np.abs(actual_numeric - predicted_numeric)
        else:
            row_error = (actual_values != predicted_values).astype(float)
        finite = np.isfinite(row_error)
        overall = float(np.mean(row_error[finite])) if finite.any() else float("nan")
        candidates: List[Tuple[str, pd.Series]] = []
        for column in frame.columns:
            if column == group_column:
                continue
            series = frame[column]
            unique = _safe_nunique(series)
            if 2 <= unique <= 12:
                candidates.append((str(column), series.fillna("__MISSING__").astype(str)))
            elif pd.api.types.is_numeric_dtype(series) and unique >= 20 and len(candidates) < 8:
                try:
                    candidates.append((f"{column}（分位组）", pd.qcut(series, 4, duplicates="drop").astype(str)))
                except (TypeError, ValueError):
                    pass
            if len(candidates) >= 8:
                break
        worst: Optional[Dict[str, Any]] = None
        minimum_group = max(10, int(len(frame) * 0.02))
        for column, labels in candidates:
            work = pd.DataFrame({"group": labels, "error": row_error})
            stats = work.groupby("group", dropna=False)["error"].agg(["mean", "size"])
            stats = stats[stats["size"] >= minimum_group]
            if len(stats) < 2:
                continue
            label = stats["mean"].idxmax()
            group_error = float(stats.loc[label, "mean"])
            disparity = group_error / max(overall, 1e-12) if overall > 0 else (1.0 if group_error == 0 else float("inf"))
            item = {
                "feature": column, "group": str(label), "group_error": group_error,
                "overall_error": overall, "error_ratio": disparity,
                "group_samples": int(stats.loc[label, "size"]),
            }
            if worst is None or item["error_ratio"] > worst["error_ratio"]:
                worst = item
        if worst is None:
            return self._credibility_check(
                "subgroup_error", "分群误差", "not_assessed", "没有样本量足够的可比分群。"
            )
        disparity = float(worst["error_ratio"])
        material_gap = (
            worst["group_error"] - overall > 0.15
            if task_value != "regression"
            else disparity > 2.5
        )
        status = "fail" if disparity > 2.5 and material_gap else ("warning" if disparity > 1.5 else "pass")
        metric_name = "MAE" if task_value == "regression" else "错误率"
        evidence = (
            f"最弱群体为 {worst['feature']}={worst['group']}：{metric_name}={worst['group_error']:.4g}，"
            f"总体={overall:.4g}，约为总体的 {disparity:.2f} 倍。"
        )
        return self._credibility_check(
            "subgroup_error", "分群误差", status, evidence,
            "单独报告弱势群体表现，补充样本或建立分群模型。" if status != "pass" else "", worst,
        )

    def _distribution_shift_check(
        self,
        X_fit: pd.DataFrame,
        X_evaluation: Optional[pd.DataFrame],
        use_time_validation: bool,
        group_column: Optional[str],
    ) -> Dict[str, Any]:
        if X_evaluation is None or X_evaluation.empty:
            return self._credibility_check(
                "distribution_shift", "分布漂移", "not_assessed", "没有独立确认特征可与开发集比较。"
            )
        train = _sample_frame(X_fit, self.credibility_max_rows, self.random_state)
        evaluation = _sample_frame(X_evaluation, self.credibility_max_rows, self.random_state + 1)
        shifts: List[Dict[str, Any]] = []
        for column in X_fit.columns[:80]:
            if column == group_column or column not in evaluation.columns:
                continue
            left, right = train[column], evaluation[column]
            if pd.api.types.is_numeric_dtype(left):
                left_numeric = pd.to_numeric(left, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                right_numeric = pd.to_numeric(right, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
                if len(left_numeric) >= 20 and len(right_numeric) >= 10:
                    pooled = math.sqrt((float(left_numeric.var()) + float(right_numeric.var())) / 2)
                    shift = abs(float(left_numeric.mean()) - float(right_numeric.mean())) / max(pooled, 1e-12)
                    shifts.append({"feature": str(column), "method": "standardized_mean_difference", "value": shift})
            elif _safe_nunique(left) <= 30:
                left_freq = left.fillna("__MISSING__").astype(str).value_counts(normalize=True)
                right_freq = right.fillna("__MISSING__").astype(str).value_counts(normalize=True)
                categories = left_freq.index.union(right_freq.index)
                shift = 0.5 * float(sum(abs(left_freq.get(key, 0.0) - right_freq.get(key, 0.0)) for key in categories))
                shifts.append({"feature": str(column), "method": "total_variation", "value": shift})
        if not shifts:
            return self._credibility_check(
                "distribution_shift", "分布漂移", "not_assessed", "没有可稳定比较的特征分布。"
            )
        shifts.sort(key=lambda item: item["value"], reverse=True)
        maximum = float(shifts[0]["value"])
        if maximum <= 0.25:
            status = "pass"
        elif maximum <= 0.8 or use_time_validation:
            status = "warning"
        else:
            status = "fail"
        evidence = f"最大漂移来自 {shifts[0]['feature']}，{shifts[0]['method']}={maximum:.4g}。"
        if use_time_validation and maximum > 0.25:
            evidence += " 时序留出天然允许时间变化，但外推风险已提高。"
        return self._credibility_check(
            "distribution_shift", "分布漂移", status, evidence,
            "定位漂移特征，采用滚动更新、重加权或限制结论适用范围。" if status != "pass" else "",
            {"largest_shifts": shifts[:10]},
        )

    def _input_sensitivity_check(
        self,
        engine: Any,
        X_evaluation: Optional[pd.DataFrame],
        actual: np.ndarray,
        task: Any,
        group_column: Optional[str],
    ) -> Dict[str, Any]:
        if engine is None or X_evaluation is None or X_evaluation.empty:
            return self._credibility_check(
                "input_sensitivity", "输入扰动敏感性", "not_assessed", "没有可调用的最终模型或验证特征。"
            )
        numeric = [
            column for column in X_evaluation.columns
            if column != group_column and pd.api.types.is_numeric_dtype(X_evaluation[column])
            and X_evaluation[column].nunique(dropna=True) > 2
        ][:12]
        if not numeric:
            return self._credibility_check(
                "input_sensitivity", "输入扰动敏感性", "not_assessed", "没有适合施加小扰动的连续特征。"
            )
        sample = _sample_frame(X_evaluation, min(300, len(X_evaluation)), self.random_state + 17).copy()
        perturbed = sample.copy()
        rng = np.random.default_rng(self.random_state + 401)
        for column in numeric:
            values = pd.to_numeric(sample[column], errors="coerce")
            scale = float(values.std())
            if np.isfinite(scale) and scale > 0:
                perturbed[column] = values + rng.normal(0.0, 0.01 * scale, len(values))
        try:
            original_prediction = np.asarray(engine.predict(sample))
            perturbed_prediction = np.asarray(engine.predict(perturbed))
        except Exception as exc:
            return self._credibility_check(
                "input_sensitivity", "输入扰动敏感性", "not_assessed", f"扰动复算失败：{exc}"
            )
        task_value = getattr(task, "value", task)
        if task_value == "regression":
            original_numeric = pd.to_numeric(pd.Series(original_prediction), errors="coerce").to_numpy(dtype=float)
            perturbed_numeric = pd.to_numeric(pd.Series(perturbed_prediction), errors="coerce").to_numpy(dtype=float)
            target_scale = float(np.nanstd(pd.to_numeric(pd.Series(actual), errors="coerce")))
            sensitivity = float(np.nanmedian(np.abs(original_numeric - perturbed_numeric)) / max(target_scale, 1e-12))
            measure = "预测变化/目标标准差"
        else:
            sensitivity = float(np.mean(original_prediction != perturbed_prediction))
            measure = "类别翻转率"
        status = "pass" if sensitivity <= 0.02 else ("warning" if sensitivity <= 0.1 else "fail")
        return self._credibility_check(
            "input_sensitivity", "输入扰动敏感性", status,
            f"连续特征加入其标准差 1% 的随机扰动后，{measure}={sensitivity:.2%}。",
            "检查决策边界和异常敏感特征，必要时加强正则化或采用稳健变换。" if status != "pass" else "",
            {"sensitivity": sensitivity, "measure": measure, "perturbed_features": numeric,
             "samples": len(sample)},
        )

    def _importance_concentration_check(
        self, feature_importance: Sequence[Dict[str, Any]]
    ) -> Dict[str, Any]:
        values: List[Tuple[str, float]] = []
        for item in feature_importance:
            raw = item.get("importance", item.get("score"))
            try:
                value = abs(float(raw))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value) and value > 0:
                values.append((str(item.get("feature", item.get("column", "-"))), value))
        if not values:
            return self._credibility_check(
                "feature_concentration", "单特征依赖", "not_assessed", "没有统一口径的特征重要性。"
            )
        values.sort(key=lambda item: item[1], reverse=True)
        share = values[0][1] / max(sum(value for _, value in values), 1e-12)
        status = "warning" if share > 0.85 else "pass"
        return self._credibility_check(
            "feature_concentration", "单特征依赖", status,
            f"最重要特征 {values[0][0]} 占已报告总重要性的 {share:.1%}。",
            "删除该特征复算并核查其生成时点，确认结论不是由单一代理变量支撑。" if status != "pass" else "",
            {"top_feature": values[0][0], "top_share": share},
        )

    def _model_hypothesis_check(
        self,
        modeling_result: Any,
        actual: pd.Series,
        task: Any,
        time_ordered: bool,
    ) -> Dict[str, Any]:
        """Audit the Rashomon set: near-optimal models should broadly agree."""
        cv_results = list(getattr(modeling_result, "cv_results", []) or [])
        task_value = getattr(task, "value", task)
        primary_metric = "rmse" if task_value == "regression" else "f1_weighted"
        candidates: List[Dict[str, Any]] = []
        actual_values = np.asarray(actual)
        try:
            from .modeling_engine import ModelLibrary
            specifications = ModelLibrary.get_models(task)
        except Exception:
            specifications = {}
        for result in cv_results:
            score = result.mean_scores.get(primary_metric)
            prediction = getattr(result, "oof_pred", None)
            if score is None or prediction is None or not np.isfinite(float(score)):
                continue
            predicted = np.asarray(prediction)
            candidate_actual = actual_values
            if time_ordered and len(predicted):
                fold_count = len(next(iter(result.fold_scores.values()), []))
                initial_window = max(1, len(predicted) // (fold_count + 1))
                predicted = predicted[initial_window:]
                candidate_actual = candidate_actual[initial_window:]
            if len(predicted) != len(candidate_actual) or not len(predicted):
                continue
            key = str(getattr(result, "model_key", "unknown"))
            spec = specifications.get(key)
            candidates.append({
                "key": key,
                "name": str(getattr(result, "model_name", key)),
                "family": getattr(spec, "category", "unknown") if spec is not None else "unknown",
                "score": float(score),
                "prediction": predicted,
                "actual": candidate_actual,
            })
        if len(candidates) < 2:
            return self._credibility_check(
                "model_hypothesis_consistency", "近优模型一致性", "not_assessed",
                "只有一个模型假设具有可比较的 OOF 预测。",
                "增加不同机理和模型族的候选，而不是只在同一算法内调参。",
                {"candidate_count": len(candidates)},
            )
        scores = [item["score"] for item in candidates]
        best_score = min(scores) if task_value == "regression" else max(scores)
        if task_value == "regression":
            tolerance = max(abs(best_score) * 0.10, 1e-12)
            near = [item for item in candidates if item["score"] <= best_score + tolerance]
        else:
            tolerance = max(abs(best_score) * 0.05, 0.02)
            near = [item for item in candidates if item["score"] >= best_score - tolerance]
        near.sort(key=lambda item: item["score"], reverse=task_value != "regression")
        public_candidates = [
            {key: value for key, value in item.items() if key not in {"prediction", "actual"}}
            for item in near
        ]
        if len(near) < 2:
            return self._credibility_check(
                "model_hypothesis_consistency", "近优模型一致性", "warning",
                f"测试了 {len(candidates)} 个模型，但只有 {near[0]['name']} 落入 10%/5% 近优范围。",
                "补充不同模型族或外部验证，确认结论不是单一算法特有。",
                {"near_optimal_models": public_candidates, "tested_models": len(candidates),
                 "primary_metric": primary_metric, "tolerance": tolerance},
            )
        pairwise: List[Dict[str, Any]] = []
        if task_value == "regression":
            reference_scale = max(
                float(np.nanstd(pd.to_numeric(pd.Series(near[0]["actual"]), errors="coerce"))),
                1e-12,
            )
            for left, right in combinations(near, 2):
                left_prediction = pd.to_numeric(pd.Series(left["prediction"]), errors="coerce")
                right_prediction = pd.to_numeric(pd.Series(right["prediction"]), errors="coerce")
                valid = left_prediction.notna() & right_prediction.notna()
                if int(valid.sum()) < 10:
                    continue
                correlation = left_prediction[valid].corr(right_prediction[valid], method="spearman")
                correlation = float(correlation) if np.isfinite(correlation) else 0.0
                disagreement = float(
                    np.median(np.abs(
                        left_prediction[valid].to_numpy(dtype=float)
                        - right_prediction[valid].to_numpy(dtype=float)
                    )) / reference_scale
                )
                pairwise.append({
                    "left": left["key"], "right": right["key"],
                    "prediction_spearman": correlation,
                    "normalized_median_disagreement": disagreement,
                })
            minimum_agreement = min((item["prediction_spearman"] for item in pairwise), default=0.0)
            maximum_disagreement = max(
                (item["normalized_median_disagreement"] for item in pairwise), default=float("inf")
            )
            if minimum_agreement >= 0.9 and maximum_disagreement <= 0.10:
                status = "pass"
            elif minimum_agreement >= 0.7 and maximum_disagreement <= 0.25:
                status = "warning"
            else:
                status = "fail"
            consistency_text = (
                f"最小预测秩相关={minimum_agreement:.3f}，"
                f"最大归一化中位分歧={maximum_disagreement:.1%}"
            )
            details_metric = {
                "minimum_prediction_spearman": minimum_agreement,
                "maximum_normalized_median_disagreement": maximum_disagreement,
            }
        else:
            for left, right in combinations(near, 2):
                agreement = float(np.mean(np.asarray(left["prediction"]) == np.asarray(right["prediction"])))
                pairwise.append({"left": left["key"], "right": right["key"], "class_agreement": agreement})
            minimum_agreement = min((item["class_agreement"] for item in pairwise), default=0.0)
            status = "pass" if minimum_agreement >= 0.9 else ("warning" if minimum_agreement >= 0.75 else "fail")
            consistency_text = f"最低类别一致率={minimum_agreement:.1%}"
            details_metric = {"minimum_class_agreement": minimum_agreement}
        families = {item["family"] for item in near if item["family"] != "unknown"}
        if status == "pass" and len(families) < 2:
            status = "warning"
        evidence = (
            f"近优集合包含 {len(near)} 个模型、{len(families)} 个模型族；{consistency_text}。"
        )
        recommendation = (
            "" if status == "pass" else
            "若近优模型结论冲突，应报告模型集合范围，补充外部验证，不能只展示排名第一的模型。"
        )
        return self._credibility_check(
            "model_hypothesis_consistency", "近优模型一致性", status,
            evidence, recommendation,
            {
                "near_optimal_models": public_candidates,
                "tested_models": len(candidates),
                "model_families": sorted(families),
                "primary_metric": primary_metric,
                "tolerance": tolerance,
                "pairwise": pairwise[:20],
                **details_metric,
            },
        )

    def _conformal_prediction_summary(
        self,
        modeling_result: Any,
        y_fit: pd.Series,
        task: Any,
        use_time_validation: bool,
        evaluation_actual: Optional[np.ndarray],
        evaluation_prediction: Optional[np.ndarray],
        independent_evaluation: bool,
        alpha: float = 0.10,
    ) -> Optional[Dict[str, Any]]:
        """Build a finite-sample corrected residual interval and audit coverage."""
        task_value = getattr(task, "value", task)
        if task_value != "regression" or evaluation_prediction is None:
            return None
        best_cv = getattr(modeling_result, "best_cv_result", None)
        calibration_prediction = getattr(best_cv, "oof_pred", None)
        if calibration_prediction is None:
            return None
        calibration_prediction = np.asarray(calibration_prediction, dtype=float)
        calibration_actual = pd.to_numeric(pd.Series(y_fit), errors="coerce").to_numpy(dtype=float)
        if use_time_validation and len(calibration_prediction):
            fold_count = len(next(iter(getattr(best_cv, "fold_scores", {}).values()), []))
            initial_window = max(1, len(calibration_prediction) // (fold_count + 1))
            calibration_prediction = calibration_prediction[initial_window:]
            calibration_actual = calibration_actual[initial_window:]
        if len(calibration_prediction) != len(calibration_actual):
            return None
        valid = np.isfinite(calibration_prediction) & np.isfinite(calibration_actual)
        residuals = np.abs(calibration_actual[valid] - calibration_prediction[valid])
        if len(residuals) < 30:
            return None
        quantile_level = min(1.0, math.ceil((len(residuals) + 1) * (1 - alpha)) / len(residuals))
        weighting = "exchangeable_equal_weight"
        if use_time_validation:
            # Fixed recency weights follow the beyond-exchangeability idea. They
            # adapt to gradual drift but do not restore exact i.i.d. coverage.
            weights = np.exp(np.linspace(-3.0, 0.0, len(residuals)))
            order = np.argsort(residuals)
            sorted_residuals = residuals[order]
            cumulative = np.cumsum(weights[order]) / np.sum(weights)
            index = min(int(np.searchsorted(cumulative, 1 - alpha)), len(sorted_residuals) - 1)
            radius = float(sorted_residuals[index])
            weighting = "fixed_recency_weighted_nonexchangeable"
        else:
            try:
                radius = float(np.quantile(residuals, quantile_level, method="higher"))
            except TypeError:
                radius = float(np.quantile(residuals, quantile_level, interpolation="higher"))
        prediction = np.asarray(evaluation_prediction, dtype=float)
        lower = prediction - radius
        upper = prediction + radius
        coverage = None
        if evaluation_actual is not None:
            actual = np.asarray(evaluation_actual, dtype=float)
            if len(actual) == len(prediction):
                finite = np.isfinite(actual) & np.isfinite(prediction)
                if int(finite.sum()):
                    coverage = float(np.mean(
                        (actual[finite] >= lower[finite]) & (actual[finite] <= upper[finite])
                    ))
        target_coverage = 1 - alpha
        if coverage is None:
            status = "not_assessed"
            evidence = "没有可比较的验证预测，无法复核区间覆盖率。"
        elif independent_evaluation:
            status = (
                "pass" if coverage >= target_coverage - 0.02 else (
                    "warning" if coverage >= target_coverage - 0.08 else "fail"
                )
            )
            evidence = (
                f"独立确认集经验覆盖率={coverage:.1%}，目标覆盖率={target_coverage:.1%}。"
            )
        else:
            status = "warning"
            evidence = (
                f"OOF 校准样本内经验覆盖率={coverage:.1%}；没有第二个独立集合验证覆盖率。"
            )
        check = self._credibility_check(
            "prediction_interval_coverage", "预测区间覆盖率", status,
            evidence,
            "扩大独立校准/确认集，或在分布漂移下使用带权保序方法并报告覆盖缺口。"
            if status != "pass" else "",
            {
                "target_coverage": target_coverage,
                "empirical_coverage": coverage,
                "independent_evaluation": independent_evaluation,
                "weighting": weighting,
            },
        )
        scale = max(float(np.std(calibration_actual[valid], ddof=1)), 1e-12)
        return {
            "method": "residual_conformal_prediction",
            "alpha": alpha,
            "target_coverage": target_coverage,
            "calibration_samples": len(residuals),
            "radius": radius,
            "mean_interval_width": 2 * radius,
            "normalized_width": 2 * radius / scale,
            "empirical_coverage": coverage,
            "coverage_evaluation": "independent" if independent_evaluation else "oof_reuse",
            "weighting": weighting,
            "credibility_audit": {
                "status": status,
                "label": {"pass": "覆盖通过", "warning": "谨慎使用", "fail": "覆盖失败",
                          "not_assessed": "未评估"}[status],
                "checks": [check],
            },
            "literature_basis": {
                "exchangeable_localization_doi": "10.1093/biomet/asac040",
                "beyond_exchangeability_doi": "10.1214/23-AOS2276",
            },
            "note": (
                "区间保证依赖交换性；时间加权用于适应漂移，但不宣称恢复精确有限样本覆盖。"
            ),
            "_credibility_check": check,
        }

    def _audit_model_credibility(
        self,
        *,
        X_fit: pd.DataFrame,
        y_fit: pd.Series,
        X_evaluation: Optional[pd.DataFrame],
        actual: Optional[np.ndarray],
        prediction: Optional[np.ndarray],
        task: Any,
        target: str,
        validation: str,
        use_time_validation: bool,
        group_column: Optional[str],
        diagnostics: Mapping[str, Any],
        join_audit: Sequence[Dict[str, Any]],
        feature_importance: Sequence[Dict[str, Any]],
        hypothesis_check: Optional[Dict[str, Any]] = None,
        conformal_check: Optional[Dict[str, Any]] = None,
        engine: Any = None,
    ) -> Dict[str, Any]:
        if not self.credibility_audit:
            return {
                "enabled": False, "status": "not_assessed", "label": "未审计",
                "decision": "不能仅凭单点指标判断可信度", "checks": [],
            }
        if actual is None or prediction is None:
            return {
                "enabled": True, "status": "not_assessed", "label": "证据不足",
                "decision": "没有独立或 OOF 预测，不能判断模型结果可信度", "checks": [],
                "limitations": ["训练集拟合分数不能替代未见数据验证。"],
            }
        actual_array = np.asarray(actual)
        prediction_array = np.asarray(prediction)
        independent = "holdout_after_inner_cv" in validation
        group_overlap = None
        if (
            independent and group_column and X_evaluation is not None
            and group_column in X_fit.columns and group_column in X_evaluation.columns
        ):
            fit_groups = set(X_fit[group_column].dropna().map(_normalise_value))
            evaluation_groups = set(X_evaluation[group_column].dropna().map(_normalise_value))
            group_overlap = len(fit_groups & evaluation_groups)
        checks = [
            self._validation_protocol_check(
                validation, len(actual_array), use_time_validation, group_column, group_overlap
            ),
            self._target_leakage_check(
                X_fit, y_fit, target, use_time_validation, join_audit, group_column
            ),
            self._baseline_check(
                actual_array, prediction_array, y_fit, task, use_time_validation, independent
            ),
            self._prediction_permutation_check(actual_array, prediction_array, task),
            self._uncertainty_check(actual_array, prediction_array, task),
            self._fold_stability_check(diagnostics),
            self._subgroup_error_check(
                X_evaluation, actual_array, prediction_array, task, group_column
            ),
            self._distribution_shift_check(
                X_fit, X_evaluation if independent else None, use_time_validation, group_column
            ),
            self._input_sensitivity_check(
                engine, X_evaluation, actual_array, task, group_column
            ),
            self._importance_concentration_check(feature_importance),
        ]
        if hypothesis_check is not None:
            checks.append(hypothesis_check)
        if conformal_check is not None:
            checks.append(conformal_check)
        failed = [item for item in checks if item["status"] == "fail"]
        warned = [item for item in checks if item["status"] == "warning"]
        unavailable = [item for item in checks if item["status"] == "not_assessed"]
        if failed:
            status, label = "fail", "不可信"
            decision = "存在反证或关键验证失败，当前结果不能作为主要结论"
        elif warned or unavailable:
            status, label = "warning", "谨慎使用"
            decision = "结果具有部分证据，但必须连同警告、区间和适用范围一起报告"
        else:
            status, label = "pass", "可信"
            decision = "已通过当前自动反证检查，可作为有边界的模型证据"
        next_actions = [item["recommendation"] for item in failed + warned if item.get("recommendation")]
        summary_parts = []
        if failed:
            summary_parts.append("失败：" + "、".join(item["name"] for item in failed))
        if warned:
            summary_parts.append("警告：" + "、".join(item["name"] for item in warned))
        if unavailable:
            summary_parts.append("未评估：" + "、".join(item["name"] for item in unavailable))
        if not summary_parts:
            summary_parts.append("全部自动审计项通过")
        return _plain({
            "enabled": True,
            "status": status,
            "label": label,
            "decision": decision,
            "summary": "；".join(summary_parts),
            "counts": {
                "pass": sum(item["status"] == "pass" for item in checks),
                "warning": len(warned), "fail": len(failed), "not_assessed": len(unavailable),
            },
            "checks": checks,
            "next_actions": list(dict.fromkeys(next_actions))[:8],
            "limitations": [
                "自动审计只能寻找反证，不能证明统计关联就是因果关系。",
                "验证预测置乱检验衡量未见数据上的结果对齐，仍需字段产生时点审计来识别稳定泄漏。",
                "未提供外部数据时，系统无法验证比赛样本之外的分布。",
            ],
        })

    @staticmethod
    def _validation_diagnostics(
        cv_result: Any,
        actual: pd.Series,
        task: Any,
        time_ordered: bool,
    ) -> Dict[str, Any]:
        primary_metric = "rmse" if getattr(task, "value", task) == "regression" else "f1_weighted"
        primary_mean = float(cv_result.mean_scores.get(primary_metric, 0.0))
        primary_std = float(cv_result.std_scores.get(primary_metric, 0.0))
        diagnostics: Dict[str, Any] = {
            "primary_metric": primary_metric,
            "fold_relative_std": abs(primary_std) / max(abs(primary_mean), 1e-12),
            "recommendations": [],
        }
        predictions = np.asarray(cv_result.oof_pred) if cv_result.oof_pred is not None else np.array([])
        actual_values = np.asarray(actual)
        if time_ordered and len(predictions):
            # TimeSeriesSplit has no OOF prediction for its initial training window.
            initial_window = max(1, len(predictions) // (len(next(iter(cv_result.fold_scores.values()), [])) + 1))
            predictions = predictions[initial_window:]
            actual_values = actual_values[initial_window:]
        if len(predictions) != len(actual_values) or not len(predictions):
            return diagnostics
        if getattr(task, "value", task) == "regression":
            actual_numeric = pd.to_numeric(pd.Series(actual_values), errors="coerce").to_numpy(dtype=float)
            predicted_numeric = pd.to_numeric(pd.Series(predictions), errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(actual_numeric) & np.isfinite(predicted_numeric)
            actual_numeric, predicted_numeric = actual_numeric[valid], predicted_numeric[valid]
            if len(actual_numeric) >= 10:
                residual = actual_numeric - predicted_numeric
                scale = max(float(np.std(actual_numeric)), 1e-12)
                diagnostics.update({
                    "normalized_bias": float(np.mean(residual) / scale),
                    "residual_prediction_correlation": float(pd.Series(residual).corr(pd.Series(predicted_numeric), method="spearman")),
                    "heteroscedasticity_signal": float(pd.Series(np.abs(residual)).corr(pd.Series(predicted_numeric), method="spearman")),
                    "residual_p90_absolute": float(np.quantile(np.abs(residual), 0.9)),
                })
                if abs(diagnostics["normalized_bias"]) > 0.1:
                    diagnostics["recommendations"].append("校正系统性预测偏差")
                if abs(diagnostics["residual_prediction_correlation"]) > 0.15:
                    diagnostics["recommendations"].append("增加模型非线性或正则化搜索")
                if abs(diagnostics["heteroscedasticity_signal"]) > 0.15:
                    diagnostics["recommendations"].append("针对异方差调整树深与叶节点约束")
        else:
            errors = np.asarray(predictions) != np.asarray(actual_values)
            diagnostics["oof_error_rate"] = float(np.mean(errors))
            diagnostics["minority_error_warning"] = bool(diagnostics["oof_error_rate"] > 0.25)
            if diagnostics["minority_error_warning"]:
                diagnostics["recommendations"].append("扩大分类正则化和树复杂度搜索")
        if diagnostics["fold_relative_std"] > 0.1:
            diagnostics["recommendations"].append("优先选择跨折更稳定的参数")
        return _plain(diagnostics)

    def _feedback_optimize_model(
        self,
        baseline_result: Any,
        baseline_engine: Any,
        X: pd.DataFrame,
        y: pd.Series,
        X_confirmation: Optional[pd.DataFrame],
        y_confirmation: Optional[pd.Series],
        task: Any,
        use_time_validation: bool,
        group_column: Optional[str],
        diagnostics: Dict[str, Any],
    ) -> Tuple[Any, Any, Dict[str, Any]]:
        from .modeling_engine import ModelLibrary, ModelingEngine

        baseline_cv = baseline_result.best_cv_result
        primary_metric = "rmse" if getattr(task, "value", task) == "regression" else "f1_weighted"
        inner_baseline_score = (
            float(baseline_cv.mean_scores.get(primary_metric, np.nan))
            if baseline_cv is not None else float("nan")
        )
        feedback: Dict[str, Any] = {
            "enabled": self.feedback_optimization,
            "attempted": False,
            "accepted": False,
            "primary_metric": primary_metric,
            "baseline_model": baseline_result.best_model_key,
            "inner_cv_baseline_score": inner_baseline_score,
            "diagnostics": diagnostics,
        }
        if (
            not self.feedback_optimization
            or baseline_cv is None
            or X_confirmation is None
            or y_confirmation is None
            or len(X_confirmation) < 12
        ):
            feedback["reason"] = "反馈优化已关闭或独立确认集不足 12 个样本"
            return baseline_result, None, feedback
        baseline_prediction = np.asarray(baseline_engine.predict(X_confirmation))
        baseline_metrics = self._confirmation_metrics(y_confirmation, baseline_prediction, task)
        baseline_score = float(baseline_metrics[primary_metric])
        confirmation_method = (
            "末段时间留出" if use_time_validation
            else (f"按实体 {group_column} 隔离留出" if group_column else "独立分层/随机留出")
        )
        feedback.update({
            "baseline_score": baseline_score,
            "confirmation": confirmation_method,
            "confirmation_samples": int(len(y_confirmation)),
            "baseline_confirmation_metrics": baseline_metrics,
            "selected_confirmation_metrics": baseline_metrics,
            "_confirmation_actual": np.asarray(y_confirmation),
            "_confirmation_prediction": baseline_prediction,
        })
        available = ModelLibrary.get_models(task)
        ranked_model_keys = [baseline_result.best_model_key] + [
            result.model_key for result in baseline_result.cv_results
            if result.model_key != baseline_result.best_model_key
        ]
        task_value = getattr(task, "value", task)
        diagnostic_priority: List[str] = []
        candidate_reason = "优先调节当前验证排名最高且有参数空间的模型"
        nonlinear_signal = max(
            abs(float(diagnostics.get("residual_prediction_correlation") or 0.0)),
            abs(float(diagnostics.get("heteroscedasticity_signal") or 0.0)),
        )
        if task_value == "regression" and nonlinear_signal > 0.15:
            diagnostic_priority = ["hist_gb", "lgb", "xgb", "rf", "gbr"]
            candidate_reason = "残差仍有结构性关系，优先反馈调节非线性模型"
        elif task_value == "classification" and diagnostics.get("minority_error_warning"):
            diagnostic_priority = ["hist_gb", "lgb", "xgb", "rf", "lr"]
            candidate_reason = "分类 OOF 错误偏高，优先调节能表达类别边界的模型"
        elif float(diagnostics.get("fold_relative_std") or 0.0) > 0.1:
            diagnostic_priority = ["ridge", "lr", "hist_gb", "rf", "lgb", "xgb"]
            candidate_reason = "跨折波动偏高，优先调节更稳定或正则化更强的模型"
        if diagnostic_priority:
            diagnostic_rank = {key: index for index, key in enumerate(diagnostic_priority)}
            original_rank = {key: index for index, key in enumerate(ranked_model_keys)}
            ranked_model_keys = sorted(
                ranked_model_keys,
                key=lambda key: (
                    diagnostic_rank.get(key, len(diagnostic_priority)),
                    original_rank[key],
                ),
            )
        candidate_keys = [
            key for key in ranked_model_keys
            if key in available and available[key].hyperparam_space
        ][:1]
        if not candidate_keys:
            feedback["reason"] = "当前最佳模型没有可搜索参数空间"
            return baseline_result, None, feedback
        feedback["attempted"] = True
        feedback["candidate_models"] = candidate_keys
        feedback["candidate_selection_reason"] = candidate_reason
        tuned_engine = ModelingEngine(
            task_type=getattr(task, "value", task),
            model_keys=candidate_keys,
            n_splits=3,
            n_jobs=1,
            optimize_hyperparams=True,
            hyperparam_trials=self.feedback_trials,
            optimizer="random",
            explainability=False,
            auto_decision_mode="stability_first" if diagnostics.get("fold_relative_std", 0) > 0.1 else "accuracy_first",
            auto_sample=True,
            max_samples=min(self.max_analysis_rows, 40_000),
            fold_type="time" if use_time_validation else ("group" if group_column else "default"),
            group_col=group_column,
            verbose=False,
            random_state=self.random_state,
        )
        try:
            tuned_result = tuned_engine.fit(X, y)
            tuned_cv = tuned_result.best_cv_result
            tuned_prediction = np.asarray(tuned_engine.predict(X_confirmation))
            tuned_metrics = self._confirmation_metrics(y_confirmation, tuned_prediction, task)
        except Exception as exc:
            feedback["reason"] = f"参数搜索失败，独立确认集仍保留基线：{exc}"
            return baseline_result, None, feedback
        tuned_score = float(tuned_metrics[primary_metric])
        if not np.isfinite(baseline_score) or not np.isfinite(tuned_score):
            feedback["reason"] = "基线或调优指标不是有限数值"
            return baseline_result, None, feedback
        if primary_metric == "rmse":
            relative_gain = (baseline_score - tuned_score) / max(abs(baseline_score), 1e-12)
        else:
            relative_gain = (tuned_score - baseline_score) / max(abs(baseline_score), 1e-12)
        acceptance_threshold = self.feedback_min_relative_gain
        improvement_probability = self._paired_improvement_probability(
            y_confirmation, baseline_prediction, tuned_prediction, task
        )
        accepted = bool(relative_gain > acceptance_threshold and improvement_probability >= 0.8)
        history_summary = {
            key: sorted(history, key=lambda item: item.get("trial", 0))[:50]
            for key, history in (tuned_result.optimization_history or {}).items()
        }
        feedback.update({
            "tuned_model": tuned_result.best_model_key,
            "tuned_score": tuned_score,
            "inner_cv_tuned_score": (
                float(tuned_cv.mean_scores.get(primary_metric, np.nan))
                if tuned_cv is not None else float("nan")
            ),
            "relative_gain": float(relative_gain),
            "acceptance_threshold": float(acceptance_threshold),
            "improvement_probability": float(improvement_probability),
            "accepted": accepted,
            "confirmation": confirmation_method,
            "confirmation_samples": int(len(y_confirmation)),
            "baseline_confirmation_metrics": baseline_metrics,
            "tuned_confirmation_metrics": tuned_metrics,
            "selected_confirmation_metrics": tuned_metrics if accepted else baseline_metrics,
            "_confirmation_actual": np.asarray(y_confirmation),
            "_confirmation_prediction": tuned_prediction if accepted else baseline_prediction,
            "trials_requested": self.feedback_trials,
            "optimized_params": tuned_result.optimized_params or {},
            "optimization_history": history_summary,
            "reason": "独立确认集复核通过" if accepted else "独立确认集未确认稳定收益，保留基线以避免过拟合",
        })
        return (tuned_result, tuned_engine, feedback) if accepted else (baseline_result, None, feedback)

    @staticmethod
    def _confirmation_metrics(actual: pd.Series, prediction: np.ndarray, task: Any) -> Dict[str, float]:
        task_value = getattr(task, "value", task)
        if task_value == "regression":
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            actual_values = pd.to_numeric(pd.Series(actual), errors="coerce").to_numpy(dtype=float)
            predicted_values = pd.to_numeric(pd.Series(prediction), errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(actual_values) & np.isfinite(predicted_values)
            actual_values, predicted_values = actual_values[valid], predicted_values[valid]
            return {
                "rmse": float(math.sqrt(mean_squared_error(actual_values, predicted_values))),
                "mae": float(mean_absolute_error(actual_values, predicted_values)),
                "r2": float(r2_score(actual_values, predicted_values)),
            }
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
        actual_values = np.asarray(actual)
        return {
            "accuracy": float(accuracy_score(actual_values, prediction)),
            "balanced_accuracy": float(balanced_accuracy_score(actual_values, prediction)),
            "f1_weighted": float(f1_score(actual_values, prediction, average="weighted", zero_division=0)),
        }

    def _paired_improvement_probability(
        self,
        actual: pd.Series,
        baseline_prediction: np.ndarray,
        tuned_prediction: np.ndarray,
        task: Any,
        iterations: int = 300,
    ) -> float:
        actual_values = np.asarray(actual)
        n_samples = len(actual_values)
        if n_samples < 12:
            return 0.0
        rng = np.random.default_rng(self.random_state + 7919)
        improved = 0
        task_value = getattr(task, "value", task)
        for _ in range(iterations):
            indices = rng.integers(0, n_samples, n_samples)
            sampled_actual = actual_values[indices]
            sampled_baseline = baseline_prediction[indices]
            sampled_tuned = tuned_prediction[indices]
            if task_value == "regression":
                baseline_loss = float(np.mean((sampled_actual.astype(float) - sampled_baseline.astype(float)) ** 2))
                tuned_loss = float(np.mean((sampled_actual.astype(float) - sampled_tuned.astype(float)) ** 2))
                improved += int(tuned_loss < baseline_loss)
            else:
                from sklearn.metrics import f1_score
                baseline_score = f1_score(sampled_actual, sampled_baseline, average="weighted", zero_division=0)
                tuned_score = f1_score(sampled_actual, sampled_tuned, average="weighted", zero_division=0)
                improved += int(tuned_score > baseline_score)
        return improved / iterations

    def _run_supervised_model(
        self,
        dataset_name: str,
        target: str,
        problem_analysis: Dict[str, Any],
        excluded_targets: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        from .modeling_engine import ModelLibrary, ModelingEngine, TaskType, TaskTypeDetector

        time_columns = [col for col in self._profiles[dataset_name].datetime_columns if col in self._datasets[dataset_name].columns]
        task_candidates = {item["task_type"] for item in problem_analysis.get("task_candidates", [])}
        use_time_validation = bool(time_columns and "prediction_forecast" in task_candidates)
        view = self._build_modeling_view(
            dataset_name,
            target,
            temporal=use_time_validation,
            base_time_column=time_columns[0] if use_time_validation else None,
        )
        join_audit = list(view.attrs.get("feature_join_audit", []))
        if use_time_validation:
            time_values = pd.to_datetime(view[time_columns[0]], errors="coerce")
            view = view.assign(__assistant_time_order=time_values).sort_values("__assistant_time_order")
            view = view.drop(columns=["__assistant_time_order"])
        if target not in view.columns:
            raise ValueError(f"目标列 {target!r} 不在建模视图中")
        view = view.loc[view[target].notna()].copy()
        if len(view) < 30:
            raise ValueError("有效目标样本少于 30，无法进行可靠交叉验证")
        group_column = None if use_time_validation else self._select_validation_group(view, target)
        y = view.pop(target)
        view = view.drop(columns=list(excluded_targets or []), errors="ignore")
        task = TaskTypeDetector.detect(y)
        if task not in {TaskType.CLASSIFICATION, TaskType.REGRESSION}:
            raise ValueError("目标列类型无法用于监督学习")

        drop_columns: List[str] = []
        for col in view.columns:
            series = view[col]
            unique_ratio = series.nunique(dropna=True) / max(series.notna().sum(), 1)
            kind = _column_kind(series, str(col))
            explicit_identifier = self._is_identifier_name(str(col))
            if (
                series.nunique(dropna=True) <= 1
                or (kind == "text" and unique_ratio > 0.5)
                or (explicit_identifier and col != group_column)
            ):
                drop_columns.append(col)
        X = view.drop(columns=drop_columns, errors="ignore")
        if X.shape[1] == 0:
            raise ValueError("清理标识符和常量列后没有可用特征")
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
                X[col] = X[col].fillna(X[col].median())
            elif not pd.api.types.is_datetime64_any_dtype(X[col]):
                X[col] = X[col].fillna("__MISSING__").astype(str)

        available = ModelLibrary.get_models(task)
        preferences = (
            ["ridge", "hist_gb", "xgb", "lgb", "rf"]
            if task == TaskType.REGRESSION
            else ["lr", "hist_gb", "xgb", "lgb", "rf"]
        )
        model_keys = [key for key in preferences if key in available][:3]
        if not model_keys:
            model_keys = list(available)[:2]
        X_fit, y_fit = X, y
        X_confirmation: Optional[pd.DataFrame] = None
        y_confirmation: Optional[pd.Series] = None
        # Hyperparameters are chosen only on the development portion.  A separate
        # confirmation set decides whether the tuned model is actually adopted.
        if self.feedback_optimization and len(X) >= 80:
            confirmation_size = max(12, int(math.ceil(len(X) * 0.2)))
            if use_time_validation:
                split_at = len(X) - confirmation_size
                if split_at >= 30:
                    X_fit, X_confirmation = X.iloc[:split_at].copy(), X.iloc[split_at:].copy()
                    y_fit, y_confirmation = y.iloc[:split_at].copy(), y.iloc[split_at:].copy()
            elif group_column and group_column in X.columns:
                from sklearn.model_selection import GroupShuffleSplit

                try:
                    splitter = GroupShuffleSplit(
                        n_splits=1, test_size=0.2, random_state=self.random_state
                    )
                    train_index, confirmation_index = next(
                        splitter.split(X, y, groups=X[group_column])
                    )
                    X_fit, X_confirmation = X.iloc[train_index].copy(), X.iloc[confirmation_index].copy()
                    y_fit, y_confirmation = y.iloc[train_index].copy(), y.iloc[confirmation_index].copy()
                except ValueError as exc:
                    self._runtime_warnings.append(
                        f"{dataset_name}.{target} 无法按实体 {group_column} 建立确认集，反馈优化已跳过：{exc}"
                    )
                    X_fit, y_fit = X, y
                    X_confirmation = y_confirmation = None
            else:
                from sklearn.model_selection import train_test_split

                stratify = None
                if task == TaskType.CLASSIFICATION:
                    class_counts = pd.Series(y).value_counts(dropna=False)
                    if len(class_counts) > 1 and int(class_counts.min()) >= 2:
                        stratify = y
                try:
                    X_fit, X_confirmation, y_fit, y_confirmation = train_test_split(
                        X,
                        y,
                        test_size=confirmation_size,
                        random_state=self.random_state,
                        stratify=stratify,
                    )
                except ValueError as exc:
                    self._runtime_warnings.append(
                        f"{dataset_name}.{target} 无法建立独立确认集，反馈优化已跳过：{exc}"
                    )
                    X_fit, y_fit = X, y
                    X_confirmation = y_confirmation = None

        engine = ModelingEngine(
            task_type=task.value,
            model_keys=model_keys,
            n_splits=3,
            n_jobs=1,
            optimize_hyperparams=False,
            explainability=False,
            auto_decision_mode="accuracy_first",
            auto_sample=True,
            max_samples=min(self.max_analysis_rows, 40_000),
            fold_type="time" if use_time_validation else ("group" if group_column else "default"),
            group_col=group_column,
            verbose=False,
            random_state=self.random_state,
        )
        result = engine.fit(X_fit, y_fit)
        diagnostic_actual = y_fit
        if task == TaskType.CLASSIFICATION and getattr(engine, "_label_encoder", None) is not None:
            diagnostic_actual = pd.Series(engine._label_encoder.transform(pd.Series(y_fit).astype(str)))
        diagnostics = self._validation_diagnostics(
            result.best_cv_result,
            diagnostic_actual,
            task,
            use_time_validation,
        ) if result.best_cv_result is not None else {}
        hypothesis_check = self._model_hypothesis_check(
            result, diagnostic_actual, task, use_time_validation
        )
        feedback_optimization: Dict[str, Any]
        try:
            optimized_result, optimized_engine, feedback_optimization = self._feedback_optimize_model(
                result,
                engine,
                X_fit,
                y_fit,
                X_confirmation,
                y_confirmation,
                task,
                use_time_validation,
                group_column,
                diagnostics,
            )
            if feedback_optimization.get("accepted"):
                result = optimized_result
                engine = optimized_engine
        except Exception as exc:
            feedback_optimization = {
                "enabled": self.feedback_optimization,
                "attempted": True,
                "accepted": False,
                "diagnostics": diagnostics,
                "reason": f"反馈优化失败，已保留基线：{exc}",
            }
            self._runtime_warnings.append(f"{dataset_name}.{target} 的反馈优化失败，已保留基线：{exc}")
        confirmation_actual = feedback_optimization.pop("_confirmation_actual", None)
        confirmation_prediction = feedback_optimization.pop("_confirmation_prediction", None)
        leaderboard = result.leaderboard.to_dict("records") if result.leaderboard is not None else []
        metrics = (
            feedback_optimization.get("selected_confirmation_metrics", {})
            if confirmation_prediction is not None
            else (result.best_cv_result.mean_scores if result.best_cv_result is not None else {})
        )
        feature_importance = []
        if result.feature_importance is not None:
            feature_importance = result.feature_importance.head(20).to_dict("records")
        predictions = None
        actual_for_plot = np.asarray(y_fit)
        evaluation_features: Optional[pd.DataFrame] = X_fit
        validation = (
            "time_ordered_cv" if use_time_validation
            else ("group_cross_validation" if group_column else "cross_validation")
        )
        note = "指标来自交叉验证的 OOF 预测，不是训练集拟合分数。"
        if confirmation_prediction is not None:
            predictions = np.asarray(confirmation_prediction)
            actual_for_plot = np.asarray(confirmation_actual)
            evaluation_features = X_confirmation
            validation = (
                "temporal_holdout_after_inner_cv" if use_time_validation
                else ("group_holdout_after_inner_cv" if group_column else "holdout_after_inner_cv")
            )
            note = "参数只在开发集内选择；指标来自未参与选参的独立确认集。"
        elif result.best_cv_result is not None and result.best_cv_result.oof_pred is not None:
            predictions = np.asarray(result.best_cv_result.oof_pred)
            if task == TaskType.CLASSIFICATION and getattr(engine, "_label_encoder", None) is not None:
                actual_for_plot = engine._label_encoder.transform(pd.Series(y_fit).astype(str))
            if use_time_validation:
                fold_count = len(next(iter(result.best_cv_result.fold_scores.values()), []))
                initial_window = max(1, len(predictions) // (fold_count + 1))
                predictions = predictions[initial_window:]
                actual_for_plot = actual_for_plot[initial_window:]
                evaluation_features = X_fit.iloc[initial_window:]
        prediction_interval = self._conformal_prediction_summary(
            result,
            y_fit,
            task,
            use_time_validation,
            actual_for_plot if predictions is not None else None,
            predictions,
            independent_evaluation="holdout_after_inner_cv" in validation,
        )
        conformal_check = (
            prediction_interval.pop("_credibility_check", None)
            if prediction_interval else None
        )
        credibility = self._audit_model_credibility(
            X_fit=X_fit,
            y_fit=y_fit,
            X_evaluation=evaluation_features,
            actual=actual_for_plot if predictions is not None else None,
            prediction=predictions,
            task=task,
            target=target,
            validation=validation,
            use_time_validation=use_time_validation,
            group_column=group_column,
            diagnostics=diagnostics,
            join_audit=join_audit,
            feature_importance=feature_importance,
            hypothesis_check=hypothesis_check,
            conformal_check=conformal_check,
            engine=engine,
        )
        return _plain({
            "dataset": dataset_name,
            "target": target,
            "task_type": task.value,
            "n_samples": len(X),
            "fit_samples": len(X_fit),
            "confirmation_samples": len(y_confirmation) if y_confirmation is not None else 0,
            "n_features": X.shape[1],
            "best_model": result.best_model_key,
            "validation": validation,
            "validation_group": group_column,
            "metrics": metrics,
            "leaderboard": leaderboard,
            "feature_importance": feature_importance,
            "feature_join_audit": join_audit,
            "feedback_optimization": feedback_optimization,
            "prediction_interval": prediction_interval,
            "credibility_audit": credibility,
            "actual": actual_for_plot[:20_000] if predictions is not None else None,
            "oof_prediction": predictions[:20_000] if predictions is not None else None,
            "note": note,
        })

    def _run_entropy_topsis(self, target: Optional[str]) -> Optional[Dict[str, Any]]:
        selected: Optional[Tuple[str, pd.DataFrame]] = None
        if target and "." in target:
            name = target.rsplit(".", 1)[0]
            if name in self._datasets:
                selected = (name, self._datasets[name])
        if selected is None:
            selected = max(self._datasets.items(), key=lambda item: len(self._profiles[item[0]].numeric_columns))
        name, df = selected
        numeric = self._top_numeric(df)
        id_columns = self._profiles[name].id_candidates
        numeric = [col for col in numeric if col not in id_columns]
        if len(numeric) < 2:
            return None
        work = _sample_frame(df, self.max_analysis_rows, self.random_state)[numeric].copy()
        work = work.replace([np.inf, -np.inf], np.nan)
        work = work.fillna(work.median())
        normalised = pd.DataFrame(index=work.index)
        directions: Dict[str, str] = {}
        for col in numeric:
            minimum, maximum = work[col].min(), work[col].max()
            scaled = (work[col] - minimum) / (maximum - minimum) if maximum > minimum else pd.Series(0.5, index=work.index)
            negative = any(token in _normalise_name(col) for token in _NEGATIVE_INDICATORS)
            normalised[col] = 1.0 - scaled if negative else scaled
            directions[col] = "negative" if negative else "positive"
        probabilities = normalised.div(normalised.sum(axis=0).replace(0, 1), axis=1).clip(lower=1e-12)
        entropy = -(probabilities * np.log(probabilities)).sum(axis=0) / math.log(max(len(probabilities), 2))
        diversity = 1 - entropy
        weights = diversity / diversity.sum() if diversity.sum() > 0 else pd.Series(1 / len(numeric), index=numeric)
        weighted = normalised.mul(weights, axis=1)
        distance_best = np.sqrt(((weighted - weighted.max()) ** 2).sum(axis=1))
        distance_worst = np.sqrt(((weighted - weighted.min()) ** 2).sum(axis=1))
        scores = distance_worst / (distance_best + distance_worst).replace(0, np.nan)
        scores = scores.fillna(0.5)
        entity_col = id_columns[0] if id_columns else None
        entity_values = (
            df.loc[work.index, entity_col].astype(str).values
            if entity_col else work.index.astype(str).to_numpy()
        )
        ranking = pd.DataFrame({
            "entity": entity_values,
            "score": scores.values,
        }).sort_values("score", ascending=False)
        ranking["rank"] = np.arange(1, len(ranking) + 1)
        # A ranking is not trustworthy merely because TOPSIS returns a number.
        # Recompute a bounded entity subset under plausible weight perturbations
        # and quantify whether the ordering and winner survive.
        sensitivity_frame = normalised.reset_index(drop=True)
        if len(sensitivity_frame) > 3_000:
            reported_winner_position = int(np.argmax(scores.to_numpy(dtype=float)))
            other_rows = sensitivity_frame.drop(index=reported_winner_position).sample(
                n=2_999, random_state=self.random_state
            )
            sensitivity_frame = pd.concat([
                sensitivity_frame.iloc[[reported_winner_position]], other_rows,
            ], ignore_index=True)
        base_values = sensitivity_frame.to_numpy(dtype=float)
        base_weights = weights.to_numpy(dtype=float)

        def topsis_scores(weight_vector: np.ndarray) -> np.ndarray:
            candidate = base_values * weight_vector
            best_point = np.max(candidate, axis=0)
            worst_point = np.min(candidate, axis=0)
            distance_to_best = np.sqrt(np.sum((candidate - best_point) ** 2, axis=1))
            distance_to_worst = np.sqrt(np.sum((candidate - worst_point) ** 2, axis=1))
            denominator = distance_to_best + distance_to_worst
            return np.divide(
                distance_to_worst, denominator,
                out=np.full_like(distance_to_worst, 0.5), where=denominator > 0,
            )

        baseline_scores = topsis_scores(base_weights)
        baseline_ranks = pd.Series(baseline_scores).rank(method="average").to_numpy()
        baseline_winner = int(np.argmax(baseline_scores))
        rng = np.random.default_rng(self.random_state + 601)
        correlations: List[float] = []
        winner_matches = 0
        sensitivity_iterations = 100
        for _ in range(sensitivity_iterations):
            perturbed_weights = base_weights * rng.lognormal(0.0, 0.20, len(base_weights))
            perturbed_weights /= perturbed_weights.sum()
            perturbed_scores = topsis_scores(perturbed_weights)
            perturbed_ranks = pd.Series(perturbed_scores).rank(method="average").to_numpy()
            correlation = pd.Series(baseline_ranks).corr(
                pd.Series(perturbed_ranks), method="spearman"
            )
            if np.isfinite(correlation):
                correlations.append(float(correlation))
            winner_matches += int(np.argmax(perturbed_scores) == baseline_winner)
        rank_stability = float(np.median(correlations)) if correlations else 0.0
        winner_retention = winner_matches / sensitivity_iterations
        leave_one_out_winners: Dict[str, bool] = {}
        if len(base_weights) > 2:
            for indicator_index, indicator in enumerate(numeric):
                candidate_weights = base_weights.copy()
                candidate_weights[indicator_index] = 0.0
                candidate_total = candidate_weights.sum()
                if candidate_total <= 0:
                    candidate_weights = np.ones_like(base_weights)
                    candidate_weights[indicator_index] = 0.0
                    candidate_total = candidate_weights.sum()
                candidate_weights /= candidate_total
                leave_one_out_winners[indicator] = bool(
                    np.argmax(topsis_scores(candidate_weights)) == baseline_winner
                )
        leave_one_out_retention = (
            float(np.mean(list(leave_one_out_winners.values())))
            if leave_one_out_winners else None
        )
        sensitivity_status = (
            "pass" if rank_stability >= 0.95 and winner_retention >= 0.80
            else ("warning" if rank_stability >= 0.80 and winner_retention >= 0.50 else "fail")
        )
        sensitivity_check = self._credibility_check(
            "weight_sensitivity", "排名权重敏感性", sensitivity_status,
            f"权重随机扰动 100 次：中位秩相关={rank_stability:.3f}，首名保持率={winner_retention:.1%}。",
            "排名对权重敏感，应报告名次区间或帕累托集合，不能只给出唯一顺序。"
            if sensitivity_status != "pass" else "",
            {
                "iterations": sensitivity_iterations,
                "median_rank_spearman": rank_stability,
                "winner_retention": winner_retention,
                "leave_one_indicator_out_winner_retention": leave_one_out_retention,
                "leave_one_indicator_out": leave_one_out_winners,
            },
        )
        audit_label = {
            "pass": "可信", "warning": "谨慎使用", "fail": "不可信",
        }[sensitivity_status]
        # A scalar score hides genuine trade-offs. Compute an exact Pareto front
        # on a bounded, winner-preserving sample and report how many alternatives
        # cannot be declared worse without introducing preferences.
        pareto_limit = min(1_200, len(normalised))
        winner_position = int(np.argmax(scores.to_numpy(dtype=float)))
        if len(normalised) > pareto_limit:
            rng = np.random.default_rng(self.random_state + 607)
            available_positions = np.delete(np.arange(len(normalised)), winner_position)
            pareto_positions = np.r_[
                winner_position,
                rng.choice(available_positions, pareto_limit - 1, replace=False),
            ]
        else:
            pareto_positions = np.arange(len(normalised))
        pareto_values = normalised.to_numpy(dtype=float)[pareto_positions]
        dominates = (
            np.all(pareto_values[:, None, :] >= pareto_values[None, :, :], axis=2)
            & np.any(pareto_values[:, None, :] > pareto_values[None, :, :], axis=2)
        )
        nondominated_mask = ~np.any(dominates, axis=0)
        nondominated_positions = pareto_positions[nondominated_mask]
        pareto_share = float(np.mean(nondominated_mask))
        score_values = scores.to_numpy(dtype=float)
        pareto_order = nondominated_positions[
            np.argsort(score_values[nondominated_positions])[::-1]
        ]
        indicator_correlation = normalised.iloc[pareto_positions].corr(method="spearman")
        conflicts: List[Dict[str, Any]] = []
        for left_index, right_index in combinations(range(len(numeric)), 2):
            correlation = indicator_correlation.iloc[left_index, right_index]
            if np.isfinite(correlation) and correlation <= -0.30:
                conflicts.append({
                    "left": numeric[left_index], "right": numeric[right_index],
                    "spearman": float(correlation),
                })
        conflicts.sort(key=lambda item: item["spearman"])
        pareto_status = (
            "warning" if pareto_share > 0.25 or conflicts else "pass"
        )
        pareto_check = self._credibility_check(
            "pareto_tradeoff", "Pareto 权衡保留", pareto_status,
            f"审计样本中 {int(nondominated_mask.sum())}/{len(nondominated_mask)} "
            f"（{pareto_share:.1%}）对象非支配；发现 {len(conflicts)} 对明显冲突指标。",
            "非支配方案较多时应展示 Pareto 集和偏好情景，不能把 TOPSIS 唯一顺序描述成客观事实。"
            if pareto_status != "pass" else "",
            {"pareto_share": pareto_share, "conflicting_indicator_pairs": conflicts[:20]},
        )
        if sensitivity_status == "fail":
            ranking_audit_status = "fail"
        elif sensitivity_status == "warning" or pareto_status == "warning":
            ranking_audit_status = "warning"
        else:
            ranking_audit_status = "pass"
        audit_label = {
            "pass": "可信", "warning": "谨慎使用", "fail": "不可信",
        }[ranking_audit_status]
        return _plain({
            "dataset": name,
            "method": "entropy_weight_topsis",
            "indicators": numeric,
            "directions": directions,
            "weights": {col: float(weights[col]) for col in numeric},
            "ranking": ranking.head(100).to_dict("records"),
            "sensitivity": sensitivity_check["details"],
            "pareto_analysis": {
                "sample_size": len(pareto_positions),
                "front_size": int(nondominated_mask.sum()),
                "front_share": pareto_share,
                "front": [
                    {
                        "entity": str(entity_values[position]),
                        "topsis_score": float(score_values[position]),
                    }
                    for position in pareto_order[:100]
                ],
                "conflicting_indicator_pairs": conflicts[:20],
                "note": "Pareto 非支配只使用指标方向，不引入人为权重。",
            },
            "credibility_audit": {
                "status": ranking_audit_status,
                "label": audit_label,
                "checks": [sensitivity_check, pareto_check],
                "decision": (
                    "当前排序对合理权重扰动较稳定，且未隐藏大规模权衡集"
                    if ranking_audit_status == "pass"
                    else "排序必须连同权重敏感性一起使用，不能把单一名次当成确定事实"
                ),
            },
            "literature_basis": {
                "idea": "retain nondominated images before preference-dependent scalarization",
                "doi": "10.1007/s00186-023-00823-2",
            },
            "note": "权重由样本离散度确定；负向指标已正向化，并已执行权重扰动、删指标和 Pareto 非支配复核。",
        })

    def _run_data_structure_analysis(self) -> List[Dict[str, Any]]:
        """Bounded PCA structure discovery with robust reconstruction anomalies."""
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer

        ranked_datasets = sorted(
            self._datasets,
            key=lambda name: (
                len([
                    column for column in self._profiles[name].numeric_columns
                    if column not in self._profiles[name].id_candidates
                ]),
                len(self._datasets[name]),
            ),
            reverse=True,
        )
        results: List[Dict[str, Any]] = []
        for dataset_name in ranked_datasets[:5]:
            source = self._datasets[dataset_name]
            numeric = [
                column for column in self._top_numeric(source)
                if column not in self._profiles[dataset_name].id_candidates
                and source[column].nunique(dropna=True) > 1
            ][:self.max_numeric_columns]
            # Two measurements do not constitute a meaningful latent-space
            # discovery problem: PCA would trivially return two dimensions and
            # dress an ordinary bivariate scatter as a "stable structure".
            if len(numeric) < 3 or len(source) < 20:
                continue
            sample_limit = min(self.max_analysis_rows, 5_000)
            sampled = _sample_frame(source[numeric], sample_limit, self.random_state).copy()
            values = sampled.replace([np.inf, -np.inf], np.nan)
            matrix = SimpleImputer(strategy="median").fit_transform(values)
            finite_variance = np.nanstd(matrix, axis=0) > 1e-12
            if int(finite_variance.sum()) < 3:
                continue
            kept_columns = [column for column, keep in zip(numeric, finite_variance) if keep]
            matrix = matrix[:, finite_variance]
            robust_center = np.median(matrix, axis=0)
            feature_mad = np.median(np.abs(matrix - robust_center), axis=0)
            feature_scale = 1.4826 * feature_mad
            standard_scale = np.std(matrix, axis=0)
            feature_scale = np.where(
                feature_scale > 1e-12, feature_scale,
                np.where(standard_scale > 1e-12, standard_scale, 1.0),
            )
            scaled = (matrix - robust_center) / feature_scale
            # Fit the basis on winsorised robust-standardised values so a handful
            # of high-leverage rows cannot rotate the entire PCA subspace.
            basis_values = np.clip(scaled, -8.0, 8.0)
            maximum_components = min(scaled.shape[1], scaled.shape[0] - 1)
            pca = PCA(n_components=maximum_components, svd_solver="full", random_state=self.random_state)
            pca.fit(basis_values)
            scores = pca.transform(scaled)
            cumulative = np.cumsum(pca.explained_variance_ratio_)
            dimensions_90 = int(np.searchsorted(cumulative, 0.90) + 1)
            dimensions_90 = min(max(dimensions_90, 1), maximum_components)
            reconstruction_dimensions = min(dimensions_90, max(1, scaled.shape[1] - 1))
            retained_scores = scores[:, :reconstruction_dimensions]
            reconstructed = retained_scores @ pca.components_[:reconstruction_dimensions] + pca.mean_
            squared_error = np.mean((scaled - reconstructed) ** 2, axis=1)
            # Log-transform the non-negative error before MAD standardisation;
            # otherwise the natural chi-square-like right tail looks anomalous.
            error_floor = max(float(np.median(squared_error)) * 1e-6, 1e-15)
            log_error = np.log(squared_error + error_floor)
            error_median = float(np.median(log_error))
            error_mad = float(np.median(np.abs(log_error - error_median)))
            robust_scale = max(1.4826 * error_mad, 1e-12)
            reconstruction_robust_z = (log_error - error_median) / robust_scale
            retained_center = np.median(retained_scores, axis=0)
            retained_mad = np.median(
                np.abs(retained_scores - retained_center), axis=0
            )
            retained_scale = np.where(1.4826 * retained_mad > 1e-12, 1.4826 * retained_mad, 1.0)
            score_distance = np.sqrt(np.mean(
                ((retained_scores - retained_center) / retained_scale) ** 2,
                axis=1,
            ))
            distance_median = float(np.median(score_distance))
            distance_mad = float(np.median(np.abs(score_distance - distance_median)))
            distance_scale = max(1.4826 * distance_mad, 1e-12)
            score_distance_robust_z = (
                score_distance - distance_median
            ) / distance_scale
            robust_z = np.maximum(reconstruction_robust_z, score_distance_robust_z)
            # Squared reconstruction errors are right-skewed. A fixed MAD cutoff
            # alone can therefore flag a large fraction of an otherwise regular
            # sample. Keep the robust cutoff, but calibrate it against the 99th
            # empirical percentile so the unsupervised false-alarm budget is bounded.
            anomaly_threshold = max(3.5, float(np.quantile(robust_z, 0.99)))
            anomaly_mask = robust_z > anomaly_threshold
            anomaly_positions = np.flatnonzero(anomaly_mask)
            top_positions = np.argsort(robust_z)[::-1][:min(50, len(robust_z))]
            top_anomalies: List[Dict[str, Any]] = []
            for position in top_positions:
                if not anomaly_mask[position] and len(top_anomalies) >= 10:
                    continue
                feature_deviation = np.maximum(
                    np.abs(scaled[position]),
                    np.abs(scaled[position] - reconstructed[position]),
                )
                dominant_positions = np.argsort(feature_deviation)[::-1][:3]
                top_anomalies.append({
                    "row_index": str(sampled.index[position]),
                    "robust_z": float(robust_z[position]),
                    "reconstruction_robust_z": float(reconstruction_robust_z[position]),
                    "score_distance": float(score_distance[position]),
                    "score_distance_robust_z": float(score_distance_robust_z[position]),
                    "reconstruction_error": float(squared_error[position]),
                    "flagged": bool(anomaly_mask[position]),
                    "dominant_deviations": [
                        {
                            "feature": kept_columns[feature_position],
                            "robust_value": float(scaled[position, feature_position]),
                            "deviation_score": float(feature_deviation[feature_position]),
                        }
                        for feature_position in dominant_positions
                    ],
                })
            top_anomalies = top_anomalies[:50]
            projection_limit = min(2_000, len(sampled))
            projection_positions = list(anomaly_positions[:min(200, projection_limit)])
            projection_set = set(projection_positions)
            projection_positions.extend(
                position for position in range(len(sampled))
                if position not in projection_set
            )
            projection_positions = projection_positions[:projection_limit]
            projection = [
                {
                    "pc1": float(scores[position, 0]),
                    "pc2": float(scores[position, 1]) if scores.shape[1] > 1 else 0.0,
                    "robust_z": float(robust_z[position]),
                    "flagged": bool(anomaly_mask[position]),
                }
                for position in projection_positions
            ]
            loadings: List[Dict[str, Any]] = []
            for component_index in range(min(dimensions_90, 3)):
                component = pca.components_[component_index]
                order = np.argsort(np.abs(component))[::-1][:min(6, len(component))]
                loadings.append({
                    "component": f"PC{component_index + 1}",
                    "variance_ratio": float(pca.explained_variance_ratio_[component_index]),
                    "dominant_features": [
                        {"feature": kept_columns[position], "loading": float(component[position])}
                        for position in order
                    ],
                })

            subspace_stability = None
            if len(scaled) >= max(60, 4 * dimensions_90):
                rng = np.random.default_rng(self.random_state + 503)
                permutation = rng.permutation(len(scaled))
                midpoint = len(permutation) // 2
                left = basis_values[permutation[:midpoint]]
                right = basis_values[permutation[midpoint:]]
                comparison_components = min(dimensions_90, len(left) - 1, len(right) - 1)
                left_basis = PCA(n_components=comparison_components, svd_solver="full").fit(left).components_
                right_basis = PCA(n_components=comparison_components, svd_solver="full").fit(right).components_
                singular_values = np.linalg.svd(left_basis @ right_basis.T, compute_uv=False)
                subspace_stability = float(np.mean(np.clip(singular_values, 0.0, 1.0)))

            rng = np.random.default_rng(self.random_state + 509)
            perturbed = scaled + rng.normal(0.0, 0.01, scaled.shape)
            perturbed_scores = pca.transform(perturbed)[:, :reconstruction_dimensions]
            perturbed_reconstruction = perturbed_scores @ pca.components_[:reconstruction_dimensions] + pca.mean_
            perturbed_error = np.mean((perturbed - perturbed_reconstruction) ** 2, axis=1)
            perturbed_reconstruction_z = (
                np.log(perturbed_error + error_floor) - error_median
            ) / robust_scale
            perturbed_score_distance = np.sqrt(np.mean(
                ((perturbed_scores - retained_center) / retained_scale) ** 2,
                axis=1,
            ))
            perturbed_score_distance_z = (
                perturbed_score_distance - distance_median
            ) / distance_scale
            perturbed_combined_score = np.maximum(
                perturbed_reconstruction_z, perturbed_score_distance_z
            )
            perturbed_mask = perturbed_combined_score > anomaly_threshold
            union = int(np.sum(anomaly_mask | perturbed_mask))
            anomaly_jaccard = (
                float(np.sum(anomaly_mask & perturbed_mask) / union) if union else 1.0
            )

            singular = np.linalg.svd(basis_values, compute_uv=False)
            positive = singular[singular > 1e-10]
            condition_number = float(positive.max() / positive.min()) if len(positive) else float("inf")
            sample_ratio = len(sampled) / max(len(source), 1)
            checks = [
                self._credibility_check(
                    "sample_adequacy", "结构样本充分性",
                    "pass" if len(sampled) >= 10 * len(kept_columns) else "warning",
                    f"使用 {len(sampled)} 行、{len(kept_columns)} 个有效指标，样本/指标比={len(sampled) / len(kept_columns):.1f}。",
                    "增加样本或减少指标后复核潜在结构。" if len(sampled) < 10 * len(kept_columns) else "",
                ),
                self._credibility_check(
                    "subspace_stability", "主子空间稳定性",
                    "not_assessed" if subspace_stability is None else (
                        "pass" if subspace_stability >= 0.85 else (
                            "warning" if subspace_stability >= 0.65 else "fail"
                        )
                    ),
                    "样本不足，未执行分半子空间复核。" if subspace_stability is None else
                    f"两半样本主子空间平均典型相似度={subspace_stability:.1%}。",
                    "潜在维度对样本敏感，应避免对单个主成分作强解释。"
                    if subspace_stability is not None and subspace_stability < 0.85 else "",
                ),
                self._credibility_check(
                    "anomaly_perturbation", "异常名单扰动稳定性",
                    "pass" if anomaly_jaccard >= 0.8 else (
                        "warning" if anomaly_jaccard >= 0.5 else "fail"
                    ),
                    f"输入加入 1% 标准差扰动后，异常名单 Jaccard={anomaly_jaccard:.1%}。",
                    "异常名单不稳定，应结合业务阈值和人工复核。" if anomaly_jaccard < 0.8 else "",
                ),
            ]
            failed = [item for item in checks if item["status"] == "fail"]
            warned = [item for item in checks if item["status"] in {"warning", "not_assessed"}]
            audit_status = "fail" if failed else ("warning" if warned else "pass")
            audit_label = "不可信" if failed else ("谨慎使用" if warned else "可信")
            results.append(_plain({
                "dataset": dataset_name,
                "method": "robust_pca_reconstruction_and_leverage",
                "source_rows": int(self._profiles[dataset_name].source_rows),
                "analysis_rows": len(sampled),
                "analysis_sample_ratio": sample_ratio,
                "features": kept_columns,
                "original_dimensions": len(kept_columns),
                "dimensions_90": dimensions_90,
                "reconstruction_dimensions": reconstruction_dimensions,
                "dimension_reduction_ratio": 1.0 - dimensions_90 / len(kept_columns),
                "cumulative_explained_variance": float(cumulative[dimensions_90 - 1]),
                "explained_variance_ratio": pca.explained_variance_ratio_[:min(10, len(pca.explained_variance_ratio_))],
                "dominant_loadings": loadings,
                "subspace_stability": subspace_stability,
                "condition_number": condition_number,
                "anomaly_count": int(len(anomaly_positions)),
                "anomaly_rate": float(np.mean(anomaly_mask)),
                "estimated_source_anomalies": int(round(np.mean(anomaly_mask) * self._profiles[dataset_name].source_rows)),
                "anomaly_threshold_robust_z": anomaly_threshold,
                "anomaly_perturbation_jaccard": anomaly_jaccard,
                "top_anomalies": top_anomalies,
                "projection": projection,
                "credibility_audit": {
                    "status": audit_status,
                    "label": audit_label,
                    "checks": checks,
                    "decision": (
                        "潜在结构或异常名单存在关键不稳定性" if failed
                        else ("结构可用于探索，但应保留稳定性警告" if warned else "结构与异常名单通过当前稳定性复核")
                    ),
                },
                "note": "异常联合考虑主子空间外的重构偏离与主子空间内的高杠杆偏离；它不自动等同于数据错误或业务故障。",
            }))
        return results

    def _run_clustering(self) -> Dict[str, Any]:
        """Select a stable K-Means solution with bounded silhouette evaluation."""
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
        from sklearn.preprocessing import StandardScaler

        candidates = sorted(
            self._datasets,
            key=lambda name: len(self._profiles[name].numeric_columns),
            reverse=True,
        )
        dataset_name = candidates[0]
        source = _sample_frame(
            self._datasets[dataset_name],
            min(self.max_analysis_rows, 20_000),
            self.random_state,
        )
        numeric = [
            col for col in self._top_numeric(source)
            if col not in self._profiles[dataset_name].id_candidates
            and source[col].nunique(dropna=True) > 1
        ]
        if len(numeric) < 2 or len(source) < 20:
            raise ValueError("聚类至少需要 20 行和 2 个非标识数值特征")
        values = SimpleImputer(strategy="median").fit_transform(source[numeric])
        scaled = StandardScaler().fit_transform(values)
        evaluation_index = np.arange(len(scaled))
        if len(evaluation_index) > 2_000:
            rng = np.random.default_rng(self.random_state)
            evaluation_index = rng.choice(evaluation_index, size=2_000, replace=False)
        max_k = min(8, max(2, len(scaled) // 20))
        trials: List[Dict[str, Any]] = []
        best = None
        for k in range(2, max_k + 1):
            model_class = MiniBatchKMeans if len(scaled) > 5_000 else KMeans
            model = model_class(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = model.fit_predict(scaled)
            eval_labels = labels[evaluation_index]
            if len(np.unique(eval_labels)) < 2:
                continue
            silhouette = float(silhouette_score(scaled[evaluation_index], eval_labels))
            db_index = float(davies_bouldin_score(scaled[evaluation_index], eval_labels))
            trial = {"k": k, "silhouette": silhouette, "davies_bouldin": db_index}
            trials.append(trial)
            if best is None or silhouette > best[0]:
                best = (silhouette, model, labels, db_index)
        if best is None:
            raise ValueError("没有形成两个以上有效簇")
        _, best_model, labels, db_index = best
        stability_scores: List[float] = []
        model_class = MiniBatchKMeans if len(scaled) > 5_000 else KMeans
        for offset in range(1, 6):
            repeated = model_class(
                n_clusters=int(best_model.n_clusters),
                random_state=self.random_state + offset,
                n_init=5,
            )
            repeated_labels = repeated.fit_predict(scaled)
            stability_scores.append(float(adjusted_rand_score(labels, repeated_labels)))
        minimum_stability = min(stability_scores) if stability_scores else 0.0
        smallest_cluster_share = float(pd.Series(labels).value_counts(normalize=True).min())
        separation_status = (
            "pass" if best[0] >= 0.35 else ("warning" if best[0] >= 0.20 else "fail")
        )
        stability_status = (
            "pass" if minimum_stability >= 0.85
            else ("warning" if minimum_stability >= 0.60 else "fail")
        )
        size_status = "warning" if smallest_cluster_share < 0.02 else "pass"
        cluster_checks = [
            self._credibility_check(
                "cluster_separation", "簇间分离度", separation_status,
                f"最优轮廓系数={best[0]:.3f}，Davies-Bouldin={db_index:.3f}。",
                "簇间边界较弱，群组只能用于探索，不宜解释为自然类别。"
                if separation_status != "pass" else "",
            ),
            self._credibility_check(
                "cluster_seed_stability", "随机种子稳定性", stability_status,
                f"更换 5 个随机种子后，最小调整兰德指数={minimum_stability:.3f}。",
                "聚类解依赖初始化，应比较其他算法或降低结论强度。"
                if stability_status != "pass" else "",
                {"adjusted_rand_scores": stability_scores},
            ),
            self._credibility_check(
                "cluster_size_balance", "极小簇检查", size_status,
                f"最小簇占样本的 {smallest_cluster_share:.1%}。",
                "核查极小簇是否只是异常点集合，而不是稳定群体。" if size_status != "pass" else "",
            ),
        ]
        cluster_failed = any(item["status"] == "fail" for item in cluster_checks)
        cluster_warned = any(item["status"] == "warning" for item in cluster_checks)
        cluster_audit_status = "fail" if cluster_failed else ("warning" if cluster_warned else "pass")
        cluster_audit_label = {
            "pass": "可信", "warning": "谨慎使用", "fail": "不可信",
        }[cluster_audit_status]
        profiled = source[numeric].copy()
        profiled["cluster"] = labels
        cluster_profile = profiled.groupby("cluster")[numeric].mean().round(6)
        cluster_sizes = pd.Series(labels).value_counts().sort_index()
        embedding = PCA(n_components=2, random_state=self.random_state).fit_transform(scaled)
        if len(embedding) > 5_000:
            rng = np.random.default_rng(self.random_state)
            plot_index = rng.choice(len(embedding), 5_000, replace=False)
            embedding = embedding[plot_index]
            plot_labels = labels[plot_index]
        else:
            plot_labels = labels
        return _plain({
            "dataset": dataset_name,
            "target": None,
            "task_type": "clustering",
            "n_samples": len(source),
            "n_features": len(numeric),
            "best_model": best_model.__class__.__name__,
            "best_k": int(best_model.n_clusters),
            "metrics": {"silhouette": best[0], "davies_bouldin": db_index},
            "trials": trials,
            "cluster_sizes": {str(key): int(value) for key, value in cluster_sizes.items()},
            "cluster_profiles": cluster_profile.reset_index().to_dict("records"),
            "credibility_audit": {
                "enabled": True,
                "status": cluster_audit_status,
                "label": cluster_audit_label,
                "checks": cluster_checks,
                "decision": (
                    "聚类结构通过分离度、随机种子和极小簇复核"
                    if cluster_audit_status == "pass"
                    else "聚类结构存在不稳定或弱分离证据，不应把簇标签当成客观真值"
                ),
            },
            "embedding": embedding,
            "cluster_labels": plot_labels,
            "note": "簇数通过轮廓系数选择；评估最多抽样 2000 行，并使用 5 个随机种子复核稳定性。",
        })

    def _run_graph_analysis(self, problem: str) -> Optional[Dict[str, Any]]:
        source_tokens = ("source", "from", "origin", "start", "起点", "出发", "源节点", "始发")
        target_tokens = ("target", "to", "destination", "end", "终点", "到达", "目的", "终到")
        weight_tokens = ("weight", "distance", "cost", "length", "time", "权重", "距离", "成本", "长度", "时间")
        selected = None
        for dataset_name, frame in self._datasets.items():
            source_columns = [str(col) for col in frame.columns if any(token in str(col).lower() for token in source_tokens)]
            target_columns = [str(col) for col in frame.columns if any(token in str(col).lower() for token in target_tokens)]
            if source_columns and target_columns:
                selected = (dataset_name, frame, source_columns[0], target_columns[0])
                break
        if selected is None:
            for dataset_name, frame in self._datasets.items():
                candidates = [
                    column for column in (
                        self._profiles[dataset_name].id_candidates
                        + self._profiles[dataset_name].categorical_columns
                    )
                    if column in frame.columns
                ]
                if len(candidates) >= 2:
                    selected = (dataset_name, frame, candidates[0], candidates[1])
                    break
        if selected is None:
            return None
        dataset_name, source, source_col, target_col = selected
        frame = _sample_frame(source, min(self.max_analysis_rows, 50_000), self.random_state)
        edges = frame[[source_col, target_col]].copy()
        edges["source"] = edges[source_col].map(_normalise_value)
        edges["target"] = edges[target_col].map(_normalise_value)
        edges = edges[(edges["source"] != "") & (edges["target"] != "")]
        if edges.empty:
            return None
        weight_col = next(
            (str(col) for col in frame.select_dtypes(include=[np.number]).columns if any(token in str(col).lower() for token in weight_tokens)),
            None,
        )
        nodes = set(edges["source"]) | set(edges["target"])
        degree: Dict[str, int] = {node: 0 for node in nodes}
        parent = {node: node for node in nodes}

        def find(node: str) -> str:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(left: str, right: str) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        unique_edges = set()
        for source_node, target_node in edges[["source", "target"]].itertuples(index=False, name=None):
            degree[source_node] += 1
            degree[target_node] += 1
            unique_edges.add((source_node, target_node))
            union(source_node, target_node)
        components: Dict[str, int] = {}
        for node in nodes:
            root = find(node)
            components[root] = components.get(root, 0) + 1
        directed = any(token in problem for token in ("有向", "流向", "起点", "终点"))
        n_nodes = len(nodes)
        denominator = n_nodes * (n_nodes - 1) if directed else n_nodes * (n_nodes - 1) / 2
        top_degree = sorted(degree.items(), key=lambda item: (-item[1], item[0]))[:20]
        result = {
            "dataset": dataset_name,
            "source_column": source_col,
            "target_column": target_col,
            "weight_column": weight_col,
            "directed": directed,
            "n_nodes": n_nodes,
            "n_edge_records": len(edges),
            "n_unique_edges": len(unique_edges),
            "density": len(unique_edges) / denominator if denominator else 0.0,
            "connected_components": len(components),
            "largest_component_size": max(components.values()) if components else 0,
            "top_degree_nodes": [{"node": node, "degree": value} for node, value in top_degree],
            "note": "中心节点按度数筛选；最短路/最大流需要题目明确起终点及权重含义。",
        }
        if weight_col:
            weights = pd.to_numeric(frame.loc[edges.index, weight_col], errors="coerce").dropna()
            if len(weights):
                result["weight_summary"] = {
                    "min": float(weights.min()), "mean": float(weights.mean()),
                    "max": float(weights.max()), "negative_count": int((weights < 0).sum()),
                }
        return _plain(result)

    def _numeric_subject(self, target: Optional[str] = None) -> Optional[Tuple[str, str]]:
        if target:
            selected = self._select_target(target)
            if selected and pd.api.types.is_numeric_dtype(self._datasets[selected[0]][selected[1]]):
                return selected
        inferred = self._select_target(None)
        if inferred and pd.api.types.is_numeric_dtype(self._datasets[inferred[0]][inferred[1]]):
            return inferred
        candidates = [
            (name, column)
            for name, profile in self._profiles.items()
            for column in profile.numeric_columns
            if column not in profile.id_candidates
        ]
        return candidates[0] if candidates else None

    def _run_cross_fitted_causal_effect(
        self, problem: str, target: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Estimate a partially linear treatment effect only with explicit roles."""
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
        from sklearn.model_selection import GroupKFold, KFold

        role_token = r"[0-9A-Za-z_\-\.\u4e00-\u9fff]+"

        def extract_role(role_names: str) -> Optional[str]:
            forward = re.search(
                rf"(?:{role_names})(?:变量)?\s*(?:为|是|=|:|：)\s*[‘’'\"]?({role_token})",
                str(problem), re.IGNORECASE,
            )
            if forward:
                return forward.group(1).strip("。；;，,、'\"‘’")
            reverse = re.search(
                rf"以\s*[‘’'\"]?({role_token})[‘’'\"]?\s*(?:为|作为)\s*(?:{role_names})(?:变量)?",
                str(problem), re.IGNORECASE,
            )
            return reverse.group(1).strip("。；;，,、'\"‘’") if reverse else None

        outcome_reference = target or extract_role("结果|结局|因变量|outcome")
        treatment_reference = extract_role("处理|干预|政策|暴露|措施|treatment")
        if not outcome_reference or not treatment_reference:
            return None

        def resolve(reference: str, preferred_dataset: Optional[str] = None) -> Optional[Tuple[str, str]]:
            reference = str(reference).strip()
            if "." in reference:
                dataset_name, column = reference.rsplit(".", 1)
                if dataset_name in self._datasets and column in self._datasets[dataset_name].columns:
                    return dataset_name, column
            if preferred_dataset and reference in self._datasets[preferred_dataset].columns:
                return preferred_dataset, reference
            matches = [
                (dataset_name, reference) for dataset_name, frame in self._datasets.items()
                if reference in frame.columns
            ]
            return matches[0] if len(matches) == 1 else None

        outcome = resolve(str(outcome_reference))
        if outcome is None:
            return None
        treatment = resolve(str(treatment_reference), outcome[0])
        if treatment is None or treatment[0] != outcome[0] or treatment[1] == outcome[1]:
            return None
        dataset_name, outcome_column = outcome
        treatment_column = treatment[1]
        source = _sample_frame(
            self._datasets[dataset_name],
            min(self.max_analysis_rows, 10_000), self.random_state,
        ).copy()
        outcome_values = pd.to_numeric(source[outcome_column], errors="coerce")
        raw_treatment = source[treatment_column]
        treatment_levels = raw_treatment.dropna().unique()
        binary_treatment = len(treatment_levels) == 2
        treatment_mapping = None
        if binary_treatment:
            ordered_levels = sorted(treatment_levels, key=lambda value: str(value))
            treatment_mapping = {
                str(ordered_levels[0]): 0, str(ordered_levels[1]): 1,
            }
            treatment_values = raw_treatment.map(
                {ordered_levels[0]: 0.0, ordered_levels[1]: 1.0}
            )
        else:
            treatment_values = pd.to_numeric(raw_treatment, errors="coerce")
            if treatment_values.nunique(dropna=True) < 5:
                return None
        valid = outcome_values.notna() & treatment_values.notna()
        if int(valid.sum()) < 80:
            return None
        source = source.loc[valid].reset_index(drop=True)
        y = outcome_values.loc[valid].to_numpy(dtype=float)
        d = treatment_values.loc[valid].to_numpy(dtype=float)
        excluded = {
            outcome_column, treatment_column,
            *self._profiles[dataset_name].datetime_columns,
            *self._profiles[dataset_name].id_candidates,
        }
        control_columns: List[str] = []
        for column in source.columns:
            if column in excluded:
                continue
            series = source[column]
            if pd.api.types.is_numeric_dtype(series):
                if series.nunique(dropna=True) > 1:
                    control_columns.append(column)
            elif series.nunique(dropna=True) <= 20:
                control_columns.append(column)
            if len(control_columns) >= 30:
                break
        if control_columns:
            controls = source[control_columns].copy()
            for column in controls.columns:
                if pd.api.types.is_numeric_dtype(controls[column]):
                    controls[column] = pd.to_numeric(controls[column], errors="coerce")
                    controls[column] = controls[column].replace([np.inf, -np.inf], np.nan)
                    controls[column] = controls[column].fillna(controls[column].median())
                else:
                    controls[column] = controls[column].fillna("__MISSING__").astype(str)
            controls = pd.get_dummies(controls, drop_first=False, dtype=float)
            X = controls.to_numpy(dtype=float)
        else:
            X = np.zeros((len(source), 1), dtype=float)
        group_column = next(
            (
                column for column in self._profiles[dataset_name].id_candidates
                if column in source.columns
                and 5 <= source[column].nunique(dropna=True) <= int(0.8 * len(source))
            ),
            None,
        )
        if group_column:
            groups = source[group_column].astype(str).to_numpy()
            n_splits = min(5, len(np.unique(groups)))
            splitter = GroupKFold(n_splits=n_splits)
            splits = list(splitter.split(X, y, groups=groups))
            validation = "group_cross_fitting"
        else:
            n_splits = 5
            splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            splits = list(splitter.split(X, y))
            validation = "cross_fitting"
        y_nuisance = np.empty(len(y), dtype=float)
        d_nuisance = np.empty(len(d), dtype=float)
        fold_effects: List[float] = []
        for fold_index, (train_index, test_index) in enumerate(splits):
            outcome_model = HistGradientBoostingRegressor(
                max_iter=120, max_depth=4, learning_rate=0.06,
                l2_regularization=0.1, random_state=self.random_state + fold_index,
            )
            outcome_model.fit(X[train_index], y[train_index])
            y_nuisance[test_index] = outcome_model.predict(X[test_index])
            if binary_treatment:
                treatment_model = HistGradientBoostingClassifier(
                    max_iter=120, max_depth=4, learning_rate=0.06,
                    l2_regularization=0.1, random_state=self.random_state + 101 + fold_index,
                )
                treatment_model.fit(X[train_index], d[train_index])
                d_nuisance[test_index] = treatment_model.predict_proba(X[test_index])[:, 1]
            else:
                treatment_model = HistGradientBoostingRegressor(
                    max_iter=120, max_depth=4, learning_rate=0.06,
                    l2_regularization=0.1, random_state=self.random_state + 101 + fold_index,
                )
                treatment_model.fit(X[train_index], d[train_index])
                d_nuisance[test_index] = treatment_model.predict(X[test_index])
        y_residual = y - y_nuisance
        d_residual = d - d_nuisance
        denominator = float(np.sum(d_residual ** 2))
        if denominator <= 1e-12:
            return None
        effect = float(np.sum(d_residual * y_residual) / denominator)
        for _, test_index in splits:
            fold_denominator = float(np.sum(d_residual[test_index] ** 2))
            if fold_denominator > 1e-12:
                fold_effects.append(float(
                    np.sum(d_residual[test_index] * y_residual[test_index])
                    / fold_denominator
                ))
        jacobian = float(np.mean(d_residual ** 2))
        influence = d_residual * (y_residual - effect * d_residual) / max(jacobian, 1e-12)
        standard_error = float(np.std(influence, ddof=1) / math.sqrt(len(influence)))
        z_value = effect / standard_error if standard_error > 1e-12 else float("inf")
        p_value = float(math.erfc(abs(z_value) / math.sqrt(2.0)))
        confidence_interval = [
            effect - 1.96 * standard_error,
            effect + 1.96 * standard_error,
        ]
        rng = np.random.default_rng(self.random_state + 701)
        placebo_effects = np.empty(200, dtype=float)
        for iteration in range(len(placebo_effects)):
            permuted = rng.permutation(d_residual)
            placebo_denominator = max(float(np.sum(permuted ** 2)), 1e-12)
            placebo_effects[iteration] = np.sum(permuted * y_residual) / placebo_denominator
        placebo_p = float(
            (1 + np.sum(np.abs(placebo_effects) >= abs(effect)))
            / (len(placebo_effects) + 1)
        )
        if binary_treatment:
            overlap_share = float(np.mean((d_nuisance >= 0.05) & (d_nuisance <= 0.95)))
            overlap_status = (
                "pass" if overlap_share >= 0.90 else (
                    "warning" if overlap_share >= 0.75 else "fail"
                )
            )
            overlap_evidence = (
                f"{overlap_share:.1%} 样本的交叉拟合倾向得分位于 [0.05, 0.95]。"
            )
        else:
            treatment_r2 = 1.0 - float(np.sum(d_residual ** 2)) / max(
                float(np.sum((d - np.mean(d)) ** 2)), 1e-12
            )
            residual_share = float(np.std(d_residual) / max(np.std(d), 1e-12))
            overlap_share = residual_share
            overlap_status = "pass" if residual_share >= 0.20 else (
                "warning" if residual_share >= 0.10 else "fail"
            )
            overlap_evidence = (
                f"处理残差标准差/原标准差={residual_share:.1%}，处理模型 R²={treatment_r2:.3f}。"
            )
        fold_mean = float(np.mean(fold_effects)) if fold_effects else effect
        fold_std = float(np.std(fold_effects, ddof=1)) if len(fold_effects) > 1 else 0.0
        fold_relative_std = fold_std / max(abs(fold_mean), standard_error, 1e-12)
        sign_agreement = float(np.mean(
            np.sign(fold_effects) == np.sign(effect)
        )) if fold_effects and effect != 0 else 0.0
        timing_explicit = any(
            token in str(problem).lower()
            for token in ("处理前", "干预前", "政策前", "事前", "基线协变量", "pre-treatment")
        )
        checks = [
            self._credibility_check(
                "causal_role_declaration", "因果角色显式声明", "pass",
                f"处理={dataset_name}.{treatment_column}；结果={dataset_name}.{outcome_column}；不会从相关性反推方向。",
            ),
            self._credibility_check(
                "causal_overlap", "处理重叠性/可辨识变异", overlap_status,
                overlap_evidence,
                "处理几乎可由协变量确定，效应依赖外推，应限制到共同支持域。"
                if overlap_status != "pass" else "",
            ),
            self._credibility_check(
                "causal_fold_stability", "处理效应跨折稳定性",
                "pass" if sign_agreement >= 0.8 and fold_relative_std <= 0.75 else (
                    "warning" if sign_agreement >= 0.6 and fold_relative_std <= 1.5 else "fail"
                ),
                f"折效应={np.round(fold_effects, 5).tolist()}；符号一致率={sign_agreement:.1%}。",
                "效应对样本切分敏感，应报告异质性或增加样本。"
                if sign_agreement < 0.8 or fold_relative_std > 0.75 else "",
            ),
            self._credibility_check(
                "causal_placebo", "处理残差安慰剂", "pass" if placebo_p < 0.05 else "warning",
                f"200 次处理残差置乱的双侧经验 p={placebo_p:.4f}。",
                "当前数据没有提供区别于随机处理排列的强效应证据；零效应仍可能是正确结论。"
                if placebo_p >= 0.05 else "",
            ),
            self._credibility_check(
                "pre_treatment_controls", "控制变量时间顺序",
                "pass" if timing_explicit else "warning",
                "题目已声明使用处理前/基线协变量。" if timing_explicit else
                "系统无法仅从表结构证明控制变量发生在处理之前。",
                "明确给出处理时间，并排除中介变量、碰撞变量和处理后的变量。"
                if not timing_explicit else "",
            ),
            self._credibility_check(
                "unobserved_confounding", "未观测混杂", "not_assessed",
                "观察数据无法仅靠算法检验无未观测混杂、SUTVA和一致性假设。",
                "使用随机化、自然实验、工具变量、负对照或正式敏感性分析增强识别。",
            ),
        ]
        failed = any(check["status"] == "fail" for check in checks)
        warned = any(check["status"] in {"warning", "not_assessed"} for check in checks)
        audit_status = "fail" if failed else ("warning" if warned else "pass")
        audit_label = "不可信" if failed else ("有条件可信" if warned else "可信")
        return _plain({
            "dataset": dataset_name,
            "method": "cross_fitted_orthogonal_partially_linear_effect",
            "treatment": treatment_column,
            "outcome": outcome_column,
            "treatment_type": "binary" if binary_treatment else "continuous",
            "treatment_mapping": treatment_mapping,
            "controls": control_columns,
            "n_samples": len(y),
            "validation": validation,
            "group_column": group_column,
            "effect": effect,
            "standard_error": standard_error,
            "confidence_interval_95": confidence_interval,
            "z_value": z_value,
            "p_value": p_value,
            "fold_effects": fold_effects,
            "placebo_p_value": placebo_p,
            "overlap_share": overlap_share,
            "credibility_audit": {
                "status": audit_status,
                "label": audit_label,
                "checks": checks,
                "decision": (
                    "存在关键识别或稳定性失败，不能作因果表述" if failed else
                    "估计通过可观测数据检查，但因果解释依赖不可由数据验证的识别假设"
                ),
            },
            "literature_basis": {
                "idea": "Neyman-orthogonal residualization with cross-fitting",
                "doi": "10.1111/ectj.12097",
            },
            "note": "该估计针对部分线性平均处理效应；显著性不能消除未观测混杂。",
        })

    def _run_bootstrap_uncertainty(self, target: Optional[str]) -> Optional[Dict[str, Any]]:
        subject = self._numeric_subject(target)
        if not subject:
            return None
        dataset_name, column = subject
        values = pd.to_numeric(self._datasets[dataset_name][column], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
        if len(values) < 10:
            return None
        rng = np.random.default_rng(self.random_state)
        if len(values) > 20_000:
            values = rng.choice(values, size=20_000, replace=False)
        resample_size = min(len(values), 5_000)
        bootstrap_means: List[np.ndarray] = []
        for _ in range(20):
            samples = rng.choice(values, size=(50, resample_size), replace=True)
            bootstrap_means.append(samples.mean(axis=1))
        means = np.concatenate(bootstrap_means)
        return _plain({
            "dataset": dataset_name,
            "variable": column,
            "method": "nonparametric_bootstrap",
            "iterations": len(means),
            "sample_size": len(values),
            "observed_mean": float(np.mean(values)),
            "observed_std": float(np.std(values, ddof=1)),
            "mean_confidence_interval_95": np.quantile(means, [0.025, 0.975]),
            "distribution_quantiles": {
                "q05": float(np.quantile(values, 0.05)),
                "q50": float(np.quantile(values, 0.50)),
                "q95": float(np.quantile(values, 0.95)),
            },
            "note": "这是对观测分布的不确定性估计；机理仿真仍需要状态转移规则。",
        })

    def _run_time_dynamics(self, target: Optional[str]) -> Optional[Dict[str, Any]]:
        selected = None
        explicit = self._numeric_subject(target)
        for dataset_name, profile in self._profiles.items():
            if not profile.datetime_columns:
                continue
            numeric = [col for col in profile.numeric_columns if col not in profile.id_candidates]
            if explicit and explicit[0] == dataset_name and explicit[1] in numeric:
                selected = (dataset_name, profile.datetime_columns[0], explicit[1])
                break
            if numeric:
                selected = (dataset_name, profile.datetime_columns[0], numeric[0])
                break
        if selected is None:
            return None
        dataset_name, time_col, value_col = selected
        source = _sample_frame(self._datasets[dataset_name], self.max_analysis_rows, self.random_state)
        time_values = pd.to_datetime(source[time_col], errors="coerce")
        values = pd.to_numeric(source[value_col], errors="coerce")
        series = pd.DataFrame({"time": time_values, "value": values}).dropna().groupby("time", as_index=False)["value"].mean().sort_values("time")
        if len(series) < 8:
            return None
        elapsed_days = (series["time"] - series["time"].iloc[0]).dt.total_seconds().to_numpy() / 86400
        slope, intercept = np.polyfit(elapsed_days, series["value"].to_numpy(), 1)
        fitted = intercept + slope * elapsed_days
        residual = series["value"].to_numpy() - fitted
        intervals = series["time"].diff().dt.total_seconds().dropna() / 86400
        autocorrelation = {}
        for lag in (1, 7, 12):
            if len(series) > lag + 3:
                value = series["value"].autocorr(lag=lag)
                if np.isfinite(value):
                    autocorrelation[str(lag)] = float(value)
        points = series.tail(500)
        return _plain({
            "dataset": dataset_name,
            "time_column": time_col,
            "variable": value_col,
            "n_time_points": len(series),
            "time_range": [series["time"].iloc[0], series["time"].iloc[-1]],
            "median_interval_days": float(intervals.median()) if len(intervals) else None,
            "linear_trend_per_day": float(slope),
            "residual_std": float(np.std(residual, ddof=1)),
            "autocorrelation": autocorrelation,
            "points": [{"time": str(time), "value": float(value)} for time, value in points.itertuples(index=False, name=None)],
            "note": "趋势和自相关是经验动力特征，不替代由机理推导的微分方程。",
        })

    def _run_hierarchical_additive_analysis(
        self,
        problem: str,
        target: Optional[Tuple[str, str]],
        relationships: Sequence[DatasetRelation],
    ) -> Optional[Dict[str, Any]]:
        """Analyze arbitrary parent/child distributions and co-movement.

        The routine is a reusable hierarchy compiler rather than a contest
        template.  It identifies an additive flow, a time key, an entity key,
        and an optional parent dimension; aggregates before analysis; removes
        trend and weekday effects; and applies BH-FDR to the remaining pairwise
        associations.
        """
        from scipy.stats import spearmanr
        from sklearn.linear_model import Ridge

        if target is None:
            return None
        dataset_name, target_column = target
        profile = self._profiles.get(dataset_name)
        if profile is None or not profile.datetime_columns:
            return None
        source = self._datasets[dataset_name]
        if source.attrs.get("aggregation_complete") is False:
            self._runtime_warnings.append(
                f"{dataset_name} 只有覆盖样本而非完整时间聚合；"
                "系统拒绝用它计算层级总量分布。"
            )
            return None
        represented_source_rows = (
            int(source.attrs.get("source_rows", len(source)))
            if source.attrs.get("aggregation_complete") is True
            else len(source)
        )
        if len(source) > 2_000_000:
            self._runtime_warnings.append(
                f"{dataset_name} 有 {len(source):,} 行，超过层级销量分析 2,000,000 行硬上限；"
                "系统不会对交易行随机抽样后声称得到总量分布。"
            )
            return None
        time_column = sorted(
            profile.datetime_columns,
            key=lambda column: (
                0 if any(token in str(column).lower() for token in ("日期", "date", "day")) else 1,
                len(str(column)),
            ),
        )[0]
        bindings = self._discover_mentioned_group_bindings(problem)
        if len(bindings) < 2:
            return None
        bindings = sorted(
            bindings,
            key=lambda item: int(item["group_column_hint"]["unique_count"]),
        )
        materialized = None
        for parent_binding in bindings:
            parent_hint = parent_binding["group_column_hint"]
            category_column = str(parent_hint["column"])
            for child_binding in reversed(bindings):
                child_hint = child_binding["group_column_hint"]
                item_column = str(child_hint["column"])
                if (
                    item_column == category_column
                    or int(child_hint["unique_count"]) <= int(parent_hint["unique_count"])
                ):
                    continue
                if item_column in source.columns and category_column in source.columns:
                    candidate_working = source[
                        [time_column, item_column, category_column, target_column]
                    ].copy()
                    materialized = (
                        candidate_working, dataset_name, None,
                        item_column, category_column,
                    )
                    break
                for relation in relationships:
                    if dataset_name not in {relation.left_dataset, relation.right_dataset}:
                        continue
                    if relation.left_dataset == dataset_name:
                        base_key, other_key, other_name = (
                            relation.left_key, relation.right_key, relation.right_dataset
                        )
                        other_unique = relation.relationship in {"many_to_one", "one_to_one"}
                    else:
                        base_key, other_key, other_name = (
                            relation.right_key, relation.left_key, relation.left_dataset
                        )
                        other_unique = relation.relationship in {"one_to_many", "one_to_one"}
                    if not other_unique or base_key not in source.columns:
                        continue
                    other = self._datasets[other_name]
                    if category_column not in other.columns:
                        continue
                    lookup_columns = [other_key, category_column]
                    child_is_direct = item_column in source.columns
                    if not child_is_direct:
                        if item_column not in other.columns:
                            continue
                        lookup_columns.append(item_column)
                    lookup = other[list(dict.fromkeys(lookup_columns))].dropna().drop_duplicates()
                    if bool((lookup.groupby(other_key).size() > 1).any()):
                        continue
                    base_columns = [time_column, target_column, base_key]
                    if child_is_direct:
                        base_columns.append(item_column)
                    candidate_working = source[list(dict.fromkeys(base_columns))].merge(
                        lookup.drop_duplicates(other_key),
                        left_on=base_key, right_on=other_key,
                        how="left", validate="many_to_one",
                    )
                    if candidate_working[[item_column, category_column]].dropna().empty:
                        continue
                    if bool(
                        (
                            candidate_working[[item_column, category_column]].dropna()
                            .drop_duplicates().groupby(item_column)[category_column].nunique()
                            > 1
                        ).any()
                    ):
                        continue
                    join_audit = {
                        "relationship": f"{dataset_name}.{base_key}->{other_name}.{other_key}",
                        "confidence": relation.confidence,
                        "matched_share": float(
                            candidate_working[category_column].notna().mean()
                        ),
                        "validation": "many_to_one_and_child_to_parent_function",
                    }
                    materialized = (
                        candidate_working, other_name, join_audit,
                        item_column, category_column,
                    )
                    break
                if materialized is not None:
                    break
            if materialized is not None:
                break
        if materialized is None:
            return None
        working, category_source, join_audit, item_column, category_column = materialized
        hierarchy_pairs = working[[item_column, category_column]].dropna().drop_duplicates()
        if hierarchy_pairs.empty or bool(
            (hierarchy_pairs.groupby(item_column)[category_column].nunique() > 1).any()
        ):
            return None

        working["__date"] = pd.to_datetime(working[time_column], errors="coerce").dt.floor("D")
        working["__value"] = pd.to_numeric(working[target_column], errors="coerce")
        working = working.dropna(subset=["__date", "__value", item_column, category_column])
        working = working[working["__value"] >= 0]
        if working.empty:
            return None
        daily_item = (
            working.groupby(["__date", category_column, item_column], observed=True, sort=True)["__value"]
            .sum().rename("value").reset_index()
        )
        daily_category = (
            daily_item.groupby(["__date", category_column], observed=True, sort=True)["value"]
            .sum().reset_index()
        )
        if daily_category["__date"].nunique() < 21:
            return None

        def summaries(frame: pd.DataFrame, group_column: str, limit: int) -> List[Dict[str, Any]]:
            grouped = frame.groupby(group_column, observed=True)["value"]
            output = pd.DataFrame({
                "total": grouped.sum(), "daily_mean": grouped.mean(),
                "daily_std": grouped.std(ddof=1).fillna(0.0),
                "median": grouped.median(),
                "q10": grouped.quantile(0.10), "q90": grouped.quantile(0.90),
                "active_days": grouped.count(),
            })
            output["coefficient_of_variation"] = (
                output["daily_std"] / output["daily_mean"].replace(0, np.nan)
            )
            output = output.sort_values("total", ascending=False).head(limit).reset_index()
            return _plain(output.replace([np.inf, -np.inf], np.nan).to_dict("records"))

        category_summary = summaries(daily_category, category_column, 100)
        item_summary = summaries(daily_item, item_column, 200)
        item_totals = daily_item.groupby(item_column, observed=True)["value"].sum().sort_values(ascending=False)
        shares = item_totals / max(float(item_totals.sum()), 1e-12)
        concentration = {
            "item_count": int(len(item_totals)),
            "child_count": int(len(item_totals)),
            "hhi": float(np.square(shares).sum()),
            "top_10_share": float(shares.head(10).sum()),
            "top_20_share": float(shares.head(20).sum()),
        }

        all_dates = pd.date_range(
            daily_item["__date"].min(), daily_item["__date"].max(), freq="D"
        )

        def residual_associations(
            frame: pd.DataFrame,
            group_column: str,
            groups: Sequence[Any],
            maximum_pairs: int,
        ) -> List[Dict[str, Any]]:
            pivot = frame[frame[group_column].isin(groups)].pivot_table(
                index="__date", columns=group_column, values="value",
                aggfunc="sum", fill_value=0.0,
            ).reindex(all_dates, fill_value=0.0)
            if pivot.shape[1] < 2:
                return []
            elapsed = np.arange(len(pivot), dtype=float)
            weekday = np.eye(7, dtype=float)[pivot.index.dayofweek]
            design_matrix = np.column_stack([elapsed, weekday])
            residual = pd.DataFrame(index=pivot.index)
            for column in pivot.columns:
                values = pivot[column].to_numpy(dtype=float)
                fitted = Ridge(alpha=10.0).fit(design_matrix, values).predict(design_matrix)
                residual[str(column)] = values - fitted
            tested: List[Dict[str, Any]] = []
            columns = list(residual.columns)
            for left_index, left in enumerate(columns):
                for right in columns[left_index + 1:]:
                    coefficient, p_value = spearmanr(residual[left], residual[right])
                    if not np.isfinite(coefficient) or not np.isfinite(p_value):
                        continue
                    tested.append({
                        "left": str(left), "right": str(right),
                        "residual_spearman": float(coefficient),
                        "p_value": float(p_value), "q_value": None,
                        "significant": False,
                    })
            order = sorted(range(len(tested)), key=lambda index: tested[index]["p_value"])
            running = 1.0
            for reverse_rank in range(len(order) - 1, -1, -1):
                index = order[reverse_rank]
                rank = reverse_rank + 1
                adjusted = min(running, tested[index]["p_value"] * len(order) / rank)
                running = adjusted
                tested[index]["q_value"] = float(adjusted)
                tested[index]["significant"] = adjusted <= 0.05
            tested.sort(
                key=lambda item: (
                    not item["significant"], -abs(item["residual_spearman"]), item["left"], item["right"]
                )
            )
            return tested[:maximum_pairs]

        category_groups = [item[category_column] for item in category_summary]
        top_items = list(item_totals.head(30).index)
        category_associations = residual_associations(
            daily_category, category_column, category_groups, 50
        )
        item_associations = residual_associations(
            daily_item, item_column, top_items, 80
        )
        weekday = (
            daily_category.assign(weekday=daily_category["__date"].dt.dayofweek)
            .groupby([category_column, "weekday"], observed=True)["value"].mean()
            .reset_index(name="mean_value")
        )
        weekday["mean_sales"] = weekday["mean_value"]
        significant_categories = sum(item["significant"] for item in category_associations)
        significant_items = sum(item["significant"] for item in item_associations)
        return _plain({
            "status": "executed",
            "method": "hierarchical_daily_aggregation_residual_association_fdr",
            "dataset": dataset_name,
            "target": target_column,
            "time_column": time_column,
            "category_column": category_column,
            "category_column_source": category_source,
            "item_column": item_column,
            "parent_dimension": category_column,
            "parent_dimension_source": category_source,
            "child_dimension": item_column,
            "source_rows_aggregated": represented_source_rows,
            "research_representation": source.attrs.get(
                "research_representation", "complete_rows"
            ),
            "daily_item_rows": len(daily_item),
            "daily_category_rows": len(daily_category),
            "join_audit": join_audit,
            "category_summary": category_summary,
            "item_summary": item_summary,
            "parent_summary": category_summary,
            "child_summary": item_summary,
            "concentration": concentration,
            "category_associations": category_associations,
            "item_associations": item_associations,
            "parent_associations": category_associations,
            "child_associations": item_associations,
            "weekday_profile": weekday.to_dict("records"),
            "credibility_audit": {
                "status": "pass" if significant_categories or significant_items else "warning",
                "label": "通过" if significant_categories or significant_items else "谨慎使用",
                "checks": [
                    {
                        "id": "aggregation_before_analysis", "status": "pass",
                        "evidence": (
                            f"先将 {represented_source_rows:,} 行聚合为 {len(daily_item):,} 个"
                            f"日×{category_column}×{item_column}观测。"
                        ),
                    },
                    {
                        "id": "trend_weekday_residualization", "status": "pass",
                        "evidence": "相互关系在逐组移除线性趋势和星期效应后计算。",
                    },
                    {
                        "id": "hierarchical_fdr", "status": "pass" if significant_categories or significant_items else "warning",
                        "evidence": (
                            f"FDR显著的上层/下层对象对为 "
                            f"{significant_categories}/{significant_items}。"
                        ),
                    },
                    {
                        "id": "zero_fill_scope", "status": "warning",
                        "evidence": "未出现的日×对象组合按零流量处理；数据漏报会破坏该假设。",
                    },
                ],
                "decision": (
                    "可报告分布、集中度、星期效应及通过FDR的剩余联动；不得作因果解释。"
                ),
            },
            "note": (
                f"{category_column}/{item_column} 联动是去除线性趋势和星期效应后的"
                "日残差 Spearman 关系；共同外部冲击和未观测控制变量仍可能造成混杂。"
            ),
        })

    def _run_grouped_time_forecast(
        self,
        problem: str,
        target: Optional[Tuple[str, str]],
        relationships: Sequence[DatasetRelation],
        requested_grain: Optional[str] = None,
        group_column_hint: Optional[Mapping[str, Any]] = None,
        source_task_ids: Optional[Sequence[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Compile requested time/entity grain before forecasting additive outcomes.

        Transaction rows are not interchangeable with daily category totals.
        This compiler follows verified dimension keys, aggregates the complete
        bounded analysis frame first, and then evaluates competing time models
        on a terminal block. No random row sample is allowed before aggregation.
        """
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error

        if target is None:
            return None
        dataset_name, target_column = target
        profile = self._profiles.get(dataset_name)
        if profile is None or not profile.datetime_columns:
            return None
        text = str(problem).lower()
        target_norm = _normalise_name(target_column)
        additive_target = any(
            token in target_norm
            for token in ("sales", "demand", "销量", "销售量", "需求", "数量", "总量", "产量")
        )
        requests_time_total = bool(re.search(
            r"(?:日|每日|每天|逐日|周|每周|月|每月)[^。；\n]{0,18}(?:总量|销量|销售量|需求|数量)|"
            r"(?:总量|销量|销售量|需求量)[^。；\n]{0,18}(?:日|每日|每天|逐日|周|月)",
            text,
        )) or bool(
            any(token in text for token in ("补货", "订购", "replenish"))
            and re.search(r"(?:未来|\d{1,2}\s*[月/-]\s*\d{1,2}|一周|次日|明日)", text)
        )
        requests_group = bool(group_column_hint) or requested_grain in {"category", "item"} or bool(re.search(
            r"(?:各|按|不同)[^。；\n]{0,10}(?:品类|分类|单品|产品|商品|item|product|category)",
            text,
        ))
        if not (additive_target and requests_time_total and requests_group):
            return None

        source = self._datasets[dataset_name]
        if source.attrs.get("aggregation_complete") is False:
            self._runtime_warnings.append(
                f"{dataset_name} 只有覆盖样本而非完整时间聚合；"
                "系统拒绝从样本推断分组总量预测。"
            )
            return None
        represented_source_rows = (
            int(source.attrs.get("source_rows", len(source)))
            if source.attrs.get("aggregation_complete") is True
            else len(source)
        )
        if len(source) > 2_000_000:
            self._runtime_warnings.append(
                f"{dataset_name} 有 {len(source):,} 行，超过分组总量编译的 2,000,000 行硬上限；"
                "系统拒绝先随机抽样再伪装成总量。"
            )
            return None
        time_column = sorted(
            profile.datetime_columns,
            key=lambda column: (
                0 if any(token in str(column).lower() for token in ("日期", "date", "day")) else 1,
                len(str(column)),
            ),
        )[0]
        wants_category = requested_grain == "category" or (
            requested_grain is None
            and any(token in text for token in ("品类", "分类", "category"))
        )
        wants_item = requested_grain == "item" or (
            requested_grain is None
            and not wants_category
            and any(token in text for token in ("单品", "产品", "商品", "item", "product", "sku"))
        )
        category_tokens = ("品类", "分类", "category")
        item_tokens = ("单品", "产品", "商品", "item", "product", "sku")

        def role_columns(frame: pd.DataFrame, tokens: Sequence[str]) -> List[str]:
            return [
                str(column) for column in frame.columns
                if str(column) not in {target_column, time_column}
                and any(token in str(column).lower() for token in tokens)
            ]

        desired_tokens = category_tokens if wants_category else item_tokens
        hinted_dataset = str((group_column_hint or {}).get("dataset") or "")
        hinted_column = str((group_column_hint or {}).get("column") or "")
        direct_groups = role_columns(source, desired_tokens)
        group_column = (
            hinted_column
            if hinted_dataset == dataset_name and hinted_column in source.columns
            else next(
                (
                    column for column in direct_groups
                    if any(token in column for token in ("名称", "name"))
                ),
                direct_groups[0] if direct_groups else None,
            )
        )
        working_columns = [time_column, target_column]
        if group_column:
            working_columns.append(group_column)
        working = source[working_columns].copy()
        group_source = dataset_name
        join_audit: Optional[Dict[str, Any]] = None

        if group_column is None:
            candidates: List[Tuple[float, DatasetRelation, str, str, str]] = []
            for relation in relationships:
                if dataset_name not in {relation.left_dataset, relation.right_dataset}:
                    continue
                if relation.left_dataset == dataset_name:
                    base_key, other_key, other_name = (
                        relation.left_key, relation.right_key, relation.right_dataset
                    )
                    other_unique = relation.relationship in {"many_to_one", "one_to_one"}
                else:
                    base_key, other_key, other_name = (
                        relation.right_key, relation.left_key, relation.left_dataset
                    )
                    other_unique = relation.relationship in {"one_to_many", "one_to_one"}
                if not other_unique or base_key not in source.columns:
                    continue
                if hinted_dataset and other_name != hinted_dataset:
                    continue
                other = self._datasets[other_name]
                candidate_columns = (
                    [hinted_column]
                    if hinted_column and hinted_column in other.columns
                    else role_columns(other, desired_tokens)
                )
                for candidate_column in candidate_columns:
                    score = relation.confidence + (
                        5.0 if any(token in candidate_column for token in ("名称", "name")) else 0.0
                    )
                    candidates.append((score, relation, base_key, other_key, candidate_column))
            if candidates:
                _, relation, base_key, other_key, group_column = max(
                    candidates, key=lambda item: item[0]
                )
                other_name = (
                    relation.right_dataset if relation.left_dataset == dataset_name
                    else relation.left_dataset
                )
                lookup = self._datasets[other_name][[other_key, group_column]].dropna().drop_duplicates()
                ambiguity = lookup.groupby(other_key, sort=False)[group_column].nunique(dropna=True)
                if bool((ambiguity > 1).any()):
                    self._runtime_warnings.append(
                        f"无法按 {group_column} 聚合：{other_name}.{other_key} 对应多个分类值。"
                    )
                    return None
                lookup = lookup.drop_duplicates(other_key)
                working = source[[time_column, target_column, base_key]].merge(
                    lookup, left_on=base_key, right_on=other_key, how="left", validate="many_to_one",
                )
                group_source = other_name
                join_audit = {
                    "relationship": f"{dataset_name}.{base_key}->{other_name}.{other_key}",
                    "confidence": relation.confidence,
                    "matched_share": float(working[group_column].notna().mean()),
                    "validation": "many_to_one",
                }
        if group_column is None:
            return None

        working["__time"] = pd.to_datetime(working[time_column], errors="coerce").dt.floor("D")
        working["__target"] = pd.to_numeric(working[target_column], errors="coerce")
        working = working.dropna(subset=["__time", "__target", group_column])
        if working.empty or working[group_column].nunique(dropna=True) > 500:
            return None
        daily = (
            working.groupby(["__time", group_column], observed=True, sort=True)["__target"]
            .sum().rename("value").reset_index()
        )
        if daily["__time"].nunique() < 21:
            return None
        global_end = daily["__time"].max()
        explicit_dates: List[pd.Timestamp] = []
        range_pattern = re.compile(
            r"(?:(20\d{2})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)?\s*"
            r"(?:[-—~～至到])\s*(\d{1,2})\s*(?:日|号)"
        )
        occupied_spans: List[Tuple[int, int]] = []
        for match in range_pattern.finditer(text):
            year_text, month_text, start_text, end_text = match.groups()
            year = int(year_text) if year_text else int(global_end.year)
            try:
                start = pd.Timestamp(year=year, month=int(month_text), day=int(start_text))
                end = pd.Timestamp(year=year, month=int(month_text), day=int(end_text))
            except ValueError:
                continue
            explicit_dates.extend(pd.date_range(start, end, freq="D").tolist())
            occupied_spans.append(match.span())
        day_pattern = re.compile(
            r"(?:(20\d{2})\s*年\s*)?(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)"
        )
        for match in day_pattern.finditer(text):
            if any(start <= match.start() < end for start, end in occupied_spans):
                continue
            year_text, month_text, day_text = match.groups()
            year = int(year_text) if year_text else int(global_end.year)
            try:
                explicit_dates.append(pd.Timestamp(
                    year=year, month=int(month_text), day=int(day_text)
                ))
            except ValueError:
                continue
        explicit_future_dates = sorted({
            date for date in explicit_dates
            if global_end < date <= global_end + pd.Timedelta(days=90)
        })
        if explicit_future_dates:
            future_dates = pd.DatetimeIndex(explicit_future_dates)
        else:
            horizon_match = re.search(r"未来\s*(\d+)\s*(?:天|日)", text)
            horizon = (
                int(horizon_match.group(1)) if horizon_match
                else (7 if "一周" in text or "未来一周" in text else 7)
            )
            horizon = min(max(horizon, 1), 90)
            future_dates = pd.date_range(
                global_end + pd.Timedelta(days=1), periods=horizon, freq="D"
            )
        future_dates = future_dates[
            (future_dates > global_end)
            & (future_dates <= global_end + pd.Timedelta(days=90))
        ]
        if len(future_dates) == 0:
            self._runtime_warnings.append(
                f"{requested_grain or '分组'}预测请求日期不晚于数据末日 {global_end.date()}，"
                "系统拒绝把历史拟合冒充未来预测。"
            )
            return None
        horizon = len(future_dates)
        forecasts: List[Dict[str, Any]] = []
        validation_rows: List[Tuple[float, float, float]] = []
        selected_models: Dict[str, int] = {}
        skipped_groups = 0

        def design(index: pd.DatetimeIndex, origin: pd.Timestamp) -> np.ndarray:
            elapsed = (index - origin).days.to_numpy(dtype=float)
            weekday = np.eye(7, dtype=float)[index.dayofweek]
            return np.column_stack([elapsed, weekday])

        for group_value, group_data in daily.groupby(group_column, observed=True, sort=True):
            series = group_data.set_index("__time")["value"].sort_index()
            full_index = pd.date_range(series.index.min(), global_end, freq="D")
            series = series.reindex(full_index, fill_value=0.0).astype(float)
            if len(series) < 21:
                skipped_groups += 1
                continue
            holdout = min(28, max(7, len(series) // 5))
            train, test = series.iloc[:-holdout], series.iloc[-holdout:]
            seasonal_prediction = series.shift(7).reindex(test.index)
            fallback = float(train.tail(28).median())
            seasonal_prediction = seasonal_prediction.fillna(fallback).clip(lower=0)
            ridge = Ridge(alpha=10.0)
            ridge.fit(design(train.index, train.index[0]), train.to_numpy())
            ridge_prediction = np.maximum(
                0.0, ridge.predict(design(test.index, train.index[0]))
            )
            candidate_predictions = {
                "seasonal_naive_7": seasonal_prediction.to_numpy(dtype=float),
                "ridge_trend_weekday": ridge_prediction,
            }
            errors = {
                name: float(np.sqrt(mean_squared_error(test, prediction)))
                for name, prediction in candidate_predictions.items()
            }
            selected_name = min(errors, key=errors.get)
            selected_models[selected_name] = selected_models.get(selected_name, 0) + 1
            selected_validation = candidate_predictions[selected_name]
            residuals = test.to_numpy(dtype=float) - selected_validation
            interval_radius = float(np.quantile(np.abs(residuals), 0.90))
            if selected_name == "seasonal_naive_7":
                extended = pd.concat([
                    series,
                    pd.Series(index=future_dates, dtype=float),
                ])
                future_prediction = []
                for date in future_dates:
                    lagged = extended.get(date - pd.Timedelta(days=7), fallback)
                    value = fallback if pd.isna(lagged) else float(lagged)
                    future_prediction.append(max(0.0, value))
                    extended.loc[date] = future_prediction[-1]
            else:
                final_model = Ridge(alpha=10.0)
                final_model.fit(design(series.index, series.index[0]), series.to_numpy())
                future_prediction = np.maximum(
                    0.0, final_model.predict(design(future_dates, series.index[0]))
                ).tolist()
            baseline_rmse = errors["seasonal_naive_7"]
            validation_rows.extend(
                (float(actual), float(predicted), float(baseline))
                for actual, predicted, baseline in zip(
                    test.to_numpy(), selected_validation,
                    candidate_predictions["seasonal_naive_7"],
                )
            )
            for date, value in zip(future_dates, future_prediction):
                forecasts.append({
                    "group": str(group_value), "date": str(date.date()),
                    "forecast": float(value),
                    "lower_90": float(max(0.0, value - interval_radius)),
                    "upper_90": float(value + interval_radius),
                    "selected_model": selected_name,
                    "validation_rmse": errors[selected_name],
                    "seasonal_baseline_rmse": baseline_rmse,
                })
        if not forecasts or not validation_rows:
            return None
        actual = np.asarray([row[0] for row in validation_rows])
        predicted = np.asarray([row[1] for row in validation_rows])
        baseline = np.asarray([row[2] for row in validation_rows])
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        baseline_rmse = float(np.sqrt(mean_squared_error(actual, baseline)))
        mae = float(mean_absolute_error(actual, predicted))
        audit_status = "pass" if rmse <= baseline_rmse * 0.98 else "warning"
        return _plain({
            "status": "executed",
            "method": "grain_compilation_then_grouped_terminal_backtest",
            "requested_grain": requested_grain or ("category" if wants_category else "item"),
            "group_column_binding": dict(group_column_hint or {}),
            "request_text": str(problem),
            "source_task_ids": list(source_task_ids or []),
            "dataset": dataset_name,
            "target": target_column,
            "time_column": time_column,
            "group_column": group_column,
            "group_column_source": group_source,
            "aggregation": "daily_sum_before_any_sampling",
            "source_rows_aggregated": represented_source_rows,
            "research_representation": source.attrs.get(
                "research_representation", "complete_rows"
            ),
            "daily_group_rows": len(daily),
            "groups_forecast": len({item["group"] for item in forecasts}),
            "skipped_groups": skipped_groups,
            "horizon_days": horizon,
            "forecast_period": [str(future_dates[0].date()), str(future_dates[-1].date())],
            "metrics": {
                "terminal_block_rmse": rmse,
                "terminal_block_mae": mae,
                "seasonal_naive_rmse": baseline_rmse,
            },
            "selected_model_counts": selected_models,
            "dimension_join_audit": join_audit,
            "forecasts": forecasts,
            "credibility_audit": {
                "status": audit_status,
                "label": "通过" if audit_status == "pass" else "谨慎使用",
                "checks": [
                    {
                        "id": "aggregation_before_sampling", "status": "pass",
                        "evidence": (
                            f"先聚合 {represented_source_rows:,} 行为 {len(daily):,} 个"
                            "日×组观测，未做交易行随机抽样。"
                        ),
                    },
                    {
                        "id": "terminal_time_backtest", "status": audit_status,
                        "evidence": f"RMSE={rmse:.6g}；7日季节基线={baseline_rmse:.6g}。",
                    },
                    {
                        "id": "zero_fill_scope", "status": "warning",
                        "evidence": "日粒度缺行按零销量处理；若原始系统遗漏交易而非真实零销量，该假设会失效。",
                    },
                ],
                "decision": (
                    "分组总量预测不劣于季节基线，可作为后续决策输入并保留区间。"
                    if audit_status == "pass" else
                    "复杂候选未稳定超过季节基线，输出已逐组回退到验证更优者。"
                ),
            },
            "note": (
                "预测对象是题面要求的日×组总量，不是随机抽取的单条交易；"
                "区间来自末段回测绝对残差，仍不覆盖结构突变。"
            ),
        })

    def _run_prescriptive_replenishment_pricing(
        self,
        problem: str,
        grouped_forecast: Mapping[str, Any],
        relationships: Sequence[DatasetRelation],
        supporting_forecasts: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Compile forecast scenarios into a reusable multiple-choice MILP.

        The domain adapter only supplies empirically identified payoffs. The
        actual selection is delegated to the generic, independently checked
        mixed-integer backend. Price changes are allowed only when a negative
        observational elasticity is stable across time halves; otherwise price
        is held at its recent reference level.
        """
        from sklearn.linear_model import Ridge

        if not grouped_forecast or not any(
            token in str(problem).lower()
            for token in ("补货", "订购", "库存", "定价", "价格决策", "replenish", "pricing")
        ):
            return None
        dataset_name = str(grouped_forecast.get("dataset", ""))
        target_column = str(grouped_forecast.get("target", ""))
        time_column = str(grouped_forecast.get("time_column", ""))
        group_column = str(grouped_forecast.get("group_column", ""))
        group_source = str(grouped_forecast.get("group_column_source", dataset_name))
        requested_grain = str(grouped_forecast.get("requested_grain") or "group")
        selection_match = re.search(
            r"(?:总数|数量)[^。；\n\d]{0,12}?(?:控制在|为|介于)?\s*(\d+)\s*[-—~～至到]\s*(\d+)\s*个",
            str(problem),
        ) or re.search(
            r"(?:控制在|介于)\s*(\d+)\s*[-—~～至到]\s*(\d+)\s*个",
            str(problem),
        )
        selection_bounds = None
        if selection_match:
            lower_count, upper_count = map(int, selection_match.groups())
            if 0 <= lower_count <= upper_count <= 500:
                selection_bounds = (lower_count, upper_count)
        display_match = re.search(
            r"(?:最小陈列量|最小订购量|最低订购量|最小补货量|最小配置量|"
            r"最低配置量|最小分配量|最低分配量|每(?:个|项|台|站|点)[^。；\n]{0,8}至少)"
            r"\s*(\d+(?:\.\d+)?)\s*(?:千克|公斤|kg|吨|台|辆|人|小时|件|个|份)?",
            str(problem).lower(),
        )
        minimum_display = float(display_match.group(1)) if display_match else 0.0
        if dataset_name not in self._datasets or group_source not in self._datasets:
            return None

        def numeric_measurement(
            frame: pd.DataFrame, preferred_tokens: Sequence[str], excluded: Sequence[str] = (),
        ) -> Optional[str]:
            ranked: List[Tuple[int, int, str]] = []
            excluded_set = set(map(str, excluded))
            for column in frame.select_dtypes(include=[np.number]).columns:
                name = str(column)
                if name in excluded_set or _is_explicit_identifier_name(name):
                    continue
                lowered = name.lower()
                for rank, token in enumerate(preferred_tokens):
                    if token in lowered:
                        ranked.append((rank, len(name), name))
                        break
            return min(ranked)[2] if ranked else None

        master = self._datasets[group_source]

        def attach_group(
            source_name: str, columns: Sequence[str],
        ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]]]:
            frame = self._datasets[source_name]
            requested = list(dict.fromkeys(str(column) for column in columns))
            if group_column in frame.columns:
                return frame[requested + ([group_column] if group_column not in requested else [])].copy(), {
                    "dataset": source_name, "strategy": "direct_group_column",
                }
            best: Optional[Tuple[float, str, str, DatasetRelation]] = None
            for relation in relationships:
                if {source_name, group_source} != {relation.left_dataset, relation.right_dataset}:
                    continue
                if relation.left_dataset == source_name:
                    source_key, master_key = relation.left_key, relation.right_key
                    master_unique = relation.relationship in {"many_to_one", "one_to_one"}
                else:
                    source_key, master_key = relation.right_key, relation.left_key
                    master_unique = relation.relationship in {"one_to_many", "one_to_one"}
                if master_unique and source_key in frame.columns and master_key in master.columns:
                    candidate = (relation.confidence, source_key, master_key, relation)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
            if best is None or group_column not in master.columns:
                return None, None
            _, source_key, master_key, relation = best
            lookup = master[[master_key, group_column]].dropna().drop_duplicates()
            if bool((lookup.groupby(master_key)[group_column].nunique() > 1).any()):
                return None, None
            lookup = lookup.drop_duplicates(master_key)
            required = list(dict.fromkeys(requested + [source_key]))
            joined = frame[required].merge(
                lookup, left_on=source_key, right_on=master_key,
                how="left", validate="many_to_one",
            )
            return joined, {
                "dataset": source_name,
                "strategy": "verified_dimension_join",
                "relationship": f"{source_name}.{source_key}->{group_source}.{master_key}",
                "confidence": relation.confidence,
                "matched_share": float(joined[group_column].notna().mean()),
            }

        sales_source = self._datasets[dataset_name]
        price_column = numeric_measurement(
            sales_source,
            ("销售单价", "售价", "销售价格", "price", "单价"),
            (target_column,),
        )
        if price_column is None:
            return None
        sales_view, sales_join_audit = attach_group(
            dataset_name, [time_column, target_column, price_column]
        )
        if sales_view is None:
            return None
        sales_view["__time"] = pd.to_datetime(sales_view[time_column], errors="coerce").dt.floor("D")
        sales_view["__quantity"] = pd.to_numeric(sales_view[target_column], errors="coerce")
        sales_view["__price"] = pd.to_numeric(sales_view[price_column], errors="coerce")
        sales_view = sales_view.dropna(subset=["__time", "__quantity", "__price", group_column])
        sales_view = sales_view[(sales_view["__quantity"] >= 0) & (sales_view["__price"] > 0)]
        if sales_view.empty:
            return None
        eligible_groups: Optional[set] = None
        if selection_bounds:
            latest_sales_day = sales_view["__time"].max()
            recent_start = latest_sales_day - pd.Timedelta(days=6)
            eligible_groups = set(
                sales_view.loc[
                    sales_view["__time"].between(recent_start, latest_sales_day),
                    group_column,
                ].astype(str)
            )
        sales_view["__revenue_proxy"] = sales_view["__quantity"] * sales_view["__price"]
        daily = sales_view.groupby(["__time", group_column], observed=True, sort=True).agg(
            demand=("__quantity", "sum"),
            revenue_proxy=("__revenue_proxy", "sum"),
        ).reset_index()
        daily["price"] = daily["revenue_proxy"] / daily["demand"].replace(0, np.nan)
        daily = daily.dropna(subset=["price"])

        elasticity: Dict[str, Dict[str, Any]] = {}
        price_summary: Dict[str, Dict[str, float]] = {}
        for group, group_data in daily.groupby(group_column, observed=True, sort=True):
            group_data = group_data.sort_values("__time")
            recent = group_data.tail(min(56, len(group_data)))
            prices = recent["price"].to_numpy(dtype=float)
            price_summary[str(group)] = {
                "reference": float(np.median(prices)),
                "lower": float(np.quantile(prices, 0.10)),
                "upper": float(np.quantile(prices, 0.90)),
            }
            record = {
                "coefficient": None, "first_half": None, "second_half": None,
                "relative_price_range": 0.0, "eligible_for_price_decision": False,
                "reason": "insufficient_price_demand_history",
            }
            if len(group_data) >= 35 and group_data["price"].nunique() >= 5:
                origin = group_data["__time"].iloc[0]

                def fit_coefficient(part: pd.DataFrame) -> float:
                    elapsed = (part["__time"] - origin).dt.days.to_numpy(dtype=float)
                    weekday = np.eye(7)[part["__time"].dt.dayofweek.to_numpy()]
                    design_matrix = np.column_stack([
                        np.log(part["price"].to_numpy(dtype=float)), elapsed, weekday,
                    ])
                    outcome = np.log1p(part["demand"].to_numpy(dtype=float))
                    model = Ridge(alpha=2.0).fit(design_matrix, outcome)
                    return float(model.coef_[0])

                midpoint = len(group_data) // 2
                coefficient = fit_coefficient(group_data)
                first = fit_coefficient(group_data.iloc[:midpoint])
                second = fit_coefficient(group_data.iloc[midpoint:])
                relative_range = (
                    price_summary[str(group)]["upper"] - price_summary[str(group)]["lower"]
                ) / max(price_summary[str(group)]["reference"], 1e-12)
                eligible = (
                    coefficient < -0.05 and first < 0 and second < 0
                    and relative_range >= 0.02
                )
                record.update({
                    "coefficient": coefficient,
                    "first_half": first,
                    "second_half": second,
                    "relative_price_range": relative_range,
                    "eligible_for_price_decision": eligible,
                    "reason": (
                        "stable_negative_observational_elasticity"
                        if eligible else "elasticity_not_stably_negative"
                    ),
                })
            elasticity[str(group)] = record

        cost_candidates: List[Tuple[int, str, str]] = []
        for candidate_name, frame in self._datasets.items():
            cost_column = numeric_measurement(
                frame, ("批发价格", "采购价格", "进价", "单位成本", "成本", "wholesale", "cost")
            )
            if cost_column:
                # Dated external costs preserve the actual price-cost time
                # relation; static master costs remain a valid decision fallback.
                is_external = candidate_name != dataset_name
                has_time = bool(self._profiles[candidate_name].datetime_columns)
                rank = 0 if is_external and has_time else (1 if is_external else 2)
                cost_candidates.append((rank, candidate_name, cost_column))
        cost_by_group: Dict[str, float] = {}
        cost_audit = None
        cost_dataset_used: Optional[str] = None
        cost_column_used: Optional[str] = None
        cost_plus_relationship: List[Dict[str, Any]] = []
        if cost_candidates:
            _, cost_dataset, cost_column = min(cost_candidates)
            cost_dataset_used = cost_dataset
            cost_column_used = cost_column
            cost_profile = self._profiles[cost_dataset]
            selected_columns = [cost_column]
            cost_time = cost_profile.datetime_columns[0] if cost_profile.datetime_columns else None
            if cost_time:
                selected_columns.append(cost_time)
            cost_view, cost_audit = attach_group(cost_dataset, selected_columns)
            if cost_view is not None:
                cost_view["__cost"] = pd.to_numeric(cost_view[cost_column], errors="coerce")
                cost_view = cost_view.dropna(subset=["__cost", group_column])
                cost_history_view = cost_view
                if cost_time:
                    cost_history_view = cost_view.copy()
                    cost_history_view["__time"] = pd.to_datetime(
                        cost_history_view[cost_time], errors="coerce"
                    ).dt.floor("D")
                    cost_history_view = cost_history_view.dropna(subset=["__time"])
                    if not cost_history_view.empty:
                        cutoff = cost_history_view["__time"].max() - pd.Timedelta(days=56)
                        cost_view = cost_history_view[cost_history_view["__time"] >= cutoff].copy()
                    else:
                        cost_view = cost_history_view
                cost_by_group = {
                    str(key): float(value)
                    for key, value in cost_view.groupby(group_column, observed=True)["__cost"].median().items()
                    if np.isfinite(value) and value >= 0
                }
                if requested_grain == "category" and cost_time:
                    from scipy.stats import spearmanr

                    daily_cost = (
                        cost_history_view.dropna(
                            subset=["__time", "__cost", group_column]
                        )
                        .groupby(["__time", group_column], observed=True)["__cost"]
                        .median().rename("cost").reset_index()
                    )
                    aligned = daily.merge(
                        daily_cost, on=["__time", group_column],
                        how="inner", validate="one_to_one",
                    )
                    for group, group_data in aligned.groupby(
                        group_column, observed=True, sort=True
                    ):
                        group_data = group_data.sort_values("__time").copy()
                        group_data = group_data[
                            (group_data["price"] > 0) & (group_data["cost"] > 0)
                        ]
                        if len(group_data) < 30:
                            continue
                        group_data["markup_rate"] = (
                            group_data["price"] / group_data["cost"] - 1.0
                        )
                        elapsed = (
                            group_data["__time"] - group_data["__time"].iloc[0]
                        ).dt.days.to_numpy(dtype=float)
                        weekday = np.eye(7, dtype=float)[
                            group_data["__time"].dt.dayofweek.to_numpy()
                        ]
                        design_matrix = np.column_stack([elapsed, weekday])

                        def residual(values: np.ndarray) -> np.ndarray:
                            fitted = Ridge(alpha=10.0).fit(
                                design_matrix, values
                            ).predict(design_matrix)
                            return values - fitted

                        demand_residual = residual(
                            group_data["demand"].to_numpy(dtype=float)
                        )
                        markup_residual = residual(
                            group_data["markup_rate"].to_numpy(dtype=float)
                        )
                        coefficient, p_value = spearmanr(
                            demand_residual, markup_residual
                        )
                        if not (
                            np.isfinite(coefficient) and np.isfinite(p_value)
                        ):
                            continue
                        midpoint = len(group_data) // 2
                        first = spearmanr(
                            demand_residual[:midpoint], markup_residual[:midpoint]
                        ).statistic
                        second = spearmanr(
                            demand_residual[midpoint:], markup_residual[midpoint:]
                        ).statistic
                        split_same_sign = bool(
                            np.isfinite(first) and np.isfinite(second)
                            and (first == 0 or second == 0 or np.sign(first) == np.sign(second))
                        )
                        cost_plus_relationship.append({
                            "group": str(group),
                            "n_aligned_days": len(group_data),
                            "median_markup_rate": float(
                                group_data["markup_rate"].median()
                            ),
                            "residual_spearman": float(coefficient),
                            "p_value": float(p_value),
                            "q_value": None,
                            "significant": False,
                            "first_half_spearman": (
                                float(first) if np.isfinite(first) else None
                            ),
                            "second_half_spearman": (
                                float(second) if np.isfinite(second) else None
                            ),
                            "split_same_sign": split_same_sign,
                        })
                    ordered = sorted(
                        range(len(cost_plus_relationship)),
                        key=lambda index: cost_plus_relationship[index]["p_value"],
                    )
                    running = 1.0
                    for reverse_rank in range(len(ordered) - 1, -1, -1):
                        index = ordered[reverse_rank]
                        rank = reverse_rank + 1
                        adjusted = min(
                            running,
                            cost_plus_relationship[index]["p_value"]
                            * len(ordered) / rank,
                        )
                        running = adjusted
                        cost_plus_relationship[index]["q_value"] = float(adjusted)
                        cost_plus_relationship[index]["significant"] = bool(
                            adjusted <= 0.05
                            and cost_plus_relationship[index]["split_same_sign"]
                        )

        loss_by_group: Dict[str, float] = {}
        loss_audit = None
        for loss_dataset, frame in self._datasets.items():
            loss_column = numeric_measurement(
                frame, ("损耗率", "损失率", "损耗", "loss_rate", "waste")
            )
            if loss_column is None:
                continue
            loss_view, loss_audit = attach_group(loss_dataset, [loss_column])
            if loss_view is None:
                continue
            values = pd.to_numeric(loss_view[loss_column], errors="coerce")
            if float(values.dropna().median()) > 1.0:
                values = values / 100.0
            loss_view["__loss"] = values
            loss_view = loss_view[
                loss_view["__loss"].between(0, 0.95) & loss_view[group_column].notna()
            ]
            loss_by_group = {
                str(key): float(value)
                for key, value in loss_view.groupby(group_column, observed=True)["__loss"].median().items()
            }
            break

        parent_targets: Dict[Tuple[str, str], float] = {}
        item_to_parent: Dict[str, str] = {}
        parent_dimension: Optional[str] = None
        if supporting_forecasts:
            child_group_count = int(grouped_forecast.get("groups_forecast", 0))
            parent_candidates = sorted(
                (
                    item for item in supporting_forecasts
                    if item is not grouped_forecast
                    and str(item.get("group_column") or "") != group_column
                    and 0 < int(item.get("groups_forecast", 0)) < child_group_count
                ),
                key=lambda item: (
                    int(item.get("groups_forecast", 0)),
                    str(item.get("group_column", "")),
                ),
            )
            decision_dates = {
                str(item.get("date")) for item in grouped_forecast.get("forecasts", [])
            }
            for parent_forecast in parent_candidates:
                parent_column = str(parent_forecast.get("group_column") or "")
                mapping: Dict[str, str] = {}
                for frame in self._datasets.values():
                    if group_column not in frame.columns or parent_column not in frame.columns:
                        continue
                    lookup = frame[[group_column, parent_column]].dropna().drop_duplicates()
                    if lookup.empty or bool(
                        (lookup.groupby(group_column)[parent_column].nunique() > 1).any()
                    ):
                        continue
                    mapping = {
                        str(child): str(parent)
                        for child, parent in lookup.drop_duplicates(group_column).itertuples(
                            index=False, name=None
                        )
                    }
                    break
                targets = {
                    (str(item.get("group")), str(item.get("date"))): max(
                        0.0, float(item.get("forecast", 0.0))
                    )
                    for item in parent_forecast.get("forecasts", [])
                    if str(item.get("date")) in decision_dates
                }
                if mapping and targets:
                    item_to_parent = mapping
                    parent_targets = targets
                    parent_dimension = parent_column
                    break

        actions: List[Dict[str, Any]] = []
        slots: List[Tuple[str, str]] = []
        for forecast in grouped_forecast.get("forecasts", []):
            group = str(forecast.get("group"))
            date = str(forecast.get("date"))
            if group not in price_summary or (
                eligible_groups is not None and group not in eligible_groups
            ):
                continue
            slot = (group, date)
            slots.append(slot)
            baseline_demand = max(0.0, float(forecast.get("forecast", 0.0)))
            reference = price_summary[group]["reference"]
            lower_price = price_summary[group]["lower"]
            upper_price = price_summary[group]["upper"]
            elasticity_record = elasticity.get(group, {})
            price_eligible = bool(elasticity_record.get("eligible_for_price_decision"))
            prices = (
                np.linspace(lower_price, upper_price, 5).tolist()
                if price_eligible and upper_price > lower_price else [reference]
            )
            loss_rate = loss_by_group.get(group, 0.0)
            unit_cost = cost_by_group.get(group)
            for price in prices:
                coefficient = float(elasticity_record.get("coefficient") or 0.0)
                price_response = (
                    (float(price) / reference) ** coefficient if price_eligible else 1.0
                )
                demand = baseline_demand * price_response
                lower_demand = max(
                    0.0, float(forecast.get("lower_90", 0.0))
                ) * price_response
                upper_demand = max(
                    0.0, float(forecast.get("upper_90", 0.0))
                ) * price_response
                usable_fraction = max(1.0 - loss_rate, 0.05)
                lower_replenishment = max(
                    lower_demand / usable_fraction, minimum_display
                )
                point_replenishment = max(
                    demand / max(1.0 - loss_rate, 0.05), minimum_display
                )
                upper_replenishment = max(
                    upper_demand / usable_fraction, minimum_display
                )
                quantity_candidates = [("lower_interval", lower_replenishment)]
                if abs(point_replenishment - lower_replenishment) > 1e-9:
                    quantity_candidates.append(("point", point_replenishment))
                else:
                    quantity_candidates[0] = ("point", point_replenishment)
                for quantity_policy, replenishment in quantity_candidates:
                    available_for_sale = replenishment * usable_fraction
                    covered_point_demand = min(demand, available_for_sale)
                    payoff = (
                        float(price) * covered_point_demand
                        - float(unit_cost) * replenishment
                        if unit_cost is not None else 0.0
                    )
                    scenario_utilities = {
                        scenario: (
                            float(price) * min(scenario_demand, available_for_sale)
                            - float(unit_cost) * replenishment
                            if unit_cost is not None else 0.0
                        )
                        for scenario, scenario_demand in {
                            "lower_90": lower_demand,
                            "point": demand,
                            "upper_90": upper_demand,
                        }.items()
                    }
                    actions.append({
                        "slot": slot, "group": group, "date": date,
                        "price": float(price), "forecast_demand": float(demand),
                        "covered_point_demand": float(covered_point_demand),
                        "replenishment": float(replenishment), "loss_rate": loss_rate,
                        "unit_cost": unit_cost, "payoff": float(payoff),
                        "scenario_utilities": scenario_utilities,
                        "quantity_policy": quantity_policy,
                        "price_eligible": price_eligible,
                        "selected": True,
                        "lower_replenishment_90": float(lower_replenishment),
                        "upper_replenishment_90": float(upper_replenishment),
                    })
            if selection_bounds:
                actions.append({
                    "slot": slot, "group": group, "date": date,
                    "price": reference, "forecast_demand": baseline_demand,
                    "covered_point_demand": 0.0,
                    "replenishment": 0.0, "loss_rate": loss_rate,
                    "unit_cost": unit_cost, "payoff": 0.0,
                    "price_eligible": False, "selected": False,
                    "quantity_policy": "inactive",
                    "scenario_utilities": {
                        "lower_90": 0.0, "point": 0.0, "upper_90": 0.0,
                    },
                    "lower_replenishment_90": 0.0,
                    "upper_replenishment_90": 0.0,
                })
        slots = list(dict.fromkeys(slots))
        if not actions or not slots:
            return None
        costs_complete = all(action["unit_cost"] is not None for action in actions)
        strategy: List[Dict[str, Any]] = []
        solver_result = None
        compiled_contract = None
        hierarchical_stage_one = None
        hierarchical_shortage = None
        generic_decision_result = None
        risk_aware_decision_result = None
        risk_comparison = None
        risk_requested = any(
            token in str(problem).lower()
            for token in (
                "稳健", "风险厌恶", "下行风险", "不确定性优化", "cvar",
                "robust decision", "risk-averse", "risk averse",
            )
        )
        if costs_complete:
            from .hierarchical_decision_compiler import HierarchicalDecisionCompiler

            covered_parent_targets = [
                (key, target_value) for key, target_value in parent_targets.items()
                if any(
                    action.get("selected")
                    and action["date"] == key[1]
                    and item_to_parent.get(action["group"]) == key[0]
                    for action in actions
                )
            ]
            requirement_ids = {
                key: f"coverage_{index}"
                for index, (key, _) in enumerate(covered_parent_targets)
            }
            generic_actions = []
            for index, action in enumerate(actions):
                parent_key = (
                    item_to_parent.get(action["group"]), action["date"]
                )
                coverage = {}
                if action.get("selected") and parent_key in requirement_ids:
                    coverage[requirement_ids[parent_key]] = float(
                        action.get("covered_point_demand", action["forecast_demand"])
                    )
                generic_actions.append({
                    "id": f"candidate_{index}",
                    "decision_unit": f"{action['group']}::{action['date']}",
                    "utility": float(action["payoff"]),
                    "active": bool(action.get("selected", True)),
                    "coverage": coverage,
                    "scenario_utilities": dict(action.get("scenario_utilities", {})),
                    "metadata": {"source_action_index": index},
                })
            generic_requirements = [
                {
                    "id": requirement_ids[key],
                    "target": float(target_value),
                    "unit": "forecast_target_unit",
                    "metadata": {
                        "parent_group": key[0],
                        "date": key[1],
                        "parent_dimension": parent_dimension,
                    },
                }
                for key, target_value in covered_parent_targets
            ]
            nominal_decision_result = HierarchicalDecisionCompiler.solve(
                generic_actions,
                coverage_requirements=generic_requirements,
                active_count_bounds=selection_bounds,
            )
            stress_probabilities = {
                "lower_90": 0.25, "point": 0.50, "upper_90": 0.25,
            }
            risk_aware_decision_result = HierarchicalDecisionCompiler.solve(
                generic_actions,
                coverage_requirements=generic_requirements,
                active_count_bounds=selection_bounds,
                scenario_probabilities=stress_probabilities,
                risk_aversion=0.5,
                cvar_confidence=0.75,
            )
            generic_decision_result = (
                risk_aware_decision_result
                if risk_requested
                and risk_aware_decision_result.get("status") == "executed"
                else nominal_decision_result
            )

            def summarize_selected_scenarios(
                selected_indices: Sequence[int],
            ) -> Dict[str, Any]:
                outcomes = {
                    scenario: sum(
                        float(generic_actions[index]["scenario_utilities"][scenario])
                        for index in selected_indices
                    )
                    for scenario in stress_probabilities
                }
                expected = sum(
                    stress_probabilities[scenario] * outcomes[scenario]
                    for scenario in stress_probabilities
                )
                lower_tail = HierarchicalDecisionCompiler._weighted_lower_tail_cvar(
                    outcomes, stress_probabilities, 0.75,
                )
                return {
                    "scenario_outcomes": outcomes,
                    "stress_weighted_expected_utility": expected,
                    "worst_case_utility": min(outcomes.values()),
                    "lower_tail_cvar": lower_tail,
                    "risk_adjusted_utility": 0.5 * expected + 0.5 * lower_tail,
                }

            nominal_indices = nominal_decision_result.get(
                "selected_action_indices", []
            )
            risk_indices = risk_aware_decision_result.get(
                "selected_action_indices", []
            )
            nominal_by_unit = {
                generic_actions[index]["decision_unit"]: generic_actions[index]["id"]
                for index in nominal_indices
            }
            risk_by_unit = {
                generic_actions[index]["decision_unit"]: generic_actions[index]["id"]
                for index in risk_indices
            }
            risk_comparison = {
                "status": (
                    "pass"
                    if (risk_aware_decision_result.get("scenario_analysis") or {}).get(
                        "status"
                    ) == "pass" else "warning"
                ),
                "scenario_source": "forecast_point_and_90_percent_interval_stress_grid",
                "scenario_weights": stress_probabilities,
                "weights_are_calibrated_probabilities": False,
                "unsold_salvage_value": 0.0,
                "shortage_penalty_included": False,
                "risk_aversion": 0.5,
                "cvar_confidence": 0.75,
                "nominal_selection": summarize_selected_scenarios(nominal_indices),
                "risk_aware_selection": summarize_selected_scenarios(risk_indices),
                "changed_action_count": len(
                    set(nominal_decision_result.get("selected_action_ids", []))
                    ^ set(risk_aware_decision_result.get("selected_action_ids", []))
                ),
                "changed_decision_unit_count": sum(
                    nominal_by_unit.get(unit) != risk_by_unit.get(unit)
                    for unit in set(nominal_by_unit) | set(risk_by_unit)
                ),
                "risk_aware_candidate_rows": [
                    actions[index] for index in risk_indices
                    if actions[index].get("selected", True)
                ],
                "adopted": bool(
                    generic_decision_result is risk_aware_decision_result
                ),
                "decision": (
                    "题目显式要求稳健/风险厌恶，采用风险感知候选。"
                    if generic_decision_result is risk_aware_decision_result else
                    "题目未声明风险偏好；保留名义最优为主方案，风险感知结果仅作压力测试候选。"
                ),
            }
            compiled_contract = generic_decision_result.get("final_contract")
            solver_result = generic_decision_result.get("final_result")
            hierarchical_stage_one = generic_decision_result.get("stage_one_result")
            hierarchical_shortage = generic_decision_result.get(
                "minimum_weighted_shortage"
            )
            selected = [
                actions[index]
                for index in generic_decision_result.get("selected_action_indices", [])
            ]
            strategy = [item for item in selected if item.get("selected", True)]
        if not strategy:
            # Replenishment remains computable even if profit cannot be compiled.
            # Price is held fixed because missing cost makes a profit optimum undefined.
            fallback_candidates = []
            for slot in slots:
                slot_actions = [
                    action for action in actions
                    if action["slot"] == slot and action.get("selected", True)
                ]
                if slot_actions:
                    fallback_candidates.append(min(
                        slot_actions,
                        key=lambda item: (
                            item.get("quantity_policy") != "point",
                            abs(item["price"] - price_summary[item["group"]]["reference"]),
                        ),
                    ))
            if selection_bounds:
                # Without a profit-complete MILP, retain a feasible but clearly
                # non-optimal fallback ranked by forecast demand.
                fallback_candidates.sort(
                    key=lambda item: (-item["forecast_demand"], item["group"])
                )
                strategy = fallback_candidates[:selection_bounds[0]]
            else:
                strategy = fallback_candidates

        hierarchical_coverage: List[Dict[str, Any]] = []
        for (parent, date), target_value in parent_targets.items():
            covered = sum(
                float(item.get("covered_point_demand", item.get("forecast_demand", 0.0)))
                for item in strategy
                if item.get("date") == date
                and item_to_parent.get(str(item.get("group"))) == parent
            )
            shortage = max(0.0, float(target_value) - covered)
            hierarchical_coverage.append({
                "parent_group": parent, "date": date,
                "target_demand": float(target_value),
                "selected_item_demand": float(covered),
                "shortage": float(shortage),
                "coverage_ratio": (
                    min(1.0, covered / float(target_value)) if target_value > 0 else 1.0
                ),
            })
        aggregate_parent_coverage = (
            1.0 - sum(item["shortage"] for item in hierarchical_coverage)
            / max(sum(item["target_demand"] for item in hierarchical_coverage), 1e-12)
            if hierarchical_coverage else None
        )
        hierarchical_stage_one_success = bool(
            hierarchical_stage_one
            and hierarchical_stage_one.get("status") == "executed"
        )
        price_decisions = sum(bool(item["price_eligible"]) for item in strategy)
        cost_plus_significant_count = sum(
            bool(item.get("significant")) for item in cost_plus_relationship
        )
        forecast_group_set = {
            str(item.get("group")) for item in grouped_forecast.get("forecasts", [])
        }
        decision_group_set = (
            forecast_group_set.intersection(eligible_groups)
            if eligible_groups is not None else forecast_group_set
        )
        cost_coverage = (
            len(decision_group_set.intersection(cost_by_group))
            / max(1, len(decision_group_set))
        )
        loss_coverage = (
            len(decision_group_set.intersection(loss_by_group))
            / max(1, len(decision_group_set))
        )
        audit_status = "warning"
        return _plain({
            "status": "partially_executed",
            "method": "forecast_to_finite_action_milp",
            "mathematical_form": "multiple_choice_mixed_integer_linear_program",
            "requested_grain": requested_grain,
            "dataset": dataset_name,
            "request_text": str(problem),
            "source_task_ids": list(grouped_forecast.get("source_task_ids", [])),
            "group_grain": group_column,
            "decision_rows": strategy,
            "decision_count": len(strategy),
            "price_decision_count": price_decisions,
            "held_price_count": len(strategy) - price_decisions,
            "selection_bounds": list(selection_bounds) if selection_bounds else None,
            "minimum_display": minimum_display,
            "eligible_group_count": len(eligible_groups) if eligible_groups is not None else None,
            "decision_group_count": len(decision_group_set),
            "parent_dimension": parent_dimension,
            "hierarchical_demand_coverage": hierarchical_coverage,
            "aggregate_parent_demand_coverage": aggregate_parent_coverage,
            "hierarchical_lexicographic_stage_one": hierarchical_stage_one,
            "hierarchical_lexicographic_verified": hierarchical_stage_one_success,
            "minimum_total_parent_shortage": hierarchical_shortage,
            "generic_decision_compilation": ({
                "primitive": "hierarchical_finite_action_lexicographic_milp",
                "status": generic_decision_result.get("status"),
                "decision_unit_count": generic_decision_result.get("decision_unit_count"),
                "action_count": generic_decision_result.get("action_count"),
                "requirement_count": generic_decision_result.get("requirement_count"),
                "selected_active_count": generic_decision_result.get("selected_active_count"),
                "lexicographic_verified": generic_decision_result.get("lexicographic_verified"),
                "risk_aware_objective": bool(
                    generic_decision_result.get("scenario_analysis")
                ),
            } if generic_decision_result else None),
            "risk_aware_stress_test": risk_comparison,
            "cost_coverage": cost_coverage,
            "loss_coverage": loss_coverage,
            "cost_dataset": cost_dataset_used,
            "cost_column": cost_column_used,
            "elasticity_audit": elasticity,
            "cost_plus_pricing_relationship": cost_plus_relationship,
            "cost_plus_pricing_tested_groups": len(cost_plus_relationship),
            "cost_plus_pricing_significant_groups": cost_plus_significant_count,
            "price_bounds": price_summary,
            "joins": [item for item in (sales_join_audit, cost_audit, loss_audit) if item],
            "compiled_contract_summary": ({
                "kind": compiled_contract.get("kind"),
                "parse_status": compiled_contract.get("parse_status"),
                "variables": len(compiled_contract.get("variables", [])),
                "equality_constraints": len(compiled_contract.get("A_eq", [])),
            } if compiled_contract else None),
            "solver_result": solver_result,
            "credibility_audit": {
                "status": audit_status,
                "label": "条件性决策候选",
                "checks": [
                    {
                        "id": "generic_solver_dispatch",
                        "status": "pass" if solver_result else "warning",
                        "evidence": (
                            "有限价格动作已编译为通用多选MILP并由HiGHS求解。"
                            if solver_result else "成本未闭合，未声称利润最优；只输出需求覆盖补货量。"
                        ),
                    },
                    {
                        "id": "price_identifiability_gate", "status": "warning",
                        "evidence": (
                            f"{price_decisions}/{len(strategy)} 个决策通过负弹性分半同号门；"
                            "价格弹性仍是观察性关联，不是因果实验结论。"
                        ),
                    },
                    *([{
                        "id": "cost_plus_pricing_alignment",
                        "status": "pass" if cost_plus_relationship else "warning",
                        "evidence": (
                            f"按日期×{group_column}对齐售价、批发成本和销量，去除线性趋势与星期效应后，"
                            f"完成 {len(cost_plus_relationship)} 组检验；其中 "
                            f"{cost_plus_significant_count} 组同时通过BH-FDR与分半同号检验。"
                            "该结果是成本加成率与销量的观察性关联，不是因果效应。"
                            if cost_plus_relationship else
                            "没有至少30个对齐日的品类，成本加成率—销量关系不可检验；"
                            "系统不会用简单相关替代对齐后的稳健检验。"
                        ),
                    }] if requested_grain == "category" else []),
                    {
                        "id": "forecast_interval_propagation", "status": "pass",
                        "evidence": "每个补货量同时输出由90%预测区间传播得到的上下界。",
                    },
                    {
                        "id": "scenario_cvar_stress_test",
                        "status": (
                            "pass"
                            if risk_comparison
                            and risk_comparison.get("status") == "pass"
                            else "warning"
                        ),
                        "evidence": (
                            f"以预测点和90%区间端点构造3个压力情景，"
                            f"用期望收益与75%下尾CVaR各占50%复算候选；"
                            f"相对名义方案改变 {risk_comparison.get('changed_decision_unit_count', 0)} 个决策单元。"
                            f"{'题面声明风险偏好，已采用。' if risk_comparison.get('adopted') else '情景权重未经校准，仅保留为压力测试。'}"
                            if risk_comparison else
                            "成本不完整，无法对收益执行有限情景CVaR压力测试。"
                        ),
                    },
                    {
                        "id": "parent_demand_coverage",
                        "status": (
                            "pass"
                            if hierarchical_coverage and hierarchical_stage_one_success
                            else "warning"
                        ),
                        "evidence": (
                            f"先最小化品类需求总缺口，再在最小缺口内最大化收益；"
                            f"当前聚合覆盖率={aggregate_parent_coverage:.1%}。"
                            if hierarchical_coverage
                            and hierarchical_stage_one_success
                            and aggregate_parent_coverage is not None
                            else (
                                "已计算逐品类覆盖缺口，但第一阶段未得到经求解器验证的最小缺口；"
                                "不得把当前覆盖率解释为最优。"
                                if hierarchical_coverage
                                else "未获得可对齐的上层需求预测，未声称满足品类需求。"
                            )
                        ),
                    },
                ],
                "decision": (
                    "可把方案作为当前预测、成本、损耗和历史价格范围内的条件性候选；"
                    "不得宣称为现实全局最优，价格变化应先做小规模实验。"
                ),
            },
            "note": (
                "补货量按预测需求/(1-损耗率)计算；价格只在历史10%—90%分位范围内搜索，"
                "且必须通过负弹性分半稳定性门。成本或弹性不足时自动保持近期中位价格。"
            ),
        })

    def _run_integral_equation_discovery(
        self, target: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """Discover a sparse derivative-free candidate ODE on a time holdout.

        This is an integral/weak-form-inspired system identification stage: over
        each window, y(t1)-y(t0) is regressed on integrals of candidate library
        terms. It deliberately returns a falsifiable candidate equation, not a
        claim that the data-generating mechanism has been proven.
        """
        from sklearn.linear_model import Lasso
        from sklearn.metrics import mean_squared_error, r2_score

        explicit = self._numeric_subject(target)
        selected = None
        for dataset_name, profile in self._profiles.items():
            if not profile.datetime_columns:
                continue
            numeric = [
                column for column in profile.numeric_columns
                if column not in profile.id_candidates
                and self._datasets[dataset_name][column].nunique(dropna=True) > 2
            ]
            if explicit and explicit[0] == dataset_name and explicit[1] in numeric:
                selected = dataset_name, profile.datetime_columns[0], explicit[1], numeric
                break
            if numeric:
                selected = dataset_name, profile.datetime_columns[0], numeric[0], numeric
                break
        if selected is None:
            return None
        dataset_name, time_column, target_column, numeric_columns = selected
        source = self._datasets[dataset_name]
        working_columns = [time_column] + numeric_columns[:min(8, len(numeric_columns))]
        frame = source[working_columns].copy()
        frame[time_column] = pd.to_datetime(frame[time_column], errors="coerce")
        for column in working_columns[1:]:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = (
            frame.dropna(subset=[time_column, target_column])
            .groupby(time_column, as_index=False)[working_columns[1:]].mean()
            .sort_values(time_column)
        )
        if len(frame) > 5_000:
            positions = np.linspace(0, len(frame) - 1, 5_000, dtype=int)
            frame = frame.iloc[np.unique(positions)].reset_index(drop=True)
        if len(frame) < 50:
            return None
        elapsed = (
            frame[time_column] - frame[time_column].iloc[0]
        ).dt.total_seconds().to_numpy(dtype=float) / 86_400.0
        positive_intervals = np.diff(elapsed)
        if not len(positive_intervals) or np.any(positive_intervals <= 0):
            return None
        available_states = [
            column for column in working_columns[1:]
            if frame[column].notna().sum() >= max(30, int(0.8 * len(frame)))
            and frame[column].nunique(dropna=True) > 2
        ]
        if target_column not in available_states:
            return None
        correlations = frame[available_states].corr(method="spearman")[target_column].abs()
        state_columns = [target_column] + [
            column for column in correlations.sort_values(ascending=False).index
            if column != target_column
        ][:3]
        state_frame = frame[state_columns].interpolate(limit_direction="both")
        matrix = state_frame.to_numpy(dtype=float)
        state_center = np.median(matrix, axis=0)
        state_mad = np.median(np.abs(matrix - state_center), axis=0)
        state_scale = np.where(
            1.4826 * state_mad > 1e-12, 1.4826 * state_mad,
            np.where(np.std(matrix, axis=0) > 1e-12, np.std(matrix, axis=0), 1.0),
        )
        states = (matrix - state_center) / state_scale
        library_columns = [np.ones(len(states))]
        term_names = ["1"]
        for index, column in enumerate(state_columns):
            library_columns.append(states[:, index])
            term_names.append(f"z({column})")
        for index, column in enumerate(state_columns):
            library_columns.append(states[:, index] ** 2)
            term_names.append(f"z({column})²")
        for left, right in combinations(range(len(state_columns)), 2):
            library_columns.append(states[:, left] * states[:, right])
            term_names.append(f"z({state_columns[left]})·z({state_columns[right]})")
        library = np.column_stack(library_columns)
        window = max(3, min(12, len(frame) // 30))
        integrated_rows: List[np.ndarray] = []
        deltas: List[float] = []
        starts: List[int] = []
        target_index = state_columns.index(target_column)
        for start in range(0, len(frame) - window):
            stop = start + window
            local_dt = np.diff(elapsed[start:stop + 1])
            integral = np.sum(
                0.5 * (library[start:stop] + library[start + 1:stop + 1])
                * local_dt[:, None],
                axis=0,
            )
            integrated_rows.append(integral)
            deltas.append(float(states[stop, target_index] - states[start, target_index]))
            starts.append(start)
        design = np.asarray(integrated_rows, dtype=float)
        response = np.asarray(deltas, dtype=float)
        starts_array = np.asarray(starts)
        split_point = int(len(frame) * 0.70)
        train_mask = starts_array + window < split_point
        validation_mask = starts_array >= split_point
        if int(train_mask.sum()) < 30 or int(validation_mask.sum()) < 12:
            return None
        feature_scale = np.std(design[train_mask], axis=0)
        feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
        scaled_design = design / feature_scale
        response_scale = max(float(np.std(response[train_mask])), 1e-12)
        alpha_grid = response_scale * np.logspace(-4, -0.7, 16)
        candidates: List[Dict[str, Any]] = []
        best = None
        for alpha in alpha_grid:
            model = Lasso(
                alpha=float(alpha), fit_intercept=False,
                max_iter=20_000, random_state=self.random_state,
            )
            model.fit(scaled_design[train_mask], response[train_mask])
            validation_prediction = model.predict(scaled_design[validation_mask])
            rmse = float(np.sqrt(mean_squared_error(
                response[validation_mask], validation_prediction
            )))
            nonzero = int(np.sum(np.abs(model.coef_) > 1e-8))
            objective = rmse / response_scale + 0.005 * nonzero
            candidate = {
                "alpha": float(alpha), "validation_rmse": rmse,
                "nonzero_terms": nonzero, "selection_objective": objective,
            }
            candidates.append(candidate)
            if best is None or objective < best[0]:
                best = objective, model, validation_prediction, candidate
        if best is None:
            return None
        _, selected_model, validation_prediction, selected_candidate = best
        validation_actual = response[validation_mask]
        # Persistence is the correct derivative-free baseline: over a window it
        # predicts no state change. A training-period mean delta can be badly
        # biased when the process moves into a new regime.
        baseline_prediction = np.zeros_like(validation_actual)
        validation_rmse = float(np.sqrt(mean_squared_error(
            validation_actual, validation_prediction
        )))
        baseline_rmse = float(np.sqrt(mean_squared_error(
            validation_actual, baseline_prediction
        )))
        validation_r2 = float(r2_score(validation_actual, validation_prediction))
        coefficients = selected_model.coef_ / feature_scale
        active = [
            {
                "term": term, "coefficient": float(coefficient),
                "absolute_coefficient": float(abs(coefficient)),
            }
            for term, coefficient in zip(term_names, coefficients)
            if abs(coefficient) > 1e-8
        ]
        active.sort(key=lambda item: item["absolute_coefficient"], reverse=True)

        midpoint = int(train_mask.sum()) // 2
        train_indices = np.flatnonzero(train_mask)
        supports: List[set] = []
        for subset in (train_indices[:midpoint], train_indices[midpoint:]):
            if len(subset) < 15:
                continue
            model = Lasso(
                alpha=float(selected_candidate["alpha"]), fit_intercept=False,
                max_iter=20_000, random_state=self.random_state,
            ).fit(scaled_design[subset], response[subset])
            supports.append({
                term_names[index] for index, coefficient in enumerate(model.coef_)
                if abs(coefficient) > 1e-8
            })
        if len(supports) == 2:
            support_union = supports[0] | supports[1]
            support_jaccard = (
                len(supports[0] & supports[1]) / len(support_union)
                if support_union else 1.0
            )
        else:
            support_jaccard = None
        residual = validation_actual - validation_prediction
        residual_autocorrelation = (
            float(pd.Series(residual).autocorr(lag=1)) if len(residual) > 5 else None
        )
        if residual_autocorrelation is not None and not np.isfinite(residual_autocorrelation):
            residual_autocorrelation = None
        checks = [
            self._credibility_check(
                "dynamics_holdout", "时间外推验证",
                "pass" if validation_rmse < baseline_rmse * 0.90 and validation_r2 >= 0.25 else (
                    "warning" if validation_rmse < baseline_rmse and validation_r2 > 0 else "fail"
                ),
                f"末段时间留出 RMSE={validation_rmse:.4g}，零变化基线={baseline_rmse:.4g}，R²={validation_r2:.3f}。",
                "候选方程未稳定优于简单变化基线，不能作为机理解释。"
                if validation_rmse >= baseline_rmse * 0.90 or validation_r2 < 0.25 else "",
            ),
            self._credibility_check(
                "equation_support_stability", "方程项稳定性",
                "not_assessed" if support_jaccard is None else (
                    "pass" if support_jaccard >= 0.70 else (
                        "warning" if support_jaccard >= 0.40 else "fail"
                    )
                ),
                "样本不足，未执行分段项集复核。" if support_jaccard is None else
                f"前后两段训练窗口的非零项 Jaccard={support_jaccard:.1%}。",
                "方程项随时间段改变，可能存在状态切换或伪相关。"
                if support_jaccard is not None and support_jaccard < 0.70 else "",
            ),
            self._credibility_check(
                "dynamics_residual_memory", "动力残差记忆",
                "not_assessed" if residual_autocorrelation is None else (
                    "pass" if abs(residual_autocorrelation) < 0.30 else (
                        "warning" if abs(residual_autocorrelation) < 0.60 else "fail"
                    )
                ),
                "残差长度不足。" if residual_autocorrelation is None else
                f"末段残差一阶自相关={residual_autocorrelation:.3f}。",
                "残差仍有明显时间结构，候选库遗漏了状态、滞后或外生驱动。"
                if residual_autocorrelation is not None and abs(residual_autocorrelation) >= 0.30 else "",
            ),
        ]
        failed = any(check["status"] == "fail" for check in checks)
        warned = any(check["status"] in {"warning", "not_assessed"} for check in checks)
        audit_status = "fail" if failed else ("warning" if warned else "pass")
        audit_label = {"pass": "可信候选", "warning": "谨慎候选", "fail": "未通过"}[audit_status]
        equation_terms = " + ".join(
            f"{item['coefficient']:.5g}·{item['term']}" for item in active
        ) or "0"
        return _plain({
            "dataset": dataset_name,
            "time_column": time_column,
            "target": target_column,
            "state_columns": state_columns,
            "method": "derivative_free_integral_sparse_dynamics",
            "equation": f"d z({target_column}) / d day = {equation_terms}",
            "standardization": {
                column: {"center": float(center), "scale": float(scale)}
                for column, center, scale in zip(state_columns, state_center, state_scale)
            },
            "window_points": window,
            "n_time_points": len(frame),
            "training_windows": int(train_mask.sum()),
            "validation_windows": int(validation_mask.sum()),
            "selected_alpha": selected_candidate["alpha"],
            "active_terms": active,
            "candidate_search": candidates,
            "validation_actual": validation_actual,
            "validation_prediction": validation_prediction,
            "metrics": {
                "validation_rmse": validation_rmse,
                "baseline_rmse": baseline_rmse,
                "validation_r2": validation_r2,
                "support_jaccard": support_jaccard,
                "residual_autocorrelation": residual_autocorrelation,
            },
            "credibility_audit": {
                "status": audit_status,
                "label": audit_label,
                "checks": checks,
                "decision": (
                    "候选动力方程通过当前外推和稳定性检查"
                    if audit_status == "pass" else
                    "该方程只能作为待验证假设，不能宣称为真实控制方程"
                ),
            },
            "literature_basis": {
                "idea": "weak/integral sparse identification avoids pointwise derivative estimation",
                "doi": "10.1137/20M1343166",
            },
            "note": "方程建立在标准化状态和观测时间尺度上；统计可辨识不等于机理正确。",
        })

    @staticmethod
    def _relationship_backbone(
        relationships: Sequence[DatasetRelation],
        max_edges: int,
    ) -> List[DatasetRelation]:
        """Build a confidence-weighted sparse backbone instead of plotting a clique."""
        best_per_pair: Dict[Tuple[str, str], DatasetRelation] = {}
        for relation in relationships:
            pair = tuple(sorted((relation.left_dataset, relation.right_dataset)))
            if pair not in best_per_pair or relation.confidence > best_per_pair[pair].confidence:
                best_per_pair[pair] = relation
        candidates = sorted(
            best_per_pair.values(),
            key=lambda relation: (relation.confidence, relation.value_overlap),
            reverse=True,
        )
        parent: Dict[str, str] = {}

        def find(node: str) -> str:
            parent.setdefault(node, node)
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        selected: List[DatasetRelation] = []
        selected_ids: set = set()
        for relation in candidates:
            left_root, right_root = find(relation.left_dataset), find(relation.right_dataset)
            if left_root != right_root:
                parent[left_root] = right_root
                selected.append(relation)
                selected_ids.add(id(relation))
        for relation in candidates:
            if len(selected) >= max_edges:
                break
            if id(relation) not in selected_ids:
                selected.append(relation)
        return selected[:max_edges]

    def _generate_charts(
        self,
        relationships: Sequence[DatasetRelation],
        interactions: Sequence[InteractionFinding],
        model_results: Sequence[Dict[str, Any]],
        ranking_result: Optional[Dict[str, Any]],
        specialized_results: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
        self._artifact_path("charts", "_layout_probe").parent.mkdir(parents=True, exist_ok=True)
        charts: List[Dict[str, Any]] = []

        mechanistic_candidate = specialized_results.get("mechanistic_model", {})
        mechanistic = (
            mechanistic_candidate
            if mechanistic_candidate.get("presentation_scope", "primary") == "primary"
            else {}
        )
        operator_graph = mechanistic.get("operator_graph", [])
        if operator_graph:
            labels = [str(node.get("key", node.get("id", "operator"))) for node in operator_graph]
            completeness = [
                1.0 - len(node.get("missing_bindings", [])) /
                max(1, len(node.get("required_bindings", [])))
                for node in operator_graph
            ]
            colors = [
                "#2E86AB" if node.get("status") == "ready_to_compile" else "#F18F01"
                for node in operator_graph
            ]
            height = max(4.5, min(10.0, 0.48 * len(labels) + 2.0))
            fig, axes = plt.subplots(1, 2, figsize=(14, height))
            positions = np.arange(len(labels))
            axes[0].barh(positions, completeness, color=colors, alpha=0.9)
            axes[0].set_yticks(positions, labels, fontsize=8)
            axes[0].invert_yaxis()
            axes[0].set_xlim(0, 1.02)
            axes[0].set_xlabel("binding completeness")
            axes[0].set_title("通用数学算子编译完整度")
            categories: Dict[str, int] = {}
            for node in operator_graph:
                category = str(node.get("category", "other"))
                categories[category] = categories.get(category, 0) + 1
            category_names = list(categories)
            axes[1].bar(
                category_names, [categories[name] for name in category_names],
                color="#64748B", alpha=0.9,
            )
            axes[1].set_ylabel("operator count")
            axes[1].set_title("算子类别覆盖")
            axes[1].tick_params(axis="x", rotation=35)
            fig.suptitle("纯题面数学 IR：算子图与求解准备度")
            fig.tight_layout()
            path = self._artifact_path("charts", "10_mechanistic_operator_graph.png")
            fig.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(fig)
            charts.append({
                "title": "通用算子图与求解准备度",
                "type": "mechanistic_operator_graph",
                "path": str(path), "datasets": [],
            })

        for result_index, numerical in enumerate(mechanistic.get("numerical_results", [])[:6], 1):
            plot_data = numerical.get("plot_data", {})
            if numerical.get("kind") in {
                "kinematic_visibility_event",
                "kinematic_visibility_optimization_solution",
            }:
                time_values = np.asarray(plot_data.get("time", []), dtype=float)
                distances = np.asarray(plot_data.get("distance", []), dtype=float)
                threshold = float(plot_data.get("threshold", 0.0))
                if len(time_values) and len(time_values) == len(distances) and threshold > 0:
                    fig, ax = plt.subplots(figsize=(10, 5.4))
                    ax.plot(time_values, distances, color="#2E86AB", linewidth=1.7,
                            label="影响区中心到源—目标线段的距离")
                    ax.axhline(threshold, color="#C73E1D", linestyle="--", linewidth=1.3,
                               label=f"有效半径 {threshold:g}")
                    for left, right in plot_data.get("intervals", []):
                        ax.axvspan(float(left), float(right), color="#28A745", alpha=0.18)
                    ax.set_xlabel("任务开始后的时间 / s")
                    ax.set_ylabel("距离 / m")
                    optimized = (
                        numerical.get("kind")
                        == "kinematic_visibility_optimization_solution"
                    )
                    ax.set_title(
                        "优化候选的连续遮蔽事件与有效区间"
                        if optimized else "连续遮蔽事件：距离阈值、精化根与有效区间"
                    )
                    ax.legend(loc="best")
                    ax.grid(alpha=0.2)
                    fig.tight_layout()
                    path = self._artifact_path(
                        "charts",
                        f"11_mechanistic_visibility_{'optimization' if optimized else 'event'}_"
                        f"{result_index:02d}.png",
                    )
                    fig.savefig(path, dpi=170, bbox_inches="tight")
                    plt.close(fig)
                    charts.append({
                        "title": (
                            f"非凸优化候选的遮蔽区间 {result_index}"
                            if optimized else f"连续遮蔽事件与有效时长 {result_index}"
                        ),
                        "type": (
                            "mechanistic_visibility_optimization"
                            if optimized else "mechanistic_visibility_event"
                        ),
                        "path": str(path), "datasets": [],
                    })
                continue
            if numerical.get("kind") == "optimization_solution":
                names = [str(name) for name in plot_data.get("names", [])]
                values = np.asarray(plot_data.get("values", []), dtype=float)
                if names and len(names) == len(values):
                    fig, ax = plt.subplots(figsize=(max(7.5, min(13.0, len(names) * 0.65)), 5.2))
                    ax.bar(names, values, color="#2E86AB", alpha=0.9)
                    ax.axhline(0.0, color="#64748B", linewidth=0.8)
                    ax.set_ylabel("decision value")
                    ax.set_title("结构化约束优化候选（多起点局部求解）")
                    ax.tick_params(axis="x", rotation=35)
                    fig.tight_layout()
                    path = self._artifact_path(
                        "charts", f"12_mechanistic_optimization_{result_index:02d}.png"
                    )
                    fig.savefig(path, dpi=170, bbox_inches="tight")
                    plt.close(fig)
                    charts.append({
                        "title": f"结构化约束优化候选 {result_index}",
                        "type": "mechanistic_optimization_solution",
                        "path": str(path), "datasets": [],
                    })
                continue
            time_values = np.asarray(plot_data.get("time", []), dtype=float)
            series = plot_data.get("series", {})
            if numerical.get("kind") != "ode_trajectory" or not len(time_values) or not series:
                continue
            fig, ax = plt.subplots(figsize=(10, 5.4))
            for name, values in list(series.items())[:12]:
                array = np.asarray(values, dtype=float)
                if len(array) == len(time_values):
                    ax.plot(time_values, array, linewidth=1.5, label=str(name))
            ax.set_xlabel(str(numerical.get("time_variable", "time")))
            ax.set_ylabel("state value")
            ax.set_title("已验证结构化 ODE 的数值轨迹")
            ax.legend(loc="best", ncol=2)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            path = self._artifact_path(
                "charts", f"11_mechanistic_ode_trajectory_{result_index:02d}.png"
            )
            fig.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(fig)
            charts.append({
                "title": f"结构化 ODE 数值轨迹 {result_index}",
                "type": "mechanistic_ode_trajectory",
                "path": str(path), "datasets": [],
            })

        for index, (name, df) in enumerate(list(self._datasets.items())[:6], start=1):
            profile = self._profiles[name]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
            missing = df.isna().mean().nlargest(15)
            axes[0].barh(missing.index.astype(str)[::-1], missing.values[::-1], color="#F18F01")
            axes[0].set_title(f"{name} · 缺失率")
            axes[0].set_xlim(0, max(1.0, float(missing.max()) * 1.05 if len(missing) else 1.0))
            axes[0].set_xlabel("missing ratio")
            numeric = self._top_numeric(df)
            if len(numeric) >= 2:
                corr = _sample_frame(df[numeric], min(10_000, len(df)), self.random_state).corr(method="spearman")
                image = axes[1].imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
                axes[1].set_xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right", fontsize=8)
                axes[1].set_yticks(range(len(corr.index)), corr.index, fontsize=8)
                axes[1].set_title("数值变量 Spearman 相关")
                fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
            else:
                axes[1].axis("off")
                axes[1].text(0.5, 0.5, "数值列不足，跳过相关矩阵", ha="center", va="center")
            fig.suptitle(f"数据集概览：{name}（{profile.source_rows:,} 行 × {profile.n_columns} 列）")
            fig.tight_layout()
            path = self._artifact_path(
                "charts", f"{index:02d}_{_safe_filename(name)}_overview.png"
            )
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            charts.append({"title": f"{name} 数据概览", "type": "dataset_overview", "path": str(path), "datasets": [name]})

        if relationships:
            fig, ax = plt.subplots(figsize=(max(7, len(self._datasets) * 1.8), 5.5))
            names = list(self._datasets)
            display_relationships = self._relationship_backbone(
                relationships,
                max_edges=max(len(names) - 1, 2 * len(names) - 2),
            )
            angles = np.linspace(0, 2 * np.pi, len(names), endpoint=False)
            positions = {name: (math.cos(angle), math.sin(angle)) for name, angle in zip(names, angles)}
            for name, (x, y) in positions.items():
                role = self._profiles[name].role
                color = {"fact": "#2E86AB", "dimension": "#28A745", "bridge": "#A23B72"}.get(role, "#6C757D")
                ax.scatter([x], [y], s=1800, color=color, alpha=0.9, edgecolors="white", linewidths=2, zorder=3)
                ax.text(x, y, f"{name}\n[{role}]", ha="center", va="center", color="white", fontsize=9, zorder=4)
            for relation in display_relationships:
                x1, y1 = positions[relation.left_dataset]
                x2, y2 = positions[relation.right_dataset]
                ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops={"arrowstyle": "-", "lw": 1 + relation.confidence / 50, "color": "#64748b", "alpha": 0.65})
                ax.text((x1 + x2) / 2, (y1 + y2) / 2, f"{relation.left_key} - {relation.right_key}\n{relation.relationship}", fontsize=7, ha="center", bbox={"boxstyle": "round,pad=.2", "fc": "white", "ec": "#ddd", "alpha": 0.9})
            ax.set_title("多数据集关系骨架（边宽表示关联置信度）")
            ax.set_xlim(-1.35, 1.35)
            ax.set_ylim(-1.25, 1.25)
            ax.axis("off")
            path = self._artifact_path("charts", "20_dataset_relationship_graph.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({
                "title": "多数据集关系骨架图",
                "type": "relationship_graph",
                "path": str(path),
                "datasets": names,
                "displayed_relationships": len(display_relationships),
                "total_relationships": len(relationships),
            })

        data_compilation = specialized_results.get("mathematical_data_compilation", {})
        stressed_relationships = [
            item for item in (
                (data_compilation.get("conclusion_stress") or {}).get("relationships") or []
            )
            if item.get("status") in {"contradicted", "restricted"}
        ][:6]
        if stressed_relationships:
            labels: List[str] = []
            values: List[float] = []
            colors: List[str] = []
            for relationship in stressed_relationships:
                predictor = str(relationship.get("predictor", "predictor"))
                for context in relationship.get("contexts", []):
                    rho = context.get("rho")
                    if rho is None or not np.isfinite(float(rho)):
                        continue
                    labels.append(f"{predictor} | {context.get('view')}")
                    values.append(float(rho))
                    colors.append("#C73E1D" if float(rho) < 0 else "#2E86AB")
            if values:
                height = max(5.0, min(12.0, len(values) * 0.32 + 2.0))
                fig, ax = plt.subplots(figsize=(12, height))
                positions = np.arange(len(values))
                ax.barh(positions, values, color=colors, alpha=0.88)
                ax.set_yticks(positions, labels, fontsize=8)
                ax.invert_yaxis()
                ax.axvline(0, color="#111827", linewidth=1.0)
                ax.set_xlim(-1.0, 1.0)
                ax.set_xlabel("Spearman ρ")
                ax.set_title("同一关系在合理数据视图下的方向与强度")
                fig.tight_layout()
                path = self._artifact_path("charts", "29_math_data_view_stability.png")
                fig.savefig(path, dpi=170, bbox_inches="tight")
                plt.close(fig)
                charts.append({
                    "title": "数学数据多视图结论稳定性",
                    "type": "mathematical_data_view_stability",
                    "path": str(path),
                    "datasets": [str(data_compilation.get("dataset", ""))],
                })

        if interactions:
            top = list(interactions[:15])[::-1]
            labels = [
                f"{item.left_dataset}.{item.left_variable} - "
                f"{item.right_dataset}.{item.right_variable}"
                for item in top
            ]
            values = [item.strength for item in top]
            fig, ax = plt.subplots(figsize=(10, max(4.5, len(top) * 0.38)))
            colors = ["#2E86AB" if value >= 0 else "#C73E1D" for value in values]
            ax.barh(labels, values, color=colors)
            ax.axvline(0, color="#333", linewidth=0.8)
            ax.set_xlim(-1, 1)
            ax.set_xlabel("effect strength（ρ 为有符号值，η²/NMI/Cramér's V 为非负值）")
            ax.set_title("跨数据集变量交互强度")
            fig.tight_layout()
            path = self._artifact_path("charts", "30_cross_dataset_interactions.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({"title": "跨数据集变量交互", "type": "interaction_bar", "path": str(path), "datasets": list(self._datasets)})

        hierarchical = specialized_results.get("hierarchical_distribution")
        if hierarchical:
            category_column = str(hierarchical.get("parent_dimension") or "parent")
            child_column = str(hierarchical.get("child_dimension") or "child")
            category_rows = pd.DataFrame(hierarchical.get("parent_summary", []))
            weekday_rows = pd.DataFrame(hierarchical.get("weekday_profile", []))
            if not category_rows.empty and category_column in category_rows:
                category_rows = category_rows.sort_values("total", ascending=True)
                fig, axes = plt.subplots(
                    1, 2, figsize=(14, max(5.2, len(category_rows) * 0.55))
                )
                axes[0].barh(
                    category_rows[category_column].astype(str), category_rows["total"],
                    color="#2E86AB", alpha=0.9,
                )
                axes[0].set_xlabel("观测期可加总量")
                axes[0].set_title(f"{category_column}规模")
                if not weekday_rows.empty:
                    for category, values in weekday_rows.groupby(category_column, observed=True):
                        values = values.sort_values("weekday")
                        axes[1].plot(
                            values["weekday"], values["mean_sales"], marker="o",
                            linewidth=1.5, label=str(category),
                        )
                    axes[1].set_xticks(
                        range(7), ["一", "二", "三", "四", "五", "六", "日"]
                    )
                    axes[1].legend(loc="best", ncol=2, fontsize=8)
                axes[1].set_xlabel("星期")
                axes[1].set_ylabel("平均销量")
                axes[1].set_title("品类星期效应")
                fig.suptitle(f"{category_column}—{child_column}层级分布")
                fig.tight_layout()
                path = self._artifact_path(
                    "charts", "34_hierarchical_sales_distribution.png"
                )
                fig.savefig(path, dpi=170, bbox_inches="tight")
                plt.close(fig)
                charts.append({
                    "title": f"{category_column}—{child_column}分布与星期效应",
                    "type": "hierarchical_sales",
                    "path": str(path),
                    "datasets": [str(hierarchical.get("dataset", ""))],
                })

        grouped_forecasts = list(specialized_results.get("grouped_forecasts") or [])
        if not grouped_forecasts and specialized_results.get("grouped_forecast"):
            grouped_forecasts = [specialized_results["grouped_forecast"]]
        for forecast_index, grouped in enumerate(grouped_forecasts[:4], start=1):
            rows = pd.DataFrame(grouped.get("forecasts", []))
            if rows.empty:
                continue
            rows["date"] = pd.to_datetime(rows["date"], errors="coerce")
            for column in ("forecast", "lower_90", "upper_90"):
                rows[column] = pd.to_numeric(rows[column], errors="coerce")
            rows = rows.dropna(subset=["date", "forecast", "group"])
            if rows.empty:
                continue
            grain = str(grouped.get("requested_grain") or "group")
            group_order = (
                rows.groupby("group", observed=True)["forecast"].mean()
                .sort_values(ascending=False).head(12).index
            )
            selected = rows[rows["group"].isin(group_order)].copy()
            fig, ax = plt.subplots(figsize=(11, 5.8))
            if selected["date"].nunique() > 1:
                for group, values in selected.groupby("group", observed=True, sort=False):
                    values = values.sort_values("date")
                    ax.plot(values["date"], values["forecast"], marker="o", linewidth=1.4, label=str(group))
                    if len(group_order) <= 8:
                        ax.fill_between(
                            values["date"], values["lower_90"], values["upper_90"], alpha=0.10
                        )
                ax.set_xlabel("预测日期")
                ax.legend(loc="best", ncol=2, fontsize=8)
            else:
                selected = selected.sort_values("forecast", ascending=True)
                error = np.vstack([
                    np.maximum(0.0, selected["forecast"] - selected["lower_90"]),
                    np.maximum(0.0, selected["upper_90"] - selected["forecast"]),
                ])
                ax.barh(selected["group"].astype(str), selected["forecast"], xerr=error,
                        color="#2E86AB", alpha=0.9, capsize=2)
                ax.set_xlabel("预测需求及 90% 区间")
            ax.set_ylabel(str(grouped.get("group_column") or "分组"))
            ax.set_title(f"{grain} 粒度需求预测（按题目粒度先聚合）")
            ax.grid(alpha=0.18, axis="y" if selected["date"].nunique() > 1 else "x")
            fig.tight_layout()
            path = self._artifact_path(
                "charts", f"35_grouped_forecast_{forecast_index:02d}_{_safe_filename(grain)}.png"
            )
            fig.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(fig)
            charts.append({
                "title": f"{grain} 粒度分组需求预测",
                "type": "grouped_forecast",
                "path": str(path),
                "datasets": [str(grouped.get("dataset", ""))],
            })

        prescriptive_decisions = list(specialized_results.get("prescriptive_decisions") or [])
        if not prescriptive_decisions and specialized_results.get("prescriptive_decision"):
            prescriptive_decisions = [specialized_results["prescriptive_decision"]]
        for decision_index, decision in enumerate(prescriptive_decisions[:4], start=1):
            rows = pd.DataFrame(decision.get("decision_rows", []))
            if rows.empty:
                continue
            for column in ("replenishment", "price", "forecast_demand"):
                rows[column] = pd.to_numeric(rows[column], errors="coerce")
            rows = rows.dropna(subset=["group", "replenishment", "price"])
            if rows.empty:
                continue
            rows = rows.sort_values("replenishment", ascending=False).head(30).iloc[::-1]
            fig, axes = plt.subplots(1, 2, figsize=(14, max(5.2, len(rows) * 0.25)))
            labels = rows["group"].astype(str) + " · " + rows["date"].astype(str)
            axes[0].barh(labels, rows["replenishment"], color="#28A745", alpha=0.9)
            axes[0].set_xlabel("建议补货量")
            axes[0].set_title("需求区间传播后的补货候选")
            axes[1].barh(labels, rows["price"], color="#F18F01", alpha=0.9)
            axes[1].set_xlabel("建议价格")
            axes[1].set_title("经可识别性门约束的价格候选")
            fig.suptitle(f"{decision.get('requested_grain', 'group')} 粒度组合决策")
            fig.tight_layout()
            path = self._artifact_path(
                "charts", f"36_prescriptive_decision_{decision_index:02d}.png"
            )
            fig.savefig(path, dpi=170, bbox_inches="tight")
            plt.close(fig)
            charts.append({
                "title": f"{decision.get('requested_grain', 'group')} 粒度补货与定价决策",
                "type": "prescriptive_decision",
                "path": str(path),
                "datasets": [],
            })

        for structure_index, structure in enumerate(
            specialized_results.get("data_structure", [])[:3], start=1
        ):
            projection = structure.get("projection", [])
            if not projection:
                continue
            pc1 = np.asarray([point["pc1"] for point in projection], dtype=float)
            pc2 = np.asarray([point["pc2"] for point in projection], dtype=float)
            flagged = np.asarray([point["flagged"] for point in projection], dtype=bool)
            variance = np.asarray(structure.get("explained_variance_ratio", []), dtype=float)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))
            axes[0].scatter(
                pc1[~flagged], pc2[~flagged], s=10, alpha=0.35,
                color="#2E86AB", label="常规样本",
            )
            if np.any(flagged):
                axes[0].scatter(
                    pc1[flagged], pc2[flagged], s=28, alpha=0.9,
                    color="#C73E1D", marker="x", label="结构异常",
                )
            axes[0].set_xlabel("PC1")
            axes[0].set_ylabel("PC2")
            axes[0].set_title("主成分空间与结构异常")
            axes[0].legend(loc="best")
            component_names = [f"PC{index}" for index in range(1, len(variance) + 1)]
            axes[1].bar(component_names, variance, color="#A23B72", alpha=0.85)
            axes[1].plot(
                component_names, np.cumsum(variance), color="#F18F01",
                marker="o", label="累计解释率",
            )
            axes[1].axhline(0.90, color="#64748b", linestyle="--", linewidth=1)
            axes[1].set_ylim(0, 1.05)
            axes[1].set_title(
                f"{structure['dimensions_90']} 维解释 "
                f"{structure['cumulative_explained_variance']:.1%}"
            )
            axes[1].tick_params(axis="x", rotation=45)
            axes[1].legend(loc="best")
            audit_label = structure.get("credibility_audit", {}).get("label", "未审计")
            fig.suptitle(
                f"{structure['dataset']} · 潜在结构与异常（可信度：{audit_label}）"
            )
            fig.tight_layout()
            path = self._artifact_path(
                "charts",
                f"35_{structure_index:02d}_{_safe_filename(structure['dataset'])}_structure.png",
            )
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({
                "title": f"{structure['dataset']} 潜在结构与异常",
                "type": "data_structure",
                "path": str(path),
                "datasets": [structure["dataset"]],
                "credibility": audit_label,
            })

        clustering_result = next(
            (model for model in model_results if model.get("task_type") == "clustering" and model.get("embedding") is not None),
            None,
        )
        if clustering_result:
            model_result = clustering_result
            embedding = np.asarray(model_result["embedding"])
            labels = np.asarray(model_result["cluster_labels"])
            fig, ax = plt.subplots(figsize=(7.2, 5.8))
            scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=10, alpha=0.55)
            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")
            ax.set_title(f"{model_result['dataset']} · 自动聚类（k={model_result['best_k']}）")
            fig.colorbar(scatter, ax=ax, label="cluster")
            fig.tight_layout()
            path = self._artifact_path("charts", "40_automatic_clustering.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({"title": "自动聚类与降维", "type": "clustering", "path": str(path), "datasets": [model_result["dataset"]]})

        for model_index, model_result in enumerate(
            [model for model in model_results if model.get("actual") is not None],
            start=1,
        ):
            actual = np.asarray(model_result["actual"])
            predicted = np.asarray(model_result["oof_prediction"])
            is_confirmation = "holdout" in str(model_result.get("validation", ""))
            prediction_label = "独立确认集预测" if is_confirmation else "OOF 预测"
            fig, ax = plt.subplots(figsize=(6.5, 5.5))
            if model_result["task_type"] == "regression":
                ax.scatter(actual, predicted, s=12, alpha=0.35, color="#2E86AB")
                limits = [float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))]
                ax.plot(limits, limits, "--", color="#C73E1D", linewidth=1)
                ax.set_xlabel("真实值")
                ax.set_ylabel(prediction_label)
                ax.set_title(f"{model_result['target']} · 真实值与{prediction_label}")
            else:
                labels, matrix = np.unique(np.concatenate([actual, predicted]), return_inverse=False), None
                from sklearn.metrics import confusion_matrix
                matrix = confusion_matrix(actual, predicted, labels=labels)
                image = ax.imshow(matrix, cmap="Blues")
                ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
                ax.set_yticks(range(len(labels)), labels)
                ax.set_xlabel("prediction")
                ax.set_ylabel("actual")
                ax.set_title(f"{model_result['target']} · {prediction_label}混淆矩阵")
                fig.colorbar(image, ax=ax)
            fig.tight_layout()
            path = self._artifact_path(
                "charts",
                f"40_{model_index:02d}_{_safe_filename(model_result.get('target', 'model'))}_validation.png",
            )
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({
                "title": f"{model_result.get('target', '模型')} 自动模型验证",
                "type": "model_validation",
                "path": str(path),
                "datasets": [model_result["dataset"]],
                "target": model_result.get("target"),
            })

            feedback = model_result.get("feedback_optimization", {})
            histories = feedback.get("optimization_history", {})
            if histories:
                fig, ax = plt.subplots(figsize=(7.5, 4.8))
                for model_key, history in histories.items():
                    ordered = sorted(history, key=lambda item: item.get("trial", 0))
                    scores = [float(item["score"]) for item in ordered if np.isfinite(item.get("score", np.nan))]
                    if not scores:
                        continue
                    running_best = np.maximum.accumulate(scores)
                    ax.plot(range(1, len(running_best) + 1), running_best, marker="o", label=model_key)
                ax.set_xlabel("trial")
                ax.set_ylabel("search CV objective")
                ax.set_title(f"{model_result.get('target', '模型')} · 参数反馈优化轨迹")
                ax.grid(alpha=0.2)
                if len(histories) > 1:
                    ax.legend()
                fig.tight_layout()
                path = self._artifact_path(
                    "charts",
                    f"41_{model_index:02d}_{_safe_filename(model_result.get('target', 'model'))}_feedback.png",
                )
                fig.savefig(path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                charts.append({
                    "title": f"{model_result.get('target', '模型')} 参数反馈优化",
                    "type": "feedback_optimization",
                    "path": str(path),
                    "datasets": [model_result["dataset"]],
                    "target": model_result.get("target"),
                })

        if ranking_result and ranking_result.get("ranking"):
            top = ranking_result["ranking"][:15][::-1]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.barh([row["entity"] for row in top], [row["score"] for row in top], color="#A23B72")
            ax.set_xlabel("TOPSIS score")
            ax.set_title("综合评价排名 Top 15")
            fig.tight_layout()
            path = self._artifact_path("charts", "50_topsis_ranking.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({"title": "TOPSIS 综合评价排名", "type": "ranking", "path": str(path), "datasets": [ranking_result["dataset"]]})

        optimization_result = specialized_results.get("optimization")
        if optimization_result and optimization_result.get("solver_success"):
            solution = optimization_result.get("solution", {})
            if solution:
                names = list(solution)
                values = [float(solution[name]) for name in names]
                ranges = optimization_result.get("near_optimal_ranges", {})
                lower_errors: List[float] = []
                upper_errors: List[float] = []
                for name, value in zip(names, values):
                    lower, upper = ranges.get(name, [None, None])
                    lower_errors.append(max(value - float(lower), 0.0) if lower is not None else 0.0)
                    upper_errors.append(max(float(upper) - value, 0.0) if upper is not None else 0.0)
                fig, ax = plt.subplots(figsize=(max(7.0, min(12.0, len(names) * 0.7)), 5.2))
                ax.bar(names, values, color="#2E86AB", alpha=0.85)
                if any(error > 1e-10 for error in lower_errors + upper_errors):
                    ax.errorbar(
                        range(len(names)), values, yerr=[lower_errors, upper_errors],
                        fmt="none", ecolor="#C73E1D", capsize=4, label="近优解范围",
                    )
                    ax.legend()
                ax.axhline(0.0, color="#666", linewidth=0.8)
                ax.set_ylabel("decision value")
                ax.set_title("显式线性模型最优方案与近优范围")
                ax.tick_params(axis="x", rotation=35)
                fig.tight_layout()
                path = self._artifact_path("charts", "55_optimization_solution.png")
                fig.savefig(path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                charts.append({
                    "title": "线性优化方案与近优范围", "type": "optimization_solution",
                    "path": str(path), "datasets": [],
                })

        graph_result = specialized_results.get("graph_network")
        if graph_result and graph_result.get("top_degree_nodes"):
            nodes = graph_result["top_degree_nodes"][:15][::-1]
            fig, ax = plt.subplots(figsize=(9, 5.5))
            ax.barh([item["node"] for item in nodes], [item["degree"] for item in nodes], color="#2E86AB")
            ax.set_xlabel("degree")
            ax.set_title("网络中心节点 Top 15")
            fig.tight_layout()
            path = self._artifact_path("charts", "60_network_degree.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({"title": "网络中心节点", "type": "network_degree", "path": str(path), "datasets": [graph_result["dataset"]]})

        dynamics = specialized_results.get("time_dynamics")
        if dynamics and dynamics.get("points"):
            points = dynamics["points"]
            times = pd.to_datetime([point["time"] for point in points], errors="coerce")
            values = [point["value"] for point in points]
            fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.plot(times, values, color="#2E86AB", linewidth=1.3)
            ax.set_title(f"{dynamics['variable']} · 时序动力特征")
            ax.set_xlabel(dynamics["time_column"])
            ax.set_ylabel(dynamics["variable"])
            fig.autofmt_xdate()
            fig.tight_layout()
            path = self._artifact_path("charts", "70_time_dynamics.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({"title": "时序动力特征", "type": "time_dynamics", "path": str(path), "datasets": [dynamics["dataset"]]})
        equation = specialized_results.get("equation_discovery")
        if equation and equation.get("validation_actual") is not None:
            actual = np.asarray(equation["validation_actual"], dtype=float)
            prediction = np.asarray(equation["validation_prediction"], dtype=float)
            fig, ax = plt.subplots(figsize=(10, 5.2))
            ax.plot(actual, color="#2E86AB", linewidth=1.5, label="观测窗口变化")
            ax.plot(prediction, color="#C73E1D", linewidth=1.2, label="候选方程预测")
            ax.axhline(0, color="#64748b", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("末段验证窗口")
            ax.set_ylabel("标准化状态变化")
            ax.set_title(
                f"{equation['target']} · 积分弱形式候选方程外推验证"
            )
            ax.legend()
            fig.tight_layout()
            path = self._artifact_path("charts", "71_equation_discovery_validation.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({
                "title": "积分弱形式候选方程验证",
                "type": "equation_discovery",
                "path": str(path),
                "datasets": [equation["dataset"]],
            })
        causal = specialized_results.get("causal_effect")
        if causal:
            lower, upper = causal["confidence_interval_95"]
            effect = causal["effect"]
            fig, ax = plt.subplots(figsize=(7.5, 3.8))
            ax.errorbar(
                [effect], [0],
                xerr=[[effect - lower], [upper - effect]],
                fmt="o", color="#A23B72", capsize=6, markersize=8,
            )
            ax.axvline(0, color="#64748b", linestyle="--", linewidth=1)
            ax.set_yticks([0], [f"{causal['treatment']} → {causal['outcome']}"])
            ax.set_xlabel("正交化处理效应及 95% 区间")
            ax.set_title("交叉拟合因果效应（依赖识别假设）")
            fig.tight_layout()
            path = self._artifact_path("charts", "72_causal_effect_interval.png")
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            charts.append({
                "title": "正交化处理效应区间",
                "type": "causal_effect",
                "path": str(path),
                "datasets": [causal["dataset"]],
            })
        return charts

    def _build_conclusions(
        self,
        relationships: Sequence[DatasetRelation],
        interactions: Sequence[InteractionFinding],
        model_results: Sequence[Dict[str, Any]],
        ranking_result: Optional[Dict[str, Any]],
        specialized_results: Dict[str, Any],
    ) -> List[str]:
        conclusions = [f"已分析 {len(self._datasets)} 个数据集，共识别 {len(relationships)} 条候选数据关系。"]
        unsafe = [relation for relation in relationships if not relation.safe_to_join]
        if unsafe:
            conclusions.append(f"其中 {len(unsafe)} 条关系存在联表膨胀风险，分析时已改为按键聚合，不能直接全量 merge。")
        if interactions:
            top = interactions[0]
            conclusions.append(top.interpretation + " 这是关联证据，不直接代表因果关系。")
        else:
            conclusions.append("未发现达到阈值的跨表数值交互；这不等于变量独立，可能需要时滞、非线性或分组分析。")
        for model_result in model_results:
            metric_text = "、".join(f"{key}={value:.4g}" for key, value in model_result.get("metrics", {}).items() if isinstance(value, (int, float)))
            if model_result["task_type"] == "clustering":
                conclusions.append(f"已对 {model_result['dataset']} 完成自动聚类，选择 k={model_result['best_k']}；{metric_text}。")
            else:
                conclusions.append(f"已对 {model_result['dataset']}.{model_result['target']} 完成自动{model_result['task_type']}，最佳模型为 {model_result['best_model']}；{metric_text}。")
            feedback = model_result.get("feedback_optimization", {})
            if feedback.get("attempted"):
                if feedback.get("accepted"):
                    conclusions.append(
                        f"参数反馈优化已通过复核，{feedback['primary_metric']} 相对改善 "
                        f"{feedback.get('relative_gain', 0):.2%}，成对重采样改善概率为 "
                        f"{feedback.get('improvement_probability', 0):.1%}。"
                    )
                else:
                    conclusions.append("参数反馈候选未通过独立确认集复核，已保留更稳健的基线结果。")
            if any(item.get("strategy") == "point_in_time" for item in model_result.get("feature_join_audit", [])):
                conclusions.append("跨表时序特征采用 point-in-time 联接，每个样本只使用当时及此前可见记录。")
            credibility = model_result.get("credibility_audit", {})
            if credibility.get("enabled"):
                conclusions.append(
                    f"结果可信度审计判定为“{credibility.get('label', '证据不足')}”："
                    f"{credibility.get('decision', credibility.get('summary', '-'))}。"
                )
            prediction_interval = model_result.get("prediction_interval")
            if prediction_interval:
                coverage = prediction_interval.get("empirical_coverage")
                coverage_text = f"，经验覆盖率 {coverage:.1%}" if coverage is not None else ""
                conclusions.append(
                    f"{model_result['target']} 的 {prediction_interval['target_coverage']:.0%} "
                    f"保序预测区间平均宽度为 {prediction_interval['mean_interval_width']:.4g}"
                    f"{coverage_text}。"
                )
        if ranking_result:
            first = ranking_result["ranking"][0]
            conclusions.append(f"熵权 TOPSIS 排名首位为 {first['entity']}（得分 {first['score']:.4f}）。")
            ranking_audit = ranking_result.get("credibility_audit", {})
            sensitivity = ranking_result.get("sensitivity", {})
            if ranking_audit:
                conclusions.append(
                    f"综合排名可信度为“{ranking_audit.get('label', '-')}”："
                    f"权重扰动后秩相关中位数 {sensitivity.get('median_rank_spearman', 0):.3f}，"
                    f"首名保持率 {sensitivity.get('winner_retention', 0):.1%}。"
                )
            pareto = ranking_result.get("pareto_analysis", {})
            if pareto:
                conclusions.append(
                    f"无权重 Pareto 审计中 {pareto.get('front_size', 0)}/"
                    f"{pareto.get('sample_size', 0)} 个对象非支配；这些方案之间存在真实指标权衡。"
                )
        graph_result = specialized_results.get("graph_network")
        if graph_result:
            conclusions.append(
                f"实体网络包含 {graph_result['n_nodes']} 个节点、{graph_result['n_unique_edges']} 条唯一边，"
                f"共 {graph_result['connected_components']} 个连通分量。"
            )
        simulation = specialized_results.get("simulation")
        if simulation:
            lower, upper = simulation["mean_confidence_interval_95"]
            conclusions.append(
                f"{simulation['dataset']}.{simulation['variable']} 的 bootstrap 均值 95% 区间为 "
                f"[{lower:.4g}, {upper:.4g}]。"
            )
        dynamics = specialized_results.get("time_dynamics")
        if dynamics:
            conclusions.append(
                f"{dynamics['dataset']}.{dynamics['variable']} 的经验线性趋势为 "
                f"{dynamics['linear_trend_per_day']:.4g}/天。"
            )
        equation = specialized_results.get("equation_discovery")
        if equation:
            conclusions.append(
                f"为 {equation['target']} 发现积分弱形式稀疏候选方程：{equation['equation']}；"
                f"审计为“{equation.get('credibility_audit', {}).get('label', '-')}”，"
                "它是可反证假设而非已证明机理。"
            )
        causal = specialized_results.get("causal_effect")
        if causal:
            lower, upper = causal["confidence_interval_95"]
            conclusions.append(
                f"{causal['treatment']} 对 {causal['outcome']} 的交叉拟合正交化效应为 "
                f"{causal['effect']:.4g}（95% 区间 [{lower:.4g}, {upper:.4g}]）；"
                f"因果审计为“{causal.get('credibility_audit', {}).get('label', '-')}”。"
            )
        structures = specialized_results.get("data_structure", [])
        for structure in structures[:3]:
            audit = structure.get("credibility_audit", {})
            conclusions.append(
                f"{structure['dataset']} 的 {structure['original_dimensions']} 个有效指标可用 "
                f"{structure['dimensions_90']} 个主维度解释 "
                f"{structure['cumulative_explained_variance']:.1%} 的标准化方差；"
                f"检测到 {structure['anomaly_count']} 个结构偏离样本，"
                f"稳定性审计为“{audit.get('label', '-')}”。"
            )
        return conclusions

    def _build_evidence_conclusions(
        self,
        evidence_bundle: Mapping[str, Any],
        relationships: Sequence[DatasetRelation],
        model_results: Sequence[Dict[str, Any]],
    ) -> List[str]:
        """Render only claims that survived the argument-graph policy.

        The returned text remains a convenience view.  Its authority comes from
        ``evidence_bundle.claims``; an optional writing API may rephrase it but is
        not allowed to introduce new mathematics or numbers.
        """
        conclusions = [
            f"已分析 {len(self._datasets)} 个数据集并识别 {len(relationships)} 条候选数据关系；"
            f"论证总状态为“{evidence_bundle.get('overall_label', '尚无数值结论')}”。"
        ]
        model_subjects = {
            (
                str(model.get("dataset", "")), str(model.get("target", "")),
                str(model.get("task_type", ""))
            ) for model in model_results
        }
        for claim in evidence_bundle.get("claims", []):
            statement = str(claim.get("statement", "")).strip()
            # Preserve the concise legacy completion phrase while attaching the
            # evidence grade.  Downstream callers can migrate without losing the
            # much stricter claim policy.
            if claim.get("claim_type") == "predictive":
                for dataset, target, task_type in model_subjects:
                    marker = f"{dataset}.{target}"
                    if marker in statement:
                        statement = f"已对 {marker} 完成自动{task_type}；{statement}"
                        break
            label = claim.get("label", "不可判定")
            disposition = claim.get("disposition")
            if disposition == "rejected":
                conclusions.append(f"[当前反证：不得采信] {statement}")
            elif disposition == "unresolved":
                conclusions.append(f"[待完成] {statement}")
            else:
                scope = str(claim.get("scope", "")).strip()
                conclusions.append(f"[{label}] {statement}" + (f" 适用边界：{scope}" if scope else ""))
        if len(conclusions) == 1:
            conclusions.append("[不可判定] 当前没有形成带计算证据的结论；方法建议不等于求解结果。")
        return conclusions

    def _write_report(self, result: ResearchResult) -> Path:
        lines = [
            "# 数学建模论证证据包", "",
            "> 本文件是可审计的数学证据摘要，不是自动生成的竞赛论文。论文写作仅允许在最后阶段显式调用 API，并且不得新增公式、数字或结论。",
            "", "## 题目", "", result.problem, "",
            "## 任务识别", "",
            f"- 主任务：{result.problem_analysis.get('model_class')}（{result.problem_analysis.get('task_type')}）",
            f"- 识别置信度：{result.problem_analysis.get('confidence')}%", "",
        ]
        spec = result.mathematical_model_spec or {}
        bundle = result.evidence_bundle or {}
        readiness_tracks = spec.get("readiness_by_track", {})
        spec_missing = [
            _mechanistic_label(item) for item in spec.get("missing_requirements", [])
        ]
        lines.extend([
            "## 数学规范与求解准备度", "",
            f"- 规范版本：{spec.get('version', '-')}",
            f"- 总准备度：**{_mechanistic_label(spec.get('readiness', 'not_assessed'))}**",
            f"- 机理数学结构：**{_mechanistic_label(readiness_tracks.get('mechanistic_structure', 'not_assessed'))}**",
            f"- 数值执行：**{_mechanistic_label(readiness_tracks.get('numerical_execution', 'not_assessed'))}**",
            f"- 观测数据建模：**{_mechanistic_label(readiness_tracks.get('observational_modeling', 'not_assessed'))}**",
            f"- 变量角色：`{json.dumps(spec.get('role_bindings', {}), ensure_ascii=False)}`",
            f"- 未执行节点仍需：{'；'.join(spec_missing) or '无'}",
            f"- 静态冲突：{len(spec.get('contradictions', []))} 项",
            "", "### 假设账本", "",
            "| 假设 | 类别 | 关键 | 可检验性 | 状态 | 当前证据 | 反证/补强方式 |",
            "|---|---|---:|---|---|---|---|",
        ])
        for assumption in spec.get("assumptions", []):
            lines.append(
                f"| {assumption.get('text', '-')} | {assumption.get('category', '-')} | "
                f"{'是' if assumption.get('critical') else '否'} | {assumption.get('testability', '-')} | "
                f"{assumption.get('status', '-')} | {str(assumption.get('evidence', '-')).replace('|', '\\|')} | "
                f"{str(assumption.get('falsification', '-')).replace('|', '\\|')} |"
            )
        lines.extend([
            "", "### 竞争模型与反证路线", "",
            "| 任务 | 候选模型 | 适用性 | 角色 | 求解器 | 准备度 | 缺失条件 | 反证检查 |",
            "|---|---|---|---|---|---|---|---|",
        ])
        for candidate in spec.get("candidate_models", []):
            lines.append(
                f"| {candidate.get('task_type', '-')} | {candidate.get('name', '-')} | "
                f"{_mechanistic_label(candidate.get('applicability', 'applicable'))} | "
                f"{candidate.get('role', '-')} | {candidate.get('solver', '-')} | "
                f"{_mechanistic_label(candidate.get('readiness', '-'))} | "
                f"{'；'.join(_mechanistic_label(item) for item in candidate.get('missing_requirements', [])) or '-'} | "
                f"{'；'.join(candidate.get('falsification_tests', [])) or '-'} |"
            )
        lines.extend([
            "", "### 安全编译计划", "",
            "| 任务 | 编译状态 | 数学结构 | 数值求解器 | 可执行 | 缺失条件 |",
            "|---|---|---|---|---:|---|",
        ])
        for compiler in spec.get("compiler_plan", []):
            lines.append(
                f"| {compiler.get('task_type', '-')} | "
                f"{_mechanistic_label(compiler.get('status', '-'))} | "
                f"{compiler.get('formulation_class', '-')} | {compiler.get('solver', '-')} | "
                f"{'是' if compiler.get('executable') else '否'} | "
                f"{'；'.join(_mechanistic_label(item) for item in compiler.get('missing_requirements', [])) or '-'} |"
            )
        lines.extend([
            "", "## 论证结论分级", "",
            f"- 总状态：**{bundle.get('overall_label', '尚无数值结论')}**（{bundle.get('overall_status', 'no_claims')}）",
            f"- 分级计数：`{json.dumps(bundle.get('grade_counts', {}), ensure_ascii=False)}`",
            f"- 论证图完整性：{bundle.get('argument_integrity', {}).get('status', 'not_assessed')}",
            "", "| 结论 | 等级 | 处置 | 适用范围 | 失效条件 |",
            "|---|---|---|---|---|",
        ])
        for claim in bundle.get("claims", []):
            lines.append(
                f"| {str(claim.get('statement', '-')).replace('|', '\\|')} | "
                f"{claim.get('label', '-')} | {claim.get('disposition', '-')} | "
                f"{str(claim.get('scope', '-')).replace('|', '\\|')} | "
                f"{'；'.join(claim.get('invalid_when', [])) or '-'} |"
            )
        writing = bundle.get("writing_contract", {})
        lines.extend([
            "", "### 最终写作 API 契约", "",
            f"- 当前启用：{writing.get('enabled', False)}",
            f"- 可写入的结论 ID：{writing.get('allowed_claim_ids', [])}",
            f"- 禁止肯定表述的结论 ID：{writing.get('prohibited_claim_ids', [])}",
            f"- 规则：{writing.get('instruction', '-')}",
        ])
        if result.dataset_profiles:
            lines.extend([
                "", "## 数据集画像", "",
                "| 数据集 | 角色 | 行数 | 列数 | 数值列 | 类别列 | 时间列 | 缺失率 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for profile in result.dataset_profiles:
                lines.append(f"| {profile.name} | {profile.role} | {profile.source_rows:,} | {profile.n_columns} | {len(profile.numeric_columns)} | {len(profile.categorical_columns)} | {len(profile.datetime_columns)} | {profile.missing_rate:.2%} |")
        task_graph = result.problem_analysis.get("task_graph", [])
        if task_graph:
            lines.extend([
                "", "## 多子问题执行图", "",
                "| 节点 | 子问题 | 任务 | 状态 | 上游依赖 | 执行证据/缺失条件 |",
                "|---|---|---|---|---|---|",
            ])
            for node in task_graph:
                text = str(node.get("text", "-")).replace("|", "\\|").replace("\n", " ")
                evidence = str(node.get("evidence", "-")).replace("|", "\\|")
                missing = "；".join(
                    _mechanistic_label(item) for item in node.get("missing_requirements", [])
                )
                detail = evidence + (f"；缺少：{missing}" if missing else "")
                dependencies = "、".join(node.get("depends_on", [])) or "-"
                lines.append(
                    f"| {node.get('id', '-')} | {text} | {node.get('task_type', '-')} | "
                    f"{node.get('status', '-')} | {dependencies} | {detail} |"
                )
        if result.dataset_profiles:
            lines.extend(["", "## 数据关系", ""])
            if result.relationships:
                lines.extend(["| 左表.键 | 右表.键 | 基数 | 置信度 | 值域覆盖 | 安全性 |", "|---|---|---|---:|---:|---|"])
                for relation in result.relationships:
                    lines.append(f"| {relation.left_dataset}.{relation.left_key} | {relation.right_dataset}.{relation.right_key} | {relation.relationship} | {relation.confidence:.1f}% | {relation.value_overlap:.1%} | {'可直接关联' if relation.safe_to_join else '先聚合'} |")
            else:
                lines.append("未找到有足够证据的跨表关联键。")
            lines.extend(["", "## 跨数据集交互", ""])
            for item in result.interactions[:15]:
                significance = f"，p={item.p_value:.4g}" if item.p_value is not None else ""
                q_value = f"，FDR q={item.q_value:.4g}" if item.q_value is not None else ""
                fdr_decision = (
                    "，FDR显著" if item.significant is True else
                    ("，FDR不显著（仅探索）" if item.significant is False else "，未检验显著性")
                )
                interval = f"，区间={item.confidence_interval}" if item.confidence_interval else ""
                conditional = f"，条件ρ={item.conditional_strength:.4g}" if item.conditional_strength is not None else ""
                stability = f"，稳定性={item.stability_score:.1%}" if item.stability_score is not None else ""
                controls = f"，控制变量={item.conditioning_variables}" if item.conditioning_variables else ""
                lines.append(f"- [{item.method}] {item.interpretation} 样本数={item.sample_size}{significance}{q_value}{fdr_decision}{interval}{conditional}{stability}{controls}")
            if not result.interactions:
                lines.append("- 未发现达到当前阈值的跨表数值交互。")
        lines.extend(["", "## 执行计划", ""])
        for index, step in enumerate(result.analysis_plan, 1):
            lines.append(f"{index}. **{step['phase']}**：{step['action']}（{step['method']}）")
        lines.extend(["", "## 题型能力边界", "", "| 任务 | 状态 | 已有证据 | 仍需条件 |", "|---|---|---|---|"])
        for capability in result.capability_report.get("tasks", []):
            capability_requirement = "；".join(
                _mechanistic_label(item)
                for item in str(capability.get("requirement") or "").split("；")
                if item
            ) or "-"
            lines.append(
                f"| {capability['task_type']} | {capability['status']} | "
                f"{capability.get('evidence') or '-'} | {capability_requirement} |"
            )
        if result.model_results:
            lines.extend(["", "## 自动模型", ""])
            for model_index, current_model in enumerate(result.model_results, 1):
                subject = current_model['dataset']
                if current_model.get('target'):
                    subject += f".{current_model['target']}"
                lines.extend([f"### 模型 {model_index}：{subject}", "", f"- 任务：{current_model['task_type']}", f"- 最佳模型：{current_model['best_model']}"])
                for metric, value in current_model.get("metrics", {}).items():
                    lines.append(f"- {metric}: {value}")
                feedback = current_model.get("feedback_optimization", {})
                if feedback:
                    lines.extend([
                        "", "#### 验证结果反馈优化", "",
                        f"- 是否尝试：{feedback.get('attempted', False)}",
                        f"- 是否采用调优结果：{feedback.get('accepted', False)}",
                        f"- 原因：{feedback.get('reason', '-')}",
                    ])
                    if feedback.get("attempted"):
                        lines.extend([
                            f"- 基线 {feedback.get('primary_metric')}：{feedback.get('baseline_score')}",
                            f"- 调优 {feedback.get('primary_metric')}：{feedback.get('tuned_score')}",
                            f"- 相对改善：{feedback.get('relative_gain')}",
                            f"- 独立确认方式：{feedback.get('confirmation', '-')}",
                            f"- 独立确认样本数：{feedback.get('confirmation_samples', 0)}",
                            f"- 成对重采样改善概率：{feedback.get('improvement_probability')}",
                            f"- 候选选择依据：{feedback.get('candidate_selection_reason', '-')}",
                            f"- 最优参数：`{json.dumps(feedback.get('optimized_params', {}), ensure_ascii=False)}`",
                        ])
                    recommendations = feedback.get("diagnostics", {}).get("recommendations", [])
                    if recommendations:
                        lines.append(f"- 诊断建议：{'；'.join(recommendations)}")
                credibility = current_model.get("credibility_audit", {})
                if credibility:
                    lines.extend([
                        "", "#### 结果可信度审计", "",
                        f"- 判定：**{credibility.get('label', '证据不足')}**（{credibility.get('status', 'not_assessed')}）",
                        f"- 使用建议：{credibility.get('decision', '-')}",
                        f"- 摘要：{credibility.get('summary', '-')}",
                        "", "| 审计项 | 状态 | 证据 | 建议 |", "|---|---|---|---|",
                    ])
                    for check in credibility.get("checks", []):
                        evidence = str(check.get("evidence", "-")).replace("|", "\\|")
                        recommendation = str(check.get("recommendation", "-") or "-").replace("|", "\\|")
                        lines.append(
                            f"| {check.get('name', '-')} | {check.get('status', '-')} | "
                            f"{evidence} | {recommendation} |"
                        )
                    if credibility.get("next_actions"):
                        lines.extend(["", "优先处理："])
                        lines.extend(f"- {item}" for item in credibility["next_actions"])
                    if credibility.get("limitations"):
                        lines.extend(["", "审计边界："])
                        lines.extend(f"- {item}" for item in credibility["limitations"])
                prediction_interval = current_model.get("prediction_interval")
                if prediction_interval:
                    lines.extend([
                        "", "#### 保序预测区间", "",
                        f"- 目标覆盖率：{prediction_interval['target_coverage']:.1%}",
                        f"- 校准样本：{prediction_interval['calibration_samples']}",
                        f"- 区间半径：{prediction_interval['radius']}",
                        f"- 平均宽度：{prediction_interval['mean_interval_width']}",
                        f"- 经验覆盖率：{prediction_interval.get('empirical_coverage')}",
                        f"- 覆盖评估：{prediction_interval.get('coverage_evaluation')}",
                        f"- 边界：{prediction_interval.get('note', '-')}",
                    ])
                join_audit = current_model.get("feature_join_audit", [])
                if join_audit:
                    lines.extend(["", "#### 跨表特征时间审计", "", "| 来源数据集 | 策略 | 新增特征 | 说明 |", "|---|---|---:|---|"])
                    for audit in join_audit:
                        lines.append(
                            f"| {audit.get('dataset', '-')} | {audit.get('strategy', '-')} | "
                            f"{audit.get('features_added', 0)} | {audit.get('reason', '-')} |"
                        )
                if current_model.get("feature_importance"):
                    lines.extend(["", "#### 重要特征", "", "| 特征 | 重要性 |", "|---|---:|"])
                    for item in current_model["feature_importance"][:15]:
                        feature = item.get("feature", item.get("column", "-"))
                        importance = item.get("importance", item.get("score", "-"))
                        lines.append(f"| {feature} | {importance} |")
                lines.append("")
        if result.ranking_result:
            lines.extend(["", "## 综合评价", "", f"方法：{result.ranking_result['method']}", "", "| 排名 | 对象 | 得分 |", "|---:|---|---:|"])
            for row in result.ranking_result["ranking"][:20]:
                lines.append(f"| {row['rank']} | {row['entity']} | {row['score']:.6f} |")
            ranking_audit = result.ranking_result.get("credibility_audit", {})
            if ranking_audit:
                sensitivity = result.ranking_result.get("sensitivity", {})
                lines.extend([
                    "", "### 排名可信度", "",
                    f"- 判定：**{ranking_audit.get('label', '-')}**",
                    f"- 结论：{ranking_audit.get('decision', '-')}",
                    f"- 权重扰动中位秩相关：{sensitivity.get('median_rank_spearman', '-')}",
                    f"- 首名保持率：{sensitivity.get('winner_retention', '-')}",
                    f"- 逐一删除指标后的首名保持率：{sensitivity.get('leave_one_indicator_out_winner_retention', '-')}",
                ])
            pareto = result.ranking_result.get("pareto_analysis", {})
            if pareto:
                lines.extend([
                    "", "### Pareto 非支配方案", "",
                    f"- 审计样本：{pareto.get('sample_size', 0)}",
                    f"- 非支配方案：{pareto.get('front_size', 0)}（{pareto.get('front_share', 0):.1%}）",
                    f"- 冲突指标对：{len(pareto.get('conflicting_indicator_pairs', []))}",
                    "", "| 对象 | TOPSIS 得分 |", "|---|---:|",
                ])
                for row in pareto.get("front", [])[:20]:
                    lines.append(f"| {row['entity']} | {row['topsis_score']:.6f} |")
        mechanistic_preview = result.specialized_results.get("mechanistic_model", {})
        mechanistic_preview_ir = mechanistic_preview.get("mathematical_ir", {})
        semantic_model_preview = mechanistic_preview.get("semantic_model_compilation", {})
        has_mechanistic_preview = mechanistic_preview.get("presentation_scope", "primary") == "primary" and (
            bool(mechanistic_preview.get("operator_graph")) or any(
                mechanistic_preview_ir.get(key) for key in (
                    "entities", "quantities", "relations", "objectives", "constraints"
                )
            ) or semantic_model_preview.get("status") not in {None, "not_configured"}
        )
        has_other_specialized = any(
            key != "mechanistic_model" and bool(value)
            for key, value in result.specialized_results.items()
        )
        if has_mechanistic_preview or has_other_specialized:
            lines.extend(["", "## 专项数学分析", ""])
            data_compilation = result.specialized_results.get(
                "mathematical_data_compilation"
            )
            if data_compilation:
                contract = data_compilation.get("contract", {})
                summary = data_compilation.get("summary", {})
                lines.extend([
                    "### 数学数据编译与多视图反证", "",
                    f"- 数据集：{data_compilation.get('dataset', '-')}；状态：**{data_compilation.get('status', '-')}**",
                    f"- 估计对象：{contract.get('estimand', '-')} ",
                    f"- 观测粒度：{contract.get('observed_grain', [])}；唯一率：{contract.get('grain_uniqueness', '-')}；"
                    f"状态：{contract.get('grain_status', '-')} ",
                    f"- 候选视图：{summary.get('candidate_views', 0)}；可采用：{summary.get('admissible_views', 0)}；阻断：{summary.get('blocked_views', 0)}",
                    f"- 审计范围：{summary.get('audited_rows', 0):,}/{summary.get('source_rows', 0):,} 行；"
                    f"{'应用时需完整复审' if summary.get('sampled_execution') else '完整数据'}",
                    f"- 多表契约：{summary.get('cross_dataset_contracts', 0)}；"
                    f"阻断原始连接：{summary.get('blocked_cross_dataset_contracts', 0)}",
                    f"- 检验关系：{summary.get('relationships_tested', 0)}；方向翻转：**{summary.get('direction_reversals', 0)}**",
                    "", "| 预测/解释变量 | 目标 | 状态 | 全局ρ（95%区间） | FDR q | 效应跨度 | Simpson风险 | 处置 |",
                    "|---|---|---|---|---:|---:|---:|---|",
                ])
                for relationship in (
                    (data_compilation.get("conclusion_stress") or {}).get(
                        "relationships", []
                    )
                ):
                    disposition = (
                        "拒绝无条件总体规律" if relationship.get("status") == "contradicted"
                        else "限定视图与粒度报告"
                    )
                    global_context = next(
                        (
                            item for item in relationship.get("contexts", [])
                            if item.get("view") == "global_complete_case"
                        ),
                        {},
                    )
                    lines.append(
                        f"| {relationship.get('predictor', '-')} | {relationship.get('target', '-')} | "
                        f"{relationship.get('status', '-')} | {global_context.get('rho', '-')} "
                        f"({global_context.get('confidence_interval_95', '-')}) | "
                        f"{relationship.get('global_fdr_q', '-')} | {relationship.get('effect_spread', '-')} | "
                        f"{'是' if relationship.get('simpson_risk') else '否'} | {disposition} |"
                    )
                cross_contracts = data_compilation.get("cross_dataset_contracts", [])
                if cross_contracts:
                    lines.extend([
                        "", "#### 跨表粒度与连接契约", "",
                        "| 数据表 | 键 | 基数 | 膨胀估计 | 时间规则 | 状态 | 复审 | 数学处置 |",
                        "|---|---|---|---:|---|---|---|---|",
                    ])
                    for cross in cross_contracts:
                        keys = "；".join(
                            f"{item.get('left')}↔{item.get('right')}"
                            for item in cross.get("key_pairs", [])
                        ) or "未发现"
                        lines.append(
                            f"| {cross.get('left_dataset', '-')} ↔ {cross.get('right_dataset', '-')} | "
                            f"{keys} | {cross.get('relationship', '-')} | "
                            f"{cross.get('estimated_expansion', '-')} | "
                            f"{'point-in-time' if cross.get('point_in_time_required') else '普通键对齐'} | "
                            f"{cross.get('status', '-')} | "
                            f"{'需要全表复审' if cross.get('full_cardinality_reaudit_required') else '完整基数'} | "
                            f"{cross.get('combined_additive_analysis', '-')} |"
                        )
                lines.extend([
                    "",
                    "> 多视图方向稳定只能排除一部分数据表述脆弱性；它不把相关性升级为因果关系。",
                    "",
                ])
            mechanistic = result.specialized_results.get("mechanistic_model")
            mechanistic_ir = mechanistic.get("mathematical_ir", {}) if mechanistic else {}
            if mechanistic and mechanistic.get("presentation_scope", "primary") == "primary" and (
                mechanistic.get("operator_graph")
                or any(mechanistic_ir.get(key) for key in (
                    "entities", "quantities", "relations", "objectives", "constraints"
                ))
                or mechanistic.get("semantic_model_compilation", {}).get("status")
                not in {None, "not_configured"}
            ):
                math_ir = mechanistic_ir
                compiler = mechanistic.get("compiler_plan", {})
                model_draft = mechanistic.get("model_draft", {})
                semantic_model = mechanistic.get("semantic_model_compilation", {})
                four_layer = mechanistic.get("four_layer_pipeline", {})
                semantic_contract = four_layer.get("semantic_contract", {})
                unified_ir = four_layer.get("mathematical_ir", {})
                solver_plan = four_layer.get("solver_plan", {})
                independent_audit = four_layer.get("independent_audit", {})
                completed_stages = [
                    _mechanistic_label(item)
                    for item in model_draft.get("completed_stages", [])
                ]
                compiler_blockers = [
                    _mechanistic_label(item)
                    for item in compiler.get("blocked_by", [])
                ]
                missing_conditions = [
                    _mechanistic_label(item)
                    for item in mechanistic.get("missing_requirements", [])
                ]
                if semantic_model.get("status") != "not_configured":
                    semantic_config = semantic_model.get("configuration", {})
                    lines.extend([
                        "### 受约束语义模型编译", "",
                        f"- 状态：**{semantic_model.get('status', '-')}**",
                        f"- 后端：{semantic_config.get('provider', '-')} / "
                        f"{semantic_config.get('model_name', '-')}；API 密钥未写入产物。",
                        f"- 接受/延后关系：{semantic_model.get('accepted_count', 0)}/"
                        f"{semantic_model.get('deferred_count', 0)}",
                        "- 权限边界：模型只提出候选 IR；逐字段题面引文、数值溯源和"
                        "确定性契约复核全部通过后才允许进入求解层。",
                    ])
                    if semantic_model.get("error"):
                        lines.append(f"- 安全降级原因：{semantic_model.get('error')}")
                    for proposal in semantic_model.get("deferred_proposals", [])[:12]:
                        lines.append(
                            f"- 延后 `{proposal.get('id') or proposal.get('index', '-')}`："
                            f"{'；'.join(proposal.get('errors', [])) or '未通过语义证据门'}"
                        )
                    lines.append("")
                if four_layer:
                    plan_budget = solver_plan.get("budget_summary", {})
                    audit_coverage = independent_audit.get("coverage", {})
                    structure_catalog = unified_ir.get("structure_catalog", [])
                    candidate_structures = semantic_contract.get("candidate_structures", [])
                    implemented_structures = sum(
                        item.get("execution_status") == "implemented"
                        for item in structure_catalog
                    )
                    lines.extend([
                        "### 四层数学建模流水线", "",
                        f"- 第一层 · 题意契约：**{semantic_contract.get('status', '-')}**；"
                        f"符号 {len(semantic_contract.get('symbol_table', []))} 个；"
                        f"来源覆盖率 {semantic_contract.get('provenance', {}).get('coverage', 0):.1%}",
                        f"- 第二层 · 统一数学 IR：**{unified_ir.get('status', '-')}**；"
                        f"可执行/延后节点 {unified_ir.get('validation', {}).get('executable_nodes', 0)}/"
                        f"{unified_ir.get('validation', {}).get('deferred_nodes', 0)}；"
                        f"仅语义候选 {unified_ir.get('validation', {}).get('semantic_candidates', 0)}",
                        f"- 第三层 · 结构选解：**{solver_plan.get('status', '-')}**；"
                        f"可运行/延后 {plan_budget.get('runnable_nodes', 0)}/"
                        f"{plan_budget.get('deferred_nodes', 0)}；失败隔离="
                        f"{plan_budget.get('node_failure_isolation', False)}",
                        f"- 第四层 · 独立审计：**{independent_audit.get('status', '-')}**；"
                        f"覆盖 {audit_coverage.get('audited_results', 0)}/"
                        f"{audit_coverage.get('executed_results', 0)} 个结果",
                        f"- 数学结构目录：{len(structure_catalog)} 类；"
                        f"已实现后端 {implemented_structures} 类；"
                        f"仅识别 {len(structure_catalog) - implemented_structures} 类。",
                        f"- 题面结构候选："
                        f"{'、'.join(item.get('key', '-') for item in candidate_structures) or '未可靠识别'}；"
                        "候选必须形成完整结构化契约后才可执行。",
                        "- 选择规则：只依据数学形式，不依据题目标题或比赛题号。",
                        "", "| IR 节点 | 数学形式 | 求解器族 | 状态 | 资源上限 |",
                        "|---|---|---|---|---|",
                    ])
                    for plan_node in solver_plan.get("nodes", [])[:40]:
                        budget = plan_node.get("resource_budget", {})
                        lines.append(
                            f"| {plan_node.get('ir_node_id', '-')} | "
                            f"{plan_node.get('mathematical_form', '-')} | "
                            f"{plan_node.get('solver_family') or '-'} | "
                            f"{plan_node.get('status', '-')} | "
                            f"变量≤{budget.get('max_variables', '-')}；"
                            f"评估≤{budget.get('max_evaluations', '-')}；"
                            f"软墙钟预算 {budget.get('wall_time_budget_seconds', '-')}s |"
                        )
                    if structure_catalog:
                        lines.extend([
                            "", "#### 通用数学结构能力矩阵", "",
                            "| 数学结构 | 家族 | 后端状态 | 求解器 | 结构化契约必需字段 |",
                            "|---|---|---|---|---|",
                        ])
                        for structure in structure_catalog:
                            solver = structure.get("solver") or {}
                            lines.append(
                                f"| {structure.get('key', '-')} | {structure.get('family', '-')} | "
                                f"{structure.get('execution_status', '-')} | "
                                f"{solver.get('solver_family') or '-'} | "
                                f"{'、'.join(structure.get('required_contract_fields', [])) or '-'} |"
                            )
                    if unified_ir.get("deferred_semantic_relations"):
                        lines.extend([
                            "", "#### 未进入求解计划的语义候选", "",
                            "| 原关系 | 类型 | 延后原因 | 原文证据 |",
                            "|---|---|---|---|",
                        ])
                        for relation in unified_ir.get("deferred_semantic_relations", [])[:30]:
                            source_text = str(relation.get("source_text", "-")).replace("|", "\\|")
                            lines.append(
                                f"| {relation.get('relation_id', '-')} | {relation.get('kind', '-')} | "
                                f"{relation.get('reason', '-')} | {source_text} |"
                            )
                    if independent_audit.get("execution_failures"):
                        lines.extend([
                            "", "#### 隔离的执行失败", "", "```json",
                            json.dumps(
                                independent_audit.get("execution_failures", []),
                                ensure_ascii=False, indent=2,
                            ),
                            "```",
                        ])
                lines.extend([
                    "### 纯题面通用数学 IR", "",
                    f"- IR 版本：{mechanistic.get('schema_version', '-')}",
                    f"- 状态：**{mechanistic.get('execution_status', '-')}**",
                    f"- 模型草案状态：**{model_draft.get('status', '-')}**",
                    f"- 已完成阶段：{' → '.join(completed_stages) or '-'}",
                    f"- 实体/显式量/关系：{len(math_ir.get('entities', []))}/"
                    f"{len(math_ir.get('quantities', []))}/{len(math_ir.get('relations', []))}",
                    f"- 通用算子数：{len(mechanistic.get('operator_graph', []))}",
                    f"- 规范方程草案数：{len(model_draft.get('equations', []))}",
                    f"- 求解路线：{'；'.join(compiler.get('solver_routes', [])) or '-'}",
                    f"- 数值求解待补条件：{'；'.join(compiler_blockers) or '无'}",
                    f"- 待绑定条件：{'；'.join(missing_conditions) or '无'}",
                    "- 边界：此处只报告题面到数学结构的编译；未通过安全门时不生成数值答案。",
                    "", "| 算子 | 类别 | 状态 | 求解器路线 | 未绑定角色 |",
                    "|---|---|---|---|---|",
                ])
                for node in mechanistic.get("operator_graph", []):
                    lines.append(
                        f"| {node.get('key', '-')} | {node.get('category', '-')} | "
                        f"{node.get('status', '-')} | {node.get('solver_route', '-')} | "
                        f"{'；'.join(node.get('missing_bindings', [])) or '-'} |"
                    )
                if model_draft.get("equations"):
                    lines.extend([
                        "", "#### 规范方程草案", "",
                        "| 通用算子 | 规范形式 | 当前状态 | 仍需绑定 |",
                        "|---|---|---|---|",
                    ])
                    for equation in model_draft.get("equations", []):
                        expression = str(equation.get("expression", "-")).replace("|", "\\|")
                        equation_missing = "；".join(
                            _mechanistic_label(item)
                            for item in equation.get("missing_bindings", [])
                        ) or "无"
                        lines.append(
                            f"| {equation.get('operator', '-')} | `{expression}` | "
                            f"{equation.get('status', '-')} | {equation_missing} |"
                        )
                if model_draft.get("assumption_questions"):
                    lines.extend(["", "#### 数值求解前必须回答的假设问题", ""])
                    lines.extend(
                        f"- {question}"
                        for question in model_draft.get("assumption_questions", [])
                    )
                for index, numerical in enumerate(mechanistic.get("numerical_results", []), 1):
                    result_independent_audit = numerical.get("independent_audit", {})
                    lines.extend([
                        "", f"#### 通用数值执行 {index}", "",
                        f"- 类型/状态：{numerical.get('kind', '-')}/{numerical.get('status', '-')}",
                        f"- 求解器：{numerical.get('solver', '-')}",
                        f"- 四层独立审计：**{result_independent_audit.get('grade', 'not_assessed')}**；"
                        f"风险标记：{'、'.join(result_independent_audit.get('false_confidence_flags', [])) or '无'}",
                    ])
                    if numerical.get("kind") == "kinematic_visibility_event":
                        lines.extend([
                            f"- 主语义有效时长：**{numerical.get('duration', 0):.6f} s**",
                            f"- 有效区间：`{json.dumps(numerical.get('effective_intervals', []), ensure_ascii=False)}`",
                            f"- 投放点：`{json.dumps(numerical.get('release_point', []), ensure_ascii=False)}`",
                            f"- 激活时刻/位置：{numerical.get('activation_time')} s / "
                            f"`{json.dumps(numerical.get('activation_point', []), ensure_ascii=False)}`",
                            f"- 目标代表点语义范围：`{json.dumps(numerical.get('semantic_duration_range', []), ensure_ascii=False)}` s",
                            f"- 语义分支：`{json.dumps(numerical.get('semantic_branches', []), ensure_ascii=False)}`",
                            f"- 网格与根精化复算：`{json.dumps(numerical.get('convergence', {}), ensure_ascii=False)}`",
                            f"- 可信度：**{numerical.get('credibility_audit', {}).get('label', '-')}**",
                            "- 解释：该数值是已声明几何语义下的条件性结果；语义范围用于识别“似对非对”。",
                        ])
                    elif numerical.get("kind") == "kinematic_visibility_optimization_solution":
                        lines.extend([
                            f"- 最佳可行候选时长：**{numerical.get('duration', 0):.6f} s**",
                            f"- 决策参数：`{json.dumps(numerical.get('solution', {}), ensure_ascii=False)}`",
                            f"- 投放点：`{json.dumps(numerical.get('release_point', []), ensure_ascii=False)}`",
                            f"- 激活点：`{json.dumps(numerical.get('activation_point', []), ensure_ascii=False)}`",
                            f"- 有效区间：`{json.dumps(numerical.get('effective_intervals', []), ensure_ascii=False)}`",
                            f"- 多起点近优相对差：{numerical.get('multistart_relative_spread')}",
                            f"- 决策扰动最大相对下降：{numerical.get('maximum_relative_sensitivity_drop')}",
                            f"- 单变量99%近优范围：`{json.dumps(numerical.get('one_at_a_time_99pct_ranges', {}), ensure_ascii=False)}`",
                            f"- 舍入实施方案：`{json.dumps(numerical.get('implementation_candidate', {}), ensure_ascii=False)}`",
                            f"- 基线反馈优化：`{json.dumps(numerical.get('feedback_optimization', {}), ensure_ascii=False)}`",
                            f"- 目标语义范围：`{json.dumps(numerical.get('semantic_duration_range', []), ensure_ascii=False)}` s",
                            f"- 可信度：**{numerical.get('credibility_audit', {}).get('label', '-')}**",
                            "- 解释：这是通过多种子、构造初值、局部精化和事件根复算的高质量可行候选，未声称非凸全局最优。",
                        ])
                    else:
                        lines.extend([
                            f"- 时间范围/输出点：{numerical.get('time_span', '-')}/{numerical.get('output_points', '-')}",
                            f"- 状态摘要：`{json.dumps(numerical.get('summary', {}), ensure_ascii=False)}`",
                            f"- 容差复算：`{json.dumps(numerical.get('convergence', {}), ensure_ascii=False)}`",
                            "- 解释：数值收敛只支持该方程契约内的计算，不自动证明题面解释或现实机理。",
                        ])
            hierarchical_sales = result.specialized_results.get("hierarchical_distribution")
            if hierarchical_sales:
                concentration = hierarchical_sales.get("concentration", {})
                hierarchical_audit = hierarchical_sales.get("credibility_audit", {})
                category_column = hierarchical_sales.get("parent_dimension") or "上层维度"
                item_column = hierarchical_sales.get("child_dimension") or "下层维度"
                lines.extend([
                    f"### {category_column}—{item_column}层级分布与剩余联动", "",
                    f"- 聚合：{hierarchical_sales.get('source_rows_aggregated', 0):,} 行 → "
                    f"{hierarchical_sales.get('daily_item_rows', 0):,} 个"
                    f"日×{category_column}×{item_column}观测",
                    f"- 下层对象数/HHI/前20份额：{concentration.get('child_count', 0)} / "
                    f"{concentration.get('hhi', 0):.6g} / {concentration.get('top_20_share', 0):.1%}",
                    f"- 可信度：**{hierarchical_audit.get('label', '-')}**；"
                    f"{hierarchical_audit.get('decision', '-')}",
                    f"- 边界：{hierarchical_sales.get('note', '-')}", "",
                    f"#### {category_column}分布", "",
                    f"| {category_column} | 总量 | 日均 | 日标准差 | 变异系数 | P10 | 中位数 | P90 |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ])
                for row in hierarchical_sales.get("parent_summary", [])[:50]:
                    lines.append(
                        f"| {row.get(category_column, '-')} | {row.get('total', 0):.6g} | "
                        f"{row.get('daily_mean', 0):.6g} | {row.get('daily_std', 0):.6g} | "
                        f"{row.get('coefficient_of_variation', 0):.6g} | {row.get('q10', 0):.6g} | "
                        f"{row.get('median', 0):.6g} | {row.get('q90', 0):.6g} |"
                    )
                lines.extend([
                    "", "#### FDR 后的剩余联动", "",
                    "| 层级 | 对象A | 对象B | 残差ρ | p | q | 结论 |",
                    "|---|---|---|---:|---:|---:|---|",
                ])
                for level, records in (
                    (str(category_column), hierarchical_sales.get("parent_associations", [])),
                    (str(item_column), hierarchical_sales.get("child_associations", [])),
                ):
                    for row in records[:30]:
                        lines.append(
                            f"| {level} | {row.get('left', '-')} | {row.get('right', '-')} | "
                            f"{row.get('residual_spearman', 0):.6g} | {row.get('p_value', 0):.4g} | "
                            f"{row.get('q_value', 0):.4g} | "
                            f"{'FDR显著' if row.get('significant') else '仅探索'} |"
                        )
                lines.append("")
            grouped_forecasts = list(result.specialized_results.get("grouped_forecasts") or [])
            if not grouped_forecasts and result.specialized_results.get("grouped_forecast"):
                grouped_forecasts = [result.specialized_results["grouped_forecast"]]
            for grouped_forecast in grouped_forecasts:
                grouped_audit = grouped_forecast.get("credibility_audit", {})
                grouped_metrics = grouped_forecast.get("metrics", {})
                lines.extend([
                    f"### 分组时间粒度预测 · {grouped_forecast.get('requested_grain', 'group')}", "",
                    f"- 对象：{grouped_forecast.get('dataset')}.{grouped_forecast.get('target')}",
                    f"- 编译粒度：日 × {grouped_forecast.get('group_column')}；"
                    f"分组字段来源：{grouped_forecast.get('group_column_source')}",
                    f"- 聚合：{grouped_forecast.get('aggregation')}；"
                    f"{grouped_forecast.get('source_rows_aggregated', 0):,} 行 → "
                    f"{grouped_forecast.get('daily_group_rows', 0):,} 个日×组观测",
                    f"- 预测范围：{grouped_forecast.get('forecast_period')}；"
                    f"组数：{grouped_forecast.get('groups_forecast')}；"
                    f"模型选择：`{json.dumps(grouped_forecast.get('selected_model_counts', {}), ensure_ascii=False)}`",
                    f"- 末段回测：`{json.dumps(grouped_metrics, ensure_ascii=False)}`",
                    f"- 可信度：**{grouped_audit.get('label', '-')}**；{grouped_audit.get('decision', '-')}",
                    f"- 边界：{grouped_forecast.get('note', '-')}", "",
                    "| 组 | 日期 | 点预测 | 90% 下界 | 90% 上界 | 选用模型 |",
                    "|---|---|---:|---:|---:|---|",
                ])
                for row in grouped_forecast.get("forecasts", [])[:500]:
                    lines.append(
                        f"| {row.get('group', '-')} | {row.get('date', '-')} | "
                        f"{row.get('forecast', 0):.6g} | {row.get('lower_90', 0):.6g} | "
                        f"{row.get('upper_90', 0):.6g} | {row.get('selected_model', '-')} |"
                    )
                lines.append("")
            prescriptive_decisions = list(
                result.specialized_results.get("prescriptive_decisions") or []
            )
            if not prescriptive_decisions and result.specialized_results.get("prescriptive_decision"):
                prescriptive_decisions = [result.specialized_results["prescriptive_decision"]]
            for prescriptive in prescriptive_decisions:
                prescriptive_audit = prescriptive.get("credibility_audit", {})
                solver_summary = (prescriptive.get("solver_result") or {}).get("summary", {})
                parent_coverage = prescriptive.get("aggregate_parent_demand_coverage")
                parent_coverage_text = (
                    f"{parent_coverage:.1%}" if parent_coverage is not None else "-"
                )
                parent_shortage = prescriptive.get("minimum_total_parent_shortage")
                parent_shortage_text = (
                    f"{parent_shortage:.6g}" if parent_shortage is not None else "-"
                )
                lines.extend([
                    f"### 预测—补货—定价组合决策 · {prescriptive.get('requested_grain', 'group')}", "",
                    f"- 数学形式：`{prescriptive.get('mathematical_form', '-')}`；"
                    f"通用求解摘要：`{json.dumps(solver_summary, ensure_ascii=False)}`",
                    f"- 决策数：{prescriptive.get('decision_count', 0)}；"
                    f"允许调价/保持参考价：{prescriptive.get('price_decision_count', 0)}/"
                    f"{prescriptive.get('held_price_count', 0)}",
                    f"- 成本/损耗覆盖率：{prescriptive.get('cost_coverage', 0):.1%}/"
                    f"{prescriptive.get('loss_coverage', 0):.1%}",
                    f"- 数量边界/最小陈列量：{prescriptive.get('selection_bounds') or '-'} / "
                    f"{prescriptive.get('minimum_display', 0):.6g}",
                    f"- 上层品类需求覆盖："
                    f"{parent_coverage_text}；"
                    f"词典序最小缺口已验证："
                    f"{'是' if prescriptive.get('hierarchical_lexicographic_verified') else '否/不适用'}；"
                    f"最小总缺口："
                    f"{parent_shortage_text}",
                    f"- 可信度：**{prescriptive_audit.get('label', '-')}**；"
                    f"{prescriptive_audit.get('decision', '-')}",
                    f"- 边界：{prescriptive.get('note', '-')}", "",
                    "| 组 | 日期 | 需求预测 | 建议补货 | 补货90%范围 | 建议价格 | 单位成本 | 期望收益 |",
                    "|---|---|---:|---:|---|---:|---:|---:|",
                ])
                for row in prescriptive.get("decision_rows", [])[:500]:
                    cost = row.get("unit_cost")
                    lines.append(
                        f"| {row.get('group', '-')} | {row.get('date', '-')} | "
                        f"{row.get('forecast_demand', 0):.6g} | {row.get('replenishment', 0):.6g} | "
                        f"[{row.get('lower_replenishment_90', 0):.6g}, "
                        f"{row.get('upper_replenishment_90', 0):.6g}] | "
                        f"{row.get('price', 0):.6g} | "
                        f"{('-' if cost is None else f'{cost:.6g}')} | {row.get('payoff', 0):.6g} |"
                    )
                risk_stress = prescriptive.get("risk_aware_stress_test") or {}
                if risk_stress:
                    nominal_risk = risk_stress.get("nominal_selection", {})
                    robust_risk = risk_stress.get("risk_aware_selection", {})
                    lines.extend([
                        "", "#### 预测区间情景与下行风险审计", "",
                        f"- 情景权重：`{json.dumps(risk_stress.get('scenario_weights', {}), ensure_ascii=False)}`；"
                        "这些是对称压力权重，**不是经校准的概率分布**。",
                        f"- 风险目标：50% 压力加权期望 + 50% 的 75% 下尾 CVaR；"
                        f"改变决策单元数：{risk_stress.get('changed_decision_unit_count', 0)}；"
                        f"是否采用：{'是' if risk_stress.get('adopted') else '否'}。",
                        f"- 处置：{risk_stress.get('decision', '-')}", "",
                        "> 压力收益暂按未售出数量残值为 0、缺货不另计信誉/机会惩罚计算；"
                        "若题目给出折价处理、报废、库存结转或缺货损失，应绑定这些参数后重算。",
                        "",
                        "| 方案 | 压力加权期望收益 | 最坏情景收益 | 下尾CVaR | 风险调整收益 |",
                        "|---|---:|---:|---:|---:|",
                        f"| 名义方案 | {nominal_risk.get('stress_weighted_expected_utility', 0):.6g} | "
                        f"{nominal_risk.get('worst_case_utility', 0):.6g} | "
                        f"{nominal_risk.get('lower_tail_cvar', 0):.6g} | "
                        f"{nominal_risk.get('risk_adjusted_utility', 0):.6g} |",
                        f"| 风险感知候选 | {robust_risk.get('stress_weighted_expected_utility', 0):.6g} | "
                        f"{robust_risk.get('worst_case_utility', 0):.6g} | "
                        f"{robust_risk.get('lower_tail_cvar', 0):.6g} | "
                        f"{robust_risk.get('risk_adjusted_utility', 0):.6g} |",
                    ])
                coverage_rows = prescriptive.get("hierarchical_demand_coverage", [])
                if coverage_rows:
                    lines.extend([
                        "", "#### 上层品类需求覆盖审计", "",
                        "| 品类 | 日期 | 品类预测需求 | 入选单品需求 | 缺口 | 覆盖率 |",
                        "|---|---|---:|---:|---:|---:|",
                    ])
                    for row in coverage_rows[:200]:
                        lines.append(
                            f"| {row.get('parent_group', '-')} | {row.get('date', '-')} | "
                            f"{row.get('target_demand', 0):.6g} | "
                            f"{row.get('selected_item_demand', 0):.6g} | "
                            f"{row.get('shortage', 0):.6g} | "
                            f"{row.get('coverage_ratio', 0):.1%} |"
                        )
                cost_plus_rows = prescriptive.get(
                    "cost_plus_pricing_relationship", []
                )
                if cost_plus_rows:
                    lines.extend([
                        "", "#### 成本加成率—销量关系审计（观察性）", "",
                        f"售价来自 `{prescriptive.get('dataset', '-')}`，成本来自 "
                        f"`{prescriptive.get('cost_dataset', '-')}."
                        f"{prescriptive.get('cost_column', '-')}`。按日期×品类对齐后，"
                        "分别去除线性趋势与星期效应，再计算 Spearman 相关；"
                        "显著判据同时要求 BH-FDR q≤0.05 和前后半段同号。",
                        "",
                        "| 品类 | 对齐天数 | 中位加成率 | 残差相关 | p值 | FDR q值 | 前半/后半相关 | 判定 |",
                        "|---|---:|---:|---:|---:|---:|---:|---|",
                    ])
                    for row in cost_plus_rows:
                        first = row.get("first_half_spearman")
                        second = row.get("second_half_spearman")
                        split_text = (
                            f"{first:.4g}/{second:.4g}"
                            if first is not None and second is not None else "-"
                        )
                        lines.append(
                            f"| {row.get('group', '-')} | {row.get('n_aligned_days', 0)} | "
                            f"{row.get('median_markup_rate', 0):.2%} | "
                            f"{row.get('residual_spearman', 0):.4g} | "
                            f"{row.get('p_value', 1):.4g} | "
                            f"{row.get('q_value', 1):.4g} | {split_text} | "
                            f"{'FDR显著且方向稳定' if row.get('significant') else '未通过联合门'} |"
                        )
                    lines.extend([
                        "",
                        "> 这是历史观察性检验。品类成本采用当日单品批发价的中位数近似；"
                        "库存删失、促销、商品结构变化或未观测混杂都可能改变结论，"
                        "不能据此单独宣称调价的因果收益。",
                    ])
                lines.append("")
            data_requirements = result.specialized_results.get("data_requirements")
            if data_requirements:
                lines.extend([
                    "### 数据需求与可识别性审计", "",
                    f"- 方法：`{data_requirements.get('method', '-')}`",
                    f"- 已审计数据集/字段：{data_requirements.get('observed_dataset_count', 0)}/"
                    f"{data_requirements.get('observed_column_count', 0)}",
                    f"- 关系证据：{data_requirements.get('relationship_evidence_count', 0)} 条",
                    f"- 边界：{data_requirements.get('note', '-')}", "",
                    "| 优先级 | 应补数据角色 | 为什么需要 | 采集设计 | 支持任务 | 证据缺口 |",
                    "|---|---|---|---|---|---|",
                ])
                for item in data_requirements.get("recommendations", []):
                    lines.append(
                        f"| {item.get('priority', '-')} | {item.get('data_role', '-')} | "
                        f"{str(item.get('reason', '-')).replace('|', '\\|')} | "
                        f"{str(item.get('collection_design', '-')).replace('|', '\\|')} | "
                        f"{'、'.join(item.get('supports_tasks', [])) or '-'} | "
                        f"{str(item.get('gap_source', '-')).replace('|', '\\|')} |"
                    )
                lines.append("")
            optimization = result.specialized_results.get("optimization")
            if optimization:
                audit = optimization.get("credibility_audit", {})
                lines.extend([
                    "### 显式连续线性优化", "",
                    f"- 求解器：{optimization.get('solver', '-')}；状态：{optimization.get('message', '-')}",
                    f"- 目标：{optimization.get('direction', '-')} `{optimization.get('objective_expression', '-')}`",
                    f"- 最优目标值：{optimization.get('objective_value')}",
                    f"- 决策方案：`{json.dumps(optimization.get('solution', {}), ensure_ascii=False)}`",
                    f"- 最大约束违反：{optimization.get('maximum_constraint_violation')}",
                    f"- 最大 KKT 残差：{optimization.get('optimality_certificate', {}).get('maximum_kkt_residual')}",
                    f"- 近优变量范围：`{json.dumps(optimization.get('near_optimal_ranges', {}), ensure_ascii=False)}`",
                    f"- 5%目标系数扰动中位方案变化：{optimization.get('sensitivity', {}).get('median_relative_solution_shift')}",
                    f"- 最小最大遗憾候选：`{json.dumps(optimization.get('robust_feedback', {}).get('candidate_solution', {}), ensure_ascii=False)}`",
                    f"- 最坏遗憾改善：{optimization.get('robust_feedback', {}).get('relative_regret_reduction')}",
                    f"- 是否替换名义解：{optimization.get('robust_feedback', {}).get('accepted_as_primary', False)}（不确定集合需先由题目确认）",
                    f"- 可信度：**{audit.get('label', '-')}**；{audit.get('decision', '-')}",
                    f"- 边界：{optimization.get('note', '-')}",
                    "", "| 优化审计项 | 状态 | 证据 |", "|---|---|---|",
                ])
                for check in audit.get("checks", []):
                    lines.append(
                        f"| {check.get('name', '-')} | {check.get('status', '-')} | "
                        f"{str(check.get('evidence', '-')).replace('|', '\\|')} |"
                    )
            graph_result = result.specialized_results.get("graph_network")
            if graph_result:
                lines.extend([
                    "### 网络结构", "",
                    f"- 节点数：{graph_result['n_nodes']}",
                    f"- 唯一边数：{graph_result['n_unique_edges']}",
                    f"- 连通分量：{graph_result['connected_components']}",
                    f"- 最大连通分量：{graph_result['largest_component_size']}",
                ])
            simulation = result.specialized_results.get("simulation")
            if simulation:
                lines.extend([
                    "", "### Bootstrap 不确定性", "",
                    f"- 变量：{simulation['dataset']}.{simulation['variable']}",
                    f"- 观测均值：{simulation['observed_mean']}",
                    f"- 均值 95% 区间：{simulation['mean_confidence_interval_95']}",
                ])
            dynamics = result.specialized_results.get("time_dynamics")
            if dynamics:
                lines.extend([
                    "", "### 时序动力特征", "",
                    f"- 变量：{dynamics['dataset']}.{dynamics['variable']}",
                    f"- 时间范围：{dynamics['time_range']}",
                    f"- 日趋势：{dynamics['linear_trend_per_day']}",
                    f"- 自相关：{dynamics['autocorrelation']}",
                ])
            equation = result.specialized_results.get("equation_discovery")
            if equation:
                audit = equation.get("credibility_audit", {})
                lines.extend([
                    "", "### 积分弱形式稀疏动力方程", "",
                    f"- 候选方程：`{equation['equation']}`",
                    f"- 状态变量：{equation['state_columns']}",
                    f"- 时间点/训练窗口/验证窗口：{equation['n_time_points']}/"
                    f"{equation['training_windows']}/{equation['validation_windows']}",
                    f"- 验证指标：`{json.dumps(equation['metrics'], ensure_ascii=False)}`",
                    f"- 可信度：**{audit.get('label', '-')}**；{audit.get('decision', '-')}",
                    f"- 边界：{equation.get('note', '-')}",
                ])
            causal = result.specialized_results.get("causal_effect")
            if causal:
                audit = causal.get("credibility_audit", {})
                lines.extend([
                    "", "### 交叉拟合正交化因果效应", "",
                    f"- 处理变量：{causal['dataset']}.{causal['treatment']}",
                    f"- 结果变量：{causal['dataset']}.{causal['outcome']}",
                    f"- 效应：{causal['effect']}",
                    f"- 95% 区间：{causal['confidence_interval_95']}",
                    f"- 安慰剂 p 值：{causal['placebo_p_value']}",
                    f"- 可信度：**{audit.get('label', '-')}**；{audit.get('decision', '-')}",
                    f"- 边界：{causal.get('note', '-')}",
                    "", "| 因果审计项 | 状态 | 证据 |", "|---|---|---|",
                ])
                for check in audit.get("checks", []):
                    evidence = str(check.get("evidence", "-")).replace("|", "\\|")
                    lines.append(f"| {check.get('name', '-')} | {check.get('status', '-')} | {evidence} |")
            structures = result.specialized_results.get("data_structure", [])
            if structures:
                lines.extend(["", "### 潜在结构与稳健异常", ""])
                for structure in structures:
                    audit = structure.get("credibility_audit", {})
                    lines.extend([
                        f"#### {structure['dataset']}", "",
                        f"- 原始有效维数：{structure['original_dimensions']}；90% 解释率所需维数：{structure['dimensions_90']}",
                        f"- 累计解释率：{structure['cumulative_explained_variance']:.2%}",
                        f"- 分半子空间稳定性：{structure.get('subspace_stability')}",
                        f"- 结构异常：{structure['anomaly_count']} / {structure['analysis_rows']}（估计全量约 {structure['estimated_source_anomalies']}）",
                        f"- 异常名单扰动 Jaccard：{structure['anomaly_perturbation_jaccard']:.2%}",
                        f"- 可信度：**{audit.get('label', '-')}**；{audit.get('decision', '-')}",
                        f"- 解释边界：{structure.get('note', '-')}", "",
                    ])
            custom_results = result.specialized_results.get("custom", {})
            for task_type, payload in custom_results.items():
                lines.extend([
                    "", f"### 扩展分析器：{task_type}", "", "```json",
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    "```",
                ])
        lines.extend(["", "## 结论", ""])
        lines.extend(f"- {conclusion}" for conclusion in result.conclusions)
        if result.charts:
            lines.extend(["", "## 自动生成图表", ""])
            for chart in result.charts:
                relative_chart = Path("..") / "charts" / Path(chart["path"]).name
                lines.append(f"- [{chart['title']}]({relative_chart.as_posix()})")
        if result.warnings:
            lines.extend(["", "## 限制与待确认项", ""])
            lines.extend(f"- {warning}" for warning in result.warnings)
        lines.extend(["", "> 统计关联不自动等价于因果关系；竞赛论文中应结合机理、实验设计或稳健性检验进行论证。", ""])
        if self._artifact_manager is None:
            raise RuntimeError("运行产物管理器尚未初始化")
        return self._artifact_manager.write_text(
            "report.mathematical_argument", "reports", "mathematical_argument.md",
            "\n".join(lines), media_type="text/markdown; charset=utf-8",
            format_version="1.0", required=True,
        )


def run_modeling_study(
    problem: str,
    datasets: Optional[Mapping[str, pd.DataFrame]] = None,
    target: Optional[Union[str, Sequence[str]]] = None,
    output_dir: Optional[str] = None,
    run_modeling: bool = True,
    generate_plots: bool = True,
    mechanistic_ir: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Convenience API returning a JSON-safe study result."""
    assistant = MathModelingAssistant(output_dir=output_dir)
    return assistant.run(
        problem=problem,
        datasets=datasets,
        target=target,
        run_modeling=run_modeling,
        generate_plots=generate_plots,
        mechanistic_ir=mechanistic_ir,
    ).to_dict()


__all__ = [
    "DatasetProfile", "DatasetRelation", "InteractionFinding", "ResearchResult",
    "MathModelingAssistant", "run_modeling_study",
]
