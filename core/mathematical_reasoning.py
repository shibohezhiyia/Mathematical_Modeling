"""Typed mathematical specifications and auditable argument construction.

The modeling assistant produces many useful numerical objects.  This module is
the layer that prevents those objects from silently turning into stronger claims
than the computations support.  It deliberately contains no language-model
calls: prose generation is an optional consumer of the verified evidence bundle,
never a source of mathematical facts.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


_ROLE_TOKEN = r"[0-9A-Za-z_\-\.\u4e00-\u9fff]+"
_DATE_WORDS = ("date", "time", "day", "month", "year", "日期", "时间", "年月", "时刻")
_ID_WORDS = ("id", "key", "code", "编号", "编码", "序号", "主键", "uuid")
_POST_EVENT_WORDS = (
    "post", "after", "future", "outcome", "result", "label", "事后", "之后", "未来", "结果",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class UnitDimension:
    """Physical/economic dimension represented as powers of base dimensions."""

    powers: Tuple[Tuple[str, float], ...] = ()
    scale: float = 1.0
    symbol: str = "1"

    @classmethod
    def from_mapping(
        cls, powers: Mapping[str, float], scale: float = 1.0, symbol: str = "1"
    ) -> "UnitDimension":
        cleaned = tuple(sorted(
            (str(key), float(value)) for key, value in powers.items()
            if abs(float(value)) > 1e-12
        ))
        return cls(cleaned, float(scale), symbol)

    @property
    def mapping(self) -> Dict[str, float]:
        return dict(self.powers)

    @property
    def is_dimensionless(self) -> bool:
        return not self.powers

    def compatible(self, other: "UnitDimension") -> bool:
        return self.powers == other.powers

    def multiply(self, other: "UnitDimension") -> "UnitDimension":
        powers = self.mapping
        for key, value in other.powers:
            powers[key] = powers.get(key, 0.0) + value
        return UnitDimension.from_mapping(
            powers, self.scale * other.scale, f"({self.symbol})*({other.symbol})"
        )

    def divide(self, other: "UnitDimension") -> "UnitDimension":
        powers = self.mapping
        for key, value in other.powers:
            powers[key] = powers.get(key, 0.0) - value
        return UnitDimension.from_mapping(
            powers, self.scale / other.scale, f"({self.symbol})/({other.symbol})"
        )

    def power(self, exponent: float) -> "UnitDimension":
        return UnitDimension.from_mapping(
            {key: value * exponent for key, value in self.powers},
            self.scale ** exponent,
            f"({self.symbol})^{exponent:g}",
        )

    def to_dict(self) -> Dict[str, Any]:
        return {"powers": self.mapping, "scale": self.scale, "symbol": self.symbol}


_ATOMIC_UNITS: Dict[str, UnitDimension] = {
    "1": UnitDimension(),
    "%": UnitDimension(scale=0.01, symbol="%"),
    "percent": UnitDimension(scale=0.01, symbol="%"),
    "百分比": UnitDimension(scale=0.01, symbol="%"),
    "m": UnitDimension.from_mapping({"length": 1}, 1.0, "m"),
    "米": UnitDimension.from_mapping({"length": 1}, 1.0, "m"),
    "km": UnitDimension.from_mapping({"length": 1}, 1000.0, "km"),
    "千米": UnitDimension.from_mapping({"length": 1}, 1000.0, "km"),
    "公里": UnitDimension.from_mapping({"length": 1}, 1000.0, "km"),
    "cm": UnitDimension.from_mapping({"length": 1}, 0.01, "cm"),
    "mm": UnitDimension.from_mapping({"length": 1}, 0.001, "mm"),
    "s": UnitDimension.from_mapping({"time": 1}, 1.0, "s"),
    "sec": UnitDimension.from_mapping({"time": 1}, 1.0, "s"),
    "秒": UnitDimension.from_mapping({"time": 1}, 1.0, "s"),
    "min": UnitDimension.from_mapping({"time": 1}, 60.0, "min"),
    "分钟": UnitDimension.from_mapping({"time": 1}, 60.0, "min"),
    "h": UnitDimension.from_mapping({"time": 1}, 3600.0, "h"),
    "hr": UnitDimension.from_mapping({"time": 1}, 3600.0, "h"),
    "小时": UnitDimension.from_mapping({"time": 1}, 3600.0, "h"),
    "d": UnitDimension.from_mapping({"time": 1}, 86400.0, "d"),
    "day": UnitDimension.from_mapping({"time": 1}, 86400.0, "d"),
    "天": UnitDimension.from_mapping({"time": 1}, 86400.0, "d"),
    "kg": UnitDimension.from_mapping({"mass": 1}, 1.0, "kg"),
    "千克": UnitDimension.from_mapping({"mass": 1}, 1.0, "kg"),
    "g": UnitDimension.from_mapping({"mass": 1}, 0.001, "g"),
    "t": UnitDimension.from_mapping({"mass": 1}, 1000.0, "t"),
    "吨": UnitDimension.from_mapping({"mass": 1}, 1000.0, "t"),
    "cny": UnitDimension.from_mapping({"currency": 1}, 1.0, "CNY"),
    "rmb": UnitDimension.from_mapping({"currency": 1}, 1.0, "CNY"),
    "yuan": UnitDimension.from_mapping({"currency": 1}, 1.0, "CNY"),
    "元": UnitDimension.from_mapping({"currency": 1}, 1.0, "CNY"),
    "人": UnitDimension.from_mapping({"people": 1}, 1.0, "person"),
    "person": UnitDimension.from_mapping({"people": 1}, 1.0, "person"),
    "辆": UnitDimension.from_mapping({"vehicle": 1}, 1.0, "vehicle"),
    "vehicle": UnitDimension.from_mapping({"vehicle": 1}, 1.0, "vehicle"),
    "件": UnitDimension.from_mapping({"count": 1}, 1.0, "count"),
    "count": UnitDimension.from_mapping({"count": 1}, 1.0, "count"),
    "k": UnitDimension.from_mapping({"temperature": 1}, 1.0, "K"),
    "°c": UnitDimension.from_mapping({"temperature": 1}, 1.0, "°C"),
}


def parse_unit(unit: Optional[str]) -> Optional[UnitDimension]:
    """Parse a conservative subset of common competition-problem units.

    Unknown units return ``None`` instead of being silently treated as
    dimensionless.  Affine conversions such as Celsius to Kelvin are deliberately
    not applied; only dimensional compatibility is represented.
    """

    if unit is None:
        return None
    token = str(unit).strip().lower().replace(" ", "")
    token = token.replace("每", "/").replace("·", "*").replace("⋅", "*")
    token = token.replace("²", "^2").replace("³", "^3")
    if not token:
        return None
    if token in _ATOMIC_UNITS:
        return _ATOMIC_UNITS[token]

    parts = re.split(r"([*/])", token)
    result = UnitDimension()
    operator = "*"
    parsed_any = False
    for part in parts:
        if not part:
            continue
        if part in {"*", "/"}:
            operator = part
            continue
        match = re.fullmatch(r"(.+?)(?:\^(-?\d+(?:\.\d+)?))?", part)
        if not match:
            return None
        atomic_name = match.group(1)
        atomic = _ATOMIC_UNITS.get(atomic_name)
        if atomic is None:
            return None
        exponent = float(match.group(2) or 1.0)
        component = atomic.power(exponent)
        result = result.multiply(component) if operator == "*" else result.divide(component)
        parsed_any = True
    return UnitDimension.from_mapping(result.mapping, result.scale, token) if parsed_any else None


def extract_column_unit(column: str) -> Optional[str]:
    text = str(column).strip()
    bracketed = re.search(r"[\[（(]([^\]）)]+)[\]）)]\s*$", text)
    if bracketed and parse_unit(bracketed.group(1)) is not None:
        return bracketed.group(1).strip()
    lower = text.lower()
    # Check slash-free encodings before atomic suffixes; otherwise speed_km_h
    # would be misread as a time variable measured only in hours.
    for suffix, unit in (("_km_h", "km/h"), ("_m_s", "m/s"), ("_yuan_day", "元/天")):
        if lower.endswith(suffix):
            return unit
    candidates = sorted(_ATOMIC_UNITS, key=len, reverse=True)
    compound_candidates = ("km/h", "m/s", "元/天", "元/人", "kg/m^3")
    for token in (*compound_candidates, *candidates):
        aliases = (f"_{token}", f"-{token}", f"/{token}")
        if any(lower.endswith(alias.lower()) for alias in aliases) and parse_unit(token) is not None:
            return token
    return None


@dataclass
class ModelSymbol:
    id: str
    name: str
    dataset: Optional[str]
    column: Optional[str]
    role: str
    dtype: str
    unit: Optional[str]
    dimension: Optional[Dict[str, Any]]
    availability: str
    bounds: Optional[List[Optional[float]]] = None
    source: str = "dataset_schema"
    confidence: float = 1.0


@dataclass
class AssumptionRecord:
    id: str
    text: str
    category: str
    critical: bool
    testability: str
    status: str
    evidence: str
    falsification: str
    affected_tasks: List[str] = field(default_factory=list)


@dataclass
class CandidateModel:
    id: str
    task_type: str
    name: str
    family: str
    role: str
    requirements: List[str]
    assumptions: List[str]
    falsification_tests: List[str]
    solver: str
    readiness: str
    missing_requirements: List[str] = field(default_factory=list)
    applicability: str = "applicable"


@dataclass
class MathematicalModelSpec:
    version: str
    problem: str
    task_types: List[str]
    symbols: List[ModelSymbol]
    objectives: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    assumptions: List[AssumptionRecord]
    candidate_models: List[CandidateModel]
    role_bindings: Dict[str, str]
    unit_checks: List[Dict[str, Any]]
    contradictions: List[Dict[str, Any]]
    missing_requirements: List[str]
    readiness: str
    readiness_by_track: Dict[str, str]
    compiler_plan: List[Dict[str, Any]]
    output_policy: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


@dataclass
class EvidenceNode:
    id: str
    kind: str
    label: str
    status: str
    source_path: str
    detail: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClaimAssessment:
    id: str
    statement: str
    claim_type: str
    grade: str
    label: str
    disposition: str
    scope: str
    supports: List[str]
    challenges: List[str]
    assumptions: List[str]
    invalid_when: List[str]
    next_actions: List[str]
    numerical_certificate: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvidenceBundle:
    version: str
    overall_status: str
    overall_label: str
    claims: List[ClaimAssessment]
    evidence_nodes: List[EvidenceNode]
    edges: List[Dict[str, str]]
    assumption_ledger: List[AssumptionRecord]
    data_manifest: List[Dict[str, Any]]
    grade_counts: Dict[str, int]
    rejected_claim_ids: List[str]
    unresolved_claim_ids: List[str]
    model_tournament: List[Dict[str, Any]]
    argument_integrity: Dict[str, Any]
    writing_contract: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return _json_safe(asdict(self))


class DimensionAnalysisError(ValueError):
    pass


class _ExpressionDimensionAnalyzer(ast.NodeVisitor):
    def __init__(self, symbols: Mapping[str, UnitDimension]) -> None:
        self.symbols = symbols

    def visit_Name(self, node: ast.Name) -> UnitDimension:
        if node.id not in self.symbols:
            raise DimensionAnalysisError(f"未知符号 {node.id}")
        return self.symbols[node.id]

    def visit_Constant(self, node: ast.Constant) -> UnitDimension:
        if not isinstance(node.value, (int, float)):
            raise DimensionAnalysisError("仅支持数值常量")
        return UnitDimension()

    def visit_UnaryOp(self, node: ast.UnaryOp) -> UnitDimension:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise DimensionAnalysisError("不支持该一元运算")
        return self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> UnitDimension:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            if not left.compatible(right):
                raise DimensionAnalysisError(
                    f"加减两侧量纲不一致：{left.mapping} 与 {right.mapping}"
                )
            return left
        if isinstance(node.op, ast.Mult):
            return left.multiply(right)
        if isinstance(node.op, ast.Div):
            return left.divide(right)
        if isinstance(node.op, ast.Pow):
            if not isinstance(node.right, ast.Constant) or not isinstance(node.right.value, (int, float)):
                raise DimensionAnalysisError("量纲幂次必须是数值常量")
            return left.power(float(node.right.value))
        raise DimensionAnalysisError("不支持该二元运算")

    def visit_Call(self, node: ast.Call) -> UnitDimension:
        if not isinstance(node.func, ast.Name):
            raise DimensionAnalysisError("不支持属性函数")
        name = node.func.id.lower()
        dimensions = [self.visit(argument) for argument in node.args]
        if not dimensions:
            raise DimensionAnalysisError(f"函数 {name} 缺少参数")
        if name in {"abs", "min", "max"}:
            if any(not dimensions[0].compatible(item) for item in dimensions[1:]):
                raise DimensionAnalysisError(f"函数 {name} 的参数量纲不一致")
            return dimensions[0]
        if name == "sqrt":
            return dimensions[0].power(0.5)
        if name in {"exp", "log", "sin", "cos", "tan"}:
            if not dimensions[0].is_dimensionless:
                raise DimensionAnalysisError(f"函数 {name} 的参数必须无量纲")
            return UnitDimension()
        raise DimensionAnalysisError(f"不支持函数 {name}")

    def generic_visit(self, node: ast.AST) -> UnitDimension:
        raise DimensionAnalysisError(f"不支持表达式节点 {type(node).__name__}")


class _ExpressionStructureAnalyzer(ast.NodeVisitor):
    """Determine polynomial degree in decision variables without evaluating it."""

    def __init__(self, decision_variables: Iterable[str], known_symbols: Iterable[str]) -> None:
        self.decision_variables = set(decision_variables)
        self.known_symbols = set(known_symbols)

    def visit_Name(self, node: ast.Name) -> float:
        if node.id not in self.known_symbols:
            raise ValueError(f"未知符号 {node.id}")
        return 1.0 if node.id in self.decision_variables else 0.0

    def visit_Constant(self, node: ast.Constant) -> float:
        if not isinstance(node.value, (int, float)):
            raise ValueError("仅支持数值常量")
        return 0.0

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            return math.inf
        return self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, (ast.Add, ast.Sub)):
            return max(left, right)
        if isinstance(node.op, ast.Mult):
            return left + right
        if isinstance(node.op, ast.Div):
            return left if right == 0 else math.inf
        if isinstance(node.op, ast.Pow):
            if (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, (int, float))
                and float(node.right.value) >= 0
            ):
                return left * float(node.right.value)
            return math.inf
        return math.inf

    def visit_Call(self, node: ast.Call) -> float:
        # Even abs(x) is piecewise-linear and needs an auxiliary formulation;
        # conservatively classify all functions as nonlinear here.
        for argument in node.args:
            self.visit(argument)
        return math.inf

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"不支持表达式节点 {type(node).__name__}")


def classify_expression_structure(
    expression: str,
    decision_variables: Iterable[str],
    known_symbols: Iterable[str],
) -> Dict[str, Any]:
    """Classify an algebraic expression as constant, linear, quadratic or nonlinear."""
    try:
        tree = ast.parse(str(expression), mode="eval")
        degree = _ExpressionStructureAnalyzer(decision_variables, known_symbols).visit(tree.body)
    except (SyntaxError, ValueError) as exc:
        return {"status": "fail", "structure": "unknown", "degree": None, "evidence": str(exc)}
    if not np.isfinite(degree):
        structure = "nonlinear"
    elif degree <= 0:
        structure = "constant"
    elif degree <= 1:
        structure = "linear"
    elif degree <= 2:
        structure = "quadratic"
    else:
        structure = "polynomial_nonlinear"
    return {
        "status": "pass", "structure": structure,
        "degree": None if not np.isfinite(degree) else float(degree),
        "evidence": "仅做符号结构分析，未执行表达式。",
    }


class _LinearExpressionCompiler(ast.NodeVisitor):
    """Compile safe AST into ``coefficient_map, constant`` for linear programs."""

    def __init__(self, decisions: Iterable[str], scalar_parameters: Mapping[str, float]) -> None:
        self.decisions = set(decisions)
        self.scalar_parameters = dict(scalar_parameters)

    @staticmethod
    def _combine(
        left: Tuple[Dict[str, float], float],
        right: Tuple[Dict[str, float], float],
        sign: float = 1.0,
    ) -> Tuple[Dict[str, float], float]:
        coefficients = dict(left[0])
        for name, value in right[0].items():
            coefficients[name] = coefficients.get(name, 0.0) + sign * value
        return coefficients, left[1] + sign * right[1]

    @staticmethod
    def _scale(
        value: Tuple[Dict[str, float], float], factor: float
    ) -> Tuple[Dict[str, float], float]:
        return ({name: coefficient * factor for name, coefficient in value[0].items()}, value[1] * factor)

    def visit_Name(self, node: ast.Name) -> Tuple[Dict[str, float], float]:
        if node.id in self.decisions:
            return {node.id: 1.0}, 0.0
        if node.id in self.scalar_parameters:
            return {}, float(self.scalar_parameters[node.id])
        raise ValueError(f"参数 {node.id} 不是已声明的标量")

    def visit_Constant(self, node: ast.Constant) -> Tuple[Dict[str, float], float]:
        if not isinstance(node.value, (int, float)):
            raise ValueError("线性模型仅支持数值常量")
        return {}, float(node.value)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Tuple[Dict[str, float], float]:
        value = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return value
        if isinstance(node.op, ast.USub):
            return self._scale(value, -1.0)
        raise ValueError("不支持该一元运算")

    def visit_BinOp(self, node: ast.BinOp) -> Tuple[Dict[str, float], float]:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return self._combine(left, right)
        if isinstance(node.op, ast.Sub):
            return self._combine(left, right, -1.0)
        if isinstance(node.op, ast.Mult):
            if left[0] and right[0]:
                raise ValueError("决策变量相乘，不是线性表达式")
            return self._scale(right, left[1]) if not left[0] else self._scale(left, right[1])
        if isinstance(node.op, ast.Div):
            if right[0] or abs(right[1]) <= 1e-15:
                raise ValueError("线性表达式只能除以非零标量")
            return self._scale(left, 1.0 / right[1])
        if isinstance(node.op, ast.Pow):
            if isinstance(node.right, ast.Constant) and float(node.right.value) == 1.0:
                return left
            if isinstance(node.right, ast.Constant) and float(node.right.value) == 0.0:
                return {}, 1.0
            raise ValueError("线性模型不支持非一次幂")
        raise ValueError("不支持该运算")

    def generic_visit(self, node: ast.AST) -> Tuple[Dict[str, float], float]:
        raise ValueError(f"不支持表达式节点 {type(node).__name__}")


def compile_linear_expression(
    expression: str,
    decision_variables: Iterable[str],
    scalar_parameters: Optional[Mapping[str, float]] = None,
) -> Tuple[Dict[str, float], float]:
    """Compile but never evaluate arbitrary code from an algebraic expression."""
    tree = ast.parse(str(expression), mode="eval")
    return _LinearExpressionCompiler(decision_variables, scalar_parameters or {}).visit(tree.body)


def check_expression_dimensions(
    expression: str,
    symbol_units: Mapping[str, Union[str, UnitDimension]],
) -> Dict[str, Any]:
    """Check dimensional consistency without executing the expression."""

    try:
        tree = ast.parse(str(expression), mode="eval")
    except SyntaxError as exc:
        return {
            "status": "fail", "expression": expression, "evidence": str(exc),
            "result_dimension": None,
        }
    function_names = {
        node.func.id for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    referenced_names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    } - function_names
    parsed_units: Dict[str, UnitDimension] = {}
    unknown_units: List[str] = []
    for name, unit in symbol_units.items():
        if str(name) not in referenced_names:
            continue
        parsed = unit if isinstance(unit, UnitDimension) else parse_unit(unit)
        if parsed is None:
            unknown_units.append(str(name))
        else:
            parsed_units[str(name)] = parsed
    if unknown_units:
        return {
            "status": "not_assessed",
            "expression": expression,
            "evidence": f"以下符号缺少可解析单位：{', '.join(unknown_units)}",
            "result_dimension": None,
        }
    try:
        dimension = _ExpressionDimensionAnalyzer(parsed_units).visit(tree.body)
        return {
            "status": "pass",
            "expression": expression,
            "evidence": "表达式的加减、乘除和函数参数通过静态量纲检查。",
            "result_dimension": dimension.to_dict(),
        }
    except (SyntaxError, DimensionAnalysisError) as exc:
        return {
            "status": "fail",
            "expression": expression,
            "evidence": str(exc),
            "result_dimension": None,
        }


def check_equation_dimensions(
    left_expression: str,
    right_expression: str,
    symbol_units: Mapping[str, Union[str, UnitDimension]],
) -> Dict[str, Any]:
    left = check_expression_dimensions(left_expression, symbol_units)
    right = check_expression_dimensions(right_expression, symbol_units)
    if left["status"] != "pass" or right["status"] != "pass":
        status = "fail" if "fail" in {left["status"], right["status"]} else "not_assessed"
        return {
            "status": status,
            "expression": f"{left_expression} = {right_expression}",
            "evidence": f"左侧：{left['evidence']}；右侧：{right['evidence']}",
            "left_dimension": left.get("result_dimension"),
            "right_dimension": right.get("result_dimension"),
        }
    left_dimension = UnitDimension.from_mapping(
        left["result_dimension"]["powers"], left["result_dimension"]["scale"], "left"
    )
    right_dimension = UnitDimension.from_mapping(
        right["result_dimension"]["powers"], right["result_dimension"]["scale"], "right"
    )
    compatible = left_dimension.compatible(right_dimension)
    return {
        "status": "pass" if compatible else "fail",
        "expression": f"{left_expression} = {right_expression}",
        "evidence": (
            "等式两侧量纲一致；单位尺度可在数值计算前统一换算。"
            if compatible else
            f"等式两侧量纲不一致：{left_dimension.mapping} 与 {right_dimension.mapping}"
        ),
        "left_dimension": left["result_dimension"],
        "right_dimension": right["result_dimension"],
    }


_CANDIDATE_LIBRARY: Dict[str, List[Dict[str, Any]]] = {
    "prediction_forecast": [
        {"name": "朴素/季节朴素基线", "family": "baseline", "role": "null_model",
         "requirements": ["目标变量"], "solver": "deterministic_baseline"},
        {"name": "正则化可解释模型", "family": "statistical", "role": "competitor",
         "requirements": ["目标变量", "可用预测变量"], "solver": "cross_validated_estimator"},
        {"name": "非线性树集成", "family": "machine_learning", "role": "competitor",
         "requirements": ["目标变量", "足够样本"], "solver": "cross_validated_estimator"},
    ],
    "classification": [
        {"name": "多数类/先验概率基线", "family": "baseline", "role": "null_model",
         "requirements": ["离散目标变量"], "solver": "deterministic_baseline"},
        {"name": "校准分类器", "family": "statistical", "role": "competitor",
         "requirements": ["离散目标变量", "每类足够样本"], "solver": "stratified_cross_validation"},
    ],
    "statistical_inference": [
        {"name": "零效应模型", "family": "null_hypothesis", "role": "null_model",
         "requirements": ["至少两个变量"], "solver": "permutation_and_bootstrap"},
        {"name": "条件关联模型", "family": "statistical", "role": "competitor",
         "requirements": ["至少两个变量"], "solver": "robust_association"},
    ],
    "causal_inference": [
        {"name": "关联而非因果零解释", "family": "null_hypothesis", "role": "null_model",
         "requirements": ["处理变量", "结果变量"], "solver": "placebo_test"},
        {"name": "交叉拟合正交效应", "family": "causal", "role": "competitor",
         "requirements": ["显式处理变量", "显式结果变量", "处理前混杂变量"],
         "solver": "cross_fitted_dml"},
    ],
    "evaluation_ranking": [
        {"name": "Pareto 非支配集", "family": "multiobjective", "role": "weight_free_reference",
         "requirements": ["至少两个指标", "指标方向"], "solver": "pareto_dominance"},
        {"name": "熵权 TOPSIS", "family": "scalarization", "role": "competitor",
         "requirements": ["至少两个指标", "指标方向"], "solver": "closed_form"},
    ],
    "optimization": [
        {"name": "确定性优化", "family": "operations_research", "role": "nominal_model",
         "requirements": ["决策变量", "目标函数", "可执行约束"], "solver": "HiGHS_or_SciPy"},
        {"name": "情景/稳健优化", "family": "robust_optimization", "role": "stress_competitor",
         "requirements": ["决策变量", "目标函数", "可执行约束", "不确定参数集合"],
         "solver": "scenario_or_DRO"},
    ],
    "differential_equations": [
        {"name": "常值/随机游走零动力学", "family": "baseline", "role": "null_model",
         "requirements": ["时间变量", "状态变量"], "solver": "deterministic_baseline"},
        {"name": "积分弱形式稀疏动力学", "family": "mechanistic_candidate", "role": "competitor",
         "requirements": ["有序时间变量", "连续状态变量", "至少50个时间点"],
         "solver": "integral_sparse_regression"},
    ],
    "clustering": [
        {"name": "单群体零结构", "family": "null_hypothesis", "role": "null_model",
         "requirements": ["至少两个指标"], "solver": "dispersion_reference"},
        {"name": "多初始化聚类", "family": "unsupervised", "role": "competitor",
         "requirements": ["至少两个指标", "足够样本"], "solver": "multi_seed_clustering"},
    ],
    "dimension_reduction": [
        {"name": "稳健主成分结构", "family": "latent_structure", "role": "competitor",
         "requirements": ["至少两个数值变量"], "solver": "robust_scaled_PCA"},
    ],
    "anomaly_detection": [
        {"name": "重构误差与杠杆联合异常", "family": "structural_anomaly", "role": "competitor",
         "requirements": ["至少两个数值变量", "足够样本"], "solver": "robust_PCA_audit"},
    ],
    "graph_network": [
        {"name": "实体关系网络", "family": "graph", "role": "competitor",
         "requirements": ["起点列", "终点列"], "solver": "bounded_graph_algorithms"},
    ],
    "simulation": [
        {"name": "非参数 Bootstrap", "family": "uncertainty", "role": "competitor",
         "requirements": ["可重采样观测"], "solver": "bounded_bootstrap"},
    ],
}


_FALSIFICATION_BY_TASK: Dict[str, List[str]] = {
    "prediction_forecast": ["严格留出/时间外推", "朴素基线", "置乱检验", "漂移与扰动", "子群误差"],
    "classification": ["分层或分组留出", "多数类基线", "概率校准", "置乱检验", "子群误差"],
    "statistical_inference": ["多重检验校正", "条件关联", "分块稳定性", "非线性替代解释"],
    "causal_inference": ["重叠性", "安慰剂", "处理前时序", "折间稳定性", "未观测混杂敏感性"],
    "evaluation_ranking": ["权重扰动", "删指标", "Pareto 前沿", "指标方向复核"],
    "optimization": ["约束残差", "最优性缺口", "极端情景", "参数敏感性", "替代最优解"],
    "differential_equations": ["时间外验证", "零变化基线", "支持集稳定性", "残差自相关", "初值扰动"],
    "clustering": ["多初始化一致性", "簇分离", "簇规模平衡", "删变量稳定性"],
    "dimension_reduction": ["分半子空间稳定性", "重采样载荷稳定性", "异常点影响"],
    "anomaly_detection": ["阈值敏感性", "扰动名单一致性", "人工注入异常检出"],
    "graph_network": ["边定义复核", "删边稳定性", "方向性与连通性检查"],
    "simulation": ["重采样假设", "蒙特卡洛误差", "尾部稳定性", "情景覆盖"],
}


class MathematicalReasoningEngine:
    """Build specifications first, then compile numerical outputs into claims."""

    version = "1.0"

    @staticmethod
    def _extract_role(problem: str, role_names: str) -> Optional[str]:
        forward = re.search(
            rf"(?:{role_names})(?:变量)?\s*(?:为|是|=|:|：)\s*[‘’'\"]?({_ROLE_TOKEN})",
            str(problem), re.IGNORECASE,
        )
        if forward:
            return forward.group(1).strip("。；;，,、'\"‘’")
        reverse = re.search(
            rf"以\s*[‘’'\"]?({_ROLE_TOKEN})[‘’'\"]?\s*(?:为|作为)\s*(?:{role_names})(?:变量)?",
            str(problem), re.IGNORECASE,
        )
        return reverse.group(1).strip("。；;，,、'\"‘’") if reverse else None

    @staticmethod
    def _resolve_column(
        reference: Optional[str], datasets: Mapping[str, pd.DataFrame]
    ) -> Optional[Tuple[str, str]]:
        if not reference:
            return None
        text = str(reference).strip()
        if "." in text:
            dataset, column = text.rsplit(".", 1)
            if dataset in datasets and column in datasets[dataset].columns:
                return dataset, column
        matches = [
            (name, text) for name, frame in datasets.items() if text in frame.columns
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _extract_declared_names(problem: str, role_names: str) -> List[str]:
        match = re.search(
            rf"(?:{role_names})(?:变量)?\s*(?:为|是|=|:|：)\s*([^；;。\n]+)",
            str(problem), re.IGNORECASE,
        )
        if not match:
            return []
        content = match.group(1)
        # Stop before a new semantic declaration in the same clause.
        content = re.split(
            r"(?:，|,)?\s*(?:目标函数|约束|状态变量|参数|处理变量|结果变量)(?:为|是|=|:|：)",
            content, maxsplit=1,
        )[0]
        names: List[str] = []
        for item in re.split(r"[,，、\s]+", content):
            token = item.strip("()（）[]{}'\"‘’")
            if re.fullmatch(r"[A-Za-z_\u4e00-\u9fff][0-9A-Za-z_\u4e00-\u9fff]*", token):
                names.append(token)
        return list(dict.fromkeys(names))[:50]

    @staticmethod
    def _decision_bounds(problem: str, name: str) -> Optional[List[Optional[float]]]:
        escaped = re.escape(name)
        number = r"[-+]?\d+(?:\.\d+)?"
        standalone = rf"(?<![0-9A-Za-z_+\-*/]){escaped}(?![0-9A-Za-z_+\-*/])"
        double = re.search(
            rf"({number})\s*(?:<=|≤)\s*{standalone}\s*(?:<=|≤)\s*({number})",
            str(problem), re.IGNORECASE,
        )
        if double:
            return [float(double.group(1)), float(double.group(2))]
        lower = re.search(rf"{standalone}\s*(?:>=|≥)\s*({number})", str(problem), re.IGNORECASE)
        upper = re.search(rf"{standalone}\s*(?:<=|≤)\s*({number})", str(problem), re.IGNORECASE)
        if lower or upper:
            return [float(lower.group(1)) if lower else None, float(upper.group(1)) if upper else None]
        return None

    @staticmethod
    def _extract_algebraic_statements(problem: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        objectives: List[Dict[str, Any]] = []
        constraints: List[Dict[str, Any]] = []
        for clause in re.split(r"[；;。\n]", str(problem)):
            text = clause.strip()
            if not text:
                continue
            objective = re.search(
                r"(?:目标函数\s*(?:为|是|=|:|：)\s*)?"
                r"(最小化|最大化|minimize|maximize|min|max)\s*[:：]?\s*(.+)$",
                text, re.IGNORECASE,
            )
            if objective:
                direction_token = objective.group(1).lower()
                objectives.append({
                    "text": text,
                    "direction": "minimize" if direction_token in {"最小化", "minimize", "min"} else "maximize",
                    "expression": objective.group(2).strip(),
                    "executable": False,
                    "source": "explicit_problem_expression",
                })
            comparator = re.search(r"<=|>=|≤|≥|(?<![<>])=(?!=)", text)
            if comparator and not any(
                marker in text.lower()
                for marker in ("处理变量", "结果变量", "目标变量", "决策变量", "treatment", "outcome")
            ):
                left = text[:comparator.start()].strip()
                right = text[comparator.end():].strip()
                if "约束" in left:
                    left = re.split(r"[:：]", left)[-1]
                    left = re.sub(r"^约束(?:条件)?(?:为|是)?", "", left).strip()
                constraints.append({
                    "text": text, "left_expression": left,
                    "operator": comparator.group(0), "right_expression": right,
                    "executable": False, "source": "explicit_problem_expression",
                })
        return objectives, constraints

    @staticmethod
    def _dataset_manifest(datasets: Mapping[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        manifest: List[Dict[str, Any]] = []
        for name, frame in datasets.items():
            source_rows = int(frame.attrs.get("source_rows", len(frame)))
            schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
            positions = sorted(set(
                list(range(min(64, len(frame))))
                + list(range(max(0, len(frame) - 64), len(frame)))
            ))
            sample = frame.iloc[positions] if positions else frame.head(0)
            try:
                row_hash = pd.util.hash_pandas_object(sample, index=True).values.tobytes()
            except (TypeError, ValueError):
                normalized = sample.astype(str)
                row_hash = pd.util.hash_pandas_object(normalized, index=True).values.tobytes()
            digest = hashlib.sha256(
                json.dumps(schema, ensure_ascii=False).encode("utf-8")
                + str(len(frame)).encode("ascii") + str(source_rows).encode("ascii") + row_hash
            ).hexdigest()
            manifest.append({
                "dataset": str(name), "analysis_rows": int(len(frame)),
                "source_rows": source_rows, "columns": int(len(frame.columns)),
                "schema": schema, "fingerprint": digest,
                "fingerprint_scope": "schema + row_count + first/last 64 rows",
            })
        return manifest

    @staticmethod
    def _task_types(problem_analysis: Mapping[str, Any]) -> List[str]:
        tasks = [
            str(item.get("task_type"))
            for item in problem_analysis.get("task_candidates", [])
            if item.get("task_type")
        ]
        primary = problem_analysis.get("task_type")
        if primary:
            tasks.insert(0, str(primary))
        for item in problem_analysis.get("task_graph", []):
            if item.get("task_type"):
                tasks.append(str(item["task_type"]))
        return list(dict.fromkeys(tasks))

    def build_spec(
        self,
        problem: str,
        datasets: Mapping[str, pd.DataFrame],
        problem_analysis: Mapping[str, Any],
        targets: Optional[Union[str, Sequence[str]]] = None,
        mechanistic_result: Optional[Mapping[str, Any]] = None,
    ) -> MathematicalModelSpec:
        if isinstance(targets, str):
            target_references = [item.strip() for item in re.split(r"[,，;；]", targets) if item.strip()]
        else:
            target_references = [str(item).strip() for item in (targets or []) if str(item).strip()]
        target_bindings = [self._resolve_column(item, datasets) for item in target_references]
        target_bindings = [item for item in target_bindings if item is not None]
        treatment_ref = self._extract_role(problem, "处理|干预|政策|暴露|措施|treatment")
        outcome_ref = self._extract_role(problem, "结果|结局|因变量|outcome")
        treatment = self._resolve_column(treatment_ref, datasets)
        outcome = self._resolve_column(outcome_ref, datasets)
        pre_treatment_declared = any(
            token in str(problem).lower()
            for token in ("处理前", "干预前", "政策前", "事前", "基线协变量", "pre-treatment")
        )
        if outcome is None and target_bindings:
            outcome = target_bindings[0]

        declared_decisions = self._extract_declared_names(problem, "决策|decision")
        declared_integers = self._extract_declared_names(
            problem, "整数决策|整数|integer decision|integer"
        )
        declared_binaries = self._extract_declared_names(
            problem, "0-1决策|0-1|二元决策|二元|binary decision|binary"
        )
        declared_decisions = list(dict.fromkeys(
            declared_decisions + declared_integers + declared_binaries
        ))
        decision_domain = {
            name: ("binary" if name in declared_binaries else (
                "integer" if name in declared_integers else "continuous"
            )) for name in declared_decisions
        }
        declared_states = self._extract_declared_names(problem, "状态|state")
        declared_parameters = self._extract_declared_names(problem, "参数|parameter")
        declared_role_by_name: Dict[str, str] = {}
        for declared_name in declared_decisions:
            declared_role_by_name[declared_name] = "decision"
        for declared_name in declared_states:
            declared_role_by_name.setdefault(declared_name, "state")
        for declared_name in declared_parameters:
            declared_role_by_name.setdefault(declared_name, "parameter")

        symbols: List[ModelSymbol] = []
        role_bindings: Dict[str, str] = {}
        for dataset_name, frame in datasets.items():
            priority_bounds = {
                column for bound_dataset, column in (
                    [item for item in (treatment, outcome) if item is not None]
                    + list(target_bindings)
                ) if bound_dataset == dataset_name
            } | set(declared_role_by_name)
            numeric_for_bounds = [
                str(column) for column in frame.columns
                if pd.api.types.is_numeric_dtype(frame[column])
            ]
            bounded_columns = priority_bounds | set(numeric_for_bounds[:256])
            for column in frame.columns:
                name = str(column)
                lower = name.lower()
                binding = (dataset_name, name)
                role = "observation"
                if name in declared_role_by_name:
                    role = declared_role_by_name[name]
                    role_bindings.setdefault(role, f"{dataset_name}.{name}")
                elif binding == treatment:
                    role = "treatment"
                    role_bindings["treatment"] = f"{dataset_name}.{name}"
                elif binding == outcome:
                    role = "outcome"
                    role_bindings["outcome"] = f"{dataset_name}.{name}"
                elif binding in target_bindings:
                    role = "target"
                elif any(word in lower for word in _DATE_WORDS) or pd.api.types.is_datetime64_any_dtype(frame[column]):
                    role = "time"
                elif any(word == lower or lower.endswith(f"_{word}") for word in _ID_WORDS):
                    role = "identifier"
                availability = "at_decision"
                if role in {"outcome", "target"} or any(word in lower for word in _POST_EVENT_WORDS):
                    availability = "post_outcome"
                elif role == "time":
                    availability = "index"
                elif (
                    pre_treatment_declared and treatment and outcome
                    and dataset_name == treatment[0] == outcome[0]
                    and binding not in {treatment, outcome}
                ):
                    role = "control_candidate"
                    availability = "pre_treatment_declared"
                unit = extract_column_unit(name)
                dimension = parse_unit(unit)
                numeric = (
                    pd.to_numeric(frame[column], errors="coerce")
                    if name in bounded_columns and pd.api.types.is_numeric_dtype(frame[column])
                    else None
                )
                bounds = None
                if numeric is not None and numeric.notna().any():
                    bounds = [float(numeric.min()), float(numeric.max())]
                dtype = decision_domain.get(name, str(frame[column].dtype)) if role == "decision" else str(frame[column].dtype)
                if role == "decision" and decision_domain.get(name) == "binary":
                    bounds = [0.0, 1.0]
                symbols.append(ModelSymbol(
                    id=_stable_id("sym", dataset_name, name), name=name, dataset=str(dataset_name),
                    column=name, role=role, dtype=dtype, unit=unit,
                    dimension=dimension.to_dict() if dimension else None,
                    availability=availability, bounds=bounds,
                ))

        existing_names = {symbol.name for symbol in symbols}
        for name, role in declared_role_by_name.items():
            if name in existing_names:
                continue
            symbol = ModelSymbol(
                id=_stable_id("sym", "abstract", name), name=name, dataset=None,
                column=None, role=role, dtype=decision_domain.get(name, "symbolic"), unit=None, dimension=None,
                availability="decision_time" if role == "decision" else "static",
                bounds=([0.0, 1.0] if decision_domain.get(name) == "binary" else
                        (self._decision_bounds(problem, name) if role == "decision" else None)),
                source="explicit_problem_declaration", confidence=1.0,
            )
            symbols.append(symbol)
            role_bindings.setdefault(role, name)

        tasks = self._task_types(problem_analysis)
        if mechanistic_result is not None and not datasets:
            mechanistic_tasks = [
                str(item.get("task_type"))
                for item in problem_analysis.get("task_graph", [])
                if item.get("task_type")
            ]
            primary_task = problem_analysis.get("task_type")
            tasks = list(dict.fromkeys(
                ([str(primary_task)] if primary_task else []) + mechanistic_tasks
            ))
        assumptions = self._build_assumptions(tasks, role_bindings, symbols, problem)
        contradictions: List[Dict[str, Any]] = []
        if treatment and outcome and treatment == outcome:
            contradictions.append({
                "id": "causal_role_collision", "severity": "error",
                "message": "处理变量与结果变量不能是同一字段。",
            })
        duplicate_roles = [
            role for role in ("treatment", "outcome")
            if list(role_bindings).count(role) > 1
        ]
        if duplicate_roles:
            contradictions.append({
                "id": "ambiguous_roles", "severity": "error",
                "message": f"角色绑定不唯一：{duplicate_roles}",
            })
        for symbol in symbols:
            if (
                symbol.role == "decision" and symbol.bounds is not None
                and symbol.bounds[0] is not None and symbol.bounds[1] is not None
                and symbol.bounds[0] > symbol.bounds[1]
            ):
                contradictions.append({
                    "id": _stable_id("bound", symbol.name), "severity": "error",
                    "message": (
                        f"决策变量 {symbol.name} 的下界 {symbol.bounds[0]} 大于上界 {symbol.bounds[1]}。"
                    ),
                })

        unit_checks = self._extract_and_check_equations(problem, symbols)
        for check in unit_checks:
            if check["status"] == "fail":
                contradictions.append({
                    "id": _stable_id("unit", check["expression"]), "severity": "error",
                    "message": check["evidence"], "expression": check["expression"],
                })

        explicit_objectives, explicit_constraints = self._extract_algebraic_statements(problem)
        known_symbol_names = {symbol.name for symbol in symbols}
        decision_names = {symbol.name for symbol in symbols if symbol.role == "decision"}
        for objective in explicit_objectives:
            structure = classify_expression_structure(
                objective["expression"], decision_names, known_symbol_names
            )
            objective["structure"] = structure
            objective["executable"] = structure["status"] == "pass"
        for constraint in explicit_constraints:
            left = classify_expression_structure(
                constraint["left_expression"], decision_names, known_symbol_names
            )
            right = classify_expression_structure(
                constraint["right_expression"], decision_names, known_symbol_names
            )
            constraint["structure"] = {"left": left, "right": right}
            constraint["executable"] = left["status"] == right["status"] == "pass"

        readiness_context = self._readiness_context(datasets, role_bindings, target_bindings, problem_analysis)
        readiness_context.update({
            "has_decision_variable": bool(decision_names),
            "has_objective": any(item["executable"] for item in explicit_objectives),
            "has_constraints": bool(explicit_constraints) and all(
                item["executable"] for item in explicit_constraints
            ),
        })
        readiness_context["has_pre_treatment"] = pre_treatment_declared and any(
            symbol.role == "control_candidate" for symbol in symbols
        )
        candidates: List[CandidateModel] = []
        missing_requirements: List[str] = []
        if mechanistic_result is not None and not datasets:
            mechanism_missing = list(mechanistic_result.get("missing_requirements", []))
            draft = mechanistic_result.get("model_draft", {})
            candidates.append(CandidateModel(
                id="candidate_mechanistic_composition_1",
                task_type="mechanistic_system",
                name="题面机理算子组合",
                family="mechanistic_compilation",
                role="primary_model_draft",
                requirements=["实体与参数", "通用数学算子", "规范方程草案"],
                assumptions=[],
                falsification_tests=[
                    "量纲一致性", "极限情形", "竞争语义", "步长或网格收敛", "参数敏感性"
                ],
                solver="verified_operator_dispatch",
                readiness=(
                    "ready" if mechanistic_result.get("execution_status") in {
                        "executed", "partially_executed"
                    }
                    else ("model_draft_ready" if draft.get("equations") else "needs_input")
                ),
                missing_requirements=mechanism_missing,
            ))
        for task in tasks:
            for index, template in enumerate(_CANDIDATE_LIBRARY.get(task, []), 1):
                observational_only = bool(
                    mechanistic_result is not None and not datasets and task != "optimization"
                )
                statement_model_draft = bool(
                    mechanistic_result is not None and not datasets and task == "optimization"
                )
                if observational_only:
                    missing = []
                elif statement_model_draft:
                    mechanism_ir = mechanistic_result.get("mathematical_ir", {})
                    missing = []
                    if not mechanism_ir.get("decision_statements"):
                        missing.append("决策语义")
                    else:
                        missing.append("决策变量的符号化、类型与边界")
                    if not mechanism_ir.get("objectives"):
                        missing.append("目标语义")
                    else:
                        missing.append("目标函数的可计算表达式")
                    if not mechanism_ir.get("constraints"):
                        missing.append("约束语义")
                    else:
                        missing.append("约束的代数表达式与单位核验")
                    if "不确定参数集合" in template["requirements"]:
                        missing.append("不确定参数集合")
                else:
                    missing = self._candidate_missing(
                        task, template["requirements"], readiness_context
                    )
                candidates.append(CandidateModel(
                    id=f"candidate_{task}_{index}", task_type=task, name=template["name"],
                    family=template["family"], role=template["role"],
                    requirements=list(template["requirements"]),
                    assumptions=[item.id for item in assumptions if task in item.affected_tasks],
                    falsification_tests=list(_FALSIFICATION_BY_TASK.get(task, [])),
                    solver=template["solver"],
                    readiness=(
                        "not_applicable" if observational_only else
                        ("model_draft_ready" if statement_model_draft else
                        ("ready" if not missing else "needs_input")
                        )
                    ),
                    missing_requirements=missing,
                    applicability=(
                        "not_applicable_without_observations"
                        if observational_only else "applicable"
                    ),
                ))
                if not observational_only:
                    missing_requirements.extend(missing)
        missing_requirements = list(dict.fromkeys(missing_requirements))
        if contradictions:
            readiness = "invalid"
        elif candidates and all(item.readiness == "needs_input" for item in candidates):
            readiness = "needs_input"
        elif any(item.readiness == "needs_input" for item in candidates):
            readiness = "partial"
        else:
            readiness = "ready"

        readiness_by_track = {
            "observational_modeling": readiness if datasets else "not_applicable",
            "mechanistic_structure": "not_assessed",
            "numerical_execution": "not_assessed",
        }
        if mechanistic_result is not None and not datasets:
            mechanism_draft = mechanistic_result.get("model_draft", {})
            has_draft = bool(
                mechanism_draft.get("equations")
                or mechanistic_result.get("operator_graph")
            )
            execution_status = str(mechanistic_result.get("execution_status", "not_assessed"))
            readiness_by_track.update({
                "mechanistic_structure": "ready" if has_draft else "needs_input",
                "numerical_execution": (
                    "executed" if execution_status == "executed" else
                    ("partial" if execution_status == "partially_executed" else
                    ("solver_ready" if execution_status == "solver_ready" else "needs_confirmation")
                    )
                ),
            })
            readiness = (
                "ready" if execution_status == "executed" else
                ("partial" if execution_status == "partially_executed" else
                ("model_draft_ready" if has_draft else "needs_input")
                )
            )
            missing_requirements = list(mechanistic_result.get("missing_requirements", []))
            math_ir = mechanistic_result.get("mathematical_ir", {})
            entity_labels = [
                str(item.get("label")) for item in math_ir.get("entities", [])
                if item.get("label")
            ]
            if entity_labels:
                role_bindings["spatial_entities"] = "、".join(entity_labels[:20])
            quantity_count = len(math_ir.get("quantities", []))
            if quantity_count:
                role_bindings["stated_parameters"] = f"{quantity_count} 个题面数值参数"

        objectives = explicit_objectives or [
            {"text": str(item), "executable": False, "source": "problem_parser_suggestion"}
            for item in problem_analysis.get("objectives", [])
        ]
        constraints = explicit_constraints or [
            {"text": str(item), "executable": False, "source": "problem_parser_suggestion"}
            for item in problem_analysis.get("constraints", [])
        ]
        compiler_plan = self._build_compiler_plan(
            tasks, objectives, constraints, symbols, contradictions
        )
        return MathematicalModelSpec(
            version=self.version, problem=str(problem).strip(), task_types=tasks, symbols=symbols,
            objectives=objectives, constraints=constraints, assumptions=assumptions,
            candidate_models=candidates, role_bindings=role_bindings, unit_checks=unit_checks,
            contradictions=contradictions, missing_requirements=missing_requirements,
            readiness=readiness, readiness_by_track=readiness_by_track,
            compiler_plan=compiler_plan,
            output_policy={
                "primary_artifact": "machine_checkable_evidence_bundle",
                "narrative_generation": "disabled_by_default",
                "api_writer_stage": "explicit_opt_in_after_validation",
                "writer_may_compute": False,
                "writer_may_invent_claims": False,
            },
        )

    @staticmethod
    def _build_compiler_plan(
        tasks: Sequence[str],
        objectives: Sequence[Mapping[str, Any]],
        constraints: Sequence[Mapping[str, Any]],
        symbols: Sequence[ModelSymbol],
        contradictions: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        plans: List[Dict[str, Any]] = []
        decision_symbols = [symbol for symbol in symbols if symbol.role == "decision"]
        method_by_task = {
            "prediction_forecast": "cross_validated_estimator",
            "classification": "stratified_or_group_cross_validation",
            "statistical_inference": "robust_tests_with_FDR",
            "causal_inference": "cross_fitted_orthogonal_estimator",
            "evaluation_ranking": "pareto_then_preference_scalarization",
            "differential_equations": "integral_weak_form_candidate_discovery",
            "clustering": "multi_seed_stability_selection",
            "dimension_reduction": "robust_scaled_PCA",
            "anomaly_detection": "reconstruction_and_leverage_audit",
            "graph_network": "bounded_graph_algorithms",
            "simulation": "bounded_resampling",
        }
        for task in tasks:
            if task != "optimization":
                plans.append({
                    "task_type": task,
                    "status": "delegated_to_verified_analyzer",
                    "formulation_class": task,
                    "solver": method_by_task.get(task, "registered_domain_analyzer"),
                    "executable": False,
                    "reason": "由对应分析器执行后再生成证据证书；本层不执行数值表达式。",
                    "safety": "语言模型不得执行或补写数值计算。",
                })
                continue

            executable_objectives = [item for item in objectives if item.get("executable")]
            executable_constraints = [item for item in constraints if item.get("executable")]
            missing: List[str] = []
            if not decision_symbols:
                missing.append("显式决策变量")
            if not executable_objectives:
                missing.append("可解析目标函数")
            if not executable_constraints:
                missing.append("至少一条可解析约束")
            degrees: List[float] = []
            for objective in executable_objectives:
                degree = objective.get("structure", {}).get("degree")
                if degree is not None:
                    degrees.append(float(degree))
                elif objective.get("structure", {}).get("structure") == "nonlinear":
                    degrees.append(math.inf)
            for constraint in executable_constraints:
                for side in ("left", "right"):
                    structure = constraint.get("structure", {}).get(side, {})
                    degree = structure.get("degree")
                    if degree is not None:
                        degrees.append(float(degree))
                    elif structure.get("structure") == "nonlinear":
                        degrees.append(math.inf)
            maximum_degree = max(degrees, default=None)
            if maximum_degree is None:
                formulation = "unknown"
                solver = "none"
            elif maximum_degree <= 1:
                formulation = "linear_program"
                solver = "scipy.optimize.linprog(method='highs')"
            elif maximum_degree <= 2:
                formulation = "quadratic_or_quadratically_constrained"
                solver = "convexity_check_then_scipy.optimize"
            else:
                formulation = "nonlinear_program"
                solver = "scipy.optimize.minimize_with_multistart"
            if contradictions:
                missing.append("消除数学规范冲突")
            discrete = [
                symbol.name for symbol in decision_symbols
                if symbol.dtype in {"integer", "binary"}
            ]
            if discrete and formulation == "linear_program":
                formulation = "mixed_integer_linear_program"
                solver = "scipy.optimize.milp(HiGHS)"
                missing.append("整数规划尚未进入自动执行白名单，需注册并审计领域求解器")
            elif formulation != "linear_program" and not missing:
                missing.append("当前自动执行仅支持显式连续线性规划")
            symbolically_compilable = bool(
                decision_symbols and executable_objectives and executable_constraints
            )
            executable = symbolically_compilable and not missing and formulation == "linear_program"
            plans.append({
                "task_type": task,
                "status": "ready_to_compile" if executable else "needs_input",
                "formulation_class": formulation,
                "solver": solver,
                "executable": executable,
                "symbolically_compilable": symbolically_compilable,
                "discrete_variables": discrete,
                "decision_variables": [
                    {"name": symbol.name, "bounds": symbol.bounds, "dtype": symbol.dtype}
                    for symbol in decision_symbols
                ],
                "objective_count": len(executable_objectives),
                "constraint_count": len(executable_constraints),
                "missing_requirements": list(dict.fromkeys(missing)),
                "safety": (
                    "只完成符号编译准备；实际求解仍必须检查可行性、约束残差、"
                    "最优性缺口、凸性和极端情景。"
                ),
            })
        return plans

    def solve_explicit_optimization(
        self,
        spec: MathematicalModelSpec,
        datasets: Mapping[str, pd.DataFrame],
        random_state: int = 42,
    ) -> Optional[Dict[str, Any]]:
        """Solve a fully explicit continuous LP and return auditable certificates.

        This intentionally refuses integer, quadratic, nonlinear, ambiguous and
        prose-only formulations.  Refusal is safer than silently changing the
        mathematical problem.
        """
        plan = next(
            (item for item in spec.compiler_plan if item.get("task_type") == "optimization"),
            None,
        )
        if not plan or not plan.get("executable") or plan.get("formulation_class") != "linear_program":
            return None
        from scipy.optimize import linprog

        decisions = [symbol for symbol in spec.symbols if symbol.role == "decision"]
        names = [symbol.name for symbol in decisions]
        scalar_parameters: Dict[str, float] = {}
        for symbol in spec.symbols:
            if symbol.name in names or not symbol.dataset or not symbol.column:
                continue
            frame = datasets.get(symbol.dataset)
            if frame is None or symbol.column not in frame.columns:
                continue
            values = pd.to_numeric(frame[symbol.column], errors="coerce").dropna().unique()
            if len(values) == 1 and np.isfinite(float(values[0])):
                scalar_parameters[symbol.name] = float(values[0])

        objective = next(item for item in spec.objectives if item.get("executable"))
        try:
            objective_map, objective_constant = compile_linear_expression(
                str(objective["expression"]), names, scalar_parameters
            )
            original_coefficients = np.asarray(
                [objective_map.get(name, 0.0) for name in names], dtype=float
            )
            direction = str(objective.get("direction", "minimize"))
            solver_coefficients = (
                -original_coefficients if direction == "maximize" else original_coefficients
            )
            inequalities: List[np.ndarray] = []
            inequality_bounds: List[float] = []
            equalities: List[np.ndarray] = []
            equality_bounds: List[float] = []
            compiled_constraints: List[Dict[str, Any]] = []
            for constraint in spec.constraints:
                if not constraint.get("executable"):
                    continue
                left_map, left_constant = compile_linear_expression(
                    str(constraint["left_expression"]), names, scalar_parameters
                )
                right_map, right_constant = compile_linear_expression(
                    str(constraint["right_expression"]), names, scalar_parameters
                )
                row = np.asarray([
                    left_map.get(name, 0.0) - right_map.get(name, 0.0) for name in names
                ], dtype=float)
                bound = float(right_constant - left_constant)
                operator = str(constraint["operator"])
                if operator in {">=", "≥"}:
                    inequalities.append(-row)
                    inequality_bounds.append(-bound)
                    canonical_row, canonical_bound, canonical_operator = -row, -bound, "<="
                elif operator in {"<=", "≤"}:
                    inequalities.append(row)
                    inequality_bounds.append(bound)
                    canonical_row, canonical_bound, canonical_operator = row, bound, "<="
                elif operator == "=":
                    equalities.append(row)
                    equality_bounds.append(bound)
                    canonical_row, canonical_bound, canonical_operator = row, bound, "="
                else:
                    raise ValueError(f"不支持约束运算符 {operator}")
                compiled_constraints.append({
                    "source": constraint.get("text"), "operator": canonical_operator,
                    "coefficients": canonical_row.tolist(), "bound": canonical_bound,
                })
        except (SyntaxError, TypeError, ValueError) as exc:
            return {
                "method": "explicit_linear_program", "solver": "HiGHS",
                "solver_success": False, "solver_status": "compile_error",
                "message": str(exc), "decision_variables": names,
                "credibility_audit": {
                    "status": "fail", "label": "不可求解",
                    "decision": "显式模型未能安全编译，不能给出优化结论。",
                    "checks": [{
                        "id": "safe_compilation", "name": "安全符号编译", "status": "fail",
                        "evidence": str(exc), "recommendation": "补充标量参数并修正代数表达式。",
                    }],
                },
            }

        bounds = [
            tuple(symbol.bounds) if symbol.bounds is not None else (None, None)
            for symbol in decisions
        ]
        A_ub = np.vstack(inequalities) if inequalities else None
        b_ub = np.asarray(inequality_bounds, dtype=float) if inequalities else None
        A_eq = np.vstack(equalities) if equalities else None
        b_eq = np.asarray(equality_bounds, dtype=float) if equalities else None
        try:
            solved = linprog(
                solver_coefficients, A_ub=A_ub, b_ub=b_ub,
                A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs",
            )
        except (TypeError, ValueError) as exc:
            return _json_safe({
                "method": "explicit_linear_program", "solver": "HiGHS",
                "solver_success": False, "solver_status": "invalid_formulation",
                "message": str(exc), "decision_variables": names,
                "compiled_constraints": compiled_constraints,
                "credibility_audit": {
                    "status": "fail", "label": "模型矛盾",
                    "decision": "变量边界或约束矩阵不合法，当前优化模型无可采信解。",
                    "checks": [{
                        "id": "formulation_validity", "name": "优化模型一致性", "status": "fail",
                        "evidence": str(exc), "recommendation": "修正相互矛盾的边界、约束或参数。",
                    }],
                },
            })
        if not solved.success:
            checks = [{
                "id": "solver_termination", "name": "求解器终止状态", "status": "fail",
                "evidence": f"HiGHS status={solved.status}: {solved.message}",
                "recommendation": "检查不可行约束、无界变量、单位和参数值。",
            }]
            return _json_safe({
                "method": "explicit_linear_program", "solver": "HiGHS",
                "solver_success": False, "solver_status": int(solved.status),
                "message": str(solved.message), "decision_variables": names,
                "compiled_constraints": compiled_constraints,
                "credibility_audit": {
                    "status": "fail", "label": "不可求解",
                    "decision": "求解器未返回有限可行最优解，不能报告最优方案。",
                    "checks": checks,
                },
            })

        solution = np.asarray(solved.x, dtype=float)
        inequality_violation = (
            float(np.max(np.maximum(A_ub @ solution - b_ub, 0.0)))
            if A_ub is not None else 0.0
        )
        equality_violation = (
            float(np.max(np.abs(A_eq @ solution - b_eq)))
            if A_eq is not None else 0.0
        )
        bound_violation = 0.0
        for value, (lower, upper) in zip(solution, bounds):
            if lower is not None:
                bound_violation = max(bound_violation, float(max(lower - value, 0.0)))
            if upper is not None:
                bound_violation = max(bound_violation, float(max(value - upper, 0.0)))
        maximum_violation = max(inequality_violation, equality_violation, bound_violation)
        objective_value = float(original_coefficients @ solution + objective_constant)

        # Reconstruct first-order LP optimality conditions from HiGHS marginals.
        # This does not rely solely on the textual solver status.
        stationarity = solver_coefficients.copy()
        complementarity_terms: List[float] = []
        if A_ub is not None:
            inequality_marginals = np.asarray(solved.ineqlin.marginals, dtype=float)
            inequality_residuals = np.asarray(solved.ineqlin.residual, dtype=float)
            stationarity -= A_ub.T @ inequality_marginals
            complementarity_terms.extend(np.abs(
                inequality_marginals * inequality_residuals
            ).tolist())
        if A_eq is not None:
            equality_marginals = np.asarray(solved.eqlin.marginals, dtype=float)
            stationarity -= A_eq.T @ equality_marginals
        lower_marginals = np.asarray(solved.lower.marginals, dtype=float)
        upper_marginals = np.asarray(solved.upper.marginals, dtype=float)
        stationarity -= lower_marginals + upper_marginals
        lower_residuals = np.asarray(solved.lower.residual, dtype=float)
        upper_residuals = np.asarray(solved.upper.residual, dtype=float)
        finite_lower = np.isfinite(lower_residuals)
        finite_upper = np.isfinite(upper_residuals)
        complementarity_terms.extend(np.abs(
            lower_marginals[finite_lower] * lower_residuals[finite_lower]
        ).tolist())
        complementarity_terms.extend(np.abs(
            upper_marginals[finite_upper] * upper_residuals[finite_upper]
        ).tolist())
        stationarity_residual = float(np.max(np.abs(stationarity))) if len(stationarity) else 0.0
        complementarity_residual = max(complementarity_terms, default=0.0)
        kkt_residual = max(stationarity_residual, complementarity_residual)
        optimality_certificate = {
            "status": "pass" if kkt_residual <= 1e-7 else "fail",
            "stationarity_residual": stationarity_residual,
            "complementarity_residual": complementarity_residual,
            "maximum_kkt_residual": kkt_residual,
            "meaning": "连续线性模型的一阶最优性与互补松弛数值证书。",
        }

        # Explore the near-optimal face.  Large ranges mean that the objective is
        # identified while the actual decision vector is not unique.
        optimality_tolerance = 1e-7 * max(1.0, abs(float(solved.fun)))
        near_A = np.vstack([A_ub, solver_coefficients]) if A_ub is not None else solver_coefficients[None, :]
        near_b = np.concatenate([b_ub, [float(solved.fun) + optimality_tolerance]]) if b_ub is not None else np.asarray([float(solved.fun) + optimality_tolerance])
        alternative_ranges: Dict[str, List[Optional[float]]] = {}
        for position, name in enumerate(names):
            direction_vector = np.zeros(len(names), dtype=float)
            direction_vector[position] = 1.0
            lower_result = linprog(
                direction_vector, A_ub=near_A, b_ub=near_b, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
            )
            upper_result = linprog(
                -direction_vector, A_ub=near_A, b_ub=near_b, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
            )
            alternative_ranges[name] = [
                float(lower_result.fun) if lower_result.success else None,
                float(-upper_result.fun) if upper_result.success else None,
            ]
        finite_widths = [
            upper - lower for lower, upper in alternative_ranges.values()
            if lower is not None and upper is not None
        ]
        unbounded_near_variables = [
            name for name, (lower, upper) in alternative_ranges.items()
            if lower is None or upper is None
        ]
        solution_scale = max(float(np.linalg.norm(solution)), 1.0)
        maximum_relative_range = (
            max(finite_widths, default=0.0) / solution_scale
        )

        rng = np.random.default_rng(random_state + 911)
        perturbed_solutions: List[np.ndarray] = []
        perturbed_objectives: List[float] = []
        scenario_coefficients: List[np.ndarray] = []
        scenario_optimal_values: List[float] = []
        for _ in range(30):
            noise = rng.normal(0.0, 0.05, len(solver_coefficients))
            perturbed_c = solver_coefficients * (1.0 + noise)
            candidate = linprog(
                perturbed_c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method="highs",
            )
            if candidate.success:
                perturbed_solutions.append(np.asarray(candidate.x, dtype=float))
                perturbed_objectives.append(float(original_coefficients @ candidate.x + objective_constant))
                scenario_coefficients.append(perturbed_c)
                scenario_optimal_values.append(float(candidate.fun))
        solution_shift = (
            float(np.median([
                np.linalg.norm(candidate - solution) / solution_scale
                for candidate in perturbed_solutions
            ])) if perturbed_solutions else None
        )
        sensitivity_status = (
            "not_assessed" if solution_shift is None else
            ("pass" if solution_shift <= 0.10 else ("warning" if solution_shift <= 0.50 else "fail"))
        )
        alternative_status = (
            "warning" if unbounded_near_variables else
            ("pass" if maximum_relative_range <= 0.10 else
             ("warning" if maximum_relative_range <= 1.0 else "fail"))
        )
        robust_feedback: Dict[str, Any] = {
            "attempted": bool(scenario_coefficients),
            "accepted_as_primary": False,
            "uncertainty_assumption": "目标系数独立高斯5%压力扰动，仅作诊断",
        }
        if scenario_coefficients:
            candidate_matrix = np.vstack([solution, *perturbed_solutions])
            normalized_regrets = np.empty((len(candidate_matrix), len(scenario_coefficients)), dtype=float)
            for scenario_index, (scenario_c, optimum) in enumerate(
                zip(scenario_coefficients, scenario_optimal_values)
            ):
                costs = candidate_matrix @ scenario_c
                normalized_regrets[:, scenario_index] = (
                    costs - optimum
                ) / max(abs(optimum), 1.0)
            worst_regrets = np.max(np.maximum(normalized_regrets, 0.0), axis=1)
            robust_position = int(np.argmin(worst_regrets))
            robust_solution = candidate_matrix[robust_position]
            nominal_regret = float(worst_regrets[0])
            robust_regret = float(worst_regrets[robust_position])
            regret_reduction = (
                (nominal_regret - robust_regret) / max(nominal_regret, 1e-12)
                if nominal_regret > 1e-12 else 0.0
            )
            robust_feedback.update({
                "candidate_solution": {
                    name: float(value) for name, value in zip(names, robust_solution)
                },
                "candidate_objective_value": float(
                    original_coefficients @ robust_solution + objective_constant
                ),
                "nominal_worst_normalized_regret": nominal_regret,
                "candidate_worst_normalized_regret": robust_regret,
                "relative_regret_reduction": float(regret_reduction),
                "differs_from_nominal": bool(
                    np.linalg.norm(robust_solution - solution) > 1e-8 * solution_scale
                ),
                "recommendation": (
                    "若题目确认5%系数扰动具有现实含义，可将该最小最大遗憾方案作为稳健候选；"
                    "在确认不确定集合前不得替换名义最优解。"
                ),
            })
        checks = [
            {
                "id": "solver_termination", "name": "求解器终止状态", "status": "pass",
                "evidence": f"HiGHS 返回最优状态：{solved.message}", "recommendation": "",
            },
            {
                "id": "primal_feasibility", "name": "原始可行性", "status": "pass" if maximum_violation <= 1e-7 else "fail",
                "evidence": f"最大约束/边界违反={maximum_violation:.3g}",
                "recommendation": "检查数值尺度并提高求解精度。" if maximum_violation > 1e-7 else "",
            },
            {
                "id": "kkt_optimality", "name": "KKT最优性与互补松弛",
                "status": optimality_certificate["status"],
                "evidence": (
                    f"驻点残差={stationarity_residual:.3g}，"
                    f"互补松弛残差={complementarity_residual:.3g}。"
                ),
                "recommendation": "重新缩放模型并提高求解精度。"
                if optimality_certificate["status"] == "fail" else "",
            },
            {
                "id": "near_optimal_identifiability", "name": "近优解唯一性", "status": alternative_status,
                "evidence": (
                    f"最优容差面上存在无界/未识别变量：{unbounded_near_variables}。"
                    if unbounded_near_variables else
                    f"最优容差面上最大相对变量范围={maximum_relative_range:.3g}。"
                ),
                "recommendation": "报告替代最优区间或加入次级目标。" if alternative_status != "pass" else "",
            },
            {
                "id": "objective_sensitivity", "name": "目标系数扰动", "status": sensitivity_status,
                "evidence": (
                    "未形成可行扰动样本。" if solution_shift is None else
                    f"30次5%系数扰动中位相对方案变化={solution_shift:.3g}。"
                ),
                "recommendation": "使用稳健/情景优化并报告参数不确定集合。"
                if sensitivity_status in {"warning", "fail", "not_assessed"} else "",
            },
        ]
        failed = any(check["status"] == "fail" for check in checks)
        limited = any(check["status"] in {"warning", "not_assessed"} for check in checks)
        audit_status = "fail" if failed else ("warning" if limited else "pass")
        return _json_safe({
            "method": "explicit_linear_program", "solver": "HiGHS",
            "solver_success": True, "solver_status": int(solved.status),
            "message": str(solved.message), "direction": direction,
            "objective_expression": objective["expression"],
            "objective_value": objective_value,
            "decision_variables": names,
            "solution": {name: float(value) for name, value in zip(names, solution)},
            "maximum_constraint_violation": maximum_violation,
            "optimality_certificate": optimality_certificate,
            "compiled_constraints": compiled_constraints,
            "near_optimal_ranges": alternative_ranges,
            "sensitivity": {
                "coefficient_perturbation_scale": 0.05,
                "successful_runs": len(perturbed_solutions),
                "median_relative_solution_shift": solution_shift,
                "objective_values": perturbed_objectives,
            },
            "robust_feedback": robust_feedback,
            "credibility_audit": {
                "status": audit_status,
                "label": "可信" if audit_status == "pass" else (
                    "有条件可信" if audit_status == "warning" else "不可信"
                ),
                "decision": (
                    "当前显式线性模型求解自洽，但现实有效性仍依赖目标、约束和参数完整性。"
                    if audit_status != "fail" else
                    "存在可行性、稳定性或识别失败，当前方案不能作为唯一最优决策。"
                ),
                "checks": checks,
            },
            "note": "最优性只针对显式输入的连续线性模型；未声明的现实规则不在证明范围内。",
        })

    @staticmethod
    def _is_executable_statement(text: str) -> bool:
        return bool(
            re.search(r"(?:<=|>=|=|≤|≥)", text)
            and not any(token in text for token in ("进一步提取", "分析题目"))
        )

    @staticmethod
    def _readiness_context(
        datasets: Mapping[str, pd.DataFrame],
        role_bindings: Mapping[str, str],
        targets: Sequence[Tuple[str, str]],
        problem_analysis: Mapping[str, Any],
    ) -> Dict[str, Any]:
        numeric_columns = sum(
            int(pd.api.types.is_numeric_dtype(frame[column]))
            for frame in datasets.values() for column in frame.columns
        )
        time_columns = sum(
            int(pd.api.types.is_datetime64_any_dtype(frame[column]) or any(
                word in str(column).lower() for word in _DATE_WORDS
            )) for frame in datasets.values() for column in frame.columns
        )
        max_rows = max((len(frame) for frame in datasets.values()), default=0)
        constraints = [str(item) for item in problem_analysis.get("constraints", [])]
        objectives = [str(item) for item in problem_analysis.get("objectives", [])]
        return {
            "numeric_columns": numeric_columns, "time_columns": time_columns,
            "max_rows": max_rows, "has_target": bool(targets or role_bindings.get("outcome")),
            "has_treatment": bool(role_bindings.get("treatment")),
            "has_outcome": bool(role_bindings.get("outcome")),
            "has_constraints": any("进一步提取" not in item for item in constraints),
            "has_objective": any("分析题目" not in item for item in objectives),
            "has_decision_variable": any(
                token in str(problem_analysis.get("variables", [])).lower()
                for token in ("决策", "方案", "分配", "路径", "数量")
            ),
            "has_pre_treatment": False,
        }

    @staticmethod
    def _candidate_missing(
        task: str, requirements: Sequence[str], context: Mapping[str, Any]
    ) -> List[str]:
        missing: List[str] = []
        for requirement in requirements:
            satisfied = True
            if "处理变量" in requirement:
                satisfied = context["has_treatment"]
            elif "结果变量" in requirement or "目标变量" in requirement:
                satisfied = context["has_outcome"] or context["has_target"]
            elif "处理前混杂变量" in requirement:
                satisfied = context["has_pre_treatment"]
            elif "时间变量" in requirement:
                satisfied = context["time_columns"] >= 1
            elif "至少50" in requirement:
                satisfied = context["max_rows"] >= 50
            elif "至少两个" in requirement:
                satisfied = context["numeric_columns"] >= 2
            elif "足够样本" in requirement:
                satisfied = context["max_rows"] >= 30
            elif "决策变量" in requirement:
                satisfied = context["has_decision_variable"]
            elif "目标函数" in requirement:
                satisfied = context["has_objective"]
            elif "可执行约束" in requirement:
                satisfied = context["has_constraints"]
            elif "不确定参数集合" in requirement:
                satisfied = False
            elif "起点列" in requirement or "终点列" in requirement:
                satisfied = False
            if not satisfied:
                missing.append(requirement)
        return list(dict.fromkeys(missing))

    @staticmethod
    def _build_assumptions(
        tasks: Sequence[str], role_bindings: Mapping[str, str], symbols: Sequence[ModelSymbol], problem: str
    ) -> List[AssumptionRecord]:
        assumptions: List[AssumptionRecord] = []

        def add(
            assumption_id: str, text: str, category: str, critical: bool,
            testability: str, status: str, evidence: str, falsification: str,
            affected: Iterable[str],
        ) -> None:
            assumptions.append(AssumptionRecord(
                assumption_id, text, category, critical, testability, status,
                evidence, falsification, list(affected),
            ))

        predictive = [task for task in tasks if task in {"prediction_forecast", "classification"}]
        if predictive:
            add("assumption_generalization", "训练/验证样本能代表实际使用环境。", "sampling", True,
                "partially_testable", "not_assessed", "内部数据不能验证比赛范围外的分布。",
                "使用时间外、地区外或外部数据验证，并报告漂移。", predictive)
            add("assumption_no_leakage", "所有预测变量在预测时刻可获得且不含目标代理。", "temporal", True,
                "testable", "partially_checked", "已记录字段可用时点；仍需业务确认。",
                "逐字段核对产生时点，并进行删除可疑变量实验。", predictive)
        if "causal_inference" in tasks:
            explicit = bool(role_bindings.get("treatment") and role_bindings.get("outcome"))
            add("assumption_causal_roles", "处理、结果和因果方向由题目显式给定。", "identification", True,
                "testable", "checked" if explicit else "failed",
                "已显式绑定角色。" if explicit else "缺少显式处理或结果变量。",
                "明确写出处理变量和结果变量，禁止由相关性反推方向。", ["causal_inference"])
            add("assumption_exchangeability", "给定处理前协变量后不存在未观测混杂。", "identification", True,
                "not_fully_testable", "not_assessed", "观察数据不能单独验证无未观测混杂。",
                "随机化、自然实验、工具变量、负对照或敏感性分析。", ["causal_inference"])
            add("assumption_sutva", "处理定义一致且个体之间不存在未建模干扰。", "identification", True,
                "domain_only", "not_assessed", "需要题目机理和实验设计信息。",
                "检查处理版本、溢出效应和网络干扰。", ["causal_inference"])
            add("assumption_positivity", "各关键协变量区域都具有足够处理重叠。", "identification", True,
                "partially_testable", "not_assessed", "需要估计共同支持域。",
                "检查倾向得分或连续处理残差变异，并限制外推区域。", ["causal_inference"])
            add("assumption_pre_treatment", "控制变量均在处理发生前确定。", "temporal", True,
                "domain_only", "partially_checked" if any(
                    token in str(problem).lower()
                    for token in ("处理前", "干预前", "政策前", "事前", "基线协变量", "pre-treatment")
                ) else "not_assessed",
                "题目已声明处理前/基线协变量。" if any(
                    token in str(problem).lower()
                    for token in ("处理前", "干预前", "政策前", "事前", "基线协变量", "pre-treatment")
                ) else "尚未声明控制变量的产生时点。",
                "排除中介、碰撞变量和处理后变量。", ["causal_inference"])
        if "evaluation_ranking" in tasks:
            add("assumption_indicator_direction", "每个评价指标的正负方向符合题意。", "preference", True,
                "domain_only", "not_assessed", "字段名推断不能替代题意确认。",
                "让使用者确认每个指标方向并比较翻转后的排名。", ["evaluation_ranking"])
            add("assumption_scalarization", "单一综合分数足以表达决策偏好。", "preference", False,
                "partially_testable", "partially_checked", "可用 Pareto 前沿揭示被压缩的权衡。",
                "报告非支配集、权重敏感性和替代方案。", ["evaluation_ranking"])
        if "differential_equations" in tasks:
            add("assumption_state_observability", "已观测变量足以近似封闭系统状态。", "mechanism", True,
                "not_fully_testable", "not_assessed", "遗漏状态可能产生伪动力项。",
                "检查残差结构、加入候选状态并做外推实验。", ["differential_equations"])
            add("assumption_library_adequacy", "候选函数库包含真实动力学的有效近似。", "mechanism", True,
                "partially_testable", "not_assessed", "稀疏回归只能在给定函数库中选择。",
                "比较多套函数库、窗口和稀疏强度下的支持集。", ["differential_equations"])
        if "optimization" in tasks:
            add("assumption_constraint_completeness", "目标、约束和决策边界完整表达现实规则。", "optimization", True,
                "domain_only", "not_assessed", "自然语言解析无法证明约束完整。",
                "逐条建立约束来源表并用已知可行/不可行方案复核。", ["optimization"])
        if any(task in tasks for task in ("clustering", "dimension_reduction", "anomaly_detection")):
            affected = [task for task in tasks if task in {"clustering", "dimension_reduction", "anomaly_detection"}]
            add("assumption_geometry", "缩放后的距离和方差能表达问题中的相似性。", "representation", True,
                "partially_testable", "partially_checked", "可检查重采样稳定性，但语义仍需领域确认。",
                "比较稳健缩放、距离度量与删变量后的结构。", affected)

        # A global unit assumption is checked only when usable unit metadata exist.
        known_units = sum(symbol.dimension is not None for symbol in symbols)
        add("assumption_units", "所有参与运算的变量单位已统一且量纲相容。", "mathematical", True,
            "testable", "partially_checked" if known_units else "not_assessed",
            f"自动识别到 {known_units}/{len(symbols)} 个字段单位。",
            "在列名或模型规范中补充单位，并执行等式/约束量纲检查。", tasks)
        return assumptions

    @staticmethod
    def _extract_and_check_equations(
        problem: str, symbols: Sequence[ModelSymbol]
    ) -> List[Dict[str, Any]]:
        unit_by_name = {
            symbol.name: symbol.unit for symbol in symbols if symbol.unit
        }
        if not unit_by_name:
            return []
        checks: List[Dict[str, Any]] = []
        # Only inspect explicit formula clauses.  Descriptive uses of '=' such as
        # "处理变量=x" are excluded, avoiding an invented algebraic equation.
        for clause in re.split(r"[；;。\n]", str(problem)):
            comparator = re.search(r"<=|>=|≤|≥|(?<![<>])=(?!=)", clause)
            if not comparator:
                continue
            if any(marker in clause.lower() for marker in ("处理变量", "结果变量", "目标变量", "treatment", "outcome")):
                continue
            left = clause[:comparator.start()]
            right = clause[comparator.end():]
            mentioned = [name for name in unit_by_name if name in clause]
            if not mentioned:
                continue
            # Remove prose prefixes ending in ':'/'为' while retaining Unicode identifiers.
            left = re.split(r"[:：]", left)[-1].strip()
            if "为" in left and left not in unit_by_name:
                left = left.rsplit("为", 1)[-1].strip()
            starts = [left.find(name) for name in mentioned if left.find(name) >= 0]
            if starts:
                left = left[min(starts):].strip()
            right = right.strip()
            check = check_equation_dimensions(left, right, unit_by_name)
            check["expression"] = f"{left} {comparator.group(0)} {right}"
            checks.append(check)
        return checks

    @staticmethod
    def _audit_status(audit: Mapping[str, Any]) -> str:
        status = str(audit.get("status", "not_assessed"))
        return status if status in {"pass", "warning", "fail", "not_assessed"} else "not_assessed"

    @staticmethod
    def _collect_audits(
        model_results: Sequence[Mapping[str, Any]],
        ranking_result: Optional[Mapping[str, Any]],
        specialized_results: Mapping[str, Any],
    ) -> List[Mapping[str, Any]]:
        audits: List[Mapping[str, Any]] = [
            item.get("credibility_audit", {}) for item in model_results
        ]
        if ranking_result:
            audits.append(ranking_result.get("credibility_audit", {}))
        for key in (
            "causal_effect", "equation_discovery", "mathematical_data_compilation",
        ):
            payload = specialized_results.get(key)
            if isinstance(payload, Mapping):
                audits.append(payload.get("credibility_audit", {}))
        for payload in specialized_results.get("data_structure", []):
            if isinstance(payload, Mapping):
                audits.append(payload.get("credibility_audit", {}))
        return [audit for audit in audits if audit]

    def _reconcile_assumptions(
        self,
        source: Sequence[AssumptionRecord],
        model_results: Sequence[Mapping[str, Any]],
        ranking_result: Optional[Mapping[str, Any]],
        specialized_results: Mapping[str, Any],
    ) -> List[AssumptionRecord]:
        assumptions = [AssumptionRecord(**asdict(item)) for item in source]
        checks: Dict[str, List[Mapping[str, Any]]] = {}
        for audit in self._collect_audits(model_results, ranking_result, specialized_results):
            for check in audit.get("checks", []):
                checks.setdefault(str(check.get("id", "")), []).append(check)

        mapping = {
            "assumption_no_leakage": ["target_leakage", "validation_protocol"],
            "assumption_generalization": ["validation_protocol", "distribution_shift"],
            "assumption_causal_roles": ["causal_role_declaration"],
            "assumption_positivity": ["causal_overlap"],
            "assumption_pre_treatment": ["pre_treatment_controls"],
            "assumption_scalarization": ["weight_sensitivity", "pareto_tradeoff"],
            "assumption_geometry": [
                "cluster_seed_stability", "cluster_separation", "subspace_stability",
                "anomaly_perturbation_stability",
            ],
            "assumption_state_observability": ["dynamics_residual_structure"],
        }
        for assumption in assumptions:
            matched = [
                check for check_id in mapping.get(assumption.id, [])
                for check in checks.get(check_id, [])
            ]
            if not matched:
                continue
            statuses = {str(check.get("status", "not_assessed")) for check in matched}
            if "fail" in statuses:
                assumption.status = "failed"
            elif statuses <= {"pass"}:
                assumption.status = (
                    "checked" if assumption.testability == "testable" else "partially_checked"
                )
            elif "warning" in statuses or "not_assessed" in statuses:
                assumption.status = "partially_checked"
            assumption.evidence = "；".join(
                str(check.get("evidence", "")) for check in matched if check.get("evidence")
            )[:1200]
        return assumptions

    @staticmethod
    def _claim_grade(claim_type: str, audit_status: str) -> Tuple[str, str, str]:
        if audit_status == "fail":
            return "refuted", "当前反证", "rejected"
        if audit_status == "not_assessed":
            return "undetermined", "不可判定", "unresolved"
        if claim_type == "optimization_certificate" and audit_status == "pass":
            return "deductively_verified", "数学上已验证", "accepted_with_scope"
        if claim_type in {"causal", "ranking", "optimization", "mechanistic_execution"}:
            return "conditionally_supported", "在明确假设下成立", "restricted"
        if audit_status == "warning":
            return "conditionally_supported", "在明确假设下成立", "restricted"
        return "empirical_support", "经验支持", "accepted_with_scope"

    @staticmethod
    def _numerical_certificate(kind: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
        checks: List[Dict[str, Any]] = []

        def add(name: str, passed: Optional[bool], evidence: str) -> None:
            checks.append({
                "name": name,
                "status": (
                    "not_assessed" if passed is None else ("pass" if passed else "fail")
                ),
                "evidence": evidence,
            })

        if kind == "ranking":
            weights = np.asarray(list(payload.get("weights", {}).values()), dtype=float)
            ranks = [row.get("rank") for row in payload.get("ranking", [])]
            add("权重单纯形", bool(len(weights) and np.all(np.isfinite(weights)) and
                              np.all(weights >= -1e-12) and abs(weights.sum() - 1.0) <= 1e-8),
                f"权重和={float(weights.sum()) if len(weights) else None}")
            add("排名序号", ranks == list(range(1, len(ranks) + 1)), f"返回 {len(ranks)} 个有序名次")
        elif kind == "causal":
            interval = payload.get("confidence_interval_95", [])
            effect = payload.get("effect")
            finite = len(interval) == 2 and all(np.isfinite(float(item)) for item in interval)
            contains = finite and effect is not None and interval[0] <= effect <= interval[1]
            add("效应区间有限有序", bool(finite and interval[0] <= interval[1]), f"区间={interval}")
            add("点估计位于区间", bool(contains), f"effect={effect}")
        elif kind == "equation":
            metrics = payload.get("metrics", {})
            required = [metrics.get("validation_rmse"), metrics.get("validation_r2")]
            add("外验证指标有限", all(value is not None and np.isfinite(float(value)) for value in required),
                f"validation_rmse={required[0]}, validation_r2={required[1]}")
            add("验证窗口非空", int(payload.get("validation_windows", 0)) > 0,
                f"validation_windows={payload.get('validation_windows', 0)}")
        elif kind in {"predictive", "clustering", "structure"}:
            metrics = payload.get("metrics", {})
            numeric_values = [
                value for value in metrics.values() if isinstance(value, (int, float, np.integer, np.floating))
            ]
            add("报告指标有限", bool(numeric_values) and all(np.isfinite(float(value)) for value in numeric_values),
                f"检查 {len(numeric_values)} 个数值指标")
        elif kind == "graph":
            nodes = int(payload.get("n_nodes", -1))
            edges = int(payload.get("n_unique_edges", -1))
            components = int(payload.get("connected_components", -1))
            density = payload.get("density")
            add("图计数边界", nodes >= 0 and edges >= 0 and 0 <= components <= max(nodes, 0),
                f"nodes={nodes}, edges={edges}, components={components}")
            add("图密度边界", density is not None and np.isfinite(float(density)) and 0 <= float(density) <= 1,
                f"density={density}")
        elif kind == "uncertainty":
            interval = payload.get("mean_confidence_interval_95", [])
            mean = payload.get("observed_mean")
            finite = len(interval) == 2 and all(np.isfinite(float(value)) for value in interval)
            add("Bootstrap 区间有限有序", bool(finite and interval[0] <= interval[1]), f"interval={interval}")
            add("观测均值位于区间", bool(finite and mean is not None and interval[0] <= mean <= interval[1]),
                f"mean={mean}")
        elif kind == "time_dynamics":
            values = [payload.get("linear_trend_per_day"), payload.get("residual_std")]
            add("时序统计量有限", all(value is not None and np.isfinite(float(value)) for value in values),
                f"trend={values[0]}, residual_std={values[1]}")
            add("时间点充分", int(payload.get("n_time_points", 0)) >= 3,
                f"n_time_points={payload.get('n_time_points', 0)}")
        elif kind == "association":
            metrics = payload.get("metrics", {})
            strength = metrics.get("strength")
            concentration = payload.get("concentration", {})
            if concentration:
                hhi = concentration.get("hhi")
                top_share = concentration.get("top_20_share")
                association_rows = [
                    *payload.get("parent_associations", payload.get("category_associations", [])),
                    *payload.get("child_associations", payload.get("item_associations", [])),
                ]
                finite_rows = all(
                    np.isfinite(float(item.get("residual_spearman")))
                    and np.isfinite(float(item.get("q_value")))
                    for item in association_rows
                )
                add(
                    "层级集中度边界",
                    hhi is not None and top_share is not None
                    and 0 <= float(hhi) <= 1 and 0 <= float(top_share) <= 1,
                    f"hhi={hhi}, top_20_share={top_share}",
                )
                add(
                    "层级关联统计量有限", bool(association_rows) and finite_rows,
                    f"checked_pairs={len(association_rows)}",
                )
            else:
                add("关联强度有限", strength is not None and np.isfinite(float(strength)),
                    f"strength={strength}")
        elif kind in {"optimization", "optimization_certificate"}:
            nested_solver = payload.get("solver_result")
            if isinstance(nested_solver, Mapping):
                summary = nested_solver.get("summary", {})
                violation = summary.get(
                    "maximum_constraint_violation",
                    nested_solver.get("maximum_constraint_violation"),
                )
                integrality = summary.get(
                    "integrality_violation",
                    nested_solver.get("integrality_violation"),
                )
                objective = summary.get(
                    "objective_value", nested_solver.get("objective_value")
                )
                executed = nested_solver.get("status") == "executed"
                add(
                    "组合优化求解器终止", executed,
                    f"status={nested_solver.get('status')}, solver={nested_solver.get('solver')}",
                )
                add(
                    "组合优化约束残差",
                    violation is not None and np.isfinite(float(violation))
                    and float(violation) <= 1e-7,
                    f"maximum_violation={violation}",
                )
                add(
                    "组合优化整数残差",
                    integrality is not None and np.isfinite(float(integrality))
                    and float(integrality) <= 1e-7,
                    f"integrality_violation={integrality}",
                )
                add(
                    "组合优化目标值有限",
                    objective is not None and np.isfinite(float(objective)),
                    f"objective={objective}",
                )
            elif payload.get("method") == "forecast_to_finite_action_milp":
                add(
                    "组合优化数值证书", None,
                    "成本或动作覆盖不完整，未生成可复算的求解器最优证书。",
                )
            else:
                success = bool(payload.get("solver_success"))
                violation = payload.get("maximum_constraint_violation")
                objective = payload.get("objective_value")
                kkt = payload.get("optimality_certificate", {})
                add("求解器最优终止", success, f"status={payload.get('solver_status')}, message={payload.get('message')}")
                add("约束残差", violation is not None and np.isfinite(float(violation)) and float(violation) <= 1e-7,
                    f"maximum_violation={violation}")
                add("KKT数值证书", kkt.get("status") == "pass",
                    f"maximum_kkt_residual={kkt.get('maximum_kkt_residual')}")
                add("目标值有限", objective is not None and np.isfinite(float(objective)), f"objective={objective}")
        elif kind == "mechanistic_specification":
            ir = payload.get("mathematical_ir", {})
            operators = payload.get("operator_graph", [])
            policy = payload.get("input_policy", {})
            extracted_count = sum(
                len(ir.get(key, []))
                for key in ("entities", "quantities", "relations", "objectives", "constraints")
            )
            add("IR结构可核验", isinstance(ir, Mapping),
                f"extracted_items={extracted_count}, operator_count={len(operators)}")
            if operators:
                operator_ids = [str(item.get("id", "")) for item in operators]
                add("算子标识唯一", len(operator_ids) == len(set(operator_ids)),
                    f"operator_count={len(operators)}")
            add("未伪造观测数据", policy.get("observations_invented") is False,
                f"observations_invented={policy.get('observations_invented')}")
        elif kind == "mechanistic_execution":
            summary = payload.get("summary", {})
            numeric_values = []
            for state in summary.values():
                if isinstance(state, Mapping):
                    numeric_values.extend(
                        value for value in state.values()
                        if isinstance(value, (int, float, np.integer, np.floating))
                    )
                elif isinstance(state, (int, float, np.integer, np.floating)):
                    numeric_values.append(state)
            add("求解状态", payload.get("status") == "executed",
                f"status={payload.get('status')}, solver={payload.get('solver')}")
            add("结果摘要有限", bool(numeric_values) and all(np.isfinite(float(value)) for value in numeric_values),
                f"checked_values={len(numeric_values)}")
            if payload.get("kind") == "optimization_solution":
                violation = payload.get("maximum_constraint_violation")
                add("约束可行性", violation is not None and np.isfinite(float(violation)) and float(violation) <= 1e-7,
                    f"maximum_constraint_violation={violation}")
                add("多起点收敛", int(payload.get("successful_starts", 0)) > 0,
                    f"successful={payload.get('successful_starts')}/{payload.get('attempted_starts')}")
            elif payload.get("kind") in {
                "kinematic_visibility_event",
                "kinematic_visibility_optimization_solution",
            }:
                duration = payload.get("duration")
                intervals = payload.get("effective_intervals", [])
                interval_measure = sum(
                    float(right) - float(left) for left, right in intervals
                    if len([left, right]) == 2
                )
                add(
                    "事件区间与时长一致",
                    duration is not None and np.isfinite(float(duration))
                    and float(duration) >= 0
                    and abs(interval_measure - float(duration)) <= 1e-8,
                    f"duration={duration}, interval_measure={interval_measure}",
                )
                convergence = payload.get("convergence", {})
                add(
                    "事件根加密复算", convergence.get("status") == "pass",
                    f"maximum_duration_difference={convergence.get('maximum_duration_difference')}",
                )
                if payload.get("kind") == "kinematic_visibility_optimization_solution":
                    add(
                        "非凸搜索存在多个有限候选",
                        int(payload.get("successful_starts", 0)) >= 2,
                        f"successful={payload.get('successful_starts')}/"
                        f"{payload.get('attempted_starts')}",
                    )
            else:
                convergence = payload.get("convergence", {})
                add("独立容差复算", convergence.get("status") in {"pass", "warning"},
                    f"relative_difference={convergence.get('relative_tolerance_comparison')}")
            independent_audit = payload.get("independent_audit", {})
            if independent_audit:
                add(
                    "四层独立审计未拒绝",
                    independent_audit.get("status") in {"pass", "warning"},
                    f"status={independent_audit.get('status')}, "
                    f"grade={independent_audit.get('grade')}, "
                    f"flags={independent_audit.get('false_confidence_flags', [])}",
                )
        failed = [item for item in checks if item["status"] == "fail"]
        unavailable = [item for item in checks if item["status"] == "not_assessed"]
        return {
            "status": (
                "fail" if failed else
                ("not_assessed" if unavailable or not checks else "pass")
            ),
            "checks": checks,
            "meaning": "仅证明算术与结果结构自洽，不证明模型假设或现实解释正确。",
        }

    def build_evidence_bundle(
        self,
        spec: MathematicalModelSpec,
        datasets: Mapping[str, pd.DataFrame],
        relationships: Sequence[Any],
        interactions: Sequence[Any],
        model_results: Sequence[Mapping[str, Any]],
        ranking_result: Optional[Mapping[str, Any]],
        specialized_results: Mapping[str, Any],
        task_graph: Sequence[Mapping[str, Any]],
    ) -> EvidenceBundle:
        nodes: List[EvidenceNode] = []
        claims: List[ClaimAssessment] = []
        edges: List[Dict[str, str]] = []
        assumptions = self._reconcile_assumptions(
            spec.assumptions, model_results, ranking_result, specialized_results
        )
        manifest = self._dataset_manifest(datasets)
        dataset_node_ids: Dict[str, str] = {}
        for item in manifest:
            node_id = _stable_id("data", item["dataset"], item["fingerprint"])
            dataset_node_ids[item["dataset"]] = node_id
            nodes.append(EvidenceNode(node_id, "data", item["dataset"], "observed", "data_manifest", item))
        for assumption in assumptions:
            nodes.append(EvidenceNode(
                assumption.id, "assumption", assumption.text, assumption.status,
                "model_spec.assumptions", {"critical": assumption.critical,
                                             "testability": assumption.testability,
                                             "evidence": assumption.evidence},
            ))

        def add_claim(
            statement: str, claim_type: str, source_path: str, dataset_names: Sequence[str],
            payload: Mapping[str, Any], audit: Optional[Mapping[str, Any]],
            assumption_tasks: Sequence[str], scope: str, invalid_when: Sequence[str],
        ) -> None:
            audit = audit or {}
            audit_status = self._audit_status(audit)
            certificate = self._numerical_certificate(claim_type, payload)
            if certificate["status"] == "fail":
                audit_status = "fail"
            grade, label, disposition = self._claim_grade(claim_type, audit_status)
            claim_id = _stable_id("claim", source_path, statement)
            support_ids = [dataset_node_ids[name] for name in dataset_names if name in dataset_node_ids]
            challenge_ids: List[str] = []
            for index, check in enumerate(audit.get("checks", []), 1):
                check_status = str(check.get("status", "not_assessed"))
                node_id = _stable_id("check", claim_id, check.get("id", index))
                nodes.append(EvidenceNode(
                    node_id, "falsification_check", str(check.get("name", check.get("id", "检查"))),
                    check_status, f"{source_path}.credibility_audit.checks[{index - 1}]",
                    {"evidence": check.get("evidence"), "recommendation": check.get("recommendation"),
                     "details": check.get("details", {})},
                ))
                relation = "supports" if check_status == "pass" else "challenges"
                edges.append({"from": node_id, "to": claim_id, "relation": relation})
                (support_ids if relation == "supports" else challenge_ids).append(node_id)
            certificate_id = _stable_id("certificate", claim_id)
            nodes.append(EvidenceNode(
                certificate_id, "numerical_certificate", "数值结果结构校验",
                certificate["status"], f"{source_path}.numerical_certificate", certificate,
            ))
            edges.append({
                "from": certificate_id, "to": claim_id,
                "relation": "supports" if certificate["status"] == "pass" else "challenges",
            })
            (support_ids if certificate["status"] == "pass" else challenge_ids).append(certificate_id)

            relevant_assumptions = [
                item for item in assumptions
                if any(task in item.affected_tasks for task in assumption_tasks)
            ]
            assumption_ids = [item.id for item in relevant_assumptions]
            for assumption in relevant_assumptions:
                edges.append({"from": assumption.id, "to": claim_id, "relation": "depends_on"})
            if any(item.critical and item.status == "failed" for item in relevant_assumptions):
                grade, label, disposition = "refuted", "当前反证", "rejected"
            elif grade == "empirical_support" and claim_type == "causal":
                grade, label, disposition = "conditionally_supported", "在明确假设下成立", "restricted"
            next_actions = [
                str(check.get("recommendation")) for check in audit.get("checks", [])
                if check.get("status") in {"fail", "warning", "not_assessed"}
                and check.get("recommendation")
            ]
            claims.append(ClaimAssessment(
                id=claim_id, statement=statement, claim_type=claim_type, grade=grade,
                label=label, disposition=disposition, scope=scope,
                supports=list(dict.fromkeys(support_ids)), challenges=list(dict.fromkeys(challenge_ids)),
                assumptions=assumption_ids, invalid_when=list(invalid_when),
                next_actions=list(dict.fromkeys(next_actions))[:8],
                numerical_certificate=certificate,
            ))
            for node_id in dataset_node_ids.values():
                if node_id in support_ids:
                    edges.append({"from": node_id, "to": claim_id, "relation": "derived_from"})

        for index, contradiction in enumerate(spec.contradictions):
            message = str(contradiction.get("message", "数学规范存在矛盾"))
            audit = {"status": "fail", "checks": [{
                "id": str(contradiction.get("id", f"contradiction_{index + 1}")),
                "name": "数学规范一致性", "status": "fail", "evidence": message,
                "recommendation": "修正变量角色、单位、边界或代数关系后重新编译。",
            }]}
            add_claim(
                f"当前数学规范不能同时成立：{message}",
                "mathematical_consistency", f"model_spec.contradictions[{index}]",
                [], {"metrics": {}}, audit, spec.task_types,
                "这是输入规范的静态反证，不依赖模型拟合分数。",
                ["题目或单位声明被修正"],
            )

        for index, model in enumerate(model_results):
            task_type = str(model.get("task_type", "predictive"))
            dataset = str(model.get("dataset", ""))
            target = model.get("target")
            if task_type == "clustering":
                statement = f"{dataset} 在当前特征与距离定义下存在 k={model.get('best_k')} 的可重复分群候选。"
                claim_type = "clustering"
                tasks = ["clustering"]
                scope = "仅描述当前样本在既定缩放和距离下的结构，不代表客观类别。"
            else:
                statement = (
                    f"{model.get('best_model', '候选模型')} 对 {dataset}.{target} 在"
                    f" {model.get('validation', '当前验证')} 上获得报告指标 {model.get('metrics', {})}。"
                )
                claim_type = "predictive"
                tasks = ["prediction_forecast", "classification"]
                scope = "指标只适用于相同字段语义、可用时点和被验证的数据范围。"
            add_claim(
                statement, claim_type, f"model_results[{index}]", [dataset], model,
                model.get("credibility_audit", {}), tasks, scope,
                ["字段产生时点变化", "数据分布超出验证范围", "目标定义变化"],
            )

        data_compilation = specialized_results.get("mathematical_data_compilation")
        if isinstance(data_compilation, Mapping):
            compilation_dataset = str(data_compilation.get("dataset", ""))
            relationships = (
                (data_compilation.get("conclusion_stress") or {}).get("relationships") or []
            )
            for relationship_index, relationship in enumerate(relationships):
                relation_status = str(relationship.get("status", ""))
                significant = bool(relationship.get("global_significant_fdr_0_05"))
                if relation_status not in {"contradicted", "restricted"} and not significant:
                    continue
                contexts = list(relationship.get("contexts") or [])
                global_context = next(
                    (item for item in contexts if item.get("view") == "global_complete_case"),
                    {},
                )
                finite_strengths = [
                    abs(float(item.get("rho"))) for item in contexts
                    if item.get("rho") is not None and np.isfinite(float(item.get("rho")))
                ]
                if not finite_strengths:
                    continue
                flips = list(relationship.get("direction_flips") or [])
                if relation_status == "contradicted":
                    alternatives = ", ".join(
                        f"{item.get('against')}={item.get('alternative_rho')}"
                        for item in flips[:4]
                    )
                    statement = (
                        f"“{relationship.get('predictor')} 与 {relationship.get('target')} 存在稳定同向关系”"
                        f"这一结论被多视图审计反证：总体 Spearman ρ={global_context.get('rho')}，"
                        f"而合理替代估计视图得到反向结果（{alternatives}）。"
                    )
                    audit_status = "fail"
                    evidence = (
                        f"direction_flips={len(flips)}, simpson_risk={relationship.get('simpson_risk')}, "
                        f"effect_spread={relationship.get('effect_spread')}"
                    )
                elif relation_status == "restricted":
                    statement = (
                        f"{relationship.get('predictor')} 与 {relationship.get('target')} 的关系方向"
                        "尚未翻转，但效应量明显依赖数据粒度或处理视图，只能限定条件采用。"
                    )
                    audit_status = "warning"
                    evidence = f"effect_spread={relationship.get('effect_spread')}"
                else:
                    statement = (
                        f"{relationship.get('predictor')} 与 {relationship.get('target')} 的总体秩相关"
                        f"在当前多视图压力测试中方向稳定，且通过全局FDR（q="
                        f"{relationship.get('global_fdr_q')}）。"
                    )
                    audit_status = "pass"
                    evidence = (
                        f"global_rho={global_context.get('rho')}, "
                        f"q={relationship.get('global_fdr_q')}, tested_views={len(contexts)}"
                    )
                relation_audit = {
                    "status": audit_status,
                    "checks": [{
                        "id": "conclusion_view_stability",
                        "name": "合理数据视图下的方向稳定性",
                        "status": audit_status,
                        "evidence": evidence,
                        "recommendation": (
                            "分别报告总体、组内、组间与时间聚合估计对象；"
                            "发生翻转时拒绝无条件总体规律。"
                        ),
                    }],
                }
                add_claim(
                    statement,
                    "association",
                    "specialized_results.mathematical_data_compilation."
                    f"conclusion_stress.relationships[{relationship_index}]",
                    [compilation_dataset],
                    {
                        "metrics": {"strength": max(finite_strengths)},
                        "contexts": contexts,
                    },
                    relation_audit,
                    ["statistical_inference"],
                    "比较的是同一目标在完整样本、填补、缩尾、组内、组间和时间聚合等"
                    "合理数据视图下的经验关系；不构成因果识别。",
                    [
                        "目标或分组语义改变", "观测粒度改变", "存在未检验的选择机制",
                        "独立数据不能复现",
                    ],
                )
            for cross_index, cross_contract in enumerate(
                data_compilation.get("cross_dataset_contracts") or []
            ):
                if cross_contract.get("status") != "blocked":
                    continue
                left_dataset = str(cross_contract.get("left_dataset", ""))
                right_dataset = str(cross_contract.get("right_dataset", ""))
                key_text = ", ".join(
                    f"{item.get('left')}↔{item.get('right')}"
                    for item in cross_contract.get("key_pairs", [])
                ) or "未验证键"
                join_audit = {
                    "status": "fail",
                    "checks": [{
                        "id": "cross_dataset_cardinality",
                        "name": "跨表键基数与连接膨胀",
                        "status": "fail",
                        "evidence": (
                            f"relationship={cross_contract.get('relationship')}, "
                            f"estimated_expansion={cross_contract.get('estimated_expansion')}, "
                            f"point_in_time_required={cross_contract.get('point_in_time_required')}"
                        ),
                        "recommendation": "至少对一侧按目标估计粒度预聚合，再重新验证复合键和时间可用性。",
                    }],
                }
                add_claim(
                    f"“可通过 {key_text} 直接连接 {left_dataset} 与 {right_dataset}，"
                    "且连接后总量和样本单位保持不变”这一数据前提被反证。",
                    "association",
                    "specialized_results.mathematical_data_compilation."
                    f"cross_dataset_contracts[{cross_index}]",
                    [left_dataset, right_dataset],
                    {"metrics": {"strength": float(
                        cross_contract.get("estimated_expansion") or 0.0
                    )}},
                    join_audit,
                    ["statistical_inference", "prediction_forecast", "optimization"],
                    "这是连接基数和估计粒度审计，不否定两张表存在可用关系；"
                    "它只禁止未经聚合的原始多对多拼接。",
                    ["任一侧已按目标粒度聚合", "采用了新的唯一复合键", "时间对齐规则被补齐"],
                )

        hierarchical_sales = specialized_results.get("hierarchical_sales")
        if hierarchical_sales:
            concentration = hierarchical_sales.get("concentration", {})
            significant_categories = sum(
                bool(item.get("significant"))
                for item in hierarchical_sales.get("category_associations", [])
            )
            significant_items = sum(
                bool(item.get("significant"))
                for item in hierarchical_sales.get("item_associations", [])
            )
            add_claim(
                f"已将 {hierarchical_sales.get('dataset')}.{hierarchical_sales.get('target')} "
                f"聚合到日×{hierarchical_sales.get('category_column')}×"
                f"{hierarchical_sales.get('item_column')}，单品 HHI={concentration.get('hhi')}、"
                f"前20单品份额={concentration.get('top_20_share')}；去除趋势和星期效应并经FDR后，"
                f"保留 {significant_categories} 个品类对和 {significant_items} 个单品对。",
                "association", "specialized_results.hierarchical_sales",
                [str(hierarchical_sales.get("dataset", ""))], hierarchical_sales,
                hierarchical_sales.get("credibility_audit", {}), ["statistical_inference"],
                "分布和联动仅适用于当前销售记录、零销量补全规则与层级映射；不作因果解释。",
                ["交易漏报", "层级映射错误", "缺货导致需求删失", "促销或陈列造成共同冲击"],
            )

        grouped_forecasts = list(specialized_results.get("grouped_forecasts") or [])
        if not grouped_forecasts and specialized_results.get("grouped_forecast"):
            grouped_forecasts = [specialized_results["grouped_forecast"]]
        for grouped_index, grouped_forecast in enumerate(grouped_forecasts):
            metrics = grouped_forecast.get("metrics", {})
            add_claim(
                f"已将 {grouped_forecast.get('dataset')}.{grouped_forecast.get('target')} "
                f"按日×{grouped_forecast.get('group_column')}先聚合，再对 "
                f"{grouped_forecast.get('groups_forecast')} 个组预测 "
                f"{grouped_forecast.get('horizon_days')} 天；末段回测指标为 {metrics}。",
                "predictive", f"specialized_results.grouped_forecasts[{grouped_index}]",
                [str(grouped_forecast.get("dataset", ""))], grouped_forecast,
                grouped_forecast.get("credibility_audit", {}), ["prediction_forecast"],
                "结论适用于当前日粒度、分组映射和末段回测范围；区间不覆盖未知结构突变。",
                ["分组映射错误", "缺行不代表零销量", "未来发生结构突变", "目标聚合口径改变"],
            )

        prescriptive_decisions = list(specialized_results.get("prescriptive_decisions") or [])
        if not prescriptive_decisions and specialized_results.get("prescriptive_decision"):
            prescriptive_decisions = [specialized_results["prescriptive_decision"]]
        for prescriptive_index, prescriptive in enumerate(prescriptive_decisions):
            solver_summary = (prescriptive.get("solver_result") or {}).get("summary", {})
            risk_stress = prescriptive.get("risk_aware_stress_test") or {}
            risk_clause = (
                f"另以预测区间构造有限情景并复算下尾CVaR，风险感知候选相对"
                f"名义方案改变 {risk_stress.get('changed_decision_unit_count', 0)} 个决策单元；"
                f"{'已按题面风险偏好采用' if risk_stress.get('adopted') else '因情景权重未校准而仅作压力测试'}。"
                if risk_stress else ""
            )
            add_claim(
                f"已把分组预测、成本、损耗和通过稳定性门的价格动作编译为"
                f"{prescriptive.get('mathematical_form')}，得到 "
                f"{prescriptive.get('decision_count')} 条条件性补货/价格候选；"
                f"通用求解摘要为 {solver_summary}。{risk_clause}",
                "optimization", f"specialized_results.prescriptive_decisions[{prescriptive_index}]", [],
                prescriptive, prescriptive.get("credibility_audit", {}), ["optimization"],
                "仅适用于当前预测区间、历史价格边界、成本/损耗映射和有限动作集合；观察性弹性不构成调价因果证据。",
                ["成本或损耗口径错误", "遗漏容量/库存/供应约束", "价格弹性不是稳定因果效应", "需求发生结构突变"],
            )
            cost_plus_rows = list(
                prescriptive.get("cost_plus_pricing_relationship") or []
            )
            if cost_plus_rows:
                significant_count = sum(
                    bool(item.get("significant")) for item in cost_plus_rows
                )
                median_strength = float(np.median([
                    abs(float(item.get("residual_spearman", 0.0)))
                    for item in cost_plus_rows
                ]))
                relationship_audit = {
                    "status": "pass",
                    "checks": [{
                        "id": "cost_plus_pricing_alignment",
                        "status": "pass",
                        "evidence": (
                            f"{len(cost_plus_rows)} 个品类均使用日期×品类内连接，"
                            "去除趋势和星期效应，并统一执行BH-FDR与分半方向复核。"
                        ),
                    }],
                }
                relationship_payload = {
                    "metrics": {"strength": median_strength},
                    "rows": cost_plus_rows,
                }
                finding = (
                    f"其中 {significant_count} 个品类通过BH-FDR和分半同号联合门"
                    if significant_count else
                    "当前没有品类通过BH-FDR和分半同号联合门"
                )
                add_claim(
                    f"已按日期×品类对齐售价、批发成本和销量，对 "
                    f"{len(cost_plus_rows)} 个品类检验去趋势、去星期效应后的"
                    f"成本加成率—销量关系；{finding}。",
                    "association",
                    f"specialized_results.prescriptive_decisions[{prescriptive_index}]"
                    ".cost_plus_pricing_relationship",
                    [
                        str(prescriptive.get("dataset", "")),
                        str(prescriptive.get("cost_dataset", "")),
                    ],
                    relationship_payload,
                    relationship_audit,
                    ["statistical_inference"],
                    "该结论是当前时间范围和品类聚合口径下的观察性证据；"
                    "成本采用当日品类内单品批发价中位数近似，不构成调价因果效应。",
                    [
                        "库存删失使销量不等于真实需求",
                        "促销或陈列同时影响价格和销量",
                        "品类内商品结构变化使中位成本失真",
                        "存在未观测共同原因或时间结构突变",
                    ],
                )

        if ranking_result:
            ranking = ranking_result.get("ranking", [])
            winner = ranking[0].get("entity") if ranking else "未知对象"
            add_claim(
                f"在当前指标方向与熵权标量化规则下，{winner} 的 TOPSIS 排名为首位。",
                "ranking", "ranking_result", [str(ranking_result.get("dataset", ""))],
                ranking_result, ranking_result.get("credibility_audit", {}),
                ["evaluation_ranking"],
                "这是偏好依赖的条件排序，Pareto 非支配关系优先于单一名次。",
                ["指标方向改变", "权重偏好改变", "候选方案集合改变"],
            )

        equation = specialized_results.get("equation_discovery")
        if equation:
            add_claim(
                f"方程 {equation.get('equation')} 是 {equation.get('target')} 的可反证动力学候选。",
                "equation", "specialized_results.equation_discovery",
                [str(equation.get("dataset", ""))], equation,
                equation.get("credibility_audit", {}), ["differential_equations"],
                "仅表明候选函数库中的外验证拟合，不证明真实机理。",
                ["采样间隔改变", "遗漏状态变量", "候选函数库不充分", "外推失败"],
            )

        causal = specialized_results.get("causal_effect")
        if causal:
            interval = causal.get("confidence_interval_95", [None, None])
            add_claim(
                f"在部分线性、条件可交换等识别假设下，{causal.get('treatment')} 对"
                f" {causal.get('outcome')} 的平均效应估计为 {causal.get('effect')}，95%区间为 {interval}。",
                "causal", "specialized_results.causal_effect",
                [str(causal.get("dataset", ""))], causal,
                causal.get("credibility_audit", {}), ["causal_inference"],
                "因果解释依赖未观测混杂、SUTVA、重叠性和处理前控制等条件。",
                ["存在未控制共同原因", "处理版本不一致", "干扰效应", "重叠性不足"],
            )

        optimization = specialized_results.get("optimization")
        if optimization:
            solution_text = optimization.get("solution", {}) if optimization.get("solver_success") else "无可行最优解"
            add_claim(
                f"对显式连续线性模型，HiGHS 返回方案 {solution_text}，目标值为"
                f" {optimization.get('objective_value')}。",
                "optimization", "specialized_results.optimization", [], optimization,
                optimization.get("credibility_audit", {}), ["optimization"],
                "最优性仅相对于已声明的目标、约束、边界和标量参数。",
                ["遗漏现实约束", "参数或单位错误", "不确定参数超出扰动范围", "模型并非线性"],
            )
            if optimization.get("solver_success"):
                certificate_checks = [
                    check for check in optimization.get("credibility_audit", {}).get("checks", [])
                    if check.get("id") in {
                        "solver_termination", "primal_feasibility", "kkt_optimality"
                    }
                ]
                certificate_status = (
                    "pass" if certificate_checks and all(
                        check.get("status") == "pass" for check in certificate_checks
                    ) else "fail"
                )
                add_claim(
                    "在已编译的连续线性规划和 1e-7 数值容差内，返回方案满足原始可行性、"
                    "KKT驻点条件与互补松弛，因此是该输入模型的数值最优解。",
                    "optimization_certificate",
                    "specialized_results.optimization.optimality_certificate", [], optimization,
                    {"status": certificate_status, "checks": certificate_checks}, [],
                    "只证明已输入连续线性规划的数值最优性，不证明现实目标与约束完整。",
                    ["数值残差超过容差", "模型表达式或参数被修改"],
                )

        mechanistic = specialized_results.get("mechanistic_model", {})
        math_ir = mechanistic.get("mathematical_ir", {})
        operators = mechanistic.get("operator_graph", [])
        has_mechanistic_content = bool(operators) or any(
            math_ir.get(key) for key in (
                "entities", "quantities", "relations", "objectives", "constraints"
            )
        )
        if (
            mechanistic
            and has_mechanistic_content
            and mechanistic.get("presentation_scope", "primary") == "primary"
        ):
            unresolved = mechanistic.get("missing_requirements", [])
            model_draft = mechanistic.get("model_draft", {})
            completed_stages = model_draft.get("completed_stages", [])
            add_claim(
                f"已从题面抽取 {len(math_ir.get('entities', []))} 个带坐标实体、"
                f"{len(math_ir.get('quantities', []))} 个显式量和 {len(operators)} 个通用数学算子；"
                f"建立 {len(model_draft.get('equations', []))} 条规范方程草案并完成 "
                f"{len(completed_stages)} 个建模阶段；当前仍有 {len(unresolved)} 类未绑定条件。",
                "mechanistic_specification",
                "specialized_results.mechanistic_model", [], mechanistic,
                mechanistic.get("credibility_audit", {}),
                ["differential_equations", "simulation", "optimization", "graph_network"],
                "这是题面到数学中间表示的可审计编译结果，不是数值答案。",
                ["题面抽取错误", "符号或单位绑定错误", "缺少初边值条件", "采用了不同数学语义"],
            )
            for index, numerical in enumerate(mechanistic.get("numerical_results", []), 1):
                if numerical.get("status") != "executed":
                    continue
                summary = numerical.get("summary", {})
                final_values = {
                    name: values.get("final") for name, values in summary.items()
                    if isinstance(values, Mapping)
                }
                if numerical.get("kind") == "kinematic_visibility_event":
                    statement = (
                        "在已编译的运动学、视线距离与连续事件定义下，"
                        f"有效事件时长为 {numerical.get('duration', 0):.6f} s，"
                        f"有效区间为 {numerical.get('effective_intervals', [])}；"
                        f"合理目标代表点语义的时长范围为 "
                        f"{numerical.get('semantic_duration_range', [])}。"
                    )
                    scope = (
                        "仅适用于题面编译出的匀速轨迹、抛体运动、球形影响区、"
                        "源—目标线段距离判据和已声明的目标代表点；不是全目标遮蔽证明。"
                    )
                    invalid_when = [
                        "目标遮蔽语义改为全圆柱或任一点", "存在未建模空气阻力或风场",
                        "坐标单位或重力常数约定改变", "事件求根加密复算失败",
                    ]
                    supported_tasks = ["simulation"]
                elif numerical.get("kind") == "kinematic_visibility_optimization_solution":
                    statement = (
                        "在已编译的连续运动—视线事件模型下，多起点非凸搜索得到"
                        f"有效时长 {numerical.get('duration', 0):.6f} s 的可行最优候选；"
                        f"决策参数为 {numerical.get('solution', {})}，"
                        f"有效区间为 {numerical.get('effective_intervals', [])}。"
                    )
                    scope = (
                        "这是当前速度、方向、释放和激活边界下，经多种子与局部精化得到的"
                        "高质量可行候选；未提供非凸全局最优性的数学证明。"
                    )
                    invalid_when = [
                        "现实航向或投放边界未被题面完整表达", "遮蔽几何语义改变",
                        "空气阻力或风场不可忽略", "可验证全局上界否定当前候选",
                    ]
                    supported_tasks = ["optimization"]
                elif numerical.get("kind") in {
                    "linear_system_solution", "scalar_root_solution",
                    "linear_least_squares_solution", "linear_program_solution",
                    "mixed_integer_linear_program_solution", "shortest_path_solution",
                    "maximum_flow_solution", "bipartite_matching_solution",
                    "markov_chain_solution", "sample_expectation_solution",
                    "quadratic_program_solution", "multiobjective_program_solution",
                    "robust_program_solution", "stochastic_program_solution",
                    "dynamic_program_solution", "minimum_cost_flow_solution",
                }:
                    statement = (
                        f"已按统一数学 IR 的 {numerical.get('mathematical_form', numerical.get('kind'))} "
                        f"契约完成通用求解；可复算摘要为 {numerical.get('summary', {})}。"
                    )
                    scope = (
                        "结论只适用于当前结构化变量、矩阵、图、概率或样本契约；"
                        "数值复算通过不证明题面到该契约的语义映射正确。"
                    )
                    invalid_when = [
                        "结构化契约与题意不一致", "单位或变量角色绑定错误",
                        "独立残差或结构不变量复算失败", "遗漏现实约束",
                    ]
                    supported_tasks = {
                        "linear_program_solution": ["optimization"],
                        "mixed_integer_linear_program_solution": ["optimization"],
                        "quadratic_program_solution": ["optimization"],
                        "multiobjective_program_solution": ["optimization"],
                        "robust_program_solution": ["optimization"],
                        "stochastic_program_solution": ["optimization"],
                        "dynamic_program_solution": ["optimization"],
                        "shortest_path_solution": ["graph_network"],
                        "maximum_flow_solution": ["graph_network"],
                        "minimum_cost_flow_solution": ["graph_network"],
                        "bipartite_matching_solution": ["graph_network"],
                        "markov_chain_solution": ["simulation"],
                        "sample_expectation_solution": ["simulation", "statistical_inference"],
                        "linear_least_squares_solution": ["statistical_inference"],
                    }.get(numerical.get("kind"), ["simulation"])
                else:
                    statement = (
                        f"已按通过安全编译的 {numerical.get('kind', 'mechanistic')} 契约执行数值求解；"
                        f"终点状态为 {final_values}。"
                    )
                    scope = "仅适用于已确认的状态、参数、单位、初值、时间范围和方程；数值收敛不等于机理真实。"
                    invalid_when = [
                        "结构化 IR 未经确认", "方程或单位改变", "积分容差复算失败",
                        "合理替代机理给出不同结论",
                    ]
                    supported_tasks = ["differential_equations", "simulation"]
                solver_audit = numerical.get("credibility_audit", {})
                independent_audit = numerical.get("independent_audit", {})
                combined_status = str(solver_audit.get("status", "not_assessed"))
                if independent_audit.get("status") == "fail":
                    combined_status = "fail"
                elif independent_audit.get("status") == "warning" and combined_status != "fail":
                    combined_status = "warning"
                combined_audit = {
                    **solver_audit,
                    "status": combined_status,
                    "checks": [
                        *solver_audit.get("checks", []),
                        *(
                            {
                                "id": f"independent_{check.get('id', check_index)}",
                                "name": f"四层独立审计：{check.get('id', check_index)}",
                                "status": check.get("status", "not_assessed"),
                                "evidence": check.get("evidence", "-"),
                                "recommendation": "按统一 IR 修正数学契约或降低结论等级。",
                            }
                            for check_index, check in enumerate(
                                independent_audit.get("checks", []), 1
                            )
                        ),
                    ],
                }
                add_claim(
                    statement,
                    "mechanistic_execution",
                    f"specialized_results.mechanistic_model.numerical_results[{index - 1}]",
                    [], numerical, combined_audit,
                    supported_tasks,
                    scope, invalid_when,
                )

        graph = specialized_results.get("graph_network")
        if graph:
            graph_audit = {"status": "warning", "checks": [{
                "id": "graph_definition", "name": "图定义边界", "status": "warning",
                "evidence": "节点和边来自当前起点/终点字段；业务语义尚需确认。",
                "recommendation": "核对边方向、重复边聚合和节点实体语义。",
            }]}
            add_claim(
                f"当前实体网络包含 {graph.get('n_nodes')} 个节点、{graph.get('n_unique_edges')} 条唯一边，"
                f"形成 {graph.get('connected_components')} 个连通分量。",
                "graph", "specialized_results.graph_network", [str(graph.get("dataset", ""))],
                graph, graph_audit, ["graph_network"],
                "这是当前边定义下的组合结构，不自动表示传播、因果或真实联系强度。",
                ["节点实体定义错误", "边方向错误", "重复边处理改变"],
            )

        simulation = specialized_results.get("simulation")
        if simulation:
            simulation_audit = {"status": "warning", "checks": [{
                "id": "bootstrap_exchangeability", "name": "重采样适用性", "status": "warning",
                "evidence": "普通行 Bootstrap 默认观测可交换，未自动证明时空/群组独立性。",
                "recommendation": "存在时间、空间或群组依赖时改用分块或分层 Bootstrap。",
            }]}
            add_claim(
                f"在当前重采样单位下，{simulation.get('dataset')}.{simulation.get('variable')} 的均值"
                f"95% Bootstrap 区间为 {simulation.get('mean_confidence_interval_95')}。",
                "uncertainty", "specialized_results.simulation",
                [str(simulation.get("dataset", ""))], simulation, simulation_audit,
                ["simulation"], "区间只覆盖抽样不确定性，不覆盖模型结构和测量偏差。",
                ["观测存在未建模依赖", "样本不代表总体", "重采样单位错误"],
            )

        dynamics = specialized_results.get("time_dynamics")
        if dynamics:
            dynamics_audit = {"status": "warning", "checks": [{
                "id": "trend_descriptive_only", "name": "趋势解释边界", "status": "warning",
                "evidence": "线性趋势与自相关是描述统计，未建立生成机理。",
                "recommendation": "比较结构突变、季节性和非线性趋势，并进行时间外验证。",
            }]}
            add_claim(
                f"{dynamics.get('dataset')}.{dynamics.get('variable')} 在当前时间范围内的经验线性趋势为"
                f" {dynamics.get('linear_trend_per_day')}/天。",
                "time_dynamics", "specialized_results.time_dynamics",
                [str(dynamics.get("dataset", ""))], dynamics, dynamics_audit,
                ["prediction_forecast", "differential_equations"],
                "仅描述已观测时间范围，不自动支持外推或机理解释。",
                ["结构突变", "季节性未建模", "时间戳或聚合粒度改变"],
            )

        for index, structure in enumerate(specialized_results.get("data_structure", [])):
            add_claim(
                f"{structure.get('dataset')} 的当前数值指标呈现 {structure.get('dimensions_90')} 维"
                f"稳健潜在结构，并标记 {structure.get('anomaly_count')} 个结构偏离样本。",
                "structure", f"specialized_results.data_structure[{index}]",
                [str(structure.get("dataset", ""))],
                {**structure, "metrics": {
                    "cumulative_explained_variance": structure.get("cumulative_explained_variance"),
                    "subspace_stability": structure.get("subspace_stability"),
                }}, structure.get("credibility_audit", {}),
                ["dimension_reduction", "anomaly_detection"],
                "异常表示相对当前样本结构的偏离，不自动等价于错误、欺诈或故障。",
                ["缩放或特征集合改变", "总体分布改变", "阈值用途改变"],
            )

        # Cross-table associations are claims too, but cap them to keep the graph bounded.
        for index, finding in enumerate(list(interactions)[:20]):
            payload = finding if isinstance(finding, Mapping) else asdict(finding)
            dataset_names = [str(payload.get("left_dataset", "")), str(payload.get("right_dataset", ""))]
            significant = payload.get("significant")
            status = (
                "pass" if significant is True else
                ("not_assessed" if significant is False else
                 (
                     "not_assessed"
                     if payload.get("p_value") is None
                     else ("warning" if payload.get("strength", 0) else "not_assessed")
                 ))
            )
            interpretation = str(payload.get("interpretation", "检测到跨表关联"))
            if significant is False:
                interpretation += (
                    f" 当前效应未通过全局 FDR 校正（q={payload.get('q_value')}），"
                    "不能作为稳定关系结论"
                )
            audit = {"status": status, "checks": [{
                "id": "fdr_and_stability", "name": "关联显著性与稳定性", "status": status,
                "evidence": (
                    f"strength={payload.get('strength')}, q={payload.get('q_value')}, "
                    f"stability={payload.get('stability_score')}"
                ),
                "recommendation": "检查条件关联、时滞、分层和独立数据复现。",
            }]}
            add_claim(
                interpretation + "（不作因果解释）。",
                "association", f"interactions[{index}]", dataset_names,
                {"metrics": {"strength": payload.get("strength")}}, audit,
                ["statistical_inference"], "当前联表键、样本和多重检验范围内的关联。",
                ["联表键错误", "条件变量解释关联", "独立样本不能复现"],
            )

        # Explicitly represent requested but unresolved tasks.  Silence must never
        # be mistaken for successful execution.
        claimed_task_types = set()
        for claim in claims:
            if claim.claim_type == "predictive":
                claimed_task_types.update({"prediction_forecast", "classification"})
            elif claim.claim_type == "equation":
                claimed_task_types.add("differential_equations")
            elif claim.claim_type == "causal":
                claimed_task_types.add("causal_inference")
            elif claim.claim_type == "ranking":
                claimed_task_types.add("evaluation_ranking")
            elif claim.claim_type == "clustering":
                claimed_task_types.add("clustering")
            elif claim.claim_type == "structure":
                claimed_task_types.update({"dimension_reduction", "anomaly_detection"})
            elif claim.claim_type == "association":
                claimed_task_types.add("statistical_inference")
            elif claim.claim_type == "graph":
                claimed_task_types.add("graph_network")
            elif claim.claim_type == "uncertainty":
                claimed_task_types.add("simulation")
            elif claim.claim_type == "optimization":
                claimed_task_types.add("optimization")
        for numerical in specialized_results.get("mechanistic_model", {}).get("numerical_results", []):
            if numerical.get("status") != "executed":
                continue
            if numerical.get("kind") == "ode_trajectory":
                claimed_task_types.update({"differential_equations", "simulation"})
            elif numerical.get("kind") == "optimization_solution":
                claimed_task_types.add("optimization")
            elif numerical.get("kind") == "kinematic_visibility_event":
                claimed_task_types.add("simulation")
            elif numerical.get("kind") == "kinematic_visibility_optimization_solution":
                claimed_task_types.add("optimization")

        mechanistic_model = specialized_results.get("mechanistic_model", {})
        mechanistic_subproblems = list(mechanistic_model.get("subproblems", []))
        subproblem_to_task = {
            str(item.get("id")): f"task_{index}"
            for index, item in enumerate(mechanistic_subproblems, 1)
        }
        mechanistic_executed_task_ids = set()
        mechanistic_executed_types = set()
        for numerical in mechanistic_model.get("numerical_results", []):
            if numerical.get("status") != "executed":
                continue
            task_id = subproblem_to_task.get(str(numerical.get("subproblem_id")))
            if task_id:
                mechanistic_executed_task_ids.add(task_id)
            if numerical.get("kind") == "kinematic_visibility_event":
                mechanistic_executed_types.add("simulation")
            elif numerical.get("kind") == "kinematic_visibility_optimization_solution":
                mechanistic_executed_types.add("optimization")

        pending_by_type: Dict[str, List[Mapping[str, Any]]] = {}
        for task in task_graph:
            task_type = str(task.get("task_type", ""))
            task_id = str(task.get("id", ""))
            if task_id in mechanistic_executed_task_ids:
                continue
            if task.get("status") in {
                "needs_input", "blocked", "partial", "ready", "planned"
            } and (
                task_type not in claimed_task_types
                or task_type in mechanistic_executed_types
            ):
                pending_by_type.setdefault(task_type or "unclassified", []).append(task)

        missing_labels = {
            "machine_readable_equations_or_algorithms": "可机器读取的方程或算法",
            "verified_symbol_and_unit_bindings": "题面符号、单位与规范方程的最终核验",
            "decision_variables": "决策变量",
            "objective": "可计算目标函数",
            "objectives": "多个可计算目标函数",
            "constraints": "约束与变量边界",
            "state": "状态变量",
            "initial_condition": "初始条件",
            "boundary_condition": "边界条件",
            "dynamics_law": "动力学关系",
            "rate_law": "速率关系",
            "geometry_definition": "明确的几何定义",
            "event_definition": "事件成立判据",
            "probability_model": "概率分布或随机机制",
            "computable_response": "可计算响应函数",
        }
        task_type_labels = {
            "data_requirements": "数据需求与采集设计",
            "optimization": "优化求解",
            "simulation": "事件仿真与时长计算",
            "differential_equations": "动力学方程求解",
            "statistical_inference": "统计推断",
            "prediction_forecast": "预测",
            "graph_network": "图与网络求解",
            "evaluation_ranking": "评价与排序",
        }
        for task_type, pending_tasks in pending_by_type.items():
            missing = list(dict.fromkeys(
                missing_labels.get(str(item), str(item))
                for task in pending_tasks
                for item in (
                    task.get("missing_requirements", [])
                    or task.get("requires", [])
                )
                if str(item).strip()
            ))
            if not missing:
                missing = ["能够由求解器复核的模型契约或数值证据"]
            claim_id = _stable_id(
                "claim", task_type,
                *(str(task.get("id")) for task in pending_tasks),
                "unresolved",
            )
            task_node_ids = []
            completed_evidence = []
            for task in pending_tasks:
                task_node_id = _stable_id("task", task.get("id"))
                task_node_ids.append(task_node_id)
                if task.get("evidence") and str(task.get("evidence")) not in completed_evidence:
                    completed_evidence.append(str(task.get("evidence")))
                nodes.append(EvidenceNode(
                    task_node_id, "execution_gap",
                    f"{task.get('id', '-')} · {task_type_labels.get(task_type, task_type)}"
                    "尚未形成数值结论",
                    str(task.get("status")), "problem_analysis.task_graph",
                    {
                        "subproblem": str(task.get("text", ""))[:500],
                        "missing_requirements": [
                            missing_labels.get(str(item), str(item))
                            for item in task.get("missing_requirements", [])
                        ],
                        "evidence": task.get("evidence"),
                    },
                ))
                edges.append({"from": task_node_id, "to": claim_id, "relation": "blocks"})
            visible_task_labels = [
                str(task.get("id", "-")) for task in pending_tasks[:8]
            ]
            task_labels = "、".join(visible_task_labels)
            if len(pending_tasks) > len(visible_task_labels):
                task_labels += " 等"
            completed = (
                "；".join(completed_evidence[:3]).rstrip("。；; ")
                or "已完成任务识别和求解路线生成"
            )
            display_task_type = task_type_labels.get(task_type, task_type)
            statement = (
                f"{display_task_type}的 {len(pending_tasks)} 个待求解节点（{task_labels}）"
                "尚未形成数值结论。"
                f"当前已完成：{completed}。下一步需补齐：{'、'.join(missing)}。"
            )
            claims.append(ClaimAssessment(
                claim_id, statement, task_type, "undetermined", "待完成", "unresolved",
                "这是合并后的执行缺口，不表示前面的题面解析、变量抽取和算子选择均失败。",
                [], task_node_ids, [], ["模型契约或数值验证尚未完成"], missing,
                {"status": "not_assessed", "checks": []},
            ))

        claim_ids = {claim.id for claim in claims}
        incoming = {claim_id: 0 for claim_id in claim_ids}
        for edge in edges:
            if edge["to"] in incoming:
                incoming[edge["to"]] += 1
        unsupported = [claim_id for claim_id, count in incoming.items() if count == 0]
        # This is a construction invariant.  If future code adds an unsupported
        # claim, it is automatically unresolved instead of leaking into prose.
        for claim in claims:
            if claim.id in unsupported:
                claim.grade, claim.label, claim.disposition = "undetermined", "不可判定", "unresolved"
                claim.challenges.append("argument_graph_missing_support")

        counts: Dict[str, int] = {}
        for claim in claims:
            counts[claim.grade] = counts.get(claim.grade, 0) + 1
        rejected = [claim.id for claim in claims if claim.disposition == "rejected"]
        unresolved = [claim.id for claim in claims if claim.disposition == "unresolved"]
        accepted = [claim for claim in claims if claim.disposition in {"accepted_with_scope", "restricted"}]
        if rejected:
            overall_status, overall_label = "contains_rejected_claims", "包含被反证结论"
        elif unresolved and not accepted:
            overall_status, overall_label = "undetermined", "当前不可判定"
        elif unresolved:
            has_mechanistic_specification = any(
                claim.claim_type == "mechanistic_specification" for claim in accepted
            )
            has_mechanistic_execution = any(
                claim.claim_type == "mechanistic_execution" for claim in accepted
            )
            overall_status, overall_label = (
                "partial",
                (
                    "数学结构已形成，部分数值结论已验证"
                    if has_mechanistic_execution else "数学结构已形成，数值结论待验证"
                )
                if has_mechanistic_specification else "部分有据、部分待验证",
            )
        elif any(claim.disposition == "restricted" for claim in claims):
            overall_status, overall_label = "conditional", "有条件支持"
        elif accepted:
            overall_status, overall_label = "empirical", "经验支持"
        else:
            overall_status, overall_label = "no_claims", "尚无数值结论"

        allowed = [
            claim.id for claim in claims if claim.disposition in {"accepted_with_scope", "restricted"}
        ]
        prohibited = [
            claim.id for claim in claims if claim.disposition in {"rejected", "unresolved"}
        ]
        tournaments: List[Dict[str, Any]] = []
        for index, model in enumerate(model_results):
            leaderboard = list(model.get("leaderboard", []))
            tournaments.append({
                "id": f"tournament_model_{index + 1}",
                "subject": f"{model.get('dataset')}.{model.get('target') or 'unsupervised'}",
                "selected": model.get("best_model"),
                "validation": model.get("validation"),
                "candidates": leaderboard[:20],
                "feedback_decision": model.get("feedback_optimization", {}),
                "selection_is_conclusion": False,
                "note": "候选选择仍受独立确认、近优模型分歧和可信度审计约束。",
            })
        if ranking_result:
            tournaments.append({
                "id": "tournament_ranking", "subject": ranking_result.get("dataset"),
                "selected": "entropy_weight_topsis",
                "candidates": ["pareto_front", "entropy_weight_topsis", "weight_perturbations"],
                "selection_is_conclusion": False,
                "note": "无权重 Pareto 结果用于反驳被单一标量排名隐藏的权衡。",
            })
        equation_result = specialized_results.get("equation_discovery")
        if equation_result:
            tournaments.append({
                "id": "tournament_equation", "subject": equation_result.get("target"),
                "selected": equation_result.get("equation"),
                "candidates": equation_result.get("candidate_search", [])[:20],
                "selection_is_conclusion": False,
                "note": "候选方程必须优于零变化基线并保持支持集稳定。",
            })
        optimization_result = specialized_results.get("optimization")
        if optimization_result:
            tournaments.append({
                "id": "tournament_optimization", "subject": "explicit_linear_program",
                "selected": optimization_result.get("solution"),
                "candidates": [
                    "nominal_HiGHS_solution", "near_optimal_face",
                    "30_objective_coefficient_perturbations",
                ],
                "selection_is_conclusion": False,
                "note": "名义最优解与近优替代解、参数扰动方案同时保留。",
            })

        return EvidenceBundle(
            version=self.version, overall_status=overall_status, overall_label=overall_label,
            claims=claims, evidence_nodes=nodes, edges=edges,
            assumption_ledger=assumptions, data_manifest=manifest, grade_counts=counts,
            rejected_claim_ids=rejected, unresolved_claim_ids=unresolved,
            model_tournament=tournaments,
            argument_integrity={
                "status": "pass" if not unsupported else "fail",
                "claims": len(claims), "evidence_nodes": len(nodes), "edges": len(edges),
                "unsupported_claim_ids": unsupported,
                "rule": "每个结论必须至少有一个数据、检查、证书或阻断证据入边。",
            },
            writing_contract={
                "enabled": False,
                "stage": "final_optional_api_only",
                "allowed_claim_ids": allowed,
                "prohibited_claim_ids": prohibited,
                "must_include_scope_and_assumptions": True,
                "may_invent_formulas": False,
                "may_invent_numbers": False,
                "instruction": (
                    "写作 API 只能改写 allowed_claim_ids 对应内容；必须保留适用范围、假设和反证，"
                    "不得把 rejected/unresolved 结论写成肯定事实。"
                ),
            },
        )


__all__ = [
    "UnitDimension", "parse_unit", "extract_column_unit",
    "check_expression_dimensions", "check_equation_dimensions", "classify_expression_structure",
    "compile_linear_expression",
    "ModelSymbol", "AssumptionRecord", "CandidateModel", "MathematicalModelSpec",
    "EvidenceNode", "ClaimAssessment", "EvidenceBundle", "MathematicalReasoningEngine",
]
