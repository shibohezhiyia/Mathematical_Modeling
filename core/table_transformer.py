"""通用、可组合且可审计的表数据变换引擎。

该模块刻意不绑定某一道建模题。它把常见的数据准备动作表示为声明式
pipeline，并在提交结果前完成列校验、规模预算和逐步审计。Web、CLI 和
自动研究流程可以共享同一份变换契约。
"""

from __future__ import annotations

import ast
import math
import operator
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


class TableTransformError(ValueError):
    """用户可修正的表变换错误。"""


@dataclass
class TransformationResult:
    data: pd.DataFrame
    audit: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    input_shape: Tuple[int, int] = (0, 0)

    @property
    def output_shape(self) -> Tuple[int, int]:
        return tuple(self.data.shape)


def _as_list(value: Any, name: str, *, allow_empty: bool = True) -> List[Any]:
    if value is None:
        result: List[Any] = []
    elif isinstance(value, (list, tuple)):
        result = list(value)
    else:
        result = [value]
    if not allow_empty and not result:
        raise TableTransformError(f"{name}不能为空")
    return result


def _deduplicate_labels(labels: Iterable[Any]) -> List[str]:
    counts: Dict[str, int] = {}
    result: List[str] = []
    for raw in labels:
        label = str(raw)
        counts[label] = counts.get(label, 0) + 1
        result.append(label if counts[label] == 1 else f"{label}__{counts[label]}")
    return result


class _SafeExpressionEvaluator:
    """只解释向量化算术 AST，不调用 Python eval。"""

    _BIN_OPS: Mapping[type, Callable[[Any, Any], Any]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _CMP_OPS: Mapping[type, Callable[[Any, Any], Any]] = {
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
    }

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def evaluate(self, expression: str) -> Any:
        if not isinstance(expression, str) or not expression.strip():
            raise TableTransformError("派生表达式不能为空")
        if len(expression) > 2_000:
            raise TableTransformError("派生表达式过长（最多2000字符）")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise TableTransformError(f"派生表达式语法错误: {exc.msg}") from exc
        return self._visit(tree.body)

    def _visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool, type(None))):
                return node.value
            raise TableTransformError("表达式包含不支持的常量")
        if isinstance(node, ast.Name):
            if node.id not in self.frame.columns:
                raise TableTransformError(f"表达式引用了不存在的列: {node.id}")
            return self.frame[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in self._BIN_OPS:
            left, right = self._visit(node.left), self._visit(node.right)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                return self._BIN_OPS[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp):
            value = self._visit(node.operand)
            if isinstance(node.op, ast.USub):
                return -value
            if isinstance(node.op, ast.UAdd):
                return +value
            if isinstance(node.op, (ast.Not, ast.Invert)):
                return ~value if isinstance(value, pd.Series) else not value
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            op = type(node.ops[0])
            if op not in self._CMP_OPS:
                raise TableTransformError("表达式比较运算符不受支持")
            return self._CMP_OPS[op](self._visit(node.left), self._visit(node.comparators[0]))
        if isinstance(node, ast.BoolOp):
            values = [self._visit(item) for item in node.values]
            if not values:
                raise TableTransformError("空布尔表达式")
            result = values[0]
            for value in values[1:]:
                result = (result & value) if isinstance(node.op, ast.And) else (result | value)
            return result
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.keywords:
                raise TableTransformError("表达式函数只接受位置参数")
            name = node.func.id
            if name == "col":
                if len(node.args) != 1 or not isinstance(node.args[0], ast.Constant):
                    raise TableTransformError('col()需要一个列名字符串，例如 col("销售额")')
                column = str(node.args[0].value)
                if column not in self.frame.columns:
                    raise TableTransformError(f"表达式引用了不存在的列: {column}")
                return self.frame[column]
            args = [self._visit(arg) for arg in node.args]
            functions: Dict[str, Callable[..., Any]] = {
                "abs": np.abs,
                "sqrt": np.sqrt,
                "log": np.log,
                "log1p": np.log1p,
                "exp": np.exp,
                "minimum": np.minimum,
                "maximum": np.maximum,
                "where": lambda c, a, b: pd.Series(np.where(c, a, b), index=self.frame.index),
                "clip": lambda x, lo, hi: x.clip(lower=lo, upper=hi),
                "round": lambda x, digits=0: x.round(int(digits)),
                "isna": pd.isna,
                "fillna": lambda x, value: x.fillna(value),
            }
            if name not in functions:
                raise TableTransformError(f"表达式函数不受支持: {name}")
            try:
                return functions[name](*args)
            except (TypeError, ValueError) as exc:
                raise TableTransformError(f"函数{name}参数无效: {exc}") from exc
        raise TableTransformError(f"表达式包含不安全或不支持的语法: {type(node).__name__}")


class TableTransformationEngine:
    """声明式 DataFrame 变换注册表与组合执行器。"""

    MAX_STEPS = 30
    MAX_RESULT_CELLS = 20_000_000
    MAX_COLUMNS = 5_000
    MAX_EXPANSION_RATIO = 50.0

    _STANDARD_AGGREGATIONS = {
        "sum", "mean", "median", "min", "max", "std", "var", "count",
        "nunique", "first", "last", "prod",
    }

    def __init__(
        self,
        *,
        max_result_cells: int = MAX_RESULT_CELLS,
        max_columns: int = MAX_COLUMNS,
        max_expansion_ratio: float = MAX_EXPANSION_RATIO,
    ) -> None:
        self.max_result_cells = int(max_result_cells)
        self.max_columns = int(max_columns)
        self.max_expansion_ratio = float(max_expansion_ratio)
        self._operations: Dict[str, Callable[[pd.DataFrame, Dict[str, Any]], pd.DataFrame]] = {
            "select_columns": self._op_select_columns,
            "drop_columns": self._op_drop_columns,
            "rename_columns": self._op_rename_columns,
            "filter_rows": self._op_filter_rows,
            "sort_rows": self._op_sort_rows,
            "deduplicate": self._op_deduplicate,
            "convert_types": self._op_convert_types,
            "fill_missing": self._op_fill_missing,
            "derive_columns": self._op_derive_columns,
            "aggregate": self._op_aggregate,
            "pivot": self._op_pivot,
            "melt": self._op_melt,
            "time_features": self._op_time_features,
            "resample_time": self._op_resample_time,
            "window_features": self._op_window_features,
            "normalize": self._op_normalize,
            "bin_numeric": self._op_bin_numeric,
            "encode_categorical": self._op_encode_categorical,
            "pairwise_distance": self._op_pairwise_distance,
        }

    @classmethod
    def capabilities(cls, frame: Optional[pd.DataFrame] = None) -> List[Dict[str, Any]]:
        """供 UI/外部模型读取的稳定操作契约，可按当前表绑定真实字段。"""
        capabilities = [
            {"name": "select_columns", "label": "选择字段", "category": "结构", "description": "保留指定字段并调整顺序", "template": {"columns": ["字段A", "字段B"]}},
            {"name": "drop_columns", "label": "删除字段", "category": "结构", "description": "删除不参与分析的字段", "template": {"columns": ["字段A"]}},
            {"name": "rename_columns", "label": "字段重命名", "category": "结构", "description": "统一多来源字段语义", "template": {"mapping": {"旧字段": "新字段"}}},
            {"name": "filter_rows", "label": "条件筛选", "category": "行处理", "description": "支持数值、类别、文本、空值与区间条件", "template": {"combine": "and", "conditions": [{"column": "字段A", "operator": "ge", "value": 0}]}},
            {"name": "sort_rows", "label": "排序", "category": "行处理", "description": "单字段或多字段稳定排序", "template": {"by": ["字段A"], "ascending": [True]}},
            {"name": "deduplicate", "label": "去重", "category": "清洗", "description": "按实体键或全行去重", "template": {"subset": ["ID"], "keep": "first"}},
            {"name": "convert_types", "label": "类型转换", "category": "清洗", "description": "数值、整数、日期、布尔、文本与类别类型转换", "template": {"mapping": {"日期": "datetime", "数值": "numeric"}, "errors": "coerce"}},
            {"name": "fill_missing", "label": "缺失值处理", "category": "清洗", "description": "常量、均值、中位数、众数、插值及分组填补", "template": {"columns": ["字段A"], "strategy": "median", "group_by": []}},
            {"name": "derive_columns", "label": "指标/公式派生", "category": "特征", "description": "安全表达式构造利润率、变化量、约束指标等", "template": {"expressions": {"利润": "收入 - 成本", "利润率": "利润 / 收入"}}},
            {"name": "aggregate", "label": "分组汇总", "category": "重构", "description": "多键、多指标及加权均值/分位数/占比汇总", "template": {"group_by": ["类别"], "aggregations": [{"column": "数值", "function": "sum", "output": "数值_合计"}]}},
            {"name": "pivot", "label": "透视表", "category": "重构", "description": "长表转宽表并限制维度爆炸", "template": {"index": ["时间"], "columns": ["类别"], "values": ["数值"], "aggfunc": "sum", "fill_value": 0}},
            {"name": "melt", "label": "宽表转长表", "category": "重构", "description": "把多个指标列还原为变量-取值结构", "template": {"id_vars": ["ID"], "value_vars": ["指标1", "指标2"], "var_name": "指标", "value_name": "数值"}},
            {"name": "time_features", "label": "时间特征", "category": "时序", "description": "构造年季月周、星期、小时和周期特征", "template": {"time_column": "日期", "features": ["year", "month", "dayofweek", "is_weekend"]}},
            {"name": "resample_time", "label": "时间重采样", "category": "时序", "description": "把不规则观测按日/周/月/季度等周期汇总", "template": {"time_column": "日期", "frequency": "D", "group_by": ["实体"], "aggregations": [{"column": "数值", "function": "sum", "output": "数值_周期合计"}]}},
            {"name": "window_features", "label": "时序/面板窗口", "category": "时序", "description": "按实体生成滞后、差分、增长率、滚动统计和累计量", "template": {"order_by": "日期", "partition_by": ["实体"], "value_columns": ["数值"], "features": [{"kind": "lag", "periods": 1}, {"kind": "rolling_mean", "window": 7, "shift": 1}]}},
            {"name": "normalize", "label": "数值变换", "category": "特征", "description": "Z-score、Min-Max、稳健缩放或对数变换", "template": {"columns": ["数值"], "method": "zscore", "suffix": "_z"}},
            {"name": "bin_numeric", "label": "数值分箱", "category": "特征", "description": "等宽、等频或自定义区间离散化", "template": {"column": "数值", "method": "quantile", "bins": 5, "output": "数值_分组"}},
            {"name": "encode_categorical", "label": "类别编码", "category": "特征", "description": "频率编码或受维度保护的独热编码", "template": {"columns": ["类别"], "method": "frequency", "max_categories": 30}},
            {"name": "pairwise_distance", "label": "坐标转距离边表", "category": "空间/网络", "description": "由二维/三维坐标生成欧氏、曼哈顿或球面距离边", "template": {"id_column": "节点", "coordinate_columns": ["经度", "纬度"], "metric": "haversine", "directed": False, "include_self": False}},
        ]
        if frame is None:
            return capabilities
        return cls._bind_capabilities(capabilities, frame)

    @classmethod
    def _bind_capabilities(
        cls, capabilities: List[Dict[str, Any]], frame: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Bind executable templates without inventing fields absent from the table."""
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] == 0:
            return [
                {**item, "availability": "unavailable", "availability_reason": "当前没有可用数据表"}
                for item in capabilities
            ]

        working = frame.copy(deep=False)
        working.columns = _deduplicate_labels(working.columns)
        columns = [str(column) for column in working.columns]
        numeric = [str(column) for column in working.select_dtypes(include=np.number).columns]
        lowered = {column: column.strip().lower() for column in columns}

        def named_like(column: str, tokens: Sequence[str]) -> bool:
            name = lowered[column]
            return any(name == token or token in name for token in tokens)

        identifier_tokens = ("id", "code", "编号", "编码", "序号", "节点", "实体号", "样本号")
        identifier_candidates = [
            column for column in columns
            if named_like(column, identifier_tokens)
        ]
        identifier_candidates.sort(
            key=lambda column: (
                float(working[column].nunique(dropna=True)) / max(len(working), 1),
                -columns.index(column),
            ),
            reverse=True,
        )

        datetime_columns: List[str] = []
        for column in columns:
            series = working[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                datetime_columns.append(column)
                continue
            if named_like(column, ("date", "time", "日期", "时间", "年月", "月份")):
                sample = series.dropna().astype(str).head(200)
                if not sample.empty and pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.8:
                    datetime_columns.append(column)

        categorical: List[str] = []
        for column in columns:
            if column in numeric or column in datetime_columns:
                continue
            unique = int(working[column].nunique(dropna=True))
            if unique <= max(50, int(math.sqrt(max(len(working), 1))) + 1):
                categorical.append(column)

        longitude = next((column for column in numeric if named_like(column, ("longitude", "经度", "lng", "lon"))), None)
        latitude = next((column for column in numeric if named_like(column, ("latitude", "纬度", "lat"))), None)
        x_coordinate = next((column for column in numeric if lowered[column] in {"x", "x坐标", "横坐标"}), None)
        y_coordinate = next((column for column in numeric if lowered[column] in {"y", "y坐标", "纵坐标"}), None)
        coordinate_columns = [longitude, latitude] if longitude and latitude else (
            [x_coordinate, y_coordinate] if x_coordinate and y_coordinate else []
        )
        coordinate_columns = [column for column in coordinate_columns if column]
        measures = [
            column for column in numeric
            if column not in identifier_candidates and column not in coordinate_columns
        ]
        missing_numeric = [column for column in measures if working[column].isna().any()]
        missing_other = [column for column in columns if column not in numeric and working[column].isna().any()]
        unique_non_coordinate = [
            column for column in columns
            if column not in coordinate_columns
            and not working[column].isna().any()
            and working[column].nunique(dropna=True) == len(working)
        ]
        entity = next((column for column in identifier_candidates if column in unique_non_coordinate), None)
        entity = entity or (unique_non_coordinate[0] if unique_non_coordinate else None)
        group = categorical[0] if categorical else None
        time_column = datetime_columns[0] if datetime_columns else None

        templates: Dict[str, Dict[str, Any]] = {}
        states: Dict[str, tuple[str, str]] = {}

        def bind(name: str, template: Optional[Dict[str, Any]], reason: str, review: bool = False) -> None:
            if template is None:
                states[name] = ("unavailable", reason)
            else:
                templates[name] = template
                states[name] = ("review" if review else "ready", reason)

        bind("select_columns", {"columns": columns[: min(5, len(columns))]}, "已绑定当前表字段")
        bind("drop_columns", {"columns": [columns[-1]]} if len(columns) > 1 else None, "至少需要保留一个字段", review=True)
        bind("rename_columns", {"mapping": {columns[0]: f"{columns[0]}_重命名"}}, "已绑定首个字段；请核验新名称", review=True)
        if measures:
            threshold = pd.to_numeric(working[measures[0]], errors="coerce").median()
            threshold = float(threshold) if pd.notna(threshold) else 0.0
            filter_template = {"combine": "and", "conditions": [{"column": measures[0], "operator": "ge", "value": threshold}]}
        elif categorical:
            mode = working[categorical[0]].mode(dropna=True)
            filter_template = {"combine": "and", "conditions": [{"column": categorical[0], "operator": "eq", "value": None if mode.empty else str(mode.iloc[0])}]}
        else:
            filter_template = {"combine": "and", "conditions": [{"column": columns[0], "operator": "not_null", "value": None}]}
        bind("filter_rows", filter_template, "已绑定当前字段与可执行筛选值", review=True)
        order_column = time_column or (measures[0] if measures else columns[0])
        bind("sort_rows", {"by": [order_column], "ascending": [True]}, "已绑定可排序字段")
        bind("deduplicate", {"subset": [entity] if entity else columns, "keep": "first"}, "优先按唯一实体键去重；没有实体键时按整行去重", review=True)
        type_mapping: Dict[str, str] = {}
        if time_column:
            type_mapping[time_column] = "datetime"
        if measures:
            type_mapping[measures[0]] = "numeric"
        if not type_mapping:
            type_mapping[columns[0]] = "text"
        bind("convert_types", {"mapping": type_mapping, "errors": "coerce"}, "已根据当前字段画像绑定类型")
        fill_columns = missing_numeric or missing_other
        fill_strategy = "median" if missing_numeric else "mode"
        bind(
            "fill_missing",
            {"columns": fill_columns[:20], "strategy": fill_strategy, "group_by": []} if fill_columns else None,
            f"已绑定{len(fill_columns)}个存在缺失的字段" if fill_columns else "当前表未检测到缺失值",
        )
        if len(measures) >= 2:
            left, right = measures[:2]
            expression = f'col("{left}") / col("{right}")'
            bind("derive_columns", {"expressions": {f"{left}_比_{right}": expression}}, "公式只保证语法可执行，业务含义与除零边界仍需核验", review=True)
        else:
            bind("derive_columns", None, "至少需要两个非编码数值度量才能生成通用公式")
        if group and measures:
            bind("aggregate", {"group_by": [group], "aggregations": [{"column": column, "function": "sum", "output": f"{column}_sum"} for column in measures[:3]]}, "已绑定分组维度和非编码数值度量", review=True)
        else:
            bind("aggregate", None, "需要至少一个分组维度和一个非编码数值度量")
        if group and measures:
            pivot_index = [time_column] if time_column else ([entity] if entity else [group])
            pivot_column = next((column for column in categorical if column not in pivot_index), None)
            bind("pivot", {"index": pivot_index, "columns": [pivot_column], "values": [measures[0]], "aggfunc": "sum", "fill_value": 0} if pivot_column else None, "已绑定行、列和值字段；需核验聚合含义" if pivot_column else "需要两个可区分维度和一个数值度量", review=True)
        else:
            bind("pivot", None, "需要两个可区分维度和一个数值度量")
        bind("melt", {"id_vars": [entity] if entity else [], "value_vars": measures[:3], "var_name": "指标", "value_name": "数值"} if len(measures) >= 2 else None, "已绑定实体键与多个度量；需确认它们确为同类指标" if len(measures) >= 2 else "至少需要两个非编码数值度量", review=True)
        bind("time_features", {"time_column": time_column, "features": ["year", "quarter", "month", "dayofweek", "is_weekend"]} if time_column else None, f"已绑定时间字段“{time_column}”" if time_column else "当前表未识别到时间字段")
        bind("resample_time", {"time_column": time_column, "frequency": "D", "group_by": [group] if group else [], "aggregations": [{"column": measures[0], "function": "sum", "output": f"{measures[0]}_周期合计"}]} if time_column and measures else None, "已绑定时间和度量；需核验日粒度及可加性" if time_column and measures else "需要时间字段和非编码数值度量", review=True)
        bind("window_features", {"order_by": time_column, "partition_by": [group] if group else [], "value_columns": measures[:3], "features": [{"kind": "lag", "periods": 1}, {"kind": "rolling_mean", "window": 7, "shift": 1}]} if time_column and measures else None, "已绑定排序、分组与度量字段；需核验7期窗口含义" if time_column and measures else "需要时间字段和非编码数值度量", review=True)
        bind("normalize", {"columns": measures[:10], "method": "zscore", "suffix": "_z"} if measures else None, f"已绑定{len(measures[:10])}个非编码数值度量" if measures else "当前表没有可缩放的非编码数值度量")
        bind("bin_numeric", {"column": measures[0], "method": "quantile", "bins": 5, "output": f"{measures[0]}_分组"} if measures else None, f"已绑定数值度量“{measures[0]}”" if measures else "当前表没有可分箱的非编码数值度量")
        bind("encode_categorical", {"columns": categorical[:3], "method": "frequency", "max_categories": 30} if categorical else None, f"已绑定{len(categorical[:3])}个低基数类别字段" if categorical else "当前表未识别到低基数类别字段", review=True)
        metric = "haversine" if longitude and latitude else "euclidean"
        pair_template = {"id_column": entity, "coordinate_columns": coordinate_columns, "metric": metric, "directed": False, "include_self": False, "max_pairs": 1_000_000} if entity and len(coordinate_columns) == 2 else None
        bind("pairwise_distance", pair_template, f"已绑定实体键“{entity}”与坐标字段；需核验坐标系和单位" if pair_template else "需要唯一实体键以及经纬度或明确命名的X/Y坐标字段", review=True)

        bound: List[Dict[str, Any]] = []
        for item in capabilities:
            availability, reason = states.get(item["name"], ("unavailable", "当前数据结构无法安全绑定该操作"))
            current = {**item, "availability": availability, "availability_reason": reason}
            if item["name"] in templates:
                current["template"] = templates[item["name"]]
            elif availability == "unavailable":
                current["template"] = None
            bound.append(current)
        return bound

    def execute(self, frame: pd.DataFrame, pipeline: Sequence[Mapping[str, Any]]) -> TransformationResult:
        if not isinstance(frame, pd.DataFrame):
            raise TableTransformError("输入必须是二维表")
        if not isinstance(pipeline, (list, tuple)) or not pipeline:
            raise TableTransformError("处理流水线不能为空")
        if len(pipeline) > self.MAX_STEPS:
            raise TableTransformError(f"单次最多执行{self.MAX_STEPS}个处理步骤")

        original_shape = tuple(frame.shape)
        current = frame.copy(deep=False)
        warnings: List[str] = []
        labels = [str(col) for col in current.columns]
        normalized = _deduplicate_labels(labels)
        if normalized != labels:
            current = current.copy(deep=False)
            current.columns = normalized
            warnings.append("检测到重复字段名，已追加__2、__3后缀以保证字段可寻址。")
        elif list(current.columns) != labels:
            current = current.copy(deep=False)
            current.columns = labels
            warnings.append("非字符串字段名已转换为字符串。")

        self._check_budget(current, baseline_rows=max(len(frame), 1))
        audit: List[Dict[str, Any]] = []
        for index, raw_step in enumerate(pipeline, start=1):
            if not isinstance(raw_step, Mapping):
                raise TableTransformError(f"第{index}步不是有效对象")
            operation = raw_step.get("operation", raw_step.get("op"))
            params = raw_step.get("params", {})
            if operation not in self._operations:
                raise TableTransformError(f"第{index}步操作不受支持: {operation}")
            if not isinstance(params, dict):
                raise TableTransformError(f"第{index}步params必须是对象")

            before_shape = tuple(current.shape)
            before_columns = list(current.columns)
            started = time.perf_counter()
            try:
                transformed = self._operations[str(operation)](current, dict(params))
            except TableTransformError as exc:
                raise TableTransformError(f"第{index}步 {operation}：{exc}") from exc
            except Exception as exc:
                raise TableTransformError(f"第{index}步 {operation} 执行失败: {exc}") from exc
            if not isinstance(transformed, pd.DataFrame):
                raise TableTransformError(f"第{index}步没有返回二维表")
            self._check_budget(
                transformed,
                baseline_rows=max(len(frame), 1),
                check_expansion=str(operation) not in {"melt", "pairwise_distance"},
            )
            elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
            after_columns = list(map(str, transformed.columns))
            audit.append({
                "step": index,
                "operation": str(operation),
                "input_shape": list(before_shape),
                "output_shape": list(transformed.shape),
                "columns_added": [col for col in after_columns if col not in before_columns],
                "columns_removed": [str(col) for col in before_columns if str(col) not in after_columns],
                "elapsed_ms": elapsed_ms,
            })
            current = transformed

        return TransformationResult(
            data=current,
            audit=audit,
            warnings=warnings,
            input_shape=original_shape,
        )

    def _check_budget(self, frame: pd.DataFrame, *, baseline_rows: int, check_expansion: bool = True) -> None:
        rows, columns = frame.shape
        if columns > self.max_columns:
            raise TableTransformError(f"结果字段数{columns:,}超过安全上限{self.max_columns:,}，请减少透视/独热维度")
        cells = int(rows) * int(columns)
        if cells > self.max_result_cells:
            raise TableTransformError(f"结果规模约{cells:,}个单元格，超过安全上限{self.max_result_cells:,}")
        if check_expansion and rows > baseline_rows * self.max_expansion_ratio:
            raise TableTransformError(
                f"结果行数相对输入膨胀超过{self.max_expansion_ratio:g}倍，请缩小melt/展开范围"
            )

    @staticmethod
    def _require_columns(frame: pd.DataFrame, columns: Iterable[Any], parameter: str = "columns") -> List[str]:
        result = [str(column) for column in columns]
        missing = [column for column in result if column not in frame.columns]
        if missing:
            raise TableTransformError(f"{parameter}包含不存在的字段: {missing}")
        return result

    def _op_select_columns(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        columns = self._require_columns(frame, _as_list(params.get("columns"), "columns", allow_empty=False))
        if len(set(columns)) != len(columns):
            raise TableTransformError("选择字段不能重复")
        return frame.loc[:, columns].copy(deep=False)

    def _op_drop_columns(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        columns = self._require_columns(frame, _as_list(params.get("columns"), "columns", allow_empty=False))
        result = frame.drop(columns=columns)
        if result.shape[1] == 0:
            raise TableTransformError("不能删除全部字段")
        return result

    def _op_rename_columns(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        mapping = params.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise TableTransformError("mapping必须是非空的旧字段到新字段映射")
        self._require_columns(frame, mapping.keys(), "mapping")
        normalized = {str(old): str(new).strip() for old, new in mapping.items()}
        if any(not value for value in normalized.values()):
            raise TableTransformError("新字段名不能为空")
        output_columns = [normalized.get(str(col), str(col)) for col in frame.columns]
        if len(set(output_columns)) != len(output_columns):
            raise TableTransformError("重命名会产生重复字段")
        return frame.rename(columns=normalized)

    def _op_filter_rows(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        conditions = params.get("conditions")
        if not isinstance(conditions, list) or not conditions or len(conditions) > 50:
            raise TableTransformError("conditions必须包含1到50个条件")
        combine = str(params.get("combine", "and")).lower()
        if combine not in {"and", "or"}:
            raise TableTransformError("combine只能是and或or")
        masks: List[pd.Series] = []
        for condition in conditions:
            if not isinstance(condition, dict):
                raise TableTransformError("每个筛选条件必须是对象")
            column = str(condition.get("column", ""))
            self._require_columns(frame, [column], "筛选条件")
            masks.append(self._condition_mask(frame[column], str(condition.get("operator", "eq")), condition.get("value")))
        mask = masks[0]
        for item in masks[1:]:
            mask = (mask & item) if combine == "and" else (mask | item)
        return frame.loc[mask.fillna(False)].copy(deep=False)

    @staticmethod
    def _coerce_condition_value(series: pd.Series, value: Any) -> Any:
        if pd.api.types.is_datetime64_any_dtype(series):
            if isinstance(value, list):
                return [pd.to_datetime(item) for item in value]
            return pd.to_datetime(value)
        if pd.api.types.is_numeric_dtype(series):
            if isinstance(value, (list, tuple)):
                return pd.to_numeric(pd.Series(list(value)), errors="raise").tolist()
            return pd.to_numeric(value)
        return value

    def _condition_mask(self, series: pd.Series, operation: str, value: Any) -> pd.Series:
        operation = operation.lower()
        if operation == "is_null":
            return series.isna()
        if operation == "not_null":
            return series.notna()
        if operation in {"contains", "not_contains", "startswith", "endswith"}:
            if value is None or len(str(value)) > 500:
                raise TableTransformError("文本筛选值不能为空且最多500字符")
            text = series.astype("string")
            if operation in {"contains", "not_contains"}:
                result = text.str.contains(str(value), regex=False, na=False)
                return ~result if operation == "not_contains" else result
            return text.str.startswith(str(value), na=False) if operation == "startswith" else text.str.endswith(str(value), na=False)
        if operation in {"in", "not_in"}:
            values = _as_list(value, "筛选值", allow_empty=False)
            if len(values) > 10_000:
                raise TableTransformError("in筛选值最多10000项")
            result = series.isin(values)
            return ~result if operation == "not_in" else result
        if operation == "between":
            values = _as_list(value, "区间", allow_empty=False)
            if len(values) != 2:
                raise TableTransformError("between需要两个边界值")
            values = self._coerce_condition_value(series, values)
            return series.between(values[0], values[1], inclusive="both")
        value = self._coerce_condition_value(series, value)
        functions = {
            "eq": series.eq, "ne": series.ne, "gt": series.gt, "ge": series.ge,
            "lt": series.lt, "le": series.le,
        }
        if operation not in functions:
            raise TableTransformError(f"筛选运算符不受支持: {operation}")
        return functions[operation](value)

    def _op_sort_rows(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        columns = self._require_columns(frame, _as_list(params.get("by"), "by", allow_empty=False), "by")
        ascending = params.get("ascending", True)
        if isinstance(ascending, list) and len(ascending) != len(columns):
            raise TableTransformError("ascending列表长度必须与by一致")
        return frame.sort_values(columns, ascending=ascending, kind="mergesort", na_position=str(params.get("na_position", "last"))).reset_index(drop=True)

    def _op_deduplicate(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        subset_raw = _as_list(params.get("subset"), "subset")
        subset = self._require_columns(frame, subset_raw, "subset") if subset_raw else None
        keep = params.get("keep", "first")
        if keep not in {"first", "last", False}:
            raise TableTransformError("keep只能是first、last或false")
        return frame.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)

    def _op_convert_types(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        mapping = params.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise TableTransformError("mapping必须指定字段及目标类型")
        self._require_columns(frame, mapping.keys(), "mapping")
        errors = str(params.get("errors", "coerce"))
        if errors not in {"coerce", "raise"}:
            raise TableTransformError("errors只能是coerce或raise")
        result = frame.copy(deep=False)
        for column, target in mapping.items():
            column, target = str(column), str(target).lower()
            series = result[column]
            if target in {"numeric", "float"}:
                result[column] = pd.to_numeric(series, errors=errors)
            elif target == "integer":
                converted = pd.to_numeric(series, errors=errors)
                result[column] = converted.astype("Int64")
            elif target == "datetime":
                result[column] = pd.to_datetime(series, errors=errors)
            elif target in {"string", "text"}:
                result[column] = series.astype("string")
            elif target == "category":
                result[column] = series.astype("category")
            elif target == "boolean":
                if pd.api.types.is_bool_dtype(series):
                    result[column] = series.astype("boolean")
                else:
                    truth = {"1": True, "true": True, "yes": True, "y": True, "是": True,
                             "0": False, "false": False, "no": False, "n": False, "否": False}
                    converted = series.astype("string").str.strip().str.lower().map(truth)
                    if errors == "raise" and converted.isna().sum() > series.isna().sum():
                        raise TableTransformError(f"字段{column}包含无法转换的布尔值")
                    result[column] = converted.astype("boolean")
            else:
                raise TableTransformError(f"不支持的目标类型: {target}")
        return result

    def _op_fill_missing(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        columns = self._require_columns(frame, _as_list(params.get("columns"), "columns", allow_empty=False))
        strategy = str(params.get("strategy", "median")).lower()
        group_by = self._require_columns(frame, _as_list(params.get("group_by"), "group_by"), "group_by")
        result = frame.copy(deep=False)
        if strategy == "drop_rows":
            return result.dropna(subset=columns).reset_index(drop=True)
        for column in columns:
            series = result[column]
            if strategy == "constant":
                if "value" not in params:
                    raise TableTransformError("constant填补需要value")
                result[column] = series.fillna(params["value"])
            elif strategy in {"mean", "median"}:
                if not pd.api.types.is_numeric_dtype(series):
                    raise TableTransformError(f"{strategy}只能用于数值字段: {column}")
                values = result.groupby(group_by, dropna=False, observed=True)[column].transform(strategy) if group_by else getattr(series, strategy)()
                result[column] = series.fillna(values)
            elif strategy == "mode":
                if group_by:
                    values = result.groupby(group_by, dropna=False, observed=True)[column].transform(
                        lambda item: item.mode(dropna=True).iloc[0] if not item.mode(dropna=True).empty else np.nan
                    )
                    result[column] = series.fillna(values)
                else:
                    mode = series.mode(dropna=True)
                    if not mode.empty:
                        result[column] = series.fillna(mode.iloc[0])
            elif strategy in {"forward", "backward"}:
                if group_by:
                    grouped = result.groupby(group_by, dropna=False, observed=True)[column]
                    result[column] = grouped.ffill() if strategy == "forward" else grouped.bfill()
                else:
                    result[column] = series.ffill() if strategy == "forward" else series.bfill()
            elif strategy == "interpolate":
                if not pd.api.types.is_numeric_dtype(series):
                    raise TableTransformError(f"interpolate只能用于数值字段: {column}")
                result[column] = (
                    result.groupby(group_by, dropna=False, observed=True)[column].transform(lambda item: item.interpolate())
                    if group_by else series.interpolate()
                )
            else:
                raise TableTransformError(f"缺失值策略不受支持: {strategy}")
        return result

    def _op_derive_columns(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        expressions = params.get("expressions")
        if not isinstance(expressions, dict) or not expressions or len(expressions) > 50:
            raise TableTransformError("expressions必须包含1到50个新字段表达式")
        result = frame.copy(deep=False)
        for raw_name, expression in expressions.items():
            name = str(raw_name).strip()
            if not name:
                raise TableTransformError("派生字段名不能为空")
            evaluator = _SafeExpressionEvaluator(result)
            values = evaluator.evaluate(str(expression))
            if isinstance(values, (pd.Series, np.ndarray, list)) and len(values) != len(result):
                raise TableTransformError(f"派生字段{name}长度与原表不一致")
            result[name] = values
            if pd.api.types.is_numeric_dtype(result[name]):
                result[name] = result[name].replace([np.inf, -np.inf], np.nan)
        return result

    def _parse_aggregations(self, frame: pd.DataFrame, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        specs = params.get("aggregations")
        if not isinstance(specs, list) or not specs or len(specs) > 100:
            raise TableTransformError("aggregations必须包含1到100个汇总规则")
        normalized: List[Dict[str, Any]] = []
        outputs: set = set()
        for raw in specs:
            if not isinstance(raw, dict):
                raise TableTransformError("每个汇总规则必须是对象")
            column = str(raw.get("column", ""))
            function = str(raw.get("function", "")).lower()
            if column != "*":
                self._require_columns(frame, [column], "aggregations")
            output = str(raw.get("output") or f"{column}_{function}")
            if not output or output in outputs:
                raise TableTransformError(f"汇总输出字段重复或为空: {output}")
            if function not in self._STANDARD_AGGREGATIONS | {"quantile", "weighted_mean", "share_of_total", "size"}:
                raise TableTransformError(f"汇总函数不受支持: {function}")
            if column == "*" and function != "size":
                raise TableTransformError("column='*'只允许与size汇总函数一起使用")
            outputs.add(output)
            normalized.append({**raw, "column": column, "function": function, "output": output})
        return normalized

    def _op_aggregate(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        group_by = self._require_columns(frame, _as_list(params.get("group_by"), "group_by"), "group_by")
        specs = self._parse_aggregations(frame, params)
        dropna = bool(params.get("dropna", False))
        sort = bool(params.get("sort", True))
        if not group_by:
            row: Dict[str, Any] = {}
            for spec in specs:
                column, function, output = spec["column"], spec["function"], spec["output"]
                series = frame[column] if column != "*" else None
                if function == "size":
                    row[output] = len(frame)
                elif function == "quantile":
                    q = float(spec.get("q", 0.5))
                    if not 0 <= q <= 1:
                        raise TableTransformError("quantile的q必须在0到1之间")
                    row[output] = series.quantile(q)
                elif function == "weighted_mean":
                    weight = str(spec.get("weight", ""))
                    self._require_columns(frame, [weight], "weight")
                    valid = series.notna() & frame[weight].notna()
                    denominator = frame.loc[valid, weight].sum()
                    row[output] = (series[valid] * frame.loc[valid, weight]).sum() / denominator if denominator != 0 else np.nan
                elif function == "share_of_total":
                    row[output] = 1.0 if series.notna().any() else np.nan
                else:
                    row[output] = getattr(series, function)()
            return pd.DataFrame([row])

        grouped = frame.groupby(group_by, dropna=dropna, sort=sort, observed=True)
        standard_specs = [spec for spec in specs if spec["function"] in self._STANDARD_AGGREGATIONS]
        if standard_specs:
            named = {
                spec["output"]: pd.NamedAgg(column=spec["column"], aggfunc=spec["function"])
                for spec in standard_specs
            }
            result = grouped.agg(**named)
        else:
            result = grouped.size().rename("__row_count").to_frame().drop(columns="__row_count")

        for spec in specs:
            column, function, output = spec["column"], spec["function"], spec["output"]
            if function in self._STANDARD_AGGREGATIONS:
                continue
            if function == "size":
                result[output] = grouped.size()
            elif function == "quantile":
                q = float(spec.get("q", 0.5))
                if not 0 <= q <= 1:
                    raise TableTransformError("quantile的q必须在0到1之间")
                result[output] = grouped[column].quantile(q)
            elif function == "share_of_total":
                totals = grouped[column].sum(min_count=1)
                denominator = totals.sum()
                result[output] = totals / denominator if denominator != 0 else np.nan
            elif function == "weighted_mean":
                weight = str(spec.get("weight", ""))
                self._require_columns(frame, [weight], "weight")
                temporary = frame[group_by + [column, weight]].copy()
                temporary["__weighted_value"] = temporary[column] * temporary[weight]
                weighted = temporary.groupby(group_by, dropna=dropna, sort=sort, observed=True)[["__weighted_value", weight]].sum(min_count=1)
                result[output] = weighted["__weighted_value"] / weighted[weight].replace(0, np.nan)
        return result.reset_index()

    def _op_pivot(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        index = self._require_columns(frame, _as_list(params.get("index"), "index", allow_empty=False), "index")
        columns = self._require_columns(frame, _as_list(params.get("columns"), "columns", allow_empty=False), "columns")
        values = self._require_columns(frame, _as_list(params.get("values"), "values", allow_empty=False), "values")
        aggfunc = str(params.get("aggfunc", "sum")).lower()
        if aggfunc not in self._STANDARD_AGGREGATIONS - {"first", "last", "prod"}:
            raise TableTransformError(f"透视汇总函数不受支持: {aggfunc}")
        index_groups = max(1, frame[index].drop_duplicates().shape[0])
        column_groups = max(1, frame[columns].drop_duplicates().shape[0])
        estimated_columns = len(index) + column_groups * len(values)
        if estimated_columns > self.max_columns or index_groups * estimated_columns > self.max_result_cells:
            raise TableTransformError(
                f"透视预计产生{index_groups:,}行×{estimated_columns:,}列，超过安全预算；请先筛选或聚合类别"
            )
        result = pd.pivot_table(
            frame,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc,
            fill_value=params.get("fill_value"),
            dropna=bool(params.get("dropna", True)),
            observed=True,
        ).reset_index()
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = _deduplicate_labels(
                "__".join(str(part) for part in label if str(part) not in {"", "None"})
                for label in result.columns
            )
        else:
            result.columns = _deduplicate_labels(result.columns)
        return result

    def _op_melt(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        id_vars = self._require_columns(frame, _as_list(params.get("id_vars"), "id_vars"), "id_vars")
        value_vars = self._require_columns(frame, _as_list(params.get("value_vars"), "value_vars", allow_empty=False), "value_vars")
        if set(id_vars) & set(value_vars):
            raise TableTransformError("id_vars与value_vars不能重叠")
        estimated_rows = len(frame) * len(value_vars)
        estimated_columns = len(id_vars) + 2
        if estimated_rows * estimated_columns > self.max_result_cells:
            raise TableTransformError(f"宽转长预计产生{estimated_rows:,}行，超过安全预算")
        return frame.melt(
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=str(params.get("var_name", "variable")),
            value_name=str(params.get("value_name", "value")),
        )

    def _op_time_features(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        column = str(params.get("time_column", ""))
        self._require_columns(frame, [column], "time_column")
        features = _as_list(params.get("features", ["year", "month", "dayofweek"]), "features", allow_empty=False)
        supported = {"year", "quarter", "month", "week", "day", "dayofyear", "dayofweek", "hour", "minute", "is_weekend", "ordinal", "month_sin", "month_cos", "weekday_sin", "weekday_cos"}
        invalid = [feature for feature in features if feature not in supported]
        if invalid:
            raise TableTransformError(f"不支持的时间特征: {invalid}")
        dates = pd.to_datetime(frame[column], errors=str(params.get("errors", "coerce")))
        result = frame.copy(deep=False)
        prefix = str(params.get("prefix") or column)
        values: Dict[str, Any] = {
            "year": dates.dt.year,
            "quarter": dates.dt.quarter,
            "month": dates.dt.month,
            "week": dates.dt.isocalendar().week.astype("Int64"),
            "day": dates.dt.day,
            "dayofyear": dates.dt.dayofyear,
            "dayofweek": dates.dt.dayofweek,
            "hour": dates.dt.hour,
            "minute": dates.dt.minute,
            "is_weekend": dates.dt.dayofweek.ge(5).astype("boolean"),
            "ordinal": pd.Series(dates.map(lambda value: value.toordinal() if pd.notna(value) else pd.NA), index=frame.index, dtype="Int64"),
            "month_sin": np.sin(2 * np.pi * dates.dt.month / 12),
            "month_cos": np.cos(2 * np.pi * dates.dt.month / 12),
            "weekday_sin": np.sin(2 * np.pi * dates.dt.dayofweek / 7),
            "weekday_cos": np.cos(2 * np.pi * dates.dt.dayofweek / 7),
        }
        for feature in features:
            result[f"{prefix}__{feature}"] = values[str(feature)]
        return result

    def _op_resample_time(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        time_column = str(params.get("time_column", ""))
        self._require_columns(frame, [time_column], "time_column")
        group_by = self._require_columns(frame, _as_list(params.get("group_by"), "group_by"), "group_by")
        frequency = str(params.get("frequency", "D")).strip()
        if not re.fullmatch(r"(?:[1-9]\d*)?(?:min|h|D|W|M|Q|Y)", frequency):
            raise TableTransformError("frequency仅支持分钟、小时、日、周、月、季度或年，例如15min、h、D、W、M、Q")
        output_column = str(params.get("output_time_column") or time_column)
        if output_column in group_by:
            raise TableTransformError("输出时间字段不能与group_by重复")
        work = frame.copy(deep=False)
        dates = pd.to_datetime(work[time_column], errors=str(params.get("errors", "coerce")))
        valid = dates.notna()
        work = work.loc[valid].copy(deep=False)
        dates = dates.loc[valid]
        if work.empty:
            raise TableTransformError("时间字段转换后没有有效观测")
        try:
            if frequency.endswith(("min", "h", "D")):
                work[output_column] = dates.dt.floor(frequency)
            else:
                work[output_column] = dates.dt.to_period(frequency).dt.start_time
        except (TypeError, ValueError) as exc:
            raise TableTransformError(f"时间频率无效: {frequency}") from exc
        if output_column != time_column:
            work = work.drop(columns=[time_column])
        aggregate_params = {
            "group_by": group_by + [output_column],
            "aggregations": params.get("aggregations"),
            "dropna": bool(params.get("dropna", False)),
            "sort": True,
        }
        return self._op_aggregate(work, aggregate_params)

    def _op_window_features(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        order_by = str(params.get("order_by", ""))
        self._require_columns(frame, [order_by], "order_by")
        partition_by = self._require_columns(frame, _as_list(params.get("partition_by"), "partition_by"), "partition_by")
        value_columns = self._require_columns(frame, _as_list(params.get("value_columns"), "value_columns", allow_empty=False), "value_columns")
        features = params.get("features")
        if not isinstance(features, list) or not features or len(features) > 30:
            raise TableTransformError("features必须包含1到30个窗口规则")
        if len(value_columns) * len(features) > 100:
            raise TableTransformError("单步最多生成100个窗口字段")
        result = frame.sort_values(partition_by + [order_by], kind="mergesort").reset_index(drop=True).copy(deep=False)
        for column in value_columns:
            if not pd.api.types.is_numeric_dtype(result[column]):
                raise TableTransformError(f"窗口值字段必须为数值型: {column}")
            grouped = result.groupby(partition_by, dropna=False, observed=True, sort=False)[column] if partition_by else None
            for spec in features:
                if not isinstance(spec, dict):
                    raise TableTransformError("窗口规则必须是对象")
                kind = str(spec.get("kind", "lag"))
                periods = int(spec.get("periods", 1))
                if periods < 1 or periods > 100_000:
                    raise TableTransformError("periods必须在1到100000之间")
                suffix = str(spec.get("output") or "")
                if kind == "lag":
                    values = grouped.shift(periods) if grouped is not None else result[column].shift(periods)
                    suffix = suffix or f"{column}__lag_{periods}"
                elif kind == "diff":
                    values = grouped.diff(periods) if grouped is not None else result[column].diff(periods)
                    suffix = suffix or f"{column}__diff_{periods}"
                elif kind == "pct_change":
                    values = grouped.pct_change(periods=periods, fill_method=None) if grouped is not None else result[column].pct_change(periods=periods, fill_method=None)
                    suffix = suffix or f"{column}__pct_change_{periods}"
                elif kind in {"cumsum", "cummax", "cummin", "rank"}:
                    if kind == "rank":
                        values = grouped.rank(method=str(spec.get("method", "average")), pct=bool(spec.get("pct", False))) if grouped is not None else result[column].rank(method=str(spec.get("method", "average")), pct=bool(spec.get("pct", False)))
                    else:
                        values = getattr(grouped, kind)() if grouped is not None else getattr(result[column], kind)()
                    suffix = suffix or f"{column}__{kind}"
                elif kind.startswith("rolling_"):
                    statistic = kind.removeprefix("rolling_")
                    if statistic not in {"mean", "sum", "std", "min", "max", "median"}:
                        raise TableTransformError(f"滚动统计不受支持: {statistic}")
                    window = int(spec.get("window", 7))
                    shift = int(spec.get("shift", 1))
                    min_periods = int(spec.get("min_periods", 1))
                    if window < 1 or window > 100_000 or shift < 0:
                        raise TableTransformError("window必须在1到100000之间且shift不能为负")
                    transform = lambda item, stat=statistic: getattr(item.shift(shift).rolling(window=window, min_periods=min_periods), stat)()
                    values = grouped.transform(transform) if grouped is not None else transform(result[column])
                    suffix = suffix or f"{column}__rolling_{statistic}_{window}_s{shift}"
                else:
                    raise TableTransformError(f"窗口特征不受支持: {kind}")
                result[suffix] = pd.Series(values, index=result.index).replace([np.inf, -np.inf], np.nan)
        return result

    def _op_normalize(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        columns = self._require_columns(frame, _as_list(params.get("columns"), "columns", allow_empty=False))
        method = str(params.get("method", "zscore")).lower()
        suffix = str(params.get("suffix", f"_{method}"))
        result = frame.copy(deep=False)
        for column in columns:
            series = pd.to_numeric(result[column], errors="coerce")
            if method == "zscore":
                scale = series.std(ddof=0)
                values = (series - series.mean()) / scale if scale and pd.notna(scale) else 0.0
            elif method == "minmax":
                scale = series.max() - series.min()
                values = (series - series.min()) / scale if scale and pd.notna(scale) else 0.0
            elif method == "robust":
                q1, q3 = series.quantile([0.25, 0.75])
                scale = q3 - q1
                values = (series - series.median()) / scale if scale and pd.notna(scale) else 0.0
            elif method == "log1p":
                values = np.log1p(series.where(series >= -1))
            else:
                raise TableTransformError(f"数值变换方法不受支持: {method}")
            result[f"{column}{suffix}"] = values
        return result

    def _op_bin_numeric(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        column = str(params.get("column", ""))
        self._require_columns(frame, [column], "column")
        method = str(params.get("method", "quantile"))
        bins = params.get("bins", 5)
        output = str(params.get("output") or f"{column}__bin")
        result = frame.copy(deep=False)
        if method == "quantile":
            count = int(bins)
            if count < 2 or count > 100:
                raise TableTransformError("等频分箱数必须在2到100之间")
            result[output] = pd.qcut(result[column], q=count, duplicates="drop")
        elif method == "equal_width":
            count = int(bins)
            if count < 2 or count > 100:
                raise TableTransformError("等宽分箱数必须在2到100之间")
            result[output] = pd.cut(result[column], bins=count)
        elif method == "custom":
            edges = [float(value) for value in _as_list(bins, "bins", allow_empty=False)]
            if len(edges) < 3 or edges != sorted(set(edges)):
                raise TableTransformError("自定义分箱边界至少3个且必须严格递增")
            result[output] = pd.cut(result[column], bins=edges, include_lowest=True)
        else:
            raise TableTransformError(f"分箱方法不受支持: {method}")
        return result

    def _op_encode_categorical(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        columns = self._require_columns(frame, _as_list(params.get("columns"), "columns", allow_empty=False))
        method = str(params.get("method", "frequency"))
        max_categories = int(params.get("max_categories", 30))
        if max_categories < 2 or max_categories > 500:
            raise TableTransformError("max_categories必须在2到500之间")
        result = frame.copy(deep=False)
        if method == "frequency":
            denominator = max(len(result), 1)
            for column in columns:
                frequencies = result[column].value_counts(dropna=False) / denominator
                result[f"{column}__frequency"] = result[column].map(frequencies).astype(float)
            return result
        if method == "one_hot":
            working = result.copy(deep=False)
            for column in columns:
                counts = working[column].value_counts(dropna=False)
                keep = set(counts.head(max_categories).index)
                working[column] = working[column].where(working[column].isin(keep), "__OTHER__").fillna("__MISSING__")
            encoded = pd.get_dummies(working[columns], prefix=columns, dtype=np.uint8)
            if result.shape[1] - len(columns) + encoded.shape[1] > self.max_columns:
                raise TableTransformError("独热编码会产生过多字段，请降低max_categories或使用frequency")
            return pd.concat([working.drop(columns=columns), encoded], axis=1)
        raise TableTransformError(f"类别编码方法不受支持: {method}")

    def _op_pairwise_distance(self, frame: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        id_column = str(params.get("id_column", ""))
        self._require_columns(frame, [id_column], "id_column")
        coordinate_columns = self._require_columns(
            frame,
            _as_list(params.get("coordinate_columns"), "coordinate_columns", allow_empty=False),
            "coordinate_columns",
        )
        metric = str(params.get("metric", "euclidean")).lower()
        if metric not in {"euclidean", "manhattan", "haversine"}:
            raise TableTransformError("metric只能是euclidean、manhattan或haversine")
        if metric == "haversine" and len(coordinate_columns) != 2:
            raise TableTransformError("haversine需要[经度, 纬度]两个坐标字段")
        if metric != "haversine" and len(coordinate_columns) not in {2, 3}:
            raise TableTransformError("欧氏/曼哈顿距离需要2或3个坐标字段")
        if frame[id_column].isna().any() or frame[id_column].duplicated().any():
            raise TableTransformError("节点ID必须非空且唯一")
        coordinates = frame[coordinate_columns].apply(pd.to_numeric, errors="coerce")
        if coordinates.isna().any().any():
            raise TableTransformError("坐标字段包含空值或非数值")
        n_entities = len(frame)
        directed = bool(params.get("directed", False))
        include_self = bool(params.get("include_self", False))
        pair_count = n_entities * (n_entities - 1) // 2
        if directed:
            pair_count *= 2
        if include_self:
            pair_count += n_entities
        max_pairs = int(params.get("max_pairs", 1_000_000))
        if max_pairs < 1 or max_pairs > 5_000_000:
            raise TableTransformError("max_pairs必须在1到5000000之间")
        if pair_count > max_pairs or pair_count * 3 > self.max_result_cells:
            raise TableTransformError(
                f"{n_entities:,}个节点将产生{pair_count:,}条距离边，超过预算；请先筛选候选节点或调低问题规模"
            )

        upper_i, upper_j = np.triu_indices(n_entities, k=1)
        values = coordinates.to_numpy(dtype=float)
        if metric == "euclidean":
            distance = np.sqrt(np.square(values[upper_i] - values[upper_j]).sum(axis=1))
        elif metric == "manhattan":
            distance = np.abs(values[upper_i] - values[upper_j]).sum(axis=1)
        else:
            longitude = np.radians(values[:, 0])
            latitude = np.radians(values[:, 1])
            delta_lon = longitude[upper_j] - longitude[upper_i]
            delta_lat = latitude[upper_j] - latitude[upper_i]
            a = np.sin(delta_lat / 2) ** 2 + np.cos(latitude[upper_i]) * np.cos(latitude[upper_j]) * np.sin(delta_lon / 2) ** 2
            distance = 2 * float(params.get("earth_radius", 6371.0088)) * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        ids = frame[id_column].to_numpy()
        source_ids, target_ids = ids[upper_i], ids[upper_j]
        if directed:
            source_ids, target_ids, distance = (
                np.concatenate([source_ids, target_ids]),
                np.concatenate([target_ids, source_ids]),
                np.concatenate([distance, distance]),
            )
        if include_self:
            source_ids = np.concatenate([source_ids, ids])
            target_ids = np.concatenate([target_ids, ids])
            distance = np.concatenate([distance, np.zeros(n_entities)])
        result = pd.DataFrame({
            str(params.get("source_column", "source")): source_ids,
            str(params.get("target_column", "target")): target_ids,
            str(params.get("distance_column", "distance")): distance,
        })
        if params.get("max_distance") is not None:
            threshold = float(params["max_distance"])
            result = result[result.iloc[:, 2].le(threshold)].reset_index(drop=True)
        return result


class TableTransformationPlanner:
    """从题目文本和字段画像产生可复核的通用候选流水线。"""

    _TIME_WORDS = ("预测", "趋势", "时序", "时间", "增长", "变化", "周期", "滞后")
    _GROUP_WORDS = ("分组", "各", "每个", "分别", "汇总", "统计", "合计", "平均")

    @classmethod
    def suggest(cls, frame: pd.DataFrame, problem: str = "") -> Dict[str, Any]:
        problem = str(problem or "")
        frame = frame.copy(deep=False)
        frame.columns = _deduplicate_labels(frame.columns)
        numeric = [str(col) for col in frame.select_dtypes(include=np.number).columns]
        categorical: List[str] = []
        datetime_columns: List[str] = [str(col) for col in frame.select_dtypes(include=["datetime", "datetimetz"]).columns]
        for raw_column in frame.columns:
            column = str(raw_column)
            series = frame[raw_column]
            if column in datetime_columns or column in numeric:
                continue
            unique = int(series.nunique(dropna=True))
            if unique <= max(50, int(math.sqrt(max(len(frame), 1))) + 1):
                categorical.append(column)
            lowered = column.lower()
            if any(token in lowered for token in ("date", "time", "日期", "时间", "年月", "月份")):
                sample = series.dropna().astype(str).head(200)
                if not sample.empty and pd.to_datetime(sample, errors="coerce").notna().mean() >= 0.8:
                    datetime_columns.append(column)

        mentioned = [str(col) for col in frame.columns if str(col) and str(col) in problem]
        mentioned_numeric = [column for column in mentioned if column in numeric]
        mentioned_groups = [column for column in mentioned if column in categorical]
        recommendations: List[Dict[str, Any]] = []

        if datetime_columns and (not problem or any(word in problem for word in cls._TIME_WORDS)):
            time_column = next((column for column in datetime_columns if column in mentioned), datetime_columns[0])
            values = mentioned_numeric or numeric[:3]
            pipeline: List[Dict[str, Any]] = [
                {"operation": "convert_types", "params": {"mapping": {time_column: "datetime"}, "errors": "coerce"}},
                {"operation": "sort_rows", "params": {"by": [time_column], "ascending": [True]}},
                {"operation": "time_features", "params": {"time_column": time_column, "features": ["year", "quarter", "month", "dayofweek", "is_weekend", "month_sin", "month_cos"]}},
            ]
            if values:
                pipeline.append({
                    "operation": "window_features",
                    "params": {
                        "order_by": time_column,
                        "partition_by": mentioned_groups[:1],
                        "value_columns": values[:3],
                        "features": [
                            {"kind": "lag", "periods": 1},
                            {"kind": "diff", "periods": 1},
                            {"kind": "pct_change", "periods": 1},
                            {"kind": "rolling_mean", "window": 7, "shift": 1},
                        ],
                    },
                })
            recommendations.append({
                "name": "时序/面板数据基础",
                "reason": "识别到时间字段；先排序并生成无未来泄漏的滞后与滚动特征。",
                "risk": "需确认一行对应的时间粒度，7表示7个观测点而不一定是7天。",
                "pipeline": pipeline,
            })

        if categorical and numeric and (not problem or any(word in problem for word in cls._GROUP_WORDS)):
            groups = mentioned_groups[:3] or categorical[:1]
            measures = mentioned_numeric[:6] or numeric[:3]
            function = "mean" if "平均" in problem or "均值" in problem else "sum"
            recommendations.append({
                "name": "分组指标汇总",
                "reason": "识别到类别维度与数值指标，可构造实体/地区/方案层面的统计表。",
                "risk": "汇总前需确认数值是可加总的流量，还是只能取均值的存量/比率。",
                "pipeline": [{
                    "operation": "aggregate",
                    "params": {
                        "group_by": groups,
                        "aggregations": [
                            {"column": column, "function": function, "output": f"{column}_{function}"}
                            for column in measures
                        ],
                    },
                }],
            })

        missing_numeric = [column for column in numeric if frame[column].isna().any()]
        missing_categorical = [column for column in categorical if frame[column].isna().any()]
        if missing_numeric or missing_categorical:
            pipeline = []
            if missing_numeric:
                pipeline.append({"operation": "fill_missing", "params": {"columns": missing_numeric[:20], "strategy": "median"}})
            if missing_categorical:
                pipeline.append({"operation": "fill_missing", "params": {"columns": missing_categorical[:20], "strategy": "mode"}})
            recommendations.append({
                "name": "缺失值稳健基线",
                "reason": "检测到缺失字段，提供不会因极端值明显偏移的数值中位数基线。",
                "risk": "若缺失并非随机，应先建立缺失指示变量或按实体/时间分组填补。",
                "pipeline": pipeline,
            })

        return {
            "profile": {
                "rows": int(len(frame)),
                "columns": int(frame.shape[1]),
                "numeric": numeric,
                "categorical": categorical,
                "datetime": list(dict.fromkeys(datetime_columns)),
                "mentioned": mentioned,
            },
            "recommendations": recommendations,
        }
