"""Bounded compiler for browser-side, multi-dimensional exploratory charts.

The compiler never mutates the source frame and never sends an unbounded table
to the browser.  It validates semantic encodings, applies vectorized filters,
uses deterministic coverage sampling when necessary, and records whether an
aggregation is exact or exploratory-only.
"""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


class InteractiveVisualizationError(ValueError):
    """Raised when a chart request is ambiguous or unsafe."""


class InteractiveVisualizationCompiler:
    """Compile a dataframe into a small, auditable browser chart payload."""

    SCHEMA_VERSION = "mathmodel.interactive-visualization/v2"
    _CHART_TYPES = {"auto", "scatter", "line", "area", "bar", "parallel"}
    _ENCODING_KEYS = {
        "x", "y", "color", "size", "facet", "animation", "tooltip", "parallel"
    }
    _AGGREGATIONS = {"none", "count", "sum", "mean", "median", "min", "max"}
    _TIME_UNITS = {"none", "day", "week", "month", "quarter", "year"}

    def __init__(
        self,
        *,
        max_profile_rows: int = 10_000,
        max_scan_rows: int = 200_000,
        max_points: int = 15_000,
    ) -> None:
        self.max_profile_rows = max(1_000, int(max_profile_rows))
        self.max_scan_rows = max(5_000, int(max_scan_rows))
        self.max_points = max(500, int(max_points))

    def describe(self, frame: pd.DataFrame) -> Dict[str, Any]:
        """Return bounded field metadata and a deterministic default encoding."""
        started = time.perf_counter()
        prepared = self._prepare_frame(frame)
        profile = self._coverage_sample(prepared, self.max_profile_rows)
        fields: List[Dict[str, Any]] = []
        for column in prepared.columns:
            name = str(column)
            sampled = profile[name]
            kind = self._infer_kind(sampled, name)
            non_missing = int(sampled.notna().sum())
            unique_count = int(sampled.nunique(dropna=True))
            unique_rate = unique_count / max(non_missing, 1)
            semantic_role = self._semantic_role(
                kind, unique_count, unique_rate, non_missing
            )
            item: Dict[str, Any] = {
                "name": name,
                "dtype": str(prepared[name].dtype),
                "kind": kind,
                "semantic_role": semantic_role,
                "unique_count_profiled": unique_count,
                "unique_rate_profiled": round(float(unique_rate), 6),
                "missing_rate_profiled": round(float(sampled.isna().mean()), 6),
                "channel_suitability": {
                    "x": semantic_role in {"measure", "time", "dimension", "label"},
                    "y": semantic_role == "measure",
                    "color": (
                        semantic_role == "measure"
                        or (semantic_role == "dimension" and unique_count <= 16)
                    ),
                    "size": semantic_role == "measure",
                    "facet": semantic_role == "dimension" and unique_count <= 24,
                    "animation": (
                        semantic_role == "time"
                        or (semantic_role == "dimension" and unique_count <= 240)
                    ),
                    "tooltip": True,
                },
            }
            if kind == "numeric":
                values = pd.to_numeric(sampled, errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                if not values.empty:
                    quantiles = values.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
                    item["range"] = [self._number(values.min()), self._number(values.max())]
                    item["quantiles"] = {
                        "p01": self._number(quantiles.loc[0.01]),
                        "p25": self._number(quantiles.loc[0.25]),
                        "p50": self._number(quantiles.loc[0.5]),
                        "p75": self._number(quantiles.loc[0.75]),
                        "p99": self._number(quantiles.loc[0.99]),
                    }
            elif kind == "datetime":
                values = pd.to_datetime(sampled, errors="coerce").dropna()
                if not values.empty:
                    item["range"] = [values.min().isoformat(), values.max().isoformat()]
            else:
                counts = sampled.astype("string").fillna("(缺失)").value_counts().head(50)
                item["levels"] = [
                    {"value": str(value), "count_profiled": int(count)}
                    for value, count in counts.items()
                ]
            fields.append(item)
        recommendation = self._recommend(fields)
        capability = self._capability(fields)
        return {
            "schema_version": self.SCHEMA_VERSION,
            "source_rows": int(len(prepared)),
            "source_columns": int(prepared.shape[1]),
            "profiled_rows": int(len(profile)),
            "profile_scope": (
                "coverage_sample" if len(profile) < len(prepared) else "full_dataset"
            ),
            "fields": fields,
            "recommendation": recommendation,
            "capability": capability,
            "limits": {
                "max_points": self.max_points,
                "max_scan_rows": self.max_scan_rows,
                "max_filters": 12,
                "max_tooltip_fields": 8,
            },
            "timing_ms": round((time.perf_counter() - started) * 1_000, 3),
        }

    def compile(
        self,
        frame: pd.DataFrame,
        specification: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Compile selected encodings, filters and aggregation into JSON rows."""
        started = time.perf_counter()
        prepared = self._prepare_frame(frame)
        spec = self._validate_specification(specification or {}, prepared)
        mask = pd.Series(True, index=prepared.index)
        filter_audit: List[Dict[str, Any]] = []
        for filter_spec in spec["filters"]:
            field = filter_spec["field"]
            kind = filter_spec["kind"]
            before = int(mask.sum())
            if kind == "range":
                values = pd.to_numeric(prepared[field], errors="coerce")
                lower = filter_spec.get("min")
                upper = filter_spec.get("max")
                condition = values.notna()
                if lower is not None:
                    condition &= values >= float(lower)
                if upper is not None:
                    condition &= values <= float(upper)
            elif kind == "time_range":
                values = pd.to_datetime(prepared[field], errors="coerce")
                condition = values.notna()
                if filter_spec.get("min"):
                    condition &= values >= pd.Timestamp(filter_spec["min"])
                if filter_spec.get("max"):
                    condition &= values <= pd.Timestamp(filter_spec["max"])
            else:
                allowed = {str(value) for value in filter_spec.get("values", [])}
                condition = prepared[field].astype("string").fillna("(缺失)").isin(allowed)
            mask &= condition.fillna(False)
            filter_audit.append({
                "field": field,
                "kind": kind,
                "rows_before": before,
                "rows_after": int(mask.sum()),
            })

        filtered_rows = int(mask.sum())
        selected_fields = self._selected_fields(spec["encodings"])
        for field in spec["aggregation"]["group_by"]:
            if field not in selected_fields:
                selected_fields.append(field)
        filtered = prepared.loc[mask, selected_fields]
        scanned = self._coverage_sample(filtered, self.max_scan_rows)
        scan_sampled = len(scanned) < len(filtered)
        aggregation_input = scanned
        color_compaction: Optional[Dict[str, Any]] = None
        if spec["aggregation"]["function"] != "none":
            aggregation_input, color_compaction = self._compact_color_levels(
                scanned, spec["encodings"].get("color"), max_levels=12
            )
        output, encodings, aggregation_audit = self._aggregate(
            aggregation_input, spec
        )
        aggregation_audit["color_compaction"] = color_compaction
        if len(output) > spec["max_points"]:
            output = self._coverage_sample(output, spec["max_points"])
        output_sampled = len(output) < int(aggregation_audit["rows_before_output_limit"])

        chart_type = self._resolve_chart_type(spec["chart_type"], output, encodings)
        if chart_type in {"line", "area"} and encodings.get("x") in output.columns:
            output = output.sort_values(str(encodings["x"]), kind="stable")
        field_types = {
            field: self._infer_kind(output[field], field) for field in output.columns
        }
        records = [
            {
                str(key): self._json_value(value, field_types.get(str(key)))
                for key, value in row.items()
            }
            for row in output.to_dict(orient="records")
        ]
        warnings: List[str] = []
        if scan_sampled:
            warnings.append(
                f"过滤后数据超过扫描预算；当前图基于{len(scanned):,}/{filtered_rows:,}行覆盖样本。"
            )
        if output_sampled:
            warnings.append(
                f"图元超过浏览器预算；已确定性抽取{len(output):,}个图元。"
            )
        if filtered_rows == 0:
            warnings.append("当前筛选条件没有保留任何记录。")
        if aggregation_audit["function"] != "none" and scan_sampled:
            warnings.append("聚合基于覆盖样本，只能用于探索，不是精确总量。")
        if aggregation_audit.get("dropped_encodings"):
            warnings.append(
                "聚合后无法保持逐行通道，已停用："
                + "、".join(aggregation_audit["dropped_encodings"])
            )
        for channel in ("x", "y"):
            field = encodings.get(channel)
            if (
                field
                and field in prepared.columns
                and self._infer_kind(prepared[field], field) == "identifier"
            ):
                warnings.append(
                    f"{channel.upper()}轴“{field}”是编码/标识符，不应解释为连续数量。"
                )
        y_field = encodings.get("y")
        if (
            y_field
            and y_field != "__count__"
            and y_field in output.columns
            and output[y_field].nunique(dropna=True) <= 1
        ):
            warnings.append(f"Y轴“{y_field}”在当前切片中为常数，无法显示变化关系。")
        color_field = encodings.get("color")
        if color_compaction:
            warnings.append(
                f"颜色字段“{color_compaction['field']}”有"
                f"{color_compaction['source_levels']}个层级；已在聚合前保留"
                f"{color_compaction['kept_levels']}个高频层级，其余合并为“其他”。"
            )
        elif color_field and color_field in output.columns:
            color_levels = int(output[color_field].nunique(dropna=True))
            if self._infer_kind(output[color_field], color_field) != "numeric" and color_levels > 12:
                warnings.append(
                    f"颜色字段“{color_field}”有{color_levels}个层级；图中只保留高频层级，其余合并为“其他”。"
                )

        facet = encodings.get("facet")
        animation = encodings.get("animation")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "chart_type": chart_type,
            "encodings": encodings,
            "records": records,
            "field_types": field_types,
            "facet_levels": self._levels(output, facet, 24),
            "animation_levels": self._levels(output, animation, 240, sort=True),
            "numeric_ranges": self._numeric_ranges(output, encodings),
            "audit": {
                "source_rows": int(len(prepared)),
                "filtered_rows": filtered_rows,
                "scanned_rows": int(len(scanned)),
                "output_rows": int(len(output)),
                "scan_scope": "coverage_sample" if scan_sampled else "full_filtered_data",
                "aggregation": aggregation_audit,
                "filters": filter_audit,
                "browser_point_limit": spec["max_points"],
                "source_mutated": False,
            },
            "warnings": warnings,
            "timing_ms": round((time.perf_counter() - started) * 1_000, 3),
        }

    def _compact_color_levels(
        self,
        frame: pd.DataFrame,
        color_field: Optional[str],
        *,
        max_levels: int,
    ) -> tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
        """Collapse long-tail series before aggregation so values stay exact."""
        if not color_field or color_field not in frame.columns:
            return frame, None
        if self._infer_kind(frame[color_field], color_field) == "numeric":
            return frame, None
        values = frame[color_field].astype("string").fillna("(缺失)")
        source_levels = int(values.nunique(dropna=False))
        if source_levels <= max_levels:
            return frame, None
        counts = values.value_counts(dropna=False)
        kept = {str(value) for value in counts.index[: max_levels - 1]}
        compacted = frame.copy()
        compacted[color_field] = values.where(values.isin(kept), "其他")
        return compacted, {
            "field": color_field,
            "source_levels": source_levels,
            "kept_levels": len(kept),
            "output_levels": int(compacted[color_field].nunique(dropna=False)),
            "dropped_records": 0,
        }

    def _validate_specification(
        self, specification: Mapping[str, Any], frame: pd.DataFrame
    ) -> Dict[str, Any]:
        if not isinstance(specification, Mapping):
            raise InteractiveVisualizationError("图形配置必须是对象")
        allowed = {"chart_type", "encodings", "filters", "aggregation", "max_points"}
        unknown = set(specification) - allowed
        if unknown:
            raise InteractiveVisualizationError(
                "图形配置包含未知字段：" + "、".join(sorted(map(str, unknown)))
            )
        chart_type = str(specification.get("chart_type", "auto"))
        if chart_type not in self._CHART_TYPES:
            raise InteractiveVisualizationError(f"不支持的交互图类型：{chart_type}")
        raw_encodings = specification.get("encodings") or {}
        if not isinstance(raw_encodings, Mapping):
            raise InteractiveVisualizationError("encodings必须是对象")
        unknown_encodings = set(raw_encodings) - self._ENCODING_KEYS
        if unknown_encodings:
            raise InteractiveVisualizationError(
                "encodings包含未知通道：" + "、".join(sorted(map(str, unknown_encodings)))
            )
        columns = {str(column) for column in frame.columns}
        encodings: Dict[str, Any] = {}
        for channel, value in raw_encodings.items():
            if channel in {"tooltip", "parallel"}:
                if value is None:
                    encodings[channel] = []
                    continue
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                    raise InteractiveVisualizationError(f"{channel}必须是字段数组")
                fields = list(dict.fromkeys(str(item) for item in value))
                if len(fields) > 8:
                    raise InteractiveVisualizationError(f"{channel}最多选择8个字段")
                missing = [field for field in fields if field not in columns]
                if missing:
                    raise InteractiveVisualizationError(
                        f"{channel}引用不存在字段：" + "、".join(missing)
                    )
                encodings[channel] = fields
                continue
            if value in {None, ""}:
                encodings[channel] = None
                continue
            field = str(value)
            if field not in columns:
                raise InteractiveVisualizationError(f"{channel}引用不存在字段：{field}")
            encodings[channel] = field
        if not encodings.get("x") and chart_type != "parallel":
            raise InteractiveVisualizationError("交互图至少需要绑定X轴")
        if chart_type in {"scatter", "line", "area"} and not encodings.get("y"):
            raise InteractiveVisualizationError(f"{chart_type}需要绑定Y轴")
        if chart_type == "parallel" and len(encodings.get("parallel") or []) < 2:
            raise InteractiveVisualizationError("平行坐标图至少需要选择2个字段")

        raw_filters = specification.get("filters") or []
        if not isinstance(raw_filters, Sequence) or isinstance(raw_filters, (str, bytes)):
            raise InteractiveVisualizationError("filters必须是数组")
        if len(raw_filters) > 12:
            raise InteractiveVisualizationError("筛选条件最多12个")
        filters: List[Dict[str, Any]] = []
        for raw_filter in raw_filters:
            if not isinstance(raw_filter, Mapping):
                raise InteractiveVisualizationError("每个筛选条件必须是对象")
            field = str(raw_filter.get("field", ""))
            kind = str(raw_filter.get("kind", "range"))
            if field not in columns:
                raise InteractiveVisualizationError(f"筛选引用不存在字段：{field}")
            if kind not in {"range", "time_range", "in"}:
                raise InteractiveVisualizationError(f"不支持的筛选类型：{kind}")
            item = {"field": field, "kind": kind}
            if kind == "in":
                values = raw_filter.get("values") or []
                if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                    raise InteractiveVisualizationError("分类筛选values必须是数组")
                if len(values) > 100:
                    raise InteractiveVisualizationError("分类筛选值最多100个")
                item["values"] = list(values)
            else:
                item["min"] = raw_filter.get("min")
                item["max"] = raw_filter.get("max")
                if kind == "range":
                    for boundary in ("min", "max"):
                        if item[boundary] is not None:
                            try:
                                number = float(item[boundary])
                            except (TypeError, ValueError) as exc:
                                raise InteractiveVisualizationError(
                                    f"{field}的{boundary}必须是有限数值"
                                ) from exc
                            if not np.isfinite(number):
                                raise InteractiveVisualizationError(
                                    f"{field}的{boundary}必须是有限数值"
                                )
                            item[boundary] = number
                    if (
                        item["min"] is not None and item["max"] is not None
                        and item["min"] > item["max"]
                    ):
                        raise InteractiveVisualizationError(f"{field}的筛选下界不能大于上界")
            filters.append(item)

        raw_aggregation = specification.get("aggregation") or {}
        if not isinstance(raw_aggregation, Mapping):
            raise InteractiveVisualizationError("aggregation必须是对象")
        function = str(raw_aggregation.get("function", "none"))
        if function not in self._AGGREGATIONS:
            raise InteractiveVisualizationError(f"不支持的聚合函数：{function}")
        group_by = list(dict.fromkeys(str(item) for item in raw_aggregation.get("group_by", []) or []))
        if len(group_by) > 4:
            raise InteractiveVisualizationError("聚合分组字段最多4个")
        missing_groups = [field for field in group_by if field not in columns]
        if missing_groups:
            raise InteractiveVisualizationError(
                "聚合引用不存在字段：" + "、".join(missing_groups)
            )
        time_unit = str(raw_aggregation.get("time_unit", "none"))
        if time_unit not in self._TIME_UNITS:
            raise InteractiveVisualizationError(f"不支持的时间粒度：{time_unit}")
        bins = int(raw_aggregation.get("bins", 20))
        if bins < 2 or bins > 100:
            raise InteractiveVisualizationError("连续轴分箱数必须在2到100之间")
        structural_channels = ("x", "y", "color", "facet", "animation")
        occupied: Dict[str, str] = {}
        for channel in structural_channels:
            field = encodings.get(channel)
            if not field:
                continue
            if field in occupied:
                raise InteractiveVisualizationError(
                    f"字段“{field}”不能同时绑定{occupied[field]}和{channel}；"
                    "重复编码不会增加信息。"
                )
            occupied[field] = channel

        y_field = encodings.get("y")
        y_kind = self._infer_kind(frame[y_field], y_field) if y_field else None
        x_field = encodings.get("x")
        x_kind = self._infer_kind(frame[x_field], x_field) if x_field else None
        size_field = encodings.get("size")
        if size_field and self._infer_kind(frame[size_field], size_field) != "numeric":
            raise InteractiveVisualizationError("点大小只能绑定真实数值度量。")
        if chart_type in {"scatter", "line", "area"} and y_kind != "numeric":
            raise InteractiveVisualizationError(
                f"{chart_type}的Y轴必须是真实数值度量，不能使用名称、类别或编码。"
            )
        if chart_type in {"line", "area"} and x_kind not in {"numeric", "datetime"}:
            raise InteractiveVisualizationError(
                f"{chart_type}的X轴必须是时间或有序数值。"
            )
        if chart_type == "parallel":
            invalid_parallel = [
                field for field in encodings.get("parallel") or []
                if self._infer_kind(frame[field], field) != "numeric"
            ]
            if invalid_parallel:
                raise InteractiveVisualizationError(
                    "平行坐标字段必须是真实数值度量：" + "、".join(invalid_parallel)
                )
        if function == "count":
            if chart_type not in {"auto", "bar"}:
                raise InteractiveVisualizationError("计数聚合只能使用柱状图。")
            if y_field:
                raise InteractiveVisualizationError(
                    "计数聚合会自动生成“记录数”，不能再绑定Y轴。"
                )
            if x_field:
                values = frame[x_field].dropna()
                unique_count = int(values.nunique(dropna=True))
                unique_rate = unique_count / max(int(len(values)), 1)
                x_role = self._semantic_role(
                    x_kind or "categorical", unique_count, unique_rate, int(len(values))
                )
                if x_role in {"label", "identifier"} and unique_count > 20 and unique_rate > 0.5:
                    raise InteractiveVisualizationError(
                        f"X轴“{x_field}”几乎一行一值，计数图只会得到一排1；"
                        "请选择低基数分组维度。"
                    )
        elif chart_type == "bar" and not y_field:
            raise InteractiveVisualizationError(
                "柱状图需要数值Y轴；若要统计类别数量，请选择计数聚合。"
            )
        max_points = int(specification.get("max_points", min(5_000, self.max_points)))
        max_points = max(200, min(max_points, self.max_points))
        return {
            "chart_type": chart_type,
            "encodings": encodings,
            "filters": filters,
            "aggregation": {
                "function": function,
                "group_by": group_by,
                "time_unit": time_unit,
                "bins": bins,
            },
            "max_points": max_points,
        }

    def _aggregate(
        self, frame: pd.DataFrame, spec: Mapping[str, Any]
    ) -> tuple[pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
        aggregation = dict(spec["aggregation"])
        function = aggregation["function"]
        encodings = dict(spec["encodings"])
        if function == "none":
            return frame.copy(deep=False), encodings, {
                "function": "none",
                "group_by": [],
                "scope": "row_level",
                "rows_before_output_limit": int(len(frame)),
                "dropped_encodings": [],
            }
        group_by = list(aggregation["group_by"])
        if not group_by:
            group_by = []
            for channel in ("x", "color", "facet", "animation"):
                field = encodings.get(channel)
                if not field:
                    continue
                # A continuous colour is a row-level measure, not a grouping
                # key.  Without an explicit aggregation contract, dropping it
                # is safer than silently converting it into arbitrary bins.
                if channel == "color" and self._infer_kind(frame[field], field) == "numeric":
                    continue
                group_by.append(field)
            group_by = list(dict.fromkeys(group_by))[:4]
        if not group_by:
            raise InteractiveVisualizationError("聚合图至少需要一个分组维度")
        working = frame.copy()
        binned_fields: List[str] = []
        for field in group_by:
            kind = self._infer_kind(working[field], field)
            if kind == "numeric" and working[field].nunique(dropna=True) > aggregation["bins"]:
                numeric = pd.to_numeric(working[field], errors="coerce")
                working[field] = pd.cut(
                    numeric, bins=aggregation["bins"], duplicates="drop"
                ).astype("string")
                binned_fields.append(field)
            elif kind == "datetime" and aggregation["time_unit"] != "none":
                dates = pd.to_datetime(working[field], errors="coerce")
                period = {
                    "day": "D", "week": "W", "month": "M",
                    "quarter": "Q", "year": "Y",
                }[aggregation["time_unit"]]
                working[field] = dates.dt.to_period(period).dt.start_time
        if function == "count":
            output = working.groupby(group_by, observed=True, dropna=False).size().reset_index(
                name="__count__"
            )
            encodings["y"] = "__count__"
        else:
            value = encodings.get("y")
            if not value:
                raise InteractiveVisualizationError(f"{function}聚合需要绑定数值Y轴")
            if not pd.api.types.is_numeric_dtype(working[value]):
                converted = pd.to_numeric(working[value], errors="coerce")
                if int(converted.notna().sum()) == 0:
                    raise InteractiveVisualizationError(f"{function}聚合的Y轴必须是数值字段")
                working[value] = converted
            output = (
                working.groupby(group_by, observed=True, dropna=False)[value]
                .agg(function).reset_index()
            )
        dropped_encodings: List[str] = []
        for channel in ("color", "size", "facet", "animation"):
            field = encodings.get(channel)
            if field and field not in output.columns:
                encodings[channel] = None
                dropped_encodings.append(channel)
        for channel in ("tooltip", "parallel"):
            retained = [
                field for field in (encodings.get(channel) or [])
                if field in output.columns
            ]
            if len(retained) != len(encodings.get(channel) or []):
                dropped_encodings.append(channel)
            encodings[channel] = retained
        return output, encodings, {
            "function": function,
            "group_by": group_by,
            "time_unit": aggregation["time_unit"],
            "binned_fields": binned_fields,
            "scope": "compiled_scan",
            "rows_before_output_limit": int(len(output)),
            "dropped_encodings": dropped_encodings,
        }

    @staticmethod
    def _prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(frame, pd.DataFrame) or frame.shape[1] == 0:
            raise InteractiveVisualizationError("交互作图需要至少一个字段的二维表")
        labels = [str(column) for column in frame.columns]
        if len(set(labels)) != len(labels):
            raise InteractiveVisualizationError("交互作图要求字段名唯一")
        prepared = frame.copy(deep=False)
        prepared.columns = labels
        return prepared

    @staticmethod
    def _looks_identifier(name: Optional[str]) -> bool:
        if not name:
            return False
        raw = str(name).strip().lower()
        normalized = re.sub(r"[\s_\-（）()\[\]]+", "", raw)
        if any(token in normalized for token in ("编码", "编号", "序号", "索引", "代码", "条码")):
            return True
        parts = [part for part in re.split(r"[^a-z0-9]+", raw) if part]
        return any(part in {"id", "code", "uuid", "identifier", "key", "sku"} for part in parts)

    @classmethod
    def _infer_kind(cls, series: pd.Series, name: Optional[str] = None) -> str:
        if cls._looks_identifier(name):
            return "identifier"
        if pd.api.types.is_bool_dtype(series):
            return "categorical"
        if pd.api.types.is_numeric_dtype(series):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"
        sample = series.dropna().astype(str).head(500)
        if not sample.empty:
            date_shape = sample.str.match(
                r"^\s*\d{4}(?:[-/年])\d{1,2}(?:[-/月])\d{1,2}", na=False
            )
            if float(date_shape.mean()) >= 0.8:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                if float(parsed.notna().mean()) >= 0.9:
                    return "datetime"
        return "categorical"

    @staticmethod
    def _semantic_role(
        kind: str, unique_count: int, unique_rate: float, non_missing: int
    ) -> str:
        if kind == "identifier":
            return "identifier"
        if kind == "datetime":
            return "time"
        if kind == "numeric":
            return "measure"
        dimension_limit = max(12, min(80, int(math.sqrt(max(non_missing, 1))) * 2))
        if 1 < unique_count <= dimension_limit and unique_rate <= 0.5:
            return "dimension"
        return "label"

    @staticmethod
    def _recommend(fields: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        measures = [
            str(field["name"]) for field in fields
            if field.get("semantic_role") == "measure"
        ]
        time_fields = [
            str(field["name"]) for field in fields
            if field.get("semantic_role") == "time"
        ]
        dimensions = [
            str(field["name"]) for field in fields
            if field.get("semantic_role") == "dimension"
        ]
        compact_dimensions = [
            str(field["name"]) for field in fields
            if field.get("semantic_role") == "dimension"
            and int(field.get("unique_count_profiled", 0)) <= 12
        ]
        labels = [
            str(field["name"]) for field in fields
            if field.get("semantic_role") == "label"
        ]
        if time_fields and measures:
            return {
                "chart_type": "line",
                "encodings": {
                    "x": time_fields[0], "y": measures[0],
                    "color": compact_dimensions[0] if compact_dimensions else None,
                },
                "aggregation": {"function": "mean", "time_unit": "day"},
                "reason": "已排除编码字段，使用时间×度量展示趋势。",
            }
        if len(measures) >= 2:
            return {
                "chart_type": "scatter",
                "encodings": {
                    "x": measures[0], "y": measures[1],
                    "color": compact_dimensions[0] if compact_dimensions else None,
                    "size": measures[2] if len(measures) >= 3 else None,
                },
                "aggregation": {"function": "none"},
                "reason": "已排除编码字段，使用两个真实度量检查关系。",
            }
        if dimensions and measures:
            return {
                "chart_type": "bar",
                "encodings": {"x": dimensions[0], "y": measures[0]},
                "aggregation": {"function": "mean"},
                "reason": "使用低基数维度比较度量，避免把编码当成连续变量。",
            }
        if dimensions:
            return {
                "chart_type": "bar",
                "encodings": {"x": dimensions[0], "y": None},
                "aggregation": {"function": "count"},
                "reason": "没有可靠数值度量，改为展示低基数类别频数。",
            }
        if labels and measures:
            return {
                "chart_type": "bar",
                "encodings": {"x": labels[0], "y": measures[0]},
                "aggregation": {"function": "none"},
                "reason": "按结果行展示标签对应度量；不对已经形成的结果再次求平均。",
            }
        first = labels[0] if labels else next(
            (str(field["name"]) for field in fields if field.get("semantic_role") != "identifier"),
            str(fields[0]["name"]) if fields else None,
        )
        return {
            "chart_type": "bar",
            "encodings": {"x": first, "y": None},
            "aggregation": {"function": "count"},
            "reason": "没有可安全解释的连续度量，退化为频数图而不是伪相关散点图。",
        }

    @staticmethod
    def _capability(fields: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        measures = [field for field in fields if field.get("semantic_role") == "measure"]
        times = [field for field in fields if field.get("semantic_role") == "time"]
        dimensions = [field for field in fields if field.get("semantic_role") == "dimension"]
        if not measures:
            return {
                "level": "composition_only",
                "summary": (
                    "当前表主要由编码、名称和类别组成，没有可解释的数值度量；"
                    "只能可靠分析类别构成，数值关系、趋势和平行坐标已停用。"
                ),
                "enabled_charts": ["bar"],
            }
        enabled = ["bar"]
        if len(measures) >= 2:
            enabled.extend(["scatter", "parallel"])
        if times:
            enabled.extend(["line", "area"])
        return {
            "level": "full" if len(measures) >= 2 else "limited",
            "summary": (
                f"识别到{len(measures)}个数值度量、{len(times)}个时间字段和"
                f"{len(dimensions)}个低基数分组维度；只开放语义条件满足的图形。"
            ),
            "enabled_charts": enabled,
        }

    @staticmethod
    def _selected_fields(encodings: Mapping[str, Any]) -> List[str]:
        fields: List[str] = []
        for channel, value in encodings.items():
            values = value if channel in {"tooltip", "parallel"} else [value]
            for field in values or []:
                if field and str(field) not in fields:
                    fields.append(str(field))
        return fields

    @staticmethod
    def _resolve_chart_type(
        chart_type: str, frame: pd.DataFrame, encodings: Mapping[str, Any]
    ) -> str:
        if chart_type != "auto":
            return chart_type
        x = encodings.get("x")
        y = encodings.get("y")
        if x in frame.columns and y in frame.columns:
            x_kind = InteractiveVisualizationCompiler._infer_kind(frame[x], str(x))
            y_kind = InteractiveVisualizationCompiler._infer_kind(frame[y], str(y))
            if x_kind == "datetime":
                return "line"
            if x_kind == "numeric" and y_kind == "numeric":
                return "scatter"
        return "bar"

    @staticmethod
    def _coverage_sample(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
        if len(frame) <= limit:
            return frame
        positions = np.linspace(0, len(frame) - 1, num=limit, dtype=np.int64)
        return frame.iloc[np.unique(positions)]

    @staticmethod
    def _levels(
        frame: pd.DataFrame, field: Optional[str], limit: int, *, sort: bool = False
    ) -> List[Any]:
        if not field or field not in frame.columns:
            return []
        values = [
            InteractiveVisualizationCompiler._json_value(
                value,
                InteractiveVisualizationCompiler._infer_kind(frame[field], str(field)),
            )
            for value in frame[field].drop_duplicates().dropna().head(limit).tolist()
        ]
        if sort:
            try:
                return sorted(values)
            except TypeError:
                return sorted(values, key=str)
        return values

    @staticmethod
    def _numeric_ranges(
        frame: pd.DataFrame, encodings: Mapping[str, Any]
    ) -> Dict[str, List[Optional[float]]]:
        output: Dict[str, List[Optional[float]]] = {}
        fields = [encodings.get("color"), encodings.get("size")]
        for field in fields:
            if not field or field not in frame.columns:
                continue
            if InteractiveVisualizationCompiler._infer_kind(frame[field], str(field)) != "numeric":
                continue
            values = pd.to_numeric(frame[field], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            if not values.empty:
                output[str(field)] = [
                    InteractiveVisualizationCompiler._number(values.min()),
                    InteractiveVisualizationCompiler._number(values.max()),
                ]
        return output

    @staticmethod
    def _number(value: Any) -> Optional[float]:
        number = float(value)
        return number if np.isfinite(number) else None

    @staticmethod
    def _json_value(value: Any, kind: Optional[str] = None) -> Any:
        if value is None or value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(value).isoformat()
        if kind == "identifier":
            return str(value)
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return None if not np.isfinite(value) else float(value)
        if isinstance(value, (np.bool_,)):
            return bool(value)
        if isinstance(value, pd.Interval):
            return str(value)
        return value if isinstance(value, (str, int, bool)) else str(value)
