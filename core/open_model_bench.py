"""可复现的开放世界数学建模基准清单与结果协议。

这个模块只负责定义基准的边界、划分和计分记录，不执行模型，也不把缺少真值的
案例强行计入“正确率”。数据文件只通过版本化指纹引用，避免基准清单把用户
数据或最终测试内容复制进搜索过程。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import math
import time
from typing import Any, Iterable, Mapping


BENCHMARK_SCHEMA = "mathmodel.open-model-bench/v1"
SPLITS = frozenset({"development", "unseen", "structure_transform", "adversarial"})
FAMILIES = frozenset({"data", "pure_mechanistic", "multi_table", "optimization", "dynamics"})
FAILURE_STAGES = frozenset({
    "representation", "proposal", "type_check", "compile", "numeric", "evidence", "validation",
})
STATUSES = frozenset({"pass", "fail", "unresolved", "not_run"})
METRIC_FIELDS = frozenset({
    "contract_correct", "ir_effective", "executable", "numerically_correct",
    "constraint_violation_rate", "counterexample_found", "interval_coverage",
})


class BenchmarkValidationError(ValueError):
    """基准协议错误，代码稳定且不携带题面或用户数据。"""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def build_reproducibility_manifest(
    benchmark: "OpenModelBench",
    *,
    seed: int,
    dependency_versions: Mapping[str, str],
    runner_version: str,
) -> dict[str, Any]:
    """Build a deterministic run manifest without reading user files."""
    if not isinstance(benchmark, OpenModelBench):
        _fail("benchmark_must_be_open_model_bench")
    if type(seed) is not int or not 0 <= seed < 2**32:
        _fail("invalid_random_seed")
    if not isinstance(dependency_versions, Mapping) or not dependency_versions or len(dependency_versions) > 256:
        _fail("invalid_dependency_versions")
    normalized = {}
    for name, version in dependency_versions.items():
        if (type(name) is not str or not name.strip() or len(name) > 160
                or type(version) is not str or not version.strip() or len(version) > 160):
            _fail("invalid_dependency_versions")
        normalized[name.strip()] = version.strip()
    if type(runner_version) is not str or not runner_version.strip() or len(runner_version) > 160:
        _fail("invalid_runner_version")
    payload = {"schema_version": "mathmodel.reproducibility-manifest/v1",
               "benchmark_digest": benchmark.digest, "seed": seed,
               "dependency_versions": dict(sorted(normalized.items())),
               "runner_version": runner_version.strip()}
    payload["manifest_digest"] = sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def _fail(code: str) -> None:
    raise BenchmarkValidationError(code)


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BenchmarkValidationError("non_json_manifest") from exc


def _digest(value: Any) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def _text(value: Any, limit: int) -> str:
    if type(value) is not str or not value.strip() or len(value) > limit:
        _fail("invalid_text")
    return value


def _id(value: Any) -> str:
    if type(value) is not str or not value or len(value) > 128 or any(ch.isspace() for ch in value):
        _fail("invalid_case_id")
    return value


def _sha(value: Any) -> str:
    if type(value) is not str or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        _fail("invalid_data_fingerprint")
    return value


@dataclass(frozen=True)
class BenchmarkCase:
    """不含可执行代码的单个基准案例描述。"""

    case_id: str
    split: str
    family: str
    statement: str
    data_fingerprints: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    has_ground_truth: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "BenchmarkCase":
        if type(payload) is not dict:
            _fail("case_must_be_object")
        allowed = {"id", "split", "family", "statement", "data_fingerprints", "tags", "has_ground_truth"}
        if set(payload) - allowed or not {"id", "split", "family", "statement"} <= set(payload):
            _fail("invalid_case_fields")
        case_id = _id(payload["id"])
        split, family = payload["split"], payload["family"]
        if split not in SPLITS:
            _fail("invalid_split")
        if family not in FAMILIES:
            _fail("invalid_family")
        statement = _text(payload["statement"], 64_000)
        fingerprints = payload.get("data_fingerprints", [])
        tags = payload.get("tags", [])
        if type(fingerprints) is not list or len(fingerprints) > 32 or any(_sha(item) is None for item in fingerprints):
            _fail("invalid_data_fingerprints")
        if type(tags) is not list or len(tags) > 32 or any(type(item) is not str or not item or len(item) > 80 for item in tags):
            _fail("invalid_tags")
        truth = payload.get("has_ground_truth", False)
        if type(truth) is not bool:
            _fail("invalid_ground_truth_flag")
        if truth and not fingerprints and family != "pure_mechanistic":
            # 有观测真值的题必须至少绑定一个数据指纹；无数据机理题可以由
            # 形式性质或独立解析解提供真值，后续结果仍需单独声明证据类型。
            _fail("ground_truth_requires_data")
        return cls(case_id, split, family, statement, tuple(fingerprints), tuple(tags), truth)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.case_id,
            "split": self.split,
            "family": self.family,
            "statement": self.statement,
            "data_fingerprints": list(self.data_fingerprints),
            "tags": list(self.tags),
            "has_ground_truth": self.has_ground_truth,
        }


@dataclass(frozen=True)
class OpenModelBench:
    """不可变基准清单；案例内容改变会改变 ``digest``。"""

    _json: str = field(repr=False)

    @classmethod
    def create(cls, cases: Iterable[BenchmarkCase | Mapping[str, Any]], *, name: str = "open-model-bench",
               revision: int = 1) -> "OpenModelBench":
        if type(name) is not str or not name.strip() or len(name) > 120:
            _fail("invalid_benchmark_name")
        if type(revision) is not int or revision < 1:
            _fail("invalid_benchmark_revision")
        if isinstance(cases, (str, bytes)):
            _fail("invalid_case_count")
        parsed = []
        try:
            for index, case in enumerate(cases):
                if index >= 10_000:
                    _fail("invalid_case_count")
                parsed.append(case if isinstance(case, BenchmarkCase) else BenchmarkCase.from_payload(case))
        except TypeError as exc:
            raise BenchmarkValidationError("invalid_case_count") from exc
        if not parsed:
            _fail("invalid_case_count")
        ids = [case.case_id for case in parsed]
        if len(ids) != len(set(ids)):
            _fail("duplicate_case_id")
        # 同一个数据指纹不能同时出现在开发和最终未见题中，否则模型可能
        # 通过数据复用而不是结构泛化获得分数。
        by_fingerprint: dict[str, set[str]] = {}
        for case in parsed:
            for fingerprint in case.data_fingerprints:
                by_fingerprint.setdefault(fingerprint, set()).add(case.split)
        for splits in by_fingerprint.values():
            if "development" in splits and (splits - {"development"}):
                _fail("data_fingerprint_split_overlap")
        payload = {
            "schema_version": BENCHMARK_SCHEMA,
            "name": name,
            "revision": revision,
            "cases": [case.public() for case in sorted(parsed, key=lambda item: item.case_id)],
        }
        return cls(_canonical(payload))

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "OpenModelBench":
        if type(payload) is not dict or set(payload) != {"schema_version", "name", "revision", "cases"}:
            _fail("invalid_benchmark_fields")
        if payload["schema_version"] != BENCHMARK_SCHEMA:
            _fail("benchmark_schema_mismatch")
        return cls.create(payload["cases"], name=payload["name"], revision=payload["revision"])

    @property
    def digest(self) -> str:
        return sha256(self._json.encode("utf-8")).hexdigest()

    def public(self) -> dict[str, Any]:
        return json.loads(self._json)

    def cases(self, *, split: str | None = None, family: str | None = None) -> tuple[BenchmarkCase, ...]:
        if split is not None and split not in SPLITS:
            _fail("invalid_split")
        if family is not None and family not in FAMILIES:
            _fail("invalid_family")
        return tuple(
            BenchmarkCase.from_payload(item)
            for item in self.public()["cases"]
            if (split is None or item["split"] == split) and (family is None or item["family"] == family)
        )

    def partition_fingerprint(self, split: str) -> str:
        return _digest({"benchmark": self.digest, "split": split, "ids": [case.case_id for case in self.cases(split=split)]})

    def record_result(self, case_id: str, *, status: str, failure_stage: str | None = None,
                      metrics: Mapping[str, Any] | None = None, elapsed_seconds: float | None = None,
                      peak_memory_bytes: int | None = None) -> dict[str, Any]:
        case = next((item for item in self.cases() if item.case_id == case_id), None)
        if case is None:
            _fail("unknown_case_id")
        if status not in STATUSES:
            _fail("invalid_result_status")
        if failure_stage is not None and failure_stage not in FAILURE_STAGES:
            _fail("invalid_failure_stage")
        if status == "fail" and failure_stage is None:
            _fail("failed_result_requires_stage")
        if status != "fail" and failure_stage is not None:
            _fail("failure_stage_on_nonfailed_result")
        if elapsed_seconds is not None and (type(elapsed_seconds) not in (int, float) or not math.isfinite(elapsed_seconds) or elapsed_seconds < 0):
            _fail("invalid_elapsed_seconds")
        if peak_memory_bytes is not None and (type(peak_memory_bytes) is not int or peak_memory_bytes < 0):
            _fail("invalid_peak_memory")
        clean_metrics = dict(metrics or {})
        if len(clean_metrics) > 32 or set(clean_metrics) - METRIC_FIELDS:
            _fail("invalid_result_metrics")
        for key, value in clean_metrics.items():
            if type(value) not in (bool, int, float) or (type(value) is float and not math.isfinite(value)):
                _fail("invalid_result_metric_value")
            if key in {"contract_correct", "ir_effective", "executable", "numerically_correct", "counterexample_found"} and type(value) is not bool:
                _fail("binary_metric_must_be_bool")
            if key in {"constraint_violation_rate", "interval_coverage"} and not 0 <= float(value) <= 1:
                _fail("metric_out_of_range")
        try:
            _canonical(clean_metrics)
        except BenchmarkValidationError:
            _fail("invalid_result_metrics")
        return {
            "schema_version": "mathmodel.open-model-result/v1",
            "benchmark_digest": self.digest,
            "case_id": case.case_id,
            "split": case.split,
            "family": case.family,
            "status": status,
            "failure_stage": failure_stage,
            "metrics": clean_metrics,
            "elapsed_seconds": elapsed_seconds,
            "peak_memory_bytes": peak_memory_bytes,
            "has_ground_truth": case.has_ground_truth,
        }

    def summarize(self, results: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        """按案例分母统计结果；没有真值时不输出准确率。"""
        expected = {case.case_id: case for case in self.cases()}
        rows = list(results)
        seen = set()
        counts = {status: 0 for status in STATUSES}
        failures = {stage: 0 for stage in FAILURE_STAGES}
        metric_values: dict[str, list[float]] = {key: [] for key in METRIC_FIELDS}
        for row in rows:
            if type(row) is not dict or row.get("benchmark_digest") != self.digest:
                _fail("result_scope_mismatch")
            case_id = row.get("case_id")
            if case_id not in expected or case_id in seen:
                _fail("duplicate_or_unknown_result")
            seen.add(case_id)
            status = row.get("status")
            if status not in counts:
                _fail("invalid_result_status")
            counts[status] += 1
            if status == "fail":
                stage = row.get("failure_stage")
                if stage not in failures:
                    _fail("invalid_failure_stage")
                failures[stage] += 1
            elif row.get("failure_stage") is not None:
                _fail("failure_stage_on_nonfailed_result")
            metrics = row.get("metrics", {})
            if type(metrics) is not dict or set(metrics) - METRIC_FIELDS:
                _fail("invalid_result_metrics")
            for key, value in metrics.items():
                if type(value) not in (bool, int, float) or (type(value) is float and not math.isfinite(value)):
                    _fail("invalid_result_metric_value")
                if key in {"contract_correct", "ir_effective", "executable", "numerically_correct", "counterexample_found"} and type(value) is not bool:
                    _fail("binary_metric_must_be_bool")
                if key in {"constraint_violation_rate", "interval_coverage"} and not 0 <= float(value) <= 1:
                    _fail("metric_out_of_range")
                metric_values[key].append(float(value))
        observed_metrics = {
            key: {"n": len(values), "mean": sum(values) / len(values)}
            for key, values in metric_values.items() if values
        }
        denominator = len(expected)
        completed = counts["pass"] + counts["fail"] + counts["unresolved"]
        summary: dict[str, Any] = {
            "benchmark_digest": self.digest,
            "case_count": denominator,
            "result_count": len(rows),
            "unrun_count": denominator - len(rows),
            "status_counts": counts,
            "failure_stage_counts": failures,
            "observed_metrics": observed_metrics,
            "completion_rate": completed / denominator if denominator else 0.0,
            "ground_truth_cases": sum(case.has_ground_truth for case in expected.values()),
            "policy": {"accuracy_reported": False, "reason": "runner metrics and labels are not supplied by this registry"},
        }
        return summary

    def compare_runs(self, runs: Mapping[str, Iterable[Mapping[str, Any]]], *,
                     allowed_splits: Iterable[str] = ("development", "structure_transform")) -> dict[str, Any]:
        """对照同一开发协议下的多个方法，禁止用未见题选择赢家。"""
        if type(runs) is not dict or not 2 <= len(runs) <= 8:
            _fail("invalid_run_count")
        splits = tuple(allowed_splits)
        if not splits or any(split not in SPLITS or split in {"unseen", "adversarial"} for split in splits):
            _fail("final_split_in_selection")
        eligible = {case.case_id for split in splits for case in self.cases(split=split)}
        summaries: dict[str, Any] = {}
        case_status: dict[str, dict[str, str]] = {case_id: {} for case_id in sorted(eligible)}
        materialized_runs: dict[str, list[Mapping[str, Any]]] = {}
        for method, rows in runs.items():
            if type(method) is not str or not method or len(method) > 80:
                _fail("invalid_method_id")
            rows = list(rows)
            materialized_runs[method] = rows
            for row in rows:
                if type(row) is not dict:
                    _fail("result_scope_mismatch")
                if row.get("case_id") not in eligible:
                    _fail("result_outside_selection_split")
                case_status[row["case_id"]][method] = row.get("status")
            summaries[method] = self.summarize(rows)
        missing = {
            method: sorted(eligible - {row.get("case_id") for row in rows})
            for method, rows in materialized_runs.items()
        }
        return {
            "benchmark_digest": self.digest,
            "selection_splits": list(splits),
            "methods": summaries,
            "paired_case_status": case_status,
            "missing_cases_by_method": missing,
            "winner_selected": False,
            "policy": {
                "final_test_used_for_selection": False,
                "selection_requires_predeclared_rule": True,
                "no_accuracy_without_ground_truth": True,
            },
        }


class BenchmarkRunner:
    """将一个外部方法适配为可复现的案例结果流。

    runner 只能接收一个公开案例字典，并返回 ``status``、可选失败阶段和
    指标；异常不会泄露堆栈或题面内容，而会保守记为提议阶段失败。
    """

    def __init__(self, benchmark: OpenModelBench):
        if not isinstance(benchmark, OpenModelBench):
            raise TypeError("benchmark_must_be_open_model_bench")
        self.benchmark = benchmark

    def run(self, runner, *, splits: Iterable[str] = ("development",), family: str | None = None,
            max_cases: int = 256) -> list[dict[str, Any]]:
        if not callable(runner):
            raise TypeError("runner_must_be_callable")
        splits = tuple(splits)
        if not splits or any(split not in SPLITS for split in splits):
            _fail("invalid_split")
        if family is not None and family not in FAMILIES:
            _fail("invalid_family")
        if type(max_cases) is not int or not 1 <= max_cases <= 1024:
            _fail("invalid_runner_case_budget")
        cases = [case for split in splits for case in self.benchmark.cases(split=split, family=family)]
        if len(cases) > max_cases:
            _fail("runner_case_budget_exhausted")
        results = []
        for case in cases:
            started = time.perf_counter()
            try:
                raw = runner(case.public())
                if type(raw) is not dict:
                    raise BenchmarkValidationError("runner_result_must_be_object")
                status = raw.get("status")
                failure_stage = raw.get("failure_stage")
                metrics = raw.get("metrics", {})
                result = self.benchmark.record_result(
                    case.case_id, status=status, failure_stage=failure_stage,
                    metrics=metrics, elapsed_seconds=time.perf_counter() - started,
                    peak_memory_bytes=raw.get("peak_memory_bytes"),
                )
            except BenchmarkValidationError as exc:
                result = self.benchmark.record_result(
                    case.case_id, status="fail", failure_stage="validation",
                    elapsed_seconds=time.perf_counter() - started,
                )
                result["runner_error_code"] = exc.code
            except Exception:
                result = self.benchmark.record_result(
                    case.case_id, status="fail", failure_stage="proposal",
                    elapsed_seconds=time.perf_counter() - started,
                )
                result["runner_error_code"] = "runner_exception"
            results.append(result)
        return results


__all__ = ["BENCHMARK_SCHEMA", "SPLITS", "FAMILIES", "FAILURE_STAGES", "STATUSES", "METRIC_FIELDS",
           "BenchmarkValidationError", "BenchmarkCase", "OpenModelBench", "BenchmarkRunner",
           "build_reproducibility_manifest"]
