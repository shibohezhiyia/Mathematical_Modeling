"""公开基准实验配置协议。

配置把题源目录摘要、固定预算、随机种子、方法对照和留出划分写在一个可审计
文件中。它只描述运行条件，不携带题面、答案或用户数据，也不自动选择赢家。
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

EXPERIMENT_SCHEMA = "mathmodel.public-experiment-config/v1"
SPLITS = frozenset({"development", "structure_transform", "unseen", "adversarial"})


class ExperimentConfigError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _fail(code: str) -> None:
    raise ExperimentConfigError(code)


def _digest(value: Any) -> str:
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        _fail("invalid_digest")
    return value


def _bounded_int(value: Any, lower: int, upper: int, code: str = "invalid_budget") -> int:
    if type(value) is not int or not lower <= value <= upper:
        _fail(code)
    return value


@dataclass(frozen=True)
class PublicExperimentConfig:
    catalog_digest: str
    benchmark_name: str
    revision: int
    seed: int
    max_seconds: int
    max_memory_mb: int
    max_api_calls: int
    max_candidates: int
    max_cases: int
    max_repeats: int
    selection_splits: tuple[str, ...]
    evaluation_splits: tuple[str, ...]
    methods: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "PublicExperimentConfig":
        if type(payload) is not dict:
            _fail("config_must_be_object")
        allowed = {"schema_version", "catalog_digest", "benchmark_name", "revision", "seed",
                   "budget", "selection_splits", "evaluation_splits", "methods", "policy"}
        if set(payload) != allowed or payload.get("schema_version") != EXPERIMENT_SCHEMA:
            _fail("invalid_config_fields")
        _digest(payload["catalog_digest"])
        name = payload["benchmark_name"]
        if type(name) is not str or not name.strip() or len(name) > 120:
            _fail("invalid_benchmark_name")
        revision = _bounded_int(payload["revision"], 1, 1_000_000, "invalid_revision")
        seed = _bounded_int(payload["seed"], 0, 2**32 - 1, "invalid_seed")
        budget = payload["budget"]
        if type(budget) is not dict or set(budget) != {"max_seconds", "max_memory_mb", "max_api_calls", "max_candidates", "max_cases", "max_repeats"}:
            _fail("invalid_budget_fields")
        values = (
            _bounded_int(budget["max_seconds"], 1, 86_400),
            _bounded_int(budget["max_memory_mb"], 32, 1_048_576),
            _bounded_int(budget["max_api_calls"], 0, 100_000),
            _bounded_int(budget["max_candidates"], 1, 100_000),
            _bounded_int(budget["max_cases"], 1, 10_000),
            _bounded_int(budget["max_repeats"], 1, 32),
        )

        def splits(value: Any, *, selection: bool) -> tuple[str, ...]:
            if type(value) is not list or not value or len(value) > 4 or any(item not in SPLITS for item in value):
                _fail("invalid_splits")
            result = tuple(dict.fromkeys(value))
            if len(result) != len(value):
                _fail("duplicate_split")
            if selection and any(item not in {"development", "structure_transform"} for item in result):
                _fail("selection_split_leakage")
            if not selection and any(item not in {"unseen", "adversarial"} for item in result):
                _fail("evaluation_split_leakage")
            return result

        selection, evaluation = splits(payload["selection_splits"], selection=True), splits(payload["evaluation_splits"], selection=False)
        methods = payload["methods"]
        if type(methods) is not list or not 2 <= len(methods) <= 32:
            _fail("invalid_methods")
        normalized = tuple(item.strip() if isinstance(item, str) else "" for item in methods)
        if any(not item or len(item) > 120 or item in {"unseen", "answer", "ground_truth"} for item in normalized) or len(set(normalized)) != len(normalized):
            _fail("invalid_methods")
        policy = payload["policy"]
        expected = {"no_selection_on_evaluation", "no_problem_text_in_config", "fixed_budget_required"}
        if policy != {key: True for key in expected}:
            _fail("invalid_policy")
        return cls(payload["catalog_digest"], name.strip(), revision, seed, *values, selection, evaluation, normalized)

    def public(self) -> dict[str, Any]:
        return {"schema_version": EXPERIMENT_SCHEMA, "catalog_digest": self.catalog_digest,
                "benchmark_name": self.benchmark_name, "revision": self.revision, "seed": self.seed,
                "budget": {"max_seconds": self.max_seconds, "max_memory_mb": self.max_memory_mb,
                            "max_api_calls": self.max_api_calls, "max_candidates": self.max_candidates,
                            "max_cases": self.max_cases, "max_repeats": self.max_repeats},
                "selection_splits": list(self.selection_splits), "evaluation_splits": list(self.evaluation_splits),
                "methods": list(self.methods),
                "policy": {"no_selection_on_evaluation": True, "no_problem_text_in_config": True,
                            "fixed_budget_required": True}}

    @property
    def digest(self) -> str:
        raw = json.dumps(self.public(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_experiment_config(path: str | Path) -> PublicExperimentConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentConfigError("config_unreadable") from exc
    return PublicExperimentConfig.from_payload(payload)


__all__ = ["EXPERIMENT_SCHEMA", "ExperimentConfigError", "PublicExperimentConfig", "load_experiment_config"]
