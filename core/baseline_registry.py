"""Versioned, conservative baselines used to calibrate model claims.

Baselines are comparison references, not fallback answers and not proof that a
candidate is useful.  This registry keeps their identity and metric direction
stable so benchmark runs can compare later search revisions to the same floor.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "mathmodel.baseline-registry/v1"
_FAMILIES = frozenset({"data", "pure_mechanistic", "multi_table", "optimization", "dynamics"})
_DIRECTIONS = frozenset({"minimize", "maximize"})


class BaselineRegistryError(ValueError):
    pass


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BaselineRegistryError("baseline_payload_not_serializable") from exc


@dataclass(frozen=True)
class BaselineSpec:
    baseline_id: str
    family: str
    task: str
    metric: str
    direction: str
    description: str
    version: int = 1

    def __post_init__(self) -> None:
        if (type(self.baseline_id) is not str or not self.baseline_id or len(self.baseline_id) > 80
                or any(ch.isspace() for ch in self.baseline_id)):
            raise BaselineRegistryError("invalid_baseline_id")
        if self.family not in _FAMILIES or type(self.task) is not str or not self.task:
            raise BaselineRegistryError("invalid_baseline_family_or_task")
        if type(self.metric) is not str or not self.metric or len(self.metric) > 80:
            raise BaselineRegistryError("invalid_baseline_metric")
        if self.direction not in _DIRECTIONS:
            raise BaselineRegistryError("invalid_baseline_direction")
        if type(self.description) is not str or not self.description or len(self.description) > 500:
            raise BaselineRegistryError("invalid_baseline_description")
        if type(self.version) is not int or self.version < 1:
            raise BaselineRegistryError("invalid_baseline_version")

    def public(self) -> dict[str, Any]:
        return {"id": self.baseline_id, "family": self.family, "task": self.task,
                "metric": self.metric, "direction": self.direction,
                "description": self.description, "version": self.version}


_DEFAULTS = (
    BaselineSpec("regression_mean", "data", "regression", "rmse", "minimize", "训练段目标均值"),
    BaselineSpec("classification_majority", "data", "classification", "f1_weighted", "maximize", "训练段多数类"),
    BaselineSpec("clustering_kmeans", "data", "clustering", "silhouette", "maximize", "标准化特征上的有界 K-Means"),
    BaselineSpec("dynamics_zero_increment", "dynamics", "differential_equations", "rmse", "minimize", "零状态增量"),
    BaselineSpec("optimization_feasible_reference", "optimization", "optimization", "objective", "minimize", "题面约束下的可行参考点"),
)


class BaselineRegistry:
    def __init__(self, specs: Iterable[BaselineSpec] = _DEFAULTS, *, revision: int = 1) -> None:
        if type(revision) is not int or revision < 1:
            raise BaselineRegistryError("invalid_registry_revision")
        parsed = tuple(specs)
        if not parsed or len(parsed) > 64 or any(not isinstance(item, BaselineSpec) for item in parsed):
            raise BaselineRegistryError("invalid_baseline_specs")
        if len({item.baseline_id for item in parsed}) != len(parsed):
            raise BaselineRegistryError("duplicate_baseline_id")
        self._payload = {"schema_version": SCHEMA_VERSION, "revision": revision,
                         "baselines": [item.public() for item in sorted(parsed, key=lambda item: item.baseline_id)]}
        self._json = _canonical(self._payload)

    @property
    def digest(self) -> str:
        return sha256(self._json.encode("utf-8")).hexdigest()

    def public(self) -> dict[str, Any]:
        return json.loads(self._json)

    def for_task(self, task: str) -> BaselineSpec | None:
        for item in self._payload["baselines"]:
            if item["task"] == task:
                return BaselineSpec(item["id"], item["family"], item["task"], item["metric"],
                                    item["direction"], item["description"], item["version"])
        return None

    def compare(self, task: str, candidate_value: Any, baseline_value: Any) -> dict[str, Any]:
        spec = self.for_task(task)
        if spec is None:
            return {"status": "not_assessed", "reason": "baseline_not_registered", "registry_digest": self.digest}
        if type(candidate_value) not in (int, float) or type(baseline_value) not in (int, float):
            return {"status": "not_assessed", "reason": "non_numeric_metric", "baseline_id": spec.baseline_id,
                    "registry_digest": self.digest}
        if not math.isfinite(float(candidate_value)) or not math.isfinite(float(baseline_value)):
            return {"status": "not_assessed", "reason": "nonfinite_metric", "baseline_id": spec.baseline_id,
                    "registry_digest": self.digest}
        margin = float(baseline_value) - float(candidate_value) if spec.direction == "minimize" else float(candidate_value) - float(baseline_value)
        return {"status": "candidate_better" if margin > 0 else ("tie" if margin == 0 else "baseline_not_beaten"),
                "baseline_id": spec.baseline_id, "metric": spec.metric, "direction": spec.direction,
                "candidate_value": float(candidate_value), "baseline_value": float(baseline_value),
                "margin": margin, "registry_digest": self.digest}


def default_baseline_registry() -> BaselineRegistry:
    return BaselineRegistry()


__all__ = ["SCHEMA_VERSION", "BaselineRegistryError", "BaselineSpec", "BaselineRegistry",
           "default_baseline_registry"]
