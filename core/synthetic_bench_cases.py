"""Small deterministic cases for exercising the five core modeling families.

These are protocol fixtures, not a claim about real-world accuracy.  They keep
the benchmark runner honest by separating a case's observable inputs from a
small set of independently checkable invariants.  No competition statement or
domain-specific solver is embedded here.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Mapping


@dataclass(frozen=True)
class SyntheticBenchmarkCase:
    case_id: str
    family: str
    statement: str
    data: Mapping[str, Any]
    expected: Mapping[str, Any]
    seed: int

    def public(self) -> dict[str, Any]:
        """Return metadata only; raw fixture values stay out of manifests."""
        return {
            "id": self.case_id,
            "family": self.family,
            "statement": self.statement,
            "data_fingerprint": _fingerprint(self.data),
            "expected_checks": sorted(str(key) for key in self.expected),
            "seed": self.seed,
        }


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False, default=str)
    return sha256(encoded.encode("utf-8")).hexdigest()


def build_minimal_case_catalog() -> tuple[SyntheticBenchmarkCase, ...]:
    """Build one bounded, reproducible fixture per core benchmark family."""
    import numpy as np

    rng = np.random.default_rng(20260908)
    x = np.arange(24, dtype=float)
    data_case = {
        "features": {"x": x.tolist(), "z": (np.sin(x / 4.0)).round(8).tolist()},
        "target": (2.5 * x - 0.7 * np.sin(x / 4.0)).round(8).tolist(),
    }

    t = np.linspace(0.0, 4.0, 25)
    mechanistic_case = {
        "time": t.round(8).tolist(),
        "state": np.exp(-0.4 * t).round(8).tolist(),
        "rate": 0.4,
        "initial_state": 1.0,
    }

    entities = ["A", "B", "C", "D"]
    multi_table_case = {
        "fact": {
            "entity": [item for item in entities for _ in range(3)],
            "time": [f"2026-01-0{day}" for day in (1, 2, 3)] * len(entities),
            "value": [float(index + 1) for index in range(12)],
        },
        "dimension": {
            "entity": entities,
            "group": ["north", "north", "south", "south"],
        },
        "join": {"left_key": "entity", "right_key": "entity", "cardinality": "many_to_one"},
    }

    costs = np.array([2.0, 3.0])
    optimization_case = {
        "objective": costs.tolist(),
        "capacity": 10.0,
        "upper_bounds": [4.0, 3.0],
        "sense": "maximize",
    }

    regime_time = np.arange(40, dtype=float)
    regime_state = np.r_[
        0.15 * regime_time[:20],
        3.0 - 0.08 * (regime_time[20:] - 20),
    ] + rng.normal(0.0, 0.02, 40)
    dynamics_case = {
        "time": regime_time.tolist(),
        "state": regime_state.round(8).tolist(),
        "known": {"noise_scale": 0.02, "regime_change_index": 20},
    }

    return (
        SyntheticBenchmarkCase(
            "synthetic-data-regression", "data",
            "从观测特征估计连续目标，并检查开发段残差。",
            data_case, {"row_count": 24, "target_is_finite": True}, 20260908,
        ),
        SyntheticBenchmarkCase(
            "synthetic-pure-mechanistic", "pure_mechanistic",
            "验证一阶衰减方程的初值、单位和积分轨迹。",
            mechanistic_case, {"closed_form": True, "state_nonnegative": True}, 20260908,
        ),
        SyntheticBenchmarkCase(
            "synthetic-multi-table", "multi_table",
            "按实体和时间键关联事实表与维度表，不制造多对多膨胀。",
            multi_table_case, {"cardinality": "many_to_one", "has_time_key": True}, 20260908,
        ),
        SyntheticBenchmarkCase(
            "synthetic-optimization", "optimization",
            "在容量和上界约束下比较可行目标方案。",
            optimization_case, {"bounded": True, "feasible_reference": True}, 20260908,
        ),
        SyntheticBenchmarkCase(
            "synthetic-dynamics-regime", "dynamics",
            "在噪声与机制切换下区分数值失败和结构变点。",
            dynamics_case, {"noise_present": True, "regime_change": True}, 20260908,
        ),
    )


def validate_case_catalog(cases: tuple[SyntheticBenchmarkCase, ...] | None = None) -> dict[str, Any]:
    """Check fixture invariants without running a modeling algorithm."""
    cases = build_minimal_case_catalog() if cases is None else cases
    if len(cases) != 5 or len({case.case_id for case in cases}) != 5:
        return {"status": "invalid", "reason": "family_catalog_must_contain_five_unique_cases"}
    families = {case.family for case in cases}
    required = {"data", "pure_mechanistic", "multi_table", "optimization", "dynamics"}
    if families != required:
        return {"status": "invalid", "reason": "family_catalog_incomplete"}
    for case in cases:
        if not isinstance(case.data, Mapping) or not case.expected:
            return {"status": "invalid", "reason": "case_payload_missing"}
        if len(_fingerprint(case.data)) != 64:
            return {"status": "invalid", "reason": "case_fingerprint_invalid"}
    return {
        "status": "valid",
        "case_count": len(cases),
        "families": sorted(families),
        "catalog_fingerprint": _fingerprint([case.public() for case in cases]),
        "policy": "synthetic_protocol_fixture_not_real_world_accuracy_evidence",
    }


def build_diagnostic_case_catalog() -> tuple[SyntheticBenchmarkCase, ...]:
    """Return deterministic stress fixtures for failure attribution.

    These cases are deliberately small and synthetic.  They exercise hidden
    state, high observation noise, bias, regime switching, non-identifiability,
    contradictory constraints and pure statement compilation without claiming
    that any fixture represents a real competition problem.
    """
    import numpy as np

    t = np.arange(32, dtype=float)
    latent = np.sin(t / 5.0)
    return (
        SyntheticBenchmarkCase("diagnostic-hidden-state", "dynamics",
            "观测 x,y 由未观测状态 z 共同驱动。",
            {"time": t.tolist(), "x": (latent + 0.1 * t).tolist(), "y": (latent ** 2).tolist()},
            {"hidden_state": True}, 20260908),
        SyntheticBenchmarkCase("diagnostic-high-noise", "data",
            "观测含高噪声但不存在隐状态。",
            {"x": t.tolist(), "y": (2 * t + np.random.default_rng(4).normal(0, 3, len(t))).tolist()},
            {"noise_present": True, "hidden_state": False}, 20260908),
        SyntheticBenchmarkCase("diagnostic-observation-bias", "data",
            "观测值带固定偏置，机制值未直接观测。",
            {"x": t.tolist(), "observed": (np.cos(t / 4) + 0.75).tolist()},
            {"observation_bias": True}, 20260908),
        SyntheticBenchmarkCase("diagnostic-regime-switch", "dynamics",
            "同一变量在中点发生机制切换。",
            {"time": t.tolist(), "state": np.r_[0.2 * t[:16], 3.2 - 0.1 * (t[16:] - 16)].tolist()},
            {"regime_switch": True}, 20260908),
        SyntheticBenchmarkCase("diagnostic-nonidentifiable", "optimization",
            "两个参数只以乘积进入观测，单独不可辨识。",
            {"x": t.tolist(), "y": (6.0 * np.ones(len(t))).tolist()},
            {"nonidentifiable": True}, 20260908),
        SyntheticBenchmarkCase("diagnostic-contradictory-contract", "optimization",
            "约束同时要求 x≥2 与 x≤1。",
            {"variables": ["x"], "constraints": [{"op": ">=", "value": 2}, {"op": "<=", "value": 1}]},
            {"contradictory_constraints": True}, 20260908),
        SyntheticBenchmarkCase("diagnostic-pure-statement", "pure_mechanistic",
            "只有题面方程和边界，没有观测数据。",
            {"statement": "x'(t)=-k x(t), x(0)=1, k>0"},
            {"data_absent": True}, 20260908),
    )


__all__ = ["SyntheticBenchmarkCase", "build_minimal_case_catalog", "build_diagnostic_case_catalog", "validate_case_catalog"]
