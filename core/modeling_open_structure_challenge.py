"""Second-generation one-shot challenges intentionally outside the current structure library."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .automated_benchmark import AutomatedBenchmarkCase


def _table(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"name": name, "format": "records", "rows": [dict(row) for row in rows]}


def build_modeling_open_structure_challenge(*, seed: int = 20260920) -> tuple[AutomatedBenchmarkCase, ...]:
    """Build four unsupported structures without adapting the candidate implementation."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("challenge_seed_invalid")
    scale = 1.0 + (seed % 11) / 25.0
    cases: list[AutomatedBenchmarkCase] = []

    xs = np.linspace(-2.5, 2.5, 13)
    queries = [-2.75, -0.4, 1.7, 2.9]
    wave = lambda x: scale * (0.4 + 1.8 * math.sin(1.2 * x))
    cases.append(AutomatedBenchmarkCase(
        "open-challenge-algebra-sinusoid", "modeling_algebra",
        "从原始观测识别周期关系并预测查询点；不得用低阶多项式冒充周期结构。",
        {"attachments": [_table("sensor", [{"input": float(x), "response": wave(float(x))} for x in xs])],
         "query_inputs": queries},
        {"family": "modeling_algebra", "structure": "sinusoidal",
         "input_variables": ["input"], "coefficients": [1.8 * scale, 1.2, 0.4 * scale],
         "predictions": [wave(x) for x in queries], "tolerance": 2e-3},
        "open-algebra-periodic", "internal-open-structure-challenge-v2",
    ))

    times = np.linspace(0.0, 6.0, 15)
    capacity, rate, initial = 12.0 * scale, 0.45, 1.5 * scale
    gompertz = lambda t: capacity * math.exp(math.log(initial / capacity) * math.exp(-rate * t))
    query_times = [6.5, 7.0]
    cases.append(AutomatedBenchmarkCase(
        "open-challenge-ode-gompertz", "modeling_ode",
        "根据状态观测区分非对称饱和增长机制并预测，提交机制与参数。",
        {"attachments": [_table("series", [{"time": float(t), "state": gompertz(float(t))} for t in times])],
         "query_times": query_times},
        {"family": "modeling_ode", "structure": "gompertz_growth",
         "parameters": {"capacity": capacity, "rate": rate},
         "trajectory": [gompertz(t) for t in query_times], "tolerance": 3e-2},
        "open-ode-gompertz", "internal-open-structure-challenge-v2",
    ))

    cases.append(AutomatedBenchmarkCase(
        "open-challenge-optimization-fixed-charge", "modeling_optimization",
        "生产某产品前需支付一次固定启动成本；构造数量与启用变量，在容量内最大化净收益。",
        {"attachments": [
            _table("products", [{"item": "A", "profit": 7.0, "fixed_cost": 5.0, "labor": 2.0},
                                {"item": "B", "profit": 4.0, "fixed_cost": 1.0, "labor": 1.0}]),
            _table("limits", [{"resource": "labor", "capacity": 5.0}]),
        ], "integer_decisions": True},
        {"family": "modeling_optimization", "structure": "fixed_charge_milp",
         "decision_variables": ["A_qty", "B_qty", "A_open", "B_open"],
         "constraint_ids": ["labor", "A_link", "B_link"],
         "decision_units": {"A_qty": "item", "B_qty": "item", "A_open": "binary", "B_open": "binary"},
         "objective_unit": "currency", "objective": 19.0,
         "objective_coefficients": [7.0, 4.0, -5.0, -1.0],
         "bounds": [[0, 2], [0, 5], [0, 1], [0, 1]],
         "A_ub": [[2, 1, 0, 0], [1, 0, -2, 0], [0, 1, 0, -5]],
         "b_ub": [5, 0, 0], "direction": "maximize", "integer_indices": [0, 1, 2, 3],
         "known_solution": [0, 5, 0, 1], "tolerance": 1e-8},
        "open-optimization-fixed-charge", "internal-open-structure-challenge-v2",
    ))

    cases.append(AutomatedBenchmarkCase(
        "open-challenge-multitable-weighted-bridge", "modeling_multi_table",
        "订单通过带权桥表分配到多个标签，按标签组汇总且每单权重和为一。",
        {"attachments": [
            _table("orders", [{"order": "O1", "amount": 10.0}, {"order": "O2", "amount": 6.0}]),
            _table("allocation", [{"order": "O1", "tag": "T1", "weight": 0.25},
                                  {"order": "O1", "tag": "T2", "weight": 0.75},
                                  {"order": "O2", "tag": "T1", "weight": 1.0}]),
            _table("tags", [{"tag": "T1", "group": "G1"}, {"tag": "T2", "group": "G2"}]),
        ]},
        {"family": "modeling_multi_table", "structure": "weighted_bridge_group_sum",
         "join_keys": ["order", "tag"], "aggregation": "weighted_sum", "point_in_time": False,
         "rows": 3, "group_totals": {"G1": 8.5, "G2": 7.5}, "tolerance": 1e-8},
        "open-multitable-weighted-bridge", "internal-open-structure-challenge-v2",
    ))
    for case in cases:
        case.validate()
    return tuple(cases)


__all__ = ["build_modeling_open_structure_challenge"]
