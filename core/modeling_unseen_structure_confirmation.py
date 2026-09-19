"""Prospective structures for a freeze-before-first-execution internal confirmation."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence
import numpy as np

from .automated_benchmark import AutomatedBenchmarkCase


def _table(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"name": name, "format": "records", "rows": [dict(row) for row in rows]}


def build_unseen_structure_confirmation(*, seed: int = 20260921) -> tuple[AutomatedBenchmarkCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("confirmation_seed_invalid")
    scale = 1.0 + (seed % 13) / 30.0
    rational = lambda x: scale * (1.0 + 1.7 * x) / (1.0 + 0.3 * x)
    xs, queries = np.linspace(-1.5, 3.0, 14), [-1.7, 0.4, 3.4]
    algebra = AutomatedBenchmarkCase(
        "unseen-algebra-rational", "modeling_algebra",
        "识别带一阶分子和分母的有理关系，并在无奇点查询处预测。",
        {"attachments": [_table("observations", [{"input": float(x), "response": rational(float(x))} for x in xs])],
         "query_inputs": queries},
        {"family": "modeling_algebra", "structure": "rational_1_1", "input_variables": ["input"],
         "coefficients": [scale, 1.7 * scale, 0.3], "predictions": [rational(x) for x in queries],
         "tolerance": 2e-3}, "unseen-algebra-rational", "internal-unseen-structure-v3")

    times = np.linspace(0.0, 7.0, 29)
    values = [1.0 * scale]
    delay, rate, capacity = 4, 0.35, 10.0 * scale
    step = float(times[1] - times[0])
    for index in range(1, len(times)):
        delayed = values[max(0, index - delay)]
        values.append(values[-1] + step * rate * values[-1] * (1.0 - delayed / capacity))
    ode = AutomatedBenchmarkCase(
        "unseen-ode-delayed-growth", "modeling_ode",
        "增长受过去状态的延迟反馈控制；识别延迟机制并预测后续状态。",
        {"attachments": [_table("series", [{"time": float(t), "state": float(y)} for t, y in zip(times, values)])],
         "query_times": [7.25, 7.5]},
        {"family": "modeling_ode", "structure": "delayed_logistic", "parameters": {"rate": rate, "delay": 1.0},
         "trajectory": [float(values[-1]), float(values[-1])], "tolerance": 0.5},
        "unseen-ode-delay", "internal-unseen-structure-v3")

    optimization = AutomatedBenchmarkCase(
        "unseen-optimization-cardinality", "modeling_optimization",
        "最多启用一种产品；构造数量与启用变量，在资源容量内最大化利润。",
        {"attachments": [_table("products", [{"item": "A", "profit": 6.0, "labor": 2.0},
                                               {"item": "B", "profit": 5.0, "labor": 1.0}]),
                         _table("limits", [{"resource": "labor", "capacity": 4.0}])],
         "integer_decisions": True, "maximum_active_items": 1},
        {"family": "modeling_optimization", "structure": "cardinality_constrained_milp",
         "decision_variables": ["A_qty", "B_qty", "A_open", "B_open"],
         "constraint_ids": ["labor", "A_link", "B_link", "cardinality"],
         "decision_units": {"A_qty": "item", "B_qty": "item", "A_open": "binary", "B_open": "binary"},
         "objective_unit": "currency", "objective": 20.0, "objective_coefficients": [6, 5, 0, 0],
         "bounds": [[0, 2], [0, 4], [0, 1], [0, 1]],
         "A_ub": [[2, 1, 0, 0], [1, 0, -2, 0], [0, 1, 0, -4], [0, 0, 1, 1]],
         "b_ub": [4, 0, 0, 1], "direction": "maximize", "known_solution": [0, 4, 0, 1],
         "integer_indices": [0, 1, 2, 3], "tolerance": 1e-8},
        "unseen-optimization-cardinality", "internal-unseen-structure-v3")

    multitable = AutomatedBenchmarkCase(
        "unseen-multitable-validity-interval", "modeling_multi_table",
        "按事件时刻落入的有效起止区间关联分组并汇总，区间外不得沿用旧记录。",
        {"attachments": [_table("events", [{"entity": "E1", "event_time": "2026-01-10", "amount": 3.0},
                                             {"entity": "E1", "event_time": "2026-03-10", "amount": 5.0}]),
                         _table("assignments", [{"entity": "E1", "effective_from": "2026-01-01",
                                                 "effective_to": "2026-02-01", "segment": "active"}])]},
        {"family": "modeling_multi_table", "structure": "validity_interval_group_sum",
         "join_keys": ["entity"], "aggregation": "sum", "point_in_time": True, "rows": 1,
         "group_totals": {"active": 3.0}, "tolerance": 1e-8},
        "unseen-multitable-validity-interval", "internal-unseen-structure-v3")
    result = (algebra, ode, optimization, multitable)
    for case in result:
        case.validate()
    return result


__all__ = ["build_unseen_structure_confirmation"]
