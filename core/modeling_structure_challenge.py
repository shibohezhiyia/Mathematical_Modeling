"""Post-development structure challenges for bounded automatic modeling."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .automated_benchmark import AutomatedBenchmarkCase


def _table(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"name": name, "format": "records", "rows": [dict(row) for row in rows]}


def build_modeling_structure_challenge(*, seed: int = 20260917) -> tuple[AutomatedBenchmarkCase, ...]:
    """Return eight new structures; the seed is bound for protocol identity."""
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("challenge_seed_invalid")
    scale = 1.0 + (seed % 7) / 20.0
    cases: list[AutomatedBenchmarkCase] = []

    xs = [-3, -2, -1, 0, 1, 2, 3]
    queries = [-2.5, 1.5, 4.0]
    cubic = lambda x: scale * (x**3 - 2.0 * x + 1.0)
    cases.append(AutomatedBenchmarkCase(
        "challenge-algebra-cubic", "modeling_algebra",
        "从原始观测识别最低充分次数的多项式，并预测查询点；不得预先假定最高只有二次。",
        {"attachments": [_table("measurements", [{"input": x, "response": cubic(x)} for x in xs])],
         "query_inputs": queries},
        {"family": "modeling_algebra", "structure": "cubic", "input_variables": ["input"],
         "coefficients": [scale, 0.0, -2.0 * scale, scale],
         "predictions": [cubic(x) for x in queries], "tolerance": 1e-6},
        "challenge-algebra-cubic", "post-development-structure-challenge-v1",
    ))
    points = [(a, b, c) for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)]
    tri = lambda a, b, c: scale * (1.0 + 2.0 * a * b * c)
    tri_queries = [[0.5, -1.0, 2.0], [2.0, 1.5, -0.5]]
    cases.append(AutomatedBenchmarkCase(
        "challenge-algebra-three-way", "modeling_algebra",
        "识别三个变量共同作用、但任何低阶子集都不足以解释的关系，并预测新组合。",
        {"attachments": [_table("raw", [{"x1": a, "x2": b, "x3": c, "response": tri(a, b, c)}
                                             for a, b, c in points])], "query_inputs": tri_queries},
        {"family": "modeling_algebra", "structure": "trilinear", "input_variables": ["x1", "x2", "x3"],
         "coefficients": [scale, 2.0 * scale],
         "predictions": [tri(*values) for values in tri_queries], "tolerance": 1e-6},
        "challenge-algebra-three-way", "post-development-structure-challenge-v1",
    ))

    times = np.linspace(0.0, 5.0, 11)
    driven = 2.0 + (1.0 - 2.0) * np.exp(-0.5 * times)
    query_times = [5.5, 6.0]
    cases.append(AutomatedBenchmarkCase(
        "challenge-ode-external-drive", "modeling_ode",
        "状态受恒定外部输入驱动并同时线性耗散。根据观测识别平衡点与速率并预测。",
        {"attachments": [_table("series", [{"time": float(t), "state": float(y)}
                                                for t, y in zip(times, driven)])],
         "query_times": query_times},
        {"family": "modeling_ode", "structure": "externally_driven_linear",
         "parameters": {"rate": 0.5, "forcing": 1.0},
         "trajectory": [float(2.0 - math.exp(-0.5 * t)) for t in query_times], "tolerance": 2e-3},
        "challenge-ode-external-drive", "post-development-structure-challenge-v1",
    ))
    coupled_rows = [{"time": float(t), "state_a": float(math.exp(-t) * math.cos(t)),
                     "state_b": float(math.exp(-t) * math.sin(t))} for t in np.linspace(0, 3, 10)]
    cases.append(AutomatedBenchmarkCase(
        "challenge-ode-coupled-state", "modeling_ode",
        "两个状态相互耦合并衰减。识别跨状态动力学并给出两个状态的后续轨迹。",
        {"attachments": [_table("coupled", coupled_rows)], "query_times": [3.5, 4.0]},
        {"family": "modeling_ode", "structure": "coupled_linear",
         "parameters": {"decay": 1.0, "coupling": 1.0},
         "trajectory": [[math.exp(-t) * math.cos(t), math.exp(-t) * math.sin(t)] for t in (3.5, 4.0)],
         "tolerance": 2e-3},
        "challenge-ode-coupled-state", "post-development-structure-challenge-v1",
    ))

    cases.append(AutomatedBenchmarkCase(
        "challenge-optimization-integer", "modeling_optimization",
        "生产批次数必须为非负整数，在工时不超过 3 时最大化利润。提交整数可行方案。",
        {"attachments": [_table("items", [{"item": "A", "profit": 5.0, "labor": 2.0},
                                             {"item": "B", "profit": 3.0, "labor": 1.0}]),
                         _table("limits", [{"resource": "labor", "capacity": 3.0}])],
         "integer_decisions": True},
        {"family": "modeling_optimization", "structure": "integer_resource_allocation_milp",
         "decision_variables": ["A", "B"], "constraint_ids": ["labor"],
         "decision_units": {"A": "item", "B": "item"}, "objective_unit": "currency",
         "objective": 9.0, "objective_coefficients": [5.0, 3.0], "bounds": [[0, 1.5], [0, 3]],
         "A_ub": [[2.0, 1.0]], "b_ub": [3.0], "direction": "maximize",
         "known_solution": [0.0, 3.0], "integer_indices": [0, 1], "tolerance": 1e-8},
        "challenge-optimization-integer", "post-development-structure-challenge-v1",
    ))
    cases.append(AutomatedBenchmarkCase(
        "challenge-optimization-equality", "modeling_optimization",
        "恰好采购 4 单位且资源使用不超过 4，选择非负数量使成本最低；必须表达等式约束。",
        {"attachments": [_table("options", [{"item": "A", "cost": 2.0, "labor": 1.0},
                                               {"item": "B", "cost": 3.0, "labor": 1.0}]),
                         _table("limits", [{"resource": "labor", "capacity": 4.0}])],
         "required_total": 4.0},
        {"family": "modeling_optimization", "structure": "equality_constrained_allocation_lp",
         "decision_variables": ["A", "B"],
         "constraint_ids": ["labor", "balance_lower", "balance_upper"],
         "decision_units": {"A": "item", "B": "item"}, "objective_unit": "currency",
         "objective": 8.0, "objective_coefficients": [2.0, 3.0], "bounds": [[0, 4], [0, 4]],
         "A_ub": [[1.0, 1.0], [-1.0, -1.0], [1.0, 1.0]],
         "b_ub": [4.0, -4.0, 4.0], "direction": "minimize",
         "known_solution": [4.0, 0.0], "tolerance": 1e-8},
        "challenge-optimization-equality", "post-development-structure-challenge-v1",
    ))

    cases.append(AutomatedBenchmarkCase(
        "challenge-multitable-three-chain", "modeling_multi_table",
        "通过实体表和地区表两级关系，按大区汇总事实金额；提交完整三表连接链。",
        {"attachments": [
            _table("facts", [{"entity": "E1", "amount": 2.0}, {"entity": "E2", "amount": 3.0},
                              {"entity": "E3", "amount": 4.0}]),
            _table("entities", [{"entity": "E1", "region": "R1"}, {"entity": "E2", "region": "R1"},
                                 {"entity": "E3", "region": "R2"}]),
            _table("regions", [{"region": "R1", "zone": "east"}, {"region": "R2", "zone": "west"}]),
        ]},
        {"family": "modeling_multi_table", "structure": "three_table_chain_group_sum",
         "join_keys": ["entity", "region"], "aggregation": "sum", "point_in_time": False,
         "rows": 3, "group_totals": {"east": 5.0, "west": 4.0}, "tolerance": 1e-8},
        "challenge-multitable-three-chain", "post-development-structure-challenge-v1",
    ))
    cases.append(AutomatedBenchmarkCase(
        "challenge-multitable-bridge-ambiguity", "modeling_multi_table",
        "订单可带多个标签。按标签组汇总金额；若未说明重复计数或分摊规则，必须指出缺失口径。",
        {"attachments": [
            _table("orders", [{"order": "O1", "amount": 10.0}, {"order": "O2", "amount": 6.0}]),
            _table("order_tags", [{"order": "O1", "tag": "T1"}, {"order": "O1", "tag": "T2"},
                                    {"order": "O2", "tag": "T1"}]),
            _table("tags", [{"tag": "T1", "group": "G1"}, {"tag": "T2", "group": "G2"}]),
        ]},
        {"family": "modeling_multi_table", "expected_status": "needs_input",
         "missing": ["many_to_many_allocation_rule"], "tolerance": 0.0},
        "challenge-multitable-bridge-ambiguity", "post-development-structure-challenge-v1",
    ))
    for case in cases:
        case.validate()
    return tuple(cases)


__all__ = ["build_modeling_structure_challenge"]
