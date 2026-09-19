"""Prospective depth-three topologies not used by the v9 beam evaluation."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .automated_benchmark import AutomatedBenchmarkCase


def build_depth_three_holdout(*, seed: int = 20260929) -> tuple[AutomatedBenchmarkCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("holdout_seed_invalid")
    scale = 1.0 + (seed % 11) / 40.0
    xs = np.linspace(-1.45, 1.45, 37)
    queries = [-1.7, -0.55, 0.4, 1.65]
    probes = [-1.9, -1.3, -0.8, -0.2, 0.2, 0.75, 1.3, 1.85]
    specs: tuple[tuple[str, Callable[[float], float]], ...] = (
        ("sin_sin_cos", lambda x: 0.13 * scale + 1.12 * scale * math.sin(math.sin(math.cos(0.85 * x + 0.1)))),
        ("sin_cos_exp", lambda x: -0.17 * scale + 0.72 * scale * math.sin(math.cos(math.exp(0.45 * x - 0.1)))),
        ("cos_cos_sin", lambda x: 0.21 * scale + 1.18 * scale * math.cos(math.cos(math.sin(0.9 * x)))),
        ("cos_exp_tanh", lambda x: -0.12 * scale + 0.92 * scale * math.cos(math.exp(math.tanh(0.65 * x)))),
        ("tanh_tanh_cos", lambda x: 0.15 * scale + 1.30 * scale * math.tanh(math.tanh(math.cos(1.05 * x - 0.2)))),
        ("tanh_exp_sin", lambda x: -0.19 * scale + 0.84 * scale * math.tanh(math.exp(math.sin(0.7 * x + 0.1)))),
        ("exp_cos_tanh", lambda x: 0.08 * scale + 0.58 * scale * math.exp(math.cos(math.tanh(0.8 * x)))),
        ("exp_tanh_cos", lambda x: -0.24 * scale + 0.66 * scale * math.exp(math.tanh(math.cos(0.75 * x + 0.15)))),
    )
    result = []
    for name, function in specs:
        rows = [{"input": float(x), "response": function(float(x))} for x in xs]
        result.append(AutomatedBenchmarkCase(
            f"depth3-{name}", "modeling_algebra",
            "从原始观测合成未给定的三层算子结构，并在训练区间外的隐藏点复算。",
            {"attachments": [{"name": "observed_series", "format": "records", "rows": rows}],
             "query_inputs": queries},
            {"family": "modeling_algebra", "structure": "compositional_symbolic",
             "input_variables": ["input"], "operator_signature": [],
             "predictions": [function(x) for x in queries],
             "equivalence_inputs": probes,
             "equivalence_outputs": [function(x) for x in probes], "tolerance": 5e-3},
            f"depth3-{name}", "internal-depth-three-holdout-v10",
        ))
    for case in result:
        case.validate()
    return tuple(result)


__all__ = ["build_depth_three_holdout"]
