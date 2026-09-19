"""Prospective unseen depth-three topologies for equal-budget beam evaluation."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .automated_benchmark import AutomatedBenchmarkCase


def build_beam_holdout(*, seed: int = 20260928) -> tuple[AutomatedBenchmarkCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("holdout_seed_invalid")
    scale = 1.0 + (seed % 7) / 30.0
    xs = np.linspace(-1.6, 1.6, 35)
    queries = [-1.85, -0.65, 0.35, 1.75]
    probes = [-2.0, -1.4, -0.9, -0.25, 0.15, 0.7, 1.25, 1.9]
    specs: tuple[tuple[str, Callable[[float], float]], ...] = (
        ("sin_cos_tanh", lambda x: 0.12 * scale + 1.15 * scale * math.sin(math.cos(math.tanh(0.8 * x + 0.1)))),
        ("sin_tanh_cos", lambda x: -0.18 * scale + 0.95 * scale * math.sin(math.tanh(math.cos(1.1 * x - 0.2)))),
        ("cos_sin_tanh", lambda x: 0.22 * scale + 1.25 * scale * math.cos(math.sin(math.tanh(0.7 * x + 0.25)))),
        ("cos_tanh_sin", lambda x: -0.11 * scale + 1.05 * scale * math.cos(math.tanh(math.sin(1.3 * x)))),
        ("tanh_sin_cos", lambda x: 0.16 * scale + 1.35 * scale * math.tanh(math.sin(math.cos(0.9 * x - 0.15)))),
        ("tanh_cos_sin", lambda x: -0.14 * scale + 1.10 * scale * math.tanh(math.cos(math.sin(1.2 * x + 0.05)))),
        ("exp_sin_tanh", lambda x: -0.28 * scale + 0.62 * scale * math.exp(math.sin(math.tanh(0.75 * x)))),
        ("tanh_exp_cos", lambda x: 0.09 * scale + 0.88 * scale * math.tanh(math.exp(math.cos(0.6 * x + 0.2)))),
    )
    result = []
    for name, function in specs:
        rows = [{"input": float(x), "response": function(float(x))} for x in xs]
        result.append(AutomatedBenchmarkCase(
            f"beam-{name}", "modeling_algebra",
            "从原始观测合成未提供的三层算子拓扑；用独立探针检验数学等价性。",
            {"attachments": [{"name": "measurements", "format": "records", "rows": rows}],
             "query_inputs": queries},
            {"family": "modeling_algebra", "structure": "compositional_symbolic",
             "input_variables": ["input"], "operator_signature": [],
             "predictions": [function(x) for x in queries],
             "equivalence_inputs": probes,
             "equivalence_outputs": [function(x) for x in probes], "tolerance": 5e-3},
            f"beam-{name}", "internal-beam-holdout-v9",
        ))
    for case in result:
        case.validate()
    return tuple(result)


__all__ = ["build_beam_holdout"]
