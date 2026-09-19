"""Prospective operator-topology holdout for the bounded expression grammar."""
from __future__ import annotations
import math
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from .automated_benchmark import AutomatedBenchmarkCase

def _table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"name": "raw_observations", "format": "records", "rows": [dict(row) for row in rows]}

def build_compositional_holdout(*, seed: int = 20260922) -> tuple[AutomatedBenchmarkCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("holdout_seed_invalid")
    scale = 1.0 + (seed % 7) / 20.0
    xs = np.linspace(-2.0, 2.0, 25)
    queries = [-2.3, -0.65, 0.85, 2.25]
    specifications: tuple[tuple[str, list[str], Callable[[float], float]], ...] = (
        ("tanh_of_sin", ["tanh", "sin"], lambda x: 0.2 * scale + 1.7 * scale * math.tanh(math.sin(x))),
        ("exp_of_tanh", ["exp", "tanh"], lambda x: -0.4 * scale + 0.8 * scale * math.exp(math.tanh(0.5 * x))),
        ("sin_of_tanh", ["sin", "tanh"], lambda x: 0.3 * scale + 1.3 * scale * math.sin(math.tanh(2.0 * x))),
        ("cos_of_sin", ["cos", "sin"], lambda x: -0.1 * scale + 2.1 * scale * math.cos(math.sin(x))),
    )
    cases = []
    for identifier, signature, function in specifications:
        cases.append(AutomatedBenchmarkCase(
            f"compositional-{identifier}", "modeling_algebra",
            "从原始观测合成有界算子表达式；算子可嵌套，不提供候选拓扑。",
            {"attachments": [_table([{"input": float(x), "response": function(float(x))} for x in xs])],
             "query_inputs": queries},
            {"family": "modeling_algebra", "structure": "compositional_symbolic",
             "operator_signature": signature, "input_variables": ["input"],
             "predictions": [function(x) for x in queries], "tolerance": 3e-3},
            f"compositional-{identifier}", "internal-compositional-holdout-v4"))
    for case in cases: case.validate()
    return tuple(cases)

__all__ = ["build_compositional_holdout"]
