"""Prospective compositional holdout scored by hidden mathematical-equivalence probes."""
from __future__ import annotations
import math
from typing import Any, Callable, Mapping, Sequence
import numpy as np
from .automated_benchmark import AutomatedBenchmarkCase

def _table(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"name": "measurements", "format": "records", "rows": [dict(row) for row in rows]}

def build_equivalence_holdout(*, seed: int = 20260923) -> tuple[AutomatedBenchmarkCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1: raise ValueError("holdout_seed_invalid")
    scale = 1.0 + (seed % 9) / 25.0
    xs, queries = np.linspace(-2.2, 2.2, 27), [-2.55, -0.45, 1.15, 2.5]
    probes = [-2.8, -1.9, -1.1, -0.1, 0.35, 0.95, 1.55, 2.15, 2.75]
    specs: tuple[tuple[str, Callable[[float], float]], ...] = (
        ("tanh_cos_phase", lambda x: 0.15 * scale + 1.4 * scale * math.tanh(math.cos(0.5 * x + 0.3))),
        ("exp_sin", lambda x: -0.25 * scale + 0.7 * scale * math.exp(math.sin(x))),
        ("cos_tanh", lambda x: 0.35 * scale + 1.6 * scale * math.cos(math.tanh(0.5 * x))),
        ("sin_cos", lambda x: -0.2 * scale + 1.2 * scale * math.sin(math.cos(2.0 * x))),
    )
    result = []
    for name, function in specs:
        result.append(AutomatedBenchmarkCase(
            f"equivalence-{name}", "modeling_algebra",
            "从观测合成嵌套算子表达式；评分按独立数学探针，不要求与参考树具有相同表面写法。",
            {"attachments": [_table([{"input": float(x), "response": function(float(x))} for x in xs])],
             "query_inputs": queries},
            {"family": "modeling_algebra", "structure": "compositional_symbolic",
             "input_variables": ["input"], "operator_signature": [],
             "predictions": [function(x) for x in queries], "equivalence_inputs": probes,
             "equivalence_outputs": [function(x) for x in probes], "tolerance": 4e-3},
            f"equivalence-{name}", "internal-equivalence-holdout-v5"))
    for case in result: case.validate()
    return tuple(result)

__all__ = ["build_equivalence_holdout"]
