"""Prospective remaining depth-two topologies for search-strategy comparison."""
from __future__ import annotations
import math
from typing import Callable
import numpy as np
from .automated_benchmark import AutomatedBenchmarkCase

def build_search_strategy_holdout(*, seed: int = 20260925) -> tuple[AutomatedBenchmarkCase, ...]:
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1: raise ValueError("holdout_seed_invalid")
    scale=1.0+(seed%5)/20.0; xs=np.linspace(-1.2,1.2,31); queries=[-1.45,-0.55,0.25,1.35]
    probes=[-1.6,-1.05,-0.7,-0.15,0.45,0.8,1.1,1.55]
    specs: tuple[tuple[str,Callable[[float],float]],...]=(
        ("sin_sin",lambda x:0.1*scale+1.2*scale*math.sin(math.sin(x))),
        ("sin_exp",lambda x:-0.2*scale+0.8*scale*math.sin(math.exp(0.5*x))),
        ("cos_cos",lambda x:0.3*scale+1.4*scale*math.cos(math.cos(x))),
        ("cos_exp",lambda x:-0.1*scale+0.9*scale*math.cos(math.exp(x))),
        ("tanh_tanh",lambda x:0.2*scale+1.5*scale*math.tanh(math.tanh(2*x))),
        ("tanh_exp",lambda x:-0.3*scale+1.1*scale*math.tanh(math.exp(0.5*x))),
        ("exp_cos",lambda x:0.15*scale+0.7*scale*math.exp(math.cos(x))),
        ("exp_exp",lambda x:-0.4*scale+0.5*scale*math.exp(math.exp(0.5*x))),)
    result=[]
    for name,function in specs:
        rows=[{"input":float(x),"response":function(float(x))} for x in xs]
        result.append(AutomatedBenchmarkCase(f"strategy-{name}","modeling_algebra",
            "从观测合成未提供的二层算子拓扑，并在独立探针检查数学等价性。",
            {"attachments":[{"name":"raw","format":"records","rows":rows}],"query_inputs":queries},
            {"family":"modeling_algebra","structure":"compositional_symbolic","operator_signature":[],
             "input_variables":["input"],"predictions":[function(x) for x in queries],
             "equivalence_inputs":probes,"equivalence_outputs":[function(x) for x in probes],"tolerance":5e-3},
            f"strategy-{name}","internal-search-strategy-holdout-v6"))
    for case in result: case.validate()
    return tuple(result)
__all__=["build_search_strategy_holdout"]
