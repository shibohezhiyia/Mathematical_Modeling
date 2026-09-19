"""Prospective delayed-derivative structures for first-run confirmation."""
from __future__ import annotations
import math
from .automated_benchmark import AutomatedBenchmarkCase

def build_delay_holdout(*,seed:int=20260926)->tuple[AutomatedBenchmarkCase,...]:
    if type(seed) is not int or not 0<=seed<=2**32-1: raise ValueError("holdout_seed_invalid")
    dt=0.2; specs=((2,[1,2],[0.0,-0.3,0.2]),(3,[1,4],[0.0,0.35,-0.035]),
                  (4,[2],[0.4,-0.25]),(5,[2,3],[0.0,0.15,-0.05])); result=[]
    for number,(lag,terms,coefficients) in enumerate(specs,1):
        states=[1.2+0.25*math.sin(1.7*i+0.3*number) for i in range(lag+1)]
        def advance(history):
            current,delayed=history[-1],history[-1-lag]; available=[1,current,delayed,current*current,current*delayed]
            derivative=sum(c*available[t] for c,t in zip(coefficients,[0,*terms]))
            return current+dt*derivative
        while len(states)<36: states.append(advance(states))
        observed=list(states)
        future=[advance(states)];states.append(future[-1]);future.append(advance(states))
        rows=[{"time":i*dt,"state":float(value)} for i,value in enumerate(observed)]
        result.append(AutomatedBenchmarkCase(f"delay-grammar-{number}","modeling_ode",
            "从等间隔状态记录合成稀疏滞后微分关系并预测后续两个网格点。",
            {"attachments":[{"name":"series","format":"records","rows":rows}],"query_times":[36*dt,37*dt]},
            {"family":"modeling_ode","structure":"delay_differential_polynomial","lag_time":lag*dt,
             "trajectory":future,"tolerance":2e-2},f"delay-grammar-{number}","internal-delay-holdout-v7"))
    for case in result: case.validate()
    return tuple(result)
__all__=["build_delay_holdout"]
