"""Compile explicitly selected table measures into a typed affine search seed.

This is a generic function-discovery starting point, not a domain-specific
solver, causal model or automatic natural-language interpretation.
"""
import math
import json

import numpy as np
import pandas as pd

from .graph_experiments import SearchExperiment
from .graph_search_artifacts import BUNDLE_VERSION
from .interactive_visualization import InteractiveVisualizationCompiler
from .model_hypotheses import HypothesisIR, ProblemContract, _keys, _require

# Multiplicative units only. Unknown units and affine temperature conversions
# cannot be treated as dimensionless or silently approximated.
UNITS = {"1": ({}, 1.), "count": ({}, 1.), "m": ({"L": 1}, 1.), "cm": ({"L": 1}, .01),
         "km": ({"L": 1}, 1000.), "s": ({"T": 1}, 1.), "min": ({"T": 1}, 60.),
         "h": ({"T": 1}, 3600.), "kg": ({"M": 1}, 1.), "g": ({"M": 1}, .001),
         "m/s": ({"L": 1, "T": -1}, 1.), "km/h": ({"L": 1, "T": -1}, 1/3.6),
         "K": ({"Theta": 1}, 1.)}


def table_fields(frame):
    _require(isinstance(frame, pd.DataFrame), "no_current_table")
    _require(frame.columns.is_unique and len(frame.columns) <= 500, "table_schema_not_supported")
    columns = []
    for name in frame.columns:
        if type(name) is not str or not 0 < len(name) <= 100:
            continue
        dtype = frame[name].dtype
        if (pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype)
                and not pd.api.types.is_complex_dtype(dtype)):
            columns.append({"name": name, "identifier": InteractiveVisualizationCompiler._looks_identifier(name)})
    return {"rows": len(frame), "columns": columns, "units": list(UNITS), "maximum_development_rows": 360}


def compile_table_graph(frame, config):
    fields = table_fields(frame)
    _keys(config, {"problem", "inputs", "target", "units", "start_row", "row_count",
                   "absolute_tolerance", "relative_tolerance"})
    inputs, target = config["inputs"], config["target"]
    _require(type(inputs) is list and 1 <= len(inputs) <= 4 and all(type(c) is str for c in inputs), "input_columns_required")
    _require(type(target) is str and target not in inputs and len(set(inputs)) == len(inputs), "distinct_columns_required")
    names = inputs + [target]
    available = {c["name"]: c for c in fields["columns"]}
    _require(all(c in available for c in names), "numeric_columns_required")
    _require(not any(available[c]["identifier"] for c in names), "identifier_is_not_a_measure")
    _keys(config["units"], set(names))
    _require(all(type(u) is str and u in UNITS for u in config["units"].values()), "explicit_supported_units_required")
    start, count = config["start_row"], config["row_count"]
    _require(type(start) is int and start >= 0 and type(count) is int and 12 <= count <= 360
             and start + count <= len(frame), "invalid_development_window")
    _require(type(config["problem"]) is str and 0 < len(config["problem"].strip()) <= 10000, "problem_text_required")
    # Bound rows BEFORE numeric conversion; no full-table array allocation.
    selected = frame.iloc[start:start+count][names].copy()
    values = {}
    for c in names:
        array = selected[c].to_numpy(dtype=float, na_value=np.nan)
        _require(bool(np.all(np.isfinite(array))) and bool(np.all(np.abs(array) <= 1e12)), "missing_or_nonfinite_table_values")
        array = array * UNITS[config["units"][c]][1]
        _require(bool(np.all(np.isfinite(array))) and bool(np.all(np.abs(array) <= 1e12)), "converted_values_out_of_range")
        values[c] = array
    _require(len({tuple(float(values[c][i]) for c in inputs) for i in range(count)}) == count,
             "repeated_input_requires_grouped_design")
    cut = int(count * .7)
    for c in inputs:
        _require(float(np.ptp(values[c][:cut])) > 0, "constant_training_input")
    contract = ProblemContract.create(config["problem"])
    assumptions = [{"id": "binding", "text": "用户指定这些列为连续数值输入和目标，单位换算到 SI；不主张因果关系。"},
        {"id": "seed", "text": "仿射关系仅为结构搜索起点；系数搜索盒由训练数据尺度生成，不是题面物理边界。"},
        {"id": "split", "text": "所选行范围前70%训练、后30%开发检查，不保证时间或实体独立；不包含最终留出数据。"}]
    output_dims = UNITS[config["units"][target]][0]
    def node(key, op, dims, arguments=(), attributes=None):
        return {"id": key, "op": op, "inputs": list(arguments), "type": {"dtype": "real", "shape": [], "dimensions": dims},
                "attributes": attributes or {}, "assumption_ids": ["binding", "seed", "split"],
                "context_fact_ids": [f["id"] for f in contract.public()["facts"]]}
    yscale = max(float(np.max(np.abs(values[target][:cut]))), float(np.ptp(values[target][:cut])), 1e-6)
    bbound = min(100 * yscale, 1e12)
    nodes = [node("bias", "parameter", output_dims, attributes={"name": "截距", "role": "parameter"})]
    bounds = {"bias": [-bbound, bbound, float(np.mean(values[target][:cut]))]}
    domain, bindings = {}, []
    accumulator = "bias"
    for i, c in enumerate(inputs):
        dims = UNITS[config["units"][c]][0]
        coefficient_dims = {d: output_dims.get(d, 0) - dims.get(d, 0) for d in set(output_dims) | set(dims)}
        coefficient_dims = {d: exponent for d, exponent in coefficient_dims.items() if exponent}
        x, a, term, total = f"x{i}", f"a{i}", f"term{i}", f"sum{i}"
        nodes.extend([node(x, "variable", dims, attributes={"name": c, "role": "observed"}),
            node(a, "parameter", coefficient_dims, attributes={"name": f"系数{i}", "role": "parameter"}),
            node(term, "multiply", output_dims, (a, x)), node(total, "add", output_dims, (accumulator, term))])
        bound = min(100 * yscale / float(np.ptp(values[c][:cut])), 1e12)
        _require(math.isfinite(bound) and bound > 0, "parameter_scale_out_of_range")
        bounds[a] = [-bound, bound, 0.]
        domain[x] = [float(np.min(values[c])), float(np.max(values[c]))]
        bindings.append({"node": x, "column": c, "unit": config["units"][c], "si_factor": UNITS[config["units"][c]][1]})
        accumulator = total
    nodes.append(node("y", "observation", output_dims, (accumulator,)))
    assumptions[0]["text"] += " 绑定记录：" + json.dumps({"inputs": bindings,
        "output": {"column": target, "unit": config["units"][target], "si_factor": UNITS[config["units"][target]][1]},
        "start_row": start, "selected_rows": count, "source_rows": len(frame)}, ensure_ascii=False)
    graph = HypothesisIR.from_payload({"id": "affine_seed", "contract_hash": contract.digest, "nodes": nodes,
                                      "outputs": ["y"], "assumptions": assumptions}, contract)
    cases = [{"id": f"row_{start+i}", "bindings": {f"x{j}": float(values[c][i]) for j, c in enumerate(inputs)},
              "expected": {"y": float(values[target][i])}} for i in range(count)]
    experiment = SearchExperiment.create(contract, graph, domain=domain, parameter_bounds=bounds,
        training_cases=cases[:cut], search_cases=cases[cut:], probe_count=12,
        absolute_tolerance=config["absolute_tolerance"], relative_tolerance=config["relative_tolerance"])
    return {"bundle": {"schema_version": BUNDLE_VERSION, "problem": contract.public(), "experiment": experiment.public(),
                       "candidates": [graph.payload()], "budget": {"wall_seconds": 60}, "grammar_search": True},
            "binding": {"inputs": bindings, "output": {"node": "y", "column": target, "unit": config["units"][target],
                         "si_factor": UNITS[config["units"][target]][1]}, "source_rows": len(frame), "start_row": start,
                         "selected_rows": count, "training_rows": cut, "search_rows": count-cut,
                         "full_table_analyzed": count == len(frame), "row_order_independence_verified": False,
                         "bounds_are_training_scale_search_boxes": True,
                         "absolute_tolerance_in_si_output_units": config["absolute_tolerance"]}}
