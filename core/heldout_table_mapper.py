"""Map heldout table measures onto an already frozen SI numerical contract."""
import numpy as np

from .graph_confirmation import FrozenGraphModel, HeldoutCases, HOLDOUT_VERSION
from .model_hypotheses import MathType, _keys, _require
from .table_graph_compiler import UNITS, table_fields


def frozen_table_roles(model):
    model = FrozenGraphModel.from_payload(model.public())
    graph = model.public()["hypothesis"]
    nodes = {n["id"]: n for n in graph["nodes"]}
    def role(node):
        dimensions = MathType.parse(node["type"]).public()["dimensions"]
        return {"node": node["id"], "name": node["attributes"].get("name", node["id"]),
                "dimensions": dimensions,
                "units": [unit for unit, (dims, _) in UNITS.items() if dims == dimensions]}
    _require(not set(graph["outputs"]) & {n["id"] for n in graph["nodes"] if n["op"] == "variable"},
             "separate_observation_node_required_for_table_mapping")
    return {"inputs": [role(n) for n in graph["nodes"] if n["op"] == "variable"],
            "outputs": [role(nodes[key]) for key in graph["outputs"]], "frozen_model_hash": model.digest}


def map_heldout_table(model, frame, config):
    """No fitting, threshold changes, row dropping or mutation of the source."""
    roles = frozen_table_roles(model)
    fields = table_fields(frame)
    _keys(config, {"bindings", "start_row", "row_count", "si_contract_confirmed"})
    _require(config["si_contract_confirmed"] is True, "si_numeric_contract_confirmation_required")
    nodes = roles["inputs"] + roles["outputs"]
    _keys(config["bindings"], {r["node"] for r in nodes})
    numeric = {c["name"] for c in fields["columns"] if not c["identifier"]}
    columns, mapping = [], []
    for role in nodes:
        binding = config["bindings"][role["node"]]
        _keys(binding, {"column", "unit"})
        _require(type(binding["column"]) is str and binding["column"] in numeric, "numeric_measure_column_required")
        _require(type(binding["unit"]) is str and binding["unit"] in role["units"], "heldout_unit_dimension_mismatch")
        columns.append(binding["column"])
        mapping.append({"node": role["node"], "column": binding["column"], "unit": binding["unit"],
                        "si_factor": UNITS[binding["unit"]][1]})
    _require(len(set(columns)) == len(columns), "distinct_heldout_columns_required")
    start, count = config["start_row"], config["row_count"]
    _require(type(start) is int and start >= 0 and type(count) is int and 1 <= count <= 256
             and start + count <= len(frame), "invalid_heldout_window")
    selected = frame.iloc[start:start+count][columns].copy()
    converted = {}
    for item in mapping:
        array = selected[item["column"]].to_numpy(dtype=float, na_value=np.nan)
        _require(bool(np.all(np.isfinite(array))) and bool(np.all(np.abs(array) <= 1e12)), "invalid_heldout_table_values")
        array = array * item["si_factor"]
        _require(bool(np.all(np.isfinite(array))) and bool(np.all(np.abs(array) <= 1e12)), "invalid_heldout_table_values")
        converted[item["node"]] = array
    cases = [{"id": f"heldout_row_{start+i}",
              "bindings": {r["node"]: float(converted[r["node"]][i]) for r in roles["inputs"]},
              "expected": {r["node"]: float(converted[r["node"]][i]) for r in roles["outputs"]}} for i in range(count)]
    data = HeldoutCases.from_payload({"schema_version": HOLDOUT_VERSION, "cases": cases}, model)
    return data, {"schema_version": "mathmodel.heldout-table-mapping/v1", "frozen_model_hash": model.digest,
        "holdout_hash": data.digest, "source_rows": len(frame), "start_row": start, "selected_rows": count,
        "bindings": mapping, "numeric_convention": "SI_confirmed_by_user", "source_provenance_authenticated": False,
        "independence_verified": False, "parameters_refitted": False, "may_feed_search": False}
