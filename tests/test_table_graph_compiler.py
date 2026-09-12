from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from core.table_graph_compiler import compile_table_graph, table_fields
from core.graph_experiments import SearchExperiment, restore_problem
from core.graph_search import GraphSearchSession
from core.model_hypotheses import HypothesisIR, HypothesisValidationError
from test_graph_confirmation import local_runner


def frame():
    x = np.arange(1., 31.)
    return pd.DataFrame({"时间": x, "距离": 3*x + 2, "编号": np.arange(30), "类别": ["A"]*30})


def config():
    return {"problem": "研究距离与时间的关系，检验是否存在稳定规律。", "inputs": ["时间"], "target": "距离",
            "units": {"时间": "s", "距离": "m"}, "start_row": 0, "row_count": 30,
            "absolute_tolerance": 1e-6, "relative_tolerance": 1e-4}


def test_table_compiles_and_executes_without_handwritten_graph(monkeypatch):
    local_runner(monkeypatch)
    source = frame()
    before = source.copy(deep=True)
    compiled = compile_table_graph(source, config())
    bundle = compiled["bundle"]
    contract = restore_problem(bundle["problem"])
    exp = SearchExperiment.from_payload(bundle["experiment"], contract)
    result = GraphSearchSession(contract, exp).run(bundle["candidates"])
    assert result["pareto_candidates"] and result["reports"][0]["search_rmse"] < 1e-5
    pd.testing.assert_frame_equal(source, before)
    graph = HypothesisIR.from_payload(bundle["candidates"][0], contract)
    nodes = {n["id"]: n for n in graph.payload()["nodes"]}
    assert nodes["a0"]["type"]["dimensions"] == {"L": 1, "T": -1}
    assert compiled["binding"]["training_rows"] == 21
    assert compiled["binding"]["search_rows"] == 9


def test_unit_conversion_and_tolerance_are_in_si():
    source, settings = frame(), config()
    source["时间"] /= 60
    source["距离"] *= 100
    settings["units"] = {"时间": "min", "距离": "cm"}
    result = compile_table_graph(source, settings)
    first = result["bundle"]["experiment"]["training_cases"][0]
    assert first["bindings"]["x0"] == pytest.approx(1)
    assert first["expected"]["y"] == pytest.approx(5)
    assert result["binding"]["output"]["si_factor"] == .01
    assert '"unit": "cm"' in result["bundle"]["candidates"][0]["assumptions"][0]["text"]


def test_search_labels_do_not_change_parameter_search_boxes():
    original = compile_table_graph(frame(), config())
    source = frame()
    source.loc[21:, "距离"] *= 100
    changed = compile_table_graph(source, config())
    left, right = original["bundle"]["experiment"], changed["bundle"]["experiment"]
    assert left["parameter_bounds"] == right["parameter_bounds"]
    assert left["training_cases"] == right["training_cases"]
    assert left["search_cases"] != right["search_cases"]


def test_only_selected_window_is_converted():
    source = frame()
    source.loc[29, "距离"] = np.nan
    settings = config()
    settings["row_count"] = 20
    result = compile_table_graph(source, settings)
    assert result["binding"]["full_table_analyzed"] is False
    assert len(result["bundle"]["experiment"]["training_cases"]) == 14


@pytest.mark.parametrize("mutate,error", [
    (lambda s: s["units"].update(时间=""), "explicit_supported_units_required"),
    (lambda s: s.update(target="时间"), "distinct_columns_required"),
    (lambda s: s.update(inputs=["编号"], units={"编号": "1", "距离": "m"}), "identifier_is_not_a_measure"),
    (lambda s: s.update(row_count=361), "invalid_development_window"),
    (lambda s: s.update(row_count=31), "invalid_development_window"),
    (lambda s: s.update(start_row=True), "invalid_development_window"),
    (lambda s: s.update(heldout=[]), "unexpected_fields"),
    (lambda s: s.update(relative_tolerance=-1), "invalid_check_tolerance"),
])
def test_ambiguous_or_invalid_bindings_are_rejected(mutate, error):
    settings = config()
    mutate(settings)
    with pytest.raises(HypothesisValidationError, match=error):
        compile_table_graph(frame(), settings)


def test_duplicate_inputs_and_missing_values_are_not_silently_removed():
    source = frame()
    source.loc[25, "时间"] = source.loc[0, "时间"]
    with pytest.raises(HypothesisValidationError, match="repeated_input_requires_grouped_design"):
        compile_table_graph(source, config())
    source = frame()
    source.loc[1, "距离"] = np.nan
    with pytest.raises(HypothesisValidationError, match="missing_or_nonfinite_table_values"):
        compile_table_graph(source, config())


def test_metadata_does_not_suggest_identifier_as_measure():
    fields = table_fields(frame())
    assert fields["rows"] == 30
    assert next(c for c in fields["columns"] if c["name"] == "编号")["identifier"]
    assert "类别" not in [c["name"] for c in fields["columns"]]


def test_multiple_variables_form_one_typed_composition(monkeypatch):
    local_runner(monkeypatch)
    source, settings = frame(), config()
    source["质量"] = np.sin(source["时间"]) + 2
    source["距离"] += 5*source["质量"]
    settings["inputs"].append("质量")
    settings["units"]["质量"] = "kg"
    bundle = compile_table_graph(source, settings)["bundle"]
    contract = restore_problem(bundle["problem"])
    result = GraphSearchSession(contract, SearchExperiment.from_payload(bundle["experiment"], contract)).run(bundle["candidates"])
    assert result["pareto_candidates"]
