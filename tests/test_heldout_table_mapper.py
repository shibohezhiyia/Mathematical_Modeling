import numpy as np
import pandas as pd
import pytest

from test_table_graph_compiler import frame, config
from test_graph_confirmation import local_runner
from core.table_graph_compiler import compile_table_graph
from core.graph_experiments import SearchExperiment, restore_problem
from core.graph_search import GraphSearchSession
from core.heldout_table_mapper import frozen_table_roles, map_heldout_table
from core.graph_confirmation import evaluate_frozen_request
from core.model_hypotheses import HypothesisValidationError


@pytest.fixture
def model(monkeypatch):
    local_runner(monkeypatch)
    bundle = compile_table_graph(frame(), config())["bundle"]
    contract = restore_problem(bundle["problem"])
    session = GraphSearchSession(contract, SearchExperiment.from_payload(bundle["experiment"], contract))
    result = session.run(bundle["candidates"])
    return session.freeze(result["pareto_candidates"][0])


def validation_table():
    x = np.array([1.137, 2.271, 13.123, 21.341])
    return pd.DataFrame({"时长分钟": x/60, "长度厘米": (3*x+2)*100, "编号": range(4)})


def mapping():
    return {"bindings": {"x0": {"column": "时长分钟", "unit": "min"}, "y": {"column": "长度厘米", "unit": "cm"}},
            "start_row": 0, "row_count": 4, "si_contract_confirmed": True}


def test_map_units_without_refitting_or_mutating(model):
    source = validation_table()
    before = source.copy(deep=True)
    frozen_before = model.public()
    data, evidence = map_heldout_table(model, source, mapping())
    result = evaluate_frozen_request({"frozen_model": model.public(), "holdout": data.public()}, max_evaluations=1000)
    assert result["status"] == "passed_finite_heldout_checks"
    assert data.public()["cases"][0]["bindings"]["x0"] == pytest.approx(1.137)
    assert evidence["frozen_model_hash"] == model.digest and not evidence["independence_verified"]
    assert evidence["bindings"][0]["si_factor"] == 60
    assert model.public() == frozen_before
    pd.testing.assert_frame_equal(source, before)


def test_unit_choices_follow_frozen_dimensions(model):
    roles = frozen_table_roles(model)
    assert "min" in roles["inputs"][0]["units"] and "kg" not in roles["inputs"][0]["units"]
    assert "cm" in roles["outputs"][0]["units"] and "s" not in roles["outputs"][0]["units"]


@pytest.mark.parametrize("mutate,error", [
    (lambda c: c.update(si_contract_confirmed=False), "si_numeric_contract_confirmation_required"),
    (lambda c: c["bindings"]["x0"].update(unit="kg"), "heldout_unit_dimension_mismatch"),
    (lambda c: c["bindings"]["y"].update(column="编号"), "numeric_measure_column_required"),
    (lambda c: c["bindings"]["y"].update(column="时长分钟"), "distinct_heldout_columns_required"),
    (lambda c: c.update(row_count=257), "invalid_heldout_window"),
    (lambda c: c.update(start_row=-1), "invalid_heldout_window"),
    (lambda c: c.update(absolute_tolerance=100), "unexpected_fields"),
    (lambda c: c["bindings"].update(extra={}), "unexpected_fields"),
])
def test_invalid_bindings_are_rejected(model, mutate, error):
    settings = mapping()
    mutate(settings)
    with pytest.raises(HypothesisValidationError, match=error):
        map_heldout_table(model, validation_table(), settings)


def test_unit_conversion_cannot_hide_development_overlap(model):
    source = validation_table()
    source.loc[0, "时长分钟"] = 1/60
    with pytest.raises(HypothesisValidationError, match="holdout_development_overlap"):
        map_heldout_table(model, source, mapping())


def test_missing_duplicate_and_outside_domain_are_not_silently_removed(model):
    for column, value, code in [("长度厘米", np.nan, "invalid_heldout_table_values"),
                                ("时长分钟", 100, "point_outside_domain"),
                                ("时长分钟", validation_table().loc[1, "时长分钟"], "duplicate_holdout_point")]:
        source = validation_table()
        source.loc[0, column] = value
        with pytest.raises(HypothesisValidationError, match=code):
            map_heldout_table(model, source, mapping())


def test_only_selected_rows_are_converted(model):
    source = validation_table()
    source.loc[3, "长度厘米"] = np.nan
    settings = mapping()
    settings["row_count"] = 3
    data, evidence = map_heldout_table(model, source, settings)
    assert len(data.public()["cases"]) == 3 and evidence["source_rows"] == 4
