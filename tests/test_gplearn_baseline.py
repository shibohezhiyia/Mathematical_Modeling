import numpy as np
import pytest

from core.gplearn_baseline import GPLearnBaselineError, fit_gplearn_baseline
from core.model_submission_evaluator import reexecute_submitted_model


def _payload():
    values = np.linspace(-2.0, 2.0, 48)
    return {
        "attachments": [{"name": "observations", "format": "records",
                         "rows": [{"x": float(value), "response": float(1.0 + 2.0 * value)}
                                  for value in values]}],
        "query_inputs": [[-1.5], [0.25], [1.75]],
        "gplearn_population_size": 100,
        "gplearn_generations": 3,
        "gplearn_seed": 7,
    }


def test_gplearn_baseline_serializes_an_independently_executable_program():
    pytest.importorskip("gplearn")
    output = fit_gplearn_baseline(_payload(), maximum_program_evaluations=300)
    reproduced = reexecute_submitted_model(output["model"], _payload()["query_inputs"])
    assert reproduced == pytest.approx(output["predictions"], rel=1e-12, abs=1e-12)
    assert output["model"]["source_revision"] == "0390aea8639ce5f6c0b388400e07b58c05acad6a"
    assert output["model"]["program_evaluation_upper_bound"] <= 300


def test_gplearn_baseline_rejects_a_search_budget_overrun_before_fitting():
    payload = _payload()
    payload["gplearn_population_size"] = 1000
    payload["gplearn_generations"] = 20
    with pytest.raises(GPLearnBaselineError, match="gplearn_search_budget_invalid"):
        fit_gplearn_baseline(payload, maximum_program_evaluations=1000)


def test_gplearn_prefix_executor_rejects_trailing_nodes():
    model = {"structure": "gplearn_prefix_program", "input_variables": ["x"],
             "program": [{"feature_index": 0}, {"constant": 1.0}],
             "x_center": [0.0], "x_scale": [1.0], "y_center": 0.0, "y_scale": 1.0}
    with pytest.raises(ValueError, match="gplearn_program_trailing_nodes"):
        reexecute_submitted_model(model, [[1.0]])
