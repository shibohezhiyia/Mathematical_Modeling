import numpy as np

from core.external_method_cegis import run_external_method_cegis


def test_typed_symbolic_runtime_enters_shared_cegis_loop():
    x = np.linspace(0.0, 1.0, 32).tolist()
    y = (2.0 * np.asarray(x) + 1.0).tolist()
    candidate = {
        "id": "linear",
        "method": "llm_sr",
        "payload": {
            "target": y, "features": [[value] for value in x],
            "expression": {"op": "add", "left": {"op": "multiply", "left": {"op": "param", "name": "a"}, "right": {"op": "var", "name": "x"}}, "right": {"op": "param", "name": "b"}},
            "parameter_bounds": {"a": [-5.0, 5.0], "b": [-5.0, 5.0]},
            "max_nfev": 200,
        },
    }
    result = run_external_method_cegis("llm_sr", [candidate], [{"id": "holdout", "max_metric": 0.1}])
    assert result["adapter_family"] == "llm_sr"
    assert result["status"] in {"accepted_candidates", "candidate_set_inadequate", "budget_exhausted"}
    assert result["records"]
