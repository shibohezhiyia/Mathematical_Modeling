import numpy as np
import pandas as pd
import pytest

from core.gnn_model_cegis import compile_gnn_candidate, evaluate_gnn_candidate


def _rows(n=60):
    rng = np.random.default_rng(4)
    a = rng.normal(size=n)
    b = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "y": 2.0 * a - b}).to_dict("records")


def test_gnn_candidate_contract_rejects_unknown_source_fields():
    with pytest.raises(ValueError, match="unknown_fields"):
        compile_gnn_candidate({"source": "import os"})


def test_gnn_adapter_returns_common_metrics_contract_when_torch_available():
    pytest.importorskip("torch")
    compiled = compile_gnn_candidate({"id": "screen", "target": "y", "epochs": 10, "restarts": 1})
    result = evaluate_gnn_candidate(compiled, [{"id": "holdout", "rows": _rows()}])
    assert result["status"] in {"pass", "fail", "not_assessed"}
    if result["status"] in {"pass", "fail"}:
        assert len(result["predictions"]) == 1
        assert set(result["metrics"]) == {"validation_loss", "complexity", "constraint_violation", "instability"}
