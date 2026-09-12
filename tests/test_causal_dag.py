import pytest

import numpy as np

from core.causal_dag import CausalDAGError, build_causal_dag_contract, discover_linear_causal_dag, restrict_interactions_to_dag


def test_causal_dag_is_acyclic_and_restricts_search_without_claiming_causality():
    contract = build_causal_dag_contract(
        variables=[{"id": "rain", "role": "treatment"}, {"id": "yield", "role": "outcome"}, {"id": "soil", "role": "confounder"}],
        edges=[{"source": "rain", "target": "soil"}, {"source": "soil", "target": "yield"}],
        evidence_status="partial",
    )
    result = restrict_interactions_to_dag(contract, [
        {"source": "soil", "target": "yield"}, {"source": "rain", "target": "yield"}
    ])
    assert len(result["kept"]) == 1
    assert contract["execution_authorized"] is False


def test_causal_cycle_is_rejected():
    with pytest.raises(CausalDAGError, match="causal_graph_cycle"):
        build_causal_dag_contract(
            variables=[{"id": "a"}, {"id": "b"}],
            edges=[{"source": "a", "target": "b"}, {"source": "b", "target": "a"}],
        )


def test_linear_causal_dag_screen_returns_bootstrap_stability_without_causal_claim():
    rng = np.random.default_rng(2)
    x = rng.normal(size=120)
    y = 1.8 * x + rng.normal(scale=0.1, size=120)
    z = -0.7 * y + rng.normal(scale=0.1, size=120)
    result = discover_linear_causal_dag(np.column_stack([x, y, z]), ["x", "y", "z"], bootstrap=8)
    assert result["status"] == "hypothesis_found"
    assert result["contract"]["evidence_status"] == "partial"
    assert result["policy"].endswith("identification")
