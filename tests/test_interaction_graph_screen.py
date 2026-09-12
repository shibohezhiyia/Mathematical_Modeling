import numpy as np
import pandas as pd

from core.interaction_graph_screen import discover_interaction_graph


def test_interaction_graph_screen_finds_stable_conditional_association():
    rng = np.random.default_rng(3)
    x = rng.normal(size=240)
    y = 0.8 * x + rng.normal(scale=0.2, size=240)
    z = rng.normal(size=240)
    result = discover_interaction_graph(pd.DataFrame({"x": x, "y": y, "z": z}), bootstrap=8, random_state=3)
    assert result["status"] == "tested_not_falsified"
    assert any({edge["source"], edge["target"]} == {"x", "y"} for edge in result["edges"])
    assert result["policy"].endswith("not_causal_discovery_or_gnn")


def test_interaction_graph_screen_does_not_invent_graph_for_insufficient_data():
    result = discover_interaction_graph(pd.DataFrame({"x": [1, 2], "y": [2, 3], "z": [3, 4]}))
    assert result["status"] == "not_assessed"
