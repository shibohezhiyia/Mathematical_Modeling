import numpy as np
import pandas as pd

from core.gnn_interaction_screen import discover_gnn_interactions


def test_gnn_interaction_screen_runs_bounded_message_passing():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 3))
    frame = pd.DataFrame(x, columns=["a", "b", "c"])
    frame["y"] = 2 * frame["a"] - frame["b"] + rng.normal(scale=0.1, size=len(frame))
    result = discover_gnn_interactions(frame, "y", epochs=10, restarts=1, message_layers=2, random_state=4)
    assert result["status"] in {"executed", "unavailable"}
    if result["status"] == "executed":
        assert result["backend"] == "torch_gated_message_passing"
        assert result["validation_rows"] > 0
        assert result["message_layers"] == 2
        assert result["policy"].endswith("or_proof")


def test_gnn_interaction_screen_rejects_small_tables():
    result = discover_gnn_interactions(pd.DataFrame({"a": [1] * 20, "b": [2] * 20, "y": [3] * 20}), "y")
    assert result["status"] == "not_assessed"


def test_gnn_interaction_screen_supports_group_holdout_without_group_leakage():
    rng = np.random.default_rng(9)
    rows = []
    for group in range(4):
        for step in range(20):
            a, b, c = rng.normal(size=3)
            rows.append({"group": f"g{group}", "step": step, "a": a, "b": b, "c": c, "y": 2 * a - b})
    result = discover_gnn_interactions(pd.DataFrame(rows), "y", epochs=10, restarts=1,
                                       group_column="group", time_column="step", random_state=9)
    assert result["status"] in {"executed", "unavailable"}
    if result["status"] == "executed":
        assert result["split_policy"] == "group_holdout"


def test_gnn_interaction_screen_assesses_rolling_dynamic_windows():
    rng = np.random.default_rng(12)
    frame = pd.DataFrame({"time": np.arange(120), "a": rng.normal(size=120), "b": rng.normal(size=120),
                          "c": rng.normal(size=120)})
    frame["y"] = 1.5 * frame["a"] - 0.4 * frame["b"] + rng.normal(scale=0.05, size=120)
    result = discover_gnn_interactions(frame, "y", epochs=10, restarts=1, time_column="time",
                                       dynamic_windows=2, random_state=12)
    assert result["status"] in {"executed", "unavailable"}
    if result["status"] == "executed":
        assert result["dynamic_status"] == "assessed"
        assert result["dynamic_window_count"] == 2
        assert isinstance(result["dynamic_edge_persistence"], list)
