"""Optional, bounded graph-message-passing interaction screen.

This is a small torch implementation rather than a dependency on
``torch-geometric``.  Each numeric column is a node and each row is one graph;
learned sigmoid edge gates control one message-passing layer before a pooled
regression head.  The output is predictive evidence only, never a causal DAG
or a proof that an edge is a real mechanism.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


SCHEMA_VERSION = "mathmodel.gnn-interaction-screen/v1"


class GNNInteractionError(ValueError):
    pass


def discover_gnn_interactions(
    frame: pd.DataFrame,
    target: str,
    columns: Sequence[str] | None = None,
    *,
    max_variables: int = 24,
    max_rows: int = 2_000,
    epochs: int = 120,
    hidden_dim: int = 16,
    restarts: int = 2,
    message_layers: int = 1,
    group_column: str | None = None,
    time_column: str | None = None,
    dynamic_windows: int = 0,
    edge_threshold: float = 0.55,
    validation_fraction: float = 0.2,
    random_state: int = 0,
) -> dict[str, Any]:
    """Fit a tiny gated message-passing model with strict resource bounds."""
    if not isinstance(frame, pd.DataFrame):
        raise GNNInteractionError("frame_must_be_dataframe")
    if not isinstance(target, str) or target not in frame.columns:
        raise GNNInteractionError("target_column_required")
    if type(max_variables) is not int or not 2 <= max_variables <= 32:
        raise GNNInteractionError("invalid_max_variables")
    if type(max_rows) is not int or not 60 <= max_rows <= 10_000:
        raise GNNInteractionError("invalid_max_rows")
    if type(epochs) is not int or not 10 <= epochs <= 500:
        raise GNNInteractionError("invalid_epochs")
    if type(hidden_dim) is not int or not 4 <= hidden_dim <= 64:
        raise GNNInteractionError("invalid_hidden_dim")
    if type(restarts) is not int or not 1 <= restarts <= 3:
        raise GNNInteractionError("invalid_restarts")
    if type(message_layers) is not int or not 1 <= message_layers <= 3:
        raise GNNInteractionError("invalid_message_layers")
    if type(dynamic_windows) is not int or not 0 <= dynamic_windows <= 5:
        raise GNNInteractionError("invalid_dynamic_windows")
    if not 0.1 <= float(validation_fraction) <= 0.4:
        raise GNNInteractionError("invalid_validation_fraction")
    if not 0 < float(edge_threshold) < 1:
        raise GNNInteractionError("invalid_edge_threshold")
    if type(random_state) is not int or not 0 <= random_state < 2**32:
        raise GNNInteractionError("invalid_random_state")
    if group_column is not None and (not isinstance(group_column, str) or group_column not in frame.columns):
        raise GNNInteractionError("invalid_group_column")
    if time_column is not None and (not isinstance(time_column, str) or time_column not in frame.columns):
        raise GNNInteractionError("invalid_time_column")
    if group_column is not None and time_column is not None and group_column == time_column:
        raise GNNInteractionError("group_and_time_columns_must_differ")
    try:
        import torch
        from torch import nn
    except ImportError:
        return {"schema_version": SCHEMA_VERSION, "status": "unavailable",
                "reason": "torch_not_installed", "edges": []}

    requested = [str(item) for item in columns] if columns is not None else [
        str(item) for item in frame.columns
        if item != target and item not in {group_column, time_column} and pd.api.types.is_numeric_dtype(frame[item])
    ]
    numeric = [item for item in requested if item in frame.columns and item != target and item not in {group_column, time_column}
               and pd.api.types.is_numeric_dtype(frame[item])]
    numeric = sorted(set(numeric), key=lambda name: (-float(pd.to_numeric(frame[name], errors="coerce").std()), name))[:max_variables]
    if len(numeric) < 2:
        return {"schema_version": SCHEMA_VERSION, "status": "not_assessed",
                "reason": "fewer_than_two_numeric_variables", "variables": numeric, "edges": []}
    metadata_columns = [item for item in (group_column, time_column) if item is not None]
    work = frame[numeric + [target] + metadata_columns].copy()
    work[numeric + [target]] = work[numeric + [target]].apply(pd.to_numeric, errors="coerce")
    work = work.replace([np.inf, -np.inf], np.nan).dropna()
    if len(work) < 60:
        return {"schema_version": SCHEMA_VERSION, "status": "not_assessed",
                "reason": "fewer_than_sixty_complete_rows", "variables": numeric,
                "rows": int(len(work)), "edges": []}
    rng = np.random.default_rng(random_state)
    if len(work) > max_rows:
        work = work.iloc[np.sort(rng.choice(len(work), size=max_rows, replace=False))]
    values = work[numeric].to_numpy(dtype=np.float32)
    labels = work[target].to_numpy(dtype=np.float32)
    split_policy = "row_random"
    if group_column is not None:
        groups = work[group_column].astype(str).to_numpy()
        unique_groups = np.unique(groups)
        if unique_groups.size < 3:
            return {"schema_version": SCHEMA_VERSION, "status": "not_assessed", "reason": "fewer_than_three_groups", "variables": numeric, "edges": [], "split_policy": "group_required_but_insufficient"}
        group_order = rng.permutation(unique_groups)
        valid_group_count = max(1, min(unique_groups.size - 1, int(round(unique_groups.size * float(validation_fraction)))))
        valid_groups = set(group_order[-valid_group_count:])
        valid_idx = np.flatnonzero(np.isin(groups, list(valid_groups)))
        train_idx = np.flatnonzero(~np.isin(groups, list(valid_groups)))
        split_policy = "group_holdout"
    elif time_column is not None:
        time_values = pd.to_datetime(work[time_column], errors="coerce")
        if time_values.isna().any():
            time_values = pd.to_numeric(work[time_column], errors="coerce")
        if pd.isna(time_values).any():
            return {"schema_version": SCHEMA_VERSION, "status": "not_assessed", "reason": "time_column_not_orderable", "variables": numeric, "edges": [], "split_policy": "time_required_but_invalid"}
        order = np.argsort(np.asarray(time_values.astype("int64")))
        split = max(1, min(len(values) - 1, int(round(len(values) * (1 - float(validation_fraction))))))
        train_idx, valid_idx = order[:split], order[split:]
        split_policy = "time_tail_holdout"
    else:
        order = rng.permutation(len(values))
        split = max(1, min(len(values) - 1, int(round(len(values) * (1 - float(validation_fraction))))))
        train_idx, valid_idx = order[:split], order[split:]
    x_train, x_valid = values[train_idx], values[valid_idx]
    y_train, y_valid = labels[train_idx], labels[valid_idx]
    means, scales = x_train.mean(0), np.maximum(x_train.std(0), 1e-6)
    y_mean, y_scale = float(y_train.mean()), max(float(y_train.std()), 1e-6)
    x_train = (x_train - means) / scales
    x_valid = (x_valid - means) / scales
    y_train = (y_train - y_mean) / y_scale
    y_valid_original = y_valid.copy()
    y_valid = (y_valid - y_mean) / y_scale

    class GatedMessagePassing(nn.Module):
        def __init__(self, node_count: int, width: int, layers: int):
            super().__init__()
            self.node = nn.Linear(1, width)
            self.updates = nn.ModuleList(nn.Linear(width, width) for _ in range(layers))
            self.head = nn.Sequential(nn.Linear(width, width), nn.Tanh(), nn.Linear(width, 1))
            self.edge_logits = nn.Parameter(torch.zeros(node_count, node_count))

        def forward(self, inputs):
            node_state = torch.tanh(self.node(inputs.unsqueeze(-1)))
            gates = torch.sigmoid(self.edge_logits)
            mask = 1.0 - torch.eye(gates.shape[0], device=gates.device)
            gates = gates * mask
            updated = node_state
            for update in self.updates:
                messages = torch.einsum("ij,njh->nih", gates, updated)
                updated = torch.tanh(update(updated + messages))
            return self.head(updated.mean(dim=1)).squeeze(-1), gates

    device = torch.device("cpu")
    train_x = torch.tensor(x_train, dtype=torch.float32, device=device)
    valid_x = torch.tensor(x_valid, dtype=torch.float32, device=device)
    train_y = torch.tensor(y_train, dtype=torch.float32, device=device)
    valid_y = torch.tensor(y_valid, dtype=torch.float32, device=device)
    gate_runs, validation_rmse = [], []
    for restart in range(restarts):
        torch.manual_seed(int(random_state + restart))
        model = GatedMessagePassing(len(numeric), hidden_dim, message_layers).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=1e-4)
        best_loss, best_state, stale = float("inf"), None, 0
        for _ in range(epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(train_x)
            loss = torch.mean((prediction - train_y) ** 2) + 1e-4 * torch.mean(torch.sigmoid(model.edge_logits))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                validation_loss = float(torch.mean((model(valid_x)[0] - valid_y) ** 2).cpu())
            if validation_loss + 1e-7 < best_loss:
                best_loss = validation_loss
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= 20:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prediction, gates = model(valid_x)
        prediction_original = prediction.cpu().numpy() * y_scale + y_mean
        validation_rmse.append(float(np.sqrt(np.mean((prediction_original - y_valid_original) ** 2))))
        gate_runs.append(gates.cpu().numpy())

    gate_stack = np.stack(gate_runs, axis=0)
    mean_gate = gate_stack.mean(axis=0)
    active = gate_stack >= float(edge_threshold)
    edges = []
    for source in range(len(numeric)):
        for destination in range(len(numeric)):
            if source == destination:
                continue
            stability = float(active[:, source, destination].mean())
            strength = float(mean_gate[source, destination])
            if strength >= float(edge_threshold) and stability >= 0.5:
                edges.append({"source": numeric[source], "target": numeric[destination],
                              "gate_strength": round(strength, 6), "stability": round(stability, 6),
                              "interpretation": "predictive message-passing interaction; not a causal edge"})
    edges.sort(key=lambda item: (-item["stability"], -item["gate_strength"], item["source"], item["target"]))
    result = {
        "schema_version": SCHEMA_VERSION, "status": "executed", "proof_status": "finite_validation_only",
        "backend": "torch_gated_message_passing", "variables": numeric, "rows": int(len(values)),
        "train_rows": int(len(train_idx)), "validation_rows": int(len(valid_idx)),
        "epochs_budget": int(epochs), "restarts": int(restarts), "message_layers": int(message_layers),
        "validation_rmse": float(np.mean(validation_rmse)), "validation_rmse_by_restart": validation_rmse,
        "edges": edges, "policy": "predictive_interaction_screen_is_not_gnn_causal_discovery_or_proof",
        "graph_scope": "static_row_graph; temporal/entity edges require an explicit grouping contract",
        "split_policy": split_policy, "group_column": group_column, "time_column": time_column,
    }
    if dynamic_windows > 1:
        if time_column is None:
            result["dynamic_status"] = "not_assessed"
            result["dynamic_reason"] = "time_column_required"
        else:
            ordered_time = pd.to_datetime(frame[time_column], errors="coerce")
            if ordered_time.isna().any():
                ordered_time = pd.to_numeric(frame[time_column], errors="coerce")
            if pd.isna(ordered_time).any():
                result["dynamic_status"] = "not_assessed"
                result["dynamic_reason"] = "time_column_not_orderable"
            else:
                ordered = frame.assign(__mm_time_order=np.asarray(ordered_time.astype("int64"))).sort_values("__mm_time_order")
                chunks = [chunk.drop(columns=["__mm_time_order"]) for chunk in np.array_split(ordered, dynamic_windows) if len(chunk) >= 60]
                if len(chunks) < 2:
                    result["dynamic_status"] = "not_assessed"
                    result["dynamic_reason"] = "dynamic_window_too_small"
                else:
                    window_results = []
                    edge_counts: dict[tuple[str, str], int] = {}
                    for window in chunks[:dynamic_windows]:
                        window_result = discover_gnn_interactions(
                            window, target, columns=numeric, max_variables=max_variables,
                            max_rows=max_rows, epochs=max(10, min(epochs, 80)), hidden_dim=hidden_dim,
                            restarts=restarts, message_layers=message_layers, edge_threshold=edge_threshold,
                            validation_fraction=validation_fraction, random_state=random_state,
                            group_column=None, time_column=None, dynamic_windows=0,
                        )
                        window_results.append({"status": window_result.get("status"), "edges": window_result.get("edges", []),
                                              "validation_rmse": window_result.get("validation_rmse"), "rows": window_result.get("rows")})
                        for edge in window_result.get("edges", []):
                            key = (str(edge.get("source")), str(edge.get("target")))
                            edge_counts[key] = edge_counts.get(key, 0) + 1
                    result["dynamic_status"] = "assessed"
                    result["dynamic_window_count"] = len(window_results)
                    result["dynamic_windows"] = window_results
                    result["dynamic_edge_persistence"] = [
                        {"source": source, "target": target_name, "persistence": count / len(window_results)}
                        for (source, target_name), count in sorted(edge_counts.items(), key=lambda item: (-item[1], item[0]))
                    ]
                    result["graph_scope"] = "rolling_time_window_row_graph; edges are predictive and not causal"
    return result


__all__ = ["SCHEMA_VERSION", "GNNInteractionError", "discover_gnn_interactions"]
