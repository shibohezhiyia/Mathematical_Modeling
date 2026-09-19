"""Evidence-gated causal DAG contract used to constrain, not prove, search."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Mapping, Sequence

import numpy as np


class CausalDAGError(ValueError):
    pass


def build_causal_dag_contract(*, variables: Sequence[Mapping[str, Any]], edges: Sequence[Mapping[str, Any]],
                              evidence_status: str = "not_verified") -> dict[str, Any]:
    if not isinstance(variables, Sequence) or isinstance(variables, (str, bytes)) or not variables:
        raise CausalDAGError("variables_required")
    names, roles = set(), {}
    allowed_roles = {"treatment", "outcome", "confounder", "mediator", "covariate", "state", "unknown"}
    for variable in variables:
        if not isinstance(variable, Mapping) or not isinstance(variable.get("id"), str) or not variable["id"].strip():
            raise CausalDAGError("variable_id_required")
        identifier = variable["id"].strip()
        if identifier in names:
            raise CausalDAGError("duplicate_variable_id")
        role = variable.get("role", "unknown")
        if role not in allowed_roles:
            raise CausalDAGError("unsupported_causal_role")
        names.add(identifier)
        roles[identifier] = role
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)) or len(edges) > 10_000:
        raise CausalDAGError("edges_must_be_bounded_sequence")
    adjacency = defaultdict(list)
    normalized = []
    for edge in edges:
        if not isinstance(edge, Mapping) or not isinstance(edge.get("source"), str) or not isinstance(edge.get("target"), str):
            raise CausalDAGError("causal_edge_endpoints_required")
        source, target = edge["source"].strip(), edge["target"].strip()
        if source not in names or target not in names or source == target:
            raise CausalDAGError("causal_edge_endpoint_unknown_or_self_loop")
        if target in adjacency[source]:
            raise CausalDAGError("duplicate_causal_edge")
        adjacency[source].append(target)
        normalized.append({"source": source, "target": target,
                           "evidence_refs": [str(item) for item in edge.get("evidence_refs", [])][:8]})
    indegree = {name: 0 for name in names}
    for source in adjacency:
        for target in adjacency[source]:
            indegree[target] += 1
    queue = deque(name for name, degree in indegree.items() if degree == 0)
    visited = []
    while queue:
        node = queue.popleft()
        visited.append(node)
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if len(visited) != len(names):
        raise CausalDAGError("causal_graph_cycle")
    if evidence_status not in {"verified", "partial", "not_verified"}:
        raise CausalDAGError("invalid_evidence_status")
    return {"schema_version": "mathmodel.causal-dag/v1", "variables": sorted(names),
            "roles": roles, "edges": normalized, "topological_order": visited,
            "evidence_status": evidence_status,
            "execution_authorized": evidence_status == "verified",
            "policy": "dag_is_a_structural_hypothesis;_not_causal_proof"}


def restrict_interactions_to_dag(contract: Mapping[str, Any], interactions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(contract, Mapping) or not isinstance(interactions, Sequence) or isinstance(interactions, (str, bytes)):
        raise CausalDAGError("contract_and_interactions_required")
    edges = {(item.get("source"), item.get("target")) for item in contract.get("edges", []) if isinstance(item, Mapping)}
    kept, deferred = [], []
    for interaction in interactions:
        if not isinstance(interaction, Mapping) or not isinstance(interaction.get("source"), str) or not isinstance(interaction.get("target"), str):
            raise CausalDAGError("interaction_endpoints_required")
        item = dict(interaction)
        if (item["source"], item["target"]) in edges or (item["target"], item["source"]) in edges:
            kept.append(item)
        else:
            deferred.append({"source": item["source"], "target": item["target"], "reason": "edge_not_in_declared_dag"})
    return {"kept": kept, "deferred": deferred,
            "policy": "deferred_edges_are_not_false;_dag_does_not_create_causal_evidence"}


def discover_linear_causal_dag(
    data: Sequence[Sequence[float]], variable_names: Sequence[str], *,
    edge_threshold: float = 0.15, bootstrap: int = 20, max_edges: int = 64,
    random_state: int = 0,
) -> dict[str, Any]:
    """Discover a bounded order-constrained linear DAG hypothesis.

    Acyclic orders are selected by residual score; coefficients are then
    estimated only from predecessors. Bootstrap resampling reports edge
    stability. This is a structure-learning screen, not an intervention-based
    causal identification method, so the returned contract remains partial.
    """
    matrix = np.asarray(data, dtype=float)
    names = [str(item) for item in variable_names]
    if matrix.ndim != 2 or matrix.shape[0] < 40 or matrix.shape[1] != len(names) or not 2 <= matrix.shape[1] <= 12 or not np.isfinite(matrix).all():
        raise CausalDAGError("causal_discovery_data_shape_invalid")
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise CausalDAGError("causal_discovery_variable_names_invalid")
    if not 0 < float(edge_threshold) <= 2 or type(bootstrap) is not int or not 8 <= bootstrap <= 200 or type(max_edges) is not int or not 1 <= max_edges <= 256:
        raise CausalDAGError("causal_discovery_options_invalid")
    rng = np.random.default_rng(random_state)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scales = np.maximum(centered.std(axis=0, keepdims=True), 1e-8)
    standardized = centered / scales

    def fit_order(sample: np.ndarray) -> tuple[list[int], float, dict[tuple[int, int], float]]:
        p = sample.shape[1]
        remaining = set(range(p)); order: list[int] = []; total = 0.0; coefficients: dict[tuple[int, int], float] = {}
        while remaining:
            best = None
            for candidate in sorted(remaining):
                parents = order
                if parents:
                    design = np.column_stack([np.ones(sample.shape[0]), sample[:, parents]])
                    coef, *_ = np.linalg.lstsq(design, sample[:, candidate], rcond=None)
                    residual = sample[:, candidate] - design @ coef
                    score = float(np.mean(residual ** 2) + 0.01 * len(parents) / p)
                else:
                    coef = np.array([sample[:, candidate].mean()]); score = float(np.mean((sample[:, candidate] - coef[0]) ** 2))
                if best is None or score < best[0]: best = (score, candidate, coef)
            assert best is not None
            total += best[0]; order.append(best[1]); remaining.remove(best[1])
            if len(order) > 1:
                for parent, coef in zip(order[:-1], best[2][1:]): coefficients[(parent, best[1])] = float(coef)
        return order, total, coefficients

    order, score, coefficients = fit_order(standardized)
    edge_pairs = [pair for pair, value in coefficients.items() if abs(value) >= float(edge_threshold)]
    edge_pairs = edge_pairs[:max_edges]
    stability_counts = {pair: 0 for pair in edge_pairs}
    for _ in range(bootstrap):
        sample = standardized[rng.integers(0, standardized.shape[0], size=standardized.shape[0])]
        _, _, boot_coef = fit_order(sample)
        for pair in edge_pairs:
            if abs(boot_coef.get(pair, 0.0)) >= float(edge_threshold): stability_counts[pair] += 1
    edges = [{"source": names[source], "target": names[target], "coefficient": coefficients[(source, target)],
              "stability": stability_counts[(source, target)] / bootstrap,
              "evidence_status": "screened_predictive"}
             for source, target in edge_pairs]
    contract = build_causal_dag_contract(
        variables=[{"id": name, "role": "unknown"} for name in names],
        edges=[{"source": item["source"], "target": item["target"]} for item in edges], evidence_status="partial")
    return {"schema_version": "mathmodel.causal-discovery/v1", "status": "hypothesis_found",
            "order": [names[index] for index in order], "score": score, "edges": edges,
            "bootstrap": bootstrap, "contract": contract,
            "policy": "order_constrained_linear_structure_screen;_not_interventional_causal_identification"}


def discover_temporal_causal_graph(
    data: Sequence[Sequence[float]], variable_names: Sequence[str], *,
    max_lag: int = 3, edge_threshold: float = 0.1,
    min_validation_gain: float = 0.0, bootstrap: int = 20,
    max_edges: int = 64, random_state: int = 0,
) -> dict[str, Any]:
    """Screen lagged predictive edges with a chronological holdout.

    This is deliberately a *temporal interaction* screen, not a causal
    identification algorithm.  Each candidate adds one source lag to a
    target's autoregressive baseline, then is retained only when it improves a
    future block.  Bootstrap sign stability is evidence for prioritising a
    follow-up model, never an intervention or causal proof.
    """
    matrix = np.asarray(data, dtype=float)
    names = [str(item) for item in variable_names]
    if (matrix.ndim != 2 or matrix.shape[0] < 80 or matrix.shape[1] != len(names)
            or not 2 <= matrix.shape[1] <= 16 or not np.isfinite(matrix).all()):
        raise CausalDAGError("temporal_data_shape_invalid")
    if len(set(names)) != len(names) or any(not name.strip() for name in names):
        raise CausalDAGError("temporal_variable_names_invalid")
    if (type(max_lag) is not int or not 1 <= max_lag <= 12
            or type(bootstrap) is not int or not 8 <= bootstrap <= 200
            or type(max_edges) is not int or not 1 <= max_edges <= 256
            or not np.isfinite(float(edge_threshold)) or float(edge_threshold) < 0
            or not np.isfinite(float(min_validation_gain))):
        raise CausalDAGError("temporal_discovery_options_invalid")
    if matrix.shape[0] <= max_lag + 30:
        raise CausalDAGError("temporal_history_too_short")

    centered = matrix - matrix.mean(axis=0, keepdims=True)
    scales = np.maximum(centered.std(axis=0, keepdims=True), 1e-8)
    standardized = centered / scales
    times = np.arange(max_lag, standardized.shape[0])
    # Keep the holdout chronological and large enough to estimate an RMSE.
    split = max(20, int(round(times.size * 0.7)))
    split = min(times.size - 10, split)
    train_times, valid_times = times[:split], times[split:]

    def design(at: np.ndarray, source: int | None = None, lag: int | None = None) -> np.ndarray:
        columns = [np.ones(len(at), dtype=float)]
        for own_lag in range(1, max_lag + 1):
            columns.append(standardized[at - own_lag, target])
        if source is not None and lag is not None:
            columns.append(standardized[at - lag, source])
        return np.column_stack(columns)

    rng = np.random.default_rng(random_state)
    edges: list[dict[str, Any]] = []
    for target in range(standardized.shape[1]):
        baseline_train = design(train_times)
        baseline_valid = design(valid_times)
        base_coef, *_ = np.linalg.lstsq(baseline_train, standardized[train_times, target], rcond=None)
        base_prediction = baseline_valid @ base_coef
        base_rmse = float(np.sqrt(np.mean((base_prediction - standardized[valid_times, target]) ** 2)))
        for source in range(standardized.shape[1]):
            if source == target:
                continue
            for lag in range(1, max_lag + 1):
                augmented_train = design(train_times, source, lag)
                augmented_valid = design(valid_times, source, lag)
                coef, *_ = np.linalg.lstsq(augmented_train, standardized[train_times, target], rcond=None)
                prediction = augmented_valid @ coef
                augmented_rmse = float(np.sqrt(np.mean((prediction - standardized[valid_times, target]) ** 2)))
                gain = base_rmse - augmented_rmse
                coefficient = float(coef[-1])
                if abs(coefficient) < float(edge_threshold) or gain < float(min_validation_gain):
                    continue
                signs = 0
                for _ in range(bootstrap):
                    # Resample contiguous-ish time rows by circular block starts;
                    # this preserves more local dependence than iid row shuffling.
                    starts = rng.integers(0, max(1, train_times.size - 4), size=max(2, train_times.size // 4))
                    sampled = np.concatenate([train_times[start:start + 4] for start in starts])[:train_times.size]
                    boot_design = design(sampled, source, lag)
                    boot_coef, *_ = np.linalg.lstsq(boot_design, standardized[sampled, target], rcond=None)
                    if np.sign(float(boot_coef[-1])) == np.sign(coefficient):
                        signs += 1
                edges.append({
                    "source": names[source], "target": names[target], "lag": int(lag),
                    "coefficient": coefficient, "validation_rmse_gain": float(gain),
                    "stability": float(signs / bootstrap),
                    "evidence_status": "screened_temporal_predictive",
                })
    edges.sort(key=lambda item: (-float(item["validation_rmse_gain"]), -float(item["stability"]), item["source"], item["target"], item["lag"]))
    edges = edges[:max_edges]
    return {
        "schema_version": "mathmodel.temporal-causal-screen/v1",
        "status": "hypothesis_found" if edges else "no_edge_found",
        "variables": names, "max_lag": int(max_lag),
        "train_rows": int(len(train_times)), "validation_rows": int(len(valid_times)),
        "edges": edges, "bootstrap": int(bootstrap),
        "graph_type": "directed_lagged_interaction_graph",
        "policy": "chronological_holdout_and_block_bootstrap_screen;_not_interventional_causal_identification",
    }


__all__ = ["CausalDAGError", "build_causal_dag_contract", "restrict_interactions_to_dag",
           "discover_linear_causal_dag", "discover_temporal_causal_graph"]
