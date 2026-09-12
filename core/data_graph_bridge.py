"""Build a bounded, typed graph-search experiment from tabular observations.

This adapter is deliberately small: it does not infer causality, units, or a
complete mechanism from column names.  It creates a scalar *candidate search*
problem with an explicit dimensionless assumption and keeps a disjoint search
partition.  The graph-search runtime remains responsible for execution,
counterexamples, and model selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .graph_experiments import SearchExperiment
from .graph_search_artifacts import run_search_bundle
from .model_hypotheses import HypothesisIR, MathType, ProblemContract


BRIDGE_VERSION = "mathmodel.data-graph-bridge/v1"
_ALLOWED_OPERATORS = [
    "add", "subtract", "multiply", "divide", "minimum", "maximum",
    "negate", "abs", "sqrt", "exp", "log", "sin", "cos", "observation",
]


def _finite_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.where(np.isfinite(values))


def _dimension_payload(value: Mapping[str, Any] | None) -> dict[str, Any]:
    dimensions = {} if value is None else dict(value)
    return MathType.parse({"dtype": "real", "shape": [], "dimensions": dimensions}).public()["dimensions"]


def _template(contract: ProblemContract, features: list[str], target: str,
              feature_dimensions: Mapping[str, Any] | None,
              target_dimensions: Mapping[str, Any] | None) -> HypothesisIR:
    feature_dimensions = feature_dimensions or {}
    scalar_by_node = {
        ("x" if len(features) == 1 else f"x{index}"): {
            "dtype": "real", "shape": [],
            "dimensions": _dimension_payload(feature_dimensions.get(name)),
        }
        for index, name in enumerate(features)
    }
    output_type = {"dtype": "real", "shape": [], "dimensions": _dimension_payload(target_dimensions)}
    node_ids = ["x"] if len(features) == 1 else [f"x{index}" for index in range(len(features))]
    nodes = [
        {"id": node_id, "op": "variable", "inputs": [], "type": scalar_by_node[node_id],
         "attributes": {"name": name, "role": "observed"},
         "assumption_ids": ["data_assumption"], "context_fact_ids": ["statement"]}
        for node_id, name in zip(node_ids, features)
    ]
    payload = {
        "id": "data_bridge_initial",
        "contract_hash": contract.digest,
        "nodes": nodes + [
            {"id": "y", "op": "unknown_mechanism", "inputs": node_ids, "type": output_type,
             "attributes": {"allowed_operators": list(_ALLOWED_OPERATORS),
                            "properties": ["fit_training_cases", "pass_disjoint_search_cases"]},
             "assumption_ids": ["data_assumption"], "context_fact_ids": ["statement"]},
        ],
        "outputs": ["y"],
        "assumptions": [{"id": "data_assumption",
                         "text": f"{', '.join(features)} and {target} are numeric scalar observations; units are assumed dimensionless until confirmed."}],
    }
    return HypothesisIR.from_payload(payload, contract)


def build_scalar_graph_bundle(
    frame: pd.DataFrame,
    feature: str,
    target: str,
    *,
    statement: str | None = None,
    max_rows: int = 384,
    random_state: int = 42,
    feature_dimensions: Mapping[str, Any] | None = None,
    target_dimensions: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create a safe graph-search bundle and a non-authoritative data audit.

    Duplicate feature values are aggregated by the median target so the
    generated scalar function has one expected value per input.  The audit
    exposes that choice; it is not silently presented as a causal conclusion.
    """
    return build_tabular_graph_bundle(frame, [feature], target, statement=statement,
                                      max_rows=max_rows, random_state=random_state,
                                      feature_dimensions=feature_dimensions,
                                      target_dimensions=target_dimensions)


def build_tabular_graph_bundle(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    statement: str | None = None,
    max_rows: int = 384,
    random_state: int = 42,
    feature_dimensions: Mapping[str, Any] | None = None,
    target_dimensions: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a one- to four-feature graph-search bundle.

    The unknown mechanism may expose up to four observed inputs.  The search
    grammar lowers dimensionless multi-input candidates to a bounded binary
    fold, while explicit non-dimensionless multi-input lowering remains
    conservative and is left for a unit-aware compiler pass.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame_must_be_dataframe")
    if not isinstance(features, list) or not 1 <= len(features) <= 4:
        raise ValueError("one_to_four_features_required")
    if not all(isinstance(item, str) and item for item in features):
        raise ValueError("feature_names_required")
    if len(set(features)) != len(features) or target in features:
        raise ValueError("features_and_target_must_be_distinct")
    if not isinstance(target, str) or not target:
        raise ValueError("target_required")
    if not all(item in frame.columns for item in [*features, target]):
        raise ValueError("features_and_target_must_exist")
    if type(max_rows) is not int or not 8 <= max_rows <= 384:
        raise ValueError("max_rows_out_of_range")
    if type(random_state) is not int or not 0 <= random_state < 2**32:
        raise ValueError("invalid_random_state")
    if (feature_dimensions is None) != (target_dimensions is None):
        raise ValueError("feature_and_target_dimensions_must_be_provided_together")
    if feature_dimensions is not None:
        if not isinstance(feature_dimensions, Mapping) or set(feature_dimensions) != set(features):
            raise ValueError("feature_dimensions_must_cover_all_features")
        if not isinstance(target_dimensions, Mapping) or set(target_dimensions) != {target}:
            raise ValueError("target_dimensions_must_cover_target")
        normalized_feature_dimensions = {
            name: _dimension_payload(feature_dimensions[name]) for name in features
        }
        normalized_target_dimensions = _dimension_payload(target_dimensions[target])
    else:
        normalized_feature_dimensions, normalized_target_dimensions = {}, {}

    clean = pd.DataFrame({name: _finite_numeric(frame, name) for name in [*features, target]}).dropna()
    if clean.empty:
        raise ValueError("no_finite_numeric_rows")
    finite_rows_before_group = int(len(clean))
    duplicate_feature_rows = int(clean.duplicated(features).sum())
    clean = clean.groupby(features, as_index=False, sort=True)[target].median()
    distinct_rows_before_sampling = int(len(clean))
    if len(clean) < 4:
        raise ValueError("at_least_four_distinct_feature_tuples_required")
    if len(clean) > max_rows:
        positions = np.linspace(0, len(clean) - 1, max_rows, dtype=int)
        clean = clean.iloc[np.unique(positions)].reset_index(drop=True)
    if len(clean) < 4:
        raise ValueError("sampling_left_too_few_feature_tuples")

    rng = np.random.default_rng(random_state)
    groups = np.arange(len(clean))
    rng.shuffle(groups)
    train_count = max(3, min(len(clean) - 1, int(round(len(clean) * 0.7))))
    train_indices = set(groups[:train_count].tolist())
    search_indices = [int(index) for index in groups[train_count:]]
    train_indices_sorted = sorted(train_indices)
    search_indices_sorted = sorted(search_indices)

    domain = {}
    for index, name in enumerate(features):
        values = clean[name].to_numpy(dtype=float)
        low, high = float(np.min(values)), float(np.max(values))
        if not low < high:
            raise ValueError(f"feature_{name}_must_have_nonzero_domain")
        domain["x" if len(features) == 1 else f"x{index}"] = [low, high]
    statement = statement or (
        f"Infer a bounded scalar relation from observed numeric field(s) {', '.join(features)} to {target}; "
        "this adapter does not establish causality or physical units."
    )
    contract = ProblemContract.create(statement)
    template = _template(contract, features, target, normalized_feature_dimensions,
                         normalized_target_dimensions)

    def cases(indices: list[int], prefix: str) -> list[dict[str, Any]]:
        return [{"id": f"{prefix}_{position}",
                 "bindings": {("x" if len(features) == 1 else f"x{index}"): float(clean.iloc[position][name])
                              for index, name in enumerate(features)},
                 "expected": {"y": float(clean.iloc[position][target])}}
                for position in indices]

    experiment = SearchExperiment.create(
        contract, template, domain=domain, training_cases=cases(train_indices_sorted, "train"),
        search_cases=cases(search_indices_sorted, "search"), probe_count=12, seed=random_state,
    )
    bundle = {
        "schema_version": "mathmodel.graph-search-bundle/v1",
        "problem": contract.public(), "experiment": experiment.public(),
        "candidates": [template.payload()],
        "budget": {"max_candidates": 16, "max_patch_attempts": 64,
                    "max_evaluations": 4000, "per_candidate_evaluations": 500,
                    "wall_seconds": 30},
        "grammar_search": True,
    }
    audit = {
        "schema_version": BRIDGE_VERSION, "status": "bound_search_input",
        "features": list(features), "target": target, "rows_input": int(len(frame)),
        "rows_finite": finite_rows_before_group,
        "rows_distinct_feature_tuples": distinct_rows_before_sampling, "duplicate_feature_rows": duplicate_feature_rows,
        "rows_used": int(len(clean)), "train_rows": len(train_indices_sorted),
        "search_rows": len(search_indices_sorted), "dropped_nonfinite_rows": int(len(frame) - finite_rows_before_group),
        "aggregation": (("median_target_per_feature" if len(features) == 1 else
                          "median_target_per_feature_tuple") if duplicate_feature_rows else "none"),
        "unit_status": ("explicit_dimensions_bound" if feature_dimensions is not None
                        else "assumed_dimensionless_requires_confirmation"),
        "feature_dimensions": normalized_feature_dimensions,
        "target_dimensions": normalized_target_dimensions,
        "causal_status": "not_assessed",
        "holdout_policy": "search_partition_disjoint_by_feature_tuple",
    }
    if len(features) == 1:
        audit["feature"] = features[0]
        audit["rows_distinct_feature"] = distinct_rows_before_sampling
    return bundle, audit


def run_scalar_graph_search(
    frame: pd.DataFrame, feature: str, target: str, *, output_root: Path,
    statement: str | None = None, max_rows: int = 384, random_state: int = 42,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    """Build and execute the bounded graph search; returns result, run path, audit."""
    bundle, audit = build_scalar_graph_bundle(frame, feature, target, statement=statement,
                                               max_rows=max_rows, random_state=random_state)
    result, path = run_search_bundle(bundle, output_root=Path(output_root))
    return result, path, audit
