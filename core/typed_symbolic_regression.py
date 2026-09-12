"""Bounded execution of typed symbolic expression trees.

This is a small LLM-SR compatible *execution* slice.  It accepts JSON trees,
never source code, and fits only explicitly bounded scalar parameters.  The
result is evidence for the supplied split, not a symbolic proof or a causal
claim.
"""

from __future__ import annotations

import math
import copy
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "mathmodel.typed-symbolic-regression/v1"
_OPS = {"add", "subtract", "multiply", "divide", "negate", "abs", "sqrt", "exp", "log", "sin", "cos"}
_DIM_KEYS = ("M", "L", "T", "I", "Theta", "N", "J")


class TypedSymbolicRegressionError(ValueError):
    pass


def _dim_vector(value: Mapping[str, Any]) -> tuple[float, ...]:
    if not isinstance(value, Mapping) or any(key not in _DIM_KEYS for key in value):
        raise TypedSymbolicRegressionError("invalid_dimension_mapping")
    result = []
    for key in _DIM_KEYS:
        item = value.get(key, 0)
        if type(item) not in (int, float) or not math.isfinite(float(item)) or abs(float(item)) > 32:
            raise TypedSymbolicRegressionError("invalid_dimension_exponent")
        result.append(float(item))
    return tuple(result)


def _dim_payload(value: tuple[float, ...]) -> dict[str, int | float]:
    return {key: int(item) if float(item).is_integer() else float(item)
            for key, item in zip(_DIM_KEYS, value) if item != 0}


def _infer_dimensions(node: Mapping[str, Any], feature_dimensions: Mapping[str, Any], parameter_dimensions: Mapping[str, Any]) -> tuple[float, ...]:
    op = node["op"]
    if op == "var":
        if node["name"] not in feature_dimensions:
            raise TypedSymbolicRegressionError("variable_dimension_missing")
        return _dim_vector(feature_dimensions[node["name"]])
    if op == "param":
        if node["name"] not in parameter_dimensions:
            raise TypedSymbolicRegressionError("parameter_dimension_missing")
        return _dim_vector(parameter_dimensions[node["name"]])
    if op == "const":
        return (0.0,) * len(_DIM_KEYS)
    if "arg" in node:
        arg = _infer_dimensions(node["arg"], feature_dimensions, parameter_dimensions)
        if op in {"exp", "log", "sin", "cos"} and arg != (0.0,) * len(_DIM_KEYS):
            raise TypedSymbolicRegressionError("transcendental_requires_dimensionless")
        if op == "sqrt":
            return tuple(value / 2 for value in arg)
        return arg
    left = _infer_dimensions(node["left"], feature_dimensions, parameter_dimensions)
    right = _infer_dimensions(node["right"], feature_dimensions, parameter_dimensions)
    if op in {"add", "subtract"} and left != right:
        raise TypedSymbolicRegressionError("additive_dimension_mismatch")
    if op == "multiply":
        return tuple(a + b for a, b in zip(left, right))
    if op == "divide":
        return tuple(a - b for a, b in zip(left, right))
    return left


def _node(node: Any, *, depth: int = 0, count: list[int] | None = None) -> dict[str, Any]:
    count = count or [0]
    if depth > 12 or count[0] >= 64 or not isinstance(node, Mapping):
        raise TypedSymbolicRegressionError("expression_tree_budget_exceeded")
    count[0] += 1
    keys = set(node)
    op = node.get("op")
    if op == "var":
        if keys != {"op", "name"} or not isinstance(node.get("name"), str) or not node["name"]:
            raise TypedSymbolicRegressionError("invalid_variable_node")
        return {"op": "var", "name": node["name"]}
    if op == "param":
        if keys != {"op", "name"} or not isinstance(node.get("name"), str) or not node["name"]:
            raise TypedSymbolicRegressionError("invalid_parameter_node")
        return {"op": "param", "name": node["name"]}
    if op == "const":
        value = node.get("value")
        if keys != {"op", "value"} or type(value) not in (int, float) or not math.isfinite(float(value)):
            raise TypedSymbolicRegressionError("invalid_constant_node")
        return {"op": "const", "value": float(value)}
    if op not in _OPS:
        raise TypedSymbolicRegressionError("unsupported_expression_operator")
    expected = {"op", "arg"} if op in {"negate", "abs", "sqrt", "exp", "log", "sin", "cos"} else {"op", "left", "right"}
    if keys != expected:
        raise TypedSymbolicRegressionError("invalid_expression_node_fields")
    if "arg" in expected:
        return {"op": op, "arg": _node(node["arg"], depth=depth + 1, count=count)}
    return {"op": op, "left": _node(node["left"], depth=depth + 1, count=count),
            "right": _node(node["right"], depth=depth + 1, count=count)}


def _evaluate(node: Mapping[str, Any], variables: Mapping[str, np.ndarray], parameters: Mapping[str, float]) -> np.ndarray:
    op = node["op"]
    if op == "var":
        return variables[node["name"]]
    if op == "param":
        return np.full(len(next(iter(variables.values()))), float(parameters[node["name"]]))
    if op == "const":
        return np.full(len(next(iter(variables.values()))), float(node["value"]))
    if "arg" in node:
        arg = _evaluate(node["arg"], variables, parameters)
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            if op == "negate": return -arg
            if op == "abs": return np.abs(arg)
            if op == "sqrt": return np.sqrt(arg)
            if op == "exp": return np.exp(arg)
            if op == "log": return np.log(arg)
            if op == "sin": return np.sin(arg)
            if op == "cos": return np.cos(arg)
    left, right = (_evaluate(node[key], variables, parameters) for key in ("left", "right"))
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        if op == "add": return left + right
        if op == "subtract": return left - right
        if op == "multiply": return left * right
        if op == "divide": return left / right
    raise TypedSymbolicRegressionError("expression_evaluation_failed")


def fit_typed_symbolic_expression(
    target: Sequence[float], features: Mapping[str, Sequence[float]], expression: Mapping[str, Any],
    parameter_bounds: Mapping[str, Sequence[float]], *, validation_fraction: float = 0.25,
    max_nfev: int = 300, random_state: int = 0, feature_dimensions: Mapping[str, Any] | None = None,
    target_dimensions: Mapping[str, Any] | None = None, parameter_dimensions: Mapping[str, Any] | None = None,
    allowed_operators: Sequence[str] | None = None, allowed_variables: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Fit a bounded typed expression with a temporal holdout."""
    del random_state  # deterministic optimizer; retained for protocol compatibility
    y = np.asarray(target, dtype=float).reshape(-1)
    if not isinstance(features, Mapping) or not features:
        raise TypedSymbolicRegressionError("features_must_be_nonempty_mapping")
    x = {str(name): np.asarray(values, dtype=float).reshape(-1) for name, values in features.items()}
    if y.size < 24 or any(values.shape != y.shape for values in x.values()) or not np.isfinite(y).all() or any(not np.isfinite(v).all() for v in x.values()):
        raise TypedSymbolicRegressionError("symbolic_regression_shapes_or_values_invalid")
    if not 0.1 <= float(validation_fraction) <= 0.4 or type(max_nfev) is not int or not 20 <= max_nfev <= 2_000:
        raise TypedSymbolicRegressionError("symbolic_regression_budget_invalid")
    tree = _node(expression)
    # A proposal may carry a small basis contract.  This is a constraint on
    # the compiler, not a claim that the LLM discovered a valid basis.
    tree_ops: set[str] = set()
    tree_vars: set[str] = set()
    def collect_symbols(n: Mapping[str, Any]) -> None:
        tree_ops.add(str(n["op"]))
        if n["op"] == "var":
            tree_vars.add(str(n["name"]))
        for key in ("arg", "left", "right"):
            if key in n:
                collect_symbols(n[key])
    collect_symbols(tree)
    if allowed_operators is not None:
        if isinstance(allowed_operators, (str, bytes)) or not isinstance(allowed_operators, Sequence):
            raise TypedSymbolicRegressionError("allowed_operators_must_be_sequence")
        allowed = {str(item) for item in allowed_operators}
        if not allowed or not allowed.issubset(_OPS):
            raise TypedSymbolicRegressionError("allowed_operators_invalid")
        used_math_ops = tree_ops - {"var", "param", "const"}
        if not used_math_ops.issubset(allowed):
            raise TypedSymbolicRegressionError("expression_uses_disallowed_operator")
    if allowed_variables is not None:
        if isinstance(allowed_variables, (str, bytes)) or not isinstance(allowed_variables, Sequence):
            raise TypedSymbolicRegressionError("allowed_variables_must_be_sequence")
        allowed_vars = {str(item) for item in allowed_variables}
        if not tree_vars.issubset(allowed_vars):
            raise TypedSymbolicRegressionError("expression_uses_disallowed_variable")
    names = set()
    def collect(n):
        if n["op"] == "param": names.add(n["name"])
        for key in ("arg", "left", "right"):
            if key in n: collect(n[key])
    collect(tree)
    if set(parameter_bounds) != names or not names or len(names) > 16:
        raise TypedSymbolicRegressionError("parameter_bounds_must_match_expression")
    unit_status = "not_assessed"
    if feature_dimensions is not None or target_dimensions is not None or parameter_dimensions is not None:
        if not isinstance(feature_dimensions, Mapping) or not isinstance(target_dimensions, Mapping) or not isinstance(parameter_dimensions, Mapping):
            raise TypedSymbolicRegressionError("dimension_contract_incomplete")
        used_features = set()
        def collect_vars(n):
            if n["op"] == "var": used_features.add(n["name"])
            for key in ("arg", "left", "right"):
                if key in n: collect_vars(n[key])
        collect_vars(tree)
        if set(feature_dimensions) != used_features or set(parameter_dimensions) != names:
            raise TypedSymbolicRegressionError("dimension_contract_symbols_mismatch")
        inferred = _infer_dimensions(tree, feature_dimensions, parameter_dimensions)
        if inferred != _dim_vector(target_dimensions):
            raise TypedSymbolicRegressionError("expression_target_dimension_mismatch")
        unit_status = "verified_for_expression"
    bounds = {}
    for name, value in parameter_bounds.items():
        if not isinstance(value, Sequence) or len(value) != 2 or not all(type(v) in (int, float) and math.isfinite(float(v)) for v in value) or float(value[0]) >= float(value[1]):
            raise TypedSymbolicRegressionError("invalid_parameter_bounds")
        bounds[name] = (float(value[0]), float(value[1]))
    split = max(16, min(y.size - 8, int(math.floor(y.size * (1.0 - float(validation_fraction))))))
    ordered = list(names)
    lower, upper = np.asarray([bounds[n][0] for n in ordered]), np.asarray([bounds[n][1] for n in ordered])
    train_vars = {name: values[:split] for name, values in x.items()}
    valid_vars = {name: values[split:] for name, values in x.items()}
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return {"schema_version": SCHEMA_VERSION, "status": "unavailable", "reason": "scipy_not_installed"}
    def residual(theta, vars_, truth):
        try:
            prediction = _evaluate(tree, vars_, dict(zip(ordered, theta)))
            if not np.isfinite(prediction).all(): raise FloatingPointError()
            return prediction - truth
        except (FloatingPointError, OverflowError, ZeroDivisionError, KeyError, ValueError):
            return np.full(len(truth), 1e6, dtype=float)
    try:
        fit = least_squares(lambda theta: residual(theta, train_vars, y[:split]), (lower + upper) / 2,
                            bounds=(lower, upper), max_nfev=max_nfev)
        train_pred = _evaluate(tree, train_vars, dict(zip(ordered, fit.x)))
        valid_pred = _evaluate(tree, valid_vars, dict(zip(ordered, fit.x)))
        if not np.isfinite(train_pred).all() or not np.isfinite(valid_pred).all(): raise FloatingPointError()
    except (FloatingPointError, OverflowError, ZeroDivisionError, ValueError, RuntimeError):
        return {"schema_version": SCHEMA_VERSION, "status": "rejected_nonfinite_or_solver_failure"}
    train_rmse = float(np.sqrt(np.mean((train_pred - y[:split]) ** 2)))
    holdout_rmse = float(np.sqrt(np.mean((valid_pred - y[split:]) ** 2)))
    return {"schema_version": SCHEMA_VERSION, "status": "fitted", "train_rows": int(split),
            "holdout_rows": int(y.size - split), "train_rmse": train_rmse, "holdout_rmse": holdout_rmse,
            "baseline_holdout_rmse": float(np.sqrt(np.mean((np.mean(y[:split]) - y[split:]) ** 2))),
            "parameters": {name: float(value) for name, value in zip(ordered, fit.x)},
            "expression": tree, "complexity": int(_node_count(tree)), "unit_status": unit_status,
            "basis_contract": {
                "allowed_operators": sorted({str(item) for item in allowed_operators}) if allowed_operators is not None else None,
                "allowed_variables": sorted({str(item) for item in allowed_variables}) if allowed_variables is not None else None,
                "used_operators": sorted(tree_ops), "used_variables": sorted(tree_vars),
            },
            "policy": "typed_tree_only; bounded_parameters; temporal_holdout; no_symbolic_or_causal_proof"}


def fit_typed_symbolic_candidates(
    target: Sequence[float], features: Mapping[str, Sequence[float]],
    candidates: Sequence[Mapping[str, Any]], *, validation_fraction: float = 0.25,
    max_nfev: int = 300, random_state: int = 0, feature_dimensions: Mapping[str, Any] | None = None,
    target_dimensions: Mapping[str, Any] | None = None, parameter_dimensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate a bounded pool of typed trees on one shared data split."""
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence) or not 1 <= len(candidates) <= 8:
        raise TypedSymbolicRegressionError("candidate_pool_size_invalid")
    reports = []
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping) or not {"expression", "parameter_bounds"}.issubset(set(candidate)):
            reports.append({"id": f"candidate_{index}", "status": "rejected", "reason": "candidate_fields_invalid"})
            continue
        unknown = set(candidate) - {"expression", "parameter_bounds", "allowed_operators", "allowed_variables"}
        if unknown:
            reports.append({"id": f"candidate_{index}", "status": "rejected", "reason": "candidate_fields_invalid"})
            continue
        try:
            report = fit_typed_symbolic_expression(
                target, features, candidate["expression"], candidate["parameter_bounds"],
                validation_fraction=validation_fraction, max_nfev=max_nfev, random_state=random_state,
                feature_dimensions=feature_dimensions, target_dimensions=target_dimensions,
                parameter_dimensions=parameter_dimensions,
                allowed_operators=candidate.get("allowed_operators"),
                allowed_variables=candidate.get("allowed_variables"),
            )
            reports.append({"id": f"candidate_{index}", **report})
        except (TypedSymbolicRegressionError, TypeError, ValueError, OverflowError) as exc:
            reports.append({"id": f"candidate_{index}", "status": "rejected", "reason": str(exc)})
    fitted = [item for item in reports if item.get("status") == "fitted" and math.isfinite(float(item.get("holdout_rmse", math.inf)))]
    fitted.sort(key=lambda item: (float(item["holdout_rmse"]), int(item.get("complexity", 10**9)), item["id"]))
    return {
        "schema_version": "mathmodel.typed-symbolic-regression-pool/v1",
        "status": "candidates_need_confirmation" if fitted else "candidate_set_inadequate",
        "candidate_count": len(reports), "fitted_count": len(fitted),
        "reports": reports, "ranked_ids": [item["id"] for item in fitted],
        "best_id": fitted[0]["id"] if fitted else None,
        "policy": "shared_temporal_holdout; ranking_is_not_approval; independent_confirmation_required",
    }


def _candidate_digest(candidate: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(candidate), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _mutate_constant_nodes(expression: Mapping[str, Any], *, limit: int = 8) -> list[dict[str, Any]]:
    """Generate a small, deterministic neighborhood without source code."""
    found: list[dict[str, Any]] = []
    def walk(node: Mapping[str, Any], path: tuple[str, ...] = ()) -> None:
        if len(found) >= limit:
            return
        if node.get("op") == "const":
            value = float(node["value"])
            step = 0.1 if abs(value) <= 10 else min(1e6, 0.1 * abs(value))
            for delta in (-step, step):
                revised = copy.deepcopy(dict(expression))
                current: Any = revised
                for key in path:
                    current = current[key]
                current["value"] = value + delta
                found.append(revised)
                if len(found) >= limit:
                    return
        for key in ("arg", "left", "right"):
            child = node.get(key)
            if isinstance(child, Mapping):
                walk(child, path + (key,))
    walk(expression)
    return found


def search_typed_symbolic_candidates(
    target: Sequence[float], features: Mapping[str, Sequence[float]],
    candidates: Sequence[Mapping[str, Any]], *, generations: int = 3,
    max_candidates: int = 24, beam_width: int = 4, validation_fraction: float = 0.25,
    max_nfev: int = 300, random_state: int = 0, feature_dimensions: Mapping[str, Any] | None = None,
    target_dimensions: Mapping[str, Any] | None = None, parameter_dimensions: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded typed-tree beam search with explicit candidate lineage.

    The search mutates only numeric constants; it is a local baseline for
    execution feedback, not the LLM-SR paper implementation or a symbolic
    discovery proof.  Every generation shares the same temporal split.
    """
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence) or not candidates:
        raise TypedSymbolicRegressionError("candidate_pool_size_invalid")
    if type(generations) is not int or not 1 <= generations <= 16 or type(max_candidates) is not int or not 1 <= max_candidates <= 128 or type(beam_width) is not int or not 1 <= beam_width <= 16:
        raise TypedSymbolicRegressionError("candidate_search_budget_invalid")
    pool = [dict(item) for item in candidates[:8] if isinstance(item, Mapping)]
    if not pool:
        raise TypedSymbolicRegressionError("candidate_pool_size_invalid")
    seen = {_candidate_digest(item) for item in pool}
    lineage: dict[str, str | None] = {digest: None for digest in seen}
    generations_out: list[dict[str, Any]] = []
    for generation in range(generations):
        report = fit_typed_symbolic_candidates(
            target, features, pool, validation_fraction=validation_fraction, max_nfev=max_nfev,
            random_state=random_state, feature_dimensions=feature_dimensions,
            target_dimensions=target_dimensions, parameter_dimensions=parameter_dimensions,
        )
        generations_out.append({"generation": generation, "candidate_count": len(pool), "report": report})
        fitted = [item for item in report["reports"] if item.get("status") == "fitted"]
        if generation + 1 >= generations or not fitted:
            break
        fitted.sort(key=lambda item: (float(item.get("holdout_rmse", math.inf)), int(item.get("complexity", 10**9)), item["id"]))
        next_pool: list[dict[str, Any]] = []
        for item in fitted[:beam_width]:
            index = int(str(item["id"]).split("_")[-1])
            if index >= len(pool):
                continue
            base = pool[index]
            base_hash = _candidate_digest(base)
            for expression in _mutate_constant_nodes(base.get("expression", {}), limit=4):
                proposal = copy.deepcopy(base)
                proposal["expression"] = expression
                digest = _candidate_digest(proposal)
                if digest in seen:
                    continue
                seen.add(digest)
                lineage[digest] = base_hash
                next_pool.append(proposal)
                if len(seen) >= max_candidates:
                    break
            if len(seen) >= max_candidates:
                break
        if not next_pool:
            break
        pool = next_pool[:8]
    final_report = generations_out[-1]["report"] if generations_out else {"ranked_ids": []}
    return {
        "schema_version": "mathmodel.typed-symbolic-regression-search/v1",
        "status": "candidates_need_confirmation" if final_report.get("ranked_ids") else "candidate_set_inadequate",
        "generations": generations_out, "candidate_count": len(seen),
        "lineage": lineage, "best_id": final_report.get("best_id"),
        "policy": "typed_tree_only; bounded_constant_mutation; shared_temporal_holdout; ranking_is_not_approval",
    }


def _node_count(node: Mapping[str, Any]) -> int:
    return 1 + sum(_node_count(node[key]) for key in ("arg", "left", "right") if key in node)


__all__ = ["SCHEMA_VERSION", "TypedSymbolicRegressionError", "fit_typed_symbolic_expression", "fit_typed_symbolic_candidates", "search_typed_symbolic_candidates"]
