"""Check that cache/parallel/warm-start routes preserve approved outputs."""
from __future__ import annotations

import math
from typing import Any, Callable, Mapping


class CacheConsistencyError(ValueError):
    pass


_DEFAULT_IGNORED = {"cache_hit", "duration_seconds", "wall_seconds", "resource_usage", "timing"}


def _compare(left: Any, right: Any, path: str, tolerance: float, ignored: set[str],
             mismatches: list[dict[str, Any]], state: dict[str, Any], *, depth: int = 0) -> None:
    state["nodes"] += 1
    if depth > state["max_depth"] or state["nodes"] > state["max_nodes"]:
        mismatches.append({"path": path, "reason": "comparison_budget_exceeded"})
        return
    if path.rsplit(".", 1)[-1] in ignored:
        return
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        pair = (id(left), id(right))
        if pair in state["pairs"]:
            mismatches.append({"path": path, "reason": "cycle_detected"})
            return
        state["pairs"].add(pair)
        try:
            for key in sorted(set(left) | set(right), key=str):
                child = f"{path}.{key}" if path else str(key)
                if key not in left or key not in right:
                    if str(key) not in ignored:
                        mismatches.append({"path": child, "reason": "missing_key"})
                else:
                    _compare(left[key], right[key], child, tolerance, ignored, mismatches, state, depth=depth + 1)
        finally:
            state["pairs"].remove(pair)
        return
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        pair = (id(left), id(right))
        if pair in state["pairs"]:
            mismatches.append({"path": path, "reason": "cycle_detected"})
            return
        state["pairs"].add(pair)
        try:
            if len(left) != len(right):
                mismatches.append({"path": path, "reason": "length_mismatch"}); return
            for index, (a, b) in enumerate(zip(left, right)):
                _compare(a, b, f"{path}[{index}]", tolerance, ignored, mismatches, state, depth=depth + 1)
        finally:
            state["pairs"].remove(pair)
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)) and not isinstance(left, bool) and not isinstance(right, bool):
        if not math.isfinite(float(left)) or not math.isfinite(float(right)) or not math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance):
            mismatches.append({"path": path, "reason": "numeric_mismatch", "left": left, "right": right})
        return
    if left != right:
        mismatches.append({"path": path, "reason": "value_mismatch", "left": str(left)[:200], "right": str(right)[:200]})


def compare_cached_uncached(
    run_uncached: Callable[[], Mapping[str, Any]], run_cached: Callable[[], Mapping[str, Any]],
    *, tolerance: float = 1e-9, ignored_fields: set[str] | None = None,
    max_nodes: int = 100_000, max_depth: int = 32,
) -> dict[str, Any]:
    if not callable(run_uncached) or not callable(run_cached):
        raise CacheConsistencyError("runners_required")
    if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) or tolerance < 0:
        raise CacheConsistencyError("invalid_tolerance")
    if type(max_nodes) is not int or not 1 <= max_nodes <= 1_000_000:
        raise CacheConsistencyError("invalid_comparison_node_budget")
    if type(max_depth) is not int or not 1 <= max_depth <= 128:
        raise CacheConsistencyError("invalid_comparison_depth_budget")
    try:
        uncached = run_uncached(); cached = run_cached()
    except Exception as exc:
        return {"schema_version": "mathmodel.cache-consistency/v1", "status": "not_assessed",
                "reason": "route_execution_failed", "error_code": type(exc).__name__}
    if not isinstance(uncached, Mapping) or not isinstance(cached, Mapping):
        raise CacheConsistencyError("runner_must_return_mapping")
    mismatches: list[dict[str, Any]] = []
    ignored = set(ignored_fields or _DEFAULT_IGNORED)
    if not all(isinstance(item, str) and item for item in ignored):
        raise CacheConsistencyError("ignored_fields_must_be_nonempty_strings")
    state = {"nodes": 0, "max_nodes": max_nodes, "max_depth": max_depth, "pairs": set()}
    _compare(uncached, cached, "", float(tolerance), ignored, mismatches, state)
    return {"schema_version": "mathmodel.cache-consistency/v1",
            "status": "tested_not_falsified" if not mismatches else "counterexample",
            "mismatch_count": len(mismatches), "mismatches": mismatches[:128],
            "comparison_nodes": state["nodes"], "comparison_limits": {"max_nodes": max_nodes, "max_depth": max_depth},
            "policy": "finite_route_comparison_does_not_prove_all_future_cache_equivalence"}


__all__ = ["CacheConsistencyError", "compare_cached_uncached"]
