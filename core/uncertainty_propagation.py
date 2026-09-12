"""Auditable propagation of semantic, structural, parameter and numerical spread.

This module does not turn model scores into probabilities.  It accepts a finite
set of already-evaluated outputs and applies the law of total variance along a
declared hierarchy::

    semantic -> structure -> parameter -> numerical replicate

The hierarchy is an accounting device: every row must carry an explicit unit
signature and layer identifiers.  Missing identifiers stop the corresponding
decomposition instead of silently reporting a zero component.  The result is
therefore useful for comparing competing explanations while remaining honest
about the evidence available.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "mathmodel.uncertainty-propagation/v1"
_LAYERS = ("semantic", "structure", "parameter", "numerical")
_ID_KEYS = tuple(f"{layer}_id" for layer in _LAYERS)


class UncertaintyPropagationError(ValueError):
    """Raised when layered uncertainty input is unsafe or inconsistent."""


def _finite(value: Any, code: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise UncertaintyPropagationError(code) from exc
    if not math.isfinite(result):
        raise UncertaintyPropagationError(code)
    return result


def _identifier(value: Any, key: str) -> str:
    if not isinstance(value, (str, int)) or isinstance(value, bool):
        raise UncertaintyPropagationError(f"{key}_must_be_scalar_identifier")
    result = str(value).strip()
    if not 1 <= len(result) <= 128:
        raise UncertaintyPropagationError(f"{key}_must_be_nonempty")
    return result


def _weighted_stats(rows: Sequence[tuple[tuple[str, ...], float, float]]) -> dict[tuple[str, ...], dict[str, float]]:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for key, value, weight in rows:
        entry = grouped[key]
        entry[0] += weight
        entry[1] += weight * value
        entry[2] += weight * value * value
    result: dict[tuple[str, ...], dict[str, float]] = {}
    for key, (mass, first, second) in grouped.items():
        mean = first / mass
        variance = max(0.0, second / mass - mean * mean)
        result[key] = {"weight": mass, "mean": mean, "variance": variance}
    return result


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], probability: float) -> float:
    order = sorted(zip(values, weights), key=lambda pair: pair[0])
    target = probability * sum(weights)
    cumulative = 0.0
    for value, weight in order:
        cumulative += weight
        if cumulative >= target:
            return float(value)
    return float(order[-1][0])


def _conditional_between(
    rows: Sequence[tuple[tuple[str, ...], float, float]],
    child_depth: int,
) -> tuple[float, int]:
    """Return E[Var(E[Y|child] | parent)] for one hierarchy edge."""
    parent_rows: dict[tuple[str, ...], list[tuple[tuple[str, ...], float, float]]] = defaultdict(list)
    for key, value, weight in rows:
        parent_rows[key[: child_depth - 1]].append((key[:child_depth], value, weight))
    numerator, denominator, group_count = 0.0, 0.0, 0
    for parent, items in parent_rows.items():
        child_stats = _weighted_stats(items)
        if len(child_stats) <= 1:
            # The component is genuinely zero for this parent, but it still
            # participates in the weighted average.
            local = 0.0
        else:
            total = sum(item["weight"] for item in child_stats.values())
            mean = sum(item["weight"] * item["mean"] for item in child_stats.values()) / total
            local = sum(item["weight"] * (item["mean"] - mean) ** 2
                        for item in child_stats.values()) / total
        parent_weight = sum(item[2] for item in items)
        numerator += parent_weight * local
        denominator += parent_weight
        group_count += len(child_stats)
    return (numerator / denominator if denominator else 0.0), group_count


def propagate_layered_uncertainty(
    evaluations: Sequence[Mapping[str, Any]], *, value_key: str = "value",
    max_records: int = 200_000, unit_signature: str | None = None,
) -> dict[str, Any]:
    """Decompose finite output variation across four declared uncertainty layers.

    ``evaluations`` contains one scalar output per numerical replicate.  Each
    row must provide ``semantic_id``, ``structure_id``, ``parameter_id`` and
    ``numerical_id`` plus an optional positive ``weight``.  A row may provide
    ``unit_signature`` instead of the function-level unit.  All rows must use
    the same unit; no conversion is inferred here.

    The returned component values are empirical variance contributions, not
    posterior variances or confidence intervals.  Missing layer identifiers
    produce a ``partial`` result and leave that component (and deeper ones)
    explicitly unassessed.
    """
    if type(max_records) is not int or not 1 <= max_records <= 1_000_000:
        raise UncertaintyPropagationError("invalid_record_budget")
    if not isinstance(evaluations, Sequence) or isinstance(evaluations, (str, bytes)):
        raise UncertaintyPropagationError("evaluations_must_be_a_sequence")
    if not 1 <= len(evaluations) <= max_records:
        raise UncertaintyPropagationError("evaluation_count_out_of_bounds")
    if not isinstance(value_key, str) or not 1 <= len(value_key) <= 128:
        raise UncertaintyPropagationError("invalid_value_key")
    declared_unit = None if unit_signature is None else _identifier(unit_signature, "unit_signature")
    rows: list[tuple[tuple[str, ...], float, float]] = []
    units: set[str] = set()
    missing_layers: set[str] = set()
    for index, raw in enumerate(evaluations):
        if not isinstance(raw, Mapping):
            raise UncertaintyPropagationError(f"evaluation_{index}_must_be_object")
        if value_key not in raw:
            raise UncertaintyPropagationError(f"missing_value:{value_key}")
        value = _finite(raw[value_key], "output_must_be_finite")
        weight = _finite(raw.get("weight", 1.0), "weight_must_be_finite")
        if weight <= 0:
            raise UncertaintyPropagationError("weight_must_be_positive")
        identifiers: list[str] = []
        for layer, key in zip(_LAYERS, _ID_KEYS):
            if key not in raw or raw[key] is None or str(raw[key]).strip() == "":
                missing_layers.add(layer)
                identifiers.append(f"__missing__:{layer}")
            else:
                identifiers.append(_identifier(raw[key], key))
        row_unit = raw.get("unit_signature", declared_unit)
        if row_unit is not None:
            units.add(_identifier(row_unit, "unit_signature"))
        rows.append((tuple(identifiers), value, weight))
    if declared_unit is not None:
        units.add(declared_unit)
    if len(units) > 1:
        raise UncertaintyPropagationError("unit_signatures_must_match")
    if not units:
        return {
            "schema_version": SCHEMA_VERSION, "status": "not_assessed",
            "reason": "unit_signature_required", "record_count": len(rows),
            "policy": "empirical_layered_variance_not_probability_or_confidence",
        }
    total_weight = sum(weight for _, _, weight in rows)
    values = [value for _, value, _ in rows]
    weights = [weight for _, _, weight in rows]
    mean = sum(value * weight for value, weight in zip(values, weights)) / total_weight
    total_variance = max(0.0, sum(weight * (value - mean) ** 2 for value, weight in zip(values, weights)) / total_weight)
    components: dict[str, dict[str, Any]] = {}
    can_decompose = True
    raw_rows = [(key, value, weight) for key, value, weight in rows]
    for index, layer in enumerate(_LAYERS):
        if can_decompose and layer not in missing_layers:
            if index < len(_LAYERS) - 1:
                variance, group_count = _conditional_between(raw_rows, index + 1)
            else:
                # Numerical is the residual within the deepest parameter
                # group; this is the exact final term in total variance.
                parameter_groups = _weighted_stats([(key[:3], value, weight) for key, value, weight in raw_rows])
                numerator = 0.0
                for group, stats in parameter_groups.items():
                    # The group-level variance includes every numerical
                    # replicate within that parameter setting.  It is exactly
                    # the residual after semantic/structure/parameter means.
                    local_variance = sum(weight * (value - stats["mean"]) ** 2
                                         for key, value, weight in raw_rows if key[:3] == group) / stats["weight"]
                    numerator += stats["weight"] * local_variance
                variance = numerator / total_weight if total_weight else 0.0
                group_count = len(parameter_groups)
            components[layer] = {"status": "assessed", "variance": float(variance),
                                "standard_deviation": float(math.sqrt(max(0.0, variance))),
                                "group_count": int(group_count)}
        else:
            can_decompose = False
            components[layer] = {"status": "not_assessed", "variance": None,
                                "standard_deviation": None, "group_count": 0,
                                "reason": f"missing_{layer}_identifier"}
    assessed = [item["variance"] for item in components.values() if item["status"] == "assessed"]
    reconstructed = float(sum(assessed)) if len(assessed) == len(_LAYERS) else None
    residual = None if reconstructed is None else float(total_variance - reconstructed)
    status = "ok" if reconstructed is not None else "partial"
    if len(rows) < 2:
        status = "partial"
    return {
        "schema_version": SCHEMA_VERSION, "status": status,
        "unit_signature": next(iter(units)), "record_count": len(rows),
        "effective_sample_size": float(total_weight ** 2 / sum(weight * weight for weight in weights)),
        "output_summary": {"mean": float(mean), "variance": float(total_variance),
                           "standard_deviation": float(math.sqrt(total_variance)),
                           "q05": _weighted_quantile(values, weights, 0.05),
                           "q50": _weighted_quantile(values, weights, 0.50),
                           "q95": _weighted_quantile(values, weights, 0.95)},
        "variance_components": components,
        "reconstructed_variance": reconstructed,
        "decomposition_residual": residual,
        "missing_layers": sorted(missing_layers, key=_LAYERS.index),
        "policy": "empirical_layered_variance_not_probability_or_confidence",
    }


__all__ = ["SCHEMA_VERSION", "UncertaintyPropagationError", "propagate_layered_uncertainty"]
