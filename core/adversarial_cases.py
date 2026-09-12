"""Deterministic boundary, extreme and perturbation case generation."""

from __future__ import annotations

import math
import random
from typing import Any, Mapping


class AdversarialCaseError(ValueError):
    pass


def build_adversarial_cases(
    bounds: Mapping[str, tuple[float, float] | list[float]], *,
    max_cases: int = 128, seed: int = 0, perturbation_fraction: float = 1e-6,
) -> list[dict[str, Any]]:
    """Create ordered endpoint/degenerate/extreme/random perturbation inputs.

    This function only creates inputs; a caller must provide the model-specific
    evaluator and feed the cases to the typed counterexample protocol. It never
    labels a generated point as a violation by itself.
    """
    if not isinstance(bounds, Mapping) or not bounds:
        raise AdversarialCaseError("bounds_required")
    if type(max_cases) is not int or not 1 <= max_cases <= 10_000:
        raise AdversarialCaseError("invalid_case_budget")
    if type(seed) is not int:
        raise AdversarialCaseError("seed_must_be_integer")
    try:
        fraction = float(perturbation_fraction)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AdversarialCaseError("invalid_perturbation_fraction") from exc
    if not math.isfinite(fraction) or not 0 < fraction <= 0.1:
        raise AdversarialCaseError("invalid_perturbation_fraction")
    normalized = []
    for name, pair in bounds.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise AdversarialCaseError("invalid_bound")
        try:
            lower, upper = float(pair[0]), float(pair[1])
        except (TypeError, ValueError, OverflowError) as exc:
            raise AdversarialCaseError("invalid_bound") from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
            raise AdversarialCaseError("invalid_bound")
        normalized.append((name.strip(), lower, upper))
    normalized.sort(key=lambda item: item[0])
    cases: list[dict[str, Any]] = []

    def add(kind: str, values: dict[str, float], suffix: str) -> None:
        if len(cases) < max_cases:
            cases.append({"case_id": f"{kind}_{suffix}_{len(cases):04d}", "kind": kind, "values": values})

    for name, lower, upper in normalized:
        add("endpoint", {name: lower}, f"{name}_low")
        add("endpoint", {name: upper}, f"{name}_high")
        if lower == upper:
            add("degenerate", {name: lower}, f"{name}_fixed")
        else:
            span = upper - lower
            add("degenerate", {name: lower + span / 2.0}, f"{name}_mid")
            add("optimized", {name: min(upper, lower + span * fraction)}, f"{name}_near_low")
            add("optimized", {name: max(lower, upper - span * fraction)}, f"{name}_near_high")
    rng = random.Random(seed)
    while len(cases) < max_cases and normalized:
        values = {name: rng.uniform(lower, upper) for name, lower, upper in normalized}
        add("random", values, f"seed_{seed}")
    return cases


__all__ = ["AdversarialCaseError", "build_adversarial_cases"]
