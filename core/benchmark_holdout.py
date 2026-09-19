"""Structure-held-out partitions for automatic benchmark cases.

The partitioner is intentionally group based: parameter variants sharing a
mathematical structure never leak across development, confirmation and final
sets.  It does not claim that a synthetic group is a real-world unseen task.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from .automated_benchmark import AutomatedBenchmarkCase


class BenchmarkHoldoutError(ValueError):
    pass


def split_structure_holdout(
    cases: Sequence[AutomatedBenchmarkCase], *,
    confirmation_groups: Sequence[str] | None = None,
    final_groups: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Split cases without separating variants from one structure group."""
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise BenchmarkHoldoutError("cases_required")
    normalized = []
    for case in cases:
        if not isinstance(case, AutomatedBenchmarkCase):
            raise BenchmarkHoldoutError("case_type_invalid")
        case.validate()
        normalized.append(case)
    groups = sorted({case.structure_group for case in normalized})
    if len(groups) < 3:
        raise BenchmarkHoldoutError("at_least_three_structure_groups_required")
    confirmation = set(str(value) for value in (confirmation_groups or ()))
    final = set(str(value) for value in (final_groups or ()))
    if not confirmation and not final:
        # Deterministic defaults: never choose by case order or random seed.
        final = {groups[-1]}
        confirmation = {groups[-2]}
    if not confirmation or not final or confirmation & final:
        raise BenchmarkHoldoutError("holdout_groups_must_be_nonempty_and_disjoint")
    unknown = (confirmation | final) - set(groups)
    if unknown:
        raise BenchmarkHoldoutError("holdout_group_unknown")
    development = set(groups) - confirmation - final
    if not development:
        raise BenchmarkHoldoutError("development_groups_must_remain")
    buckets: dict[str, list[AutomatedBenchmarkCase]] = defaultdict(list)
    for case in normalized:
        bucket = "final" if case.structure_group in final else "confirmation" if case.structure_group in confirmation else "development"
        buckets[bucket].append(case)
    return {
        "schema_version": "mathmodel.benchmark-holdout/v1",
        "development": tuple(buckets["development"]),
        "confirmation": tuple(buckets["confirmation"]),
        "final": tuple(buckets["final"]),
        "groups": {
            "development": sorted(development),
            "confirmation": sorted(confirmation),
            "final": sorted(final),
        },
        "case_counts": {name: len(buckets[name]) for name in ("development", "confirmation", "final")},
        "policy": "structure_group_is_atomic;_do_not_tune_on_confirmation_or_final",
    }


__all__ = ["BenchmarkHoldoutError", "split_structure_holdout"]
