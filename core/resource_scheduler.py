"""Deterministic admission control for large-data modeling workloads."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any, Mapping


class ResourceSchedulerError(ValueError):
    """Raised for invalid resource estimates or budgets."""


@dataclass(frozen=True)
class ResourceBudget:
    max_rows: int = 2_000_000
    max_columns: int = 512
    max_memory_mb: int = 2048
    max_pairwise_cells: int = 5_000_000
    max_candidates: int = 32
    max_threads: int = 2

    def __post_init__(self) -> None:
        for name in asdict(self):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ResourceSchedulerError(f"invalid_{name}")


def estimate_workload(*, rows: int, columns: int, pairwise: bool = False,
                      candidates: int = 1, bytes_per_cell: int = 8) -> dict[str, int]:
    if type(pairwise) is not bool:
        raise ResourceSchedulerError("pairwise_must_be_bool")
    values = {"rows": rows, "columns": columns, "candidates": candidates,
              "bytes_per_cell": bytes_per_cell}
    if (any(type(value) is not int or value < 0 for value in values.values()) or
            bytes_per_cell == 0 or candidates == 0):
        raise ResourceSchedulerError("workload_dimensions_must_be_nonnegative_integers")
    cells = rows * columns
    pairwise_cells = rows * rows if pairwise else 0
    estimated_bytes = (pairwise_cells if pairwise else cells) * bytes_per_cell
    estimated_memory_mb = max(1, math.ceil(estimated_bytes / (1024 * 1024)))
    return {"rows": rows, "columns": columns, "cells": cells,
            "pairwise_cells": pairwise_cells, "estimated_memory_mb": estimated_memory_mb,
            "candidates": candidates}


def admit_workload(workload: Mapping[str, Any], budget: ResourceBudget | None = None) -> dict[str, Any]:
    """Return an explicit admission decision; never silently sample or shrink work."""
    if not isinstance(workload, Mapping):
        raise ResourceSchedulerError("workload_must_be_mapping")
    budget = budget or ResourceBudget()
    if not isinstance(budget, ResourceBudget):
        raise ResourceSchedulerError("budget_must_be_resource_budget")
    required = ("rows", "columns", "pairwise_cells", "estimated_memory_mb", "candidates")
    if any(key not in workload for key in required):
        raise ResourceSchedulerError("workload_estimate_incomplete")
    normalized = {}
    for key in required:
        value = workload[key]
        if type(value) is not int or value < 0:
            raise ResourceSchedulerError("workload_estimate_values_must_be_nonnegative_integers")
        normalized[key] = value
    checks = {
        "rows": normalized["rows"] <= budget.max_rows,
        "columns": normalized["columns"] <= budget.max_columns,
        "pairwise_cells": normalized["pairwise_cells"] <= budget.max_pairwise_cells,
        "memory": normalized["estimated_memory_mb"] <= budget.max_memory_mb,
        "candidates": normalized["candidates"] <= budget.max_candidates,
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {"status": "admitted" if not failed else "rejected_resource_budget",
            "checks": checks, "failed_checks": failed, "budget": asdict(budget),
            "workload": normalized,
            "policy": "rejection_is_not_a_mathematical_counterexample"}


__all__ = ["ResourceSchedulerError", "ResourceBudget", "estimate_workload", "admit_workload"]
