"""Conservative backend and fidelity planning for typed mathematical IR.

The planner only describes an execution route.  It never turns a proposed
model into executable code and never overrides contract or evidence gates.
This keeps scale-aware routing separate from solver correctness claims.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Optional


class BackendPlanError(ValueError):
    """The IR is too malformed or too large to produce a safe route."""


@dataclass(frozen=True)
class BackendPlan:
    status: str
    route: str
    executor_key: Optional[str]
    backend: str
    fidelity: str
    reason: str
    estimated_variables: int
    estimated_points: int
    estimated_cells: int
    resource_budget: dict[str, Any]

    def public(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "route": self.route,
            "executor_key": self.executor_key,
            "backend": self.backend,
            "fidelity": self.fidelity,
            "reason": self.reason,
            "estimated_variables": self.estimated_variables,
            "estimated_points": self.estimated_points,
            "estimated_cells": self.estimated_cells,
            "resource_budget": dict(self.resource_budget),
            "authority": "routing_plan_only",
        }


_SUPERVISED_EXECUTORS = frozenset({
    "adaptive_ode/v1", "bounded_nlp/v1", "scalar_graph/v1", "scalar_graph_confirm/v1",
    "linear_ode/v1", "threshold_event/v1", "quadratic_program/v1",
    "distance/v1", "interval_union/v1", "region_membership/v1", "segment_intersection/v1",
    "line_of_sight/v1", "normal_log_likelihood/v1", "bootstrap_mean/v1", "permutation_test/v1",
})


def _bounded_int(value: Any, *, default: int = 0, maximum: int = 10_000_000) -> int:
    if value is None or isinstance(value, bool):
        return default
    if type(value) is not int:
        return default
    return max(0, min(value, maximum))


def _estimate_contract(contract: Mapping[str, Any]) -> tuple[int, int, int]:
    variables = contract.get("variables", contract.get("decision_variables", []))
    variable_count = len(variables) if isinstance(variables, list) else 0
    variable_count = max(variable_count, len(contract.get("state_variables", [])) if isinstance(contract.get("state_variables"), list) else 0)
    scale_keys = ("output_points", "horizon", "steps", "sample_count", "n_samples")
    invalid_scale = any(
        key in contract and contract[key] is not None
        and (type(contract[key]) is not int or contract[key] < 0 or contract[key] > 10_000_000)
        for key in scale_keys
    )
    points = max(
        *(_bounded_int(contract.get(key)) for key in scale_keys),
        len(contract.get("times", [])) if isinstance(contract.get("times"), list) else 0,
        len(contract.get("values", [])) if isinstance(contract.get("values"), list) else 0,
    )
    cells = 0
    invalid_matrix = False
    for name in ("coefficient_matrix", "design_matrix", "quadratic_matrix", "A_ub", "A_eq", "transition_matrix", "matrix"):
        value = contract.get(name)
        if isinstance(value, list):
            rows = len(value)
            invalid_matrix = invalid_matrix or any(not isinstance(row, list) for row in value)
            cols = max((len(row) for row in value if isinstance(row, list)), default=0)
            cells = max(cells, rows * cols)
        elif value is not None:
            invalid_matrix = True
    if variable_count and points:
        cells = max(cells, variable_count * points)
    # A malformed declared scale/matrix must not be treated as zero.  Return a
    # deliberately over-budget estimate so the caller defers execution rather
    # than routing an underestimated workload to a solver.
    if invalid_scale or invalid_matrix:
        return variable_count, 10_000_001, 10_000_001
    return variable_count, points, cells


def plan_backend(
    node: Mapping[str, Any],
    specification: Any = None,
    *,
    max_cells: int = 2_000_000,
    max_points: int = 1_000_000,
) -> BackendPlan:
    """Choose a bounded route from a validated IR node and solver metadata.

    ``specification`` is intentionally duck-typed so the planner stays free of
    registry imports and can be used by custom registries.  A missing or
    deferred contract is reported as such; it is never downgraded to an
    approximate numerical answer.
    """
    if not isinstance(node, Mapping):
        raise BackendPlanError("node_must_be_mapping")
    if type(max_cells) is not int or max_cells < 1 or max_cells > 50_000_000:
        raise BackendPlanError("invalid_cell_budget")
    if type(max_points) is not int or max_points < 1 or max_points > 50_000_000:
        raise BackendPlanError("invalid_point_budget")
    contract = node.get("execution_contract", {})
    if not isinstance(contract, Mapping):
        contract = {}
    variable_count, points, cells = _estimate_contract(contract)
    form = str(node.get("mathematical_form", "unknown"))
    if specification is None:
        return BackendPlan("deferred", "defer", None, "none", "none",
                           "no_solver_specification", variable_count, points, cells, {})
    executor = str(getattr(specification, "executor_key", "") or "")
    max_variables = _bounded_int(getattr(specification, "max_variables", 0), maximum=1_000_000)
    if node.get("status") != "executable":
        return BackendPlan("deferred", "defer", executor or None, "none", "none",
                           "mathematical_contract_not_verified", variable_count, points, cells, {})
    if not executor or max_variables <= 0:
        return BackendPlan("deferred", "defer", None, "none", "none",
                           "invalid_solver_specification", variable_count, points, cells, {})
    if variable_count > max_variables:
        return BackendPlan("deferred", "defer", executor, "none", "none",
                           "variable_count_exceeds_solver_budget", variable_count, points, cells, {})
    if points > max_points or cells > max_cells:
        return BackendPlan("deferred", "defer", executor, "none", "none",
                           "contract_scale_exceeds_safe_plan", variable_count, points, cells, {})
    supervised = executor in _SUPERVISED_EXECUTORS
    family = str(getattr(specification, "solver_family", "validated_backend"))
    fidelity = "exact_contract_backend" if supervised else "validated_backend"
    route = "trusted_worker" if supervised else "trusted_backend_api"
    evaluations = _bounded_int(getattr(specification, "max_evaluations", 0), maximum=1_000_000)
    timeout = _bounded_int(getattr(specification, "timeout_seconds", 0), maximum=120)
    from .numerical_strategy import select_numerical_strategy

    numerical_strategy = select_numerical_strategy(contract.get("diagnostics", {}))
    return BackendPlan("ready", route, executor, family, fidelity,
                       f"selected_from_{form}", variable_count, points, cells,
                       {"max_evaluations": evaluations, "wall_time_seconds": timeout,
                        "max_cells": max_cells, "max_points": max_points,
                        "process_isolated": supervised,
                        "arbitrary_code_allowed": False,
                        "numerical_strategy": numerical_strategy})


__all__ = ["BackendPlan", "BackendPlanError", "plan_backend"]
