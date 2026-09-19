"""Single source of truth for supervised execution capabilities."""
from __future__ import annotations

from typing import Any

CAPABILITIES: dict[str, dict[str, Any]] = {
    "problem_statement": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "primitive_graph": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "ode_cegis": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "optimization_cegis": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "multitable_cegis": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "dynamic_competition": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "ode": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "optimization": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "multi_table": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "external_method": {"family": "dynamic", "max_wall_seconds": 600, "isolated": True},
    "research": {"family": "research", "max_wall_seconds": 1800, "isolated": True},
    "training": {"family": "training", "max_wall_seconds": 1800, "isolated": True},
}


def get_capabilities() -> dict[str, dict[str, Any]]:
    return {key: dict(value) for key, value in CAPABILITIES.items()}


def validate_budget(kind: str, wall_seconds: float) -> float:
    if kind not in CAPABILITIES:
        raise ValueError("unsupported_execution_backend")
    if type(wall_seconds) not in (int, float) or wall_seconds <= 0:
        raise ValueError("execution_wall_limit_invalid")
    limit = CAPABILITIES[kind]["max_wall_seconds"]
    if float(wall_seconds) > limit:
        raise ValueError("execution_wall_limit_exceeded")
    return float(wall_seconds)


__all__ = ["CAPABILITIES", "get_capabilities", "validate_budget"]
