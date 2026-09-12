"""Dependency-neutral discovery for optional mathematical backends."""
from __future__ import annotations

import importlib.util
from typing import Iterable


class OptionalBackendError(ValueError):
    pass


_MODULES = {"scipy": "scipy", "sympy": "sympy", "cvxpy": "cvxpy", "pyomo": "pyomo", "jax": "jax"}


def discover_optional_backends(names: Iterable[str] | None = None) -> dict[str, dict[str, object]]:
    requested = list(_MODULES) if names is None else list(names)
    if (len(requested) > len(_MODULES) or any(type(name) is not str for name in requested) or
            len(requested) != len(set(requested)) or any(name not in _MODULES for name in requested)):
        raise OptionalBackendError("unknown_optional_backend")
    result = {}
    for name in requested:
        module = _MODULES[name]
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            # Broken import metadata should not abort a research run; it is a
            # discovery miss, never permission to execute the backend.
            available = False
        result[name] = {"module": module, "available": bool(available),
                        "status": "available" if available else "unavailable",
                        "execution_authorized": False,
                        "policy": "discovery_only_until_backend_contract_and_resource_gate"}
    return {"schema_version": "mathmodel.optional-backends/v1", "backends": result,
            "available_count": sum(item["available"] for item in result.values())}


__all__ = ["OptionalBackendError", "discover_optional_backends"]
