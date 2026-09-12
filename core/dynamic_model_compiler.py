"""Single guarded entry point for dynamically composed mathematical models.

The compiler accepts a typed JSON contract and dispatches to existing local
backends.  It is intentionally a dispatcher/contract checker, not an LLM
code executor: source strings, imports and filesystem paths are rejected.
"""
from __future__ import annotations

from typing import Any, Mapping


SCHEMA_VERSION = "mathmodel.dynamic-compiler/v1"
_KINDS = {"primitive_graph", "ode_cegis", "optimization_cegis", "multitable_cegis", "external_method"}


class DynamicCompilerError(ValueError):
    pass


def compile_and_execute_model(kind: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(kind, str) or kind not in _KINDS:
        raise DynamicCompilerError("dynamic_model_kind_invalid")
    if not isinstance(payload, Mapping):
        raise DynamicCompilerError("dynamic_model_payload_invalid")
    data = dict(payload)
    # Never accept generated source or path-like execution controls.
    forbidden = {"source", "code", "python", "command", "path", "module", "import"}
    if forbidden.intersection(data):
        raise DynamicCompilerError("dynamic_model_source_or_path_forbidden")
    if kind == "primitive_graph":
        from .primitive_graph_runtime import execute_primitive_graph
        result = execute_primitive_graph(data.get("nodes"), data.get("bindings"), output_ids=data.get("output_ids", []))
    elif kind == "ode_cegis":
        from .ode_cegis import run_ode_cegis
        candidate = data.get("candidate")
        candidates = data.get("candidates", [candidate])
        if not isinstance(candidates, list) or not candidates or any(not isinstance(item, Mapping) for item in candidates):
            raise DynamicCompilerError("ode_candidates_invalid")
        result = run_ode_cegis(candidates, data.get("cases"), tolerance=data.get("tolerance", 1e-2))
    elif kind == "optimization_cegis":
        from .optimization_cegis import run_optimization_family_cegis
        candidate = data.get("candidate")
        if not isinstance(candidate, Mapping):
            raise DynamicCompilerError("optimization_candidate_invalid")
        result = run_optimization_family_cegis(str(data.get("optimization_kind", candidate.get("kind", "linear_program"))),
                                               [candidate], data.get("cases"), tolerance=data.get("tolerance", 1e-7))
    elif kind == "multitable_cegis":
        from .multitable_cegis import run_multitable_cegis
        candidate = data.get("candidate")
        candidates = data.get("candidates", [candidate])
        if not isinstance(candidates, list) or not candidates or any(not isinstance(item, Mapping) for item in candidates):
            raise DynamicCompilerError("multitable_candidates_invalid")
        result = run_multitable_cegis(candidates, data.get("cases"))
    else:
        from .external_method_runtime import execute_external_method
        result = execute_external_method(str(data.get("method")), data.get("payload", {}))
    inner_status = result.get("status") if isinstance(result, Mapping) else None
    status = "executed" if inner_status in {None, "executed", "accepted", "accepted_candidates", "verified"} else str(inner_status)
    if inner_status in {"not_assessed", "candidate_set_inadequate", "budget_exhausted", "failed"}:
        status = inner_status
    return {"schema_version": SCHEMA_VERSION, "status": status, "kind": kind,
            "result": result, "policy": "typed_json_dispatch; no_generated_source_execution"}


__all__ = ["SCHEMA_VERSION", "DynamicCompilerError", "compile_and_execute_model"]
