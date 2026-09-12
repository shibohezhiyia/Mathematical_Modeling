"""Tiny typed differentiable expression IR with optional JAX lowering.

Only a closed arithmetic graph is accepted. No Python callback, attribute access
or generated source can enter this evaluator. When JAX is unavailable, callers
may request the bounded central-difference fallback and receive an explicit
``finite_difference`` backend label.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


class DifferentiableIRError(ValueError):
    pass


_OPS = frozenset({"constant", "variable", "add", "sub", "mul", "div", "pow", "sin", "exp"})


def _walk(node: Any, names: set[str], *, depth: int = 0, budget: list[int] | None = None) -> None:
    if budget is None:
        budget = [0]
    if depth > 32:
        raise DifferentiableIRError("ir_depth_exceeded")
    if not isinstance(node, Mapping) or not isinstance(node.get("op"), str) or node["op"] not in _OPS:
        raise DifferentiableIRError("invalid_ir_node")
    budget[0] += 1
    if budget[0] > 256:
        raise DifferentiableIRError("ir_node_budget_exceeded")
    op = node["op"]
    if op == "constant":
        value = node.get("value")
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise DifferentiableIRError("constant_must_be_finite")
        return
    if op == "variable":
        name = node.get("name")
        if type(name) is not str or not name.strip() or len(name) > 64 or any(c.isspace() for c in name):
            raise DifferentiableIRError("invalid_variable_name")
        names.add(name)
        return
    args = node.get("args")
    expected = 1 if op in {"sin", "exp"} else 2
    if not isinstance(args, list) or len(args) != expected:
        raise DifferentiableIRError("invalid_ir_arity")
    for child in args:
        _walk(child, names, depth=depth + 1, budget=budget)


def validate_differentiable_ir(node: Mapping[str, Any], variables: Sequence[str] | None = None) -> dict[str, Any]:
    names: set[str] = set()
    _walk(node, names)
    if variables is not None:
        declared = tuple(variables)
        if any(type(item) is not str or not item.strip() for item in declared) or len(set(declared)) != len(declared):
            raise DifferentiableIRError("invalid_declared_variables")
        if names - set(declared):
            raise DifferentiableIRError("undeclared_variable")
    return {"schema_version": "mathmodel.differentiable-ir/v1", "variables": sorted(names),
            "node_count": _count_nodes(node), "backend_authority": "expression_only"}


def _count_nodes(node: Mapping[str, Any]) -> int:
    args = node.get("args", [])
    return 1 + sum(_count_nodes(item) for item in args)


def _evaluate(node: Mapping[str, Any], values: Mapping[str, Any], xp: Any) -> Any:
    op = node["op"]
    if op == "constant":
        return xp.asarray(node["value"])
    if op == "variable":
        return values[node["name"]]
    args = [_evaluate(child, values, xp) for child in node["args"]]
    if op == "add": return args[0] + args[1]
    if op == "sub": return args[0] - args[1]
    if op == "mul": return args[0] * args[1]
    if op == "div": return args[0] / args[1]
    if op == "pow": return args[0] ** args[1]
    if op == "sin": return xp.sin(args[0])
    if op == "exp": return xp.exp(args[0])
    raise DifferentiableIRError("unsupported_ir_operation")


def evaluate_differentiable_ir(
    node: Mapping[str, Any], values: Mapping[str, float], *, backend: str = "auto",
    finite_difference_step: float = 1e-6,
) -> dict[str, Any]:
    contract = validate_differentiable_ir(node, variables=list(values))
    if type(finite_difference_step) not in (int, float) or not math.isfinite(float(finite_difference_step)) or finite_difference_step <= 0:
        raise DifferentiableIRError("invalid_finite_difference_step")
    if set(values) != set(contract["variables"]):
        raise DifferentiableIRError("value_variables_mismatch")
    if any(type(value) not in (int, float) or not math.isfinite(float(value)) for value in values.values()):
        raise DifferentiableIRError("values_must_be_finite")
    if backend not in {"auto", "jax", "finite_difference"}:
        raise DifferentiableIRError("invalid_backend")
    if backend in {"auto", "jax"}:
        try:
            import jax
            import jax.numpy as jnp
            names = contract["variables"]
            vector = jnp.asarray([values[name] for name in names], dtype=float)
            def scalar(argument):
                return _evaluate(node, dict(zip(names, argument)), jnp)
            value = float(scalar(vector))
            gradient = np.asarray(jax.grad(scalar)(vector), dtype=float)
            if not math.isfinite(value) or not np.isfinite(gradient).all():
                raise DifferentiableIRError("jax_nonfinite_result")
            return {**contract, "status": "executed", "backend": "jax", "value": value,
                    "gradient": gradient.tolist(), "policy": "typed_ir_only"}
        except ImportError:
            if backend == "jax":
                return {**contract, "status": "unavailable", "backend": "jax",
                        "reason": "jax_not_installed", "policy": "optional_backend_not_authorized"}
    names = contract["variables"]
    base = {name: float(values[name]) for name in names}
    value = float(_evaluate(node, base, np))
    gradient = []
    for name in names:
        plus, minus = dict(base), dict(base)
        plus[name] += float(finite_difference_step)
        minus[name] -= float(finite_difference_step)
        gradient.append((float(_evaluate(node, plus, np)) - float(_evaluate(node, minus, np))) /
                        (2.0 * float(finite_difference_step)))
    if not math.isfinite(value) or not np.isfinite(gradient).all():
        raise DifferentiableIRError("finite_difference_nonfinite_result")
    return {**contract, "status": "executed", "backend": "finite_difference", "value": value,
            "gradient": gradient, "policy": "typed_ir_only"}


__all__ = ["DifferentiableIRError", "validate_differentiable_ir", "evaluate_differentiable_ir"]
