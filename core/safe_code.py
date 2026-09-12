"""AST allow-list for tiny generated numeric programs.

This is a language validator, not an operating-system sandbox.  It accepts a
small sequence of numeric assignments and expressions, rejects imports,
attributes, indexing, loops and comprehensions, and returns an auditable AST
digest.  Execution must still happen in the supervised solver worker.
"""

from __future__ import annotations

import ast
from hashlib import sha256
import json
import math
from typing import Any, Iterable


class SafeCodeError(ValueError):
    pass


_FUNCTIONS = {"abs", "min", "max", "sqrt", "exp", "log", "sin", "cos", "tan", "tanh"}
_BINARY = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
_UNARY = (ast.UAdd, ast.USub)


def _names(values: Iterable[str], code: str) -> set[str]:
    result = set()
    for value in values:
        if type(value) is not str or not value.isidentifier() or len(value) > 64:
            raise SafeCodeError(code)
        result.add(value)
    return result


def validate_generated_code(
    source: str, *, allowed_variables: Iterable[str] = (), allowed_outputs: Iterable[str] = (),
    max_text_length: int = 32_768, max_nodes: int = 1024, max_depth: int = 64,
) -> dict[str, Any]:
    """Validate source without executing it or resolving imports."""
    if type(source) is not str or not source.strip() or len(source) > max_text_length:
        raise SafeCodeError("source_text_limit")
    for name, value, upper in (("max_text_length", max_text_length, 1_000_000),
                               ("max_nodes", max_nodes, 100_000), ("max_depth", max_depth, 512)):
        if type(value) is not int or not 1 <= value <= upper:
            raise SafeCodeError(f"invalid_{name}")
    variables = _names(allowed_variables, "invalid_allowed_variables")
    outputs = _names(allowed_outputs, "invalid_allowed_outputs")
    try:
        tree = ast.parse(source, mode="exec")
    except (SyntaxError, RecursionError, MemoryError) as exc:
        raise SafeCodeError("source_parse_error") from exc
    assigned: set[str] = set()
    count = 0

    def visit(node: ast.AST, depth: int = 0) -> None:
        nonlocal count
        count += 1
        if count > max_nodes or depth > max_depth:
            raise SafeCodeError("ast_budget_exceeded")
        if isinstance(node, (ast.Module, ast.Expr, ast.Load, ast.Store)):
            pass
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise SafeCodeError("assignment_target_not_allowlisted")
            target = node.targets[0].id
            if target not in outputs:
                raise SafeCodeError("assignment_output_not_allowlisted")
            assigned.add(target)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                if node.id not in outputs:
                    raise SafeCodeError("assignment_output_not_allowlisted")
            elif node.id not in variables and node.id not in _FUNCTIONS and node.id not in assigned:
                raise SafeCodeError("unknown_identifier")
        elif isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
                raise SafeCodeError("numeric_constants_only")
            if not math.isfinite(float(node.value)):
                raise SafeCodeError("nonfinite_constant")
        elif isinstance(node, ast.BinOp):
            if not isinstance(node.op, _BINARY):
                raise SafeCodeError("binary_operator_not_allowlisted")
        elif isinstance(node, ast.UnaryOp):
            if not isinstance(node.op, _UNARY):
                raise SafeCodeError("unary_operator_not_allowlisted")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS or node.keywords:
                raise SafeCodeError("function_not_allowlisted")
            if node.func.id in {"min", "max"} and not 2 <= len(node.args) <= 16:
                raise SafeCodeError("function_arity_invalid")
            if node.func.id not in {"min", "max"} and len(node.args) not in {1, 2}:
                raise SafeCodeError("function_arity_invalid")
        elif isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.UAdd, ast.USub)):
            pass
        else:
            raise SafeCodeError("syntax_node_not_allowlisted")
        for child in ast.iter_child_nodes(node):
            visit(child, depth + 1)

    visit(tree)
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return {"schema_version": "mathmodel.safe-code/v1", "status": "validated",
            "assigned_outputs": sorted(assigned), "ast_nodes": count,
            "source_digest": sha256(canonical.encode("utf-8")).hexdigest(),
            "execution_authorized": False,
            "policy": "AST_allowlist_only;_run_in_supervised_worker_with_resource_limits"}


__all__ = ["SafeCodeError", "validate_generated_code"]
