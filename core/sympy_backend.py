"""Safe SymPy expression backend built from the restricted expression AST."""

from __future__ import annotations

import ast
from typing import Any, Iterable

from .safe_expression import SafeExpressionError, SafeNumericExpression


class SymPyBackendError(ValueError):
    pass


def compile_sympy_expression(expression: str, symbols: Iterable[str]) -> dict[str, Any]:
    """Compile a scalar expression to SymPy without ``sympify`` or eval."""
    try:
        import sympy as sp
    except (ImportError, ModuleNotFoundError):
        return {"schema_version": "mathmodel.sympy-backend/v1", "status": "unavailable", "reason": "sympy_not_installed"}
    compiler = SafeNumericExpression(symbols)
    try:
        tree = compiler.compile(expression)
    except (SafeExpressionError, SyntaxError) as exc:
        raise SymPyBackendError("safe_expression_rejected") from exc
    symbol_map = {name: sp.Symbol(name, real=True) for name in compiler.symbols}
    functions = {"abs": sp.Abs, "min": sp.Min, "max": sp.Max, "sqrt": sp.sqrt,
                 "exp": sp.exp, "log": sp.log, "sin": sp.sin, "cos": sp.cos,
                 "tan": sp.tan, "tanh": sp.tanh}

    def convert(node: ast.AST):
        if isinstance(node, ast.Constant):
            return sp.Float(node.value) if isinstance(node.value, float) else sp.Integer(node.value)
        if isinstance(node, ast.Name):
            if node.id not in symbol_map:
                raise SymPyBackendError("unknown_symbol")
            return symbol_map[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add): return left + right
            if isinstance(node.op, ast.Sub): return left - right
            if isinstance(node.op, ast.Mult): return left * right
            if isinstance(node.op, ast.Div): return left / right
            if isinstance(node.op, ast.Pow): return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
            return functions[node.func.id](*(convert(argument) for argument in node.args))
        raise SymPyBackendError("ast_conversion_rejected")

    result = convert(tree)
    return {
        "schema_version": "mathmodel.sympy-backend/v1",
        "status": "compiled",
        "expression": str(result),
        "free_symbols": sorted(str(item) for item in result.free_symbols),
        "backend": "sympy",
        "policy": "constructed_from_safe_ast; no_sympify_or_eval; symbolic_result_requires_numeric_evidence",
    }


__all__ = ["SymPyBackendError", "compile_sympy_expression"]
