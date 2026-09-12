"""Small, deterministic mathematical expression language.

This module is a *language restriction*, not an operating-system sandbox.  It
accepts one scalar expression, validates every AST node, and interprets the
validated tree without calling ``eval`` or executing user supplied objects.
Generated code must still run through :mod:`core.solver_runtime` when it needs
resource or process isolation.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Iterable, Mapping


class SafeExpressionError(ValueError):
    """Stable validation error that does not include untrusted source text."""


class SafeNumericExpression(ast.NodeVisitor):
    """Compile and evaluate a bounded scalar expression tree.

    The grammar intentionally excludes attributes, subscripts, comprehensions,
    comparisons, assignments, strings, containers and keyword arguments.  A
    caller supplies the complete symbol set at compile time and plain Python
    real scalars at evaluation time.
    """

    _functions = {
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "tanh": math.tanh,
    }
    max_text_length = 4096
    max_nodes = 256
    max_depth = 32

    def __init__(self, symbols: Iterable[str]) -> None:
        normalized = set()
        for symbol in symbols:
            if type(symbol) is not str or not symbol.isidentifier():
                raise SafeExpressionError("invalid_symbol_set")
            normalized.add(symbol)
        self.symbols = normalized

    def compile(self, expression: str) -> ast.AST:
        if type(expression) is not str or not expression or len(expression) > self.max_text_length:
            raise SafeExpressionError("expression_text_limit")
        try:
            tree = ast.parse(expression, mode="eval")
        except (SyntaxError, RecursionError, MemoryError) as exc:
            raise SafeExpressionError("expression_parse_error") from exc
        pending = [(tree, 0)]
        count = 0
        while pending:
            current, depth = pending.pop()
            count += 1
            if count > self.max_nodes or depth > self.max_depth:
                raise SafeExpressionError("expression_ast_limit")
            pending.extend((child, depth + 1) for child in ast.iter_child_nodes(current))
        self.visit(tree)
        return tree.body

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.symbols and node.id not in self._functions:
            raise SafeExpressionError("unknown_symbol")

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise SafeExpressionError("numeric_constants_only")
        try:
            finite = math.isfinite(float(node.value))
        except (OverflowError, ValueError) as exc:
            raise SafeExpressionError("nonfinite_constant") from exc
        if not finite:
            raise SafeExpressionError("nonfinite_constant")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise SafeExpressionError("unsupported_unary_operator")
        self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise SafeExpressionError("unsupported_binary_operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in self._functions:
            raise SafeExpressionError("function_not_allowlisted")
        if node.keywords:
            raise SafeExpressionError("keyword_arguments_forbidden")
        allowed_counts = (
            range(2, 17) if node.func.id in {"min", "max"} else
            ((1, 2) if node.func.id == "log" else (1,))
        )
        if len(node.args) not in allowed_counts:
            raise SafeExpressionError("function_arity_invalid")
        for argument in node.args:
            self.visit(argument)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Expression, ast.Load)):
            super().generic_visit(node)
            return
        raise SafeExpressionError("syntax_node_not_allowlisted")

    @classmethod
    def evaluate(cls, node: ast.AST, values: Mapping[str, Any]) -> float:
        value = cls._evaluate(node, values)
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise FloatingPointError("expression_result_nonfinite")
        return float(value)

    @classmethod
    def _evaluate(cls, node: ast.AST, values: Mapping[str, Any]) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in values or type(values[node.id]) not in (int, float):
                raise SafeExpressionError("numeric bindings must be plain real scalars")
            value = float(values[node.id])
            if not math.isfinite(value):
                raise FloatingPointError("numeric_binding_nonfinite")
            return value
        if isinstance(node, ast.UnaryOp):
            value = cls.evaluate(node.operand, values)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = cls.evaluate(node.left, values), cls.evaluate(node.right, values)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                try:
                    return left / right
                except ZeroDivisionError as exc:
                    raise FloatingPointError("division_by_zero") from exc
            try:
                return math.pow(left, right)
            except (ValueError, OverflowError) as exc:
                raise FloatingPointError("power_outside_real_finite_domain") from exc
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = cls._functions[node.func.id]
            try:
                result = function(*(cls.evaluate(argument, values) for argument in node.args))
            except (ValueError, OverflowError, ZeroDivisionError) as exc:
                raise FloatingPointError("function_outside_real_finite_domain") from exc
            if type(result) not in (int, float) or not math.isfinite(float(result)):
                raise FloatingPointError("function_result_nonfinite")
            return float(result)
        raise SafeExpressionError("syntax_node_not_allowlisted")


__all__ = ["SafeExpressionError", "SafeNumericExpression"]
