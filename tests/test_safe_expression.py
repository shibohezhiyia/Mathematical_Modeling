import ast
import math

import pytest

from core.safe_expression import SafeExpressionError, SafeNumericExpression


@pytest.mark.parametrize("expression", [
    "__import__('os')", "x.__class__", "x[0]", "[x]", "'x'", "True",
    "(lambda: 1)()", "(x := 1)", "sqrt(x, x)", "exp(x, y=2)",
    "1e309", "+" * 40 + "x", "+".join(["x"] * 150),
])
def test_language_rejects_code_and_unbounded_forms(expression):
    with pytest.raises((SafeExpressionError, SyntaxError, ValueError)):
        SafeNumericExpression(["x"]).compile(expression)


def test_expression_compiles_and_evaluates_only_plain_scalars():
    compiler = SafeNumericExpression(["x", "y"])
    node = compiler.compile("sqrt(x*x + y*y) + log(1 + x)")
    assert compiler.evaluate(node, {"x": 3.0, "y": 4.0}) == pytest.approx(5 + math.log(4))


def test_expression_result_is_deterministic_and_finite():
    compiler = SafeNumericExpression(["x"])
    node = compiler.compile("exp(x)")
    assert compiler.evaluate(node, {"x": 0.0}) == 1.0
    with pytest.raises(FloatingPointError):
        compiler.evaluate(node, {"x": 1000.0})


def test_expression_division_by_zero_is_a_bounded_numeric_failure():
    compiler = SafeNumericExpression(["x"])
    node = compiler.compile("1 / x")
    with pytest.raises(FloatingPointError, match="division_by_zero"):
        compiler.evaluate(node, {"x": 0.0})


def test_expression_does_not_accept_overloaded_numeric_objects():
    class Imposter:
        def __float__(self):
            raise AssertionError("untrusted conversion ran")

    node = SafeNumericExpression(["x"]).compile("x")
    with pytest.raises(SafeExpressionError, match="plain real"):
        SafeNumericExpression.evaluate(node, {"x": Imposter()})


def test_compile_returns_expression_body_not_executable_module():
    node = SafeNumericExpression(["x"]).compile("x + 1")
    assert isinstance(node, ast.BinOp)
