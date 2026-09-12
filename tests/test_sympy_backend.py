import pytest

from core.sympy_backend import SymPyBackendError, compile_sympy_expression


def test_sympy_backend_compiles_only_restricted_ast():
    pytest.importorskip("sympy")
    result = compile_sympy_expression("sqrt(x*x + y*y)", ["x", "y"])
    assert result["status"] == "compiled"
    assert result["free_symbols"] == ["x", "y"]
    with pytest.raises(SymPyBackendError):
        compile_sympy_expression("__import__('os')", ["x"])
