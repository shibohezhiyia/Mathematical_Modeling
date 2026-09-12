import pytest

from core.safe_code import SafeCodeError, validate_generated_code


def test_safe_code_validates_numeric_assignments_without_authorizing_execution():
    result = validate_generated_code("y = exp(x) + 1", allowed_variables=["x"], allowed_outputs=["y"])
    assert result["status"] == "validated"
    assert result["assigned_outputs"] == ["y"]
    assert result["execution_authorized"] is False


@pytest.mark.parametrize("source", [
    "import os",
    "y = x.__class__",
    "y = data[0]",
    "for x in range(3):\n y = x",
    "y = __import__('os')",
])
def test_safe_code_rejects_side_effects_and_dynamic_access(source):
    with pytest.raises(SafeCodeError):
        validate_generated_code(source, allowed_variables=["x", "data"], allowed_outputs=["y"])


def test_safe_code_rejects_unknown_output_and_nonfinite_literal():
    with pytest.raises(SafeCodeError, match="output_not_allowlisted"):
        validate_generated_code("z = x + 1", allowed_variables=["x"], allowed_outputs=["y"])
    with pytest.raises(SafeCodeError, match="nonfinite_constant"):
        validate_generated_code("y = 1e309", allowed_outputs=["y"])
