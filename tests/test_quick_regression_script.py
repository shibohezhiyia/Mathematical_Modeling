from pathlib import Path


def test_quick_regression_script_declares_core_safety_and_math_checks():
    text = (Path(__file__).parents[1] / "scripts" / "quick_regression.py").read_text(encoding="utf-8")
    assert "test_primitive_runtime.py" in text
    assert "test_solver_runtime.py" in text
    assert "test_safe_code.py" in text
    assert "test_batch_candidate_evaluation.py" in text
    assert "test_numerical_stability.py" in text
    assert "test_parameter_uncertainty.py" in text
    assert "test_reversible_scaling.py" in text
    assert "test_input_snapshot.py" in text
    assert "test_variable_projection.py" in text
