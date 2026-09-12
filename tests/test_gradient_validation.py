import pytest

from core.gradient_validation import GradientCheckError, check_gradient


def test_gradient_cross_check_passes_for_quadratic_objective():
    result = check_gradient(
        lambda point: point["x"] ** 2 + 3 * point["y"],
        lambda point: {"x": 2 * point["x"], "y": 3},
        {"x": 1.5, "y": 2}, bounds={"x": [-2, 2], "y": [0, 4]},
    )
    assert result["status"] == "pass"
    assert {row["method"] for row in result["checks"]} == {"central"}
    assert result["policy"].endswith("global_gradient_proof")


def test_gradient_cross_check_detects_wrong_gradient_and_uses_boundary_one_sided_step():
    result = check_gradient(lambda point: point["x"] ** 2 + point["x"],
                            lambda point: {"x": 0}, {"x": 0}, bounds={"x": [0, 2]})
    assert result["status"] == "fail"
    assert result["checks"][0]["method"] == "forward"


def test_gradient_check_rejects_bad_contract_and_budget():
    with pytest.raises(GradientCheckError, match="point_outside_bound"):
        check_gradient(lambda point: 1, lambda point: {"x": 0}, {"x": 2}, bounds={"x": [0, 1]})
    result = check_gradient(lambda point: point["x"] ** 2, lambda point: {"x": 2 * point["x"]},
                            {"x": 1}, max_evaluations=1)
    assert result["status"] == "not_assessed"
