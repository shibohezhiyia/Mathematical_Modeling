import pytest

from core.numerical_stability import (
    NumericalStabilityError,
    assess_numerical_error_budget,
    assess_solver_tolerance_stability,
)


def test_numerical_error_budget_records_declared_solver_metadata():
    result = assess_numerical_error_budget([
        {"tolerance": 1e-3, "discretization_error": 0.02,
         "solver_error_bound": 0.01, "condition_number": 10, "unit_signature": "m"},
        {"tolerance": 1e-5, "discretization_error": 0.005,
         "solver_error_bound": 0.002, "condition_number": 20, "unit_signature": "m"},
    ])
    assert result["status"] == "assessed"
    assert result["declared_additive_bound"] == pytest.approx(0.03)
    assert result["condition_number_max"] == 20


def test_numerical_error_budget_rejects_mixed_units_and_negative_bounds():
    with pytest.raises(NumericalStabilityError, match="units_must_match"):
        assess_numerical_error_budget([
            {"tolerance": 1e-3, "unit_signature": "m"},
            {"tolerance": 1e-4, "unit_signature": "s"},
        ])
    with pytest.raises(NumericalStabilityError, match="nonnegative"):
        assess_numerical_error_budget([{"tolerance": 1e-3, "solver_error_bound": -1}])


def test_solver_outputs_stable_across_tolerances():
    result = assess_solver_tolerance_stability(lambda tol: {"objective": 2.0 + tol},
                                               tolerances=(1e-2, 1e-4, 1e-6),
                                               absolute_tolerance=2e-2)
    assert result["status"] == "stable_on_tested_tolerances"
    assert result["successful_runs"] == 3


def test_solver_tolerance_change_is_recorded_as_numerical_counterexample():
    result = assess_solver_tolerance_stability(lambda tol: {"objective": 1.0 if tol > 1e-5 else 2.0},
                                               tolerances=(1e-2, 1e-6), absolute_tolerance=1e-4)
    assert result["status"] == "unstable"
    assert result["comparison"]["counterexamples"]
    assert result["policy"].startswith("finite_tolerance")


def test_solver_failure_is_not_relabelled_as_instability():
    result = assess_solver_tolerance_stability(lambda tol: (_ for _ in ()).throw(RuntimeError()),
                                               tolerances=(1e-3, 1e-5), max_failures=2)
    assert result["status"] == "not_assessed"
    assert result["failed_runs"] == 2


def test_tolerance_uncertainty_direction_is_checked_separately():
    result = assess_solver_tolerance_stability(
        lambda tol: {"objective": 1.0, "numerical_uncertainty": tol * 2},
        tolerances=(1e-2, 1e-4, 1e-6), uncertainty_field="numerical_uncertainty",
    )
    assert result["comparison"]["uncertainty_direction"]["status"] == "pass"

    failing = assess_solver_tolerance_stability(
        lambda tol: {"objective": 1.0, "numerical_uncertainty": 1.0 / tol},
        tolerances=(1e-2, 1e-4), uncertainty_field="numerical_uncertainty",
    )
    assert failing["comparison"]["uncertainty_direction"]["status"] == "fail"


@pytest.mark.parametrize("kwargs", [
    {"tolerances": (1e-3,)}, {"tolerances": (1e-3, 1e-3)},
    {"absolute_tolerance": -1}, {"max_failures": 4},
    {"uncertainty_field": ""},
])
def test_solver_tolerance_contract_is_strict(kwargs):
    with pytest.raises(NumericalStabilityError):
        assess_solver_tolerance_stability(lambda tol: {"x": 1}, **kwargs)
