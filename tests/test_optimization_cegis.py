import pytest

from core.cegis_controller import CEGISConfig
from core.optimization_cegis import (
    OptimizationCEGISError,
    compile_linear_program_candidate,
    evaluate_linear_program_candidate,
    run_optimization_cegis,
    compile_optimization_candidate,
    evaluate_optimization_candidate,
    run_optimization_family_cegis,
)


def _candidate():
    return {
        "id": "seed",
        "kind": "linear_program",
        "variables": ["x", "y"],
        "units": {"x": "u", "y": "u"},
        "objective_coefficients": [1.0, 1.0],
        "direction": "minimize",
        "bounds": [[0.0, 10.0], [0.0, 10.0]],
        "A_ub": [[-1.0, -1.0]], "b_ub": [-1.0],
        "A_eq": [], "b_eq": [],
    }


def test_linear_program_candidate_executes_and_replays():
    candidate = compile_linear_program_candidate(_candidate())
    result = evaluate_linear_program_candidate(candidate, [{"id": "base", "expected_objective": 1.0}])
    assert result["status"] == "pass"
    run = run_optimization_cegis([_candidate()], [{"id": "base", "expected_objective": 1.0}], config=CEGISConfig(max_rounds=4, max_candidates=8))
    assert run["adapter_family"] == "linear_program"
    assert run["accepted_candidate_hashes"]


def test_linear_program_invalid_case_is_not_silently_refuted():
    with pytest.raises(OptimizationCEGISError, match="lp_cases_invalid"):
        evaluate_linear_program_candidate(compile_linear_program_candidate(_candidate()), [])


def test_mixed_integer_and_quadratic_contracts_use_the_same_cegis_adapter():
    milp = {"id": "milp", "kind": "mixed_integer_linear_program", "variables": ["x"],
            "units": {"x": "u"}, "objective_coefficients": [1.0], "direction": "minimize",
            "bounds": [[0.0, 3.0]], "A_ub": [], "b_ub": [], "A_eq": [], "b_eq": [], "integrality": [1]}
    qp = {"id": "qp", "kind": "quadratic_program", "variables": ["x"],
          "units": {"x": "u"}, "linear_coefficients": [0.0], "quadratic_matrix": [[2.0]],
          "direction": "minimize", "bounds": [[-1.0, 1.0]], "A_ub": [], "b_ub": [], "A_eq": [], "b_eq": []}
    milp_result = evaluate_optimization_candidate(compile_optimization_candidate(milp), [{"expected_objective": 0.0}])
    qp_result = evaluate_optimization_candidate(compile_optimization_candidate(qp), [{"expected_objective": 0.0}])
    assert milp_result["status"] == "pass"
    assert qp_result["status"] == "pass"
    run = run_optimization_family_cegis("mixed_integer_linear_program", [milp], [{"expected_objective": 0.0}], config=CEGISConfig(max_rounds=2, max_candidates=4))
    assert run["adapter_family"] == "mixed_integer_linear_program"


def test_optimization_evaluator_does_not_accept_a_forged_infeasible_result(monkeypatch):
    candidate = compile_optimization_candidate(_candidate())

    def forged_execute(self, _executor, _relation):
        return {
            "objective_value": 1.0,
            "maximum_constraint_violation": 0.5,
            "convergence": {"status": "fail"},
        }

    monkeypatch.setattr("core.optimization_cegis.UniversalSolverRegistry.execute", forged_execute)
    result = evaluate_optimization_candidate(candidate, [{"id": "forged"}])
    assert result["status"] == "fail"
    assert {item["reason"] for item in result["violations"]} >= {
        "optimization_constraint_violation", "optimization_solver_certificate_failed",
    }
    assert result["solver_evidence"][0]["maximum_constraint_violation"] == 0.5
