import numpy as np
import pytest

from core.cegis_controller import CEGISConfig
from core.ode_cegis import ODECEGISError, evaluate_ode_candidate, patch_ode_coefficients, run_ode_cegis


def _decay_case():
    times = np.linspace(0.0, 1.0, 11)
    return {"id": "decay", "times": times.tolist(), "initial": [1.0],
            "observations": np.exp(-0.5 * times).reshape(-1, 1).tolist()}


def test_ode_candidate_evaluator_runs_and_rejects_bad_rhs():
    case = _decay_case()
    result = evaluate_ode_candidate({"state_dim": 1, "basis": ["linear"], "coefficients": [[-0.5]]}, [case])
    assert result["status"] == "pass"
    failed = evaluate_ode_candidate({"state_dim": 1, "basis": ["linear"], "coefficients": [[0.5]]}, [case])
    assert failed["status"] == "fail"
    assert failed["violations"]


def test_ode_cegis_repairs_a_bounded_linear_decay():
    result = run_ode_cegis(
        [{"id": "seed", "state_dim": 1, "basis": ["linear"], "coefficients": [[0.0]]}],
        [_decay_case()],
        config=CEGISConfig(max_rounds=64, max_candidates=128, max_repairs=64, max_counterexamples=16),
        tolerance=0.03,
    )
    assert result["adapter_family"] == "ode"
    assert result["records"]
    assert result["policy"]["compile_before_evaluate"] is True
    assert result["accepted_candidate_hashes"]


def test_ode_cegis_can_mutate_the_basis_family():
    mutations = list(patch_ode_coefficients(
        {"state_dim": 1, "basis": ["constant"], "coefficients": [[0.0]]},
        {"step": 0.1, "counterexample_archive_count": 8},
    ))
    assert any(item["basis"] == ["constant", "linear"] for item in mutations)
    structural = next(item for item in mutations if item["basis"] == ["constant", "linear"])
    assert np.asarray(structural["coefficients"]).shape == (1, 2)


def test_ode_invalid_case_contract_uses_public_error_type():
    with pytest.raises(ODECEGISError, match="ode_cases_invalid"):
        evaluate_ode_candidate({"state_dim": 1, "basis": ["constant"], "coefficients": [[0.0]]}, [])
