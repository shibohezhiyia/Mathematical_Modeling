import numpy as np
import pytest

from core.cegis_controller import CEGISConfig
from core.ode_cegis import (
    ODECEGISError,
    compile_ode_candidate,
    evaluate_ode_candidate,
    patch_ode_coefficients,
    run_ode_cegis,
)


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


def test_ode_supports_cross_state_coupling_with_bounded_matrix_contract():
    # x' = x*y, y' = 0 has a closed form for y(0)=0.5.
    times = np.array([0.0, 0.2, 0.4])
    y = np.full(times.shape, 0.5)
    x = np.exp(0.5 * times)
    result = evaluate_ode_candidate(
        {"state_dim": 2, "basis": ["cross"], "coefficients": [[1.0], [0.0]]},
        [{"id": "coupled", "times": times, "initial": [1.0, 0.5],
          "observations": np.column_stack([x, y])}],
        tolerance=2e-4,
    )
    assert result["status"] == "pass"


def test_ode_supports_explicit_time_drive_and_mixed_basis():
    # x' = t, x(0)=0 -> x=t^2/2. Mixing constant and time exercises the
    # broadcasted feature-row contract used by structural mutations.
    times = np.array([0.0, 0.5, 1.0, 1.5])
    expected = (times ** 2 / 2.0).reshape(-1, 1)
    result = evaluate_ode_candidate(
        {"state_dim": 1, "basis": ["constant", "time"], "coefficients": [[0.0, 1.0]]},
        [{"id": "forced", "times": times, "initial": [0.0], "observations": expected}],
        tolerance=2e-4,
    )
    assert result["status"] == "pass"


def test_ode_supports_bounded_external_driver_series():
    # The driver is supplied on the same time grid and linearly interpolated
    # inside the solver; x' = u(t) with u=[1,2,3] gives x=[0,1.5,4].
    times = np.array([0.0, 1.0, 2.0])
    result = evaluate_ode_candidate(
        {"state_dim": 1, "drivers": ["u"], "basis": ["driver:u"],
         "coefficients": [[1.0]]},
        [{"id": "driven", "times": times, "initial": [0.0],
          "drivers": {"u": [1.0, 2.0, 3.0]},
          "observations": [[0.0], [1.5], [4.0]]}],
        tolerance=2e-4,
    )
    assert result["status"] == "pass"


def test_ode_driver_data_is_required_when_candidate_declares_a_driver():
    result = evaluate_ode_candidate(
        {"state_dim": 1, "drivers": ["u"], "basis": ["driver:u"],
         "coefficients": [[1.0]]},
        [{"times": [0.0, 1.0, 2.0], "initial": [0.0],
          "observations": [[0.0], [1.0], [2.0]]}],
    )
    assert result["status"] == "not_assessed"
    assert result["failure_code"] == "ode_driver_data_missing"


def test_ode_supports_bounded_threshold_event_and_state_reset():
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    result = evaluate_ode_candidate(
        {"state_dim": 1, "basis": ["constant"], "coefficients": [[-1.0]],
         "events": [{"state_index": 0, "threshold": 0.5,
                     "direction": -1, "reset_delta": 1.0}]},
        [{"id": "reset", "times": times, "initial": [1.0],
          "observations": [[1.0], [0.75], [1.5], [1.25], [1.0]]}],
        tolerance=2e-4,
    )
    assert result["status"] == "pass"
    assert result["event_count"] == 1
    assert result["events"][0]["state_index"] == 0


def test_ode_event_reset_must_be_nonzero():
    with pytest.raises(ODECEGISError, match="event_reset"):
        compile_ode_candidate({
            "state_dim": 1, "basis": ["constant"], "coefficients": [[-1.0]],
            "events": [{"state_index": 0, "threshold": 0.5,
                        "direction": -1, "reset_delta": 0.0}],
        })


def test_ode_supports_bounded_method_of_steps_delay():
    # x'=-x(t-1), constant history x(t)=1 before t=0; explicit bounded Euler
    # gives [1, 0, -1, -1] on a unit grid.
    result = evaluate_ode_candidate(
        {"state_dim": 1, "delays": [{"state_index": 0, "tau": 1.0}],
         "basis": ["delay:0"], "coefficients": [[-1.0]]},
        [{"id": "delay", "times": [0.0, 1.0, 2.0, 3.0], "initial": [1.0],
          "history": [1.0], "observations": [[1.0], [0.0], [-1.0], [-1.0]]}],
        tolerance=1e-8,
    )
    assert result["status"] == "pass"


def test_ode_delay_event_layer_handles_sampled_reset_network():
    result = evaluate_ode_candidate(
        {"state_dim": 1, "delays": [{"state_index": 0, "tau": 1.0}],
         "basis": ["delay:0"], "coefficients": [[-1.0]],
         "events": [{"state_index": 0, "threshold": 0.5, "direction": -1,
                     "reset_delta": 1.0, "max_occurrences": 1}]},
        [{"id": "delay_event", "times": [0.0, 1.0, 2.0, 3.0], "initial": [1.0],
          "history": [1.0], "observations": [[1.0], [1.0], [0.0], [-1.0]]}],
        tolerance=1e-8,
    )
    assert result["status"] == "pass"
    assert result["integration_evidence"][0]["event_count"] == 1
    assert result["events"][0]["detection"] == "sampled_method_of_steps"


def test_ode_supports_stiff_solver_selection():
    times = np.array([0.0, 0.01, 0.02, 0.03])
    result = evaluate_ode_candidate(
        {"state_dim": 1, "solver": "BDF", "basis": ["linear"], "coefficients": [[-100.0]]},
        [{"times": times, "initial": [1.0], "observations": np.exp(-100.0 * times).reshape(-1, 1)}],
        tolerance=2e-3,
    )
    assert result["status"] == "pass"


def test_ode_auto_solver_records_local_stiffness_routing():
    times = np.linspace(0.0, 0.02, 6)
    result = evaluate_ode_candidate(
        {"state_dim": 1, "solver": "auto", "basis": ["linear"], "coefficients": [[-1000.0]]},
        [{"times": times, "initial": [1.0], "observations": np.exp(-1000.0 * times).reshape(-1, 1).tolist()}],
        tolerance=2e-3,
    )
    assert result["status"] == "pass"
    evidence = result["integration_evidence"][0]
    assert evidence["solver_requested"] == "auto"
    assert evidence["solver"] in {"Radau", "RK45"}
    assert evidence["stiffness"]["status"] in {"estimated", "not_assessed"}


def test_ode_unit_contract_propagates_and_rejects_wrong_coefficient_units():
    candidate = {
        "state_dim": 1, "basis": ["linear"], "coefficients": [[-0.5]],
        "units": {"states": [{"L": 1}], "time": {"T": 1},
                  "coefficients": [[{"L": 0, "T": -1}]]},
    }
    compiled = compile_ode_candidate(candidate)
    assert compiled["units"]["status"] == "checked"
    wrong = dict(candidate)
    wrong["units"] = {**candidate["units"], "coefficients": [[{"L": 1}]]}
    with pytest.raises(ODECEGISError, match="coefficient_units_mismatch"):
        compile_ode_candidate(wrong)


def test_ode_unit_contract_can_auto_propagate_coefficient_dimensions():
    compiled = compile_ode_candidate({
        "state_dim": 1, "basis": ["linear"], "coefficients": [[-0.5]],
        "units": {"states": [{"L": 1}], "time": {"T": 1}},
    })
    assert compiled["units"]["auto_propagated"] is True
    assert compiled["units"]["coefficient_units"] == [[{"T": -1.0}]]


def test_ode_event_can_reset_multiple_states_as_one_atomic_transition():
    compiled = compile_ode_candidate({
        "state_dim": 2, "basis": ["constant"], "coefficients": [[-1.0], [0.0]],
        "events": [{"state_index": 0, "threshold": 0.5, "direction": -1,
                    "reset": [1.0, 2.0]}],
    })
    result = evaluate_ode_candidate(
        compiled,
        [{"times": [0.0, 0.25, 0.5, 0.75, 1.0], "initial": [1.0, 0.0],
          "observations": [[1.0, 0.0], [0.75, 0.0], [1.5, 2.0], [1.25, 2.0], [1.0, 2.0]]}],
        tolerance=2e-4,
    )
    assert result["status"] == "pass"
    assert result["events"][0]["reset"] == [1.0, 2.0]


def test_ode_event_network_respects_priority_and_occurrence_budget():
    # Two threshold transitions are evaluated in one bounded event network;
    # each rule is allowed once and carries its priority into the audit log.
    compiled = compile_ode_candidate({
        "state_dim": 1, "basis": ["constant"], "coefficients": [[-1.0]],
        "events": [
            {"state_index": 0, "threshold": 0.5, "direction": -1,
             "reset_delta": 1.0, "priority": 2, "max_occurrences": 1},
            {"state_index": 0, "threshold": 0.25, "direction": -1,
             "reset_delta": 1.0, "priority": 1, "max_occurrences": 1, "requires": [0]},
        ],
    })
    result = evaluate_ode_candidate(
        compiled,
        [{"times": [0.0, 0.5, 1.0, 1.5, 2.0], "initial": [1.0],
          "observations": [[1.0], [1.5], [1.0], [0.5], [1.0]]}],
        tolerance=2e-4,
    )
    assert result["status"] == "pass"
    assert result["event_count"] == 2
    assert [item["priority"] for item in result["events"]] == [2, 1]


def test_cross_basis_is_rejected_for_scalar_ode():
    with pytest.raises(ODECEGISError, match="cross_basis"):
        compile_ode_candidate({"state_dim": 1, "basis": ["cross"], "coefficients": [[1.0]]})


def test_ode_invalid_case_contract_uses_public_error_type():
    with pytest.raises(ODECEGISError, match="ode_cases_invalid"):
        evaluate_ode_candidate({"state_dim": 1, "basis": ["constant"], "coefficients": [[0.0]]}, [])
