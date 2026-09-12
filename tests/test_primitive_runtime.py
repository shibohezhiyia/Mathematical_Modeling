import pytest

from core.primitive_runtime import PrimitiveRuntime, PrimitiveRuntimeError


def test_domain_neutral_primitives_execute_with_residuals():
    runtime = PrimitiveRuntime()
    linear = runtime.execute("linear_system", {"matrix": [[2, 0], [0, 4]], "rhs": [4, 8]})
    assert linear["solution"] == pytest.approx([2, 2])
    integral = runtime.execute("integral", {"polynomial_coefficients": [0, 1], "lower": 0, "upper": 2})
    assert integral["integral"] == pytest.approx(2, rel=1e-10)


def test_runtime_rejects_unbounded_or_unknown_operations():
    with pytest.raises(PrimitiveRuntimeError):
        PrimitiveRuntime().execute("python", {"code": "2+2"})
    with pytest.raises(PrimitiveRuntimeError):
        PrimitiveRuntime().execute("linear_system", {"matrix": [[1]], "rhs": [1, 2]})


def test_basic_primitives_reject_nonfinite_and_empty_inputs_before_solver_call():
    runtime = PrimitiveRuntime()
    with pytest.raises(PrimitiveRuntimeError, match="linear_system_shape_limit"):
        runtime.execute("linear_system", {"matrix": [[float("nan")]], "rhs": [1]})
    with pytest.raises(PrimitiveRuntimeError, match="least_squares_shape_limit"):
        runtime.execute("least_squares", {"design": [[float("inf")]], "target": [1]})
    with pytest.raises(PrimitiveRuntimeError, match="integral_contract_invalid"):
        runtime.execute("integral", {"polynomial_coefficients": [float("nan")], "lower": 0, "upper": 1})
    with pytest.raises(PrimitiveRuntimeError, match="quadratic_program_iteration_limit"):
        runtime.execute("quadratic_program", {"quadratic_matrix": [[1.0]], "linear": [0.0],
                                                "bounds": [[0, 1]], "max_iterations": "10"})


def test_linear_ode_and_threshold_event_are_bounded_and_reproducible():
    runtime = PrimitiveRuntime()
    ode = runtime.execute(
        "linear_ode",
        {"matrix": [[-1.0]], "initial": [1.0], "times": [0.0, 1.0, 2.0]},
    )
    assert ode["method"] == "matrix_exponential"
    assert ode["states"][0] == pytest.approx([1.0])
    assert ode["states"][-1][0] == pytest.approx(0.135335, rel=1e-5)

    event = runtime.execute(
        "threshold_event",
        {"times": [0.0, 1.0, 2.0], "values": [0.0, 2.0, 0.0], "threshold": 1.0, "direction": "rise"},
    )
    assert event["crossing_found"] is True
    assert event["event_time"] == pytest.approx(0.5)


def test_ode_and_event_reject_nonfinite_or_nonmonotone_inputs():
    runtime = PrimitiveRuntime()
    with pytest.raises(PrimitiveRuntimeError):
        runtime.execute("linear_ode", {"matrix": [[1.0]], "initial": [1.0], "times": [0.0, 0.0]})
    with pytest.raises(PrimitiveRuntimeError):
        runtime.execute("threshold_event", {"times": [0.0, 1.0], "values": [0.0, float("nan")], "threshold": 1.0})


def test_quadratic_program_returns_feasibility_evidence_without_callbacks():
    result = PrimitiveRuntime().execute(
        "quadratic_program",
        {
            "quadratic_matrix": [[2.0]],
            "linear": [-4.0],
            "inequality_matrix": [[-1.0]],
            "inequality_rhs": [0.0],
            "bounds": [[0.0, 10.0]],
        },
    )
    assert result["status"] == "executed"
    assert result["solution"][0] == pytest.approx(2.0, abs=1e-5)
    assert result["maximum_constraint_violation"] <= 1e-6


def test_quadratic_program_does_not_silently_accept_nonconvex_objectives():
    result = PrimitiveRuntime().execute(
        "quadratic_program",
        {"quadratic_matrix": [[-1.0]], "linear": [0.0], "bounds": [[-1.0, 1.0]]},
    )
    assert result["status"] == "not_assessed"
    assert result["reason"] == "nonconvex_quadratic_objective"


def test_quadratic_program_rejects_empty_or_impossible_bound_intervals():
    runtime = PrimitiveRuntime()
    with pytest.raises(PrimitiveRuntimeError):
        runtime.execute("quadratic_program", {
            "quadratic_matrix": [], "linear": [], "bounds": [],
        })
    with pytest.raises(PrimitiveRuntimeError):
        runtime.execute("quadratic_program", {
            "quadratic_matrix": [[1.0]], "linear": [0.0],
            "bounds": [[float("inf"), float("inf")]],
        })
    with pytest.raises(PrimitiveRuntimeError):
        runtime.execute("quadratic_program", {
            "quadratic_matrix": [[1.0]], "linear": [0.0],
            "bounds": [[float("-inf"), float("-inf")]],
        })


def test_quadratic_program_reports_infeasible_constraints_without_a_fake_solution():
    result = PrimitiveRuntime().execute(
        "quadratic_program",
        {
            "quadratic_matrix": [[2.0]], "linear": [-4.0],
            "inequality_matrix": [[1.0], [-1.0]],
            "inequality_rhs": [0.0, -1.0],
            "bounds": [[-10.0, 10.0]],
        },
    )
    assert result["status"] == "not_converged"
    assert result["maximum_constraint_violation"] is not None


def test_geometry_and_event_primitives_cover_boundaries_and_overlap():
    runtime = PrimitiveRuntime()
    assert runtime.execute("distance", {"left": [0, 0], "right": [3, 4]})["distance"] == pytest.approx(5)
    assert runtime.execute("distance", {"left": [0, 0], "right": [3, 4], "metric": "manhattan"})["distance"] == pytest.approx(7)

    union = runtime.execute("interval_union", {"intervals": [[0, 1], [0.5, 2], [3, 3], [2, 3.5]]})
    assert union["intervals"] == [[0.0, 3.5]]
    assert union["measure"] == pytest.approx(3.5)

    assert runtime.execute("region_membership", {
        "point": [1, 2], "region": {"lower": [0, 0], "upper": [1, 2]},
    })["inside"] is True
    assert runtime.execute("region_membership", {
        "point": [0.5, 0.5], "region": [[0, 0], [1, 0], [1, 1], [0, 1]],
    })["inside"] is True
    crossing = runtime.execute("segment_intersection", {
        "first": [[0, 0], [2, 2]], "second": [[0, 2], [2, 0]],
    })
    assert crossing["intersects"] is True
    assert crossing["intersection_point"] == pytest.approx([1, 1])
    visibility = runtime.execute("line_of_sight", {
        "source": [0, 0], "target": [3, 0],
        "obstacles": [[[1, -1], [2, -1], [2, 1], [1, 1]]],
    })
    assert visibility["visible"] is False
    assert visibility["blocked_by"] == [0]
    assert runtime.execute("line_of_sight", {
        "source": [0, 0], "target": [3, 0], "obstacles": [],
    })["visible"] is True


def test_geometry_primitives_reject_nonfinite_and_malformed_inputs():
    runtime = PrimitiveRuntime()
    with pytest.raises(PrimitiveRuntimeError, match="distance_input_invalid"):
        runtime.execute("distance", {"left": [0, float("nan")], "right": [0, 1]})
    with pytest.raises(PrimitiveRuntimeError, match="interval_union_interval_invalid"):
        runtime.execute("interval_union", {"intervals": [[2, 1]]})
    with pytest.raises(PrimitiveRuntimeError, match="region_polygon_invalid"):
        runtime.execute("region_membership", {"point": [0, 0], "region": [[0, 0], [1, 1]]})
    with pytest.raises(PrimitiveRuntimeError, match="segment_input_invalid"):
        runtime.execute("segment_intersection", {"first": [[0, 0]], "second": [[0, 1], [1, 0]]})
    with pytest.raises(PrimitiveRuntimeError, match="line_of_sight_degenerate_segment"):
        runtime.execute("line_of_sight", {"source": [0, 0], "target": [0, 0], "obstacles": []})


def test_statistical_primitives_are_seeded_and_reproducible():
    runtime = PrimitiveRuntime()
    likelihood = runtime.execute("normal_log_likelihood", {
        "observations": [0.0, 1.0, -1.0], "mean": 0.0, "standard_deviation": 1.0,
    })
    assert likelihood["log_likelihood"] == pytest.approx(-3.7568155996)
    first = runtime.execute("bootstrap_mean", {"values": [1, 2, 3, 4], "resamples": 200, "seed": 7})
    second = runtime.execute("bootstrap_mean", {"values": [1, 2, 3, 4], "resamples": 200, "seed": 7})
    assert first == second
    test = runtime.execute("permutation_test", {
        "first": [1, 2, 3], "second": [1, 1, 1], "permutations": 100, "seed": 4,
    })
    assert 0 < test["p_value_two_sided"] <= 1


def test_statistical_primitives_reject_unbounded_work_or_invalid_parameters():
    runtime = PrimitiveRuntime()
    with pytest.raises(PrimitiveRuntimeError, match="normal_likelihood_input_invalid"):
        runtime.execute("normal_log_likelihood", {"observations": [1], "mean": 0, "standard_deviation": 0})
    with pytest.raises(PrimitiveRuntimeError, match="normal_likelihood_nonfinite"):
        runtime.execute("normal_log_likelihood", {"observations": [1e308, -1e308], "mean": 0, "standard_deviation": 1})
    with pytest.raises(PrimitiveRuntimeError, match="bootstrap_input_invalid"):
        runtime.execute("bootstrap_mean", {"values": [1, 2], "resamples": 20_000, "seed": 1.0})
    with pytest.raises(PrimitiveRuntimeError, match="permutation_input_invalid"):
        runtime.execute("permutation_test", {"first": [1, 2], "second": [1, 2], "permutations": 99, "seed": 1})
