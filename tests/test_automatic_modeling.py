from copy import deepcopy

import numpy as np
import pytest

from core.automatic_modeling import (
    AutomaticModelingError,
    bind_modeling_task,
    induce_and_solve_modeling_task,
    induce_and_solve_modeling_task_isolated,
)
from core.modeling_benchmark_suite import build_modeling_benchmark_suite, score_modeling_benchmark_case
from core.modeling_open_structure_challenge import build_modeling_open_structure_challenge
from core.modeling_unseen_structure_confirmation import build_unseen_structure_confirmation
from core.modeling_delay_holdout import build_delay_holdout
from core.modeling_logic_temporal_holdout import build_logic_temporal_holdout
from core.model_submission_evaluator import reexecute_submitted_model


def _payload(case):
    return deepcopy(case.public_input)


def test_multitable_induction_is_invariant_to_attachment_order_and_names():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-multitable-many-to-one")
    first = induce_and_solve_modeling_task(_payload(case))
    transformed = _payload(case)
    transformed["attachments"].reverse()
    transformed["attachments"][0]["name"] = "lookup_raw"
    transformed["attachments"][1]["name"] = "events_raw"
    second = induce_and_solve_modeling_task(transformed)
    assert first["model"] == second["model"]
    assert first["group_totals"] == second["group_totals"]
    assert score_modeling_benchmark_case(second, case.hidden_reference)["valid"] is True


def test_ode_structure_selection_uses_data_not_problem_keywords():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-ode-data-driven-structure-selection")
    result = induce_and_solve_modeling_task(_payload(case))
    assert result["model"]["structure"] == "logistic_growth"
    assert score_modeling_benchmark_case(result, case.hidden_reference)["valid"] is True


def test_ode_induction_is_invariant_to_observation_row_order():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-ode-exponential-decay")
    transformed = _payload(case)
    transformed["attachments"][0]["rows"].reverse()
    result = induce_and_solve_modeling_task(transformed)
    assert score_modeling_benchmark_case(result, case.hidden_reference)["valid"] is True


def test_automatic_modeling_has_process_and_memory_supervision():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-algebra-affine")
    result = induce_and_solve_modeling_task_isolated(_payload(case), wall_seconds=10, memory_mb=512)
    assert result["status"] == "completed"
    assert result["execution_supervision"]["process_isolated"] is True
    assert result["execution_supervision"]["permission_isolated"] is True
    assert result["execution_supervision"]["limits"]["memory_mb"] == 512


def test_optimization_induction_is_invariant_to_attachment_order():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-optimization-minimum-cost")
    transformed = _payload(case)
    transformed["attachments"].reverse()
    result = induce_and_solve_modeling_task(transformed)
    assert result["model"]["direction"] == "minimize"
    assert score_modeling_benchmark_case(result, case.hidden_reference)["valid"] is True


def test_optimization_uses_bounded_semantic_aliases_not_reserved_field_names():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-optimization-two-resource")
    payload = _payload(case)
    for attachment in payload["attachments"]:
        for row in attachment["rows"]:
            if "item" in row:
                row["product"] = row.pop("item")
                row["benefit"] = row.pop("profit")
            if "resource" in row:
                row["resource_name"] = row.pop("resource")
                row["available"] = row.pop("capacity")
    result = induce_and_solve_modeling_task(payload)
    assert result["status"] == "completed"
    assert result["model"]["field_binding"]["objective"] == "benefit"
    assert score_modeling_benchmark_case(result, case.hidden_reference)["valid"] is True


def test_ambiguous_duplicate_dimension_is_rejected_before_join():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-multitable-ambiguous-duplicate")
    result = induce_and_solve_modeling_task(_payload(case))
    assert result["status"] == "needs_input"
    assert result["missing"] == ["dimension_deduplication_or_time_rule"]


def test_raw_attachment_budget_and_format_are_bounded():
    with pytest.raises(AutomaticModelingError, match="record_attachments_required"):
        induce_and_solve_modeling_task({"attachments": []})
    with pytest.raises(AutomaticModelingError, match="record_attachment_invalid"):
        induce_and_solve_modeling_task({"attachments": [{"name": "x", "format": "csv", "rows": [{"x": 1}]}]})


def test_model_can_be_identified_before_missing_prediction_query_is_supplied():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-algebra-affine")
    payload = _payload(case)
    payload.pop("query_inputs")
    result = induce_and_solve_modeling_task(payload)
    assert result["status"] == "needs_input"
    assert result["model"]["structure"] == "affine"
    assert result["missing"] == ["query_inputs"]


def test_response_alias_and_explicit_problem_query_are_bound_without_structured_query():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-algebra-affine")
    payload = _payload(case)
    payload.pop("query_inputs")
    for row in payload["attachments"][0]["rows"]:
        row["observed_value"] = row.pop("response")
    payload["problem"] = "识别关系，并计算 input = 5 时的 observed_value。"
    result = induce_and_solve_modeling_task(payload)
    assert result["status"] == "completed"
    assert result["model"]["response_variable"] == "observed_value"
    assert result["predictions"] == pytest.approx([11.5])


def test_explicit_response_column_supports_arbitrary_business_field_name():
    rows = [
        {"temperature": float(value), "net_demand_kw": float(3.0 * value + 2.0)}
        for value in np.linspace(-4.0, 4.0, 40)
    ]
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "meter_readings", "format": "records", "rows": rows}],
        "response_column": "net_demand_kw",
        "problem": "识别关系，并预测 temperature = 5 时的净需求。",
    })
    assert result["status"] == "completed"
    assert result["model"]["response_variable"] == "net_demand_kw"
    assert result["predictions"] == pytest.approx([17.0])


def test_invalid_explicit_response_column_is_not_guessed():
    with pytest.raises(AutomaticModelingError, match="explicit_column_binding_invalid"):
        induce_and_solve_modeling_task({
            "attachments": [{"name": "meter_readings", "format": "records",
                             "rows": [{"temperature": 1.0, "net_demand_kw": 5.0}]}],
            "response_column": "missing_target",
            "problem": "预测 temperature = 5 时的目标。",
        })


def test_ode_aliases_and_explicit_time_query_are_bound():
    case = next(item for item in build_modeling_benchmark_suite()
                if item.structure_group == "modeling-ode-exponential-decay")
    payload = _payload(case)
    payload.pop("query_times")
    for row in payload["attachments"][0]["rows"]:
        row["timestamp"] = row.pop("time")
        row["observed_state"] = row.pop("state")
    payload["problem"] = "识别动力学，并预测 t = 2.5 时的状态。"
    result = induce_and_solve_modeling_task(payload)
    assert result["status"] == "completed"
    assert result["model"]["time_variable"] == "timestamp"
    assert result["model"]["state_variable"] == "observed_state"
    assert result["trajectory"] == pytest.approx([3.0 * np.exp(-0.7 * 2.5)], rel=2e-3)


def test_explicit_target_does_not_reclassify_time_state_records_as_algebra():
    bound = bind_modeling_task({
        "attachments": [{"name": "measurements", "format": "records", "rows": [
            {"time": float(index), "state": float(np.exp(-index))} for index in range(8)
        ]}],
        "response_column": "state",
        "problem": "预测 time = 9 时的状态。",
    })
    assert bound["family"] == "modeling_ode"
    assert bound["bound_query_times"] == [9.0]


@pytest.mark.parametrize("case_factory,signature", [
    (build_modeling_open_structure_challenge, ["cos"]),
    (build_unseen_structure_confirmation, ["divide", "affine", "affine"]),
])
def test_operator_grammar_learns_exposed_nonpolynomial_development_cases(case_factory, signature):
    case = case_factory()[0]
    result = induce_and_solve_modeling_task(_payload(case))
    assert result["model"]["structure"] == "compositional_symbolic"
    assert result["model"]["operator_signature"] == signature
    assert result["predictions"] == pytest.approx(case.hidden_reference["predictions"], rel=2e-3)
    assert result["model"]["search_strategy"] == "deterministic_bounded_enumeration"


def test_expression_depth_budget_is_bounded():
    case = build_modeling_open_structure_challenge()[0]
    payload = _payload(case)
    payload["expression_max_depth"] = 4
    with pytest.raises(AutomaticModelingError, match="expression_depth_budget_invalid"):
        induce_and_solve_modeling_task(payload)


def test_multivariate_power_law_recovers_unlisted_ratio_topology():
    xs = np.linspace(0.4, 4.0, 48)
    zs = np.linspace(1.1, 5.5, 48)[::-1]
    rows = [{"mass": float(x), "scale": float(z),
             "response": float(-2.5 * np.sqrt(x / z))} for x, z in zip(xs, zs)]
    queries = [[0.8, 2.0], [3.2, 5.0]]
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries,
    })
    assert result["model"]["structure"] == "signed_absolute_power_law"
    assert result["model"]["exponents"][:2] == pytest.approx([0.5, -0.5], abs=1e-8)
    assert result["model"]["exponents"][2:] == pytest.approx([0.0] * 4, abs=1e-8)
    assert result["predictions"] == pytest.approx(
        [-2.5 * np.sqrt(0.8 / 2.0), -2.5 * np.sqrt(3.2 / 5.0)], rel=1e-8,
    )
    ablated = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries, "enable_multivariate_nonlinear": False,
    })
    assert ablated["model"]["structure"] != "signed_absolute_power_law"


def test_multivariate_rational_recovers_affine_over_interaction_denominator():
    left = np.linspace(0.5, 3.5, 80)
    right = np.linspace(0.8, 2.4, 80)[np.random.default_rng(4).permutation(80)]
    function = lambda a, b: 3.0 * (a - 1.0) / (b * (a - 4.0))
    rows = [{"alpha": float(b), "theta": float(a), "response": float(function(a, b))}
            for a, b in zip(left, right)]
    queries = [[1.1, 0.7], [2.0, 3.0]]  # sorted inputs: alpha, theta
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries,
    })
    assert result["model"]["structure"] == "multivariate_rational"
    assert result["predictions"] == pytest.approx(
        [function(0.7, 1.1), function(3.0, 2.0)], rel=1e-5,
    )
    assert reexecute_submitted_model(result["model"], queries) == pytest.approx(
        result["predictions"], rel=1e-7, abs=1e-9)


def test_multivariate_rational_composes_unknown_cosine_denominator_position():
    rng = np.random.default_rng(9)
    alpha = rng.uniform(0.2, 1.5, 120)
    force = rng.uniform(0.5, 3.0, 120)
    theta = rng.uniform(-1.2, 1.2, 120)
    function = lambda a, f, t: f / (1.0 + a * np.cos(t))
    rows = [{"alpha": float(a), "force": float(f), "theta": float(t),
             "response": float(function(a, f, t))} for a, f, t in zip(alpha, force, theta)]
    queries = [[0.4, 1.7, -0.3], [1.1, 2.2, 0.8]]
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries,
    })
    assert result["model"]["structure"] == "multivariate_rational"
    assert result["predictions"] == pytest.approx(
        [function(*query) for query in queries], rel=1e-5,
    )
    ablated = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries, "enable_trigonometric_rational_features": False,
    })
    assert "pair_argument" in result["model"]["denominator_feature_grammar"]
    assert "pair_argument" not in ablated["model"].get("denominator_feature_grammar", "")


def test_multivariate_rational_composes_cosine_of_unknown_variable_pair():
    rng = np.random.default_rng(11)
    energy = rng.uniform(0.5, 2.0, 140)
    distance = rng.uniform(0.3, 1.4, 140)
    wave = rng.uniform(0.4, 1.6, 140)
    function = lambda d, e, k: -e / (2.0 * np.cos(d * k) - 2.0)
    rows = [{"distance": float(d), "energy": float(e), "wave": float(k),
             "response": float(function(d, e, k))} for d, e, k in zip(distance, energy, wave)]
    queries = [[0.45, 1.2, 0.8], [1.1, 0.7, 1.3]]
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries,
    })
    assert result["model"]["structure"] == "multivariate_rational"
    assert result["predictions"] == pytest.approx(
        [function(*query) for query in queries], rel=1e-5,
    )
    ablated = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries, "enable_extended_multivariate_composition": False,
    })
    assert "pair_argument" in result["model"]["denominator_feature_grammar"]
    assert "pair_argument" not in ablated["model"].get("denominator_feature_grammar", "")


def test_multivariate_rational_composes_unknown_integer_harmonic_denominator():
    rng = np.random.default_rng(20261004)
    rows = []
    for _ in range(700):
        ef = float(rng.uniform(0.7, 2.2))
        epsilon = float(rng.uniform(0.5, 1.8))
        radius = float(rng.uniform(0.6, 1.7))
        theta = float(rng.uniform(-1.35, -0.2) if len(rows) % 2 else rng.uniform(0.2, 1.35))
        response = 8.0 * np.pi * ef * epsilon * radius ** 3 / (3.0 * np.sin(2.0 * theta))
        rows.append({"Ef": ef, "epsilon": epsilon, "r": radius,
                     "theta": theta, "response": float(response)})
    queries = [[1.1, 0.9, 1.2, 0.6], [1.8, 1.4, 0.8, -1.0]]
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries,
    })
    expected = [8.0 * np.pi * ef * epsilon * radius ** 3 /
                (3.0 * np.sin(2.0 * theta)) for ef, epsilon, radius, theta in queries]
    assert result["model"]["structure"] == "multivariate_rational"
    assert "integer_harmonic_2_3" in result["model"]["denominator_feature_grammar"]
    assert result["predictions"] == pytest.approx(expected, rel=2e-4, abs=2e-4)
    assert reexecute_submitted_model(result["model"], queries) == pytest.approx(
        result["predictions"], rel=1e-7, abs=1e-9)
    ablated = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries, "enable_harmonic_trigonometric_features": False,
    })
    assert ablated["predictions"] != pytest.approx(expected, rel=2e-4, abs=2e-4)


def test_angular_projection_root_composes_unknown_variable_positions():
    rng = np.random.default_rng(20261005)
    rows = []
    for _ in range(500):
        x = float(rng.uniform(2.0, 4.0))
        x1 = float(rng.uniform(0.3, 1.2))
        theta1 = float(rng.uniform(-0.8, 0.8))
        theta2 = float(rng.uniform(-0.8, 0.8))
        delta = theta1 - theta2
        response = x1 * np.cos(delta) - np.sqrt(x * x - x1 * x1 * np.sin(delta) ** 2)
        rows.append({"theta2": theta2, "x": x, "theta1": theta1,
                     "x1": x1, "response": float(response)})
    queries = [[0.5, -0.2, 3.2, 0.8], [-0.4, 0.7, 2.7, 1.0]]
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries,
    })
    expected = []
    for theta1, theta2, x, x1 in [[0.5, -0.2, 3.2, 0.8], [-0.4, 0.7, 2.7, 1.0]]:
        delta = theta1 - theta2
        expected.append(x1 * np.cos(delta) - np.sqrt(x * x - x1 * x1 * np.sin(delta) ** 2))
    assert result["model"]["structure"] == "angular_projection_root"
    assert result["predictions"] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert reexecute_submitted_model(result["model"], queries) == pytest.approx(
        result["predictions"], rel=1e-7, abs=1e-9)
    ablated = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries, "enable_angular_projection_composition": False,
    })
    assert ablated["predictions"] != pytest.approx(expected, rel=1e-5, abs=1e-5)


@pytest.mark.parametrize("variant", ["quotient_difference", "affine_cosine_square", "half_angle_ratio"])
def test_product_angle_rational_composes_unknown_positions(variant):
    rng = np.random.default_rng(20261006)
    rows = []
    if variant == "quotient_difference":
        for _ in range(420):
            omega, time = float(rng.uniform(0.2, 0.8)), float(rng.uniform(0.2, 1.0))
            x, x1 = float(rng.uniform(1.2, 3.0)), float(rng.uniform(0.5, 1.1))
            cosine = np.cos(omega * time)
            rows.append({"omega": omega, "t": time, "x": x, "x1": x1,
                         "response": float((x / cosine - x1) / (x1 * cosine))})
        queries = [[0.35, 0.7, 2.2, 0.8], [0.65, 0.4, 1.7, 0.6]]
        expected = [(x / np.cos(o * t) - x1) / (x1 * np.cos(o * t))
                    for o, t, x, x1 in queries]
    elif variant == "affine_cosine_square":
        for _ in range(420):
            alpha, omega = float(rng.uniform(0.4, 1.5)), float(rng.uniform(0.2, 0.8))
            time, x = float(rng.uniform(0.2, 1.0)), float(rng.uniform(0.8, 2.5))
            cosine = np.cos(omega * time)
            rows.append({"alpha": alpha, "omega": omega, "t": time, "x": x,
                         "response": float(x / (alpha * cosine ** 2 + cosine))})
        queries = [[0.7, 0.4, 0.8, 1.6], [1.2, 0.7, 0.3, 2.1]]
        expected = [x / (a * np.cos(o * t) ** 2 + np.cos(o * t))
                    for a, o, t, x in queries]
    else:
        for _ in range(420):
            intensity = float(rng.uniform(0.6, 2.0))
            n = float(rng.uniform(0.7, 2.2))
            theta = float(rng.uniform(0.25, 1.0))
            rows.append({"Int": intensity, "n": n, "theta": theta,
                         "response": float(2.0 * intensity * np.sin(theta / 2.0) ** 2 /
                                           (1.0 - np.cos(n * theta)))})
        queries = [[1.1, 1.3, 0.6], [1.7, 0.9, 0.8]]
        expected = [2.0 * intensity * np.sin(theta / 2.0) ** 2 /
                    (1.0 - np.cos(n * theta)) for intensity, n, theta in queries]
    payload = {"attachments": [{"name": "observations", "format": "records", "rows": rows}],
               "query_inputs": queries}
    result = induce_and_solve_modeling_task(payload)
    assert result["model"]["structure"] == "product_angle_rational"
    assert result["model"]["substructure"] in {
        "quotient_difference", "affine_cosine_square_denominator",
        "half_angle_over_product_angle",
    }
    assert result["predictions"] == pytest.approx(expected, rel=2e-4, abs=2e-4)
    assert reexecute_submitted_model(result["model"], queries) == pytest.approx(
        result["predictions"], rel=1e-7, abs=1e-9)
    ablated = induce_and_solve_modeling_task(
        {**payload, "enable_product_angle_rational_composition": False})
    assert ablated["predictions"] != pytest.approx(expected, rel=1e-5, abs=1e-5)


def test_root_form_composes_signed_quadratic_and_ratio_terms():
    rng = np.random.default_rng(12)
    by = rng.uniform(0.1, 0.5, 160)
    bz = rng.uniform(0.1, 0.5, 160)
    energy = rng.uniform(2.0, 4.0, 160)
    momentum = rng.uniform(1.0, 1.8, 160)
    function = lambda a, b, e, m: np.sqrt(-a * a - b * b + (e / m) ** 2)
    rows = [{"By": float(a), "Bz": float(b), "energy": float(e), "momentum": float(m),
             "response": float(function(a, b, e, m))}
            for a, b, e, m in zip(by, bz, energy, momentum)]
    queries = [[0.2, 0.3, 2.8, 1.4], [0.4, 0.15, 3.5, 1.2]]
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries,
    })
    assert result["model"]["structure"] == "sqrt_quadratic_form"
    assert result["predictions"] == pytest.approx(
        [function(*query) for query in queries], rel=1e-6,
    )
    assert reexecute_submitted_model(result["model"], queries) == pytest.approx(
        result["predictions"], rel=1e-7, abs=1e-9)
    ablated = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries, "enable_extended_multivariate_composition": False,
    })
    assert ablated["model"]["structure"] != "sqrt_quadratic_form"


def test_power_law_composes_trigonometric_factors_before_fractional_power():
    rng = np.random.default_rng(13)
    field = rng.uniform(0.7, 2.0, 180)
    epsilon = rng.uniform(0.8, 1.7, 180)
    dipole = rng.uniform(0.5, 2.5, 180)
    theta = rng.uniform(0.2, 1.2, 180)
    function = lambda ef, ep, pd, th: (pd * np.sin(th) * np.cos(th) / (ef * ep)) ** (1 / 3)
    rows = [{"Ef": float(ef), "epsilon": float(ep), "p_d": float(pd), "theta": float(th),
             "response": float(function(ef, ep, pd, th))}
            for ef, ep, pd, th in zip(field, epsilon, dipole, theta)]
    queries = [[1.1, 1.2, 0.9, 0.4], [1.7, 0.95, 2.0, 0.9]]
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries,
    })
    assert result["model"]["structure"] == "signed_absolute_power_law"
    assert result["predictions"] == pytest.approx([function(*query) for query in queries], rel=1e-6)
    assert reexecute_submitted_model(result["model"], queries) == pytest.approx(
        result["predictions"], rel=1e-7, abs=1e-9)
    ablated = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries, "enable_transformed_power_laurent": False,
    })
    assert ablated["model"].get("power_feature_grammar") != result["model"]["power_feature_grammar"]


def test_root_form_composes_laurent_term_under_root_then_division():
    rng = np.random.default_rng(14)
    energy = rng.uniform(2.0, 5.0, 180)
    mass = rng.uniform(0.8, 1.8, 180)
    omega = rng.uniform(0.1, 0.6, 180)
    position = rng.uniform(0.7, 1.7, 180)
    function = lambda e, m, o, x: np.sqrt(4.0 * e / m - o * o * x * x) / x
    rows = [{"energy": float(e), "mass": float(m), "omega": float(o), "x": float(x),
             "response": float(function(e, m, o, x))}
            for e, m, o, x in zip(energy, mass, omega, position)]
    queries = [[2.5, 1.1, 0.3, 0.9], [4.2, 1.5, 0.5, 1.3]]
    result = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries,
    })
    assert result["model"]["structure"] == "sqrt_quadratic_form"
    assert result["predictions"] == pytest.approx([function(*query) for query in queries], rel=1e-5)
    assert reexecute_submitted_model(result["model"], queries) == pytest.approx(
        result["predictions"], rel=1e-7, abs=1e-9)
    ablated = induce_and_solve_modeling_task({
        "attachments": [{"name": "observations", "format": "records", "rows": rows}],
        "query_inputs": queries, "enable_transformed_power_laurent": False,
    })
    assert ablated["predictions"] != pytest.approx(result["predictions"], rel=1e-5)


def test_delay_feature_grammar_replays_confirmed_structures():
    for case in build_delay_holdout():
        result = induce_and_solve_modeling_task(_payload(case))
        assert result["model"]["structure"] == "delay_differential_polynomial"
        assert score_modeling_benchmark_case(result, case.hidden_reference)["valid"] is True


def test_logic_and_temporal_grammars_replay_confirmed_structures():
    for case in build_logic_temporal_holdout():
        result = induce_and_solve_modeling_task({**_payload(case), "problem": case.statement})
        assert score_modeling_benchmark_case(result, case.hidden_reference)["valid"] is True
