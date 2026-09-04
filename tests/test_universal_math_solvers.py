"""Cross-domain regression tests for the mathematical structure registry."""

import math

import pytest

from core.four_layer_modeling import (
    MathematicalStructureDefinition,
    MathematicalStructureRegistry,
    SolverSpecification,
)
from core.mechanistic_modeling import MechanisticModelingEngine
from core.universal_math_solvers import UniversalRelationValidator, UniversalSolverRegistry


def _run(relations):
    return MechanisticModelingEngine().analyze(
        "执行已核验的结构化数学契约；各节点只按数学形式选择求解器。",
        ir_override={"relations": relations},
    )


def test_structure_catalog_has_broad_honest_coverage():
    registry = MathematicalStructureRegistry()
    catalog = registry.catalog()
    assert len(catalog) >= 25
    implemented = {item["key"] for item in catalog if item["execution_status"] == "implemented"}
    recognized_only = {item["key"] for item in catalog if item["execution_status"] == "recognized_only"}
    assert len(implemented) >= 20
    assert {
        "linear_system", "scalar_polynomial_root", "linear_least_squares",
        "linear_program", "mixed_integer_linear_program", "shortest_path",
        "maximum_flow", "bipartite_matching", "markov_chain", "sample_expectation",
        "hierarchical_finite_action_program",
    } <= implemented
    assert {
        "partial_differential_system", "optimal_control", "nonlinear_system",
        "nonlinear_least_squares", "boundary_value_problem",
        "differential_algebraic_system", "discrete_event_simulation",
    } <= recognized_only
    candidates = registry.recognize(
        "建立偏微分方程并进行鲁棒优化，同时在网络上求最短路径。"
    )
    assert {item["key"] for item in candidates} >= {
        "partial_differential_system", "robust_program", "shortest_path",
    }
    assert all(item["recognition_status"] == "candidate_not_executable" for item in candidates)


def test_hierarchical_finite_actions_execute_through_four_layers():
    result = _run([{
        "id": "hierarchical_selection",
        "kind": "hierarchical_finite_action_program",
        "actions": [
            {
                "id": "open", "decision_unit": "site", "utility": 3.0,
                "active": True, "coverage": {"regional_need": 5.0},
            },
            {
                "id": "closed", "decision_unit": "site", "utility": 0.0,
                "active": False, "coverage": {},
            },
        ],
        "coverage_requirements": [
            {"id": "regional_need", "target": 4.0, "unit": "capacity"},
        ],
        "active_count_bounds": [1, 1],
    }])

    assert result["execution_status"] == "executed"
    numerical = result["numerical_results"][0]
    assert numerical["solver"] == "lexicographic_compiler+scipy.milp.highs"
    assert numerical["selected_action_ids"] == ["open"]
    assert numerical["lexicographic_verified"] is True


def test_ten_generic_backends_compose_in_one_solver_plan():
    relations = [
        {
            "id": "linear", "kind": "linear_system", "variables": ["x", "y"],
            "coefficient_matrix": [[2, 1], [1, -1]], "right_hand_side": [5, 1],
            "units": {"x": "1", "y": "1"},
        },
        {
            "id": "root", "kind": "polynomial_root", "variable": "z",
            "coefficients": [1, 0, -2], "bracket": [0, 2], "units": {"z": "1"},
        },
        {
            "id": "least_squares", "kind": "linear_least_squares",
            "variables": ["intercept", "slope"],
            "design_matrix": [[1, 0], [1, 1], [1, 2]], "observations": [1, 3, 5],
            "units": {"intercept": "1", "slope": "1"},
        },
        {
            "id": "lp", "kind": "linear_program", "variables": ["a", "b"],
            "objective_coefficients": [3, 2], "direction": "maximize",
            "A_ub": [[1, 1], [1, 0]], "b_ub": [4, 2], "A_eq": [], "b_eq": [],
            "bounds": [[0, 4], [0, 4]], "units": {"a": "件", "b": "件"},
        },
        {
            "id": "milp", "kind": "mixed_integer_linear_program", "variables": ["p", "q"],
            "objective_coefficients": [3, 2], "direction": "maximize",
            "A_ub": [[1, 1]], "b_ub": [3], "A_eq": [], "b_eq": [],
            "bounds": [[0, 3], [0, 3]], "integrality": [1, 1],
            "units": {"p": "件", "q": "件"},
        },
        {
            "id": "path", "kind": "shortest_path_problem", "directed": True,
            "source_node": "A", "target_node": "C",
            "edges": [
                {"source": "A", "target": "B", "weight": 1},
                {"source": "B", "target": "C", "weight": 2},
                {"source": "A", "target": "C", "weight": 10},
            ],
        },
        {
            "id": "flow", "kind": "maximum_flow_problem", "directed": True,
            "source_node": "S", "sink_node": "T",
            "edges": [
                {"source": "S", "target": "A", "capacity": 3},
                {"source": "S", "target": "B", "capacity": 2},
                {"source": "A", "target": "T", "capacity": 2},
                {"source": "B", "target": "T", "capacity": 2},
                {"source": "A", "target": "B", "capacity": 1},
            ],
        },
        {
            "id": "matching", "kind": "bipartite_matching_problem",
            "left_nodes": ["L1", "L2"], "right_nodes": ["R1", "R2"],
            "edges": [
                {"source": "L1", "target": "R1"},
                {"source": "L1", "target": "R2"},
                {"source": "L2", "target": "R2"},
            ],
        },
        {
            "id": "markov", "kind": "markov_chain", "state_labels": ["sun", "rain"],
            "transition_matrix": [[0.8, 0.2], [0.4, 0.6]],
            "initial_distribution": [1, 0], "steps": 5,
        },
        {
            "id": "expectation", "kind": "sample_expectation",
            "values": [[1], [2], [1], [2]], "weights": [1, 1, 1, 1],
            "quantity_names": ["cost"], "units": {"cost": "元"},
        },
    ]
    result = _run(relations)
    assert result["execution_status"] == "executed"
    pipeline = result["four_layer_pipeline"]
    assert pipeline["solver_plan"]["budget_summary"]["runnable_nodes"] == 10
    assert pipeline["solver_plan"]["dependency_errors"] == []
    assert len(result["numerical_results"]) == 10
    by_relation = {item["relation_id"]: item for item in result["numerical_results"]}
    assert by_relation["linear"]["solution"] == pytest.approx({"x": 2.0, "y": 1.0})
    assert by_relation["root"]["root"] == pytest.approx(math.sqrt(2), rel=1e-11)
    assert by_relation["least_squares"]["solution"] == pytest.approx({"intercept": 1, "slope": 2})
    assert by_relation["lp"]["objective_value"] == pytest.approx(10.0)
    assert by_relation["milp"]["objective_value"] == pytest.approx(9.0)
    assert by_relation["path"]["path"] == ["A", "B", "C"]
    assert by_relation["path"]["path_length"] == pytest.approx(3.0)
    assert by_relation["flow"]["maximum_flow"] == pytest.approx(4.0)
    assert by_relation["matching"]["matching_size"] == 2
    assert sum(by_relation["markov"]["distribution"].values()) == pytest.approx(1.0)
    assert by_relation["expectation"]["expectation"]["cost"] == pytest.approx(1.5)
    assert pipeline["independent_audit"]["coverage"]["audited_results"] == 10


def test_composition_binds_upstream_result_then_revalidates_contract():
    result = _run([
        {
            "id": "root", "kind": "polynomial_root", "variable": "z",
            "coefficients": [1, 0, -4], "bracket": [0, 3], "units": {"z": "1"},
        },
        {
            "id": "downstream", "kind": "linear_system", "variables": ["x"],
            "coefficient_matrix": [[1]], "right_hand_side": [0], "units": {"x": "1"},
            "input_bindings": [{
                "source_relation_id": "root", "source_path": "root",
                "target_path": "right_hand_side.0",
            }],
        },
    ])
    assert result["execution_status"] == "executed"
    assert result["four_layer_pipeline"]["solver_plan"]["execution_order"] == [
        "plan_math_node_root", "plan_math_node_downstream",
    ]
    assert result["numerical_results"][1]["solution"]["x"] == pytest.approx(2.0)


def test_six_advanced_generic_backends_execute_with_explicit_boundaries():
    result = _run([
        {
            "id": "qp", "kind": "quadratic_program", "variables": ["x", "y"],
            "quadratic_matrix": [[2, 0], [0, 2]], "linear_coefficients": [-2, -4],
            "direction": "minimize", "A_ub": [], "b_ub": [], "A_eq": [], "b_eq": [],
            "bounds": [[-5, 5], [-5, 5]], "units": {"x": "1", "y": "1"},
        },
        {
            "id": "pareto", "kind": "multiobjective_program", "variables": ["x", "y"],
            "objectives": [
                {"name": "x_cost", "coefficients": [1, 0], "direction": "minimize"},
                {"name": "y_cost", "coefficients": [0, 1], "direction": "minimize"},
            ],
            "A_ub": [], "b_ub": [], "A_eq": [[1, 1]], "b_eq": [10],
            "bounds": [[0, 10], [0, 10]], "units": {"x": "件", "y": "件"},
        },
        {
            "id": "robust", "kind": "robust_program", "variables": ["x"],
            "scenario_objective_coefficients": [[1], [2]], "direction": "maximize",
            "A_ub": [], "b_ub": [], "A_eq": [], "b_eq": [],
            "bounds": [[0, 10]], "units": {"x": "件"},
        },
        {
            "id": "stochastic", "kind": "stochastic_program", "variables": ["x"],
            "scenario_objective_coefficients": [[1], [3]], "probabilities": [0.25, 0.75],
            "direction": "maximize", "A_ub": [], "b_ub": [], "A_eq": [], "b_eq": [],
            "bounds": [[0, 10]], "units": {"x": "件"},
        },
        {
            "id": "dp", "kind": "dynamic_program", "states": ["s"],
            "actions": ["low", "high"], "horizon": 2, "direction": "maximize",
            "transition_probabilities": [[[[1], [1]]], [[[1], [1]]]],
            "stage_values": [[[1, 2]], [[1, 2]]], "terminal_values": [0],
            "initial_state": "s",
        },
        {
            "id": "mincost", "kind": "minimum_cost_flow_problem",
            "nodes": ["S", "T"], "node_demands": {"S": -3, "T": 3},
            "edges": [{"source": "S", "target": "T", "capacity": 5, "cost": 2}],
        },
    ])
    assert result["execution_status"] == "executed"
    by_relation = {item["relation_id"]: item for item in result["numerical_results"]}
    assert by_relation["qp"]["solution"] == pytest.approx({"x": 1.0, "y": 2.0}, abs=1e-6)
    assert by_relation["qp"]["objective_value"] == pytest.approx(-5.0, abs=1e-8)
    assert by_relation["pareto"]["nondominated_count"] >= 2
    assert by_relation["pareto"]["credibility_audit"]["status"] == "warning"
    assert by_relation["robust"]["solution"]["x"] == pytest.approx(10.0)
    assert by_relation["robust"]["worst_case_objective"] == pytest.approx(10.0)
    assert by_relation["stochastic"]["expected_objective"] == pytest.approx(25.0)
    assert by_relation["dp"]["initial_state_value"] == pytest.approx(4.0)
    assert by_relation["dp"]["policy"][0]["actions"]["s"] == "high"
    assert by_relation["mincost"]["minimum_cost"] == pytest.approx(6.0)


def test_unverified_or_malformed_universal_contract_is_never_executed():
    result = _run([{
        "id": "bad_lp", "kind": "linear_program", "variables": ["x"],
        "objective_coefficients": [1, 2], "direction": "maximize",
        "A_ub": [], "b_ub": [], "A_eq": [], "b_eq": [],
        "bounds": [[0, 1]], "units": {"x": "1"},
        "parse_status": "machine_verified",
    }])
    relation = result["mathematical_ir"]["relations"][0]
    assert relation["parse_status"] != "machine_verified"
    assert relation["validation_errors"]
    assert result["numerical_results"] == []


def test_recognized_only_structure_is_deferred_instead_of_faking_execution():
    result = _run([{
        "id": "pde", "kind": "pde_system", "equation": "u_t=u_xx",
        "parse_status": "machine_verified", "units": {"u": "1"},
    }])
    node = result["four_layer_pipeline"]["mathematical_ir"]["nodes"][0]
    plan = result["four_layer_pipeline"]["solver_plan"]["nodes"][0]
    assert node["mathematical_form"] == "partial_differential_system"
    assert node["status"] == "deferred"
    assert plan["status"] == "deferred"
    assert "mathematical_contract_not_verified" in plan["deferred_reasons"]
    assert result["numerical_results"] == []


def test_structure_validator_and_solver_registries_extend_without_question_branch():
    structures = MathematicalStructureRegistry()
    solvers = UniversalSolverRegistry()
    definition = MathematicalStructureDefinition(
        key="affine_transform", family="algebraic_system", contract_type="affine_transform/v1",
        relation_kinds=("affine_transform",), description="scalar affine transform",
        solver=SolverSpecification(
            "affine_transform", "affine_transform/v1", "direct affine evaluation",
            ("direct_recalculation",), 3, 10, 1,
        ),
    )

    def validate(payload):
        for field in ("input_value", "scale", "offset"):
            value = float(payload[field])
            if not math.isfinite(value):
                raise ValueError(f"{field}_must_be_finite")
            payload[field] = value
        return payload

    def execute(payload):
        value = payload["scale"] * payload["input_value"] + payload["offset"]
        return {
            "kind": "affine_transform_solution", "status": "executed",
            "relation_id": payload.get("id"), "solver": "direct_affine",
            "value": value, "summary": {"value": value},
            "convergence": {"status": "pass", "relative_tolerance_comparison": 0.0, "acceptance_tolerance": 0.0},
            "credibility_audit": {"status": "pass", "checks": [], "label": "direct", "decision": "direct"},
        }

    structures.register(definition)
    solvers.register("affine_transform/v1", execute)
    UniversalRelationValidator.register("affine_transform", validate)
    try:
        result = MechanisticModelingEngine(
            structure_registry=structures, solver_registry=solvers,
        ).analyze(
            "执行已核验仿射变换。",
            ir_override={"relations": [{
                "id": "affine", "kind": "affine_transform",
                "input_value": 3, "scale": 2, "offset": -1,
            }]},
        )
        assert result["execution_status"] == "executed"
        assert result["numerical_results"][0]["value"] == pytest.approx(5.0)
        assert result["four_layer_pipeline"]["solver_plan"]["selection_rule"] == "mathematical_form_only"
    finally:
        UniversalRelationValidator.unregister_custom("affine_transform")
