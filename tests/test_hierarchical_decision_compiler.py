from core.hierarchical_decision_compiler import HierarchicalDecisionCompiler
from core.universal_math_solvers import UniversalRelationValidator, UniversalSolverRegistry


def test_generic_region_site_lexicographic_decision_has_no_domain_dependency():
    actions = []
    candidates = {
        "site_north_a": ("north", 8.0, 5.0),
        "site_north_b": ("north", 10.0, 2.0),
        "site_south_a": ("south", 7.0, 4.0),
        "site_south_b": ("south", 10.0, 1.0),
    }
    for site, (region, capacity, utility) in candidates.items():
        actions.extend([
            {
                "id": f"{site}_open",
                "decision_unit": site,
                "utility": utility,
                "active": True,
                "coverage": {f"demand_{region}": capacity},
            },
            {
                "id": f"{site}_closed",
                "decision_unit": site,
                "utility": 0.0,
                "active": False,
                "coverage": {},
            },
        ])

    result = HierarchicalDecisionCompiler.solve(
        actions,
        coverage_requirements=[
            {"id": "demand_north", "target": 10.0, "unit": "service_capacity"},
            {"id": "demand_south", "target": 10.0, "unit": "service_capacity"},
        ],
        active_count_bounds=(2, 2),
    )

    assert result["status"] == "executed"
    assert result["lexicographic_verified"] is True
    assert result["selected_active_count"] == 2
    assert {item for item in result["selected_action_ids"] if item.endswith("_open")} == {
        "site_north_b_open", "site_south_b_open",
    }
    assert result["aggregate_coverage"] == 1.0
    assert result["minimum_weighted_shortage"] == 0.0
    assert result["stage_one_result"]["solver"] == "scipy.milp.highs"
    assert result["final_result"]["solver"] == "scipy.milp.highs"


def test_generic_compiler_reports_best_achievable_shortage_before_utility():
    result = HierarchicalDecisionCompiler.solve(
        [
            {
                "id": "plan_high_coverage",
                "decision_unit": "plan",
                "utility": 1.0,
                "coverage": {"requirement": 6.0},
            },
            {
                "id": "plan_high_utility",
                "decision_unit": "plan",
                "utility": 100.0,
                "coverage": {"requirement": 2.0},
            },
        ],
        coverage_requirements=[{"id": "requirement", "target": 10.0}],
    )

    assert result["selected_action_ids"] == ["plan_high_coverage"]
    assert result["coverage"][0]["shortage"] == 4.0
    assert result["minimum_weighted_shortage"] == 4.0


def test_hierarchical_program_is_a_first_class_registered_contract():
    contract = UniversalRelationValidator.verify({
        "id": "cross_domain_selection",
        "kind": "hierarchical_finite_action_program",
        "actions": [
            {
                "id": "activate_a", "decision_unit": "a", "utility": 2.0,
                "active": True, "coverage": {"service": 5.0},
            },
            {
                "id": "skip_a", "decision_unit": "a", "utility": 0.0,
                "active": False, "coverage": {},
            },
        ],
        "coverage_requirements": [
            {"id": "service", "target": 4.0, "unit": "capacity"},
        ],
        "active_count_bounds": [1, 1],
    })

    assert contract["parse_status"] == "machine_verified"
    registry = UniversalSolverRegistry()
    assert registry.has("hierarchical_finite_action/v1")
    result = registry.execute("hierarchical_finite_action/v1", contract)
    assert result["status"] == "executed"
    assert result["selected_action_ids"] == ["activate_a"]
    assert result["aggregate_coverage"] == 1.0


def test_generic_scenario_cvar_changes_risky_choice_and_is_recomputed():
    actions = [
        {
            "id": "safe_plan",
            "decision_unit": "plan",
            "utility": 4.0,
            "scenario_utilities": {"adverse": 4.0, "favorable": 4.0},
        },
        {
            "id": "risky_plan",
            "decision_unit": "plan",
            "utility": 5.0,
            "scenario_utilities": {"adverse": -10.0, "favorable": 20.0},
        },
    ]
    probabilities = {"adverse": 0.5, "favorable": 0.5}

    expected = HierarchicalDecisionCompiler.solve(
        actions,
        scenario_probabilities=probabilities,
        risk_aversion=0.0,
        cvar_confidence=0.5,
    )
    risk_averse = HierarchicalDecisionCompiler.solve(
        actions,
        scenario_probabilities=probabilities,
        risk_aversion=1.0,
        cvar_confidence=0.5,
    )

    assert expected["selected_action_ids"] == ["risky_plan"]
    assert expected["scenario_analysis"]["expected_utility"] == 5.0
    assert risk_averse["selected_action_ids"] == ["safe_plan"]
    assert risk_averse["scenario_analysis"]["lower_tail_cvar"] == 4.0
    assert risk_averse["scenario_analysis"]["worst_case_utility"] == 4.0
    assert risk_averse["scenario_analysis"]["status"] == "pass"
    assert risk_averse["scenario_analysis"]["solver_objective_residual"] <= 1e-8


def test_registered_hierarchical_contract_accepts_scenario_risk_fields():
    contract = UniversalRelationValidator.verify({
        "id": "risk_aware_cross_domain_selection",
        "kind": "hierarchical_finite_action_program",
        "actions": [
            {
                "id": "stable", "decision_unit": "choice", "utility": 2.0,
                "scenario_utilities": {"low": 2.0, "high": 2.0},
            },
            {
                "id": "volatile", "decision_unit": "choice", "utility": 3.0,
                "scenario_utilities": {"low": -4.0, "high": 10.0},
            },
        ],
        "scenario_probabilities": {"low": 0.5, "high": 0.5},
        "risk_aversion": 1.0,
        "cvar_confidence": 0.5,
    })

    assert contract["parse_status"] == "machine_verified"
    result = UniversalSolverRegistry().execute(
        "hierarchical_finite_action/v1", contract
    )
    assert result["status"] == "executed"
    assert result["selected_action_ids"] == ["stable"]
    assert result["scenario_analysis"]["status"] == "pass"
    assert any(
        check["id"] == "scenario_cvar_recalculation"
        and check["status"] == "pass"
        for check in result["credibility_audit"]["checks"]
    )
