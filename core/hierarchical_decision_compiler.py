"""Domain-neutral compiler for hierarchical discrete decisions.

The compiler deliberately knows nothing about products, vegetables, stations,
facilities, or contest statements.  A semantic adapter supplies finite actions,
one-choice decision units, optional activation-count bounds, and optional
upper-level coverage requirements.  This module turns those bindings into a
verified two-stage MILP:

1. minimize weighted unmet requirements;
2. preserve that optimum and maximize utility.  When a caller supplies finite
   scenario utilities, the second stage can maximize a convex combination of
   expected utility and lower-tail CVaR without changing the domain adapter.

Keeping this layer free of domain words makes the same mathematical primitive
usable for assortment selection, facility activation, fleet assignment,
production planning, portfolio baskets, and other hierarchical decisions.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .universal_math_solvers import UniversalRelationValidator, UniversalSolverRegistry


def _finite(value: Any, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field}_must_be_finite")
    return number


class HierarchicalDecisionCompiler:
    """Compile and solve a finite-action hierarchical decision contract."""

    backend_key = "mixed_integer_linear_program/v1"
    maximum_variables = 500

    @classmethod
    def solve(
        cls,
        actions: Sequence[Mapping[str, Any]],
        *,
        coverage_requirements: Sequence[Mapping[str, Any]] = (),
        active_count_bounds: Optional[Tuple[int, int]] = None,
        shortage_tolerance: float = 1e-7,
        scenario_probabilities: Optional[Mapping[str, float]] = None,
        risk_aversion: float = 0.0,
        cvar_confidence: float = 0.9,
    ) -> Dict[str, Any]:
        normalized_actions = cls._normalize_actions(actions)
        requirements = cls._normalize_requirements(coverage_requirements)
        scenario_configuration = cls._normalize_scenario_configuration(
            normalized_actions,
            scenario_probabilities=scenario_probabilities,
            risk_aversion=risk_aversion,
            cvar_confidence=cvar_confidence,
        )
        scenario_variable_count = (
            1 + len(scenario_configuration["probabilities"])
            if scenario_configuration and scenario_configuration["risk_aversion"] > 0
            else 0
        )
        if (
            len(normalized_actions) + len(requirements) + scenario_variable_count
            > cls.maximum_variables
        ):
            raise ValueError(
                f"hierarchical_contract_exceeds_{cls.maximum_variables}_variables"
            )

        decision_units = list(dict.fromkeys(
            str(action["decision_unit"]) for action in normalized_actions
        ))
        if active_count_bounds is not None:
            lower, upper = map(int, active_count_bounds)
            if not 0 <= lower <= upper <= len(decision_units):
                raise ValueError("active_count_bounds_are_infeasible")
            active_count_bounds = (lower, upper)

        action_variables = [f"action_{index}" for index in range(len(normalized_actions))]
        shortage_variables = [f"shortage_{index}" for index in range(len(requirements))]
        variables = action_variables + shortage_variables
        shortage_padding = [0.0] * len(shortage_variables)

        a_eq = [
            [
                1.0 if action["decision_unit"] == unit else 0.0
                for action in normalized_actions
            ] + shortage_padding
            for unit in decision_units
        ]
        a_ub: List[List[float]] = []
        b_ub: List[float] = []
        if active_count_bounds is not None:
            lower, upper = active_count_bounds
            active = [
                1.0 if action["active"] else 0.0 for action in normalized_actions
            ] + shortage_padding
            a_ub.extend([active, [-value for value in active]])
            b_ub.extend([float(upper), float(-lower)])

        for requirement_index, requirement in enumerate(requirements):
            requirement_id = requirement["id"]
            row = [
                -float(action["coverage"].get(requirement_id, 0.0))
                for action in normalized_actions
            ] + [
                -1.0 if index == requirement_index else 0.0
                for index in range(len(shortage_variables))
            ]
            a_ub.append(row)
            b_ub.append(-float(requirement["target"]))

        bounds = (
            [[0.0, 1.0] for _ in action_variables]
            + [[0.0, requirement["target"]] for requirement in requirements]
        )
        integrality = [1] * len(action_variables) + [0] * len(shortage_variables)
        units = {
            **{variable: "binary_action" for variable in action_variables},
            **{
                variable: str(requirements[index]["unit"])
                for index, variable in enumerate(shortage_variables)
            },
        }
        base = {
            "variables": variables,
            "A_ub": a_ub,
            "b_ub": b_ub,
            "A_eq": a_eq,
            "b_eq": [1.0] * len(decision_units),
            "bounds": bounds,
            "integrality": integrality,
            "units": units,
        }

        registry = UniversalSolverRegistry()
        stage_one = None
        minimum_weighted_shortage = None
        lexicographic_verified = not requirements
        if requirements:
            stage_one_contract = UniversalRelationValidator.verify({
                "id": "hierarchical_coverage_stage_one",
                "kind": "mixed_integer_linear_program",
                "objective_coefficients": (
                    [0.0] * len(action_variables)
                    + [-float(item["priority_weight"]) for item in requirements]
                ),
                "direction": "maximize",
                **base,
            })
            if stage_one_contract.get("parse_status") == "machine_verified":
                stage_one = registry.execute(cls.backend_key, stage_one_contract)
                if stage_one.get("status") == "executed":
                    minimum_weighted_shortage = sum(
                        float(requirements[index]["priority_weight"])
                        * float(stage_one.get("solution", {}).get(variable, 0.0))
                        for index, variable in enumerate(shortage_variables)
                    )
                    a_ub = list(a_ub) + [[
                        *([0.0] * len(action_variables)),
                        *[float(item["priority_weight"]) for item in requirements],
                    ]]
                    b_ub = list(b_ub) + [
                        float(minimum_weighted_shortage) + max(0.0, shortage_tolerance)
                    ]
                    base = {**base, "A_ub": a_ub, "b_ub": b_ub}
                    lexicographic_verified = True

        final_base = base
        final_action_objective = [
            float(action["utility"]) for action in normalized_actions
        ]
        final_extra_objective: List[float] = []
        if scenario_configuration:
            probabilities = scenario_configuration["probabilities"]
            risk_weight = float(scenario_configuration["risk_aversion"])
            confidence = float(scenario_configuration["cvar_confidence"])
            final_action_objective = [
                (1.0 - risk_weight) * sum(
                    probability * float(action["scenario_utilities"][scenario])
                    for scenario, probability in probabilities.items()
                )
                for action in normalized_actions
            ]
            if risk_weight > 0:
                scenario_names = list(probabilities)
                action_groups = {
                    unit: [
                        action for action in normalized_actions
                        if action["decision_unit"] == unit
                    ]
                    for unit in decision_units
                }
                scenario_lower_bounds = {
                    scenario: sum(
                        min(
                            float(action["scenario_utilities"][scenario])
                            for action in unit_actions
                        )
                        for unit_actions in action_groups.values()
                    )
                    for scenario in scenario_names
                }
                scenario_upper_bounds = {
                    scenario: sum(
                        max(
                            float(action["scenario_utilities"][scenario])
                            for action in unit_actions
                        )
                        for unit_actions in action_groups.values()
                    )
                    for scenario in scenario_names
                }
                reward_lower = min(scenario_lower_bounds.values())
                reward_upper = max(scenario_upper_bounds.values())
                shortfall_upper = max(0.0, reward_upper - reward_lower)
                risk_variables = ["risk_eta"] + [
                    f"risk_shortfall_{index}" for index in range(len(scenario_names))
                ]
                padding = [0.0] * len(risk_variables)
                risk_a_ub = [list(row) + padding for row in base["A_ub"]]
                risk_a_eq = [list(row) + padding for row in base["A_eq"]]
                for scenario_index, scenario in enumerate(scenario_names):
                    risk_a_ub.append([
                        *[
                            -float(action["scenario_utilities"][scenario])
                            for action in normalized_actions
                        ],
                        *([0.0] * len(shortage_variables)),
                        1.0,
                        *[
                            -1.0 if index == scenario_index else 0.0
                            for index in range(len(scenario_names))
                        ],
                    ])
                final_base = {
                    **base,
                    "variables": list(base["variables"]) + risk_variables,
                    "A_ub": risk_a_ub,
                    "b_ub": list(base["b_ub"]) + [0.0] * len(scenario_names),
                    "A_eq": risk_a_eq,
                    "bounds": list(base["bounds"]) + [
                        [reward_lower, reward_upper],
                        *[[0.0, shortfall_upper] for _ in scenario_names],
                    ],
                    "integrality": list(base["integrality"]) + [0] * len(risk_variables),
                    "units": {
                        **base["units"],
                        **{variable: "scenario_utility" for variable in risk_variables},
                    },
                }
                final_extra_objective = [
                    risk_weight,
                    *[
                        -risk_weight * probabilities[scenario] / (1.0 - confidence)
                        for scenario in scenario_names
                    ],
                ]

        final_contract = UniversalRelationValidator.verify({
            "id": (
                "hierarchical_risk_adjusted_utility_stage_two"
                if scenario_configuration else "hierarchical_utility_stage_two"
            ),
            "kind": "mixed_integer_linear_program",
            "objective_coefficients": (
                final_action_objective
                + [0.0] * len(shortage_variables)
                + final_extra_objective
            ),
            "direction": "maximize",
            **final_base,
        })
        final_result = None
        if final_contract.get("parse_status") == "machine_verified":
            final_result = registry.execute(cls.backend_key, final_contract)

        selected_indices: List[int] = []
        if final_result and final_result.get("status") == "executed":
            selected_indices = [
                index for index, variable in enumerate(action_variables)
                if float(final_result.get("solution", {}).get(variable, 0.0)) >= 0.5
            ]
        coverage = []
        for requirement in requirements:
            achieved = sum(
                float(normalized_actions[index]["coverage"].get(requirement["id"], 0.0))
                for index in selected_indices
            )
            target = float(requirement["target"])
            shortage = max(0.0, target - achieved)
            coverage.append({
                "id": requirement["id"],
                "target": target,
                "achieved": achieved,
                "shortage": shortage,
                "coverage_ratio": min(1.0, achieved / target) if target > 0 else 1.0,
                "priority_weight": float(requirement["priority_weight"]),
                "unit": requirement["unit"],
                "metadata": requirement["metadata"],
            })
        total_target = sum(item["target"] for item in coverage)
        aggregate_coverage = (
            1.0 - sum(item["shortage"] for item in coverage) / max(total_target, 1e-12)
            if coverage else None
        )
        scenario_analysis = None
        if scenario_configuration and selected_indices:
            probabilities = scenario_configuration["probabilities"]
            outcomes = {
                scenario: sum(
                    float(normalized_actions[index]["scenario_utilities"][scenario])
                    for index in selected_indices
                )
                for scenario in probabilities
            }
            expected_utility = sum(
                probabilities[scenario] * outcomes[scenario]
                for scenario in probabilities
            )
            lower_tail_cvar = cls._weighted_lower_tail_cvar(
                outcomes,
                probabilities,
                float(scenario_configuration["cvar_confidence"]),
            )
            risk_weight = float(scenario_configuration["risk_aversion"])
            risk_adjusted_utility = (
                (1.0 - risk_weight) * expected_utility
                + risk_weight * lower_tail_cvar
            )
            solver_objective = (
                float(final_result["objective_value"])
                if final_result and final_result.get("objective_value") is not None
                else None
            )
            objective_residual = (
                abs(solver_objective - risk_adjusted_utility)
                if solver_objective is not None else None
            )
            tolerance = 1e-7 * max(1.0, abs(risk_adjusted_utility))
            certificate_passed = bool(
                all(math.isfinite(value) for value in outcomes.values())
                and math.isfinite(expected_utility)
                and math.isfinite(lower_tail_cvar)
                and (objective_residual is None or objective_residual <= tolerance)
            )
            scenario_analysis = {
                "status": "pass" if certificate_passed else "fail",
                "scenario_probabilities": probabilities,
                "scenario_outcomes": outcomes,
                "expected_utility": expected_utility,
                "worst_case_utility": min(outcomes.values()),
                "lower_tail_cvar": lower_tail_cvar,
                "cvar_confidence": float(
                    scenario_configuration["cvar_confidence"]
                ),
                "risk_aversion": risk_weight,
                "risk_adjusted_utility": risk_adjusted_utility,
                "solver_objective_residual": objective_residual,
                "acceptance_tolerance": tolerance,
            }
        return {
            "status": (
                "executed"
                if final_result and final_result.get("status") == "executed"
                else "not_executed"
            ),
            "selected_action_indices": selected_indices,
            "selected_action_ids": [
                normalized_actions[index]["id"] for index in selected_indices
            ],
            "selected_active_count": sum(
                bool(normalized_actions[index]["active"]) for index in selected_indices
            ),
            "coverage": coverage,
            "aggregate_coverage": aggregate_coverage,
            "minimum_weighted_shortage": minimum_weighted_shortage,
            "lexicographic_verified": lexicographic_verified,
            "stage_one_contract": stage_one_contract if requirements else None,
            "stage_one_result": stage_one,
            "final_contract": final_contract,
            "final_result": final_result,
            "decision_unit_count": len(decision_units),
            "action_count": len(normalized_actions),
            "requirement_count": len(requirements),
            "active_count_bounds": list(active_count_bounds) if active_count_bounds else None,
            "scenario_analysis": scenario_analysis,
        }

    @staticmethod
    def _normalize_actions(
        actions: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not actions:
            raise ValueError("actions_must_not_be_empty")
        normalized: List[Dict[str, Any]] = []
        ids = set()
        for index, raw in enumerate(actions):
            action_id = str(raw.get("id", f"action_{index}"))
            decision_unit = str(raw.get("decision_unit", "")).strip()
            if not action_id or action_id in ids or not decision_unit:
                raise ValueError("action_ids_and_decision_units_must_be_valid")
            ids.add(action_id)
            coverage_raw = raw.get("coverage", {})
            if not isinstance(coverage_raw, Mapping):
                raise ValueError("action_coverage_must_be_a_mapping")
            coverage = {
                str(key): _finite(value, "coverage")
                for key, value in coverage_raw.items()
            }
            if any(value < 0 for value in coverage.values()):
                raise ValueError("coverage_contributions_must_be_nonnegative")
            scenario_raw = raw.get("scenario_utilities", {})
            if not isinstance(scenario_raw, Mapping):
                raise ValueError("scenario_utilities_must_be_a_mapping")
            scenario_utilities = {
                str(key): _finite(value, "scenario_utility")
                for key, value in scenario_raw.items()
            }
            normalized.append({
                "id": action_id,
                "decision_unit": decision_unit,
                "utility": _finite(raw.get("utility", 0.0), "utility"),
                "active": bool(raw.get("active", True)),
                "coverage": coverage,
                "scenario_utilities": scenario_utilities,
                "metadata": dict(raw.get("metadata", {})),
            })
        return normalized

    @staticmethod
    def _normalize_scenario_configuration(
        actions: Sequence[Mapping[str, Any]],
        *,
        scenario_probabilities: Optional[Mapping[str, float]],
        risk_aversion: float,
        cvar_confidence: float,
    ) -> Optional[Dict[str, Any]]:
        risk_weight = _finite(risk_aversion, "risk_aversion")
        confidence = _finite(cvar_confidence, "cvar_confidence")
        if not 0.0 <= risk_weight <= 1.0:
            raise ValueError("risk_aversion_must_be_between_zero_and_one")
        if not 0.0 < confidence < 1.0:
            raise ValueError("cvar_confidence_must_be_between_zero_and_one")
        if scenario_probabilities is None:
            if risk_weight > 0:
                raise ValueError("risk_aversion_requires_scenario_probabilities")
            return None
        if not isinstance(scenario_probabilities, Mapping) or not scenario_probabilities:
            raise ValueError("scenario_probabilities_must_be_a_nonempty_mapping")
        probabilities = {
            str(key): _finite(value, "scenario_probability")
            for key, value in scenario_probabilities.items()
        }
        if any(not key or value <= 0 for key, value in probabilities.items()):
            raise ValueError("scenario_probabilities_must_be_positive")
        probability_sum = sum(probabilities.values())
        if not math.isfinite(probability_sum) or probability_sum <= 0:
            raise ValueError("scenario_probability_sum_must_be_positive")
        probabilities = {
            key: value / probability_sum for key, value in probabilities.items()
        }
        expected_keys = set(probabilities)
        for action in actions:
            if set(action.get("scenario_utilities", {})) != expected_keys:
                raise ValueError(
                    "every_action_must_bind_exactly_the_configured_scenarios"
                )
        return {
            "probabilities": probabilities,
            "risk_aversion": risk_weight,
            "cvar_confidence": confidence,
        }

    @staticmethod
    def _weighted_lower_tail_cvar(
        outcomes: Mapping[str, float],
        probabilities: Mapping[str, float],
        confidence: float,
    ) -> float:
        """Return the exact discrete lower-tail CVaR of utility."""
        tail_mass = 1.0 - float(confidence)
        remaining = tail_mass
        weighted_sum = 0.0
        for scenario, outcome in sorted(
            outcomes.items(), key=lambda item: (float(item[1]), str(item[0]))
        ):
            if remaining <= 1e-15:
                break
            taken = min(remaining, float(probabilities[scenario]))
            weighted_sum += taken * float(outcome)
            remaining -= taken
        if remaining > 1e-10:
            raise ValueError("scenario_probabilities_do_not_cover_cvar_tail")
        return weighted_sum / tail_mass

    @staticmethod
    def _normalize_requirements(
        requirements: Sequence[Mapping[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        ids = set()
        for index, raw in enumerate(requirements):
            requirement_id = str(raw.get("id", f"requirement_{index}"))
            target = _finite(raw.get("target"), "coverage_target")
            weight = _finite(raw.get("priority_weight", 1.0), "priority_weight")
            if not requirement_id or requirement_id in ids or target < 0 or weight <= 0:
                raise ValueError("coverage_requirements_must_have_unique_ids_and_valid_bounds")
            ids.add(requirement_id)
            normalized.append({
                "id": requirement_id,
                "target": target,
                "priority_weight": weight,
                "unit": str(raw.get("unit", "coverage_quantity")) or "coverage_quantity",
                "metadata": dict(raw.get("metadata", {})),
            })
        return normalized
