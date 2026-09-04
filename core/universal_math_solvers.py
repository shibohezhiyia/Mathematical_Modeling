"""Validated, question-independent numerical contracts and solver backends.

Every backend consumes a structured mathematical object.  No backend inspects
problem prose, contest names, or domain labels.  Validation overwrites caller
trust flags and applies finite-size guards before numerical libraries receive
the contract.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_MAX_MATRIX_CELLS = 2_000_000
_MAX_GRAPH_EDGES = 200_000


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        normalized = str(value)
        if normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def _finite_array(
    value: Any, name: str, *, dimensions: Optional[int] = None,
    max_cells: int = _MAX_MATRIX_CELLS,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name}_must_be_numeric") from exc
    if dimensions is not None and array.ndim != dimensions:
        raise ValueError(f"{name}_must_have_{dimensions}_dimensions")
    if array.size == 0 or array.size > max_cells:
        raise ValueError(f"{name}_size_must_be_between_1_and_{max_cells}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}_must_be_finite")
    return array


def _variables(payload: Mapping[str, Any], *, maximum: int) -> List[str]:
    values = payload.get("decision_variables", payload.get("variables", []))
    if (
        not isinstance(values, list) or not 1 <= len(values) <= maximum
        or len(values) != len(set(values))
        or any(not isinstance(name, str) or not _IDENTIFIER.fullmatch(name) for name in values)
    ):
        raise ValueError(f"variables_must_be_1_to_{maximum}_unique_identifiers")
    return list(values)


def _require_units(payload: Mapping[str, Any], names: Sequence[str]) -> Dict[str, str]:
    raw = payload.get("units", {})
    if not isinstance(raw, Mapping) or not set(names) <= set(raw):
        raise ValueError("units_must_cover_all_named_variables")
    units = {str(name): str(raw[name]).strip() for name in names}
    if any(not value for value in units.values()):
        raise ValueError("units_must_be_nonempty")
    return units


def _bounds(
    payload: Mapping[str, Any], variables: Sequence[str], *, finite_required: bool,
) -> List[List[Optional[float]]]:
    raw = payload.get("bounds")
    if isinstance(raw, Mapping):
        if set(raw) != set(variables):
            raise ValueError("bounds_must_bind_every_variable")
        values = [raw[name] for name in variables]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = list(raw)
    else:
        raise ValueError("bounds_must_be_a_mapping_or_sequence")
    if len(values) != len(variables):
        raise ValueError("bounds_length_must_match_variables")
    normalized: List[List[Optional[float]]] = []
    for index, pair in enumerate(values):
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError(f"invalid_bound:{variables[index]}")
        lower = None if pair[0] is None else float(pair[0])
        upper = None if pair[1] is None else float(pair[1])
        if finite_required and (lower is None or upper is None):
            raise ValueError(f"finite_bound_required:{variables[index]}")
        if lower is not None and not math.isfinite(lower):
            raise ValueError(f"invalid_lower_bound:{variables[index]}")
        if upper is not None and not math.isfinite(upper):
            raise ValueError(f"invalid_upper_bound:{variables[index]}")
        if lower is not None and upper is not None and upper < lower:
            raise ValueError(f"reversed_bound:{variables[index]}")
        normalized.append([lower, upper])
    return normalized


def _linear_constraints(
    payload: Mapping[str, Any], n_variables: int,
) -> Tuple[List[List[float]], List[float], List[List[float]], List[float]]:
    output = []
    for matrix_name, vector_name in (("A_ub", "b_ub"), ("A_eq", "b_eq")):
        raw_matrix = payload.get(matrix_name, [])
        raw_vector = payload.get(vector_name, [])
        if raw_matrix is None or (
            isinstance(raw_matrix, Sequence)
            and not isinstance(raw_matrix, (str, bytes))
            and len(raw_matrix) == 0
        ):
            matrix = np.empty((0, n_variables), dtype=float)
            vector = np.empty(0, dtype=float)
        else:
            matrix = _finite_array(raw_matrix, matrix_name, dimensions=2)
            vector = _finite_array(raw_vector, vector_name, dimensions=1)
            if matrix.shape[1] != n_variables or matrix.shape[0] != vector.size:
                raise ValueError(f"{matrix_name}_and_{vector_name}_shape_mismatch")
        output.extend((matrix.tolist(), vector.tolist()))
    return output[0], output[1], output[2], output[3]


def _audit(status: str, label: str, checks: List[Dict[str, Any]], decision: str) -> Dict[str, Any]:
    return {
        "enabled": True,
        "status": status,
        "label": label,
        "checks": checks,
        "decision": decision,
    }


class UniversalRelationValidator:
    """Strict validators for reusable mathematical contracts."""

    _VALIDATORS = {
        "linear_system": "_linear_system",
        "polynomial_root": "_polynomial_root",
        "linear_least_squares": "_linear_least_squares",
        "linear_program": "_linear_program",
        "mixed_integer_linear_program": "_mixed_integer_linear_program",
        "hierarchical_finite_action_program": "_hierarchical_finite_action_program",
        "quadratic_program": "_quadratic_program",
        "multiobjective_program": "_multiobjective_program",
        "robust_program": "_robust_program",
        "stochastic_program": "_stochastic_program",
        "dynamic_program": "_dynamic_program",
        "shortest_path_problem": "_shortest_path",
        "maximum_flow_problem": "_maximum_flow",
        "minimum_cost_flow_problem": "_minimum_cost_flow",
        "bipartite_matching_problem": "_bipartite_matching",
        "markov_chain": "_markov_chain",
        "sample_expectation": "_sample_expectation",
    }

    @classmethod
    def supports(cls, relation_kind: Any) -> bool:
        return str(relation_kind) in cls._VALIDATORS

    @classmethod
    def register(
        cls, relation_kind: str,
        validator: Callable[[Dict[str, Any]], Dict[str, Any]],
    ) -> None:
        kind = str(relation_kind).strip()
        if not kind or kind in cls._VALIDATORS:
            raise ValueError(f"relation validator key is empty or already registered: {kind}")
        if not callable(validator):
            raise TypeError("validator must be callable")
        cls._VALIDATORS[kind] = validator

    @classmethod
    def unregister_custom(cls, relation_kind: str) -> None:
        kind = str(relation_kind)
        handler = cls._VALIDATORS.get(kind)
        if handler is None:
            return
        if isinstance(handler, str):
            raise ValueError("built-in validators cannot be unregistered")
        del cls._VALIDATORS[kind]

    @classmethod
    def verify(cls, relation: Mapping[str, Any]) -> Dict[str, Any]:
        payload = dict(relation)
        payload["parse_status"] = "requires_symbol_and_unit_binding"
        validator = cls._VALIDATORS.get(str(payload.get("kind")))
        if validator is None:
            payload["validation_errors"] = ["unsupported_universal_relation_kind"]
            return payload
        errors = []
        try:
            payload = (
                validator(payload) if callable(validator)
                else getattr(cls, validator)(payload)
            )
            payload["input_bindings"] = cls._input_bindings(payload)
        except (TypeError, ValueError, OverflowError) as exc:
            errors.append(str(exc) or type(exc).__name__)
        payload["validation_errors"] = errors
        if not errors:
            payload["parse_status"] = "machine_verified"
        return payload

    @staticmethod
    def _input_bindings(payload: Mapping[str, Any]) -> List[Dict[str, str]]:
        bindings = payload.get("input_bindings", [])
        if not isinstance(bindings, list) or len(bindings) > 50:
            raise ValueError("input_bindings_must_be_a_list_with_at_most_50_items")
        path_pattern = re.compile(r"^[A-Za-z_]\w*(?:\.(?:[A-Za-z_]\w*|\d+))*$")
        protected = {"id", "kind", "parse_status", "source", "source_text", "input_bindings"}
        normalized = []
        for index, binding in enumerate(bindings):
            if not isinstance(binding, Mapping):
                raise ValueError(f"input_binding_{index}_must_be_an_object")
            source_relation = str(binding.get("source_relation_id", "")).strip()
            source_path = str(binding.get("source_path", "")).strip()
            target_path = str(binding.get("target_path", "")).strip()
            if (
                not source_relation or not path_pattern.fullmatch(source_path)
                or not path_pattern.fullmatch(target_path)
                or target_path.split(".", 1)[0] in protected
            ):
                raise ValueError(f"invalid_input_binding:{index}")
            normalized.append({
                "source_relation_id": source_relation,
                "source_path": source_path,
                "target_path": target_path,
            })
        return normalized

    @staticmethod
    def _hierarchical_finite_action_program(payload: Dict[str, Any]) -> Dict[str, Any]:
        from .hierarchical_decision_compiler import HierarchicalDecisionCompiler

        actions = HierarchicalDecisionCompiler._normalize_actions(
            payload.get("actions", [])
        )
        requirements = HierarchicalDecisionCompiler._normalize_requirements(
            payload.get("coverage_requirements", [])
        )
        scenario_configuration = (
            HierarchicalDecisionCompiler._normalize_scenario_configuration(
                actions,
                scenario_probabilities=payload.get("scenario_probabilities"),
                risk_aversion=payload.get("risk_aversion", 0.0),
                cvar_confidence=payload.get("cvar_confidence", 0.9),
            )
        )
        scenario_variable_count = (
            1 + len(scenario_configuration["probabilities"])
            if scenario_configuration and scenario_configuration["risk_aversion"] > 0
            else 0
        )
        if (
            len(actions) + len(requirements) + scenario_variable_count
            > HierarchicalDecisionCompiler.maximum_variables
        ):
            raise ValueError("hierarchical_contract_exceeds_variable_budget")
        raw_bounds = payload.get("active_count_bounds")
        active_bounds = None
        if raw_bounds is not None:
            if (
                not isinstance(raw_bounds, Sequence)
                or isinstance(raw_bounds, (str, bytes))
                or len(raw_bounds) != 2
            ):
                raise ValueError("active_count_bounds_must_have_two_values")
            active_bounds = [int(raw_bounds[0]), int(raw_bounds[1])]
            unit_count = len({item["decision_unit"] for item in actions})
            if not 0 <= active_bounds[0] <= active_bounds[1] <= unit_count:
                raise ValueError("active_count_bounds_are_infeasible")
        payload.update({
            "actions": actions,
            "coverage_requirements": requirements,
            "active_count_bounds": active_bounds,
            "scenario_probabilities": (
                scenario_configuration["probabilities"]
                if scenario_configuration else None
            ),
            "risk_aversion": (
                scenario_configuration["risk_aversion"]
                if scenario_configuration else 0.0
            ),
            "cvar_confidence": (
                scenario_configuration["cvar_confidence"]
                if scenario_configuration else float(payload.get("cvar_confidence", 0.9))
            ),
        })
        return payload

    @staticmethod
    def _linear_system(payload: Dict[str, Any]) -> Dict[str, Any]:
        variables = _variables(payload, maximum=500)
        matrix = _finite_array(payload.get("coefficient_matrix"), "coefficient_matrix", dimensions=2)
        rhs = _finite_array(payload.get("right_hand_side"), "right_hand_side", dimensions=1)
        if matrix.shape[1] != len(variables) or matrix.shape[0] != rhs.size:
            raise ValueError("linear_system_shape_mismatch")
        _require_units(payload, variables)
        payload["variables"] = variables
        payload["coefficient_matrix"] = matrix.tolist()
        payload["right_hand_side"] = rhs.tolist()
        return payload

    @staticmethod
    def _polynomial_root(payload: Dict[str, Any]) -> Dict[str, Any]:
        coefficients = _finite_array(payload.get("coefficients"), "coefficients", dimensions=1, max_cells=101)
        if coefficients.size < 2 or abs(float(coefficients[0])) <= np.finfo(float).eps:
            raise ValueError("polynomial_must_have_degree_at_least_one_and_nonzero_leading_coefficient")
        bracket = _finite_array(payload.get("bracket"), "bracket", dimensions=1, max_cells=2)
        if bracket.size != 2 or bracket[1] <= bracket[0]:
            raise ValueError("bracket_must_be_two_finite_increasing_values")
        variable = str(payload.get("variable", "x"))
        if not _IDENTIFIER.fullmatch(variable):
            raise ValueError("root_variable_must_be_an_identifier")
        _require_units(payload, [variable])
        left, right = float(np.polyval(coefficients, bracket[0])), float(np.polyval(coefficients, bracket[1]))
        if left != 0.0 and right != 0.0 and math.copysign(1.0, left) == math.copysign(1.0, right):
            raise ValueError("bracket_does_not_contain_a_sign_change")
        payload.update({
            "coefficients": coefficients.tolist(), "bracket": bracket.tolist(),
            "variable": variable,
        })
        return payload

    @staticmethod
    def _linear_least_squares(payload: Dict[str, Any]) -> Dict[str, Any]:
        variables = _variables(payload, maximum=300)
        design = _finite_array(payload.get("design_matrix"), "design_matrix", dimensions=2)
        observed = _finite_array(payload.get("observations"), "observations", dimensions=1)
        if design.shape[1] != len(variables) or design.shape[0] != observed.size:
            raise ValueError("least_squares_shape_mismatch")
        _require_units(payload, variables)
        bounds = payload.get("bounds")
        normalized_bounds = None if bounds is None else _bounds(payload, variables, finite_required=True)
        payload.update({
            "variables": variables, "design_matrix": design.tolist(),
            "observations": observed.tolist(), "bounds": normalized_bounds,
        })
        return payload

    @classmethod
    def _linear_program(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        return cls._program(payload, mixed_integer=False)

    @classmethod
    def _mixed_integer_linear_program(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        return cls._program(payload, mixed_integer=True)

    @classmethod
    def _quadratic_program(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        variables = _variables(payload, maximum=300)
        linear = _finite_array(payload.get("linear_coefficients"), "linear_coefficients", dimensions=1)
        quadratic = _finite_array(payload.get("quadratic_matrix"), "quadratic_matrix", dimensions=2)
        if linear.size != len(variables) or quadratic.shape != (len(variables), len(variables)):
            raise ValueError("quadratic_program_shape_mismatch")
        if not np.allclose(quadratic, quadratic.T, atol=1e-12):
            raise ValueError("quadratic_matrix_must_be_symmetric")
        direction = str(payload.get("direction", "minimize")).lower()
        if direction not in {"minimize", "maximize", "min", "max"}:
            raise ValueError("direction_must_be_minimize_or_maximize")
        normalized_direction = "maximize" if direction in {"maximize", "max"} else "minimize"
        eigenvalues = np.linalg.eigvalsh(quadratic)
        if normalized_direction == "minimize" and float(np.min(eigenvalues)) < -1e-10:
            raise ValueError("minimization_quadratic_matrix_must_be_positive_semidefinite")
        if normalized_direction == "maximize" and float(np.max(eigenvalues)) > 1e-10:
            raise ValueError("maximization_quadratic_matrix_must_be_negative_semidefinite")
        bounds = _bounds(payload, variables, finite_required=True)
        a_ub, b_ub, a_eq, b_eq = _linear_constraints(payload, len(variables))
        _require_units(payload, variables)
        payload.update({
            "variables": variables, "decision_variables": variables,
            "linear_coefficients": linear.tolist(), "quadratic_matrix": quadratic.tolist(),
            "direction": normalized_direction, "bounds": bounds,
            "A_ub": a_ub, "b_ub": b_ub, "A_eq": a_eq, "b_eq": b_eq,
            "convexity_eigenvalue_range": [float(np.min(eigenvalues)), float(np.max(eigenvalues))],
        })
        return payload

    @classmethod
    def _multiobjective_program(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        objectives = payload.get("objectives", [])
        if not isinstance(objectives, list) or not 2 <= len(objectives) <= 8:
            raise ValueError("objectives_must_contain_2_to_8_linear_objectives")
        variables = _variables(payload, maximum=200)
        normalized_objectives = []
        for index, objective in enumerate(objectives):
            if not isinstance(objective, Mapping):
                raise ValueError(f"objective_{index}_must_be_an_object")
            coefficients = _finite_array(objective.get("coefficients"), f"objective_{index}_coefficients", dimensions=1)
            if coefficients.size != len(variables):
                raise ValueError(f"objective_{index}_length_mismatch")
            direction = str(objective.get("direction", "minimize")).lower()
            if direction not in {"minimize", "maximize", "min", "max"}:
                raise ValueError(f"objective_{index}_direction_invalid")
            normalized_objectives.append({
                "name": str(objective.get("name", f"objective_{index + 1}")),
                "coefficients": coefficients.tolist(),
                "direction": "maximize" if direction in {"maximize", "max"} else "minimize",
            })
        base = dict(payload)
        base["objective_coefficients"] = normalized_objectives[0]["coefficients"]
        base["direction"] = normalized_objectives[0]["direction"]
        base = cls._program(base, mixed_integer=False)
        base["objectives"] = normalized_objectives
        return base

    @classmethod
    def _robust_program(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        return cls._scenario_program(payload, require_probabilities=False)

    @classmethod
    def _stochastic_program(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        return cls._scenario_program(payload, require_probabilities=True)

    @classmethod
    def _scenario_program(
        cls, payload: Dict[str, Any], *, require_probabilities: bool,
    ) -> Dict[str, Any]:
        variables = _variables(payload, maximum=300)
        scenarios = _finite_array(
            payload.get("scenario_objective_coefficients"),
            "scenario_objective_coefficients", dimensions=2,
        )
        if scenarios.shape[1] != len(variables) or not 2 <= scenarios.shape[0] <= 1000:
            raise ValueError("scenario_objectives_must_have_2_to_1000_rows_and_match_variables")
        base = dict(payload)
        base["objective_coefficients"] = scenarios[0].tolist()
        base = cls._program(base, mixed_integer=False)
        base["scenario_objective_coefficients"] = scenarios.tolist()
        if require_probabilities:
            probabilities = _finite_array(payload.get("probabilities"), "probabilities", dimensions=1, max_cells=1000)
            if probabilities.size != scenarios.shape[0] or np.min(probabilities) < 0 or probabilities.sum() <= 0:
                raise ValueError("probabilities_must_be_nonnegative_and_match_scenarios")
            probabilities = probabilities / probabilities.sum()
            base["probabilities"] = probabilities.tolist()
        return base

    @staticmethod
    def _dynamic_program(payload: Dict[str, Any]) -> Dict[str, Any]:
        states = [str(value) for value in payload.get("states", [])]
        actions = [str(value) for value in payload.get("actions", [])]
        if (
            not 1 <= len(states) <= 1000 or len(states) != len(set(states))
            or not 1 <= len(actions) <= 200 or len(actions) != len(set(actions))
        ):
            raise ValueError("states_and_actions_must_be_unique_and_bounded")
        transition = _finite_array(
            payload.get("transition_probabilities"), "transition_probabilities",
            dimensions=4, max_cells=2_000_000,
        )
        values = _finite_array(payload.get("stage_values"), "stage_values", dimensions=3, max_cells=2_000_000)
        horizon = payload.get("horizon")
        if not isinstance(horizon, int) or isinstance(horizon, bool) or not 1 <= horizon <= 1000:
            raise ValueError("horizon_must_be_an_integer_between_1_and_1000")
        expected_transition_shape = (horizon, len(states), len(actions), len(states))
        expected_value_shape = (horizon, len(states), len(actions))
        if transition.shape != expected_transition_shape or values.shape != expected_value_shape:
            raise ValueError("dynamic_program_tensor_shape_mismatch")
        if np.min(transition) < -1e-12 or not np.allclose(transition.sum(axis=3), 1.0, atol=1e-10):
            raise ValueError("dynamic_program_transitions_must_be_stochastic")
        terminal = payload.get("terminal_values", [0.0] * len(states))
        terminal_values = _finite_array(terminal, "terminal_values", dimensions=1, max_cells=1000)
        if terminal_values.size != len(states):
            raise ValueError("terminal_values_must_match_states")
        direction = str(payload.get("direction", "maximize")).lower()
        if direction not in {"minimize", "maximize", "min", "max"}:
            raise ValueError("direction_must_be_minimize_or_maximize")
        initial_state = payload.get("initial_state")
        if initial_state is not None and str(initial_state) not in states:
            raise ValueError("initial_state_must_be_a_declared_state")
        payload.update({
            "states": states, "actions": actions, "transition_probabilities": transition.tolist(),
            "stage_values": values.tolist(), "terminal_values": terminal_values.tolist(),
            "horizon": horizon, "direction": "maximize" if direction in {"maximize", "max"} else "minimize",
            "initial_state": None if initial_state is None else str(initial_state),
        })
        return payload

    @staticmethod
    def _program(payload: Dict[str, Any], *, mixed_integer: bool) -> Dict[str, Any]:
        maximum = 500 if mixed_integer else 1000
        variables = _variables(payload, maximum=maximum)
        objective = _finite_array(payload.get("objective_coefficients"), "objective_coefficients", dimensions=1)
        if objective.size != len(variables):
            raise ValueError("objective_length_must_match_variables")
        direction = str(payload.get("direction", "minimize")).lower()
        if direction not in {"minimize", "maximize", "min", "max"}:
            raise ValueError("direction_must_be_minimize_or_maximize")
        bounds = _bounds(payload, variables, finite_required=mixed_integer)
        a_ub, b_ub, a_eq, b_eq = _linear_constraints(payload, len(variables))
        _require_units(payload, variables)
        payload.update({
            "variables": variables, "decision_variables": variables,
            "objective_coefficients": objective.tolist(),
            "direction": "maximize" if direction in {"maximize", "max"} else "minimize",
            "bounds": bounds, "A_ub": a_ub, "b_ub": b_ub, "A_eq": a_eq, "b_eq": b_eq,
        })
        if mixed_integer:
            integrality = _finite_array(payload.get("integrality"), "integrality", dimensions=1)
            if integrality.size != len(variables) or any(int(value) not in {0, 1, 2, 3} or value != int(value) for value in integrality):
                raise ValueError("integrality_must_use_scipy_codes_0_to_3_for_every_variable")
            payload["integrality"] = [int(value) for value in integrality]
        return payload

    @staticmethod
    def _normalize_edges(payload: Dict[str, Any], value_field: str, *, nonnegative: bool) -> List[Dict[str, Any]]:
        edges = payload.get("edges", [])
        if not isinstance(edges, list) or not 1 <= len(edges) <= _MAX_GRAPH_EDGES:
            raise ValueError(f"edges_must_be_1_to_{_MAX_GRAPH_EDGES}_items")
        directed = bool(payload.get("directed", True))
        normalized = []
        seen = set()
        for index, edge in enumerate(edges):
            if not isinstance(edge, Mapping) or "source" not in edge or "target" not in edge:
                raise ValueError(f"invalid_edge:{index}")
            source, target = str(edge["source"]), str(edge["target"])
            if not source or not target or source == target:
                raise ValueError(f"invalid_edge_endpoints:{index}")
            signature = (source, target) if directed else tuple(sorted((source, target)))
            if signature in seen:
                raise ValueError(f"duplicate_edge_requires_explicit_aggregation:{source}:{target}")
            seen.add(signature)
            value = float(edge.get(value_field, 1.0))
            if not math.isfinite(value) or (nonnegative and value < 0):
                raise ValueError(f"invalid_{value_field}:{index}")
            normalized.append({"source": source, "target": target, value_field: value})
        payload["directed"] = directed
        return normalized

    @classmethod
    def _shortest_path(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        edges = cls._normalize_edges(payload, "weight", nonnegative=False)
        source, target = str(payload.get("source_node", "")), str(payload.get("target_node", ""))
        nodes = {item[key] for item in edges for key in ("source", "target")}
        if source not in nodes or target not in nodes or source == target:
            raise ValueError("source_and_target_must_be_distinct_graph_nodes")
        payload.update({"edges": edges, "source_node": source, "target_node": target})
        return payload

    @classmethod
    def _maximum_flow(cls, payload: Dict[str, Any]) -> Dict[str, Any]:
        edges = cls._normalize_edges(payload, "capacity", nonnegative=True)
        source, sink = str(payload.get("source_node", "")), str(payload.get("sink_node", ""))
        nodes = {item[key] for item in edges for key in ("source", "target")}
        if source not in nodes or sink not in nodes or source == sink:
            raise ValueError("source_and_sink_must_be_distinct_graph_nodes")
        if not payload.get("directed", True):
            raise ValueError("maximum_flow_contract_requires_a_directed_graph")
        payload.update({"edges": edges, "source_node": source, "sink_node": sink})
        return payload

    @staticmethod
    def _minimum_cost_flow(payload: Dict[str, Any]) -> Dict[str, Any]:
        nodes = _unique(payload.get("nodes", []))
        if not 2 <= len(nodes) <= 100_000:
            raise ValueError("nodes_must_contain_2_to_100000_unique_labels")
        raw_demands = payload.get("node_demands", {})
        if not isinstance(raw_demands, Mapping) or set(map(str, raw_demands)) != set(nodes):
            raise ValueError("node_demands_must_bind_every_node")
        demands = {str(node): float(raw_demands[node]) for node in raw_demands}
        if not all(math.isfinite(value) for value in demands.values()) or abs(sum(demands.values())) > 1e-9:
            raise ValueError("node_demands_must_be_finite_and_sum_to_zero")
        edges = payload.get("edges", [])
        if not isinstance(edges, list) or not 1 <= len(edges) <= _MAX_GRAPH_EDGES:
            raise ValueError("invalid_minimum_cost_flow_edges")
        normalized, seen = [], set()
        for index, edge in enumerate(edges):
            if not isinstance(edge, Mapping):
                raise ValueError(f"invalid_edge:{index}")
            source, target = str(edge.get("source", "")), str(edge.get("target", ""))
            if source not in nodes or target not in nodes or source == target or (source, target) in seen:
                raise ValueError(f"invalid_or_duplicate_flow_edge:{index}")
            seen.add((source, target))
            capacity, cost = float(edge.get("capacity")), float(edge.get("cost"))
            if not (math.isfinite(capacity) and capacity >= 0 and math.isfinite(cost)):
                raise ValueError(f"invalid_capacity_or_cost:{index}")
            normalized.append({"source": source, "target": target, "capacity": capacity, "cost": cost})
        payload.update({"nodes": nodes, "node_demands": demands, "edges": normalized, "directed": True})
        return payload

    @staticmethod
    def _bipartite_matching(payload: Dict[str, Any]) -> Dict[str, Any]:
        left = _unique(payload.get("left_nodes", []))
        right = _unique(payload.get("right_nodes", []))
        if not left or not right or len(left) + len(right) > 100_000 or set(left) & set(right):
            raise ValueError("bipartite_partitions_must_be_nonempty_disjoint_and_bounded")
        edges = payload.get("edges", [])
        if not isinstance(edges, list) or not 1 <= len(edges) <= _MAX_GRAPH_EDGES:
            raise ValueError("invalid_bipartite_edges")
        normalized, seen = [], set()
        for index, edge in enumerate(edges):
            if not isinstance(edge, Mapping):
                raise ValueError(f"invalid_edge:{index}")
            source, target = str(edge.get("source", "")), str(edge.get("target", ""))
            if source not in left or target not in right or (source, target) in seen:
                raise ValueError(f"edge_must_cross_partitions_without_duplicates:{index}")
            seen.add((source, target))
            weight = float(edge.get("weight", 1.0))
            if not math.isfinite(weight):
                raise ValueError(f"invalid_weight:{index}")
            normalized.append({"source": source, "target": target, "weight": weight})
        payload.update({"left_nodes": left, "right_nodes": right, "edges": normalized})
        return payload

    @staticmethod
    def _markov_chain(payload: Dict[str, Any]) -> Dict[str, Any]:
        transition = _finite_array(payload.get("transition_matrix"), "transition_matrix", dimensions=2, max_cells=4_000_000)
        if transition.shape[0] != transition.shape[1] or transition.shape[0] > 2000:
            raise ValueError("transition_matrix_must_be_square_with_at_most_2000_states")
        if np.min(transition) < -1e-12 or not np.allclose(transition.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("transition_matrix_must_be_row_stochastic")
        initial = _finite_array(payload.get("initial_distribution"), "initial_distribution", dimensions=1, max_cells=2000)
        if initial.size != transition.shape[0] or np.min(initial) < -1e-12 or not np.isclose(initial.sum(), 1.0, atol=1e-10):
            raise ValueError("initial_distribution_must_be_a_probability_vector")
        steps = payload.get("steps", 1)
        if not isinstance(steps, int) or isinstance(steps, bool) or not 0 <= steps <= 1_000_000:
            raise ValueError("steps_must_be_an_integer_between_0_and_1000000")
        labels = payload.get("state_labels") or [f"state_{index}" for index in range(initial.size)]
        if not isinstance(labels, list) or len(labels) != initial.size or len(set(map(str, labels))) != len(labels):
            raise ValueError("state_labels_must_be_unique_and_match_matrix_size")
        payload.update({
            "transition_matrix": transition.tolist(), "initial_distribution": initial.tolist(),
            "steps": steps, "state_labels": [str(label) for label in labels],
        })
        return payload

    @staticmethod
    def _sample_expectation(payload: Dict[str, Any]) -> Dict[str, Any]:
        values = _finite_array(payload.get("values"), "values", max_cells=2_000_000)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] > 100:
            raise ValueError("values_must_be_a_2d_sample_with_2_or_more_rows_and_at_most_100_outputs")
        raw_weights = payload.get("weights")
        weights = np.ones(values.shape[0], dtype=float) if raw_weights is None else _finite_array(raw_weights, "weights", dimensions=1)
        if weights.size != values.shape[0] or np.min(weights) < 0 or weights.sum() <= 0:
            raise ValueError("weights_must_be_nonnegative_and_match_sample_rows")
        names = payload.get("quantity_names") or [f"q_{index}" for index in range(values.shape[1])]
        if not isinstance(names, list) or len(names) != values.shape[1] or len(set(map(str, names))) != len(names):
            raise ValueError("quantity_names_must_be_unique_and_match_sample_columns")
        _require_units(payload, [str(name) for name in names])
        payload.update({
            "values": values.tolist(), "weights": weights.tolist(),
            "quantity_names": [str(name) for name in names],
        })
        return payload


class UniversalSolverRegistry:
    """Extensible executor registry keyed by mathematical contract version."""

    def __init__(self) -> None:
        self._executors: Dict[str, Callable[[Mapping[str, Any]], Dict[str, Any]]] = {
            "linear_system/v1": self._solve_linear_system,
            "polynomial_root/v1": self._solve_polynomial_root,
            "linear_least_squares/v1": self._solve_linear_least_squares,
            "linear_program/v1": self._solve_linear_program,
            "mixed_integer_linear_program/v1": self._solve_mixed_integer_linear_program,
            "hierarchical_finite_action/v1": self._solve_hierarchical_finite_action,
            "quadratic_program/v1": self._solve_quadratic_program,
            "multiobjective/v1": self._solve_multiobjective_program,
            "robust_program/v1": self._solve_robust_program,
            "stochastic_program/v1": self._solve_stochastic_program,
            "dynamic_program/v1": self._solve_dynamic_program,
            "shortest_path/v1": self._solve_shortest_path,
            "maximum_flow/v1": self._solve_maximum_flow,
            "minimum_cost_flow/v1": self._solve_minimum_cost_flow,
            "bipartite_matching/v1": self._solve_bipartite_matching,
            "markov_chain/v1": self._solve_markov_chain,
            "sample_expectation/v1": self._solve_sample_expectation,
        }

    def register(
        self, executor_key: str,
        executor: Callable[[Mapping[str, Any]], Dict[str, Any]],
    ) -> None:
        key = str(executor_key).strip()
        if not key or key in self._executors:
            raise ValueError(f"executor key is empty or already registered: {key}")
        if not callable(executor):
            raise TypeError("executor must be callable")
        self._executors[key] = executor

    def has(self, executor_key: str) -> bool:
        return str(executor_key) in self._executors

    def execute(self, executor_key: str, contract: Mapping[str, Any]) -> Dict[str, Any]:
        try:
            executor = self._executors[str(executor_key)]
        except KeyError as exc:
            raise KeyError(f"no universal executor registered for {executor_key}") from exc
        return executor(contract)

    @staticmethod
    def _solve_hierarchical_finite_action(
        relation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        from .hierarchical_decision_compiler import HierarchicalDecisionCompiler

        compiled = HierarchicalDecisionCompiler.solve(
            relation.get("actions", []),
            coverage_requirements=relation.get("coverage_requirements", []),
            active_count_bounds=(
                tuple(relation["active_count_bounds"])
                if relation.get("active_count_bounds") is not None else None
            ),
            scenario_probabilities=relation.get("scenario_probabilities"),
            risk_aversion=float(relation.get("risk_aversion", 0.0)),
            cvar_confidence=float(relation.get("cvar_confidence", 0.9)),
        )
        final = compiled.get("final_result") or {}
        credibility = final.get("credibility_audit", _audit(
            "warning", "层级有限动作求解未完成", [],
            "仅在两阶段均成功时使用结果。",
        ))
        scenario_analysis = compiled.get("scenario_analysis")
        if scenario_analysis:
            credibility = {
                **credibility,
                "status": (
                    "fail"
                    if scenario_analysis.get("status") == "fail"
                    else credibility.get("status", "warning")
                ),
                "checks": [
                    *credibility.get("checks", []),
                    {
                        "id": "scenario_cvar_recalculation",
                        "name": "有限情景CVaR独立复算",
                        "status": scenario_analysis.get("status", "not_assessed"),
                        "evidence": (
                            f"probability_sum={sum(scenario_analysis.get('scenario_probabilities', {}).values()):.12g}, "
                            f"objective_residual={scenario_analysis.get('solver_objective_residual')}"
                        ),
                    },
                ],
            }
        return {
            "kind": "hierarchical_finite_action_solution",
            "status": compiled["status"],
            "relation_id": relation.get("id"),
            "solver": "lexicographic_compiler+scipy.milp.highs",
            "selected_action_ids": compiled["selected_action_ids"],
            "selected_active_count": compiled["selected_active_count"],
            "coverage": compiled["coverage"],
            "aggregate_coverage": compiled["aggregate_coverage"],
            "minimum_weighted_shortage": compiled["minimum_weighted_shortage"],
            "lexicographic_verified": compiled["lexicographic_verified"],
            "scenario_analysis": scenario_analysis,
            "summary": {
                "selected_active_count": compiled["selected_active_count"],
                "aggregate_coverage": compiled["aggregate_coverage"],
                "minimum_weighted_shortage": compiled["minimum_weighted_shortage"],
                "objective_value": final.get("objective_value"),
                "risk_adjusted_utility": (
                    (compiled.get("scenario_analysis") or {}).get(
                        "risk_adjusted_utility"
                    )
                ),
            },
            "convergence": final.get("convergence", {"status": "not_executed"}),
            "credibility_audit": credibility,
        }

    @staticmethod
    def _solve_linear_system(relation: Mapping[str, Any]) -> Dict[str, Any]:
        matrix = np.asarray(relation["coefficient_matrix"], dtype=float)
        rhs = np.asarray(relation["right_hand_side"], dtype=float)
        variables = list(relation["variables"])
        rank = int(np.linalg.matrix_rank(matrix))
        square_full_rank = matrix.shape[0] == matrix.shape[1] == rank
        solution = np.linalg.solve(matrix, rhs) if square_full_rank else np.linalg.lstsq(matrix, rhs, rcond=None)[0]
        confirmation = np.linalg.pinv(matrix) @ rhs
        residual = matrix @ solution - rhs
        relative_residual = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(rhs)))
        confirmation_difference = float(np.linalg.norm(solution - confirmation) / max(1.0, np.linalg.norm(solution)))
        condition = float(np.linalg.cond(matrix))
        unique = rank == matrix.shape[1]
        status = "pass" if relative_residual <= 1e-9 and unique and condition <= 1e12 else "warning"
        checks = [
            {"id": "linear_residual", "name": "线性方程残差", "status": "pass" if relative_residual <= 1e-9 else "fail", "evidence": f"relative_residual={relative_residual:.3g}"},
            {"id": "rank_and_uniqueness", "name": "秩与唯一性", "status": "pass" if unique else "warning", "evidence": f"rank={rank}, shape={matrix.shape}"},
            {"id": "pseudoinverse_confirmation", "name": "伪逆独立复算", "status": "pass" if confirmation_difference <= 1e-8 else "warning", "evidence": f"relative_difference={confirmation_difference:.3g}"},
            {"id": "conditioning", "name": "条件数", "status": "pass" if condition <= 1e12 else "warning", "evidence": f"condition_number={condition:.3g}"},
        ]
        return {
            "kind": "linear_system_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "numpy.solve_or_lstsq", "variables": variables,
            "solution": {name: float(solution[index]) for index, name in enumerate(variables)},
            "rank": rank, "condition_number": condition, "relative_residual": relative_residual,
            "summary": {"relative_residual": relative_residual, "rank": rank, "condition_number": condition},
            "convergence": {"status": "pass" if relative_residual <= 1e-9 else "fail", "relative_tolerance_comparison": relative_residual, "acceptance_tolerance": 1e-9},
            "credibility_audit": _audit(status, "线性系统已复算" if status == "pass" else "线性系统存在病态或非唯一风险", checks, "残差、秩、条件数和伪逆复算共同决定结论等级。"),
        }

    @staticmethod
    def _solve_polynomial_root(relation: Mapping[str, Any]) -> Dict[str, Any]:
        from scipy.optimize import bisect, brentq

        coefficients = np.asarray(relation["coefficients"], dtype=float)
        left, right = (float(value) for value in relation["bracket"])
        function = lambda value: float(np.polyval(coefficients, value))
        primary = float(brentq(function, left, right, xtol=1e-12, rtol=1e-12, maxiter=200))
        confirmation = float(bisect(function, left, right, xtol=1e-11, rtol=1e-11, maxiter=300))
        residual = abs(function(primary))
        difference = abs(primary - confirmation) / max(1.0, abs(primary))
        passed = residual <= 1e-8 * max(1.0, float(np.max(np.abs(coefficients)))) and difference <= 1e-9
        checks = [
            {"id": "polynomial_residual", "name": "多项式残差", "status": "pass" if residual <= 1e-8 else "fail", "evidence": f"absolute_residual={residual:.3g}"},
            {"id": "independent_bisection", "name": "二分法独立复算", "status": "pass" if difference <= 1e-9 else "fail", "evidence": f"relative_difference={difference:.3g}"},
        ]
        return {
            "kind": "scalar_root_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "scipy.brentq+bisect", "variable": relation["variable"], "root": primary,
            "solution": {relation["variable"]: primary}, "absolute_residual": residual,
            "summary": {"root": primary, "absolute_residual": residual},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": difference, "acceptance_tolerance": 1e-9},
            "credibility_audit": _audit("pass" if passed else "fail", "括区间根通过独立复算" if passed else "求根复算失败", checks, "结论只覆盖给定括区间内的这个根，不证明已找到全部根。"),
        }

    @staticmethod
    def _solve_linear_least_squares(relation: Mapping[str, Any]) -> Dict[str, Any]:
        from scipy.optimize import lsq_linear

        design = np.asarray(relation["design_matrix"], dtype=float)
        observed = np.asarray(relation["observations"], dtype=float)
        variables = list(relation["variables"])
        raw_bounds = relation.get("bounds")
        if raw_bounds is None:
            solution = np.linalg.lstsq(design, observed, rcond=None)[0]
            confirmation = np.linalg.pinv(design) @ observed
            solver = "numpy.lstsq"
        else:
            lower = np.asarray([pair[0] for pair in raw_bounds], dtype=float)
            upper = np.asarray([pair[1] for pair in raw_bounds], dtype=float)
            primary = lsq_linear(design, observed, bounds=(lower, upper), tol=1e-12, max_iter=1000)
            confirm = lsq_linear(design, observed, bounds=(lower, upper), method="bvls", tol=1e-11, max_iter=2000)
            if not primary.success or not confirm.success:
                raise RuntimeError("bounded least-squares solver did not converge")
            solution, confirmation = primary.x, confirm.x
            solver = "scipy.lsq_linear.trf+bvls"
        residual = design @ solution - observed
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        relative_residual = float(np.linalg.norm(residual) / max(1.0, np.linalg.norm(observed)))
        difference = float(np.linalg.norm(solution - confirmation) / max(1.0, np.linalg.norm(solution)))
        rank, condition = int(np.linalg.matrix_rank(design)), float(np.linalg.cond(design))
        stable = difference <= 1e-7 and condition <= 1e12 and rank == design.shape[1]
        checks = [
            {"id": "least_squares_residual", "name": "最小二乘残差", "status": "pass", "evidence": f"rmse={rmse:.6g}, relative_residual={relative_residual:.3g}"},
            {"id": "solver_confirmation", "name": "独立算法复算", "status": "pass" if difference <= 1e-7 else "warning", "evidence": f"relative_solution_difference={difference:.3g}"},
            {"id": "identifiability", "name": "参数可辨识性", "status": "pass" if rank == design.shape[1] and condition <= 1e12 else "warning", "evidence": f"rank={rank}, condition={condition:.3g}"},
        ]
        return {
            "kind": "linear_least_squares_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": solver, "variables": variables,
            "solution": {name: float(solution[index]) for index, name in enumerate(variables)},
            "rmse": rmse, "relative_residual": relative_residual, "rank": rank, "condition_number": condition,
            "summary": {"rmse": rmse, "relative_residual": relative_residual, "rank": rank, "condition_number": condition},
            "convergence": {"status": "pass" if difference <= 1e-7 else "warning", "relative_tolerance_comparison": difference, "acceptance_tolerance": 1e-7},
            "credibility_audit": _audit("pass" if stable else "warning", "线性标定稳定" if stable else "线性标定存在可辨识性风险", checks, "拟合残差小不自动证明参数具有因果或机理含义。"),
        }

    @staticmethod
    def _constraint_violation(relation: Mapping[str, Any], vector: np.ndarray) -> float:
        violations = []
        for matrix_name, vector_name, equality in (("A_ub", "b_ub", False), ("A_eq", "b_eq", True)):
            matrix = np.asarray(relation.get(matrix_name, []), dtype=float)
            rhs = np.asarray(relation.get(vector_name, []), dtype=float)
            if matrix.size:
                residual = matrix @ vector - rhs
                violations.append(float(np.max(np.abs(residual) if equality else np.maximum(residual, 0.0))))
        for value, (lower, upper) in zip(vector, relation["bounds"]):
            if lower is not None:
                violations.append(max(0.0, float(lower) - float(value)))
            if upper is not None:
                violations.append(max(0.0, float(value) - float(upper)))
        return max(violations, default=0.0)

    @classmethod
    def _solve_linear_program(cls, relation: Mapping[str, Any]) -> Dict[str, Any]:
        from scipy.optimize import linprog

        coefficients = np.asarray(relation["objective_coefficients"], dtype=float)
        maximize = relation["direction"] == "maximize"
        result = linprog(
            -coefficients if maximize else coefficients,
            A_ub=np.asarray(relation["A_ub"], dtype=float) if relation["A_ub"] else None,
            b_ub=np.asarray(relation["b_ub"], dtype=float) if relation["b_ub"] else None,
            A_eq=np.asarray(relation["A_eq"], dtype=float) if relation["A_eq"] else None,
            b_eq=np.asarray(relation["b_eq"], dtype=float) if relation["b_eq"] else None,
            bounds=[tuple(pair) for pair in relation["bounds"]], method="highs",
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"linear program did not return an optimum: {result.message}")
        objective = float(coefficients @ result.x)
        violation = cls._constraint_violation(relation, result.x)
        passed = violation <= 1e-8 and math.isfinite(objective)
        variables = list(relation["variables"])
        checks = [
            {"id": "linear_program_feasibility", "name": "线性规划可行性", "status": "pass" if violation <= 1e-8 else "fail", "evidence": f"maximum_violation={violation:.3g}"},
            {"id": "objective_recalculation", "name": "目标值复算", "status": "pass" if math.isfinite(objective) else "fail", "evidence": f"objective={objective:.12g}"},
            {"id": "highs_certificate_scope", "name": "HiGHS终止证书", "status": "pass", "evidence": str(result.message)},
        ]
        return {
            "kind": "linear_program_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "scipy.linprog.highs", "solver_success": True, "direction": relation["direction"],
            "solution": {name: float(result.x[index]) for index, name in enumerate(variables)},
            "objective_value": objective, "maximum_constraint_violation": violation,
            "summary": {"objective_value": objective, "maximum_constraint_violation": violation},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": violation, "acceptance_tolerance": 1e-8},
            "credibility_audit": _audit("pass" if passed else "fail", "线性规划最优解通过残差复算" if passed else "线性规划结果未通过复算", checks, "结论针对当前线性目标、约束和边界；未建模约束仍会使现实方案失效。"),
        }

    @classmethod
    def _solve_mixed_integer_linear_program(cls, relation: Mapping[str, Any]) -> Dict[str, Any]:
        from scipy.optimize import Bounds, LinearConstraint, milp

        coefficients = np.asarray(relation["objective_coefficients"], dtype=float)
        maximize = relation["direction"] == "maximize"
        matrices, lower_constraints, upper_constraints = [], [], []
        if relation["A_ub"]:
            matrix = np.asarray(relation["A_ub"], dtype=float)
            matrices.append(matrix)
            lower_constraints.extend([-np.inf] * matrix.shape[0])
            upper_constraints.extend(relation["b_ub"])
        if relation["A_eq"]:
            matrix = np.asarray(relation["A_eq"], dtype=float)
            matrices.append(matrix)
            lower_constraints.extend(relation["b_eq"])
            upper_constraints.extend(relation["b_eq"])
        constraints = None
        if matrices:
            constraints = LinearConstraint(np.vstack(matrices), np.asarray(lower_constraints), np.asarray(upper_constraints))
        lower = np.asarray([pair[0] for pair in relation["bounds"]], dtype=float)
        upper = np.asarray([pair[1] for pair in relation["bounds"]], dtype=float)
        result = milp(
            -coefficients if maximize else coefficients,
            integrality=np.asarray(relation["integrality"], dtype=int),
            bounds=Bounds(lower, upper), constraints=constraints,
            options={"time_limit": 45.0, "node_limit": 100_000, "mip_rel_gap": 1e-8},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"mixed-integer program did not return a certified candidate: {result.message}")
        objective = float(coefficients @ result.x)
        violation = cls._constraint_violation(relation, result.x)
        integrality_violation = max(
            (abs(float(value) - round(float(value))) for value, code in zip(result.x, relation["integrality"]) if code in {1, 3}),
            default=0.0,
        )
        passed = violation <= 1e-7 and integrality_violation <= 1e-7
        variables = list(relation["variables"])
        gap = getattr(result, "mip_gap", None)
        checks = [
            {"id": "milp_feasibility", "name": "MILP可行性", "status": "pass" if violation <= 1e-7 else "fail", "evidence": f"maximum_violation={violation:.3g}"},
            {"id": "integrality", "name": "整数性", "status": "pass" if integrality_violation <= 1e-7 else "fail", "evidence": f"maximum_integrality_violation={integrality_violation:.3g}"},
            {"id": "mip_gap", "name": "整数规划界差", "status": "pass" if gap is not None and float(gap) <= 1e-7 else "warning", "evidence": f"mip_gap={gap}"},
        ]
        audit_status = "pass" if passed and gap is not None and float(gap) <= 1e-7 else ("warning" if passed else "fail")
        return {
            "kind": "mixed_integer_linear_program_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "scipy.milp.highs", "solver_success": True, "direction": relation["direction"],
            "solution": {name: float(result.x[index]) for index, name in enumerate(variables)},
            "objective_value": objective, "maximum_constraint_violation": violation,
            "maximum_integrality_violation": integrality_violation, "mip_gap": None if gap is None else float(gap),
            "summary": {"objective_value": objective, "maximum_constraint_violation": violation, "integrality_violation": integrality_violation},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": max(violation, integrality_violation), "acceptance_tolerance": 1e-7},
            "credibility_audit": _audit(audit_status, "MILP方案通过可行性与整数性复算", checks, "只有足够小的MIP界差才能声称当前离散模型上的全局最优。"),
        }

    @classmethod
    def _solve_quadratic_program(cls, relation: Mapping[str, Any]) -> Dict[str, Any]:
        from scipy.optimize import minimize

        quadratic = np.asarray(relation["quadratic_matrix"], dtype=float)
        linear = np.asarray(relation["linear_coefficients"], dtype=float)
        maximize = relation["direction"] == "maximize"

        def raw_objective(vector: np.ndarray) -> float:
            return float(0.5 * vector @ quadratic @ vector + linear @ vector)

        def solver_objective(vector: np.ndarray) -> float:
            value = raw_objective(vector)
            return -value if maximize else value

        constraints = []
        if relation["A_ub"]:
            matrix, rhs = np.asarray(relation["A_ub"], dtype=float), np.asarray(relation["b_ub"], dtype=float)
            constraints.append({"type": "ineq", "fun": lambda x, a=matrix, b=rhs: b - a @ x})
        if relation["A_eq"]:
            matrix, rhs = np.asarray(relation["A_eq"], dtype=float), np.asarray(relation["b_eq"], dtype=float)
            constraints.append({"type": "eq", "fun": lambda x, a=matrix, b=rhs: a @ x - b})
        bounds = [tuple(pair) for pair in relation["bounds"]]
        lower = np.asarray([pair[0] for pair in bounds], dtype=float)
        upper = np.asarray([pair[1] for pair in bounds], dtype=float)
        rng = np.random.default_rng(0)
        starts = [(lower + upper) / 2.0]
        starts.extend(rng.uniform(lower, upper) for _ in range(5))
        feasible = []
        for start in starts:
            outcome = minimize(
                solver_objective, start, method="SLSQP", bounds=bounds,
                constraints=constraints, options={"maxiter": 2000, "ftol": 1e-12},
            )
            if outcome.x is None or not np.all(np.isfinite(outcome.x)):
                continue
            violation = cls._constraint_violation(relation, outcome.x)
            if outcome.success and violation <= 1e-7:
                feasible.append((outcome.x, raw_objective(outcome.x), violation))
        if not feasible:
            raise RuntimeError("convex quadratic program found no converged feasible solution")
        feasible.sort(key=lambda item: item[1], reverse=maximize)
        vector, objective, violation = feasible[0]
        spread = float(max(abs(value - objective) for _, value, _ in feasible) / max(1.0, abs(objective)))
        eigenvalues = np.linalg.eigvalsh(quadratic)
        convexity_pass = (
            float(np.min(eigenvalues)) >= -1e-10 if not maximize
            else float(np.max(eigenvalues)) <= 1e-10
        )
        passed = violation <= 1e-7 and spread <= 1e-7 and convexity_pass
        names = list(relation["variables"])
        checks = [
            {"id": "qp_convexity", "name": "二次目标凸性", "status": "pass" if convexity_pass else "fail", "evidence": f"eigenvalue_range=[{np.min(eigenvalues):.3g}, {np.max(eigenvalues):.3g}]"},
            {"id": "qp_feasibility", "name": "二次规划可行性", "status": "pass" if violation <= 1e-7 else "fail", "evidence": f"maximum_violation={violation:.3g}"},
            {"id": "qp_multistart", "name": "多起点一致性", "status": "pass" if spread <= 1e-7 else "warning", "evidence": f"successful={len(feasible)}/{len(starts)}, objective_spread={spread:.3g}"},
        ]
        return {
            "kind": "quadratic_program_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "scipy.SLSQP_convex_multistart", "direction": relation["direction"],
            "solution": {name: float(vector[index]) for index, name in enumerate(names)},
            "objective_value": float(objective), "maximum_constraint_violation": float(violation),
            "convexity_eigenvalue_range": [float(np.min(eigenvalues)), float(np.max(eigenvalues))],
            "successful_starts": len(feasible), "attempted_starts": len(starts),
            "summary": {"objective_value": float(objective), "maximum_constraint_violation": float(violation), "multistart_spread": spread},
            "convergence": {"status": "pass" if passed else "warning", "relative_tolerance_comparison": max(violation, spread), "acceptance_tolerance": 1e-7},
            "credibility_audit": _audit("pass" if passed else "warning", "凸二次规划候选通过多起点复算", checks, "全局性依赖已验证的半正定性和线性可行域，现实约束完整性仍需单独核验。"),
        }

    @staticmethod
    def _linprog_vector(relation: Mapping[str, Any], coefficients: np.ndarray) -> np.ndarray:
        from scipy.optimize import linprog

        outcome = linprog(
            coefficients,
            A_ub=np.asarray(relation["A_ub"], dtype=float) if relation["A_ub"] else None,
            b_ub=np.asarray(relation["b_ub"], dtype=float) if relation["b_ub"] else None,
            A_eq=np.asarray(relation["A_eq"], dtype=float) if relation["A_eq"] else None,
            b_eq=np.asarray(relation["b_eq"], dtype=float) if relation["b_eq"] else None,
            bounds=[tuple(pair) for pair in relation["bounds"]], method="highs",
        )
        if not outcome.success or outcome.x is None:
            raise RuntimeError(f"linear scalarization failed: {outcome.message}")
        return np.asarray(outcome.x, dtype=float)

    @classmethod
    def _solve_multiobjective_program(cls, relation: Mapping[str, Any]) -> Dict[str, Any]:
        objectives = list(relation["objectives"])
        coefficients = np.asarray([item["coefficients"] for item in objectives], dtype=float)
        signs = np.asarray([1.0 if item["direction"] == "minimize" else -1.0 for item in objectives])
        normalized_coefficients = coefficients * signs[:, None]
        count = len(objectives)
        weights = [np.eye(count)[index] for index in range(count)]
        weights.append(np.ones(count) / count)
        for left in range(count):
            for right in range(left + 1, count):
                weight = np.zeros(count)
                weight[left] = weight[right] = 0.5
                weights.append(weight)
        rng = np.random.default_rng(0)
        weights.extend(rng.dirichlet(np.ones(count), size=min(32, 8 * count)))
        candidates = []
        seen = set()
        for weight in weights:
            vector = cls._linprog_vector(relation, weight @ normalized_coefficients)
            signature = tuple(np.round(vector, 10))
            if signature in seen:
                continue
            seen.add(signature)
            raw_values = coefficients @ vector
            normalized_values = raw_values * signs
            candidates.append((vector, raw_values, normalized_values))
        nondominated = []
        for index, candidate in enumerate(candidates):
            current = candidate[2]
            dominated = any(
                other_index != index
                and np.all(other[2] <= current + 1e-9)
                and np.any(other[2] < current - 1e-9)
                for other_index, other in enumerate(candidates)
            )
            if not dominated:
                nondominated.append(candidate)
        names = list(relation["variables"])
        front = [{
            "solution": {name: float(vector[index]) for index, name in enumerate(names)},
            "objectives": {item["name"]: float(values[index]) for index, item in enumerate(objectives)},
        } for vector, values, _ in nondominated]
        checks = [
            {"id": "pareto_nondominance", "name": "非支配复算", "status": "pass", "evidence": f"candidates={len(candidates)}, nondominated={len(front)}"},
            {"id": "pareto_coverage", "name": "Pareto覆盖边界", "status": "warning", "evidence": "加权和标量化只能给出采样的受支持非支配点，非凸前沿可能不完整。"},
        ]
        return {
            "kind": "multiobjective_program_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "HiGHS_weighted_scalarization", "pareto_front": front,
            "candidate_count": len(candidates), "nondominated_count": len(front),
            "summary": {"candidate_count": len(candidates), "nondominated_count": len(front)},
            "convergence": {"status": "warning", "relative_tolerance_comparison": 0.0, "acceptance_tolerance": 1e-8},
            "credibility_audit": _audit("warning", "已得到有限采样Pareto候选集", checks, "可以比较当前非支配候选，但不能声称枚举了完整Pareto前沿。"),
        }

    @classmethod
    def _solve_robust_program(cls, relation: Mapping[str, Any]) -> Dict[str, Any]:
        from scipy.optimize import linprog

        scenarios = np.asarray(relation["scenario_objective_coefficients"], dtype=float)
        n = scenarios.shape[1]
        minimize = relation["direction"] == "minimize"
        objective = np.zeros(n + 1)
        objective[-1] = 1.0 if minimize else -1.0
        a_ub = []
        b_ub = []
        if relation["A_ub"]:
            a_ub.extend(np.column_stack([np.asarray(relation["A_ub"], dtype=float), np.zeros(len(relation["A_ub"]))]))
            b_ub.extend(relation["b_ub"])
        for scenario in scenarios:
            a_ub.append(np.r_[scenario if minimize else -scenario, -1.0 if minimize else 1.0])
            b_ub.append(0.0)
        a_eq = np.column_stack([np.asarray(relation["A_eq"], dtype=float), np.zeros(len(relation["A_eq"]))]) if relation["A_eq"] else None
        outcome = linprog(
            objective, A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub),
            A_eq=a_eq, b_eq=np.asarray(relation["b_eq"]) if relation["b_eq"] else None,
            bounds=[tuple(pair) for pair in relation["bounds"]] + [(None, None)], method="highs",
        )
        if not outcome.success or outcome.x is None:
            raise RuntimeError(f"robust linear program failed: {outcome.message}")
        vector, epigraph = outcome.x[:n], float(outcome.x[-1])
        values = scenarios @ vector
        worst = float(np.max(values) if minimize else np.min(values))
        epigraph_gap = abs(worst - epigraph)
        violation = cls._constraint_violation(relation, vector)
        passed = epigraph_gap <= 1e-8 and violation <= 1e-8
        names = list(relation["variables"])
        checks = [
            {"id": "robust_epigraph", "name": "最坏情景上图复算", "status": "pass" if epigraph_gap <= 1e-8 else "fail", "evidence": f"worst={worst}, epigraph={epigraph}, gap={epigraph_gap:.3g}"},
            {"id": "robust_feasibility", "name": "公共约束可行性", "status": "pass" if violation <= 1e-8 else "fail", "evidence": f"maximum_violation={violation:.3g}"},
        ]
        return {
            "kind": "robust_program_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "scipy.linprog.epigraph_robust", "direction": relation["direction"],
            "solution": {name: float(vector[index]) for index, name in enumerate(names)},
            "scenario_objective_values": values.tolist(), "worst_case_objective": worst,
            "maximum_constraint_violation": violation,
            "summary": {"worst_case_objective": worst, "epigraph_gap": epigraph_gap, "maximum_constraint_violation": violation},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": max(epigraph_gap, violation), "acceptance_tolerance": 1e-8},
            "credibility_audit": _audit("pass" if passed else "fail", "显式情景鲁棒解通过最坏值复算", checks, "鲁棒性只覆盖用户明确给出的有限情景，不覆盖情景集合之外的不确定性。"),
        }

    @classmethod
    def _solve_stochastic_program(cls, relation: Mapping[str, Any]) -> Dict[str, Any]:
        scenarios = np.asarray(relation["scenario_objective_coefficients"], dtype=float)
        probabilities = np.asarray(relation["probabilities"], dtype=float)
        expected_coefficients = probabilities @ scenarios
        minimize = relation["direction"] == "minimize"
        vector = cls._linprog_vector(
            relation, expected_coefficients if minimize else -expected_coefficients
        )
        scenario_values = scenarios @ vector
        expected_value = float(probabilities @ scenario_values)
        direct_value = float(expected_coefficients @ vector)
        gap = abs(expected_value - direct_value)
        violation = cls._constraint_violation(relation, vector)
        passed = gap <= 1e-10 and violation <= 1e-8
        names = list(relation["variables"])
        checks = [
            {"id": "probability_sum", "name": "情景概率归一", "status": "pass", "evidence": f"probability_sum={probabilities.sum():.12g}"},
            {"id": "expected_objective", "name": "期望目标复算", "status": "pass" if gap <= 1e-10 else "fail", "evidence": f"expected={expected_value}, coefficient_form={direct_value}"},
            {"id": "stochastic_feasibility", "name": "公共约束可行性", "status": "pass" if violation <= 1e-8 else "fail", "evidence": f"maximum_violation={violation:.3g}"},
        ]
        return {
            "kind": "stochastic_program_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "scipy.linprog.expected_scenario_objective", "direction": relation["direction"],
            "solution": {name: float(vector[index]) for index, name in enumerate(names)},
            "scenario_objective_values": scenario_values.tolist(), "expected_objective": expected_value,
            "maximum_constraint_violation": violation,
            "summary": {"expected_objective": expected_value, "scenario_minimum": float(np.min(scenario_values)), "scenario_maximum": float(np.max(scenario_values)), "maximum_constraint_violation": violation},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": max(gap, violation), "acceptance_tolerance": 1e-8},
            "credibility_audit": _audit("pass" if passed else "fail", "显式情景期望规划通过复算", checks, "期望最优依赖情景概率可信；它不自动控制尾部风险或最坏情形。"),
        }

    @staticmethod
    def _solve_dynamic_program(relation: Mapping[str, Any]) -> Dict[str, Any]:
        transition = np.asarray(relation["transition_probabilities"], dtype=float)
        stage = np.asarray(relation["stage_values"], dtype=float)
        terminal = np.asarray(relation["terminal_values"], dtype=float)
        horizon, n_states, _ = stage.shape
        maximize = relation["direction"] == "maximize"
        values = np.empty((horizon + 1, n_states), dtype=float)
        policy = np.empty((horizon, n_states), dtype=int)
        values[-1] = terminal
        bellman_residual = 0.0
        for time_index in range(horizon - 1, -1, -1):
            action_values = stage[time_index] + np.einsum(
                "sak,k->sa", transition[time_index], values[time_index + 1]
            )
            selected = np.argmax(action_values, axis=1) if maximize else np.argmin(action_values, axis=1)
            policy[time_index] = selected
            values[time_index] = action_values[np.arange(n_states), selected]
            replay = stage[time_index, np.arange(n_states), selected] + np.sum(
                transition[time_index, np.arange(n_states), selected] * values[time_index + 1], axis=1
            )
            bellman_residual = max(bellman_residual, float(np.max(np.abs(replay - values[time_index]))))
        state_names, action_names = relation["states"], relation["actions"]
        policy_rows = [
            {"time": time_index, "actions": {state: action_names[int(policy[time_index, state_index])] for state_index, state in enumerate(state_names)}}
            for time_index in range(horizon)
        ]
        initial_state = relation.get("initial_state")
        initial_value = None if initial_state is None else float(values[0, state_names.index(initial_state)])
        passed = bellman_residual <= 1e-10
        checks = [
            {"id": "bellman_replay", "name": "Bellman递推复算", "status": "pass" if passed else "fail", "evidence": f"maximum_residual={bellman_residual:.3g}"},
            {"id": "policy_domain", "name": "策略动作域", "status": "pass", "evidence": f"horizon={horizon}, states={n_states}, actions={len(action_names)}"},
        ]
        return {
            "kind": "dynamic_program_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "backward_bellman_recursion", "direction": relation["direction"],
            "policy": policy_rows, "value_table": values.tolist(), "initial_state_value": initial_value,
            "summary": {"initial_state_value": initial_value, "bellman_residual": bellman_residual, "horizon": horizon},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": bellman_residual, "acceptance_tolerance": 1e-10},
            "credibility_audit": _audit("pass" if passed else "fail", "有限时域动态规划通过Bellman复算", checks, "策略最优性只适用于已给定状态、动作、转移概率、阶段价值和有限时域。"),
        }

    @staticmethod
    def _graph(relation: Mapping[str, Any], value_field: str) -> Any:
        import networkx as nx

        graph = nx.DiGraph() if relation.get("directed", True) else nx.Graph()
        for edge in relation["edges"]:
            graph.add_edge(edge["source"], edge["target"], **{value_field: float(edge[value_field])})
        return graph

    @classmethod
    def _solve_shortest_path(cls, relation: Mapping[str, Any]) -> Dict[str, Any]:
        import networkx as nx

        graph = cls._graph(relation, "weight")
        source, target = relation["source_node"], relation["target_node"]
        has_negative = any(data["weight"] < 0 for _, _, data in graph.edges(data=True))
        method = "bellman-ford" if has_negative else "dijkstra"
        path = nx.shortest_path(graph, source, target, weight="weight", method=method)
        length = float(nx.path_weight(graph, path, weight="weight"))
        recomputed = float(sum(graph[left][right]["weight"] for left, right in zip(path, path[1:])))
        passed = path[0] == source and path[-1] == target and abs(length - recomputed) <= 1e-10
        checks = [
            {"id": "path_endpoints", "name": "路径端点", "status": "pass" if path[0] == source and path[-1] == target else "fail", "evidence": f"path={path}"},
            {"id": "path_weight_recalculation", "name": "路径权重复算", "status": "pass" if abs(length - recomputed) <= 1e-10 else "fail", "evidence": f"reported={length}, recomputed={recomputed}"},
        ]
        return {
            "kind": "shortest_path_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": f"networkx.{method}", "path": path, "path_length": length,
            "summary": {"path_length": length, "edge_count": len(path) - 1},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": abs(length - recomputed), "acceptance_tolerance": 1e-10},
            "credibility_audit": _audit("pass" if passed else "fail", "最短路径通过逐边复算", checks, "最短性只针对当前节点、边、方向和权重定义。"),
        }

    @classmethod
    def _solve_maximum_flow(cls, relation: Mapping[str, Any]) -> Dict[str, Any]:
        import networkx as nx

        graph = cls._graph(relation, "capacity")
        source, sink = relation["source_node"], relation["sink_node"]
        value, flow = nx.maximum_flow(graph, source, sink, capacity="capacity")
        cut_value, cut_partition = nx.minimum_cut(graph, source, sink, capacity="capacity")
        capacity_violation, conservation_violation = 0.0, 0.0
        for left, targets in flow.items():
            for right, amount in targets.items():
                capacity = float(graph[left][right]["capacity"])
                capacity_violation = max(capacity_violation, max(0.0, -amount, amount - capacity))
        for node in graph.nodes:
            if node in {source, sink}:
                continue
            inflow = sum(flow[left].get(node, 0.0) for left in graph.predecessors(node))
            outflow = sum(flow[node].values())
            conservation_violation = max(conservation_violation, abs(inflow - outflow))
        gap = abs(float(value) - float(cut_value))
        passed = max(capacity_violation, conservation_violation, gap) <= 1e-8
        checks = [
            {"id": "capacity", "name": "容量约束", "status": "pass" if capacity_violation <= 1e-8 else "fail", "evidence": f"maximum_violation={capacity_violation:.3g}"},
            {"id": "flow_conservation", "name": "流量守恒", "status": "pass" if conservation_violation <= 1e-8 else "fail", "evidence": f"maximum_violation={conservation_violation:.3g}"},
            {"id": "max_flow_min_cut", "name": "最大流最小割复核", "status": "pass" if gap <= 1e-8 else "fail", "evidence": f"flow={value}, cut={cut_value}"},
        ]
        return {
            "kind": "maximum_flow_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "networkx.preflow_push+minimum_cut", "maximum_flow": float(value), "flow": flow,
            "minimum_cut_partition": [sorted(cut_partition[0]), sorted(cut_partition[1])],
            "summary": {"maximum_flow": float(value), "cut_value": float(cut_value), "maximum_violation": max(capacity_violation, conservation_violation)},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": max(capacity_violation, conservation_violation, gap), "acceptance_tolerance": 1e-8},
            "credibility_audit": _audit("pass" if passed else "fail", "最大流通过最小割证书复核", checks, "证书只适用于当前有向容量网络。"),
        }

    @staticmethod
    def _solve_minimum_cost_flow(relation: Mapping[str, Any]) -> Dict[str, Any]:
        import networkx as nx

        graph = nx.DiGraph()
        for node in relation["nodes"]:
            graph.add_node(node, demand=float(relation["node_demands"][node]))
        for edge in relation["edges"]:
            graph.add_edge(
                edge["source"], edge["target"],
                capacity=float(edge["capacity"]), weight=float(edge["cost"]),
            )
        cost, flow = nx.network_simplex(graph, demand="demand", capacity="capacity", weight="weight")
        recomputed_cost = 0.0
        capacity_violation = 0.0
        balance_violation = 0.0
        for source, targets in flow.items():
            for target, amount in targets.items():
                edge = graph[source][target]
                recomputed_cost += float(amount) * float(edge["weight"])
                capacity_violation = max(
                    capacity_violation,
                    max(0.0, -float(amount), float(amount) - float(edge["capacity"])),
                )
        for node in graph.nodes:
            inflow = sum(float(flow[source].get(node, 0.0)) for source in graph.predecessors(node))
            outflow = sum(float(value) for value in flow[node].values())
            balance_violation = max(
                balance_violation,
                abs((inflow - outflow) - float(graph.nodes[node]["demand"])),
            )
        cost_gap = abs(float(cost) - recomputed_cost)
        passed = max(capacity_violation, balance_violation, cost_gap) <= 1e-8
        checks = [
            {"id": "flow_cost", "name": "费用逐边复算", "status": "pass" if cost_gap <= 1e-8 else "fail", "evidence": f"reported={cost}, recomputed={recomputed_cost}"},
            {"id": "flow_capacity", "name": "费用流容量", "status": "pass" if capacity_violation <= 1e-8 else "fail", "evidence": f"maximum_violation={capacity_violation:.3g}"},
            {"id": "flow_demand_balance", "name": "节点供需平衡", "status": "pass" if balance_violation <= 1e-8 else "fail", "evidence": f"maximum_violation={balance_violation:.3g}"},
        ]
        return {
            "kind": "minimum_cost_flow_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "networkx.network_simplex", "minimum_cost": float(cost), "flow": flow,
            "summary": {"minimum_cost": float(cost), "cost_recalculation_gap": cost_gap, "maximum_capacity_violation": capacity_violation, "maximum_balance_violation": balance_violation},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": max(capacity_violation, balance_violation, cost_gap), "acceptance_tolerance": 1e-8},
            "credibility_audit": _audit("pass" if passed else "fail", "最小费用流通过费用、容量与供需复算", checks, "最优性只针对当前网络、容量、单位费用和节点供需。"),
        }

    @staticmethod
    def _solve_bipartite_matching(relation: Mapping[str, Any]) -> Dict[str, Any]:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(relation["left_nodes"], bipartite=0)
        graph.add_nodes_from(relation["right_nodes"], bipartite=1)
        for edge in relation["edges"]:
            graph.add_edge(edge["source"], edge["target"], weight=float(edge["weight"]))
        weighted = bool(relation.get("maximize_weight", False))
        if weighted:
            raw_pairs = nx.max_weight_matching(graph, maxcardinality=bool(relation.get("max_cardinality_first", True)), weight="weight")
            pairs = [tuple(pair) for pair in raw_pairs]
        else:
            matching = nx.algorithms.bipartite.maximum_matching(graph, top_nodes=set(relation["left_nodes"]))
            pairs = [(left, matching[left]) for left in relation["left_nodes"] if left in matching]
        normalized = sorted(
            ([left, right] if left in relation["left_nodes"] else [right, left] for left, right in pairs),
            key=lambda item: (item[0], item[1]),
        )
        unique_nodes = len({node for pair in normalized for node in pair}) == 2 * len(normalized)
        all_edges = all(graph.has_edge(left, right) for left, right in normalized)
        total_weight = float(sum(graph[left][right]["weight"] for left, right in normalized))
        passed = unique_nodes and all_edges
        checks = [
            {"id": "matching_uniqueness", "name": "匹配节点唯一性", "status": "pass" if unique_nodes else "fail", "evidence": f"pairs={len(normalized)}"},
            {"id": "matching_edge_membership", "name": "匹配边存在性", "status": "pass" if all_edges else "fail", "evidence": "all returned pairs are input edges"},
        ]
        return {
            "kind": "bipartite_matching_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "networkx.bipartite_matching", "matching": normalized,
            "matching_size": len(normalized), "total_weight": total_weight,
            "summary": {"matching_size": len(normalized), "total_weight": total_weight},
            "convergence": {"status": "pass" if passed else "fail", "relative_tolerance_comparison": 0.0 if passed else 1.0, "acceptance_tolerance": 0.0},
            "credibility_audit": _audit("pass" if passed else "fail", "二部匹配通过结构复核", checks, "匹配最优性针对当前二部分区、边集和权重规则。"),
        }

    @staticmethod
    def _solve_markov_chain(relation: Mapping[str, Any]) -> Dict[str, Any]:
        transition = np.asarray(relation["transition_matrix"], dtype=float)
        initial = np.asarray(relation["initial_distribution"], dtype=float)
        distribution = initial @ np.linalg.matrix_power(transition, int(relation["steps"]))
        mass_error = abs(float(distribution.sum()) - 1.0)
        values, vectors = np.linalg.eig(transition.T)
        stationary_index = int(np.argmin(np.abs(values - 1.0)))
        stationary = np.real(vectors[:, stationary_index])
        if stationary.sum() < 0:
            stationary = -stationary
        stationary = np.maximum(stationary, 0.0)
        stationary = stationary / stationary.sum()
        stationary_residual = float(np.linalg.norm(stationary @ transition - stationary, ord=1))
        multiplicity = int(np.sum(np.abs(values - 1.0) <= 1e-8))
        stable = mass_error <= 1e-10 and stationary_residual <= 1e-8 and multiplicity == 1
        labels = relation["state_labels"]
        checks = [
            {"id": "probability_mass", "name": "概率质量守恒", "status": "pass" if mass_error <= 1e-10 else "fail", "evidence": f"mass_error={mass_error:.3g}"},
            {"id": "stationarity", "name": "平稳分布残差", "status": "pass" if stationary_residual <= 1e-8 else "fail", "evidence": f"L1_residual={stationary_residual:.3g}"},
            {"id": "stationary_uniqueness", "name": "平稳分布唯一性", "status": "pass" if multiplicity == 1 else "warning", "evidence": f"unit_eigenvalue_multiplicity={multiplicity}"},
        ]
        return {
            "kind": "markov_chain_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "numpy.matrix_power+eigendecomposition", "steps": relation["steps"],
            "distribution": {label: float(distribution[index]) for index, label in enumerate(labels)},
            "stationary_distribution": {label: float(stationary[index]) for index, label in enumerate(labels)},
            "summary": {"mass_error": mass_error, "stationarity_residual": stationary_residual, "stationary_multiplicity": multiplicity},
            "convergence": {"status": "pass" if mass_error <= 1e-10 and stationary_residual <= 1e-8 else "fail", "relative_tolerance_comparison": max(mass_error, stationary_residual), "acceptance_tolerance": 1e-8},
            "credibility_audit": _audit("pass" if stable else "warning", "马尔可夫传播与平稳性已复核" if stable else "链可能不可约性不足或平稳分布不唯一", checks, "长期解释还需要检查不可约性、周期性和转移概率稳定性。"),
        }

    @staticmethod
    def _solve_sample_expectation(relation: Mapping[str, Any]) -> Dict[str, Any]:
        values = np.asarray(relation["values"], dtype=float)
        weights = np.asarray(relation["weights"], dtype=float)
        weights = weights / weights.sum()
        means = weights @ values
        centered = values - means
        variances = weights @ (centered ** 2)
        effective_n = float(1.0 / np.sum(weights ** 2))
        standard_errors = np.sqrt(variances / max(1.0, effective_n))
        lower, upper = means - 1.96 * standard_errors, means + 1.96 * standard_errors
        midpoint = values.shape[0] // 2
        first, second = values[:midpoint].mean(axis=0), values[midpoint:].mean(axis=0)
        split_difference = float(np.max(np.abs(first - second) / np.maximum(1.0, np.abs(means))))
        stable = split_difference <= 0.1
        names = relation["quantity_names"]
        checks = [
            {"id": "weight_normalization", "name": "权重归一化", "status": "pass", "evidence": f"normalized_sum={weights.sum():.12g}, effective_n={effective_n:.3g}"},
            {"id": "finite_moments", "name": "有限样本矩", "status": "pass", "evidence": f"outputs={values.shape[1]}, samples={values.shape[0]}"},
            {"id": "split_sample_stability", "name": "分半稳定性", "status": "pass" if stable else "warning", "evidence": f"maximum_relative_mean_shift={split_difference:.3g}"},
        ]
        return {
            "kind": "sample_expectation_solution", "status": "executed", "relation_id": relation.get("id"),
            "solver": "weighted_sample_moments", "effective_sample_size": effective_n,
            "expectation": {name: float(means[index]) for index, name in enumerate(names)},
            "standard_error": {name: float(standard_errors[index]) for index, name in enumerate(names)},
            "confidence_interval_95": {name: [float(lower[index]), float(upper[index])] for index, name in enumerate(names)},
            "summary": {"effective_sample_size": effective_n, "maximum_split_shift": split_difference, **{f"mean_{name}": float(means[index]) for index, name in enumerate(names)}},
            "convergence": {"status": "pass" if stable else "warning", "relative_tolerance_comparison": split_difference, "acceptance_tolerance": 0.1},
            "credibility_audit": _audit("pass" if stable else "warning", "样本期望与分半稳定性已复核" if stable else "样本期望对样本分半敏感", checks, "区间只表达当前加权样本的不确定性，不包含模型结构与抽样偏差。"),
        }
