"""Four-layer, structure-driven mathematical modeling contracts.

The module deliberately does not parse contest titles and does not import any
domain solver.  It turns already extracted evidence into four independently
auditable layers:

1. a semantic problem contract;
2. a normalized mathematical intermediate representation;
3. a resource-bounded solver plan selected by mathematical structure; and
4. an independent audit of the numerical results.

Domain adapters may produce normalized nodes, but they cannot select a solver
by problem name.  This keeps reusable mathematics separate from wording and
makes unsupported structures explicit instead of silently fabricating answers.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


EXECUTABLE_PARSE_STATES = frozenset({"machine_verified", "machine_compiled"})
_SUBPROBLEM = re.compile(
    r"(?<![0-9A-Za-z_\u4e00-\u9fff])(?:问题|任务|小问|Problem|Task)\s*"
    r"([一二三四五六七八九十]+|\d+(?!\.\d))\s*[：:.、]?",
    re.IGNORECASE,
)


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(normalized)
    return output


def _finite_scalar(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_payload(value: Any, *, depth: int = 0) -> bool:
    """Check result summaries without recursively walking huge plot arrays."""
    if depth > 8:
        return True
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return _finite_scalar(value)
    if isinstance(value, Mapping):
        return all(_finite_payload(item, depth=depth + 1) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return all(_finite_payload(item, depth=depth + 1) for item in list(value)[:500])
    return True


def _interval_union_measure(intervals: Sequence[Sequence[Any]]) -> Tuple[float, List[List[float]]]:
    normalized = []
    for interval in intervals:
        if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)):
            raise ValueError("interval must be a two-value sequence")
        if len(interval) != 2:
            raise ValueError("interval must contain exactly two endpoints")
        start, end = float(interval[0]), float(interval[1])
        if not (math.isfinite(start) and math.isfinite(end) and end >= start):
            raise ValueError("interval endpoints must be finite and ordered")
        normalized.append([start, end])
    normalized.sort(key=lambda item: (item[0], item[1]))
    merged: List[List[float]] = []
    for start, end in normalized:
        if not merged or start > merged[-1][1] + 1e-12:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged), merged


@dataclass(frozen=True)
class SolverSpecification:
    mathematical_form: str
    executor_key: str
    solver_family: str
    verification_strategy: Tuple[str, ...]
    max_variables: int
    max_evaluations: int
    timeout_seconds: int

    def public(self) -> Dict[str, Any]:
        return {
            "mathematical_form": self.mathematical_form,
            "executor_key": self.executor_key,
            "solver_family": self.solver_family,
            "verification_strategy": list(self.verification_strategy),
            "resource_budget": {
                "max_variables": self.max_variables,
                "max_evaluations": self.max_evaluations,
                "wall_time_budget_seconds": self.timeout_seconds,
                "enforcement": (
                    "hard contract-size and solver-iteration caps; wall-clock value is advisory "
                    "unless the selected backend exposes a native time limit"
                ),
            },
        }


@dataclass(frozen=True)
class MathematicalStructureDefinition:
    """A question-independent mathematical form and its optional backend."""

    key: str
    family: str
    contract_type: str
    relation_kinds: Tuple[str, ...]
    description: str
    solver: Optional[SolverSpecification] = None

    def public(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "family": self.family,
            "contract_type": self.contract_type,
            "relation_kinds": list(self.relation_kinds),
            "description": self.description,
            "execution_status": "implemented" if self.solver else "recognized_only",
            "solver": self.solver.public() if self.solver else None,
        }


class MathematicalStructureRegistry:
    """Versioned catalog of common mathematical forms, not contest templates."""

    schema_version = "mathmodel.structure-registry/v1"
    _TRIGGERS = {
        "linear_system": (r"线性方程组|矩阵方程|Ax\s*=\s*b|linear\s+system",),
        "scalar_polynomial_root": (r"多项式.{0,8}(?:根|零点)|polynomial\s+root",),
        "nonlinear_system": (r"非线性方程组|nonlinear\s+(?:equation|system)",),
        "linear_least_squares": (r"线性最小二乘|least\s+squares|线性拟合",),
        "nonlinear_least_squares": (r"非线性.{0,8}(?:拟合|标定|参数估计)|nonlinear\s+least",),
        "initial_value_problem": (r"初值问题|常微分方程|ODE|initial.value",),
        "boundary_value_problem": (r"边值问题|boundary.value",),
        "differential_algebraic_system": (r"微分代数|DAE",),
        "partial_differential_system": (r"偏微分|PDE|扩散方程|传热方程|波动方程",),
        "continuous_event_measure": (r"事件.{0,8}(?:检测|时长|区间)|持续时间|event.{0,8}(?:duration|measure)",),
        "linear_program": (r"线性规划|linear\s+program",),
        "mixed_integer_linear_program": (r"整数规划|混合整数|MILP|integer\s+program",),
        "hierarchical_finite_action_program": (
            r"(?:层级|上下层|父子).{0,16}(?:选择|配置|覆盖|分配)|"
            r"(?:选择|启用|保留).{0,16}(?:数量|个数).{0,12}(?:覆盖|满足)",
        ),
        "quadratic_program": (r"二次规划|quadratic\s+program",),
        "bounded_nonlinear_program": (r"非线性规划|非凸优化|nonlinear\s+program",),
        "multiobjective_program": (r"多目标|Pareto|帕累托|multi.?objective",),
        "robust_program": (r"鲁棒优化|最坏情形|robust\s+(?:program|optimization)",),
        "stochastic_program": (r"随机规划|机会约束|stochastic\s+program",),
        "dynamic_program": (r"动态规划|Bellman|贝尔曼|dynamic\s+program",),
        "optimal_control": (r"最优控制|控制泛函|optimal\s+control",),
        "shortest_path": (r"最短路|最短路径|shortest\s+path",),
        "maximum_flow": (r"最大流|max(?:imum)?\s+flow",),
        "minimum_cost_flow": (r"最小费用流|minimum.cost\s+flow",),
        "bipartite_matching": (r"二部图匹配|指派问题|bipartite\s+matching|assignment\s+problem",),
        "markov_chain": (r"马尔可夫链|Markov\s+chain|转移矩阵",),
        "sample_expectation": (r"蒙特卡洛.{0,8}(?:期望|均值)|样本期望|Monte\s+Carlo",),
        "discrete_event_simulation": (r"离散事件仿真|排队仿真|discrete.event\s+simulation",),
    }
    _REQUIRED_FIELDS = {
        "linear_system": ("variables", "coefficient_matrix", "right_hand_side", "units"),
        "scalar_polynomial_root": ("variable", "coefficients", "bracket", "units"),
        "linear_least_squares": ("variables", "design_matrix", "observations", "units"),
        "initial_value_problem": (
            "state_variables", "rhs", "initial_values", "parameters",
            "time_variable", "time_span", "units",
        ),
        "continuous_event_measure": ("state_trajectory", "event_condition", "time_window"),
        "linear_program": (
            "variables", "objective_coefficients", "direction", "A_ub", "b_ub",
            "A_eq", "b_eq", "bounds", "units",
        ),
        "mixed_integer_linear_program": (
            "variables", "objective_coefficients", "direction", "A_ub", "b_ub",
            "A_eq", "b_eq", "bounds", "integrality", "units",
        ),
        "hierarchical_finite_action_program": (
            "actions", "coverage_requirements", "active_count_bounds",
        ),
        "quadratic_program": (
            "variables", "quadratic_matrix", "linear_coefficients", "direction",
            "A_ub", "b_ub", "A_eq", "b_eq", "bounds", "units",
        ),
        "bounded_nonlinear_program": (
            "decision_variables", "objective", "direction", "constraints", "bounds",
            "initial_values", "parameters", "units",
        ),
        "multiobjective_program": (
            "variables", "objectives", "A_ub", "b_ub", "A_eq", "b_eq",
            "bounds", "units",
        ),
        "robust_program": (
            "variables", "scenario_objective_coefficients", "direction", "A_ub",
            "b_ub", "A_eq", "b_eq", "bounds", "units",
        ),
        "stochastic_program": (
            "variables", "scenario_objective_coefficients", "probabilities", "direction",
            "A_ub", "b_ub", "A_eq", "b_eq", "bounds", "units",
        ),
        "dynamic_program": ("states", "actions", "transition_probabilities", "stage_values", "horizon"),
        "shortest_path": ("edges", "source_node", "target_node", "directed"),
        "maximum_flow": ("edges", "source_node", "sink_node", "directed"),
        "minimum_cost_flow": ("nodes", "node_demands", "edges"),
        "bipartite_matching": ("left_nodes", "right_nodes", "edges"),
        "markov_chain": ("transition_matrix", "initial_distribution", "steps"),
        "sample_expectation": ("values", "weights", "quantity_names", "units"),
    }

    def __init__(self) -> None:
        def solver(
            form: str, executor: str, family: str, checks: Tuple[str, ...],
            variables: int, evaluations: int, seconds: int,
        ) -> SolverSpecification:
            return SolverSpecification(
                form, executor, family, checks, variables, evaluations, seconds,
            )

        self._definitions: Tuple[MathematicalStructureDefinition, ...] = (
            MathematicalStructureDefinition(
                "linear_system", "algebraic_system", "linear_system/v1",
                ("linear_system",), "finite linear equation system",
                solver("linear_system", "linear_system/v1", "rank-aware direct or least-squares linear algebra",
                       ("rank_check", "residual_recalculation", "pseudoinverse_confirmation"), 500, 2_000_000, 20),
            ),
            MathematicalStructureDefinition(
                "scalar_polynomial_root", "algebraic_system", "polynomial_root/v1",
                ("polynomial_root",), "bracketed scalar polynomial root",
                solver("scalar_polynomial_root", "polynomial_root/v1", "bracketed scalar root refinement",
                       ("bracket_check", "residual_recalculation", "independent_bisection"), 1, 20_000, 10),
            ),
            MathematicalStructureDefinition(
                "nonlinear_system", "algebraic_system", "nonlinear_system/v1",
                ("nonlinear_system",), "finite nonlinear equation system"),
            MathematicalStructureDefinition(
                "linear_least_squares", "inverse_problem", "linear_least_squares/v1",
                ("linear_least_squares",), "linear least-squares estimation with optional bounds",
                solver("linear_least_squares", "linear_least_squares/v1", "rank-aware bounded least squares",
                       ("residual_recalculation", "rank_check", "pseudoinverse_confirmation"), 300, 2_000_000, 25),
            ),
            MathematicalStructureDefinition(
                "nonlinear_least_squares", "inverse_problem", "nonlinear_least_squares/v1",
                ("nonlinear_least_squares",), "nonlinear parameter calibration"),
            MathematicalStructureDefinition(
                "initial_value_problem", "dynamical_system", "adaptive_ode/v1",
                ("ode_system",), "ordinary differential initial-value problem",
                solver("initial_value_problem", "adaptive_ode/v1", "adaptive Runge-Kutta",
                       ("tolerance_refinement", "finite_trajectory", "initial_condition_replay"), 50, 250_000, 30),
            ),
            MathematicalStructureDefinition(
                "boundary_value_problem", "dynamical_system", "boundary_value/v1",
                ("boundary_value_problem",), "ordinary differential boundary-value problem"),
            MathematicalStructureDefinition(
                "differential_algebraic_system", "dynamical_system", "dae/v1",
                ("dae_system",), "differential-algebraic system"),
            MathematicalStructureDefinition(
                "partial_differential_system", "field_system", "pde/v1",
                ("pde_system",), "partial differential initial-boundary-value problem"),
            MathematicalStructureDefinition(
                "continuous_event_measure", "event_system", "continuous_event_measure/v1",
                ("kinematic_visibility_event",), "continuous event detection and interval measure",
                solver("continuous_event_measure", "continuous_event_measure/v1", "bracketing, root refinement, and interval union",
                       ("grid_refinement", "root_recalculation", "interval_union_recalculation"), 20, 100_000, 30),
            ),
            MathematicalStructureDefinition(
                "linear_program", "mathematical_program", "linear_program/v1",
                ("linear_program",), "continuous linear program",
                solver("linear_program", "linear_program/v1", "HiGHS linear programming",
                       ("primal_feasibility", "objective_recalculation", "termination_certificate"), 1000, 3_000_000, 30),
            ),
            MathematicalStructureDefinition(
                "mixed_integer_linear_program", "mathematical_program", "mixed_integer_linear_program/v1",
                ("mixed_integer_linear_program",), "mixed-integer linear program",
                solver("mixed_integer_linear_program", "mixed_integer_linear_program/v1", "HiGHS mixed-integer programming",
                       ("primal_feasibility", "integrality_recalculation", "mip_gap_scope"), 500, 2_000_000, 45),
            ),
            MathematicalStructureDefinition(
                "hierarchical_finite_action_program", "mathematical_program",
                "hierarchical_finite_action/v1",
                ("hierarchical_finite_action_program",),
                "finite actions with activation bounds, upper-level coverage, and optional scenario-CVaR utility",
                solver(
                    "hierarchical_finite_action_program", "hierarchical_finite_action/v1",
                    "lexicographic MILP compilation followed by HiGHS",
                    (
                        "one_action_per_decision_unit", "activation_count_bounds",
                        "minimum_weighted_shortage", "utility_recalculation",
                        "scenario_probability_normalization", "cvar_objective_recalculation",
                    ),
                    500, 4_000_000, 60,
                ),
            ),
            MathematicalStructureDefinition(
                "quadratic_program", "mathematical_program", "quadratic_program/v1",
                ("quadratic_program",), "quadratic objective with linear constraints",
                solver("quadratic_program", "quadratic_program/v1", "convex quadratic programming with multistart confirmation",
                       ("convexity_certificate", "primal_feasibility", "multistart_confirmation"), 300, 500_000, 35)),
            MathematicalStructureDefinition(
                "bounded_nonlinear_program", "mathematical_program", "bounded_nlp/v1",
                ("optimization_problem",), "bounded nonlinear program",
                solver("bounded_nonlinear_program", "bounded_nlp/v1", "bounded multistart nonlinear programming",
                       ("constraint_recalculation", "independent_starts", "objective_recalculation"), 30, 50_000, 45),
            ),
            MathematicalStructureDefinition(
                "simulation_based_bounded_program", "mathematical_program", "simulation_program/v1",
                ("kinematic_visibility_optimization",), "bounded simulation-driven program",
                solver("simulation_based_bounded_program", "simulation_program/v1", "bounded global-local simulation optimization",
                       ("independent_seeds", "exact_event_recalculation", "decision_perturbation"), 20, 300_000, 60),
            ),
            MathematicalStructureDefinition(
                "multiobjective_program", "mathematical_program", "multiobjective/v1",
                ("multiobjective_program",), "Pareto multi-objective linear program",
                solver("multiobjective_program", "multiobjective/v1", "bounded Pareto scalarization with nondominance filtering",
                       ("objective_recalculation", "nondominance_check", "coverage_scope"), 200, 1_000_000, 40)),
            MathematicalStructureDefinition(
                "robust_program", "mathematical_program", "robust_program/v1",
                ("robust_program",), "linear minimax decision over explicit scenarios",
                solver("robust_program", "robust_program/v1", "epigraph robust linear programming",
                       ("scenario_recalculation", "worst_case_epigraph", "primal_feasibility"), 300, 1_000_000, 35)),
            MathematicalStructureDefinition(
                "stochastic_program", "mathematical_program", "stochastic_program/v1",
                ("stochastic_program",), "expected-value linear program over explicit scenarios",
                solver("stochastic_program", "stochastic_program/v1", "probability-weighted linear programming",
                       ("probability_normalization", "expected_objective_recalculation", "scenario_distribution"), 300, 1_000_000, 35)),
            MathematicalStructureDefinition(
                "dynamic_program", "sequential_decision", "dynamic_program/v1",
                ("dynamic_program",), "finite-horizon finite-state dynamic program",
                solver("dynamic_program", "dynamic_program/v1", "backward Bellman recursion",
                       ("transition_stochasticity", "bellman_replay", "policy_feasibility"), 100_000, 2_000_000, 30)),
            MathematicalStructureDefinition(
                "optimal_control", "sequential_decision", "optimal_control/v1",
                ("optimal_control_problem",), "continuous-time optimal control problem"),
            MathematicalStructureDefinition(
                "shortest_path", "graph_problem", "shortest_path/v1",
                ("shortest_path_problem",), "weighted shortest path",
                solver("shortest_path", "shortest_path/v1", "Dijkstra or Bellman-Ford graph search",
                       ("path_edge_recalculation", "endpoint_check", "negative_cycle_scope"), 100_000, 2_000_000, 20),
            ),
            MathematicalStructureDefinition(
                "maximum_flow", "graph_problem", "maximum_flow/v1",
                ("maximum_flow_problem",), "capacitated maximum flow",
                solver("maximum_flow", "maximum_flow/v1", "preflow-push maximum flow",
                       ("capacity_check", "flow_conservation", "cut_value_confirmation"), 100_000, 2_000_000, 25),
            ),
            MathematicalStructureDefinition(
                "minimum_cost_flow", "graph_problem", "minimum_cost_flow/v1",
                ("minimum_cost_flow_problem",), "minimum-cost capacitated flow",
                solver("minimum_cost_flow", "minimum_cost_flow/v1", "network simplex minimum-cost flow",
                       ("demand_balance", "capacity_check", "cost_recalculation"), 100_000, 2_000_000, 25)),
            MathematicalStructureDefinition(
                "bipartite_matching", "graph_problem", "bipartite_matching/v1",
                ("bipartite_matching_problem",), "maximum-cardinality or maximum-weight bipartite matching",
                solver("bipartite_matching", "bipartite_matching/v1", "augmenting-path bipartite matching",
                       ("partition_check", "edge_membership", "matching_uniqueness"), 100_000, 2_000_000, 20),
            ),
            MathematicalStructureDefinition(
                "markov_chain", "stochastic_process", "markov_chain/v1",
                ("markov_chain",), "finite-state discrete-time Markov chain",
                solver("markov_chain", "markov_chain/v1", "stochastic matrix propagation and stationary analysis",
                       ("row_stochasticity", "mass_conservation", "stationarity_residual"), 2000, 4_000_000, 20),
            ),
            MathematicalStructureDefinition(
                "sample_expectation", "uncertainty_propagation", "sample_expectation/v1",
                ("sample_expectation",), "weighted expectation and uncertainty from evaluated samples",
                solver("sample_expectation", "sample_expectation/v1", "weighted sample moment estimation",
                       ("weight_normalization", "finite_moment_check", "split_sample_stability"), 100, 2_000_000, 15),
            ),
            MathematicalStructureDefinition(
                "discrete_event_simulation", "simulation", "discrete_event_simulation/v1",
                ("discrete_event_simulation",), "state-transition discrete-event simulation"),
        )

    def resolve(self, relation_kind: str) -> Optional[MathematicalStructureDefinition]:
        normalized = str(relation_kind)
        for definition in self._definitions:
            if normalized in definition.relation_kinds:
                return definition
        return None

    def solver_specifications(self) -> Tuple[SolverSpecification, ...]:
        return tuple(item.solver for item in self._definitions if item.solver is not None)

    def register(self, definition: MathematicalStructureDefinition) -> None:
        if not isinstance(definition, MathematicalStructureDefinition):
            raise TypeError("definition must be a MathematicalStructureDefinition")
        if any(item.key == definition.key for item in self._definitions):
            raise ValueError(f"mathematical structure already registered: {definition.key}")
        occupied = {kind for item in self._definitions for kind in item.relation_kinds}
        if occupied.intersection(definition.relation_kinds):
            raise ValueError("one or more relation kinds are already registered")
        self._definitions = (*self._definitions, definition)

    def catalog(self) -> List[Dict[str, Any]]:
        return [
            {**item.public(), "required_contract_fields": list(self._REQUIRED_FIELDS.get(item.key, ())) }
            for item in self._definitions
        ]

    def recognize(self, text: str) -> List[Dict[str, Any]]:
        """Return bounded structure candidates; recognition never authorizes execution."""
        candidates = []
        for definition in self._definitions:
            matched = [
                pattern for pattern in self._TRIGGERS.get(definition.key, ())
                if re.search(pattern, str(text), re.IGNORECASE)
            ]
            if matched:
                candidates.append({
                    "key": definition.key,
                    "family": definition.family,
                    "execution_status": "implemented" if definition.solver else "recognized_only",
                    "recognition_status": "candidate_not_executable",
                    "matched_patterns": matched,
                    "required_contract_fields": list(self._REQUIRED_FIELDS.get(definition.key, ())),
                    "warning": "keyword structure recognition is a hypothesis, not a verified model contract",
                })
        return candidates


class SemanticContractLayer:
    """Layer 1: preserve meaning, roles, assumptions, and provenance."""

    schema_version = "mathmodel.semantic-contract/v1"

    @classmethod
    def compile(
        cls,
        problem: str,
        extracted: Mapping[str, Sequence[Mapping[str, Any]]],
        operator_graph: Sequence[Mapping[str, Any]],
        structure_registry: Optional[MathematicalStructureRegistry] = None,
    ) -> Dict[str, Any]:
        structure_registry = structure_registry or MathematicalStructureRegistry()
        symbol_table: List[Dict[str, Any]] = []
        seen_symbols = set()

        def add_symbol(name: Any, role: str, unit: Any, source_ref: str) -> None:
            normalized = str(name).strip()
            signature = (normalized, role)
            if not normalized or signature in seen_symbols:
                return
            seen_symbols.add(signature)
            symbol_table.append({
                "name": normalized,
                "role": role,
                "unit": str(unit or "unresolved"),
                "source_ref": source_ref,
            })

        for entity in extracted.get("entities", []):
            add_symbol(entity.get("label") or entity.get("id"), "entity", entity.get("unit"), str(entity.get("id", "")))
        for quantity in extracted.get("quantities", []):
            add_symbol(quantity.get("id"), str(quantity.get("semantic_role", "parameter")), quantity.get("unit"), str(quantity.get("id", "")))
        for relation in extracted.get("relations", []):
            relation_id = str(relation.get("id", "relation"))
            units = relation.get("units", {}) if isinstance(relation.get("units"), Mapping) else {}
            for name in relation.get("state_variables", []):
                add_symbol(name, "state", units.get(name), relation_id)
            for name in relation.get("decision_variables", []):
                add_symbol(name, "decision", units.get(name), relation_id)
            for name in relation.get("parameters", {}):
                add_symbol(name, "parameter", units.get(name), relation_id)
            time_name = relation.get("time_variable")
            if time_name:
                add_symbol(time_name, "independent_variable", units.get(time_name), relation_id)

        semantic_items = []
        for role, key in (("objective", "objectives"), ("constraint", "constraints"), ("decision", "decisions")):
            for item in extracted.get(key, []):
                semantic_items.append({
                    "id": str(item.get("id", f"{role}_{len(semantic_items) + 1}")),
                    "role": role,
                    "statement": str(item.get("source_text", ""))[:1000],
                    "source": str(item.get("source", "explicit_problem_statement")),
                })

        assumptions = _unique(
            assumption
            for relation in extracted.get("relations", [])
            for assumption in relation.get("assumptions", [])
        )
        unresolved = _unique([
            *(
                missing
                for node in operator_graph
                for missing in node.get("missing_bindings", [])
            ),
            *(
                error
                for relation in extracted.get("relations", [])
                for error in relation.get("validation_errors", [])
            ),
        ])
        source_items = [
            item
            for key in (
                "entities", "quantities", "relations", "objectives", "constraints",
                "decisions", "initial_conditions", "boundary_conditions",
            )
            for item in extracted.get(key, [])
        ]
        sourced = sum(bool(str(item.get("source_text", "")).strip()) for item in source_items)
        markers = list(_SUBPROBLEM.finditer(problem))[:100]
        subproblems = []
        for index, marker in enumerate(markers):
            end = markers[index + 1].start() if index + 1 < len(markers) else len(problem)
            subproblems.append({
                "id": f"problem_{marker.group(1)}",
                "statement": problem[marker.end():end].strip()[:1200],
            })
        return {
            "schema_version": cls.schema_version,
            "status": "complete_with_gaps" if unresolved else "complete",
            "symbol_table": symbol_table,
            "semantic_items": semantic_items,
            "subproblems": subproblems,
            "assumptions": assumptions,
            "candidate_structures": structure_registry.recognize(problem),
            "unresolved_bindings": unresolved,
            "provenance": {
                "sourced_items": sourced,
                "total_items": len(source_items),
                "coverage": sourced / max(1, len(source_items)),
                "policy": "every executable mathematical node must retain a source reference",
            },
        }


class UnifiedMathematicalIRLayer:
    """Layer 2: normalize domain adapters into mathematical structures."""

    schema_version = "mathmodel.unified-ir/v1"
    @classmethod
    def compile(
        cls,
        extracted: Mapping[str, Sequence[Mapping[str, Any]]],
        semantic_contract: Mapping[str, Any],
        registry: Optional[MathematicalStructureRegistry] = None,
    ) -> Dict[str, Any]:
        registry = registry or MathematicalStructureRegistry()
        nodes = []
        deferred_semantic_relations = []
        seen_ids = set()
        validation_errors = []
        for index, relation in enumerate(extracted.get("relations", []), 1):
            relation_id = str(relation.get("id") or f"relation_{index}")
            node_id = f"math_node_{relation_id}"
            if node_id in seen_ids:
                validation_errors.append(f"duplicate_node_id:{node_id}")
                node_id = f"{node_id}_{index}"
            seen_ids.add(node_id)
            kind = str(relation.get("kind", "unclassified_relation"))
            definition = registry.resolve(kind)
            if definition is None:
                deferred_semantic_relations.append({
                    "relation_id": relation_id,
                    "kind": kind,
                    "status": "semantic_candidate_only",
                    "reason": "no_registered_mathematical_structure",
                    "source": relation.get("source", "unknown"),
                    "source_text": str(relation.get("source_text", ""))[:1200],
                    "validation_errors": list(relation.get("validation_errors", [])),
                })
                # Unclassified prose remains auditable semantic evidence, but it
                # is not a mathematical node and must never appear in a solver
                # queue as an allegedly failed computation.
                continue
            family, form, contract_type = (
                definition.family, definition.key, definition.contract_type
            )
            verified = relation.get("parse_status") in EXECUTABLE_PARSE_STATES
            variables = _unique([
                *relation.get("state_variables", []),
                *relation.get("decision_variables", []),
                *relation.get("variables", []),
                *(relation.get("parameters", {}).keys() if isinstance(relation.get("parameters"), Mapping) else []),
            ])
            objective = relation.get("objective")
            objective_contract = (
                {"expression": objective, "direction": relation.get("direction", "maximize")}
                if objective is not None else None
            )
            nodes.append({
                "id": node_id,
                "relation_id": relation_id,
                "subproblem_id": relation.get("subproblem_id"),
                "mathematical_family": family,
                "mathematical_form": form,
                "contract_type": contract_type,
                "status": "executable" if verified and form != "unsupported" else "deferred",
                "variables": variables,
                "equations": dict(relation.get("rhs", {})) if isinstance(relation.get("rhs"), Mapping) else [],
                "objective": objective_contract,
                "constraints": list(relation.get("constraints", [])),
                "units": dict(relation.get("units", {})) if isinstance(relation.get("units"), Mapping) else {},
                "assumptions": list(relation.get("assumptions", [])),
                "dependencies": _unique([relation.get("depends_on_subproblem", "")]),
                "relation_dependencies": _unique(
                    binding.get("source_relation_id", "")
                    for binding in relation.get("input_bindings", [])
                    if isinstance(binding, Mapping)
                ),
                "verification": {
                    "parse_status": relation.get("parse_status", "unverified"),
                    "errors": list(relation.get("validation_errors", [])),
                    "dimension_checks": list(relation.get("dimension_checks", [])),
                },
                "provenance": {
                    "source": relation.get("source", "unknown"),
                    "source_text": str(relation.get("source_text", ""))[:1200],
                },
                # The contract contains values and safe expressions only.  Prose is
                # metadata and is never evaluated by an executor.
                "execution_contract": dict(relation),
            })

        executable_count = sum(node["status"] == "executable" for node in nodes)
        return {
            "schema_version": cls.schema_version,
            "status": (
                "ready" if nodes and executable_count == len(nodes) else
                ("partially_ready" if executable_count else "draft_only")
            ),
            "semantic_contract_ref": semantic_contract.get("schema_version"),
            "structure_registry_ref": registry.schema_version,
            "structure_catalog": registry.catalog(),
            "nodes": nodes,
            "deferred_semantic_relations": deferred_semantic_relations,
            "dependency_edges": [
                {"from": dependency, "to": node["relation_id"], "relation": "feeds"}
                for node in nodes for dependency in node.get("relation_dependencies", [])
            ],
            "validation": {
                "status": "pass" if not validation_errors else "fail",
                "errors": validation_errors,
                "executable_nodes": executable_count,
                "deferred_nodes": len(nodes) - executable_count,
                "semantic_candidates": len(deferred_semantic_relations),
            },
            "policy": {
                "prose_is_executable": False,
                "unclassified_prose_is_solver_node": False,
                "partial_execution_allowed": True,
                "unknown_structure_is_deferred": True,
            },
        }


class StructureAwareSolverPlanner:
    """Layer 3: select algorithms from form and enforce finite budgets."""

    schema_version = "mathmodel.solver-plan/v1"

    def __init__(
        self, specifications: Optional[Sequence[SolverSpecification]] = None,
        structure_registry: Optional[MathematicalStructureRegistry] = None,
    ) -> None:
        structure_registry = structure_registry or MathematicalStructureRegistry()
        self._specifications = {
            item.mathematical_form: item for item in (
                specifications or structure_registry.solver_specifications()
            )
        }

    def plan(self, mathematical_ir: Mapping[str, Any]) -> Dict[str, Any]:
        nodes = []
        runnable = 0
        total_evaluations = 0
        for node in mathematical_ir.get("nodes", []):
            specification = self._specifications.get(str(node.get("mathematical_form")))
            variable_count = len(node.get("variables", []))
            reasons = []
            if node.get("status") != "executable":
                reasons.append("mathematical_contract_not_verified")
            if specification is None:
                reasons.append("no_solver_for_mathematical_form")
            elif variable_count > specification.max_variables:
                reasons.append("variable_count_exceeds_solver_budget")
            status = "runnable" if not reasons else "deferred"
            if status == "runnable":
                runnable += 1
                total_evaluations += specification.max_evaluations
            nodes.append({
                "id": f"plan_{node.get('id')}",
                "ir_node_id": node.get("id"),
                "relation_id": node.get("relation_id"),
                "subproblem_id": node.get("subproblem_id"),
                "relation_dependencies": list(node.get("relation_dependencies", [])),
                "mathematical_form": node.get("mathematical_form"),
                "status": status,
                "deferred_reasons": reasons,
                **(specification.public() if specification else {
                    "executor_key": None,
                    "solver_family": None,
                    "verification_strategy": [],
                    "resource_budget": {},
                }),
                "failure_policy": "isolate_node_and_continue",
            })
        execution_order, dependency_errors = self._execution_order(nodes)
        return {
            "schema_version": self.schema_version,
            "status": (
                "partially_ready" if dependency_errors and runnable else
                ("ready" if nodes and runnable == len(nodes) else
                ("partially_ready" if runnable else "blocked")
                )
            ),
            "nodes": nodes,
            "execution_order": execution_order,
            "dependency_errors": dependency_errors,
            "budget_summary": {
                "runnable_nodes": runnable,
                "deferred_nodes": len(nodes) - runnable,
                "max_total_evaluations": total_evaluations,
                "node_failure_isolation": True,
                "dependency_errors": len(dependency_errors),
            },
            "selection_rule": "mathematical_form_only",
        }

    @staticmethod
    def _execution_order(nodes: Sequence[Mapping[str, Any]]) -> Tuple[List[str], List[str]]:
        runnable = [node for node in nodes if node.get("status") == "runnable"]
        by_relation = {str(node.get("relation_id")): node for node in runnable}
        dependencies = {
            str(node.get("id")): set(map(str, node.get("relation_dependencies", [])))
            for node in runnable
        }
        errors = _unique(
            f"missing_upstream_relation:{dependency}"
            for values in dependencies.values() for dependency in values
            if dependency not in by_relation
        )
        order: List[str] = []
        pending = list(runnable)
        resolved_relations = set()
        while pending:
            ready = [
                node for node in pending
                if set(map(str, node.get("relation_dependencies", []))) <= resolved_relations
            ]
            if not ready:
                errors.append("cyclic_or_unresolved_relation_dependencies")
                order.extend(str(node.get("id")) for node in pending)
                break
            for node in ready:
                order.append(str(node.get("id")))
                resolved_relations.add(str(node.get("relation_id")))
                pending.remove(node)
        return order, _unique(errors)


class IndependentResultAuditor:
    """Layer 4: distrust solver labels and independently recompute invariants."""

    schema_version = "mathmodel.independent-audit/v1"

    @classmethod
    def audit(
        cls,
        mathematical_ir: Mapping[str, Any],
        solver_plan: Mapping[str, Any],
        execution: Mapping[str, Any],
    ) -> Dict[str, Any]:
        ir_nodes = {str(node.get("id")): node for node in mathematical_ir.get("nodes", [])}
        plan_nodes = {str(node.get("id")): node for node in solver_plan.get("nodes", [])}
        resolved_contracts = execution.get("resolved_contracts", {})
        result_audits = []
        for index, result in enumerate(execution.get("results", []), 1):
            plan_id = str(result.get("solver_plan_node_id", ""))
            plan_node = plan_nodes.get(plan_id)
            ir_node = ir_nodes.get(str(result.get("ir_node_id", "")))
            checks: List[Dict[str, Any]] = []

            cls._check(checks, "contract_linkage", plan_node is not None and ir_node is not None,
                       "数值结果可追溯到求解计划和统一 IR 节点。",
                       "结果缺少可追溯的计划或 IR 节点。")
            finite = _finite_payload(result.get("summary", {})) and _finite_payload(result.get("solution", {}))
            cls._check(checks, "finite_result", finite,
                       "摘要和决策变量均为有限数。", "摘要或决策变量包含 NaN/Inf。")

            form = str((ir_node or {}).get("mathematical_form", "unknown"))
            execution_contract = resolved_contracts.get(
                str(result.get("relation_id")),
                (ir_node or {}).get("execution_contract", {}),
            )
            if form == "initial_value_problem":
                convergence = result.get("convergence", {})
                tolerance = convergence.get("relative_tolerance_comparison")
                accepted = convergence.get("acceptance_tolerance", 1e-6)
                passed = _finite_scalar(tolerance) and _finite_scalar(accepted) and float(tolerance) <= float(accepted)
                cls._check(checks, "independent_tolerance_gate", passed,
                           f"加密积分相对差 {tolerance} 不超过阈值 {accepted}。",
                           f"加密积分相对差 {tolerance} 超过或无法对照阈值 {accepted}。")
            elif form == "bounded_nonlinear_program":
                violation = result.get("maximum_constraint_violation")
                passed = _finite_scalar(violation) and float(violation) <= 1e-7
                cls._check(checks, "independent_feasibility_gate", passed,
                           f"重新读取的最大约束违反为 {violation}。",
                           f"最大约束违反 {violation} 未通过 1e-7 门槛。")
                starts = result.get("successful_starts", 0)
                cls._warn(checks, "global_optimality_scope", int(starts or 0) >= 2,
                          f"有 {starts} 个独立可行起点，但这仍不构成非凸全局最优证明。",
                          "缺少至少两个独立可行起点，局部最优稳定性不足。")
                convexity = (ir_node or {}).get("execution_contract", {}).get(
                    "convexity_certificate", {}
                )
                globally_certified = bool(
                    isinstance(convexity, Mapping) and convexity.get("verified") is True
                )
                checks.append({
                    "id": "global_certificate_boundary",
                    "status": "pass" if globally_certified else "warning",
                    "evidence": (
                        "已提供通过验证的凸性/全局最优证书。" if globally_certified else
                        "没有通过验证的凸性证明或全局上下界，只允许声称局部或高质量候选。"
                    ),
                })
            elif form in {"continuous_event_measure", "simulation_based_bounded_program"}:
                intervals = result.get("effective_intervals", [])
                reported = result.get("duration")
                try:
                    recomputed, merged = _interval_union_measure(intervals)
                    passed = _finite_scalar(reported) and abs(recomputed - float(reported)) <= 1e-7 * max(1.0, abs(recomputed))
                    detail = f"独立区间并集复算={recomputed:.12g}，报告值={reported}，合并区间数={len(merged)}。"
                except (TypeError, ValueError, OverflowError) as exc:
                    passed, detail = False, f"区间并集无法复算：{exc}"
                cls._check(checks, "independent_interval_union", passed, detail, detail)
                convergence = result.get("convergence", {})
                refined = convergence.get("event_refinement", convergence).get("status") == "pass"
                cls._check(checks, "event_refinement", refined,
                           "事件边界经过加密或根求解复核。", "事件边界没有通过独立加密状态门。")
                if form == "simulation_based_bounded_program":
                    starts = int(result.get("successful_starts", 0) or 0)
                    cls._warn(checks, "nonconvex_reproducibility", starts >= 2,
                              f"记录到 {starts} 个可行候选；只能声称高质量候选。",
                              "少于两个可行候选，非凸结果容易是偶然局部解。")
                    sensitivity = result.get("maximum_relative_sensitivity_drop")
                    stable = not _finite_scalar(sensitivity) or float(sensitivity) <= 0.2
                    cls._warn(checks, "decision_sensitivity", stable,
                              f"局部扰动最大相对损失为 {sensitivity}。",
                              f"局部扰动最大相对损失为 {sensitivity}，结果对实施误差敏感。")
                    checks.append({
                        "id": "simulation_global_certificate_boundary",
                        "status": "warning",
                        "evidence": "仿真驱动非凸搜索没有可验证全局界，只能输出高质量可行候选。",
                    })
            elif form == "linear_system":
                import numpy as np

                contract = execution_contract
                names = list(contract.get("variables", []))
                try:
                    matrix = np.asarray(contract.get("coefficient_matrix"), dtype=float)
                    rhs = np.asarray(contract.get("right_hand_side"), dtype=float)
                    vector = np.asarray([result.get("solution", {})[name] for name in names], dtype=float)
                    residual = float(np.linalg.norm(matrix @ vector - rhs) / max(1.0, np.linalg.norm(rhs)))
                    passed = math.isfinite(residual) and residual <= 1e-9
                except (KeyError, TypeError, ValueError, OverflowError):
                    residual, passed = math.inf, False
                cls._check(checks, "independent_linear_residual", passed,
                           f"独立矩阵乘法复算相对残差为 {residual:.3g}。",
                           f"独立矩阵乘法复算失败或残差为 {residual}。")
            elif form == "scalar_polynomial_root":
                import numpy as np

                contract = execution_contract
                try:
                    residual = abs(float(np.polyval(contract.get("coefficients", []), result.get("root"))))
                    scale = max(1.0, float(np.max(np.abs(contract.get("coefficients", [])))))
                    passed = math.isfinite(residual) and residual <= 1e-8 * scale
                except (TypeError, ValueError, OverflowError):
                    residual, passed = math.inf, False
                cls._check(checks, "independent_polynomial_residual", passed,
                           f"独立代回多项式的绝对残差为 {residual:.3g}。",
                           f"根无法独立代回或绝对残差为 {residual}。")
            elif form == "linear_least_squares":
                import numpy as np

                contract = execution_contract
                names = list(contract.get("variables", []))
                try:
                    design = np.asarray(contract.get("design_matrix"), dtype=float)
                    observed = np.asarray(contract.get("observations"), dtype=float)
                    vector = np.asarray([result.get("solution", {})[name] for name in names], dtype=float)
                    rmse = float(np.sqrt(np.mean((design @ vector - observed) ** 2)))
                    reported = float(result.get("rmse"))
                    passed = math.isfinite(rmse) and abs(rmse - reported) <= 1e-9 * max(1.0, rmse)
                except (KeyError, TypeError, ValueError, OverflowError):
                    rmse, reported, passed = math.inf, math.inf, False
                cls._check(checks, "independent_least_squares_residual", passed,
                           f"独立RMSE复算={rmse:.12g}，报告值={reported:.12g}。",
                           "最小二乘残差无法独立复算或与报告值不一致。")
            elif form in {"linear_program", "mixed_integer_linear_program"}:
                import numpy as np

                contract = execution_contract
                names = list(contract.get("variables", []))
                try:
                    vector = np.asarray([result.get("solution", {})[name] for name in names], dtype=float)
                    coefficients = np.asarray(contract.get("objective_coefficients"), dtype=float)
                    objective = float(coefficients @ vector)
                    reported = float(result.get("objective_value"))
                    objective_ok = abs(objective - reported) <= 1e-9 * max(1.0, abs(objective))
                    violations = []
                    for matrix_name, rhs_name, equality in (
                        ("A_ub", "b_ub", False), ("A_eq", "b_eq", True),
                    ):
                        matrix = np.asarray(contract.get(matrix_name, []), dtype=float)
                        rhs = np.asarray(contract.get(rhs_name, []), dtype=float)
                        if matrix.size:
                            delta = matrix @ vector - rhs
                            violations.append(float(np.max(np.abs(delta) if equality else np.maximum(delta, 0.0))))
                    for value, pair in zip(vector, contract.get("bounds", [])):
                        if pair[0] is not None:
                            violations.append(max(0.0, float(pair[0]) - float(value)))
                        if pair[1] is not None:
                            violations.append(max(0.0, float(value) - float(pair[1])))
                    violation = max(violations, default=0.0)
                    feasibility_ok = violation <= 1e-7
                except (KeyError, TypeError, ValueError, OverflowError):
                    objective, reported, violation = math.inf, math.inf, math.inf
                    objective_ok = feasibility_ok = False
                cls._check(checks, "independent_program_objective", objective_ok,
                           f"独立线性目标复算={objective:.12g}，报告值={reported:.12g}。",
                           "线性目标无法独立复算或与报告值不一致。")
                cls._check(checks, "independent_program_feasibility", feasibility_ok,
                           f"独立约束与边界复算的最大违反为 {violation:.3g}。",
                           f"独立可行性复算失败或最大违反为 {violation}。")
                if form == "mixed_integer_linear_program":
                    try:
                        integrality = max(
                            (abs(float(value) - round(float(value)))
                             for value, code in zip(vector, contract.get("integrality", []))
                             if int(code) in {1, 3}),
                            default=0.0,
                        )
                        integer_ok = integrality <= 1e-7
                    except (TypeError, ValueError, OverflowError):
                        integrality, integer_ok = math.inf, False
                    cls._check(checks, "independent_integrality", integer_ok,
                               f"独立整数性复算最大偏差为 {integrality:.3g}。",
                               f"独立整数性复算失败或最大偏差为 {integrality}。")
            elif form == "shortest_path":
                contract = execution_contract
                path = list(result.get("path", []))
                directed = bool(contract.get("directed", True))
                edge_weights = {}
                for edge in contract.get("edges", []):
                    edge_weights[(str(edge.get("source")), str(edge.get("target")))] = float(edge.get("weight"))
                    if not directed:
                        edge_weights[(str(edge.get("target")), str(edge.get("source")))] = float(edge.get("weight"))
                try:
                    recomputed = sum(edge_weights[(str(left), str(right))] for left, right in zip(path, path[1:]))
                    endpoint_ok = (
                        path and path[0] == contract.get("source_node")
                        and path[-1] == contract.get("target_node")
                    )
                    length_ok = abs(float(recomputed) - float(result.get("path_length"))) <= 1e-10
                    nodes = {node for edge in edge_weights for node in edge}
                    distances = {node: math.inf for node in nodes}
                    distances[str(contract.get("source_node"))] = 0.0
                    for _ in range(max(0, len(nodes) - 1)):
                        changed = False
                        for (left, right), weight in edge_weights.items():
                            candidate = distances[left] + weight
                            if candidate < distances[right] - 1e-12:
                                distances[right] = candidate
                                changed = True
                        if not changed:
                            break
                    negative_cycle = any(
                        distances[left] + weight < distances[right] - 1e-10
                        for (left, right), weight in edge_weights.items()
                        if math.isfinite(distances[left])
                    )
                    optimum = distances[str(contract.get("target_node"))]
                    optimality_ok = not negative_cycle and abs(float(recomputed) - optimum) <= 1e-9
                except (KeyError, TypeError, ValueError, OverflowError):
                    recomputed, optimum = math.inf, math.inf
                    endpoint_ok = length_ok = optimality_ok = False
                cls._check(checks, "independent_path_membership", endpoint_ok and length_ok,
                           f"逐边复算路径长度为 {recomputed}，端点正确。",
                           "返回路径含不存在的边、端点错误或长度复算不一致。")
                cls._check(checks, "independent_shortest_path_optimality", optimality_ok,
                           f"独立Bellman-Ford松弛得到最短距离 {optimum}。",
                           f"独立最短性复算失败、存在可达负环或最短距离为 {optimum}。")
            elif form == "maximum_flow":
                contract = execution_contract
                flow = result.get("flow", {})
                source, sink = str(contract.get("source_node")), str(contract.get("sink_node"))
                capacities = {
                    (str(edge.get("source")), str(edge.get("target"))): float(edge.get("capacity"))
                    for edge in contract.get("edges", [])
                }
                try:
                    capacity_violation = max(
                        (max(0.0, -float(flow.get(left, {}).get(right, 0.0)),
                             float(flow.get(left, {}).get(right, 0.0)) - capacity)
                         for (left, right), capacity in capacities.items()),
                        default=0.0,
                    )
                    nodes = {node for edge in capacities for node in edge}
                    conservation = 0.0
                    for node in nodes - {source, sink}:
                        inflow = sum(float(flow.get(left, {}).get(node, 0.0)) for left, right in capacities if right == node)
                        outflow = sum(float(flow.get(node, {}).get(right, 0.0)) for left, right in capacities if left == node)
                        conservation = max(conservation, abs(inflow - outflow))
                    source_out = sum(float(flow.get(source, {}).get(right, 0.0)) for left, right in capacities if left == source)
                    source_in = sum(float(flow.get(left, {}).get(source, 0.0)) for left, right in capacities if right == source)
                    flow_value = source_out - source_in
                    partition = result.get("minimum_cut_partition", [[], []])
                    left_partition, right_partition = set(map(str, partition[0])), set(map(str, partition[1]))
                    partition_ok = left_partition | right_partition == nodes and not left_partition & right_partition and source in left_partition and sink in right_partition
                    cut_value = sum(
                        capacity for (left, right), capacity in capacities.items()
                        if left in left_partition and right in right_partition
                    )
                    gap = abs(flow_value - cut_value)
                    passed = capacity_violation <= 1e-8 and conservation <= 1e-8 and partition_ok and gap <= 1e-8
                except (TypeError, ValueError, OverflowError, IndexError):
                    capacity_violation = conservation = gap = math.inf
                    partition_ok = passed = False
                cls._check(checks, "independent_maxflow_feasibility", capacity_violation <= 1e-8 and conservation <= 1e-8,
                           f"独立复算容量违反={capacity_violation:.3g}，守恒违反={conservation:.3g}。",
                           f"独立流可行性失败：容量={capacity_violation}，守恒={conservation}。")
                cls._check(checks, "independent_maxflow_mincut", partition_ok and gap <= 1e-8,
                           f"由返回割集逐边复算的最大流—最小割差为 {gap:.3g}。",
                           f"割集无效或最大流—最小割差为 {gap}。")
            elif form == "bipartite_matching":
                contract = execution_contract
                pairs = result.get("matching", [])
                allowed = {
                    (str(edge.get("source")), str(edge.get("target")))
                    for edge in contract.get("edges", [])
                }
                used = [str(node) for pair in pairs for node in pair]
                passed = (
                    len(used) == len(set(used))
                    and all((str(pair[0]), str(pair[1])) in allowed for pair in pairs)
                )
                cls._check(checks, "independent_matching_structure", passed,
                           f"{len(pairs)} 对匹配均来自输入边且节点不重复。",
                           "匹配包含非输入边或重复使用节点。")
            elif form == "markov_chain":
                contract = execution_contract
                distribution = result.get("distribution", {})
                mass = sum(float(value) for value in distribution.values()) if distribution else math.inf
                stationary_residual = result.get("summary", {}).get("stationarity_residual")
                passed = abs(mass - 1.0) <= 1e-10 and _finite_scalar(stationary_residual) and float(stationary_residual) <= 1e-8
                cls._check(checks, "independent_probability_invariants", passed,
                           f"传播后概率和={mass:.12g}，平稳残差={stationary_residual}。",
                           f"概率质量或平稳性不满足：mass={mass}, residual={stationary_residual}。")
            elif form == "sample_expectation":
                import numpy as np

                contract = execution_contract
                try:
                    values = np.asarray(contract.get("values"), dtype=float)
                    if values.ndim == 1:
                        values = values[:, None]
                    weights = np.asarray(contract.get("weights"), dtype=float)
                    weights = weights / weights.sum()
                    recomputed = weights @ values
                    names = contract.get("quantity_names", [])
                    reported = np.asarray([result.get("expectation", {})[name] for name in names])
                    difference = float(np.max(np.abs(recomputed - reported)))
                    passed = difference <= 1e-10
                except (KeyError, TypeError, ValueError, OverflowError):
                    difference, passed = math.inf, False
                cls._check(checks, "independent_weighted_expectation", passed,
                           f"独立加权均值复算最大绝对差为 {difference:.3g}。",
                           f"加权均值无法独立复算或最大差为 {difference}。")
            else:
                convergence = result.get("convergence", {})
                status = convergence.get("status")
                cls._check(checks, "generic_convergence_contract", status in {"pass", "warning"},
                           f"通用收敛状态为 {status}。", f"通用收敛状态为 {status}。")

            legacy_audit = result.get("credibility_audit", {})
            legacy_status = str(legacy_audit.get("status", "not_assessed"))
            checks.append({
                "id": "solver_self_audit",
                "status": (
                    "pass" if legacy_status == "pass" else
                    ("warning" if legacy_status in {"warning", "not_assessed"} else "fail")
                ),
                "evidence": (
                    f"求解器自审计状态为 {legacy_status}；这里只作为旁证，不代替独立审计。"
                ),
            })
            fail_count = sum(check["status"] == "fail" for check in checks)
            warning_count = sum(check["status"] == "warning" for check in checks)
            grade = "rejected" if fail_count else ("conditionally_supported" if warning_count else "supported")
            result_audits.append({
                "result_index": index,
                "relation_id": result.get("relation_id"),
                "subproblem_id": result.get("subproblem_id"),
                "mathematical_form": form,
                "status": "fail" if fail_count else ("warning" if warning_count else "pass"),
                "grade": grade,
                "checks": checks,
                "false_confidence_flags": [
                    check["id"] for check in checks if check["status"] in {"fail", "warning"}
                ],
                "decision": (
                    "拒绝该数值结论。" if fail_count else
                    ("只允许作为带假设、边界和敏感性说明的候选结论。" if warning_count else
                     "在当前数学契约与数值门槛内可采用。")
                ),
            })

        failures = list(execution.get("failures", []))
        if failures:
            overall = "fail" if not result_audits else "warning"
        elif any(item["status"] == "fail" for item in result_audits):
            overall = "fail"
        elif any(item["status"] == "warning" for item in result_audits):
            overall = "warning"
        else:
            overall = "pass" if result_audits else "not_assessed"
        return {
            "schema_version": cls.schema_version,
            "status": overall,
            "result_audits": result_audits,
            "execution_failures": failures,
            "coverage": {
                "audited_results": len(result_audits),
                "executed_results": len(execution.get("results", [])),
                "complete": len(result_audits) == len(execution.get("results", [])),
            },
            "claim_policy": (
                "numerical success is not model truth; rejected results cannot become claims, "
                "and warnings must preserve assumptions and optimality boundaries"
            ),
        }

    @staticmethod
    def _check(
        checks: List[Dict[str, Any]], check_id: str, condition: bool,
        pass_evidence: str, fail_evidence: str,
    ) -> None:
        checks.append({
            "id": check_id,
            "status": "pass" if condition else "fail",
            "evidence": pass_evidence if condition else fail_evidence,
        })

    @staticmethod
    def _warn(
        checks: List[Dict[str, Any]], check_id: str, condition: bool,
        pass_evidence: str, warning_evidence: str,
    ) -> None:
        checks.append({
            "id": check_id,
            "status": "pass" if condition else "warning",
            "evidence": pass_evidence if condition else warning_evidence,
        })


class FourLayerModelingPipeline:
    """Facade used by the assistant and by standalone structured-IR callers."""

    schema_version = "mathmodel.four-layer-pipeline/v1"

    def __init__(
        self, planner: Optional[StructureAwareSolverPlanner] = None,
        structure_registry: Optional[MathematicalStructureRegistry] = None,
    ) -> None:
        self.structure_registry = structure_registry or MathematicalStructureRegistry()
        self.planner = planner or StructureAwareSolverPlanner(
            structure_registry=self.structure_registry
        )

    def compile(
        self,
        problem: str,
        extracted: Mapping[str, Sequence[Mapping[str, Any]]],
        operator_graph: Sequence[Mapping[str, Any]],
    ) -> Dict[str, Any]:
        semantic = SemanticContractLayer.compile(
            problem, extracted, operator_graph, self.structure_registry
        )
        mathematical_ir = UnifiedMathematicalIRLayer.compile(
            extracted, semantic, self.structure_registry
        )
        solver_plan = self.planner.plan(mathematical_ir)
        return {
            "schema_version": self.schema_version,
            "semantic_contract": semantic,
            "mathematical_ir": mathematical_ir,
            "solver_plan": solver_plan,
        }

    @staticmethod
    def audit(
        mathematical_ir: Mapping[str, Any],
        solver_plan: Mapping[str, Any],
        execution: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return IndependentResultAuditor.audit(mathematical_ir, solver_plan, execution)
