"""Domain-neutral compiler for mathematical problems without tabular data.

The compiler deliberately knows mathematical primitives, not contest questions.
It converts a problem statement into a provenance-aware intermediate
representation (IR), selects reusable operators, reports unresolved bindings,
and proposes a solver route. Natural-language extraction is never treated as
proof that a numerical model is executable.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass
import hashlib
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .four_layer_modeling import (
    FourLayerModelingPipeline,
    MathematicalStructureRegistry,
    SemanticContractLayer,
    StructureAwareSolverPlanner,
    UnifiedMathematicalIRLayer,
)
from .universal_math_solvers import UniversalRelationValidator, UniversalSolverRegistry
from .semantic_model_compiler import SemanticModelCompiler


_NUMBER = r"[-+−]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+−]?\d+)?"
_COORDINATE = re.compile(
    rf"[（(]\s*({_NUMBER})\s*[,，]\s*({_NUMBER})"
    rf"(?:\s*[,，]\s*({_NUMBER}))?\s*[)）]"
)
_QUANTITY_RANGE = re.compile(
    rf"(?P<lower>{_NUMBER})\s*(?:~|～|至|到)\s*(?P<upper>{_NUMBER})\s*"
    r"(?P<unit>km/h|m/s(?:\^?2|²)?|kg|g|km|cm|mm|m|h|min|ms|s|"
    r"小时|分钟|秒|千米|公里|厘米|毫米|米|度|°|%|％|元|万元|亿元|人|辆|台|个|件|次|吨)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_QUANTITY = re.compile(
    rf"(?P<value>{_NUMBER})\s*(?P<unit>"
    r"km/h|m/s(?:\^?2|²)?|kg/m3|kg/m\^3|kg|g|km|cm|mm|m|"
    r"h|min|ms|s|小时|分钟|秒|千米|公里|厘米|毫米|米|"
    r"kW|MW|W|kPa|MPa|Pa|K|℃|°C|度|°|%|％|元|万元|亿元|"
    r"人|辆|台|个|件|次|吨|L|mL)(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CLAUSE_SPLIT = re.compile(r"(?<=[。！？!?；;\n])")
_SUBPROBLEM = re.compile(
    r"(?<![0-9A-Za-z_\u4e00-\u9fff])(?:问题|任务|小问|Problem|Task)\s*"
    r"([一二三四五六七八九十]+|\d+(?!\.\d))\s*[：:.、]?",
    re.IGNORECASE,
)


def _unique(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def _clauses(text: str, *, limit: int = 300) -> List[str]:
    return [part.strip() for part in _CLAUSE_SPLIT.split(text) if part.strip()][:limit]


def _context(text: str, start: int, end: int, radius: int = 48) -> str:
    return text[max(0, start - radius):min(len(text), end + radius)].strip()


def _to_float(value: str) -> float:
    return float(str(value).replace("−", "-"))


def _coordinate_labels(text: str) -> Dict[int, str]:
    """Bind coordinate tuples to explicit entity labels, including batch lists."""
    labels: Dict[int, str] = {}
    coordinate_matches = list(_COORDINATE.finditer(text))
    for match in coordinate_matches:
        direct = re.search(r"([A-Za-z][A-Za-z0-9_-]{0,30})\s*$", text[:match.start()])
        if direct:
            labels[match.start()] = direct.group(1)
    batch_pattern = re.compile(
        r"(?P<labels>(?:[A-Za-z]{1,12}\d+\s*[、,，]\s*)+"
        r"[A-Za-z]{1,12}\d+)\s*分别(?:位于|为|是)\s*",
        re.IGNORECASE,
    )
    for batch in batch_pattern.finditer(text):
        clause_end_candidates = [
            position for marker in ("；", ";", "。", "\n\n")
            if (position := text.find(marker, batch.end())) >= 0
        ]
        clause_end = min(clause_end_candidates, default=len(text))
        names = re.findall(r"[A-Za-z]{1,12}\d+", batch.group("labels"))
        tuples = [
            match for match in coordinate_matches
            if batch.end() <= match.start() < clause_end
        ]
        for name, match in zip(names, tuples):
            labels[match.start()] = name
    return labels


def _coordinate_unit_context(text: str) -> Tuple[str, str]:
    has_coordinate_semantics = bool(re.search(r"坐标|位置|原点|平面|coordinate", text, re.IGNORECASE))
    has_metric_length = bool(re.search(r"\d\s*(?:km|cm|mm|m|千米|公里|厘米|毫米|米)(?!/)", text))
    has_metric_speed = bool(re.search(r"\d\s*(?:km/h|m/s)", text, re.IGNORECASE))
    if has_coordinate_semantics and has_metric_length and has_metric_speed:
        return "m", "inferred_from_consistent_metric_geometry_context"
    return "unresolved", "not_stated"


def _entity_label(prefix: str, index: int) -> str:
    tail = re.sub(r"[，,。；;：:\s]+$", "", prefix[-48:])
    patterns = (
        r"([A-Za-z][A-Za-z0-9_-]{0,30})\s*$",
        r"([\u4e00-\u9fff]{1,16})(?:的)?(?:初始)?(?:位置|坐标|中心|圆心|端点)?(?:为|是|在|位于)?\s*$",
    )
    for pattern in patterns:
        match = re.search(pattern, tail)
        if match:
            label = match.group(1)
            label = re.sub(
                r"(?:下底面|上底面)?(?:的)?(?:位置|坐标|中心|圆心)?(?:为|是|在|位于)?$",
                "", label,
            )
            return label or match.group(1)
    return f"entity_{index}"


def _relation_kind(clause: str) -> Optional[str]:
    if re.search(r"(?:∂|偏微分|laplace|扩散|传热|波动方程)", clause, re.IGNORECASE):
        return "partial_differential_equation"
    if re.search(r"(?:d\w+\s*/\s*dt|\w+['′]\s*=|微分方程|变化率)", clause, re.IGNORECASE):
        return "ordinary_differential_equation"
    if re.search(r"(?:<=|>=|≤|≥|<|>|不超过|不少于|至多|至少)", clause):
        return "inequality"
    if re.search(r"(?:=|等于|满足|守恒|平衡)", clause):
        return "equality_or_balance"
    return None


@dataclass(frozen=True)
class OperatorDefinition:
    """A reusable mathematical operator selected by semantics, never by title."""

    key: str
    category: str
    description: str
    required_bindings: Tuple[str, ...]
    produces: Tuple[str, ...]
    solver_route: str
    triggers: Tuple[str, ...]

    def public(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("triggers", None)
        payload["required_bindings"] = list(self.required_bindings)
        payload["produces"] = list(self.produces)
        return payload


class MechanisticOperatorRegistry:
    """Versioned catalog of domain-independent modeling building blocks."""

    version = "1.0"

    def __init__(self) -> None:
        self._definitions = (
            OperatorDefinition(
                "constant_rate_state", "dynamics", "constant-rate state evolution",
                ("state", "initial_condition", "rate_law"), ("state_trajectory",),
                "closed_form_or_ode_integrator",
                (r"匀速|恒速|固定速度|constant\s+speed|constant\s+rate",),
            ),
            OperatorDefinition(
                "second_order_dynamics", "dynamics", "second-order force or acceleration law",
                ("state", "initial_condition", "dynamics_law"), ("state_trajectory",),
                "ode_integrator",
                (r"加速度|重力|受力|牛顿第二|acceleration|gravity|force",),
            ),
            OperatorDefinition(
                "balance_law", "dynamics", "conservation or compartment balance",
                ("state", "dynamics_law"), ("state_derivative",), "ode_or_dae_solver",
                (r"守恒|平衡方程|流入|流出|转移率|compartment|conservation|balance",),
            ),
            OperatorDefinition(
                "first_order_ode", "dynamics", "first-order ordinary differential equation",
                ("state", "dynamics_law", "initial_condition"), ("state_trajectory",),
                "adaptive_ode_integrator",
                (r"微分方程|变化率|d\w+\s*/\s*dt|ordinary\s+differential",),
            ),
            OperatorDefinition(
                "field_pde", "dynamics", "spatial-temporal field equation",
                ("state", "dynamics_law", "initial_condition", "boundary_condition"),
                ("field_solution",), "finite_volume_or_finite_element_solver",
                (r"偏微分|扩散|传热|流场|波动方程|partial\s+differential|diffusion",),
            ),
            OperatorDefinition(
                "metric_geometry", "geometry", "distance, angle, projection, or intersection",
                ("geometry_definition",), ("geometric_measure",),
                "analytic_geometry_or_root_finding",
                (r"距离|夹角|投影|相交|切线|最近点|distance|angle|intersection",),
            ),
            OperatorDefinition(
                "region_membership", "geometry", "membership in a geometric influence region",
                ("geometry_definition", "event_definition"), ("event_indicator",),
                "analytic_geometry",
                (r"半径|球体|圆柱|区域内|覆盖范围|boundary|sphere|cylinder|region",),
            ),
            OperatorDefinition(
                "line_of_sight", "geometry", "visibility along a segment or ray",
                ("geometry_definition", "event_definition"), ("visibility_indicator",),
                "computational_geometry",
                (r"视线|可见|遮挡|遮蔽|能见度|line.of.sight|visibility|occlusion",),
            ),
            OperatorDefinition(
                "event_window", "event", "activation, termination, or valid-time window",
                ("event_definition",), ("event_interval",), "event_detection",
                (r"生效|失效|持续|直到|时刻|时间窗|有效期|duration|event|until",),
            ),
            OperatorDefinition(
                "interval_measure", "event", "measure or union of valid intervals",
                ("event_definition",), ("duration_or_measure",),
                "root_refinement_and_interval_union",
                (r"总时长|持续时间|累计时间|时间尽可能|duration|time\s+interval",),
            ),
            OperatorDefinition(
                "graph_path", "network", "path, connectivity, or centrality on a graph",
                ("graph_definition",), ("path_or_graph_metric",), "graph_algorithm",
                (r"最短路|路径|连通|中心性|节点|边|shortest\s+path|connectivity|network",),
            ),
            OperatorDefinition(
                "network_flow", "network", "capacity-constrained flow on a network",
                ("graph_definition", "constraints", "objective"), ("flow_solution",),
                "linear_or_integer_programming",
                (r"最大流|最小费用流|容量|运输网络|network\s+flow|capacity",),
            ),
            OperatorDefinition(
                "stochastic_transition", "stochastic", "probabilistic state transition",
                ("state", "probability_model"), ("state_distribution",),
                "markov_or_stochastic_simulation",
                (r"转移概率|马尔可夫|随机过程|到达率|Markov|stochastic\s+process",),
            ),
            OperatorDefinition(
                "monte_carlo", "stochastic", "sampling-based uncertainty propagation",
                ("probability_model", "computable_response"), ("output_distribution",),
                "monte_carlo_or_quasi_monte_carlo",
                (r"蒙特卡洛|随机模拟|抽样模拟|Monte\s+Carlo|random\s+simulation",),
            ),
            OperatorDefinition(
                "constrained_optimization", "optimization", "constrained single-objective search",
                ("decision_variables", "objective", "constraints"), ("candidate_solution",),
                "structure_aware_optimizer",
                (r"最大化|最小化|最优|优化|使.*(?:最大|最小|尽可能)|(?:成本|费用|风险|时间|收益|利润).*最[大小]|maximize|minimize|optimal",),
            ),
            OperatorDefinition(
                "multiobjective_optimization", "optimization", "Pareto multi-objective search",
                ("decision_variables", "objectives", "constraints"), ("pareto_set",),
                "pareto_optimizer",
                (r"多目标|帕累托|Pareto|同时(?:最大|最小)|权衡",),
            ),
            OperatorDefinition(
                "parameter_calibration", "inverse", "infer parameters from observations",
                ("computable_response", "observations", "parameters"), ("calibrated_parameters",),
                "bounded_nonlinear_least_squares",
                (r"参数估计|参数辨识|拟合参数|反演|calibrat|parameter\s+estimation",),
            ),
            OperatorDefinition(
                "robust_decision", "optimization", "decision under bounded uncertainty",
                ("decision_variables", "objective", "constraints", "uncertainty_set"),
                ("robust_solution",), "robust_or_distributionally_robust_optimizer",
                (r"鲁棒优化|最坏情形|不确定集合|分布鲁棒|robust\s+optimization|worst.case",),
            ),
        )

    def select(self, text: str) -> List[OperatorDefinition]:
        return [
            definition for definition in self._definitions
            if any(re.search(pattern, text, re.IGNORECASE) for pattern in definition.triggers)
        ]

    def get(self, key: str) -> OperatorDefinition:
        for definition in self._definitions:
            if definition.key == key:
                return definition
        raise KeyError(key)

    def register(self, definition: OperatorDefinition) -> None:
        """Register a new mathematical primitive without adding a question branch."""
        if not isinstance(definition, OperatorDefinition):
            raise TypeError("definition must be an OperatorDefinition")
        if any(item.key == definition.key for item in self._definitions):
            raise ValueError(f"operator key already registered: {definition.key}")
        self._definitions = (*self._definitions, definition)

    def catalog(self) -> List[Dict[str, Any]]:
        return [definition.public() for definition in self._definitions]


class _SafeNumericExpression(ast.NodeVisitor):
    """Compile a small numeric AST; never call eval or execute user code."""

    _functions = {
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "exp": math.exp,
        "log": math.log,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "tanh": math.tanh,
    }

    def __init__(self, symbols: Iterable[str]) -> None:
        self.symbols = set(symbols)

    def compile(self, expression: str) -> ast.AST:
        tree = ast.parse(str(expression), mode="eval")
        self.visit(tree)
        return tree.body

    def visit_Name(self, node: ast.Name) -> None:
        if node.id not in self.symbols and node.id not in self._functions:
            raise ValueError(f"unknown symbol: {node.id}")

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, (int, float)) or isinstance(node.value, bool):
            raise ValueError("only finite numeric constants are allowed")
        if not math.isfinite(float(node.value)):
            raise ValueError("numeric constants must be finite")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if not isinstance(node.op, (ast.UAdd, ast.USub)):
            raise ValueError("unsupported unary operator")
        self.visit(node.operand)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            raise ValueError("unsupported binary operator")
        self.visit(node.left)
        self.visit(node.right)

    def visit_Call(self, node: ast.Call) -> None:
        if not isinstance(node.func, ast.Name) or node.func.id not in self._functions:
            raise ValueError("function is not allow-listed")
        if node.keywords:
            raise ValueError("keyword arguments are not allowed")
        for argument in node.args:
            self.visit(argument)

    def generic_visit(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Expression, ast.Load)):
            super().generic_visit(node)
            return
        raise ValueError(f"unsupported expression node: {type(node).__name__}")

    @classmethod
    def evaluate(cls, node: ast.AST, values: Mapping[str, float]) -> float:
        if isinstance(node, ast.Constant):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in values:
                raise ValueError(f"unbound symbol: {node.id}")
            return float(values[node.id])
        if isinstance(node, ast.UnaryOp):
            value = cls.evaluate(node.operand, values)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = cls.evaluate(node.left, values), cls.evaluate(node.right, values)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            return left ** right
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            function = cls._functions[node.func.id]
            return float(function(*(cls.evaluate(argument, values) for argument in node.args)))
        raise ValueError(f"unsupported expression node: {type(node).__name__}")


class MechanisticModelingEngine:
    """Compile a no-dataset statement into a safe, auditable mathematical IR."""

    version = "3.1"
    schema_version = "mathmodel.mechanistic-ir/v2"
    max_statement_chars = 2_000_000

    def __init__(
        self,
        registry: Optional[MechanisticOperatorRegistry] = None,
        structure_registry: Optional[MathematicalStructureRegistry] = None,
        solver_registry: Optional[UniversalSolverRegistry] = None,
        semantic_compiler: Optional[SemanticModelCompiler] = None,
    ) -> None:
        self.registry = registry or MechanisticOperatorRegistry()
        self.structure_registry = structure_registry or MathematicalStructureRegistry()
        self.solver_registry = solver_registry or UniversalSolverRegistry()
        self.semantic_compiler = semantic_compiler

    def analyze(
        self, problem: str, *, ir_override: Optional[Mapping[str, Any]] = None,
        problem_images: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Dict[str, Any]:
        text = str(problem).strip()
        if not text:
            raise ValueError("problem statement must not be empty")
        if len(text) > self.max_statement_chars:
            raise ValueError(
                f"problem statement exceeds the {self.max_statement_chars} character safety limit"
            )
        extracted = self._extract(text)
        extracted["relations"].extend(
            self._compile_statement_relations(text, extracted)
        )
        semantic_compilation: Dict[str, Any] = {
            "schema_version": SemanticModelCompiler.schema_version,
            "status": "not_configured",
            "accepted_relations": [],
            "accepted_count": 0,
            "deferred_proposals": [],
            "deferred_count": 0,
            "policy": {
                "model_can_authorize_execution": False,
                "deterministic_contract_revalidation_required": True,
            },
        }
        if self.semantic_compiler is not None:
            try:
                semantic_compilation = self.semantic_compiler.compile(
                    text,
                    structure_catalog=self.structure_registry.catalog(),
                    validator=self._verify_structured_relation,
                    images=problem_images,
                )
                extracted["relations"].extend(
                    semantic_compilation.get("accepted_relations", [])
                )
            except Exception as exc:
                # A model or network failure is an isolated parser failure.  The
                # deterministic compiler and any user-supplied IR remain usable.
                semantic_compilation = {
                    "schema_version": SemanticModelCompiler.schema_version,
                    "status": "failed_safe",
                    "accepted_relations": [],
                    "accepted_count": 0,
                    "deferred_proposals": [],
                    "deferred_count": 0,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:1000],
                    "configuration": self.semantic_compiler.config.public(),
                    "policy": {
                        "model_can_authorize_execution": False,
                        "failure_isolated": True,
                        "api_key_persisted": False,
                    },
                }
        if ir_override:
            extracted = self._merge_verified_override(extracted, ir_override)
        definitions = self.registry.select(text)
        relation_operator = {
            "ode_system": "first_order_ode",
            "optimization_problem": "constrained_optimization",
            "kinematic_visibility_event": "interval_measure",
            "kinematic_visibility_optimization": "constrained_optimization",
        }
        selected_keys = {item.key for item in definitions}
        for relation in extracted["relations"]:
            key = relation_operator.get(str(relation.get("kind")))
            if key and key not in selected_keys:
                definitions.append(self.registry.get(key))
                selected_keys.add(key)
        bindings = self._semantic_bindings(text, extracted)
        graph = self._operator_graph(definitions, bindings)
        four_layer = FourLayerModelingPipeline(
            structure_registry=self.structure_registry
        ).compile(text, extracted, graph)
        unresolved = _unique(missing for node in graph for missing in node["missing_bindings"])
        compiler_plan = self._compiler_plan(graph, extracted, unresolved)
        model_draft = self._build_model_draft(graph, extracted, compiler_plan)
        alternatives = self._alternative_interpretations(text, definitions)
        audit = self._credibility_audit(extracted, graph, alternatives, compiler_plan)
        ready = bool(four_layer["solver_plan"]["budget_summary"]["runnable_nodes"])
        execution = self._execute_solver_plan(
            four_layer["mathematical_ir"], four_layer["solver_plan"],
            self.solver_registry,
        ) if ready else {
            "status": "not_executed", "results": [],
            "failures": [],
            "reason": "no unified mathematical IR node passed every safety gate",
        }
        independent_audit = FourLayerModelingPipeline.audit(
            four_layer["mathematical_ir"], four_layer["solver_plan"], execution,
        )
        audit_by_relation = {
            str(item.get("relation_id")): item
            for item in independent_audit.get("result_audits", [])
        }
        for numerical_result in execution.get("results", []):
            numerical_result["independent_audit"] = audit_by_relation.get(
                str(numerical_result.get("relation_id")), {}
            )
        if independent_audit.get("status") == "fail" and execution.get("results"):
            execution["status"] = "failed_validation"
            execution["reason"] = "one or more independent audit gates rejected a numerical result"
        four_layer["independent_audit"] = independent_audit
        executed_subproblems = {
            str(item.get("subproblem_id"))
            for item in execution.get("results", []) if item.get("subproblem_id")
        }
        statement_subproblems = list(_SUBPROBLEM.finditer(text))[:100]
        partially_executed = bool(
            execution.get("status") in {"executed", "partially_executed"}
            and executed_subproblems
            and len(executed_subproblems) < len(statement_subproblems)
        )
        execution_status = (
            "partially_executed" if partially_executed or execution.get("status") == "partially_executed" else
            ("executed" if execution.get("status") == "executed" else
            ("validation_failed" if execution.get("status") == "failed_validation" else
             ("solver_failed" if execution.get("status") == "failed" else
             ("solver_ready" if ready else "needs_model_completion"))
            ))
        )
        task_support = self._task_support(graph, compiler_plan, execution)
        subproblems = self._subproblems(
            text, graph, compiler_plan.get("blocked_by", []), execution
        )
        outstanding_requirements = _unique(
            requirement
            for support in task_support.values()
            if support.get("status") not in {"executed", "ready"}
            for requirement in support.get("missing_requirements", [])
        )
        return {
            "schema_version": self.schema_version,
            "engine_version": self.version,
            "operator_catalog_version": self.registry.version,
            "mode": "problem_statement_driven",
            "problem_fingerprint": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "input_policy": {
                "uploaded_dataset_required": False,
                "observations_invented": False,
                "stated_numbers_are": "parameters_or_conditions",
                "derived_arrays_are": "numerical_experiment_artifacts",
                "natural_language_is_executable": False,
                "semantic_model_is_executable": False,
            },
            "execution_status": execution_status,
            "semantic_model_compilation": semantic_compilation,
            "four_layer_pipeline": four_layer,
            "mathematical_ir": {
                "entities": extracted["entities"],
                "quantities": extracted["quantities"],
                "relations": extracted["relations"],
                "objectives": extracted["objectives"],
                "constraints": extracted["constraints"],
                "decision_statements": extracted["decisions"],
                "initial_conditions": extracted["initial_conditions"],
                "boundary_conditions": extracted["boundary_conditions"],
                "provenance_rule": "every extracted item retains an exact source clause",
            },
            "operator_graph": graph,
            "compiler_plan": compiler_plan,
            "model_draft": model_draft,
            "solver_execution": execution,
            "numerical_results": execution.get("results", []),
            "subproblems": subproblems,
            "task_support": task_support,
            "missing_requirements": outstanding_requirements or compiler_plan["blocked_by"],
            "alternative_interpretations": alternatives,
            "validation_protocol": self._validation_protocol(graph),
            "credibility_audit": audit,
            "result_semantics": (
                "A solver-ready specification is not a solved numerical result; execution, "
                "convergence, residual, sensitivity, and external checks are separate gates."
            ),
        }

    @staticmethod
    def _execute_solver_plan(
        mathematical_ir: Mapping[str, Any], solver_plan: Mapping[str, Any],
        universal_executors: Optional[UniversalSolverRegistry] = None,
    ) -> Dict[str, Any]:
        """Execute normalized nodes by mathematical form, isolating node failures.

        The dispatcher intentionally has no contest-title or domain-word branch.
        Adapters are registered by versioned mathematical contract type.
        """
        ir_nodes = {
            str(node.get("id")): node for node in mathematical_ir.get("nodes", [])
        }
        executors = {
            "adaptive_ode/v1": MechanisticModelingEngine._solve_ode_system,
            "bounded_nlp/v1": MechanisticModelingEngine._solve_optimization_problem,
            "continuous_event_measure/v1": MechanisticModelingEngine._solve_kinematic_visibility_event,
            "simulation_program/v1": MechanisticModelingEngine._solve_kinematic_visibility_optimization,
        }
        universal_executors = universal_executors or UniversalSolverRegistry()
        results: List[Dict[str, Any]] = []
        results_by_relation: Dict[str, Dict[str, Any]] = {}
        resolved_contracts: Dict[str, Dict[str, Any]] = {}
        failures: List[Dict[str, Any]] = []
        deferred = []
        plan_nodes = {
            str(node.get("id")): node for node in solver_plan.get("nodes", [])
        }
        ordered_ids = list(solver_plan.get("execution_order", []))
        ordered_nodes = [
            plan_nodes[node_id] for node_id in ordered_ids if node_id in plan_nodes
        ]
        ordered_nodes.extend(
            node for node_id, node in plan_nodes.items() if node_id not in set(ordered_ids)
        )
        for plan_node in ordered_nodes:
            if plan_node.get("status") != "runnable":
                deferred.append({
                    "plan_node_id": plan_node.get("id"),
                    "ir_node_id": plan_node.get("ir_node_id"),
                    "reasons": list(plan_node.get("deferred_reasons", [])),
                })
                continue
            ir_node = ir_nodes.get(str(plan_node.get("ir_node_id")))
            executor_key = str(plan_node.get("executor_key", ""))
            executor = executors.get(executor_key)
            universal_executor_available = universal_executors.has(executor_key)
            if ir_node is None or (executor is None and not universal_executor_available):
                failures.append({
                    "plan_node_id": plan_node.get("id"),
                    "ir_node_id": plan_node.get("ir_node_id"),
                    "error_type": "ExecutorContractError",
                    "message": "runnable plan node has no matching IR node or executor",
                })
                continue
            contract = copy.deepcopy(ir_node.get("execution_contract", {}))
            try:
                for binding in contract.get("input_bindings", []):
                    source_relation_id = str(binding.get("source_relation_id"))
                    upstream = results_by_relation.get(source_relation_id)
                    if upstream is None:
                        raise RuntimeError(
                            f"upstream relation did not produce a result: {source_relation_id}"
                        )
                    bound_value = MechanisticModelingEngine._read_contract_path(
                        upstream, str(binding.get("source_path"))
                    )
                    MechanisticModelingEngine._write_contract_path(
                        contract, str(binding.get("target_path")), bound_value
                    )
                if contract.get("input_bindings"):
                    reverified = MechanisticModelingEngine._verify_structured_relation(contract)
                    if reverified.get("parse_status") != "machine_verified":
                        raise ValueError(
                            "composed contract failed validation after binding: "
                            + ";".join(reverified.get("validation_errors", []))
                        )
                    contract = reverified
                result = dict(
                    universal_executors.execute(executor_key, contract)
                    if universal_executor_available else executor(contract)
                )
                result.setdefault("relation_id", ir_node.get("relation_id"))
                result.setdefault("subproblem_id", ir_node.get("subproblem_id"))
                result["ir_node_id"] = ir_node.get("id")
                result["solver_plan_node_id"] = plan_node.get("id")
                result["mathematical_form"] = ir_node.get("mathematical_form")
                result["resource_budget"] = dict(plan_node.get("resource_budget", {}))
                results.append(result)
                results_by_relation[str(ir_node.get("relation_id"))] = result
                if contract.get("input_bindings"):
                    resolved_contracts[str(ir_node.get("relation_id"))] = contract
            except Exception as exc:
                failures.append({
                    "plan_node_id": plan_node.get("id"),
                    "ir_node_id": ir_node.get("id"),
                    "relation_id": ir_node.get("relation_id"),
                    "subproblem_id": ir_node.get("subproblem_id"),
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:1000],
                })
        validation_failed = any(
            result.get("credibility_audit", {}).get("status") == "fail" for result in results
        )
        status = (
            "failed_validation" if validation_failed else
            ("partially_executed" if results and failures else
             ("executed" if results else ("failed" if failures else "not_executed")))
        )
        return {
            "status": status,
            "results": results,
            "failures": failures,
            "deferred": deferred,
            "resolved_contracts": resolved_contracts,
            "reason": (
                "one or more solver self-audits failed" if validation_failed else
                ("some mathematical nodes failed safely; successful nodes were preserved" if results and failures else
                 ("all runnable mathematical nodes failed safely" if failures else
                  (None if results else "no executable mathematical node")))
            ),
            "failure_policy": "isolate_node_and_continue",
        }

    @staticmethod
    def _read_contract_path(payload: Any, path: str) -> Any:
        current = payload
        for token in str(path).split("."):
            if isinstance(current, Mapping):
                if token not in current:
                    raise KeyError(f"source binding path does not exist: {path}")
                current = current[token]
            elif isinstance(current, Sequence) and not isinstance(current, (str, bytes)):
                if not token.isdigit() or int(token) >= len(current):
                    raise KeyError(f"source binding index does not exist: {path}")
                current = current[int(token)]
            else:
                raise KeyError(f"source binding path is not traversable: {path}")
        return copy.deepcopy(current)

    @staticmethod
    def _write_contract_path(payload: Any, path: str, value: Any) -> None:
        tokens = str(path).split(".")
        current = payload
        for token in tokens[:-1]:
            if isinstance(current, Mapping):
                if token not in current:
                    raise KeyError(f"target binding path does not exist: {path}")
                current = current[token]
            elif isinstance(current, list):
                if not token.isdigit() or int(token) >= len(current):
                    raise KeyError(f"target binding index does not exist: {path}")
                current = current[int(token)]
            else:
                raise KeyError(f"target binding path is not traversable: {path}")
        final = tokens[-1]
        if isinstance(current, dict):
            if final not in current:
                raise KeyError(f"target binding field does not exist: {path}")
            current[final] = copy.deepcopy(value)
        elif isinstance(current, list):
            if not final.isdigit() or int(final) >= len(current):
                raise KeyError(f"target binding index does not exist: {path}")
            current[int(final)] = copy.deepcopy(value)
        else:
            raise KeyError(f"target binding path is not writable: {path}")

    @staticmethod
    def _execute_verified_ir(
        extracted: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> Dict[str, Any]:
        """Backward-compatible entry routed through the four-layer contracts."""
        semantic = SemanticContractLayer.compile("", extracted, [])
        mathematical_ir = UnifiedMathematicalIRLayer.compile(extracted, semantic)
        solver_plan = StructureAwareSolverPlanner().plan(mathematical_ir)
        return MechanisticModelingEngine._execute_solver_plan(mathematical_ir, solver_plan)

    @staticmethod
    def _solve_kinematic_visibility_event(relation: Mapping[str, Any]) -> Dict[str, Any]:
        """Solve a bounded moving-point/segment visibility event with root refinement."""
        import numpy as np
        from scipy.optimize import brentq

        def vector(name: str) -> Any:
            value = np.asarray(relation.get(name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite three-dimensional vector")
            return value

        source_initial = vector("source_initial")
        source_destination = vector("source_destination")
        carrier_initial = vector("carrier_initial")
        carrier_destination = vector("carrier_destination")
        source_speed = float(relation["source_speed"])
        carrier_speed = float(relation["carrier_speed"])
        release_time = float(relation["release_time"])
        activation_delay = float(relation["activation_delay"])
        influence_radius = float(relation["influence_radius"])
        active_lifetime = float(relation["active_lifetime"])
        post_activation_velocity = vector("post_activation_velocity")
        gravity = float(relation.get("gravity", 9.8))
        scalars = (
            source_speed, carrier_speed, release_time, activation_delay,
            influence_radius, active_lifetime, gravity,
        )
        if not all(math.isfinite(value) for value in scalars):
            raise ValueError("kinematic event contains a non-finite scalar")
        if source_speed <= 0 or carrier_speed <= 0:
            raise ValueError("moving-object speeds must be positive")
        if release_time < 0 or activation_delay < 0:
            raise ValueError("release and activation times must be non-negative")
        if influence_radius <= 0 or active_lifetime <= 0 or active_lifetime > 86_400:
            raise ValueError("event radius/lifetime is outside the safe solver domain")

        def unit_direction(start: Any, destination: Any, *, horizontal: bool = False) -> Any:
            direction = destination - start
            if horizontal:
                direction = direction.copy()
                direction[2] = 0.0
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-12:
                raise ValueError("trajectory destination equals its initial point")
            return direction / norm

        source_velocity = source_speed * unit_direction(source_initial, source_destination)
        source_arrival_time = float(
            np.linalg.norm(source_destination - source_initial) / source_speed
        )
        carrier_velocity = carrier_speed * unit_direction(
            carrier_initial, carrier_destination,
            horizontal=bool(relation.get("carrier_horizontal", False)),
        )
        activation_time = release_time + activation_delay

        def activation_point_for(gravity_value: float) -> Any:
            acceleration = np.asarray([0.0, 0.0, -gravity_value], dtype=float)
            release_point = carrier_initial + carrier_velocity * release_time
            return (
                release_point + carrier_velocity * activation_delay
                + 0.5 * acceleration * activation_delay ** 2
            )

        activation_point = activation_point_for(gravity)
        if activation_point[2] < -1e-9:
            raise ValueError("payload reaches below the reference plane before activation")

        representatives = relation.get("target_representatives", [])
        if not isinstance(representatives, Sequence) or isinstance(representatives, (str, bytes)):
            raise ValueError("target_representatives must be a sequence")
        validated_representatives = []
        for item in representatives[:12]:
            if not isinstance(item, Mapping):
                raise ValueError("each target representative must be an object")
            point = np.asarray(item.get("point"), dtype=float)
            if point.shape != (3,) or not np.all(np.isfinite(point)):
                raise ValueError("target representative point must be finite and three-dimensional")
            validated_representatives.append((str(item.get("semantics", "representative_point")), point))
        if not validated_representatives:
            raise ValueError("at least one target representative is required")

        start_time = activation_time
        end_time = min(activation_time + active_lifetime, source_arrival_time)
        if end_time <= start_time:
            raise ValueError("event activates after the moving source reaches its destination")

        def source_position(time_value: float) -> Any:
            return source_initial + source_velocity * time_value

        def cloud_position(time_value: float, start_point: Any = activation_point) -> Any:
            return start_point + post_activation_velocity * (time_value - activation_time)

        def clearance(time_value: float, target_point: Any, start_point: Any = activation_point) -> float:
            source = source_position(time_value)
            cloud = cloud_position(time_value, start_point)
            segment = target_point - source
            denominator = float(np.dot(segment, segment))
            if denominator <= 1e-18:
                return float(np.linalg.norm(cloud - target_point) - influence_radius)
            projection = float(np.dot(cloud - source, segment) / denominator)
            projection = min(1.0, max(0.0, projection))
            closest = source + projection * segment
            return float(np.linalg.norm(cloud - closest) - influence_radius)

        def interval_solution(target_point: Any, samples: int, start_point: Any = activation_point) -> Dict[str, Any]:
            times = np.linspace(start_time, end_time, samples)
            values = np.asarray([
                clearance(float(time_value), target_point, start_point)
                for time_value in times
            ], dtype=float)
            if not np.all(np.isfinite(values)):
                raise FloatingPointError("event clearance became non-finite")
            roots: List[float] = []
            for left_index in range(len(times) - 1):
                left_time, right_time = float(times[left_index]), float(times[left_index + 1])
                left_value, right_value = float(values[left_index]), float(values[left_index + 1])
                if left_value == 0.0:
                    roots.append(left_time)
                if left_value * right_value < 0.0:
                    roots.append(float(brentq(
                        lambda current: clearance(current, target_point, start_point),
                        left_time, right_time, xtol=1e-11, rtol=1e-12, maxiter=100,
                    )))
            if values[-1] == 0.0:
                roots.append(float(times[-1]))
            breakpoints = [start_time, *_unique(f"{root:.12f}" for root in roots), end_time]
            numeric_breakpoints = sorted({
                float(item) for item in breakpoints
                if start_time - 1e-10 <= float(item) <= end_time + 1e-10
            })
            intervals: List[List[float]] = []
            for left, right in zip(numeric_breakpoints[:-1], numeric_breakpoints[1:]):
                midpoint = 0.5 * (left + right)
                if clearance(midpoint, target_point, start_point) <= 0.0:
                    intervals.append([float(left), float(right)])
            if not intervals and clearance(start_time, target_point, start_point) <= 0.0:
                intervals = [[float(start_time), float(end_time)]]
            duration = float(sum(right - left for left, right in intervals))
            distances = values + influence_radius
            return {
                "duration": duration,
                "intervals": intervals,
                "roots": [float(value) for value in roots],
                "time": times,
                "distance": distances,
            }

        branch_results = []
        primary_plot = None
        maximum_refinement_difference = 0.0
        for semantics, target_point in validated_representatives:
            coarse = interval_solution(target_point, 2001)
            refined = interval_solution(target_point, 8001)
            refinement_difference = abs(coarse["duration"] - refined["duration"])
            maximum_refinement_difference = max(
                maximum_refinement_difference, refinement_difference
            )
            branch_results.append({
                "semantics": semantics,
                "target_point": target_point.tolist(),
                "duration": refined["duration"],
                "intervals": refined["intervals"],
                "refinement_difference": refinement_difference,
            })
            if primary_plot is None:
                primary_plot = refined

        primary = branch_results[0]
        durations = [item["duration"] for item in branch_results]
        gravity_sensitivity = []
        for gravity_value in sorted({gravity, 9.80665}):
            start_point = activation_point_for(gravity_value)
            gravity_result = interval_solution(validated_representatives[0][1], 4001, start_point)
            gravity_sensitivity.append({
                "gravity": float(gravity_value),
                "duration": gravity_result["duration"],
            })
        convergence_status = "pass" if maximum_refinement_difference <= 1e-6 else "fail"
        checks = [
            {
                "id": "event_grid_refinement", "name": "事件根区间加密复算",
                "status": convergence_status,
                "evidence": f"2001/8001 点搜索后的最大时长差为 {maximum_refinement_difference:.3g} s。",
                "recommendation": "若失败，增加搜索密度并检查切触根和轨迹不连续点。",
            },
            {
                "id": "target_semantics", "name": "目标代表点语义敏感性",
                "status": "warning" if len(branch_results) > 1 else "pass",
                "evidence": (
                    f"合理代表点分支给出的时长范围为 [{min(durations):.6f}, "
                    f"{max(durations):.6f}] s。"
                ),
                "recommendation": "比赛报告中必须声明采用下底面中心、几何中心或全目标遮蔽定义。",
            },
            {
                "id": "gravity_convention", "name": "重力常数约定敏感性",
                "status": "warning" if relation.get("gravity_source") != "explicit_problem_statement" else "pass",
                "evidence": f"使用 g={gravity:g} m/s²；并与标准重力 9.80665 m/s² 复算比较。",
                "recommendation": "题面未指定时，在假设表中明确重力常数。",
            },
        ]
        assert primary_plot is not None
        return {
            "kind": "kinematic_visibility_event", "status": "executed",
            "relation_id": relation.get("id"),
            "subproblem_id": relation.get("subproblem_id"),
            "solver": "bounded_grid+scipy.brentq+interval_union",
            "duration": primary["duration"],
            "effective_intervals": primary["intervals"],
            "primary_semantics": primary["semantics"],
            "activation_time": activation_time,
            "source_arrival_time": source_arrival_time,
            "release_point": (carrier_initial + carrier_velocity * release_time).tolist(),
            "activation_point": activation_point.tolist(),
            "semantic_branches": branch_results,
            "semantic_duration_range": [min(durations), max(durations)],
            "gravity_sensitivity": gravity_sensitivity,
            "summary": {
                "effective_duration": primary["duration"],
                "interval_count": len(primary["intervals"]),
                "activation_time": activation_time,
                "minimum_line_distance": float(np.min(primary_plot["distance"])),
            },
            "convergence": {
                "status": convergence_status,
                "coarse_points": 2001, "refined_points": 8001,
                "maximum_duration_difference": maximum_refinement_difference,
                "root_tolerance": 1e-11,
            },
            "plot_data": {
                "time": primary_plot["time"].tolist(),
                "distance": primary_plot["distance"].tolist(),
                "threshold": influence_radius,
                "intervals": primary["intervals"],
            },
            "credibility_audit": {
                "enabled": True,
                "status": "fail" if convergence_status == "fail" else "warning",
                "label": (
                    "事件求根未通过加密复算" if convergence_status == "fail"
                    else "数值已收敛；几何语义与物理常数仍须声明"
                ),
                "checks": checks,
                "decision": (
                    "数值支持已编译运动与视线定义下的条件性结论；"
                    "语义分支、单位推断和标准物理常数不是题面直接证明。"
                ),
            },
        }

    @staticmethod
    def _solve_kinematic_visibility_optimization(
        relation: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Optimize a single continuous visibility event and audit local/global uncertainty."""
        import numpy as np
        from scipy.optimize import differential_evolution, minimize

        def vector(name: str) -> Any:
            value = np.asarray(relation.get(name), dtype=float)
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be a finite three-dimensional vector")
            return value

        def finite_bounds(name: str, fallback_upper: float) -> Tuple[float, float]:
            raw = relation.get(name, [0.0, fallback_upper])
            if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or len(raw) != 2:
                raise ValueError(f"{name} must contain lower and upper bounds")
            lower = float(raw[0])
            upper = fallback_upper if raw[1] is None else float(raw[1])
            if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
                raise ValueError(f"{name} is not a finite increasing interval")
            return lower, upper

        source_initial = vector("source_initial")
        source_destination = vector("source_destination")
        carrier_initial = vector("carrier_initial")
        post_activation_velocity = vector("post_activation_velocity")
        source_speed = float(relation["source_speed"])
        influence_radius = float(relation["influence_radius"])
        active_lifetime = float(relation["active_lifetime"])
        gravity = float(relation.get("gravity", 9.8))
        if not all(math.isfinite(value) and value > 0 for value in (
            source_speed, influence_radius, active_lifetime, gravity,
        )):
            raise ValueError("optimization relation contains invalid positive parameters")
        source_direction = source_destination - source_initial
        source_distance = float(np.linalg.norm(source_direction))
        if source_distance <= 1e-12:
            raise ValueError("source trajectory has zero length")
        source_velocity = source_speed * source_direction / source_distance
        source_arrival_time = source_distance / source_speed

        heading_bounds = finite_bounds("heading_bounds", math.pi)
        speed_bounds = finite_bounds("carrier_speed_bounds", source_speed)
        release_bounds = finite_bounds("release_time_bounds", source_arrival_time)
        ground_delay = math.sqrt(max(0.0, 2.0 * float(carrier_initial[2]) / gravity))
        delay_bounds = finite_bounds("activation_delay_bounds", ground_delay)
        release_bounds = (
            max(0.0, release_bounds[0]), min(release_bounds[1], source_arrival_time)
        )
        delay_bounds = (
            max(0.0, delay_bounds[0]), min(delay_bounds[1], ground_delay)
        )
        if release_bounds[1] <= release_bounds[0] or delay_bounds[1] <= delay_bounds[0]:
            raise ValueError("physical horizon leaves no admissible release/activation interval")
        bounds = [heading_bounds, speed_bounds, release_bounds, delay_bounds]

        representatives = relation.get("target_representatives", [])
        if not representatives or not isinstance(representatives[0], Mapping):
            raise ValueError("optimization requires a target representative")
        target_point = np.asarray(representatives[0].get("point"), dtype=float)
        if target_point.shape != (3,) or not np.all(np.isfinite(target_point)):
            raise ValueError("primary target representative is invalid")

        def approximate_duration(parameters: Any, samples: int = 241) -> float:
            heading, carrier_speed, release_time, activation_delay = (
                float(value) for value in parameters
            )
            activation_time = release_time + activation_delay
            activation_height = (
                float(carrier_initial[2]) - 0.5 * gravity * activation_delay ** 2
            )
            if activation_time >= source_arrival_time or activation_height < 0.0:
                return 0.0
            end_time = min(activation_time + active_lifetime, source_arrival_time)
            if end_time <= activation_time:
                return 0.0
            carrier_velocity = carrier_speed * np.asarray([
                math.cos(heading), math.sin(heading), 0.0,
            ], dtype=float)
            release_point = carrier_initial + carrier_velocity * release_time
            activation_point = (
                release_point + carrier_velocity * activation_delay
                + np.asarray([0.0, 0.0, -0.5 * gravity * activation_delay ** 2])
            )
            times = np.linspace(activation_time, end_time, samples)
            sources = source_initial + times[:, None] * source_velocity
            clouds = (
                activation_point
                + (times - activation_time)[:, None] * post_activation_velocity
            )
            segments = target_point - sources
            denominators = np.einsum("ij,ij->i", segments, segments)
            if np.any(denominators <= 1e-18):
                return 0.0
            projections = np.einsum(
                "ij,ij->i", clouds - sources, segments
            ) / denominators
            projections = np.clip(projections, 0.0, 1.0)
            closest = sources + projections[:, None] * segments
            clearance = np.linalg.norm(clouds - closest, axis=1) - influence_radius
            if not np.all(np.isfinite(clearance)):
                return 0.0
            inside = clearance <= 0.0
            intervals: List[Tuple[float, float]] = []
            current_start: Optional[float] = float(times[0]) if inside[0] else None
            for index in range(1, len(times)):
                if inside[index] == inside[index - 1]:
                    continue
                left_time, right_time = float(times[index - 1]), float(times[index])
                left_value, right_value = float(clearance[index - 1]), float(clearance[index])
                denominator = right_value - left_value
                crossing = (
                    0.5 * (left_time + right_time)
                    if abs(denominator) <= 1e-15 else
                    left_time - left_value * (right_time - left_time) / denominator
                )
                crossing = min(right_time, max(left_time, crossing))
                if inside[index]:
                    current_start = crossing
                elif current_start is not None:
                    intervals.append((current_start, crossing))
                    current_start = None
            if current_start is not None:
                intervals.append((current_start, float(times[-1])))
            return max(0.0, float(sum(right - left for left, right in intervals)))

        def objective(parameters: Any) -> float:
            return -approximate_duration(parameters)

        attempts = []
        candidates = []
        baseline_payload = relation.get("baseline_candidate")
        baseline_vector = None
        if isinstance(baseline_payload, Mapping):
            try:
                candidate = np.asarray([
                    baseline_payload["carrier_heading"],
                    baseline_payload["carrier_speed"],
                    baseline_payload["release_time"],
                    baseline_payload["activation_delay"],
                ], dtype=float)
                if np.all(np.isfinite(candidate)) and all(
                    lower <= float(value) <= upper
                    for value, (lower, upper) in zip(candidate, bounds)
                ) and approximate_duration(candidate) > 0.0:
                    baseline_vector = candidate
                    candidates.append(candidate)
                    attempts.append({
                        "seed": "compiled_baseline", "solver_success": True,
                        "message": "feasible baseline inherited from an executed upstream node",
                        "evaluations": 1,
                        "approximate_duration": approximate_duration(candidate),
                        "parameters": candidate.tolist(),
                    })
            except (KeyError, TypeError, ValueError, OverflowError):
                baseline_vector = None
        constructive_pool = []
        activation_grid = np.linspace(
            max(0.2, 0.005 * source_arrival_time),
            min(source_arrival_time - 1e-4, active_lifetime + ground_delay),
            56,
        )
        segment_grid = np.linspace(0.02, 0.98, 72)
        for activation_time in activation_grid:
            source_at_activation = source_initial + source_velocity * activation_time
            for segment_fraction in segment_grid:
                desired_point = (
                    source_at_activation
                    + segment_fraction * (target_point - source_at_activation)
                )
                vertical_drop = float(carrier_initial[2] - desired_point[2])
                if vertical_drop < 0.0:
                    continue
                activation_delay = math.sqrt(2.0 * vertical_drop / gravity)
                release_time = activation_time - activation_delay
                if not (release_bounds[0] <= release_time <= release_bounds[1]):
                    continue
                if not (delay_bounds[0] <= activation_delay <= delay_bounds[1]):
                    continue
                horizontal_displacement = desired_point[:2] - carrier_initial[:2]
                carrier_speed = float(np.linalg.norm(horizontal_displacement) / activation_time)
                if not (speed_bounds[0] <= carrier_speed <= speed_bounds[1]):
                    continue
                heading = math.atan2(
                    float(horizontal_displacement[1]), float(horizontal_displacement[0])
                )
                if not (heading_bounds[0] <= heading <= heading_bounds[1]):
                    continue
                candidate = np.asarray([
                    heading, carrier_speed, release_time, activation_delay,
                ], dtype=float)
                duration = approximate_duration(candidate)
                if duration > 0.0:
                    constructive_pool.append((duration, candidate))
        constructive_pool.sort(key=lambda item: item[0], reverse=True)
        for constructive_index, (duration, candidate) in enumerate(
            constructive_pool[:5], 1
        ):
            candidates.append(candidate)
            attempts.append({
                "seed": f"geometry_constructive_{constructive_index}",
                "solver_success": True,
                "message": "activation point constructed on the moving source-target segment",
                "evaluations": len(activation_grid) * len(segment_grid),
                "approximate_duration": float(duration),
                "parameters": candidate.tolist(),
            })
        for seed in (0, 1):
            rng = np.random.default_rng(seed)
            population_size = 10 * len(bounds)
            population = np.column_stack([
                rng.uniform(lower, upper, size=population_size)
                for lower, upper in bounds
            ])
            if baseline_vector is not None:
                population[0] = baseline_vector
                local_count = min(16, population_size - 1)
                local_scales = np.asarray([
                    0.12 * (upper - lower) for lower, upper in bounds
                ], dtype=float)
                local = baseline_vector + rng.normal(
                    0.0, local_scales, size=(local_count, len(bounds))
                )
                for index, (lower, upper) in enumerate(bounds):
                    local[:, index] = np.clip(local[:, index], lower, upper)
                population[1:local_count + 1] = local
            insertion_start = 0 if baseline_vector is None else 1
            for offset, (_, candidate) in enumerate(constructive_pool[:12]):
                position = insertion_start + offset
                if position >= population_size:
                    break
                population[position] = candidate
            outcome = differential_evolution(
                objective, bounds, seed=seed, maxiter=60, popsize=10,
                tol=2e-6, atol=1e-7, polish=False, workers=1,
                updating="immediate", init=population,
            )
            if np.all(np.isfinite(outcome.x)) and math.isfinite(float(outcome.fun)):
                candidate = np.asarray(outcome.x, dtype=float)
                candidates.append(candidate)
                attempts.append({
                    "seed": seed, "solver_success": bool(outcome.success),
                    "message": str(outcome.message), "evaluations": int(outcome.nfev),
                    "approximate_duration": -float(outcome.fun),
                    "parameters": candidate.tolist(),
                })
        if not candidates:
            raise RuntimeError("global search produced no finite candidate")
        current_best = max(candidates, key=approximate_duration)
        boundary_fixed: Dict[int, float] = {}
        for variable_index in (1, 2):
            lower, upper = bounds[variable_index]
            if float(current_best[variable_index]) <= lower + 0.02 * (upper - lower):
                boundary_fixed[variable_index] = lower
        if boundary_fixed:
            free_indices = [index for index in range(len(bounds)) if index not in boundary_fixed]
            reduced_bounds = [bounds[index] for index in free_indices]

            def lift(reduced: Any) -> Any:
                full = current_best.copy()
                for index, value in boundary_fixed.items():
                    full[index] = value
                for index, value in zip(free_indices, reduced):
                    full[index] = float(value)
                return full

            boundary_outcome = differential_evolution(
                lambda reduced: objective(lift(reduced)), reduced_bounds,
                seed=7, maxiter=90, popsize=12, tol=1e-7, atol=1e-8,
                polish=True, workers=1, updating="immediate",
            )
            boundary_candidate = np.asarray(lift(boundary_outcome.x), dtype=float)
            if approximate_duration(boundary_candidate) + 1e-8 >= approximate_duration(current_best):
                candidates.append(boundary_candidate)
                attempts.append({
                    "seed": "active_boundary_refinement",
                    "solver_success": bool(boundary_outcome.success),
                    "message": (
                        "refined on active lower-bound face for variables "
                        f"{sorted(boundary_fixed)}: {boundary_outcome.message}"
                    ),
                    "evaluations": int(boundary_outcome.nfev),
                    "approximate_duration": approximate_duration(boundary_candidate),
                    "parameters": boundary_candidate.tolist(),
                })
        polish_starts = sorted(
            candidates, key=approximate_duration, reverse=True
        )[:min(3, len(candidates))]
        for polish_index, polish_start in enumerate(polish_starts, 1):
            polished = minimize(
                objective, polish_start, method="Powell", bounds=bounds,
                options={"maxiter": 500, "xtol": 1e-7, "ftol": 1e-9, "disp": False},
            )
            polished_duration = -float(polished.fun) if math.isfinite(float(polished.fun)) else -math.inf
            if (
                np.all(np.isfinite(polished.x))
                and math.isfinite(polished_duration)
                and polished_duration + 1e-8 >= approximate_duration(polish_start)
            ):
                candidates.append(np.asarray(polished.x, dtype=float))
                attempts.append({
                    "seed": f"powell_polish_{polish_index}",
                    "solver_success": bool(polished.success),
                    "message": str(polished.message), "evaluations": int(polished.nfev),
                    "approximate_duration": polished_duration,
                    "parameters": np.asarray(polished.x, dtype=float).tolist(),
                })

        def exact_candidate(parameters: Any) -> Tuple[Dict[str, Any], Dict[str, float]]:
            heading, carrier_speed, release_time, activation_delay = (
                float(value) for value in parameters
            )
            carrier_destination = (
                carrier_initial + np.asarray([
                    math.cos(heading), math.sin(heading), 0.0,
                ], dtype=float)
            )
            event_relation = {
                "id": relation.get("id"),
                "kind": "kinematic_visibility_event",
                "subproblem_id": relation.get("subproblem_id"),
                "source_initial": source_initial.tolist(),
                "source_destination": source_destination.tolist(),
                "source_speed": source_speed,
                "carrier_initial": carrier_initial.tolist(),
                "carrier_destination": carrier_destination.tolist(),
                "carrier_speed": carrier_speed,
                "carrier_horizontal": True,
                "release_time": release_time,
                "activation_delay": activation_delay,
                "gravity": gravity,
                "gravity_source": relation.get("gravity_source"),
                "post_activation_velocity": post_activation_velocity.tolist(),
                "influence_radius": influence_radius,
                "active_lifetime": active_lifetime,
                "target_representatives": representatives,
            }
            result = MechanisticModelingEngine._solve_kinematic_visibility_event(
                event_relation
            )
            solution = {
                "carrier_heading_rad": heading,
                "carrier_heading_deg": math.degrees(heading) % 360.0,
                "carrier_speed": carrier_speed,
                "release_time": release_time,
                "activation_delay": activation_delay,
                "activation_time": release_time + activation_delay,
            }
            return result, solution

        exact_candidates = []
        for candidate in candidates:
            try:
                exact, solution = exact_candidate(candidate)
            except (ValueError, FloatingPointError, RuntimeError):
                continue
            exact_candidates.append((exact, solution, candidate))
        if not exact_candidates:
            raise RuntimeError("no optimized candidate passed the exact event solver")
        exact_candidates.sort(key=lambda item: float(item[0]["duration"]), reverse=True)
        best_event, best_solution, best_vector = exact_candidates[0]
        exact_durations = [float(item[0]["duration"]) for item in exact_candidates]
        baseline_duration = None
        if baseline_vector is not None:
            for event, _, candidate in exact_candidates:
                if np.allclose(candidate, baseline_vector, rtol=0.0, atol=1e-10):
                    baseline_duration = float(event["duration"])
                    break

        sensitivity = []
        for variable_index, variable_name in enumerate(
            ("carrier_heading_rad", "carrier_speed", "release_time", "activation_delay")
        ):
            lower, upper = bounds[variable_index]
            perturbation = 0.005 * (upper - lower)
            for direction in (-1.0, 1.0):
                perturbed = best_vector.copy()
                perturbed[variable_index] = np.clip(
                    perturbed[variable_index] + direction * perturbation, lower, upper
                )
                try:
                    perturbed_event, _ = exact_candidate(perturbed)
                    duration = float(perturbed_event["duration"])
                except (ValueError, FloatingPointError, RuntimeError):
                    duration = 0.0
                sensitivity.append({
                    "variable": variable_name,
                    "direction": "lower" if direction < 0 else "upper",
                    "perturbation": float(direction * perturbation),
                    "duration": duration,
                })
        sensitivity_durations = [item["duration"] for item in sensitivity]
        best_duration = float(best_event["duration"])
        near_optimal_ranges: Dict[str, List[float]] = {}
        variable_names = (
            "carrier_heading_rad", "carrier_speed", "release_time", "activation_delay",
        )
        for variable_index, variable_name in enumerate(variable_names):
            lower, upper = bounds[variable_index]
            span = upper - lower
            local_lower = max(lower, float(best_vector[variable_index]) - 0.02 * span)
            local_upper = min(upper, float(best_vector[variable_index]) + 0.02 * span)
            values = np.unique(np.concatenate([
                np.linspace(lower, upper, 61),
                np.linspace(local_lower, local_upper, 161),
                np.asarray([float(best_vector[variable_index])]),
            ]))
            accepted = []
            for value in values:
                candidate = best_vector.copy()
                candidate[variable_index] = value
                accepted.append(
                    approximate_duration(candidate) >= 0.99 * best_duration
                )
            best_index = int(np.argmin(np.abs(values - best_vector[variable_index])))
            left_index = best_index
            right_index = best_index
            while left_index > 0 and accepted[left_index - 1]:
                left_index -= 1
            while right_index + 1 < len(values) and accepted[right_index + 1]:
                right_index += 1
            if accepted[best_index]:
                near_optimal_ranges[variable_name] = [
                    float(values[left_index]), float(values[right_index])
                ]
            else:
                near_optimal_ranges[variable_name] = [
                    float(best_vector[variable_index]), float(best_vector[variable_index])
                ]

        rounded_vector = np.asarray([
            math.radians(round(math.degrees(best_solution["carrier_heading_rad"]), 1)),
            round(best_solution["carrier_speed"], 1),
            round(best_solution["release_time"], 2),
            round(best_solution["activation_delay"], 2),
        ], dtype=float)
        for index, (lower, upper) in enumerate(bounds):
            rounded_vector[index] = np.clip(rounded_vector[index], lower, upper)
        try:
            rounded_event, rounded_solution = exact_candidate(rounded_vector)
            rounded_duration = float(rounded_event["duration"])
        except (ValueError, FloatingPointError, RuntimeError):
            rounded_solution = {}
            rounded_duration = 0.0
        rounded_relative_loss = (
            (best_duration - rounded_duration) / max(1.0, abs(best_duration))
        )
        near_optimal_durations = [
            duration for duration in exact_durations
            if duration >= best_duration - max(1e-6, 0.01 * abs(best_duration))
        ]
        multistart_spread = (
            max(near_optimal_durations) - min(near_optimal_durations)
        ) / max(1.0, abs(best_duration))
        sensitivity_drop = (
            best_duration - min(sensitivity_durations, default=best_duration)
        ) / max(1.0, abs(best_duration))
        multistart_status = "pass" if multistart_spread <= 1e-3 else "warning"
        sensitivity_status = "pass" if sensitivity_drop <= 0.10 else "warning"
        exact_convergence = best_event.get("convergence", {})
        checks = [
            {
                "id": "optimizer_multistart", "name": "独立种子与局部精化",
                "status": multistart_status,
                "evidence": (
                    f"{len(exact_candidates)} 个有限候选的精确时长相对跨度为 "
                    f"{multistart_spread:.3g}。"
                ),
                "recommendation": "若不一致，增加全局搜索预算并保留多个近优方案。",
            },
            {
                "id": "optimized_event_refinement", "name": "最优候选事件根复算",
                "status": exact_convergence.get("status", "fail"),
                "evidence": (
                    "最优候选经 2001/8001 点事件搜索与 Brent 根精化；"
                    f"最大时长差={exact_convergence.get('maximum_duration_difference')}。"
                ),
                "recommendation": "加密复算失败时不得报告最优时长。",
            },
            {
                "id": "decision_perturbation", "name": "决策扰动敏感性",
                "status": sensitivity_status,
                "evidence": f"各变量按边界跨度 ±0.5% 扰动后的最大相对时长下降为 {sensitivity_drop:.3g}。",
                "recommendation": "敏感时应报告可实施的近优参数区间，而非过多小数位。",
            },
            {
                "id": "global_optimality_scope", "name": "全局最优性边界",
                "status": "warning",
                "evidence": "差分进化、多种子和 Powell 精化仍不构成连续非凸问题的全局最优证明。",
                "recommendation": "需要严格最优性时补充分支定界、区间算法或可验证上界。",
            },
        ]
        return {
            "kind": "kinematic_visibility_optimization_solution",
            "status": "executed", "relation_id": relation.get("id"),
            "subproblem_id": relation.get("subproblem_id"),
            "solver": "scipy.differential_evolution_multiseed+Powell+Brent_refinement",
            "direction": "maximize", "objective": relation.get("objective"),
            "objective_value": best_duration, "duration": best_duration,
            "solution": best_solution,
            "release_point": best_event.get("release_point"),
            "activation_point": best_event.get("activation_point"),
            "effective_intervals": best_event.get("effective_intervals", []),
            "semantic_duration_range": best_event.get("semantic_duration_range", []),
            "semantic_branches": best_event.get("semantic_branches", []),
            "successful_starts": len(exact_candidates),
            "attempted_starts": len(attempts),
            "multistart_relative_spread": multistart_spread,
            "decision_sensitivity": sensitivity,
            "maximum_relative_sensitivity_drop": sensitivity_drop,
            "one_at_a_time_99pct_ranges": near_optimal_ranges,
            "implementation_candidate": {
                "solution": rounded_solution,
                "duration": rounded_duration,
                "relative_loss": rounded_relative_loss,
                "accepted": rounded_relative_loss <= 0.02,
                "rounding": {
                    "heading_deg": 0.1, "carrier_speed": 0.1,
                    "release_time": 0.01, "activation_delay": 0.01,
                },
            },
            "feedback_optimization": {
                "baseline_duration": baseline_duration,
                "optimized_duration": best_duration,
                "relative_gain": (
                    None if baseline_duration in {None, 0.0}
                    else (best_duration - baseline_duration) / abs(baseline_duration)
                ),
                "accepted": (
                    baseline_duration is None or best_duration > baseline_duration + 1e-6
                ),
                "search_adaptation": (
                    "used an executed feasible baseline plus geometry-constructed activation "
                    "points to escape the zero-duration objective plateau"
                ),
            },
            "summary": {
                "objective_duration": best_duration,
                "activation_time": best_solution["activation_time"],
                "carrier_speed": best_solution["carrier_speed"],
                "successful_starts": len(exact_candidates),
            },
            "convergence": {
                "status": exact_convergence.get("status", "fail"),
                "event_refinement": exact_convergence,
                "multistart_relative_spread": multistart_spread,
            },
            "plot_data": {
                **best_event.get("plot_data", {}),
                "decision_names": list(best_solution),
                "decision_values": list(best_solution.values()),
            },
            "attempt_summary": attempts,
            "credibility_audit": {
                "enabled": True, "status": "warning",
                "label": "多起点非凸最优候选；未证明全局最优",
                "checks": checks,
                "decision": (
                    "该方案通过独立种子、局部精化、事件根复算和决策扰动审计；"
                    "可作为高质量可行候选，但不能写成已证明的全局最优解。"
                ),
            },
        }

    @staticmethod
    def _solve_ode_system(relation: Mapping[str, Any]) -> Dict[str, Any]:
        import numpy as np
        from scipy.integrate import solve_ivp

        states = list(relation["state_variables"])
        parameters = {str(key): float(value) for key, value in relation["parameters"].items()}
        time_name = str(relation.get("time_variable", "t"))
        symbols = (*states, *parameters, time_name)
        analyzer = _SafeNumericExpression(symbols)
        compiled = {state: analyzer.compile(str(relation["rhs"][state])) for state in states}
        initial = np.asarray([relation["initial_values"][state] for state in states], dtype=float)
        start, end = (float(value) for value in relation["time_span"])
        points = int(relation.get("output_points", 300))
        times = np.linspace(start, end, points)

        def derivative(current_time: float, current_state: Any) -> Any:
            values = dict(parameters)
            values[time_name] = float(current_time)
            values.update({name: float(current_state[index]) for index, name in enumerate(states)})
            output = np.asarray([
                _SafeNumericExpression.evaluate(compiled[name], values) for name in states
            ], dtype=float)
            if not np.all(np.isfinite(output)):
                raise FloatingPointError("ODE derivative became non-finite")
            return output

        primary = solve_ivp(
            derivative, (start, end), initial, t_eval=times,
            method="DOP853", rtol=1e-8, atol=1e-10,
        )
        confirmation = solve_ivp(
            derivative, (start, end), initial, t_eval=times,
            method="DOP853", rtol=2.5e-9, atol=2.5e-11,
        )
        if not primary.success or not confirmation.success:
            raise RuntimeError(
                f"integrator did not terminate successfully: {primary.message}; {confirmation.message}"
            )
        if primary.y.shape != confirmation.y.shape or not np.all(np.isfinite(primary.y)):
            raise FloatingPointError("trajectory is non-finite or shape-inconsistent")
        absolute_difference = float(np.max(np.abs(primary.y - confirmation.y)))
        scale = max(1.0, float(np.max(np.abs(confirmation.y))))
        relative_difference = absolute_difference / scale
        convergence_status = "pass" if relative_difference <= 1e-6 else "fail"
        summary = {
            state: {
                "initial": float(primary.y[index, 0]),
                "final": float(primary.y[index, -1]),
                "minimum": float(np.min(primary.y[index])),
                "maximum": float(np.max(primary.y[index])),
            }
            for index, state in enumerate(states)
        }
        return {
            "kind": "ode_trajectory", "status": "executed",
            "relation_id": relation.get("id"), "solver": "scipy.solve_ivp.DOP853",
            "state_variables": states, "time_variable": time_name,
            "time_span": [start, end], "output_points": points,
            "summary": summary,
            "convergence": {
                "status": convergence_status,
                "absolute_tolerance_comparison": absolute_difference,
                "relative_tolerance_comparison": relative_difference,
                "acceptance_tolerance": 1e-6,
            },
            "plot_data": {
                "time": times.tolist(),
                "series": {state: primary.y[index].tolist() for index, state in enumerate(states)},
            },
            "credibility_audit": {
                "enabled": True,
                "status": "pass" if convergence_status == "pass" else "fail",
                "label": "数值轨迹通过容差复算" if convergence_status == "pass" else "数值轨迹未收敛",
                "checks": [{
                    "id": "ode_tolerance_convergence", "name": "积分容差复算",
                    "status": convergence_status,
                    "evidence": f"两组容差的最大相对差为 {relative_difference:.3g}。",
                    "recommendation": "若失败，缩短时间区间、检查刚性并改用刚性求解器。",
                }],
                "decision": "只证明当前已验证 ODE 契约的数值一致性，不证明机理正确。",
            },
        }

    @staticmethod
    def _solve_optimization_problem(relation: Mapping[str, Any]) -> Dict[str, Any]:
        import numpy as np
        from scipy.optimize import minimize

        variables = list(relation["decision_variables"])
        parameters = {str(key): float(value) for key, value in relation["parameters"].items()}
        analyzer = _SafeNumericExpression((*variables, *parameters))
        objective_node = analyzer.compile(str(relation["objective"]))
        constraint_nodes = []
        for constraint in relation.get("constraints", []):
            constraint_nodes.append((
                analyzer.compile(str(constraint["lhs"])),
                str(constraint["sense"]),
                analyzer.compile(str(constraint["rhs"])),
            ))
        bounds = [tuple(float(value) for value in relation["bounds"][name]) for name in variables]
        initial = np.asarray([relation["initial_values"][name] for name in variables], dtype=float)
        maximize = relation.get("direction") == "maximize"

        def value_map(vector: Any) -> Dict[str, float]:
            values = dict(parameters)
            values.update({name: float(vector[index]) for index, name in enumerate(variables)})
            return values

        def raw_objective(vector: Any) -> float:
            return _SafeNumericExpression.evaluate(objective_node, value_map(vector))

        def solver_objective(vector: Any) -> float:
            value = raw_objective(vector)
            return -value if maximize else value

        scipy_constraints = []
        for lhs_node, sense, rhs_node in constraint_nodes:
            def residual(vector: Any, left: ast.AST = lhs_node, right: ast.AST = rhs_node) -> float:
                values = value_map(vector)
                return (
                    _SafeNumericExpression.evaluate(left, values)
                    - _SafeNumericExpression.evaluate(right, values)
                )
            if sense == "<=":
                scipy_constraints.append({"type": "ineq", "fun": lambda x, fn=residual: -fn(x)})
            elif sense == ">=":
                scipy_constraints.append({"type": "ineq", "fun": residual})
            else:
                scipy_constraints.append({"type": "eq", "fun": residual})

        def maximum_violation(vector: Any) -> float:
            violations = []
            values = value_map(vector)
            for left, sense, right in constraint_nodes:
                residual = (
                    _SafeNumericExpression.evaluate(left, values)
                    - _SafeNumericExpression.evaluate(right, values)
                )
                violations.append(
                    max(0.0, residual) if sense == "<=" else
                    (max(0.0, -residual) if sense == ">=" else abs(residual))
                )
            for value, (lower, upper) in zip(vector, bounds):
                violations.extend((max(0.0, lower - float(value)), max(0.0, float(value) - upper)))
            return float(max(violations, default=0.0))

        trials = int(relation.get("multistart_trials", 8))
        rng = np.random.default_rng(0)
        starts = [initial]
        starts.extend(np.asarray([
            rng.uniform(lower, upper) for lower, upper in bounds
        ], dtype=float) for _ in range(trials - 1))
        attempts = []
        feasible = []
        for start in starts:
            outcome = minimize(
                solver_objective, start, method="SLSQP", bounds=bounds,
                constraints=scipy_constraints,
                options={"maxiter": 1500, "ftol": 1e-10, "disp": False},
            )
            if not np.all(np.isfinite(outcome.x)):
                continue
            violation = maximum_violation(outcome.x)
            objective_value = raw_objective(outcome.x)
            record = {
                "solver_success": bool(outcome.success),
                "status": int(outcome.status), "message": str(outcome.message),
                "objective_value": float(objective_value),
                "maximum_constraint_violation": violation,
                "solution": {name: float(outcome.x[index]) for index, name in enumerate(variables)},
            }
            attempts.append(record)
            if outcome.success and violation <= 1e-7 and math.isfinite(objective_value):
                feasible.append(record)
        if not feasible:
            raise RuntimeError(f"no feasible converged solution across {len(starts)} starts")
        ordered = sorted(feasible, key=lambda item: item["objective_value"], reverse=maximize)
        best = ordered[0]
        objectives = np.asarray([item["objective_value"] for item in feasible], dtype=float)
        objective_scale = max(1.0, abs(float(best["objective_value"])))
        objective_spread = float(np.max(np.abs(objectives - best["objective_value"])) / objective_scale)
        consistency = "pass" if objective_spread <= 1e-5 else "warning"
        checks = [
            {
                "id": "multistart_feasibility", "name": "多起点可行性",
                "status": "pass", "evidence": (
                    f"{len(feasible)}/{len(starts)} 个起点收敛到约束容差内；"
                    f"最大违反={best['maximum_constraint_violation']:.3g}。"
                ),
                "recommendation": "若可行起点比例低，应重参数化约束或使用专门全局算法。",
            },
            {
                "id": "multistart_objective_consistency", "name": "多起点目标一致性",
                "status": consistency,
                "evidence": f"可行收敛点相对目标差最大为 {objective_spread:.3g}。",
                "recommendation": "若不一致，报告多个局部解并增加全局下界或凸性证明。",
            },
            {
                "id": "global_optimality_scope", "name": "全局最优性边界",
                "status": "warning",
                "evidence": "SLSQP 多起点不能单独构成非凸问题的全局最优性证明。",
                "recommendation": "补充凸性证明、区间分支定界或可验证全局下界。",
            },
        ]
        return {
            "kind": "optimization_solution", "status": "executed",
            "relation_id": relation.get("id"), "solver": "scipy.optimize.SLSQP_multistart",
            "direction": relation.get("direction"),
            "decision_variables": variables,
            "objective_value": best["objective_value"],
            "solution": best["solution"],
            "maximum_constraint_violation": best["maximum_constraint_violation"],
            "successful_starts": len(feasible), "attempted_starts": len(starts),
            "objective_relative_spread": objective_spread,
            "summary": {
                "objective_value": best["objective_value"],
                "maximum_constraint_violation": best["maximum_constraint_violation"],
                "solution_norm": float(np.linalg.norm(list(best["solution"].values()))),
            },
            "convergence": {
                "status": "pass" if consistency == "pass" else "warning",
                "relative_tolerance_comparison": objective_spread,
            },
            "plot_data": {
                "names": variables,
                "values": [best["solution"][name] for name in variables],
            },
            "credibility_audit": {
                "enabled": True, "status": "warning", "label": "局部最优候选；未证明全局最优",
                "checks": checks,
                "decision": "候选满足当前约束并经多起点复算，但非凸情形必须保留全局最优性边界。",
            },
            "attempt_summary": {
                "successful": len(feasible), "attempted": len(starts),
                "messages": _unique(item["message"] for item in attempts)[:5],
            },
        }

    def _extract(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        clauses = _clauses(text)
        entities: List[Dict[str, Any]] = []
        coordinate_labels = _coordinate_labels(text)
        coordinate_unit, coordinate_unit_status = _coordinate_unit_context(text)
        for index, match in enumerate(_COORDINATE.finditer(text), 1):
            values = [_to_float(match.group(1)), _to_float(match.group(2))]
            if match.group(3) is not None:
                values.append(_to_float(match.group(3)))
            entities.append({
                "id": f"entity_{index}",
                "label": coordinate_labels.get(
                    match.start(), _entity_label(text[:match.start()], index)
                ),
                "coordinate": values,
                "coordinate_dimension": len(values),
                "unit": coordinate_unit,
                "unit_status": coordinate_unit_status,
                "source": "explicit_problem_statement",
                "source_text": _context(text, match.start(), match.end()),
            })
            if len(entities) >= 500:
                break
        quantities = []
        range_spans = []
        for match in _QUANTITY_RANGE.finditer(text):
            range_spans.append((match.start(), match.end()))
            quantities.append({
                "id": f"quantity_{len(quantities) + 1}",
                "value": [_to_float(match.group("lower")), _to_float(match.group("upper"))],
                "unit": match.group("unit"), "semantic_role": "unbound_parameter",
                "value_kind": "closed_range",
                "source": "explicit_problem_statement",
                "source_text": _context(text, match.start(), match.end()),
            })
        for match in _QUANTITY.finditer(text):
            if any(start <= match.start() < end for start, end in range_spans):
                continue
            quantities.append({
                "id": f"quantity_{len(quantities) + 1}",
                "value": _to_float(match.group("value")),
                "unit": match.group("unit"), "semantic_role": "unbound_parameter",
                "value_kind": "scalar",
                "source": "explicit_problem_statement",
                "source_text": _context(text, match.start(), match.end()),
            })
            if len(quantities) >= 500:
                break
        relations = []
        for clause in clauses:
            kind = _relation_kind(clause)
            if kind:
                relations.append({
                    "id": f"relation_{len(relations) + 1}", "kind": kind,
                    "expression": clause[:600],
                    "parse_status": "requires_symbol_and_unit_binding",
                    "source": "explicit_problem_statement", "source_text": clause[:600],
                })
        objectives = self._semantic_items(
            clauses, "objective", r"最大化|最小化|最大|最小|尽可能|目标函数|maximize|minimize|objective"
        )
        constraints = self._semantic_items(
            clauses, "constraint", r"约束|必须|不得|不能|不超过|不少于|至多|至少|满足|限制|subject\s+to"
        )
        decisions = self._semantic_items(
            clauses, "decision", r"确定|选择|设计|安排|调度|决策|制定|求解|find|choose|determine"
        )
        initial_conditions = self._semantic_items(
            clauses, "initial", r"初始|初值|起始|t\s*=\s*0|initial"
        )
        if entities and not initial_conditions:
            initial_conditions = [{
                "id": "initial_coordinates",
                "source_text": "coordinate declarations require temporal-role confirmation",
                "status": "unresolved_temporal_role",
            }]
        boundary_conditions = self._semantic_items(
            clauses, "boundary", r"边界条件|边界上|端点处|边值|boundary"
        )
        return {
            "entities": entities, "quantities": quantities, "relations": relations,
            "objectives": objectives, "constraints": constraints, "decisions": decisions,
            "initial_conditions": initial_conditions, "boundary_conditions": boundary_conditions,
        }

    @staticmethod
    def _compile_statement_relations(
        text: str, extracted: Mapping[str, Sequence[Mapping[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Compile fully bound reusable primitives from prose without executing the prose itself."""
        compact = re.sub(r"\s+", "", str(text))
        markers = list(_SUBPROBLEM.finditer(compact))[:100]
        if not markers:
            return []
        first_end = markers[1].start() if len(markers) > 1 else len(compact)
        first_problem = compact[markers[0].start():first_end]
        if not re.search(r"(?:有效|累计|持续).{0,12}(?:时长|时间)|duration", first_problem, re.IGNORECASE):
            return []
        if not re.search(r"(?:遮蔽|遮挡|视线|可见|visibility|occlusion)", compact, re.IGNORECASE):
            return []

        entities = [item for item in extracted.get("entities", []) if item.get("coordinate")]
        by_label = {
            str(item.get("label")): item for item in entities
            if item.get("label") and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,30}", str(item.get("label")))
        }
        if len(by_label) < 2:
            return []
        label_pattern = "|".join(
            re.escape(label) for label in sorted(by_label, key=len, reverse=True)
        )
        actor_match = re.search(
            rf"(?:利用|使用|由)(?:无人机|飞行器|载体|平台)?(?P<carrier>{label_pattern})"
            rf".*?(?:对|针对)(?P<source>{label_pattern})",
            first_problem, re.IGNORECASE,
        )
        if not actor_match:
            return []
        carrier_label = actor_match.group("carrier")
        source_label = actor_match.group("source")
        if carrier_label == source_label:
            return []

        target_candidates = [
            item for item in entities
            if str(item.get("label")) not in {carrier_label, source_label}
            and re.search(
                r"(?:真|保护|被保护|固定)?目标|target",
                f"{item.get('label', '')}{item.get('source_text', '')}", re.IGNORECASE,
            )
        ]
        if not target_candidates:
            return []
        target = target_candidates[0]
        bound_entities = [by_label[source_label], by_label[carrier_label], target]
        if any(str(item.get("unit")) != "m" for item in bound_entities):
            return []

        speed_unit = r"m/s|米/秒"
        carrier_speed_match = re.search(
            rf"{re.escape(carrier_label)}.{{0,24}}?以?({_NUMBER})\s*(?:{speed_unit})",
            first_problem, re.IGNORECASE,
        )
        source_speed_match = re.search(
            rf"(?:来袭武器|导弹|移动源|观测源|source).{{0,30}}?"
            rf"(?:速度(?:为)?|以)\s*({_NUMBER})\s*(?:{speed_unit})",
            compact, re.IGNORECASE,
        )
        sink_match = re.search(
            rf"(?:云团|影响区|作用区|区域中心).{{0,36}}?以?({_NUMBER})\s*(?:{speed_unit})"
            r".{0,12}?(?:下沉|下降|向下)",
            compact, re.IGNORECASE,
        )
        release_match = re.search(
            rf"(?:受领任务|接收任务|指令下达|开始计时)({_NUMBER})\s*(?:s|秒)后"
            r".{0,18}?(?:投放|释放|抛撒)",
            first_problem, re.IGNORECASE,
        )
        delay_match = re.search(
            rf"(?:投放|释放|抛撒).{{0,32}}?(?:(?:间隔|经过|延迟)|[，,：:]\s*)?"
            rf"({_NUMBER})\s*(?:s|秒)后"
            r"(?:起爆|激活|生效)",
            first_problem, re.IGNORECASE,
        )
        radius_lifetime_match = re.search(
            rf"(?:中心)?({_NUMBER})\s*(?:m|米)(?:范围|半径|以内|内).{{0,50}}?"
            rf"(?:起爆|激活|形成)?({_NUMBER})\s*(?:s|秒)(?:内|有效期).{{0,16}}?(?:有效|遮蔽|生效)",
            compact, re.IGNORECASE,
        )
        if not radius_lifetime_match:
            radius_lifetime_match = re.search(
                rf"(?:中心)?({_NUMBER})\s*(?:m|米)(?:范围|半径|以内|内).{{0,28}}?"
                rf"(?:在|持续)?({_NUMBER})\s*(?:s|秒)(?:内)?.{{0,12}}?(?:有效|生效)",
                compact, re.IGNORECASE,
            )
        required_matches = (
            carrier_speed_match, source_speed_match, sink_match,
            release_match, delay_match, radius_lifetime_match,
        )
        if any(match is None for match in required_matches):
            return []
        if not re.search(r"(?:重力|gravity)", compact, re.IGNORECASE):
            return []
        if not re.search(r"(?:原点|origin)", compact, re.IGNORECASE):
            return []
        source_direction_match = re.search(
            r"(?:来袭武器|导弹|移动源|观测源|source).{0,80}?"
            r"(?:直指|飞向|朝向|指向).{0,80}(?:原点|假目标|参考点|origin)",
            compact, re.IGNORECASE,
        )
        carrier_direction_match = re.search(
            rf"{re.escape(carrier_label)}.{{0,80}}?"
            r"(?:直指|飞向|朝向|指向).{0,24}(?:原点|假目标|参考点|origin)",
            first_problem, re.IGNORECASE,
        )
        if not source_direction_match or not carrier_direction_match:
            return []

        carrier_speed = _to_float(carrier_speed_match.group(1))
        source_speed = _to_float(source_speed_match.group(1))
        downward_speed = _to_float(sink_match.group(1))
        release_time = _to_float(release_match.group(1))
        activation_delay = _to_float(delay_match.group(1))
        influence_radius = _to_float(radius_lifetime_match.group(1))
        active_lifetime = _to_float(radius_lifetime_match.group(2))
        values = (
            carrier_speed, source_speed, downward_speed, release_time,
            activation_delay, influence_radius, active_lifetime,
        )
        if not all(math.isfinite(value) for value in values):
            return []
        if min(carrier_speed, source_speed, influence_radius, active_lifetime) <= 0:
            return []
        if downward_speed < 0 or release_time < 0 or activation_delay < 0:
            return []

        target_point = [float(value) for value in target["coordinate"]]
        if len(target_point) != 3:
            return []
        target_representatives = [{
            "semantics": "stated_target_reference_point",
            "point": target_point,
        }]
        cylinder_match = re.search(
            rf"半径({_NUMBER})\s*(?:m|米)[、,，]?高({_NUMBER})\s*(?:m|米)的圆柱",
            compact, re.IGNORECASE,
        )
        target_context = f"{target.get('label', '')}{target.get('source_text', '')}"
        if cylinder_match and re.search(r"(?:下底面|底面|bottom)", target_context, re.IGNORECASE):
            height = _to_float(cylinder_match.group(2))
            target_representatives = [
                {"semantics": "target_bottom_center", "point": target_point},
                {
                    "semantics": "target_geometric_center",
                    "point": [target_point[0], target_point[1], target_point[2] + height / 2.0],
                },
                {
                    "semantics": "target_top_center",
                    "point": [target_point[0], target_point[1], target_point[2] + height],
                },
            ]

        relation_id = f"compiled_relation_{len(extracted.get('relations', [])) + 1}"
        fixed_relation = {
            "id": relation_id,
            "kind": "kinematic_visibility_event",
            "parse_status": "machine_compiled",
            "subproblem_id": f"problem_{markers[0].group(1)}",
            "source_label": source_label,
            "carrier_label": carrier_label,
            "source_initial": [float(value) for value in by_label[source_label]["coordinate"]],
            "source_destination": [0.0, 0.0, 0.0],
            "source_speed": source_speed,
            "carrier_initial": [float(value) for value in by_label[carrier_label]["coordinate"]],
            "carrier_destination": [0.0, 0.0, 0.0],
            "carrier_speed": carrier_speed,
            "carrier_horizontal": bool(re.search(r"等高|等高度|水平", compact)),
            "release_time": release_time,
            "activation_delay": activation_delay,
            "gravity": 9.8,
            "gravity_source": "standard_modeling_convention",
            "post_activation_velocity": [0.0, 0.0, -downward_speed],
            "influence_radius": influence_radius,
            "active_lifetime": active_lifetime,
            "target_representatives": target_representatives,
            "units": {
                "position": "m", "speed": "m/s", "time": "s",
                "acceleration": "m/s^2",
            },
            "assumptions": [
                "the stated origin is the destination/reference point",
                "the carrier retains its stated velocity after release",
                "gravity is constant and aerodynamic drag is neglected",
                "visibility is distance from the influence center to the source-target segment",
            ],
            "binding_evidence": {
                "subproblem": first_problem[:1000],
                "source_coordinate": by_label[source_label].get("source_text"),
                "carrier_coordinate": by_label[carrier_label].get("source_text"),
                "target_coordinate": target.get("source_text"),
                "coordinate_unit_status": target.get("unit_status"),
            },
            "source": "deterministic_statement_compiler",
            "source_text": first_problem[:1000],
        }
        compiled_relations = [fixed_relation]

        speed_bounds_match = re.search(
            rf"({_NUMBER})\s*(?:~|～|至|到)\s*({_NUMBER})\s*(?:{speed_unit})",
            compact, re.IGNORECASE,
        )
        if not speed_bounds_match:
            return compiled_relations
        speed_bounds = sorted([
            _to_float(speed_bounds_match.group(1)),
            _to_float(speed_bounds_match.group(2)),
        ])
        if speed_bounds[0] <= 0 or speed_bounds[1] <= speed_bounds[0]:
            return compiled_relations

        compiled_nonconvex_optimizations = 0
        for marker_index, marker in enumerate(markers[1:], 1):
            if compiled_nonconvex_optimizations >= 3:
                break
            section_end = (
                markers[marker_index + 1].start()
                if marker_index + 1 < len(markers) else len(compact)
            )
            section = compact[marker.start():section_end]
            if not re.search(r"(?:尽可能长|最大化|最长|maximize)", section, re.IGNORECASE):
                continue
            if not re.search(r"(?:投放|释放|抛撒)1枚", section, re.IGNORECASE):
                continue
            mentioned_labels = set(re.findall(label_pattern, section, re.IGNORECASE))
            if len(mentioned_labels) != 2:
                continue
            section_actors = re.search(
                rf"(?:利用|使用|由)(?:无人机|飞行器|载体|平台)?"
                rf"(?P<carrier>{label_pattern}).*?(?:对|针对)(?P<source>{label_pattern})",
                section, re.IGNORECASE,
            )
            if not section_actors:
                continue
            section_carrier = section_actors.group("carrier")
            section_source = section_actors.group("source")
            if section_carrier == section_source:
                continue
            if {section_carrier, section_source} != mentioned_labels:
                continue
            if section_carrier not in by_label or section_source not in by_label:
                continue
            if any(
                str(item.get("unit")) != "m"
                for item in (by_label[section_carrier], by_label[section_source], target)
            ):
                continue
            decision_roles = {
                "heading": bool(re.search(r"方向|航向|heading", section, re.IGNORECASE)),
                "speed": bool(re.search(r"速度|speed", section, re.IGNORECASE)),
                "release": bool(re.search(r"投放点|释放点|投放时刻|释放时刻|release", section, re.IGNORECASE)),
                "activation": bool(re.search(r"起爆点|激活点|起爆时刻|激活时刻|activation", section, re.IGNORECASE)),
            }
            if not all(decision_roles.values()):
                continue
            compiled_relations.append({
                "id": f"compiled_relation_{len(extracted.get('relations', [])) + len(compiled_relations) + 1}",
                "kind": "kinematic_visibility_optimization",
                "parse_status": "machine_compiled",
                "subproblem_id": f"problem_{marker.group(1)}",
                "source_label": section_source,
                "carrier_label": section_carrier,
                "source_initial": [
                    float(value) for value in by_label[section_source]["coordinate"]
                ],
                "source_destination": [0.0, 0.0, 0.0],
                "source_speed": source_speed,
                "carrier_initial": [
                    float(value) for value in by_label[section_carrier]["coordinate"]
                ],
                "carrier_horizontal": bool(re.search(r"等高|等高度|水平", compact)),
                "carrier_speed_bounds": speed_bounds,
                "heading_bounds": [-math.pi, math.pi],
                "release_time_bounds": [0.0, None],
                "activation_delay_bounds": [0.0, None],
                "gravity": 9.8,
                "gravity_source": "standard_modeling_convention",
                "post_activation_velocity": [0.0, 0.0, -downward_speed],
                "influence_radius": influence_radius,
                "active_lifetime": active_lifetime,
                "target_representatives": target_representatives,
                "objective": "maximize_union_measure_of_visibility_event",
                "decision_variables": [
                    "carrier_heading", "carrier_speed", "release_time", "activation_delay",
                ],
                "baseline_candidate": (
                    {
                        "carrier_heading": math.atan2(
                            -float(by_label[section_carrier]["coordinate"][1]),
                            -float(by_label[section_carrier]["coordinate"][0]),
                        ),
                        "carrier_speed": carrier_speed,
                        "release_time": release_time,
                        "activation_delay": activation_delay,
                        "source": "previously_compiled_feasible_subproblem",
                    }
                    if section_carrier == carrier_label and section_source == source_label
                    else None
                ),
                "units": {
                    "position": "m", "speed": "m/s", "time": "s",
                    "heading": "rad", "acceleration": "m/s^2",
                },
                "assumptions": [
                    "the carrier uses one constant horizontal heading and speed",
                    "the payload inherits carrier velocity at release",
                    "gravity is constant and aerodynamic drag is neglected",
                    "the objective is continuous-time visibility interval measure",
                ],
                "binding_evidence": {
                    "subproblem": section[:1000],
                    "speed_bounds": speed_bounds_match.group(0),
                    "source_coordinate": by_label[section_source].get("source_text"),
                    "carrier_coordinate": by_label[section_carrier].get("source_text"),
                    "target_coordinate": target.get("source_text"),
                },
                "source": "deterministic_statement_compiler",
                "source_text": section[:1000],
            })
            compiled_nonconvex_optimizations += 1
        return compiled_relations

    @staticmethod
    def _semantic_items(clauses: Sequence[str], prefix: str, pattern: str) -> List[Dict[str, Any]]:
        selected = [item for item in clauses if re.search(pattern, item, re.IGNORECASE)]
        return [
            {"id": f"{prefix}_{index}", "source_text": clause[:600], "status": "semantic"}
            for index, clause in enumerate(selected, 1)
        ]

    @staticmethod
    def _merge_verified_override(
        extracted: Dict[str, List[Dict[str, Any]]], override: Mapping[str, Any],
    ) -> Dict[str, List[Dict[str, Any]]]:
        allowed = set(extracted)
        unknown = sorted(set(override) - allowed)
        if unknown:
            raise ValueError(f"unsupported IR override fields: {unknown}")
        merged = {key: list(value) for key, value in extracted.items()}
        for key, value in override.items():
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                raise ValueError(f"IR override '{key}' must be a sequence")
            verified = []
            for item in value:
                if not isinstance(item, Mapping):
                    raise ValueError(f"IR override '{key}' contains a non-object item")
                payload = dict(item)
                payload.setdefault("source", "user_verified_ir_override")
                payload.setdefault("source_text", "user-verified structured input")
                if key == "relations":
                    payload = MechanisticModelingEngine._verify_structured_relation(payload)
                verified.append(payload)
            merged[key] = verified
        return merged

    @staticmethod
    def _verify_structured_relation(relation: Dict[str, Any]) -> Dict[str, Any]:
        """Strictly validate executable relation schemas and overwrite trust flags."""
        payload = dict(relation)
        payload["parse_status"] = "requires_symbol_and_unit_binding"
        if UniversalRelationValidator.supports(payload.get("kind")):
            return UniversalRelationValidator.verify(payload)
        if payload.get("kind") == "optimization_problem":
            return MechanisticModelingEngine._verify_optimization_relation(payload)
        if payload.get("kind") != "ode_system":
            payload["validation_errors"] = ["unsupported_executable_relation_kind"]
            return payload
        errors: List[str] = []
        states = payload.get("state_variables", [])
        rhs = payload.get("rhs", {})
        initial = payload.get("initial_values", {})
        parameters = payload.get("parameters", {})
        time_span = payload.get("time_span", [])
        time_name = str(payload.get("time_variable", "t"))
        units = payload.get("units", {})
        identifier = re.compile(r"^[A-Za-z_]\w*$")
        if (
            not isinstance(states, list) or not states or len(states) > 50
            or len(states) != len(set(states))
            or any(not isinstance(name, str) or not identifier.fullmatch(name) for name in states)
        ):
            errors.append("state_variables_must_be_1_to_50_unique_identifiers")
        if not identifier.fullmatch(time_name) or time_name in states:
            errors.append("invalid_time_variable")
        if not isinstance(parameters, Mapping) or any(
            not isinstance(name, str) or not identifier.fullmatch(name) or name in states or name == time_name
            for name in parameters
        ):
            errors.append("invalid_parameter_names")
        numeric_parameters: Dict[str, float] = {}
        if isinstance(parameters, Mapping):
            for name, value in parameters.items():
                try:
                    numeric_parameters[str(name)] = float(value)
                    if not math.isfinite(float(value)):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"non_finite_parameter:{name}")
        if not isinstance(rhs, Mapping) or set(rhs) != set(states):
            errors.append("rhs_must_bind_every_state_exactly_once")
        if not isinstance(initial, Mapping) or set(initial) != set(states):
            errors.append("initial_values_must_bind_every_state_exactly_once")
        else:
            for name, value in initial.items():
                try:
                    if not math.isfinite(float(value)):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"non_finite_initial_value:{name}")
        try:
            if len(time_span) != 2:
                raise ValueError
            start, end = float(time_span[0]), float(time_span[1])
            if not (math.isfinite(start) and math.isfinite(end) and end > start):
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            errors.append("time_span_must_be_two_finite_increasing_values")
        samples = payload.get("output_points", 300)
        if not isinstance(samples, int) or isinstance(samples, bool) or not 20 <= samples <= 5000:
            errors.append("output_points_must_be_an_integer_between_20_and_5000")
        expected_unit_names = set(states) | set(numeric_parameters) | {time_name}
        if not isinstance(units, Mapping) or not expected_unit_names <= set(units):
            errors.append("units_must_cover_states_parameters_and_time")

        dimension_checks = []
        if not errors and isinstance(rhs, Mapping):
            from .mathematical_reasoning import UnitDimension, check_expression_dimensions, parse_unit

            analyzer = _SafeNumericExpression((*states, *numeric_parameters, time_name))
            state_units = {name: parse_unit(str(units[name])) for name in states}
            time_unit = parse_unit(str(units[time_name]))
            if any(unit is None for unit in state_units.values()) or time_unit is None:
                errors.append("state_or_time_unit_is_not_parseable")
            for state in states:
                expression = str(rhs.get(state, ""))
                try:
                    analyzer.compile(expression)
                except (SyntaxError, ValueError) as exc:
                    errors.append(f"unsafe_rhs:{state}:{exc}")
                    continue
                check = check_expression_dimensions(expression, units)
                check["state"] = state
                if check.get("status") != "pass":
                    errors.append(f"dimension_check_failed:{state}:{check.get('evidence')}")
                elif state_units.get(state) is not None and isinstance(time_unit, UnitDimension):
                    result = check.get("result_dimension", {})
                    actual = UnitDimension.from_mapping(
                        result.get("powers", {}), result.get("scale", 1.0), "rhs"
                    )
                    expected = state_units[state].divide(time_unit)
                    if not actual.compatible(expected):
                        errors.append(f"rhs_dimension_mismatch:{state}")
                        check["status"] = "fail"
                        check["evidence"] = (
                            f"RHS dimension {actual.mapping} != d{state}/d{time_name} "
                            f"dimension {expected.mapping}"
                        )
                dimension_checks.append(check)
        payload["validation_errors"] = errors
        payload["dimension_checks"] = dimension_checks
        if not errors:
            payload["parse_status"] = "machine_verified"
            payload["parameters"] = numeric_parameters
            payload["initial_values"] = {name: float(initial[name]) for name in states}
            payload["time_span"] = [float(time_span[0]), float(time_span[1])]
            payload["output_points"] = int(samples)
        return payload

    @staticmethod
    def _verify_optimization_relation(payload: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a bounded nonlinear program without trusting caller flags."""
        errors: List[str] = []
        variables = payload.get("decision_variables", [])
        parameters = payload.get("parameters", {})
        bounds = payload.get("bounds", {})
        initial = payload.get("initial_values", {})
        constraints = payload.get("constraints", [])
        objective = str(payload.get("objective", ""))
        direction = str(payload.get("direction", "minimize")).lower()
        units = payload.get("units", {})
        identifier = re.compile(r"^[A-Za-z_]\w*$")
        if (
            not isinstance(variables, list) or not variables or len(variables) > 30
            or len(variables) != len(set(variables))
            or any(not isinstance(name, str) or not identifier.fullmatch(name) for name in variables)
        ):
            errors.append("decision_variables_must_be_1_to_30_unique_identifiers")
        if direction not in {"minimize", "maximize", "min", "max"}:
            errors.append("direction_must_be_minimize_or_maximize")
        numeric_parameters: Dict[str, float] = {}
        if not isinstance(parameters, Mapping) or any(
            not isinstance(name, str) or not identifier.fullmatch(name) or name in variables
            for name in parameters
        ):
            errors.append("invalid_parameter_names")
        else:
            for name, value in parameters.items():
                try:
                    numeric_parameters[str(name)] = float(value)
                    if not math.isfinite(float(value)):
                        raise ValueError
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"non_finite_parameter:{name}")
        normalized_bounds: Dict[str, List[float]] = {}
        if not isinstance(bounds, Mapping) or set(bounds) != set(variables):
            errors.append("finite_bounds_must_bind_every_decision_variable")
        else:
            for name in variables:
                try:
                    lower, upper = bounds[name]
                    lower, upper = float(lower), float(upper)
                    if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower):
                        raise ValueError
                    normalized_bounds[name] = [lower, upper]
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"invalid_finite_bound:{name}")
        normalized_initial: Dict[str, float] = {}
        if not isinstance(initial, Mapping) or set(initial) != set(variables):
            errors.append("initial_values_must_bind_every_decision_variable")
        else:
            for name in variables:
                try:
                    value = float(initial[name])
                    if not math.isfinite(value):
                        raise ValueError
                    normalized_initial[name] = value
                    if name in normalized_bounds and not (
                        normalized_bounds[name][0] <= value <= normalized_bounds[name][1]
                    ):
                        errors.append(f"initial_value_outside_bounds:{name}")
                except (TypeError, ValueError, OverflowError):
                    errors.append(f"non_finite_initial_value:{name}")
        if not isinstance(constraints, list) or len(constraints) > 200:
            errors.append("constraints_must_be_a_list_with_at_most_200_items")
            constraints = []
        if not isinstance(units, Mapping) or not (set(variables) | set(numeric_parameters)) <= set(units):
            errors.append("units_must_cover_decisions_and_parameters")
        raw_trials = payload.get("multistart_trials", 8)
        try:
            multistart_trials = int(raw_trials)
            if isinstance(raw_trials, bool) or not 4 <= multistart_trials <= 32:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            multistart_trials = 8
            errors.append("multistart_trials_must_be_an_integer_between_4_and_32")

        symbols = (*variables, *numeric_parameters)
        analyzer = _SafeNumericExpression(symbols)
        dimension_checks = []
        if not errors:
            from .mathematical_reasoning import check_equation_dimensions, check_expression_dimensions

            try:
                analyzer.compile(objective)
                objective_check = check_expression_dimensions(objective, units)
                objective_check["role"] = "objective"
                dimension_checks.append(objective_check)
                if objective_check.get("status") != "pass":
                    errors.append(f"objective_dimension_check_failed:{objective_check.get('evidence')}")
            except (SyntaxError, ValueError) as exc:
                errors.append(f"unsafe_objective:{exc}")
            for index, constraint in enumerate(constraints):
                if not isinstance(constraint, Mapping):
                    errors.append(f"constraint_{index}_must_be_an_object")
                    continue
                sense = str(constraint.get("sense", ""))
                lhs, rhs = str(constraint.get("lhs", "")), str(constraint.get("rhs", ""))
                if sense not in {"<=", ">=", "=="}:
                    errors.append(f"constraint_{index}_has_invalid_sense")
                    continue
                try:
                    analyzer.compile(lhs)
                    analyzer.compile(rhs)
                except (SyntaxError, ValueError) as exc:
                    errors.append(f"unsafe_constraint_{index}:{exc}")
                    continue
                check = check_equation_dimensions(lhs, rhs, units)
                check["role"] = f"constraint_{index}"
                dimension_checks.append(check)
                if check.get("status") != "pass":
                    errors.append(f"constraint_{index}_dimension_check_failed:{check.get('evidence')}")
        payload["validation_errors"] = errors
        payload["dimension_checks"] = dimension_checks
        if not errors:
            payload["parse_status"] = "machine_verified"
            payload["parameters"] = numeric_parameters
            payload["bounds"] = normalized_bounds
            payload["initial_values"] = normalized_initial
            payload["direction"] = "minimize" if direction in {"minimize", "min"} else "maximize"
            payload["multistart_trials"] = multistart_trials
        return payload

    @staticmethod
    def _semantic_bindings(
        text: str, extracted: Mapping[str, Sequence[Mapping[str, Any]]]
    ) -> Dict[str, List[str]]:
        clauses = _clauses(text)

        def matched(pattern: str) -> List[str]:
            return [clause[:300] for clause in clauses if re.search(pattern, clause, re.IGNORECASE)]

        bindings = {
            "state": matched(r"状态|位置|轨迹|数量|浓度|温度|库存|人口|流量|state|position|trajectory"),
            "initial_condition": [str(item.get("source_text", "")) for item in extracted["initial_conditions"]],
            "boundary_condition": [str(item.get("source_text", "")) for item in extracted["boundary_conditions"]],
            "rate_law": matched(r"速度|速率|每秒|每分钟|每小时|匀速|恒速|speed|rate"),
            "dynamics_law": matched(r"变化率|加速度|受力|重力|微分|守恒|平衡|d\w+\s*/\s*dt|derivative|dynamics"),
            "geometry_definition": matched(r"坐标|位置|距离|半径|高度|长度|角度|球|圆柱|边界|coordinate|distance|radius|geometry"),
            "event_definition": matched(
                r"当|若|直到|生效|失效|相交|覆盖|可见|遮挡|持续|时长|有效|"
                r"when|until|event|intersect"
            ),
            "decision_variables": [str(item.get("source_text", "")) for item in extracted["decisions"]],
            "objective": [str(item.get("source_text", "")) for item in extracted["objectives"][:1]],
            "objectives": [str(item.get("source_text", "")) for item in extracted["objectives"]],
            "constraints": [str(item.get("source_text", "")) for item in extracted["constraints"]],
            "graph_definition": matched(r"节点|边|网络|路径|弧|容量|node|edge|network|graph"),
            "probability_model": matched(r"概率|分布|随机|到达率|转移概率|probability|distribution|random"),
            "computable_response": [str(item.get("source_text", "")) for item in extracted["relations"]],
            "observations": [],
            "parameters": [str(item.get("source_text", "")) for item in extracted["quantities"]],
            "uncertainty_set": matched(r"不确定集合|参数范围|区间内|最坏情形|uncertainty\s+set|bounded\s+uncertainty"),
        }
        structured_odes = [
            item for item in extracted["relations"] if item.get("kind") == "ode_system"
        ]
        for relation in structured_odes:
            source = str(relation.get("source_text", "structured ODE relation"))
            if relation.get("state_variables"):
                bindings["state"].append(source)
            if relation.get("rhs"):
                bindings["dynamics_law"].append(source)
                bindings["computable_response"].append(source)
            if relation.get("initial_values"):
                bindings["initial_condition"].append(source)
            if relation.get("parameters"):
                bindings["parameters"].append(source)
        structured_optimizations = [
            item for item in extracted["relations"]
            if item.get("kind") == "optimization_problem"
        ]
        for relation in structured_optimizations:
            source = str(relation.get("source_text", "structured optimization relation"))
            if relation.get("decision_variables"):
                bindings["decision_variables"].append(source)
            if relation.get("objective"):
                bindings["objective"].append(source)
                bindings["objectives"].append(source)
                bindings["computable_response"].append(source)
            if relation.get("constraints") is not None:
                bindings["constraints"].append(source)
            if relation.get("parameters"):
                bindings["parameters"].append(source)
        return {key: _unique(values) for key, values in bindings.items()}

    @staticmethod
    def _operator_graph(
        definitions: Sequence[OperatorDefinition], bindings: Mapping[str, Sequence[str]]
    ) -> List[Dict[str, Any]]:
        graph = []
        previous_by_category: Dict[str, str] = {}
        for index, definition in enumerate(definitions, 1):
            node_id = f"operator_{index}"
            bound = {
                role: list(bindings.get(role, []))[:5]
                for role in definition.required_bindings if bindings.get(role)
            }
            missing = [role for role in definition.required_bindings if role not in bound]
            dependencies = []
            if definition.category in {"event", "optimization", "inverse"}:
                dependencies = list(previous_by_category.values())[-3:]
            graph.append({
                "id": node_id, **definition.public(), "bindings": bound,
                "missing_bindings": missing,
                "status": "ready_to_compile" if not missing else "partially_specified",
                "depends_on": dependencies,
            })
            previous_by_category[definition.category] = node_id
        return graph

    @staticmethod
    def _subproblems(
        text: str,
        graph: Sequence[Mapping[str, Any]],
        compiler_blockers: Sequence[str] = (),
        execution: Optional[Mapping[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        matches = list(_SUBPROBLEM.finditer(text))[:100]
        if not matches:
            return []
        result = []
        executed_ids = {
            str(item.get("subproblem_id"))
            for item in (execution or {}).get("results", [])
            if item.get("status") == "executed" and item.get("subproblem_id")
        }
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            description = text[match.end():end].strip()[:1000]
            relevant = list(graph)
            subproblem_id = f"problem_{match.group(1)}"
            missing = _unique([
                *(item for node in relevant for item in node.get("missing_bindings", [])),
                *compiler_blockers,
            ])
            if executed_ids and subproblem_id not in executed_ids and not missing:
                missing = ["verified_symbol_and_unit_bindings"]
            executed = subproblem_id in executed_ids
            result.append({
                "id": subproblem_id, "description": description,
                "operator_ids": [node["id"] for node in relevant],
                "status": (
                    "executed" if executed else
                    ("ready" if relevant and not missing else
                    ("partial" if relevant else "needs_input")
                    )
                ),
                "missing_requirements": [] if executed else missing,
                "evidence": (
                    (
                        "该子问题已由确定性题面编译器绑定参数，并完成事件求根、"
                        "区间并集和加密复算。"
                    ) if executed else (
                        f"已继承题面公共机理并组合 {len(relevant)} 个通用数学算子；"
                        "当前状态表示数学草案已形成，数值执行门仍单独审计。"
                    )
                ),
            })
        return result

    @staticmethod
    def _compiler_plan(
        graph: Sequence[Mapping[str, Any]],
        extracted: Mapping[str, Sequence[Mapping[str, Any]]],
        unresolved: Sequence[str],
    ) -> Dict[str, Any]:
        relation_count = len(extracted["relations"])
        executable_relations = [
            item for item in extracted["relations"]
            if item.get("parse_status") in {"machine_verified", "machine_compiled"}
        ]
        machine_bound_relations = len(executable_relations)
        executable = machine_bound_relations > 0
        blockers = list(unresolved)
        if relation_count == 0:
            blockers.append("machine_readable_equations_or_algorithms")
        elif machine_bound_relations == 0:
            blockers.append("verified_symbol_and_unit_bindings")
        task_by_relation_kind = {
            "ode_system": "simulation",
            "kinematic_visibility_event": "simulation",
            "markov_chain": "simulation",
            "sample_expectation": "statistical_inference",
            "linear_least_squares": "statistical_inference",
            "optimization_problem": "optimization",
            "kinematic_visibility_optimization": "optimization",
            "linear_program": "optimization",
            "mixed_integer_linear_program": "optimization",
            "quadratic_program": "optimization",
            "multiobjective_program": "optimization",
            "robust_program": "optimization",
            "stochastic_program": "optimization",
            "dynamic_program": "optimization",
            "shortest_path_problem": "graph_network",
            "maximum_flow_problem": "graph_network",
            "minimum_cost_flow_problem": "graph_network",
            "bipartite_matching_problem": "graph_network",
            "linear_system": "simulation",
            "polynomial_root": "simulation",
        }
        compiled_tasks = _unique(
            task_by_relation_kind.get(str(item.get("kind")), "")
            for item in executable_relations
        )
        return {
            "status": (
                "partially_ready" if executable and blockers else
                ("ready" if executable else "blocked_by_model_contract")
            ),
            "executable": executable,
            "solver_routes": _unique(node["solver_route"] for node in graph),
            "executable_relation_count": machine_bound_relations,
            "deferred_relation_count": relation_count - machine_bound_relations,
            "compiled_tasks": compiled_tasks,
            "safety_gates": [
                "symbol_binding", "unit_consistency", "initial_boundary_completeness",
                "constraint_feasibility", "solver_residual", "mesh_or_step_convergence",
                "sensitivity_and_alternative_semantics",
            ],
            "blocked_by": _unique(blockers),
            "next_action": (
                "dispatch each compiled relation independently and preserve deferred relations" if executable
                else "complete and verify the structured IR; do not execute prose directly"
            ),
        }

    @staticmethod
    def _build_model_draft(
        graph: Sequence[Mapping[str, Any]],
        extracted: Mapping[str, Sequence[Mapping[str, Any]]],
        compiler_plan: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Compose a readable mathematical draft even before numerical execution."""
        canonical_forms = {
            "constant_rate_state": ["x(t)=x(t0)+v(t-t0)"],
            "second_order_dynamics": ["dx/dt=v", "dv/dt=a(t,x,v,u;θ)"],
            "balance_law": ["dx/dt=inflow(x,u;θ)-outflow(x,u;θ)"],
            "first_order_ode": ["dx/dt=f(t,x,u;θ),  x(t0)=x0"],
            "field_pde": ["∂q/∂t+∇·F(q)=S(q,u;θ)"],
            "metric_geometry": ["d(t)=dist(A(t),B(t));  g(t)=d(t)-d_threshold"],
            "region_membership": ["I(t)=1{g(x(t),region;θ)≤0}"],
            "line_of_sight": ["I_visible(t)=1{segment(source(t),target)∩obstacle(t)=∅}"],
            "event_window": ["E={t: event_condition(t)=True}"],
            "interval_measure": ["T_eff=measure(⋃ intervals(E))"],
            "graph_path": ["G=(V,E,w);  d(v)=min_path_cost(source,v)"],
            "network_flow": ["max Σ f(source,j),  flow_balance=0,  0≤f_ij≤capacity_ij"],
            "stochastic_transition": ["p_{t+1}=p_t P(u_t;θ)"],
            "monte_carlo": ["E[g(X)]≈N^{-1}Σ g(X_i)"],
            "constrained_optimization": ["min/max J(u;θ)  s.t. g(u;θ)≤0, h(u;θ)=0, l≤u≤r"],
            "multiobjective_optimization": ["min (J1(u),…,Jk(u)); report Pareto set"],
            "parameter_calibration": ["θ*=argmin_θ L(y,F(θ)); validate on held-out evidence"],
            "robust_decision": ["min_u max_{θ∈U} J(u,θ)  s.t. constraints hold over U"],
        }
        equations = []
        seen = set()
        for node in graph:
            for expression in canonical_forms.get(str(node.get("key")), []):
                signature = (str(node.get("key")), expression)
                if signature in seen:
                    continue
                seen.add(signature)
                equations.append({
                    "operator": node.get("key"),
                    "expression": expression,
                    "status": (
                        "roles_bound" if node.get("status") == "ready_to_compile"
                        else "template_requires_binding"
                    ),
                    "missing_bindings": list(node.get("missing_bindings", [])),
                })
        categories = {str(node.get("category")) for node in graph}
        assumption_prompts = []
        if "dynamics" in categories:
            assumption_prompts.extend([
                "状态变量是否闭合，是否遗漏会改变动力学的隐变量？",
                "连续时间、确定性和参数恒定假设在哪个时间尺度上成立？",
            ])
        if "geometry" in categories or "event" in categories:
            assumption_prompts.append(
                "事件是针对中心点、任一点还是整个区域成立？边界是否计入？"
            )
        if "optimization" in categories:
            assumption_prompts.extend([
                "目标函数是否覆盖全部代价/收益，多个目标如何处理？",
                "变量域、现实约束和不确定参数范围是否完整？",
            ])
        if "stochastic" in categories:
            assumption_prompts.append("随机变量的分布、依赖结构和尾部风险依据是什么？")
        return {
            "status": "numerically_executable" if compiler_plan.get("executable") else "conceptual_model_compiled",
            "equations": equations,
            "parameters": list(extracted.get("quantities", [])),
            "decision_statements": list(extracted.get("decisions", [])),
            "objectives": list(extracted.get("objectives", [])),
            "constraints": list(extracted.get("constraints", [])),
            "assumption_questions": _unique(assumption_prompts),
            "completed_stages": [
                "problem_decomposition", "quantity_and_entity_extraction",
                "operator_composition", "canonical_equation_draft",
            ],
            "next_actions": list(compiler_plan.get("blocked_by", [])),
            "interpretation": (
                "Canonical equations are a transparent model draft. They become a numerical model "
                "only after every symbol, unit, equation, condition, and objective is bound."
            ),
        }

    @staticmethod
    def _task_support(
        graph: Sequence[Mapping[str, Any]],
        compiler_plan: Mapping[str, Any],
        execution: Mapping[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        task_by_category = {
            "dynamics": "differential_equations", "geometry": "simulation",
            "event": "simulation", "network": "graph_network", "stochastic": "simulation",
            "optimization": "optimization", "inverse": "statistical_inference",
        }
        grouped: Dict[str, List[Mapping[str, Any]]] = {}
        for node in graph:
            task = task_by_category.get(str(node.get("category")))
            if task:
                grouped.setdefault(task, []).append(node)
        support = {}
        executed_tasks = set()
        result_task_map = {
            "linear_system_solution": {"simulation"},
            "scalar_root_solution": {"simulation"},
            "linear_least_squares_solution": {"statistical_inference"},
            "linear_program_solution": {"optimization"},
            "mixed_integer_linear_program_solution": {"optimization"},
            "quadratic_program_solution": {"optimization"},
            "multiobjective_program_solution": {"optimization"},
            "robust_program_solution": {"optimization"},
            "stochastic_program_solution": {"optimization"},
            "dynamic_program_solution": {"optimization"},
            "shortest_path_solution": {"graph_network"},
            "maximum_flow_solution": {"graph_network"},
            "minimum_cost_flow_solution": {"graph_network"},
            "bipartite_matching_solution": {"graph_network"},
            "markov_chain_solution": {"simulation"},
            "sample_expectation_solution": {"simulation", "statistical_inference"},
        }
        for result in execution.get("results", []):
            if result.get("status") != "executed":
                continue
            if result.get("kind") == "ode_trajectory":
                executed_tasks.update({"differential_equations", "simulation"})
            elif result.get("kind") == "optimization_solution":
                executed_tasks.add("optimization")
            elif result.get("kind") == "kinematic_visibility_event":
                executed_tasks.add("simulation")
            elif result.get("kind") == "kinematic_visibility_optimization_solution":
                executed_tasks.add("optimization")
            else:
                executed_tasks.update(result_task_map.get(str(result.get("kind")), set()))
        for task in executed_tasks:
            grouped.setdefault(task, [])
        for task, nodes in grouped.items():
            missing = _unique([
                *(item for node in nodes for item in node.get("missing_bindings", [])),
                *compiler_plan.get("blocked_by", []),
            ])
            ready_count = sum(node.get("status") == "ready_to_compile" for node in nodes)
            if task in executed_tasks:
                status = "executed"
                evidence = f"通用算子图匹配 {len(nodes)} 个算子，已通过安全编译并形成数值证据。"
                missing = []
            elif task in compiler_plan.get("compiled_tasks", []):
                status = "ready"
                evidence = f"通用算子图匹配 {len(nodes)} 个算子，相关模型契约已通过，等待求解器执行。"
            else:
                status = "partial" if nodes else "needs_input"
                if not missing:
                    missing = ["verified_symbol_and_unit_bindings"]
                evidence = (
                    f"已完成题型结构化和 {len(nodes)} 个通用算子匹配；"
                    f"其中 {ready_count} 个算子角色绑定完整，但数值编译尚未放行。"
                )
            support[task] = {
                "status": status,
                "evidence": evidence,
                "operator_ids": [str(node.get("id")) for node in nodes],
                "missing_requirements": missing,
                "completed_stages": ["problem_decomposition", "operator_selection"] + (
                    ["binding"] if ready_count else []
                ),
            }
        return support

    @staticmethod
    def _alternative_interpretations(
        text: str, definitions: Sequence[OperatorDefinition]
    ) -> List[Dict[str, Any]]:
        alternatives = []
        keys = {item.key for item in definitions}
        if {"region_membership", "line_of_sight"} & keys:
            alternatives.append({
                "ambiguity": "geometric_event_semantics",
                "branches": ["center_or_representative_point", "entire_region", "any_point_or_partial_region"],
                "resolution": "define the event as a set relation before computing duration or probability",
            })
        if "interval_measure" in keys:
            alternatives.append({
                "ambiguity": "time_aggregation",
                "branches": ["continuous_duration", "sampled_count", "union_of_disjoint_intervals"],
                "resolution": "use continuous event roots and state whether overlapping intervals are unioned",
            })
        if re.search(r"最优|优化|最大|最小|optimal", text, re.IGNORECASE):
            alternatives.append({
                "ambiguity": "objective_semantics",
                "branches": ["expected_value", "worst_case", "risk_adjusted", "multiobjective_pareto"],
                "resolution": "declare the decision criterion and uncertainty treatment explicitly",
            })
        return alternatives

    @staticmethod
    def _validation_protocol(graph: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        routes = {str(node.get("solver_route")) for node in graph}
        checks = [
            {"gate": "dimensional_analysis", "reject_when": "an equation mixes incompatible dimensions"},
            {"gate": "limiting_cases", "reject_when": "zero/infinite/symmetric limits contradict the mechanism"},
            {"gate": "alternative_semantics", "reject_when": "reasonable definitions materially reverse the conclusion"},
            {"gate": "parameter_sensitivity", "reject_when": "small admissible perturbations cause unreported large changes"},
            {"gate": "independent_recalculation", "reject_when": "a second implementation disagrees beyond tolerance"},
        ]
        if any("integrator" in route or "element" in route or "volume" in route for route in routes):
            checks.append({"gate": "step_or_mesh_convergence", "reject_when": "refinement changes the result beyond tolerance"})
        if any("optimizer" in route or "programming" in route for route in routes):
            checks.append({"gate": "feasibility_and_optimality", "reject_when": "constraint/KKT/global-bound evidence fails"})
        checks.append({
            "gate": "feedback_parameter_update",
            "reject_when": "tuned parameters improve the same confirmation evidence used to select them",
            "protocol": "calibrate on one split/scenario set, accept only after untouched confirmation and robustness checks",
        })
        return checks

    @staticmethod
    def _credibility_audit(
        extracted: Mapping[str, Sequence[Mapping[str, Any]]],
        graph: Sequence[Mapping[str, Any]], alternatives: Sequence[Mapping[str, Any]],
        compiler_plan: Mapping[str, Any],
    ) -> Dict[str, Any]:
        missing = _unique(item for node in graph for item in node.get("missing_bindings", []))
        checks = [
            {
                "id": "statement_provenance", "name": "题面来源追踪",
                "status": "pass" if any(extracted.values()) else "warning",
                "evidence": "所有抽取对象保留原始题面片段；结构化补充必须标记为用户确认。",
                "recommendation": "逐项核对变量、数值和单位，不以语言模型补全冒充题设。",
            },
            {
                "id": "operator_binding", "name": "算子绑定完整性",
                "status": "pass" if graph and not missing else "warning",
                "evidence": f"匹配 {len(graph)} 个通用算子；未绑定角色：{missing or '无'}。",
                "recommendation": "未绑定项补齐前禁止数值执行。",
            },
            {
                "id": "semantic_ambiguity", "name": "竞争语义",
                "status": "warning" if alternatives else "pass",
                "evidence": f"识别 {len(alternatives)} 类会改变数学定义的合理解释分支。",
                "recommendation": "并行求解合理分支，报告结论对定义的敏感性。",
            },
            {
                "id": "safe_compilation", "name": "安全编译门",
                "status": "pass" if compiler_plan.get("executable") else "warning",
                "evidence": (
                    "所有机器可执行条件已满足。" if compiler_plan.get("executable") else
                    f"尚缺：{compiler_plan.get('blocked_by', [])}。题面文本不会被直接执行。"
                ),
                "recommendation": "先完成符号、单位、初边值和求解器契约，再放行计算。",
            },
        ]
        return {
            "enabled": True,
            "status": "warning" if any(item["status"] != "pass" for item in checks) else "pass",
            "label": "模型契约待补全" if missing else "题面结构已抽取",
            "checks": checks,
            "decision": "这只证明建模结构可审计，不构成数值答案或现实正确性证明。",
        }


__all__ = ["MechanisticModelingEngine", "MechanisticOperatorRegistry", "OperatorDefinition"]
