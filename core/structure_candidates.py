"""Bounded, domain-independent structure proposals for incomplete problem contracts.

These are deliberately *not* solvers.  They are small search seeds that keep a
pure-text problem moving when the statement does not yet contain enough
information for numerical compilation.  Every proposal carries unresolved
obligations and is marked ``proposed_not_executed``; downstream gates must still
type-check, bind units, compile and validate it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from .candidate_gate import gate_candidates
from .primitive_graph import validate_primitive_graph


SCHEMA_VERSION = "mathmodel.structure-candidates/v1"

# A bounded vocabulary for unresolved mechanism holes.  This is only a search
# hint; it is not an assertion that any operator is valid for the task.
_DEFAULT_UNKNOWN_OPERATORS = ["add", "subtract", "multiply", "divide", "minimum", "maximum"]

_TEMPLATES: dict[str, tuple[dict[str, Any], ...]] = {
    "optimization": (
        {"family": "separable_objective", "operators": ["decision", "objective", "constraint"],
         "unknown_mechanisms": ["objective_terms", "feasible_region"],
         "questions": ["决策变量、目标函数和每条约束的定义与单位是什么？"]},
        {"family": "robust_scenario", "operators": ["decision", "scenario", "objective", "constraint"],
         "unknown_mechanisms": ["uncertainty_set", "scenario_weights"],
         "questions": ["参数不确定性应使用区间、情景集合还是概率分布？"]},
    ),
    "differential_equations": (
        {"family": "state_space_ode", "operators": ["state", "derivative", "equation", "initial_condition"],
         "unknown_mechanisms": ["state_transition", "initial_conditions"],
         "questions": ["状态变量是否闭合？初始条件、边界条件和时间尺度是什么？"]},
        {"family": "memory_or_forcing", "operators": ["state", "lag_or_input", "derivative", "equation"],
         "unknown_mechanisms": ["external_forcing_or_memory"],
         "questions": ["变化是否依赖历史状态或外部输入？"]},
    ),
    "simulation": (
        {"family": "event_simulation", "operators": ["state", "transition", "event", "interval_measure"],
         "unknown_mechanisms": ["transition_rule", "event_boundary"],
         "questions": ["事件的触发条件、边界语义和状态转移规则是什么？"]},
        {"family": "monte_carlo", "operators": ["distribution", "sample", "outcome", "interval_measure"],
         "unknown_mechanisms": ["input_distributions", "outcome_function"],
         "questions": ["随机输入的分布、相关性和样本数量如何确定？"]},
    ),
    "graph_network": (
        {"family": "shortest_path_or_flow", "operators": ["graph", "weight", "path_or_flow", "objective"],
         "unknown_mechanisms": ["node_semantics", "edge_semantics", "weight_definition"],
         "questions": ["节点、边、方向、权重以及起点/终点如何定义？"]},
        {"family": "network_intervention", "operators": ["graph", "centrality", "intervention", "objective"],
         "unknown_mechanisms": ["intervention_cost", "network_response"],
         "questions": ["允许改变哪些节点或边？干预成本和网络响应如何衡量？"]},
    ),
    "prediction_forecast": (
        {"family": "structural_forecast", "operators": ["time", "state", "trend", "observation", "interval"],
         "unknown_mechanisms": ["target_definition", "forecast_horizon", "observation_noise"],
         "questions": ["预测目标、预测时点、可用信息集和评价损失是什么？"]},
        {"family": "regime_or_lag", "operators": ["time", "lag", "regime", "observation", "interval"],
         "unknown_mechanisms": ["lag_structure", "regime_change"],
         "questions": ["是否存在滞后、季节性或机制切换？"]},
    ),
    "statistical_inference": (
        {"family": "estimand_and_error_model", "operators": ["variable", "estimand", "likelihood", "interval"],
         "unknown_mechanisms": ["sampling_unit", "error_model", "estimand"],
         "questions": ["估计对象、抽样单位、误差模型和可检验假设是什么？"]},
    ),
    "evaluation_ranking": (
        {"family": "multi_criteria_decision", "operators": ["indicator", "direction", "weight", "score", "sensitivity"],
         "unknown_mechanisms": ["indicator_direction", "weight_rule", "decision_set"],
         "questions": ["评价对象、指标方向、权重来源和可接受的权重敏感性是什么？"]},
    ),
}


def _graph_node(node_id: str, op: str, *, inputs: tuple[str, ...] = (),
                kind: str = "quantity", dimensions: Mapping[str, float] | None = None,
                attributes: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build a JSON-only primitive node for a proposal skeleton."""
    return {
        "id": node_id, "op": op, "inputs": list(inputs), "kind": kind,
        "dimensions": dict(dimensions or {"Q": 1}),
        "attributes": dict(attributes or {}),
    }


def _typed_graph_skeleton(task_type: str, family: str) -> dict[str, Any]:
    """Return a small typed graph; it is checked, never executed.

    The graph intentionally stops at unresolved nodes.  This gives the search
    layer a legal starting point without pretending that a missing mechanism,
    objective, or observation model has already been identified.
    """
    q = {"Q": 1}
    nodes: list[dict[str, Any]]
    outputs: list[str]
    if task_type == "differential_equations":
        nodes = [
            _graph_node("state", "variable", dimensions=q),
            _graph_node("time", "coordinate", kind="coordinate", dimensions={"T": 1}),
            _graph_node("rate", "derivative", inputs=("state", "time"),
                        dimensions={"Q": 1, "T": -1},
                        attributes={"initial_condition": "unbound"}),
            _graph_node("mechanism", "unknown_mechanism", inputs=("state",),
                        dimensions={"Q": 1, "T": -1},
                        attributes={"search_budget": 8, "candidate_language": "bounded_basis",
                                    "allowed_operators": list(_DEFAULT_UNKNOWN_OPERATORS),
                                    "properties": ["待由数据与边界条件检验"]}),
        ]
        outputs = ["rate", "mechanism"]
    elif task_type == "simulation" and family == "event_simulation":
        nodes = [
            _graph_node("state", "variable", dimensions=q),
            _graph_node("threshold", "constant", dimensions=q),
            _graph_node("condition", "equation", inputs=("state", "threshold"), kind="boolean",
                        dimensions=None),
            _graph_node("event", "event", inputs=("condition",), kind="event", dimensions=None,
                        attributes={"boundary_semantics": "unbound"}),
            _graph_node("transition", "unknown_mechanism", inputs=("state",), dimensions=q,
                        attributes={"search_budget": 8, "candidate_language": "bounded_transition",
                                    "allowed_operators": list(_DEFAULT_UNKNOWN_OPERATORS),
                                    "properties": ["事件边界与转移规则待确认"]}),
        ]
        outputs = ["event", "transition"]
    elif task_type == "graph_network":
        nodes = [_graph_node("network", "graph", kind="graph", dimensions=None,
                             attributes={"node_semantics": "unbound", "edge_semantics": "unbound"})]
        outputs = ["network"]
    elif task_type == "optimization":
        nodes = [
            _graph_node("decision", "variable", dimensions=q),
            _graph_node("objective", "unknown_mechanism", inputs=("decision",), dimensions=q,
                        attributes={"search_budget": 8, "candidate_language": "bounded_objective",
                                    "allowed_operators": list(_DEFAULT_UNKNOWN_OPERATORS),
                                    "properties": ["目标函数形式待确认"]}),
            _graph_node("feasible", "unknown_mechanism", inputs=("decision",), dimensions=q,
                        attributes={"search_budget": 8, "candidate_language": "bounded_constraint",
                                    "allowed_operators": list(_DEFAULT_UNKNOWN_OPERATORS),
                                    "properties": ["可行域与约束形式待确认"]}),
        ]
        outputs = ["objective", "feasible"]
    elif task_type == "statistical_inference":
        nodes = [
            _graph_node("quantity", "variable", dimensions=q),
            _graph_node("observation", "observation", inputs=("quantity",), dimensions=q,
                        attributes={"noise_model": "unbound"}),
            _graph_node("latent", "latent_state", dimensions=q,
                        attributes={"identification_strategy": "unbound"}),
        ]
        outputs = ["observation", "latent"]
    elif task_type == "prediction_forecast":
        nodes = [
            _graph_node("time", "coordinate", kind="coordinate", dimensions={"T": 1}),
            _graph_node("target", "variable", dimensions=q),
            _graph_node("observation", "observation", inputs=("target",), dimensions=q,
                        attributes={"identity": True}),
            _graph_node("mechanism", "unknown_mechanism", inputs=("target",), dimensions=q,
                        attributes={"search_budget": 8, "candidate_language": "bounded_forecast",
                                    "allowed_operators": list(_DEFAULT_UNKNOWN_OPERATORS),
                                    "properties": ["趋势、滞后与噪声结构待检验"]}),
        ]
        outputs = ["observation", "mechanism"]
    else:
        nodes = [
            _graph_node("quantity", "variable", dimensions=q),
            _graph_node("mechanism", "unknown_mechanism", inputs=("quantity",), dimensions=q,
                        attributes={"search_budget": 8, "candidate_language": "bounded_basis",
                                    "allowed_operators": list(_DEFAULT_UNKNOWN_OPERATORS),
                                    "properties": ["机制项待发现"]}),
        ]
        outputs = ["mechanism"]
    validation = validate_primitive_graph(nodes, output_ids=outputs)
    return {"nodes": nodes, "output_ids": outputs, "validation": validation}


def _candidate_signature(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    """Build a name-independent structural signature for diversity checks."""
    graph = candidate.get("primitive_graph", {})
    nodes = graph.get("nodes", []) if isinstance(graph, Mapping) else []
    graph_signature = tuple(sorted(
        (str(node.get("op")), str(node.get("kind")), tuple(sorted(node.get("inputs", []))))
        for node in nodes if isinstance(node, Mapping)
    ))
    return (
        tuple(sorted(str(item) for item in candidate.get("operators", []))),
        tuple(sorted(str(item) for item in candidate.get("unknown_mechanisms", []))),
        graph_signature,
    )


def _diversity_audit(candidates: list[Mapping[str, Any]]) -> dict[str, Any]:
    signatures = [_candidate_signature(candidate) for candidate in candidates]
    unique = len(set(signatures))
    duplicate_indices: list[list[int]] = []
    groups: dict[tuple[Any, ...], list[int]] = {}
    for index, signature in enumerate(signatures):
        groups.setdefault(signature, []).append(index)
    duplicate_indices = [indices for indices in groups.values() if len(indices) > 1]
    if len(candidates) < 2:
        status = "not_assessed"
    elif duplicate_indices:
        status = "collapsed"
    else:
        status = "structurally_distinct"
    return {
        "status": status,
        "candidate_count": len(candidates),
        "unique_structures": unique,
        "duplicate_groups": duplicate_indices,
        "policy": "name_independent_structural_screen_not_semantic_novelty_or_correctness_proof",
    }


def build_structure_candidates(
    problem_analysis: Mapping[str, Any],
    *,
    has_observations: bool = False,
    max_candidates: int = 3,
) -> dict[str, Any]:
    """Create bounded composition seeds without claiming execution or truth."""
    if not isinstance(problem_analysis, Mapping):
        raise TypeError("problem_analysis_must_be_mapping")
    if type(max_candidates) is not int or not 1 <= max_candidates <= 6:
        raise ValueError("invalid_candidate_budget")
    task_type = str(problem_analysis.get("task_type") or "evaluation_ranking")
    ranked = [
        str(item.get("task_type"))
        for item in problem_analysis.get("task_candidates", [])
        if isinstance(item, Mapping) and item.get("task_type") in _TEMPLATES
    ]
    task_types = list(dict.fromkeys([task_type, *ranked]))
    candidates: list[dict[str, Any]] = []
    questions: list[str] = []
    for current_type in task_types:
        for template in _TEMPLATES.get(current_type, ()):
            candidate_id = f"{current_type}_{template['family']}"
            graph = _typed_graph_skeleton(current_type, template["family"])
            candidates.append({
                "id": candidate_id,
                "task_type": current_type,
                "family": template["family"],
                "operators": list(template["operators"]),
                "unknown_mechanisms": list(template["unknown_mechanisms"]),
                "status": "proposed_not_executed",
                "source": "deterministic_primitive_composition",
                "numeric_execution": "available_after_contract_binding" if has_observations else "blocked_until_contract_and_observations",
                "hard_checks": {
                    "type": "pending", "unit": "pending", "shape": "pending",
                    "boundary": "pending", "constraint": "pending", "resource": "pending",
                },
                "assumptions": list(template["questions"]),
                "primitive_graph": graph,
            })
            questions.extend(template["questions"])
            if len(candidates) >= max_candidates:
                break
        if len(candidates) >= max_candidates:
            break
    if not candidates:
        graph = _typed_graph_skeleton(task_type, "unknown_mechanism")
        candidates = [{
            "id": "generic_unknown_structure",
            "task_type": task_type,
            "family": "unknown_mechanism",
            "operators": ["variable", "unknown_mechanism", "observation"],
            "unknown_mechanisms": ["task_structure"],
            "status": "proposed_not_executed",
            "source": "deterministic_primitive_composition",
            "numeric_execution": "blocked_until_contract_and_observations",
            "hard_checks": {
                "type": "pending", "unit": "pending", "shape": "pending",
                "boundary": "pending", "constraint": "pending", "resource": "pending",
            },
            "assumptions": ["题面尚不足以选择具体数学结构。"],
            "primitive_graph": graph,
        }]
        questions.append("题面尚不足以绑定可执行数学结构；请确认任务目标和观测/参数来源。")
    gate = gate_candidates(candidates)
    diversity = _diversity_audit(candidates)
    candidate_digest = hashlib.sha256(
        json.dumps(candidates, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "candidate_structures_only",
        "candidate_count": len(candidates),
        "candidate_digest": candidate_digest,
        "candidates": candidates,
        "preflight_gate": gate,
        "diversity_audit": diversity,
        "clarification_questions": list(dict.fromkeys(questions))[:3],
        "policy": "候选结构不是事实、数值答案、证明或求解器；必须经过完整编译和独立验证。",
    }


__all__ = ["SCHEMA_VERSION", "build_structure_candidates"]
