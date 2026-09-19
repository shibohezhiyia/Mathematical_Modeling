"""End-to-end modeling benchmark starting from prose and raw observations.

Unlike :mod:`core.automated_benchmark_suite`, these public tasks do not expose
an equation graph, ODE coefficients, an optimization matrix, or a join plan.
Systems must return both a model description and an independently checkable
answer.  The small bundled suite is a protocol/regression fixture, not a claim
of representative real-world coverage.
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import math
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .automated_benchmark import AutomatedBenchmarkCase
from .automated_benchmark_suite import score_automated_benchmark_case


_ARMS = ("frozen_old", "candidate_new", "simple_tool_baseline")


def _table(name: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"name": name, "format": "records", "rows": [dict(row) for row in rows]}


def build_modeling_benchmark_suite() -> tuple[AutomatedBenchmarkCase, ...]:
    """Return twelve structurally distinct tasks with hidden models and truth."""
    cases: list[AutomatedBenchmarkCase] = []

    for identifier, structure, xs, function, queries in (
        ("affine", "affine", [-3, -1, 0, 2, 4], lambda x: 2.5 * x - 1.0, [-2, 1, 5]),
        ("quadratic", "quadratic", [-3, -2, -1, 0, 1, 2, 3], lambda x: x * x - 2 * x + 0.5, [-2.5, 1.5, 4]),
    ):
        observations = [{"input": float(x), "response": float(function(x))} for x in xs]
        cases.append(AutomatedBenchmarkCase(
            f"modeling-algebra-{identifier}", "modeling_algebra",
            "根据观测识别输入与响应的简洁数学关系，并预测查询点；必须同时提交关系类型和预测值。",
            {"attachments": [_table("observations", observations)], "query_inputs": list(queries)},
            {"family": "modeling_algebra", "structure": structure,
             "input_variables": ["input"], "predictions": [float(function(x)) for x in queries],
             "coefficients": ([2.5, -1.0] if structure == "affine" else [1.0, -2.0, 0.5]),
             "tolerance": 1e-5},
            f"modeling-algebra-{structure}", source_group="generated-modeling-algebra",
        ))

    for identifier, structure, function, coefficients in (
        ("multivariate-affine", "multivariate_affine", lambda a, b: 1.0 + 2.0 * a - 0.5 * b,
         [1.0, 2.0, -0.5, 0.0]),
        ("bilinear", "bilinear", lambda a, b: 1.0 + 2.0 * a - 0.5 * b + 1.5 * a * b,
         [1.0, 2.0, -0.5, 1.5]),
    ):
        pairs = [(-2, -1), (-2, 1), (-1, 2), (0, -2), (0, 1), (1, -1), (1, 2), (2, 0), (2, 2)]
        queries = [[-1.5, 0.5], [0.5, -1.5], [2.5, 1.0]]
        cases.append(AutomatedBenchmarkCase(
            f"modeling-algebra-{identifier}", "modeling_algebra",
            "根据两项输入的原始观测识别简洁关系；候选可包含主效应和二阶交互，并预测查询组合。",
            {"attachments": [_table("observations", [
                {"x1": float(a), "x2": float(b), "response": float(function(a, b))} for a, b in pairs
            ])], "query_inputs": queries},
            {"family": "modeling_algebra", "structure": structure, "input_variables": ["x1", "x2"],
             "predictions": [float(function(a, b)) for a, b in queries], "coefficients": coefficients,
             "tolerance": 1e-5},
            f"modeling-algebra-{structure}", source_group="generated-modeling-algebra",
        ))

    decay_times = np.linspace(0.0, 2.0, 7)
    decay_values = 3.0 * np.exp(-0.7 * decay_times)
    forecast_times = [2.5, 3.0]
    cases.append(AutomatedBenchmarkCase(
        "modeling-ode-decay", "modeling_ode",
        "一项封闭系统实验记录了随时间单调衰减的状态。识别最简动力学机制并预测后续状态；提交机制类型、参数和轨迹。",
        {"attachments": [_table("observations", [{"time": float(t), "state": float(y)}
                                                   for t, y in zip(decay_times, decay_values)])],
         "query_times": forecast_times},
        {"family": "modeling_ode", "structure": "exponential_decay", "parameter": 0.7,
         "trajectory": [float(3.0 * math.exp(-0.7 * t)) for t in forecast_times], "tolerance": 2e-3},
        "modeling-ode-exponential-decay", source_group="generated-modeling-ode",
    ))
    logistic_times = np.linspace(0.0, 4.0, 9)
    capacity, rate, initial = 10.0, 0.8, 1.0
    logistic = capacity / (1.0 + ((capacity - initial) / initial) * np.exp(-rate * logistic_times))
    logistic_queries = [4.5, 5.0]
    cases.append(AutomatedBenchmarkCase(
        "modeling-ode-logistic", "modeling_ode",
        "种群在资源有限环境中增长并逐渐接近承载上限。根据原始观测识别动力学结构并预测后续状态；提交机制类型、参数和轨迹。",
        {"attachments": [_table("observations", [{"time": float(t), "state": float(y)}
                                                   for t, y in zip(logistic_times, logistic)])],
         "query_times": logistic_queries},
        {"family": "modeling_ode", "structure": "logistic_growth", "parameters": {"rate": rate, "capacity": capacity},
         "trajectory": [float(capacity / (1.0 + 9.0 * math.exp(-rate * t))) for t in logistic_queries],
         "tolerance": 3e-2},
        "modeling-ode-logistic-growth", source_group="generated-modeling-ode",
    ))
    neutral_times = np.linspace(0.0, 5.0, 11)
    neutral_capacity, neutral_rate, neutral_initial = 15.0, 0.55, 2.0
    neutral_values = neutral_capacity / (1.0 + ((neutral_capacity - neutral_initial) / neutral_initial)
                                           * np.exp(-neutral_rate * neutral_times))
    neutral_queries = [5.5, 6.0]
    cases.append(AutomatedBenchmarkCase(
        "modeling-ode-structure-selection", "modeling_ode",
        "根据时间和状态观测，在候选动力学中选择有数据支持的简洁结构并预测后续状态；不要依赖题目中的机制关键词。",
        {"attachments": [_table("measurements", [{"time": float(t), "state": float(y)}
                                                   for t, y in zip(neutral_times, neutral_values)])],
         "query_times": neutral_queries},
        {"family": "modeling_ode", "structure": "logistic_growth",
         "parameters": {"rate": neutral_rate, "capacity": neutral_capacity},
         "trajectory": [float(neutral_capacity / (1.0 + ((neutral_capacity - neutral_initial) / neutral_initial)
                              * math.exp(-neutral_rate * t))) for t in neutral_queries], "tolerance": 3e-2},
        "modeling-ode-data-driven-structure-selection", source_group="generated-modeling-ode",
    ))

    optimization_specs = (
        ("two-resource", [{"item": "A", "profit": 6.0, "labor": 2.0, "material": 1.0},
                          {"item": "B", "profit": 5.0, "labor": 1.0, "material": 2.0}],
         [{"resource": "labor", "capacity": 8.0}, {"resource": "material", "capacity": 8.0}],
         [8.0 / 3.0, 8.0 / 3.0], 88.0 / 3.0),
        ("three-item", [{"item": "A", "profit": 7.0, "machine": 3.0, "labor": 1.0},
                        {"item": "B", "profit": 5.0, "machine": 1.0, "labor": 2.0},
                        {"item": "C", "profit": 4.0, "machine": 1.0, "labor": 1.0}],
         [{"resource": "machine", "capacity": 9.0}, {"resource": "labor", "capacity": 8.0}],
         [0.5, 0.0, 7.5], 33.5),
    )
    for identifier, items, capacities, solution, optimum in optimization_specs:
        names = [str(row["item"]) for row in items]
        resources = [str(row["resource"]) for row in capacities]
        cases.append(AutomatedBenchmarkCase(
            f"modeling-optimization-{identifier}", "modeling_optimization",
            "决定各产品的非负生产量，在不超过每种资源容量的条件下最大化总利润。请从附件构造变量、目标与资源约束，并提交可行生产方案。",
            {"attachments": [_table("products", items), _table("capacities", capacities)]},
            {"family": "modeling_optimization", "structure": "resource_allocation_lp",
             "decision_variables": names, "constraint_ids": resources,
             "decision_units": {name: "item" for name in names}, "objective_unit": "currency",
             "objective": float(optimum), "objective_coefficients": [float(row["profit"]) for row in items],
             "bounds": [[0.0, min(float(capacity["capacity"]) / float(item[str(capacity["resource"])])
                                  for capacity in capacities)] for item in items],
             "A_ub": [[float(row[resource]) for row in items] for resource in resources],
             "b_ub": [float(row["capacity"]) for row in capacities], "direction": "maximize",
             "tolerance": 1e-6, "known_solution": solution},
            f"modeling-optimization-{identifier}", source_group="generated-modeling-optimization",
        ))

    minimum_cost_items = [{"item": "A", "cost": 3.0, "labor": 2.0},
                          {"item": "B", "cost": 5.0, "labor": 1.0}]
    cases.append(AutomatedBenchmarkCase(
        "modeling-optimization-minimum-cost", "modeling_optimization",
        "选择非负采购量，在总量至少达到需求且不超过资源容量时最小化成本；从附件构造变量、方向和全部约束。",
        {"attachments": [_table("options", minimum_cost_items),
                         _table("capacities", [{"resource": "labor", "capacity": 8.0}])],
         "minimum_total": 4.0},
        {"family": "modeling_optimization", "structure": "minimum_cost_allocation_lp",
         "decision_variables": ["A", "B"], "constraint_ids": ["labor", "minimum_total"],
         "decision_units": {"A": "item", "B": "item"}, "objective_unit": "currency",
         "objective": 12.0, "objective_coefficients": [3.0, 5.0],
         "bounds": [[0.0, 4.0], [0.0, 8.0]], "A_ub": [[2.0, 1.0], [-1.0, -1.0]],
         "b_ub": [8.0, -4.0], "direction": "minimize", "tolerance": 1e-6,
         "known_solution": [4.0, 0.0]},
        "modeling-optimization-minimum-cost", source_group="generated-modeling-optimization",
    ))

    sales = [{"entity": "E1", "amount": 4.0}, {"entity": "E1", "amount": 3.0},
             {"entity": "E2", "amount": 5.0}, {"entity": "E3", "amount": 2.0}]
    entities = [{"entity": "E1", "region": "north"}, {"entity": "E2", "region": "north"},
                {"entity": "E3", "region": "south"}]
    cases.append(AutomatedBenchmarkCase(
        "modeling-multitable-many-to-one", "modeling_multi_table",
        "汇总每个地区的销售额。识别事实表与实体表的关系，避免连接膨胀，并提交连接键、聚合方式和地区总额。",
        {"attachments": [_table("sales", sales), _table("entities", entities)]},
        {"family": "modeling_multi_table", "structure": "many_to_one_group_sum", "join_keys": ["entity"],
         "aggregation": "sum", "point_in_time": False, "rows": 4,
         "group_totals": {"north": 12.0, "south": 2.0}, "tolerance": 1e-8},
        "modeling-multitable-many-to-one", source_group="generated-modeling-multitable",
    ))
    events = [{"entity": "E1", "event_time": "2026-01-10", "amount": 3.0},
              {"entity": "E1", "event_time": "2026-02-10", "amount": 5.0},
              {"entity": "E2", "event_time": "2026-02-05", "amount": 7.0}]
    assignments = [{"entity": "E1", "effective_from": "2026-01-01", "segment": "basic"},
                   {"entity": "E1", "effective_from": "2026-02-01", "segment": "plus"},
                   {"entity": "E2", "effective_from": "2026-01-01", "segment": "basic"}]
    cases.append(AutomatedBenchmarkCase(
        "modeling-multitable-point-in-time", "modeling_multi_table",
        "按事件发生当时有效的客户分群汇总金额，不得使用事件之后才生效的记录。提交连接键、时间语义、聚合方式和分群总额。",
        {"attachments": [_table("events", events), _table("assignments", assignments)]},
        {"family": "modeling_multi_table", "structure": "point_in_time_group_sum", "join_keys": ["entity"],
         "aggregation": "sum", "point_in_time": True, "rows": 3,
         "group_totals": {"basic": 10.0, "plus": 5.0}, "tolerance": 1e-8},
        "modeling-multitable-point-in-time", source_group="generated-modeling-multitable",
    ))
    cases.append(AutomatedBenchmarkCase(
        "modeling-multitable-ambiguous-duplicate", "modeling_multi_table",
        "按客户类别汇总金额。若实体表同一客户存在多个类别且没有时间或去重口径，必须指出缺失条件，不能任意连接。",
        {"attachments": [_table("facts", [{"entity": "E1", "amount": 4.0},
                                            {"entity": "E2", "amount": 5.0}]),
                         _table("labels", [{"entity": "E1", "category": "a"},
                                           {"entity": "E1", "category": "b"},
                                           {"entity": "E2", "category": "a"}])]},
        {"family": "modeling_multi_table", "expected_status": "needs_input",
         "missing": ["dimension_deduplication_or_time_rule"], "tolerance": 0.0},
        "modeling-multitable-ambiguous-duplicate", source_group="generated-modeling-multitable",
    ))
    return tuple(cases)


def _numeric_error(actual: Any, expected: Sequence[float]) -> float | None:
    try:
        values = np.asarray(actual, dtype=float)
        truth = np.asarray(expected, dtype=float)
    except (TypeError, ValueError):
        return None
    if values.shape != truth.shape or not np.isfinite(values).all():
        return None
    return float(np.sqrt(np.mean((values - truth) ** 2)))


def _evaluate_compositional_model(model: Mapping[str, Any], inputs: Sequence[float]) -> np.ndarray | None:
    operators = {"sin": np.sin, "cos": np.cos, "tanh": np.tanh,
                 "exp": lambda values: np.exp(np.clip(values, -40.0, 40.0))}
    try:
        signature = list(model["operator_signature"])
        parameters = np.asarray(model["parameters"], dtype=float)
        values = np.asarray(inputs, dtype=float)
        if signature == ["divide", "affine", "affine"] and len(parameters) == 3:
            result = (parameters[0] + parameters[1] * values) / (1.0 + parameters[2] * values)
        elif 1 <= len(signature) <= 3 and len(parameters) == 4 and all(name in operators for name in signature):
            result = parameters[2] * values + parameters[3]
            for name in reversed(signature):
                result = operators[name](result)
            result = parameters[0] + parameters[1] * result
        else:
            return None
        return result if np.isfinite(result).all() else None
    except (KeyError, TypeError, ValueError, FloatingPointError, ZeroDivisionError):
        return None


def score_modeling_benchmark_case(output: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Require both a recovered model and an independently valid answer."""
    if not isinstance(output, Mapping) or not isinstance(reference, Mapping):
        return {"valid": False, "score": 1.0, "reason": "mapping_required"}
    if reference.get("expected_status") == "needs_input":
        missing = output.get("missing")
        valid = output.get("status") == "needs_input" and isinstance(missing, Sequence) \
            and not isinstance(missing, (str, bytes)) and set(missing) == set(reference.get("missing", []))
        return {"valid": bool(valid), "score": 0.0 if valid else 1.0,
                "reason": "required_ambiguity_detected" if valid else "unsafe_or_incomplete_ambiguity_response"}
    model = output.get("model")
    if not isinstance(model, Mapping):
        return {"valid": False, "score": 1.0, "reason": "model_description_missing"}
    family, structure = reference.get("family"), reference.get("structure")
    if model.get("family") != family:
        return {"valid": False, "score": 1.0, "reason": "model_family_incorrect"}
    if model.get("structure") != structure:
        return {"valid": False, "score": 1.0, "reason": "model_structure_incorrect"}
    tolerance = float(reference.get("tolerance", 1e-6))
    if family == "modeling_algebra":
        if list(model.get("input_variables", [])) != list(reference.get("input_variables", [])):
            return {"valid": False, "score": 1.0, "reason": "variable_binding_incorrect"}
        if structure == "compositional_symbolic":
            signature = model.get("operator_signature")
            probes = reference.get("equivalence_inputs")
            expected_probes = reference.get("equivalence_outputs")
            if probes is not None or expected_probes is not None:
                evaluated = _evaluate_compositional_model(model, probes)
                equivalence_error = _numeric_error(evaluated, expected_probes) if evaluated is not None else None
                if equivalence_error is None or equivalence_error > tolerance:
                    return {"valid": False, "score": 1.0 if equivalence_error is None else equivalence_error,
                            "reason": "symbolic_equivalence_probe_failed"}
            elif not isinstance(signature, Sequence) or isinstance(signature, (str, bytes)) \
                    or list(signature) != list(reference.get("operator_signature", [])):
                return {"valid": False, "score": 1.0, "reason": "operator_structure_incorrect"}
        else:
            coefficient_error = _numeric_error(model.get("coefficients"), reference.get("coefficients", []))
            if coefficient_error is None or coefficient_error > tolerance:
                return {"valid": False, "score": 1.0 if coefficient_error is None else coefficient_error,
                        "reason": "equation_parameters_incorrect"}
        error = _numeric_error(output.get("predictions"), reference.get("predictions", []))
    elif family == "modeling_ode":
        if structure == "delay_differential_polynomial":
            try:
                lag_error = abs(float(model["lag_time"]) - float(reference["lag_time"]))
                coefficients = np.asarray(model["coefficients"], dtype=float)
                parameter_error = lag_error if coefficients.size and np.isfinite(coefficients).all() else math.inf
            except (KeyError, TypeError, ValueError):
                parameter_error = math.inf
        elif structure == "exponential_decay":
            try:
                parameter_error = abs(float(model["parameter"]) - float(reference["parameter"]))
            except (KeyError, TypeError, ValueError):
                parameter_error = math.inf
        else:
            parameters = model.get("parameters", {})
            try:
                parameter_error = max(abs(float(parameters[name]) - float(value))
                                      for name, value in reference.get("parameters", {}).items())
            except (KeyError, TypeError, ValueError):
                parameter_error = math.inf
        if not math.isfinite(parameter_error) or parameter_error > tolerance:
            return {"valid": False, "score": 1.0 if not math.isfinite(parameter_error) else parameter_error,
                    "reason": "dynamics_parameters_incorrect"}
        error = _numeric_error(output.get("trajectory"), reference.get("trajectory", []))
    elif family == "modeling_optimization":
        if set(model.get("decision_variables", [])) != set(reference.get("decision_variables", [])):
            return {"valid": False, "score": 1.0, "reason": "decision_variables_incorrect"}
        if set(model.get("constraint_ids", [])) != set(reference.get("constraint_ids", [])):
            return {"valid": False, "score": 1.0, "reason": "optimization_constraints_incomplete"}
        if model.get("decision_units") != reference.get("decision_units") or model.get("objective_unit") != reference.get("objective_unit"):
            return {"valid": False, "score": 1.0, "reason": "optimization_units_incorrect"}
        optimization_reference = dict(reference)
        optimization_reference["family"] = "optimization"
        scored = score_automated_benchmark_case(output, optimization_reference)
        return {**scored, "model_valid": True}
    elif family == "modeling_multi_table":
        if list(model.get("join_keys", [])) != list(reference.get("join_keys", [])):
            return {"valid": False, "score": 1.0, "reason": "join_key_incorrect"}
        if model.get("aggregation") != reference.get("aggregation"):
            return {"valid": False, "score": 1.0, "reason": "aggregation_incorrect"}
        if model.get("point_in_time") is not reference.get("point_in_time"):
            return {"valid": False, "score": 1.0, "reason": "time_semantics_incorrect"}
        rows = output.get("rows")
        groups = output.get("group_totals")
        if type(rows) is not int or rows != reference.get("rows") or not isinstance(groups, Mapping):
            return {"valid": False, "score": 1.0, "reason": "join_result_missing"}
        if set(groups) != set(reference.get("group_totals", {})):
            return {"valid": False, "score": 1.0, "reason": "join_groups_incorrect"}
        try:
            errors = [abs(float(groups.get(key, math.nan)) - float(value))
                      for key, value in reference.get("group_totals", {}).items()]
        except (TypeError, ValueError):
            return {"valid": False, "score": 1.0, "reason": "join_result_non_numeric"}
        error = max(errors, default=0.0)
    else:
        return {"valid": False, "score": 1.0, "reason": "unknown_modeling_family"}
    if error is None:
        return {"valid": False, "score": 1.0, "reason": "answer_missing_or_shape_mismatch"}
    return {"valid": math.isfinite(error) and error <= tolerance, "score": float(error),
            "reason": "model_and_answer_verified" if error <= tolerance else "answer_error_exceeds_tolerance",
            "model_valid": True}


def current_problem_compiler_adapter(task: Mapping[str, Any], _context: Mapping[str, Any]) -> dict[str, Any]:
    """Run the current prose/raw-attachment compiler and return only executed modeling output."""
    from .dynamic_model_compiler import compile_and_execute_model
    public_input = task.get("input", {})
    compiled = compile_and_execute_model("problem_statement", {
        "problem": task.get("statement", ""), "has_observations": bool(public_input.get("attachments")),
        **{key: deepcopy(public_input[key]) for key in (
            "attachments", "query_inputs", "query_times", "minimum_total",
            "required_total", "integer_decisions",
        ) if key in public_input},
    })
    automatic = compiled.get("result", {}).get("automatic_modeling")
    if isinstance(automatic, Mapping):
        return dict(automatic)
    return {"compiler_status": compiled.get("status"), "compiler_result": compiled.get("result")}


def legacy_typed_executor_adapter(_task: Mapping[str, Any], _context: Mapping[str, Any]) -> dict[str, Any]:
    """Interface probe for the typed-only path, not an executable historical baseline."""
    return {"status": "incomplete", "reason": "typed_contract_not_supplied",
            "usage": {"model_api_calls": 0, "numerical_solver_calls": 0,
                      "manual_interventions": 1}}


def isolated_baseline_adapter(arm: str) -> Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]:
    """Return a baseline adapter supervised by the same worker runtime as the candidate."""
    keys = {"frozen_old": "modeling_legacy_proxy/v1",
            "simple_tool_baseline": "modeling_simple_tools/v1"}
    if arm not in keys:
        raise ValueError("isolated_baseline_arm_invalid")

    def execute(task: Mapping[str, Any], context: Mapping[str, Any]) -> Mapping[str, Any]:
        from .solver_runtime import SolverLimits, SolverProcessRunner, SolverRuntimeError
        budget = context.get("budget", {})
        try:
            return SolverProcessRunner().execute(
                keys[arm], dict(task), limits=SolverLimits(
                    wall_seconds=float(budget.get("per_case_wall_seconds", 30.0)),
                    memory_mb=int(budget.get("memory_mb", 1024)),
                ),
            )
        except SolverRuntimeError as exc:
            return {"status": "not_assessed", "reason": exc.code,
                    "usage": {"model_api_calls": 0, "numerical_solver_calls": 0,
                              "manual_interventions": 0},
                    "execution_supervision": exc.metadata,
                    "policy": "resource_failure_is_not_a_modeling_failure;no_in_process_fallback"}
    return execute


def simple_tool_modeling_adapter(task: Mapping[str, Any], _context: Mapping[str, Any]) -> dict[str, Any]:
    """A transparent numerical-tool baseline; no benchmark reference is read."""
    family = task.get("family")
    public = task.get("input", {})
    attachments = {item["name"]: item["rows"] for item in public.get("attachments", [])}
    def incomplete(reason: str, solver_calls: int = 0) -> dict[str, Any]:
        return {"status": "incomplete", "reason": reason,
                "usage": {"model_api_calls": 0, "numerical_solver_calls": solver_calls,
                          "manual_interventions": 0}}
    if family == "modeling_algebra":
        rows = attachments.get("observations")
        if not rows or "input" not in rows[0]:
            return incomplete("single_input_observation_schema_required")
        x = np.asarray([row["input"] for row in rows], dtype=float)
        y = np.asarray([row["response"] for row in rows], dtype=float)
        fits = []
        for degree, structure in ((1, "affine"), (2, "quadratic")):
            coefficients = np.polyfit(x, y, degree)
            residual = float(np.mean((np.polyval(coefficients, x) - y) ** 2))
            fits.append((residual + degree * np.finfo(float).eps, degree, structure, coefficients))
        _, _degree, structure, coefficients = min(fits, key=lambda item: item[0])
        return {"model": {"family": family, "structure": structure, "input_variables": ["input"],
                          "coefficients": coefficients.tolist()},
                "predictions": np.polyval(coefficients, public["query_inputs"]).tolist(),
                "usage": {"model_api_calls": 0, "numerical_solver_calls": 2, "manual_interventions": 0}}
    if family == "modeling_ode":
        from scipy.optimize import curve_fit
        rows = attachments.get("observations")
        if not rows:
            return incomplete("named_observations_table_required")
        times = np.asarray([row["time"] for row in rows], dtype=float)
        states = np.asarray([row["state"] for row in rows], dtype=float)
        queries = np.asarray(public["query_times"], dtype=float)
        if "承载" in str(task.get("statement", "")):
            def logistic(t, capacity, rate):
                return capacity / (1.0 + ((capacity - states[0]) / states[0]) * np.exp(-rate * t))
            params, _ = curve_fit(logistic, times, states, p0=[max(states) * 1.2, 0.5],
                                  bounds=([max(states), 1e-6], [1e6, 20.0]), maxfev=5000)
            return {"model": {"family": family, "structure": "logistic_growth",
                              "parameters": {"capacity": float(params[0]), "rate": float(params[1])}},
                    "trajectory": logistic(queries, *params).tolist(),
                    "usage": {"model_api_calls": 0, "numerical_solver_calls": 1, "manual_interventions": 0}}
        shifted = times - times[0]
        rate = -float(np.polyfit(shifted, np.log(states), 1)[0])
        return {"model": {"family": family, "structure": "exponential_decay", "parameter": rate},
                "trajectory": (states[0] * np.exp(-rate * (queries - times[0]))).tolist(),
                "usage": {"model_api_calls": 0, "numerical_solver_calls": 1, "manual_interventions": 0}}
    if family == "modeling_optimization":
        from scipy.optimize import linprog
        items, capacities = attachments.get("products"), attachments.get("capacities")
        if not items or not capacities or "profit" not in items[0]:
            return incomplete("profit_products_and_capacities_schema_required")
        names = [str(row["item"]) for row in items]
        resources = [str(row["resource"]) for row in capacities]
        objective = np.asarray([row["profit"] for row in items], dtype=float)
        matrix = np.asarray([[row[resource] for row in items] for resource in resources], dtype=float)
        rhs = np.asarray([row["capacity"] for row in capacities], dtype=float)
        result = linprog(-objective, A_ub=matrix, b_ub=rhs, bounds=[(0.0, None)] * len(names), method="highs")
        if not result.success:
            raise RuntimeError("simple_lp_solver_failed")
        return {"model": {"family": family, "structure": "resource_allocation_lp",
                          "decision_variables": names, "constraint_ids": resources,
                          "decision_units": {name: "item" for name in names}, "objective_unit": "currency"},
                "solution": {name: float(result.x[index]) for index, name in enumerate(names)},
                "objective": float(objective @ result.x),
                "usage": {"model_api_calls": 0, "numerical_solver_calls": 1, "manual_interventions": 0}}
    if family == "modeling_multi_table":
        import pandas as pd
        names = list(attachments)
        left, right = pd.DataFrame(attachments[names[0]]), pd.DataFrame(attachments[names[1]])
        common = sorted(set(left.columns) & set(right.columns))
        if common != ["entity"]:
            raise ValueError("simple_join_key_not_unique")
        point_in_time = "event_time" in left.columns and "effective_from" in right.columns
        if point_in_time:
            left["event_time"] = pd.to_datetime(left["event_time"])
            right["effective_from"] = pd.to_datetime(right["effective_from"])
            joined = pd.merge_asof(left.sort_values("event_time"), right.sort_values("effective_from"),
                                   left_on="event_time", right_on="effective_from", by="entity",
                                   direction="backward", allow_exact_matches=True)
            group = "segment"
        else:
            try:
                joined = left.merge(right, on="entity", how="left", validate="many_to_one")
            except Exception:
                return incomplete("many_to_one_join_required", solver_calls=1)
            group = "region"
        totals = joined.groupby(group, dropna=False)["amount"].sum()
        structure = "point_in_time_group_sum" if point_in_time else "many_to_one_group_sum"
        return {"model": {"family": family, "structure": structure, "join_keys": ["entity"],
                          "aggregation": "sum", "point_in_time": point_in_time},
                "rows": int(len(joined)), "group_totals": {str(key): float(value) for key, value in totals.items()},
                "usage": {"model_api_calls": 0, "numerical_solver_calls": 1, "manual_interventions": 0}}
    raise ValueError("simple_tool_family_unsupported")


def run_three_arm_modeling_comparison(
    systems: Mapping[str, Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]]],
    *, cases: Sequence[AutomatedBenchmarkCase] | None = None,
    fixed_budget: Mapping[str, Any] | None = None,
    system_versions: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Run old/new/tool adapters on one paired grid and retain every failure.

    The runner passes an identical budget contract and rejects returned rows
    that exceed the per-case wall limit.  Actual model/API/worker limits must
    still be enforced by each version adapter or an outer process supervisor.
    """
    if set(systems) != set(_ARMS) or any(not callable(systems[name]) for name in _ARMS):
        raise ValueError("exact_three_modeling_arms_required")
    suite = tuple(cases or build_modeling_benchmark_suite())
    budget = dict(fixed_budget or {"per_case_wall_seconds": 30.0, "memory_mb": 1024,
                                  "max_model_api_calls": 0,
                                  "max_numerical_solver_calls": 2,
                                  "max_manual_interventions": 1, "seed": 20260915})
    per_case = budget.get("per_case_wall_seconds")
    if type(per_case) not in (int, float) or not 0.01 <= float(per_case) <= 3600:
        raise ValueError("per_case_wall_budget_invalid")
    versions = dict(system_versions or {})
    version_bound = set(versions) == set(_ARMS) and all(
        isinstance(versions[name], Mapping)
        and isinstance(versions[name].get("version_id"), str) and versions[name]["version_id"]
        and isinstance(versions[name].get("source_digest"), str) and len(versions[name]["source_digest"]) == 64
        for name in _ARMS
    )
    rows: list[dict[str, Any]] = []
    for case in suite:
        case.validate()
        public = {"id": case.case_id, "family": case.family, "statement": case.statement,
                  "input": deepcopy(case.public_input), "structure_group": case.structure_group,
                  "source_group": case.source_group}
        for arm in _ARMS:
            tick = time.monotonic()
            try:
                output = systems[arm](deepcopy(public), {"budget": deepcopy(budget), "arm": arm})
                if not isinstance(output, Mapping):
                    raise TypeError("system_output_must_be_mapping")
                elapsed = time.monotonic() - tick
                if elapsed > float(per_case):
                    rows.append({"task_id": case.case_id, "structure_group": case.structure_group,
                                 "family": case.family, "arm_id": arm, "status": "timeout", "valid": False,
                                 "score": 1.0, "reason": "per_case_wall_budget_exceeded",
                                 "duration_seconds": elapsed})
                    continue
                usage = output.get("usage")
                usage_attested = isinstance(usage, Mapping) and all(
                    type(usage.get(key)) in (int, float) and math.isfinite(float(usage[key]))
                    and float(usage[key]) >= 0
                    for key in ("model_api_calls", "numerical_solver_calls", "manual_interventions")
                )
                normalized_usage = ({key: float(usage[key]) for key in (
                    "model_api_calls", "numerical_solver_calls", "manual_interventions"
                )} if usage_attested else None)
                supervision = output.get("execution_supervision")
                resource_enforced = bool(
                    isinstance(supervision, Mapping)
                    and supervision.get("process_isolated") is True
                    and isinstance(supervision.get("memory_backend"), str)
                    and isinstance(supervision.get("limits"), Mapping)
                    and supervision["limits"].get("memory_mb") == budget.get("memory_mb", 1024)
                )
                limits = {"model_api_calls": "max_model_api_calls",
                          "numerical_solver_calls": "max_numerical_solver_calls",
                          "manual_interventions": "max_manual_interventions"}
                if usage_attested and any(
                    limits[key] in budget and normalized_usage[key] > float(budget[limits[key]])
                    for key in limits
                ):
                    rows.append({"task_id": case.case_id, "structure_group": case.structure_group,
                                 "family": case.family, "arm_id": arm, "status": "rejected", "valid": False,
                                 "score": 1.0, "reason": "reported_resource_budget_exceeded",
                                 "duration_seconds": elapsed, "usage": normalized_usage,
                                 "usage_attested": True})
                    continue
                if output.get("status") == "not_assessed":
                    rows.append({"task_id": case.case_id, "structure_group": case.structure_group,
                                 "family": case.family, "arm_id": arm,
                                 "status": str(output.get("reason") or "not_assessed"),
                                 "valid": False, "score": 1.0,
                                 "reason": str(output.get("reason") or "not_assessed"),
                                 "duration_seconds": elapsed, "usage": normalized_usage,
                                 "usage_attested": usage_attested,
                                 "resource_enforced": resource_enforced,
                                 "execution_supervision": dict(supervision)
                                 if isinstance(supervision, Mapping) else None})
                    continue
                scored = score_modeling_benchmark_case(output, case.hidden_reference)
                rows.append({"task_id": case.case_id, "structure_group": case.structure_group,
                             "family": case.family, "arm_id": arm, "status": "completed",
                             "valid": bool(scored["valid"]), "score": float(scored["score"]),
                             "reason": scored.get("reason"), "duration_seconds": elapsed,
                             "usage": normalized_usage, "usage_attested": usage_attested,
                             "resource_enforced": resource_enforced,
                             "execution_supervision": dict(supervision) if isinstance(supervision, Mapping) else None})
            except Exception as exc:
                rows.append({"task_id": case.case_id, "structure_group": case.structure_group,
                             "family": case.family, "arm_id": arm, "status": "error", "valid": False,
                             "score": 1.0, "reason": type(exc).__name__, "duration_seconds": time.monotonic() - tick})
    summaries: dict[str, Any] = {}
    for arm in _ARMS:
        arm_rows = [row for row in rows if row["arm_id"] == arm]
        groups: dict[str, list[bool]] = defaultdict(list)
        for row in arm_rows:
            groups[row["structure_group"]].append(bool(row["valid"]))
        summaries[arm] = {"case_count": len(arm_rows), "valid_count": sum(row["valid"] for row in arm_rows),
                          "valid_rate": sum(row["valid"] for row in arm_rows) / len(arm_rows),
                          "structure_group_count": len(groups),
                          "structure_group_success_rate": sum(all(values) for values in groups.values()) / len(groups),
                          "duration_seconds_sum": sum(float(row["duration_seconds"]) for row in arm_rows),
                          "usage_attested_count": sum(row.get("usage_attested") is True for row in arm_rows),
                          "resource_enforced_count": sum(row.get("resource_enforced") is True for row in arm_rows),
                          "resource_totals": {key: sum(float(row["usage"][key]) for row in arm_rows
                                                       if isinstance(row.get("usage"), Mapping))
                                              for key in ("model_api_calls", "numerical_solver_calls",
                                                          "manual_interventions")}}
    from .benchmark_statistics import paired_benchmark_effect
    paired = {}
    for baseline in ("simple_tool_baseline",):
        baseline_rows = {row["task_id"]: row for row in rows if row["arm_id"] == baseline}
        candidate_rows = {row["task_id"]: row for row in rows if row["arm_id"] == "candidate_new"}
        samples = [{"baseline": float(baseline_rows[case.case_id]["valid"]),
                    "treatment": float(candidate_rows[case.case_id]["valid"]),
                    "structure_group": case.structure_group} for case in suite]
        paired[baseline] = paired_benchmark_effect(
            samples, baseline="baseline", treatment="treatment", min_samples=5,
            cluster="structure_group",
        )
    probe_rows = [row for row in rows if row["arm_id"] == "frozen_old"]
    return {"schema_version": "mathmodel.modeling-comparison/v1", "status": "assessed",
            "arms": list(_ARMS), "fixed_budget": budget, "system_versions": versions,
            "version_binding_status": "bound" if version_bound else "proxy_only",
            "arm_roles": {
                "candidate_new": "method_under_test",
                "simple_tool_baseline": "executable_method_baseline",
                "frozen_old": "typed_contract_interface_probe_not_historical_baseline",
            },
            "historical_baseline_status": "not_available",
            "rows": rows, "arm_summary": summaries, "paired_valid_rate_effects": paired,
            "interface_probe": {"arm_id": "frozen_old", "case_count": len(probe_rows),
                                "valid_count": sum(row["valid"] for row in probe_rows),
                                "comparison_eligible": False},
            "method_comparison_eligibility": {
                "candidate_new_vs_simple_tool_baseline": version_bound,
                "candidate_new_vs_historical_system": False,
            },
            "resource_comparison_eligible": version_bound and all(
                summaries[arm]["usage_attested_count"] == summaries[arm]["case_count"]
                and summaries[arm]["resource_enforced_count"] == summaries[arm]["case_count"]
                for arm in ("candidate_new", "simple_tool_baseline")),
            "policy": "raw_problem_inputs;model_and_answer_required;same_budget_contract;reported_usage_checked;os_resource_evidence_required_for_resource_claims;failures_in_denominator;typed_contract_probe_excluded_from_method_effects;structure_group_cluster_bootstrap;small_fixture_not_generalization"}


def run_automatic_modeling_ablation(*, cases: Sequence[AutomatedBenchmarkCase] | None = None) -> dict[str, Any]:
    """Disable one induction mechanism at a time on the same task grid."""
    suite = tuple(cases or build_modeling_benchmark_suite())
    variants = ("full", "without_pairwise_interactions", "without_ode_model_selection",
                "without_optimization_direction_inference", "without_ambiguity_gate")
    rows = []
    for case in suite:
        task = {"id": case.case_id, "family": case.family, "statement": case.statement,
                "input": deepcopy(case.public_input), "structure_group": case.structure_group}
        full = current_problem_compiler_adapter(task, {})
        for variant in variants:
            output = deepcopy(full)
            model = output.get("model", {}) if isinstance(output, Mapping) else {}
            if variant == "without_pairwise_interactions" and model.get("structure") == "bilinear":
                output = {"status": "incomplete", "reason": "pairwise_interaction_disabled"}
            elif variant == "without_ode_model_selection" and model.get("family") == "modeling_ode":
                output = {"status": "incomplete", "reason": "ode_model_selection_disabled"}
            elif (variant == "without_optimization_direction_inference"
                  and model.get("direction") == "minimize"):
                output = {"status": "incomplete", "reason": "optimization_direction_inference_disabled"}
            elif (variant == "without_ambiguity_gate"
                  and output.get("status") == "needs_input"):
                output = {"status": "incomplete", "reason": "ambiguity_gate_disabled"}
            scored = score_modeling_benchmark_case(output, case.hidden_reference)
            rows.append({"task_id": case.case_id, "structure_group": case.structure_group,
                         "variant": variant, "valid": bool(scored["valid"]),
                         "score": float(scored["score"]), "reason": scored.get("reason")})
    summary = {}
    for variant in variants:
        selected = [row for row in rows if row["variant"] == variant]
        summary[variant] = {"case_count": len(selected), "valid_count": sum(row["valid"] for row in selected),
                            "valid_rate": sum(row["valid"] for row in selected) / len(selected)}
    full_by_task = {row["task_id"]: row for row in rows if row["variant"] == "full"}
    lost = {variant: [row["task_id"] for row in rows if row["variant"] == variant
                      and full_by_task[row["task_id"]]["valid"] and not row["valid"]]
            for variant in variants if variant != "full"}
    return {"schema_version": "mathmodel.modeling-ablation/v1", "status": "descriptive_development_only",
            "rows": rows, "summary": summary, "lost_successes": lost,
            "policy": "single_mechanism_output_gate_ablation;same_tasks;development_fixture;not_independent_confirmation"}


def run_modeling_scorer_attack_audit(
    *, cases: Sequence[AutomatedBenchmarkCase] | None = None,
) -> dict[str, Any]:
    """Attack the scorer with plausible but invalid submissions.

    A valid output from the current bounded modeler is retained as a control for
    every case.  Attack rows are then created without consulting scorer output,
    so the audit records both false rejection of controls and false acceptance
    of malformed, semantically incomplete, or numerically invalid answers.
    """
    suite = tuple(cases or build_modeling_benchmark_suite())
    controls: list[dict[str, Any]] = []
    attacks: list[dict[str, Any]] = []

    def add_attack(case: AutomatedBenchmarkCase, attack_id: str, output: Mapping[str, Any]) -> None:
        scored = score_modeling_benchmark_case(output, case.hidden_reference)
        raw_score = scored.get("score")
        score = float(raw_score) if type(raw_score) in (int, float) and math.isfinite(float(raw_score)) else None
        attacks.append({"task_id": case.case_id, "family": case.family,
                        "structure_group": case.structure_group, "attack_id": attack_id,
                        "accepted": bool(scored["valid"]), "score": score,
                        "reason": scored.get("reason")})

    for case in suite:
        task = {"id": case.case_id, "family": case.family, "statement": case.statement,
                "input": deepcopy(case.public_input), "structure_group": case.structure_group}
        valid_output = current_problem_compiler_adapter(task, {})
        control_score = score_modeling_benchmark_case(valid_output, case.hidden_reference)
        controls.append({"task_id": case.case_id, "family": case.family,
                         "valid": bool(control_score["valid"]),
                         "reason": control_score.get("reason")})
        if not control_score["valid"]:
            continue

        reference = case.hidden_reference
        if reference.get("expected_status") == "needs_input":
            add_attack(case, "invent_answer_for_ambiguous_input", {
                "status": "completed",
                "model": {"family": case.family, "structure": "many_to_one_group_sum",
                          "join_keys": ["entity"], "aggregation": "sum", "point_in_time": False},
                "rows": 3, "group_totals": {"a": 9.0, "b": 4.0},
            })
            add_attack(case, "generic_refusal_without_required_missing_field", {
                "status": "needs_input", "missing": ["unknown"],
            })
            continue

        answer_only = {key: deepcopy(valid_output[key]) for key in (
            "predictions", "trajectory", "solution", "objective", "rows", "group_totals"
        ) if key in valid_output}
        add_attack(case, "correct_numeric_answer_without_model", answer_only)

        if case.family == "modeling_algebra":
            wrong = deepcopy(valid_output)
            coefficients = list(wrong["model"]["coefficients"])
            coefficients[0] = float(coefficients[0]) + 1.0
            wrong["model"]["coefficients"] = coefficients
            add_attack(case, "correct_predictions_wrong_equation", wrong)
            nonfinite = deepcopy(valid_output)
            nonfinite["predictions"][0] = math.nan
            add_attack(case, "nonfinite_prediction", nonfinite)
        elif case.family == "modeling_ode":
            wrong = deepcopy(valid_output)
            if wrong["model"]["structure"] == "exponential_decay":
                wrong["model"]["parameter"] = float(wrong["model"]["parameter"]) + 1.0
            else:
                wrong["model"]["parameters"]["rate"] = float(
                    wrong["model"]["parameters"]["rate"]
                ) + 1.0
            add_attack(case, "correct_trajectory_wrong_dynamics", wrong)
            nonfinite = deepcopy(valid_output)
            nonfinite["trajectory"][0] = math.inf
            add_attack(case, "nonfinite_trajectory", nonfinite)
        elif case.family == "modeling_optimization":
            no_decisions = deepcopy(valid_output)
            no_decisions.pop("solution", None)
            add_attack(case, "objective_only_without_decisions", no_decisions)
            infeasible = deepcopy(valid_output)
            infeasible["solution"] = {name: 1e9 for name in reference["decision_variables"]}
            infeasible["objective"] = reference["objective"]
            infeasible["maximum_constraint_violation"] = 0.0
            add_attack(case, "forged_feasibility_and_objective", infeasible)
            missing_constraint = deepcopy(valid_output)
            missing_constraint["model"]["constraint_ids"] = []
            add_attack(case, "omitted_constraints", missing_constraint)
            wrong_units = deepcopy(valid_output)
            wrong_units["model"]["decision_units"] = {
                name: "forged_unit" for name in reference["decision_variables"]
            }
            add_attack(case, "wrong_decision_units", wrong_units)
        elif case.family == "modeling_multi_table":
            wrong_key = deepcopy(valid_output)
            wrong_key["model"]["join_keys"] = ["forged_key"]
            add_attack(case, "wrong_join_key", wrong_key)
            wrong_rows = deepcopy(valid_output)
            wrong_rows["rows"] = int(wrong_rows["rows"]) + 1
            add_attack(case, "hidden_join_expansion", wrong_rows)
            extra_group = deepcopy(valid_output)
            extra_group["group_totals"]["forged_group"] = 0.0
            add_attack(case, "extra_aggregation_group", extra_group)
            if reference.get("point_in_time"):
                future = deepcopy(valid_output)
                future["model"]["point_in_time"] = False
                add_attack(case, "future_information_semantics", future)

    false_accepts = [row for row in attacks if row["accepted"]]
    control_failures = [row for row in controls if not row["valid"]]
    status = "passed" if not false_accepts and not control_failures else "failed"
    return {
        "schema_version": "mathmodel.modeling-scorer-attack-audit/v1",
        "status": status,
        "control_count": len(controls),
        "control_failure_count": len(control_failures),
        "attack_count": len(attacks),
        "false_accept_count": len(false_accepts),
        "attack_detection_rate": 1.0 - len(false_accepts) / len(attacks) if attacks else 0.0,
        "controls": controls,
        "attacks": attacks,
        "policy": "valid_controls_required;all_injected_invalid_outputs_must_be_rejected;attack_catalog_is_not_exhaustive",
    }


def run_compositional_depth_ablation(
    *, cases: Sequence[AutomatedBenchmarkCase] | None = None,
) -> dict[str, Any]:
    """Compare depth-two enumeration with the same implementation restricted to flat operators."""
    from .automatic_modeling import induce_and_solve_modeling_task
    from .benchmark_statistics import paired_benchmark_effect
    if cases is None:
        from .modeling_equivalence_holdout import build_equivalence_holdout
        cases = build_equivalence_holdout()
    rows, samples = [], []
    for case in cases:
        by_variant = {}
        for variant, depth in (("depth_two", 2), ("flat_only", 1)):
            payload = deepcopy(case.public_input)
            payload["expression_max_depth"] = depth
            output = induce_and_solve_modeling_task(payload)
            scored = score_modeling_benchmark_case(output, case.hidden_reference)
            model = output.get("model", {})
            row = {"task_id": case.case_id, "structure_group": case.structure_group,
                   "variant": variant, "valid": bool(scored["valid"]), "reason": scored.get("reason"),
                   "search_evaluations": model.get("search_evaluations"),
                   "selected_signature": model.get("operator_signature")}
            rows.append(row)
            by_variant[variant] = row
        samples.append({"baseline": float(by_variant["flat_only"]["valid"]),
                        "treatment": float(by_variant["depth_two"]["valid"]),
                        "structure_group": case.structure_group})
    summary = {variant: {"case_count": sum(row["variant"] == variant for row in rows),
                         "valid_count": sum(row["valid"] for row in rows if row["variant"] == variant)}
               for variant in ("depth_two", "flat_only")}
    return {"schema_version": "mathmodel.compositional-depth-ablation/v1",
            "status": "descriptive_development_only", "rows": rows, "summary": summary,
            "paired_effect": paired_benchmark_effect(samples, cluster="structure_group", min_samples=5),
            "policy": "post_confirmation_development_ablation;deterministic_enumeration;not_search_algorithm_novelty"}


def run_compositional_search_strategy_comparison(
    *, cases: Sequence[AutomatedBenchmarkCase] | None = None, seed: int = 20260924,
) -> dict[str, Any]:
    """Compare exhaustive depth-two enumeration with equal-topology-budget random search."""
    from .automatic_modeling import induce_and_solve_modeling_task
    from .benchmark_statistics import paired_benchmark_effect
    if cases is None:
        from .modeling_equivalence_holdout import build_equivalence_holdout
        cases = build_equivalence_holdout()
    rows, samples = [], []
    for index, case in enumerate(cases):
        outcomes = {}
        for variant, additions in (
            ("exhaustive_depth_two", {"expression_max_depth": 2}),
            ("random_equal_budget", {"expression_max_depth": 3,
                                     "expression_search_strategy": "random_equal_budget",
                                     "expression_search_seed": seed + index}),
        ):
            payload = {**deepcopy(case.public_input), **additions}
            output = induce_and_solve_modeling_task(payload)
            scored = score_modeling_benchmark_case(output, case.hidden_reference)
            model = output.get("model", {})
            row = {"task_id": case.case_id, "structure_group": case.structure_group,
                   "variant": variant, "valid": bool(scored["valid"]), "reason": scored.get("reason"),
                   "topology_budget": model.get("topology_budget"),
                   "search_evaluations": model.get("search_evaluations"),
                   "selected_signature": model.get("operator_signature")}
            rows.append(row); outcomes[variant] = row
        samples.append({"baseline": float(outcomes["random_equal_budget"]["valid"]),
                        "treatment": float(outcomes["exhaustive_depth_two"]["valid"]),
                        "structure_group": case.structure_group})
    summary = {variant: {"case_count": sum(row["variant"] == variant for row in rows),
                         "valid_count": sum(row["valid"] for row in rows if row["variant"] == variant)}
               for variant in ("exhaustive_depth_two", "random_equal_budget")}
    return {"schema_version": "mathmodel.compositional-search-comparison/v1", "status": "development_only",
            "seed": seed, "rows": rows, "summary": summary,
            "paired_effect": paired_benchmark_effect(samples, cluster="structure_group", min_samples=5),
            "policy": "equal_topology_proposal_budget_21;parameter_optimizer_work_may_differ;post_confirmation_development_only"}


def run_compositional_beam_comparison(
    *, cases: Sequence[AutomatedBenchmarkCase], seed: int = 20260928,
    wall_seconds: float = 30.0, memory_mb: int = 1024,
) -> dict[str, Any]:
    """Compare adaptive beam and random search under isolated equal proposal budgets."""
    from .automatic_modeling import induce_and_solve_modeling_task_isolated
    from .benchmark_statistics import paired_benchmark_effect

    rows, samples = [], []
    variants = ("beam_equal_budget", "random_equal_budget")
    for index, case in enumerate(cases):
        case.validate()
        outcomes = {}
        for variant in variants:
            payload = deepcopy(case.public_input)
            payload.update({"expression_max_depth": 3,
                            "expression_search_strategy": variant,
                            "expression_search_seed": seed + index})
            tick = time.monotonic()
            output = induce_and_solve_modeling_task_isolated(
                payload, wall_seconds=wall_seconds, memory_mb=memory_mb,
            )
            elapsed = time.monotonic() - tick
            scored = score_modeling_benchmark_case(output, case.hidden_reference)
            model = output.get("model", {})
            supervision = output.get("execution_supervision")
            resource_enforced = bool(
                isinstance(supervision, Mapping)
                and supervision.get("process_isolated") is True
                and isinstance(supervision.get("memory_backend"), str)
                and isinstance(supervision.get("limits"), Mapping)
                and supervision["limits"].get("memory_mb") == memory_mb
            )
            row = {"task_id": case.case_id, "structure_group": case.structure_group,
                   "variant": variant, "valid": bool(scored["valid"]),
                   "reason": scored.get("reason"), "duration_seconds": elapsed,
                   "topology_budget": model.get("topology_budget"),
                   "search_evaluations": model.get("search_evaluations"),
                   "selected_signature": model.get("operator_signature"),
                   "resource_enforced": resource_enforced,
                   "execution_supervision": dict(supervision)
                   if isinstance(supervision, Mapping) else None}
            rows.append(row)
            outcomes[variant] = row
        samples.append({"baseline": float(outcomes["random_equal_budget"]["valid"]),
                        "treatment": float(outcomes["beam_equal_budget"]["valid"]),
                        "structure_group": case.structure_group})
    summary = {variant: {
        "case_count": sum(row["variant"] == variant for row in rows),
        "valid_count": sum(row["valid"] for row in rows if row["variant"] == variant),
        "duration_seconds_sum": sum(row["duration_seconds"] for row in rows
                                    if row["variant"] == variant),
        "resource_enforced_count": sum(row["resource_enforced"] for row in rows
                                       if row["variant"] == variant),
    } for variant in variants}
    return {"schema_version": "mathmodel.compositional-beam-comparison/v1",
            "status": "assessed", "seed": seed,
            "budget": {"topology_evaluations_per_case": 21,
                       "wall_seconds_per_case": wall_seconds, "memory_mb": memory_mb},
            "rows": rows, "summary": summary,
            "paired_effect": paired_benchmark_effect(samples, cluster="structure_group", min_samples=5),
            "policy": "equal_topology_proposal_budget_21;isolated_workers;failures_in_denominator;parameter_optimizer_work_may_differ"}


def run_compositional_depth_three_comparison(
    *, cases: Sequence[AutomatedBenchmarkCase], wall_seconds: float = 30.0,
    memory_mb: int = 1024,
) -> dict[str, Any]:
    """Isolate the gain from admitting depth-three topology enumeration."""
    from .automatic_modeling import induce_and_solve_modeling_task_isolated
    from .benchmark_statistics import paired_benchmark_effect

    rows, samples = [], []
    variants = (("exhaustive_depth_three", 3), ("exhaustive_depth_two", 2))
    for case in cases:
        case.validate()
        outcomes = {}
        for variant, depth in variants:
            payload = deepcopy(case.public_input)
            payload.update({"expression_max_depth": depth,
                            "expression_search_strategy": "exhaustive"})
            tick = time.monotonic()
            output = induce_and_solve_modeling_task_isolated(
                payload, wall_seconds=wall_seconds, memory_mb=memory_mb,
            )
            elapsed = time.monotonic() - tick
            scored = score_modeling_benchmark_case(output, case.hidden_reference)
            model = output.get("model", {})
            supervision = output.get("execution_supervision")
            resource_enforced = bool(
                isinstance(supervision, Mapping)
                and supervision.get("process_isolated") is True
                and isinstance(supervision.get("memory_backend"), str)
                and isinstance(supervision.get("limits"), Mapping)
                and supervision["limits"].get("memory_mb") == memory_mb
            )
            row = {"task_id": case.case_id, "structure_group": case.structure_group,
                   "variant": variant, "valid": bool(scored["valid"]),
                   "reason": scored.get("reason"), "duration_seconds": elapsed,
                   "topology_budget": model.get("topology_budget"),
                   "search_evaluations": model.get("search_evaluations"),
                   "selected_signature": model.get("operator_signature"),
                   "resource_enforced": resource_enforced,
                   "execution_supervision": dict(supervision)
                   if isinstance(supervision, Mapping) else None}
            rows.append(row)
            outcomes[variant] = row
        samples.append({"baseline": float(outcomes["exhaustive_depth_two"]["valid"]),
                        "treatment": float(outcomes["exhaustive_depth_three"]["valid"]),
                        "structure_group": case.structure_group})
    summary = {variant: {
        "case_count": sum(row["variant"] == variant for row in rows),
        "valid_count": sum(row["valid"] for row in rows if row["variant"] == variant),
        "duration_seconds_sum": sum(row["duration_seconds"] for row in rows
                                    if row["variant"] == variant),
        "resource_enforced_count": sum(row["resource_enforced"] for row in rows
                                       if row["variant"] == variant),
    } for variant, _depth in variants}
    return {"schema_version": "mathmodel.compositional-depth-three-comparison/v1",
            "status": "assessed",
            "budget": {"wall_seconds_per_case": wall_seconds, "memory_mb": memory_mb,
                       "depth_three_topology_budget": 85, "depth_two_topology_budget": 21},
            "rows": rows, "summary": summary,
            "paired_effect": paired_benchmark_effect(samples, cluster="structure_group", min_samples=5),
            "policy": "isolated_workers;failures_in_denominator;depth_budgets_intentionally_differ;attributes_grammar_reach_not_search_efficiency"}


__all__ = ["build_modeling_benchmark_suite", "score_modeling_benchmark_case",
           "legacy_typed_executor_adapter", "current_problem_compiler_adapter",
           "simple_tool_modeling_adapter", "isolated_baseline_adapter",
           "run_three_arm_modeling_comparison",
           "run_automatic_modeling_ablation", "run_compositional_depth_ablation",
           "run_compositional_search_strategy_comparison", "run_compositional_beam_comparison",
           "run_compositional_depth_three_comparison",
           "run_modeling_scorer_attack_audit"]
