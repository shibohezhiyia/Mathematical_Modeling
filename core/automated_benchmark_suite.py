"""Deterministic, self-contained task families for automated evaluation.

The suite deliberately exposes a typed *task input* rather than its hidden
reference.  It is suitable for testing the execution/validation path and for
building a structure-held-out development set; it is not a claim that these
small tasks represent real competition problems.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .automated_benchmark import AutomatedBenchmarkCase, run_automated_benchmark


def typed_execution_baseline(task: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the bundled public contracts through existing typed backends.

    This is intentionally a *baseline adapter*, not the research assistant or
    a hidden-reference oracle.  It only reads ``task['input']`` and dispatches
    to the same bounded runtimes that are exposed by the project.  Keeping the
    adapter here makes the benchmark runnable out of the box while retaining
    the generator/system/scorer separation.
    """
    if not isinstance(task, Mapping) or not isinstance(task.get("input"), Mapping):
        raise ValueError("typed_benchmark_task_input_required")
    payload = task["input"]
    family = str(task.get("family", ""))
    if family == "algebra":
        from .primitive_graph_runtime import execute_primitive_graph

        result = execute_primitive_graph(payload["nodes"], payload.get("bindings", {}),
                                         output_ids=payload.get("output_ids", []))
        raw = result.get("outputs", {}).get("out")
        values = np.asarray(raw, dtype=float)
        if values.size != 1 or not np.isfinite(values).all():
            raise ValueError("typed_algebra_output_not_scalar")
        return {"value": float(values.reshape(-1)[0]), "backend": "primitive_graph_runtime"}

    if family == "ode":
        from scipy.integrate import solve_ivp

        times = np.asarray(payload["times"], dtype=float)
        initial = np.asarray(payload["initial"], dtype=float)
        candidate = payload["candidate"]
        coefficients = np.asarray(candidate["coefficients"], dtype=float)
        if coefficients.shape != (1, 1) or initial.shape != (1,) or times.ndim != 1:
            raise ValueError("typed_ode_contract_shape_invalid")
        if times.size < 2 or np.any(np.diff(times) <= 0):
            raise ValueError("typed_ode_time_grid_invalid")
        rate = float(coefficients[0, 0])
        solution = solve_ivp(lambda _t, state: np.asarray([rate * state[0]], dtype=float),
                             (float(times[0]), float(times[-1])), initial,
                             t_eval=times, rtol=1e-8, atol=1e-10, max_step=np.inf)
        if not solution.success or solution.y.shape[1] != times.size:
            raise RuntimeError("typed_ode_solver_failed")
        return {"trajectory": solution.y[0].tolist(), "backend": "scipy.solve_ivp"}

    if family == "optimization":
        from .universal_math_solvers import UniversalRelationValidator, UniversalSolverRegistry

        candidate = dict(payload["candidate"])
        candidate.setdefault("decision_variables", [f"x{index}" for index in range(len(candidate.get("objective_coefficients", [])))])
        candidate.setdefault("units", {name: "1" for name in candidate["decision_variables"]})
        verified = UniversalRelationValidator.verify(candidate)
        if verified.get("validation_errors"):
            raise ValueError("typed_optimization_contract_invalid:" + ",".join(verified["validation_errors"]))
        return UniversalSolverRegistry().execute("linear_program/v1", verified)

    if family == "multi_table":
        from .multitable_cegis import compile_multitable_candidate, materialize_multitable_candidate

        frame = materialize_multitable_candidate(compile_multitable_candidate(payload["candidate"]))
        group_by, value_column = str(payload["group_by"]), str(payload["value_column"])
        if group_by not in frame.columns or value_column not in frame.columns:
            raise ValueError("typed_multitable_aggregation_column_missing")
        totals = frame.groupby(group_by, dropna=False)[value_column].sum()
        return {"rows": int(len(frame)), "group_totals": {str(key): float(value) for key, value in totals.items()},
                "backend": "multitable_cegis"}

    raise ValueError(f"typed_benchmark_family_unsupported:{family}")


def _node(identifier: str, op: str, inputs: Sequence[str] = (), *, value: Any = None,
          dimensions: Mapping[str, float] | None = None) -> dict[str, Any]:
    attributes = {} if value is None else {"value": value}
    return {"id": identifier, "op": op, "inputs": list(inputs),
            "kind": "quantity", "dimensions": dict(dimensions if dimensions is not None else {"Q": 1}),
            "attributes": attributes}


def _algebra_case(index: int, rng: np.random.Generator) -> AutomatedBenchmarkCase:
    family = ("linear", "quadratic", "product", "exponential")[index % 4]
    x, y = float(rng.uniform(-3, 3)), float(rng.uniform(-3, 3))
    if family == "linear":
        a, b = float(rng.uniform(-2, 2)), float(rng.uniform(-2, 2))
        nodes = [_node("x", "variable"), _node("a", "constant", value=a, dimensions={}),
                 _node("b", "constant", value=b),
                 _node("ax", "multiply", ("a", "x")), _node("out", "add", ("ax", "b"))]
        expected = a * x + b
    elif family == "quadratic":
        a, b = float(rng.uniform(-2, 2)), float(rng.uniform(-1, 1))
        nodes = [_node("x", "variable"), _node("a", "constant", value=a, dimensions={}),
                 _node("b", "constant", value=b, dimensions={"Q": 2}),
                 _node("xx", "multiply", ("x", "x"), dimensions={"Q": 2}),
                 _node("axx", "multiply", ("a", "xx"), dimensions={"Q": 2}),
                 _node("out", "add", ("axx", "b"), dimensions={"Q": 2})]
        expected = a * x * x + b
    elif family == "product":
        nodes = [_node("x", "variable"), _node("y", "variable"),
                 _node("out", "multiply", ("x", "y"), dimensions={"Q": 2})]
        expected = x * y
    else:
        k = float(rng.uniform(0.1, 1.5))
        nodes = [_node("x", "variable", dimensions={}), _node("k", "constant", value=-k, dimensions={}),
                 _node("kx", "multiply", ("k", "x"), dimensions={}),
                 _node("out", "exp", ("kx",), dimensions={})]
        expected = math.exp(-k * x)
    return AutomatedBenchmarkCase(
        f"algebra-{index:02d}", "algebra", f"计算 {family} 类型的封闭数学关系。",
        {"kind": "primitive_graph", "nodes": nodes, "bindings": {"x": x, "y": y}, "output_ids": ["out"]},
        {"family": "algebra", "output": float(expected), "tolerance": 1e-8, "structure": family},
        f"algebra-{family}",
    )


def _ode_case(index: int, rng: np.random.Generator) -> AutomatedBenchmarkCase:
    k = float(rng.uniform(0.1, 1.2))
    initial = float(rng.uniform(0.5, 2.0))
    times = np.linspace(0.0, 2.0, 9).round(8).tolist()
    trajectory = (initial * np.exp(-k * np.asarray(times))).tolist()
    candidate = {"id": f"ode-{index}", "state_dim": 1, "basis": ["linear"], "coefficients": [[-k]]}
    public = {"kind": "ode", "times": times, "initial": [initial], "candidate": candidate}
    return AutomatedBenchmarkCase(
        f"ode-{index:02d}", "ode", "从初值模拟一阶衰减动力学并检查留出轨迹。", public,
        {"family": "ode", "trajectory": trajectory, "tolerance": 0.03, "structure": "linear_decay"},
        "ode-linear-decay", source_group="generated-ode",
    )


def _optimization_case(index: int, rng: np.random.Generator) -> AutomatedBenchmarkCase:
    objective = [float(rng.uniform(0.5, 4.0)), float(rng.uniform(0.5, 4.0))]
    upper = [float(rng.uniform(1.0, 5.0)), float(rng.uniform(1.0, 5.0))]
    capacity = float(rng.uniform(upper[0] * 0.7, sum(upper)))
    # Independent closed-form reference for positive maximization coefficients.
    remaining = capacity
    allocation = [0.0, 0.0]
    for position in sorted(range(2), key=lambda item: objective[item], reverse=True):
        allocation[position] = min(upper[position], max(0.0, remaining))
        remaining -= allocation[position]
    optimum = objective[0] * allocation[0] + objective[1] * allocation[1]
    public = {"kind": "optimization", "candidate": {
        "kind": "linear_program", "objective_coefficients": objective,
        "bounds": [[0.0, upper[0]], [0.0, upper[1]]],
        "A_ub": [[1.0, 1.0]], "b_ub": [capacity], "direction": "maximize",
    }}
    return AutomatedBenchmarkCase(
        f"optimization-{index:02d}", "optimization", "在容量和变量上界下最大化线性收益。", public,
        {"family": "optimization", "objective": float(optimum), "tolerance": 1e-6,
         "structure": "bounded_lp", "decision_variables": ["x0", "x1"],
         "objective_coefficients": objective, "bounds": [[0.0, upper[0]], [0.0, upper[1]]],
         "A_ub": [[1.0, 1.0]], "b_ub": [capacity], "direction": "maximize"},
        "optimization-bounded-lp", source_group="generated-optimization",
    )


def _multitable_case(index: int, rng: np.random.Generator) -> AutomatedBenchmarkCase:
    entities = [f"E{j}" for j in range(3)]
    values = [float(rng.uniform(1, 10)) for _ in entities]
    fact = [{"entity": entity, "value": value} for entity, value in zip(entities, values) for _ in range(2)]
    dimension = [{"entity": entity, "group": "north" if j < 2 else "south"} for j, entity in enumerate(entities)]
    expected_group = {"north": float(values[0] + values[0] + values[1] + values[1]),
                      "south": float(values[2] + values[2])}
    public = {"kind": "multitable", "candidate": {
        "tables": {"fact": fact, "dimension": dimension},
        "joins": [{"left_table": "fact", "right_table": "dimension", "left_key": "entity", "right_key": "entity", "how": "left", "aggregate": "first"}],
    }, "group_by": "group", "value_column": "value"}
    return AutomatedBenchmarkCase(
        f"multitable-{index:02d}", "multi_table", "按实体键合并事实表与维度表并检查分组聚合。", public,
        {"family": "multi_table", "rows": 6, "group_totals": expected_group, "tolerance": 1e-8, "structure": "many_to_one_group_sum"},
        "multitable-many-to-one", source_group="generated-multitable",
    )


def build_automated_benchmark_suite(*, cases_per_family: int = 8, seed: int = 20260914) -> tuple[AutomatedBenchmarkCase, ...]:
    if type(cases_per_family) is not int or not 1 <= cases_per_family <= 32:
        raise ValueError("cases_per_family_out_of_bounds")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed_out_of_bounds")
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(cases_per_family):
        cases.extend((_algebra_case(index, rng), _ode_case(index, rng),
                      _optimization_case(index, rng), _multitable_case(index, rng)))
    return tuple(cases)


def _extract_scalar(output: Mapping[str, Any]) -> float | None:
    for key in ("value", "objective", "objective_value", "score"):
        value = output.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    outputs = output.get("outputs")
    if isinstance(outputs, Mapping):
        value = outputs.get("out")
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _score_optimization_solution(output: Mapping[str, Any], reference: Mapping[str, Any],
                                 tolerance: float) -> dict[str, Any]:
    """Recompute feasibility and objective from the submitted decisions.

    A reported optimum is not a solution certificate.  The scorer therefore
    rejects scalar-only answers, ignores solver-authored feasibility claims,
    and independently checks bounds, inequalities, and the objective value.
    """
    names = reference.get("decision_variables")
    raw_solution = output.get("solution", output.get("decisions"))
    if not isinstance(names, Sequence) or isinstance(names, (str, bytes)) or not names:
        return {"valid": False, "score": None, "reason": "reference_decision_variables_missing"}
    try:
        if isinstance(raw_solution, Mapping):
            if set(raw_solution) != set(names):
                raise ValueError
            vector = np.asarray([raw_solution[name] for name in names], dtype=float)
        elif isinstance(raw_solution, Sequence) and not isinstance(raw_solution, (str, bytes)):
            vector = np.asarray(raw_solution, dtype=float)
        else:
            raise ValueError
    except (TypeError, ValueError):
        return {"valid": False, "score": None, "reason": "decision_solution_missing_or_invalid"}
    if vector.shape != (len(names),) or not np.isfinite(vector).all():
        return {"valid": False, "score": None, "reason": "decision_solution_missing_or_invalid"}
    try:
        coefficients = np.asarray(reference["objective_coefficients"], dtype=float)
        bounds = np.asarray(reference["bounds"], dtype=float)
        matrix = np.asarray(reference.get("A_ub", []), dtype=float)
        rhs = np.asarray(reference.get("b_ub", []), dtype=float)
        optimum = float(reference["objective"])
    except (KeyError, TypeError, ValueError):
        return {"valid": False, "score": None, "reason": "optimization_reference_invalid"}
    if coefficients.shape != vector.shape or bounds.shape != (len(names), 2):
        return {"valid": False, "score": None, "reason": "optimization_reference_shape_invalid"}
    bound_violation = float(max(np.max(bounds[:, 0] - vector), np.max(vector - bounds[:, 1]), 0.0))
    if matrix.size:
        if matrix.ndim != 2 or matrix.shape[1] != len(names) or rhs.shape != (matrix.shape[0],):
            return {"valid": False, "score": None, "reason": "optimization_reference_shape_invalid"}
        constraint_violation = float(max(np.max(matrix @ vector - rhs), 0.0))
    else:
        constraint_violation = 0.0
    integer_indices = reference.get("integer_indices", [])
    if (not isinstance(integer_indices, Sequence) or isinstance(integer_indices, (str, bytes))
            or any(type(index) is not int or not 0 <= index < len(names) for index in integer_indices)):
        return {"valid": False, "score": None, "reason": "optimization_reference_integer_indices_invalid"}
    integrality_violation = max(
        (abs(float(vector[index]) - round(float(vector[index]))) for index in integer_indices),
        default=0.0,
    )
    recomputed = float(coefficients @ vector)
    reported = _extract_scalar(output)
    report_mismatch = 0.0 if reported is None else abs(reported - recomputed)
    objective_gap = ((optimum - recomputed) if reference.get("direction") == "maximize"
                     else (recomputed - optimum))
    objective_gap = max(0.0, float(objective_gap))
    score = max(bound_violation, constraint_violation, integrality_violation,
                report_mismatch, objective_gap)
    if bound_violation > tolerance:
        reason = "bound_violation"
    elif constraint_violation > tolerance:
        reason = "constraint_violation"
    elif integrality_violation > tolerance:
        reason = "integrality_violation"
    elif report_mismatch > tolerance:
        reason = "reported_objective_mismatch"
    elif objective_gap > tolerance:
        reason = "objective_gap_exceeds_tolerance"
    else:
        reason = "feasible_solution_within_tolerance"
    return {"valid": score <= tolerance, "score": score, "reason": reason,
            "recomputed_objective": recomputed,
            "maximum_constraint_violation": max(bound_violation, constraint_violation),
            "maximum_integrality_violation": integrality_violation,
            "reported_objective_mismatch": report_mismatch,
            "objective_gap": objective_gap}


def score_automated_benchmark_case(output: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """Independent scorer for the bundled suite's four task families."""
    if not isinstance(output, Mapping) or not isinstance(reference, Mapping):
        return {"valid": False, "score": None, "reason": "mapping_required"}
    family = reference.get("family")
    tolerance = float(reference.get("tolerance", 1e-6))
    if family == "optimization":
        return _score_optimization_solution(output, reference, tolerance)
    if family == "algebra":
        actual = _extract_scalar(output)
        expected = reference.get("output", reference.get("objective"))
        if actual is None or not isinstance(expected, (int, float)):
            return {"valid": False, "score": None, "reason": "scalar_result_missing"}
        error = abs(actual - float(expected))
        return {"valid": error <= tolerance, "score": error,
                "reason": "within_tolerance" if error <= tolerance else "scalar_error_exceeds_tolerance"}
    if family == "ode":
        actual = output.get("trajectory", output.get("predictions"))
        expected = reference.get("trajectory")
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)) or not isinstance(expected, Sequence) or len(actual) != len(expected):
            return {"valid": False, "score": None, "reason": "trajectory_missing_or_shape_mismatch"}
        try:
            error = float(np.sqrt(np.mean((np.asarray(actual, dtype=float) - np.asarray(expected, dtype=float)) ** 2)))
        except (TypeError, ValueError):
            return {"valid": False, "score": None, "reason": "trajectory_non_numeric"}
        return {"valid": math.isfinite(error) and error <= tolerance, "score": error,
                "reason": "trajectory_rmse"}
    if family == "multi_table":
        rows = output.get("rows", output.get("row_count"))
        groups = output.get("group_totals", {})
        expected_groups = reference.get("group_totals", {})
        if not isinstance(rows, (int, float)) or int(rows) != int(reference.get("rows", -1)) or not isinstance(groups, Mapping):
            return {"valid": False, "score": None, "reason": "join_shape_or_group_result_missing"}
        errors = [abs(float(groups.get(key, float("nan"))) - float(value)) for key, value in expected_groups.items()]
        error = max(errors, default=0.0)
        return {"valid": math.isfinite(error) and error <= tolerance, "score": error,
                "reason": "join_aggregate_error"}
    return {"valid": False, "score": None, "reason": "unknown_reference_family"}


def run_automated_benchmark_suite(system, *, cases_per_family: int = 8, seed: int = 20260914, wall_seconds: float = 600.0) -> dict[str, Any]:
    cases = build_automated_benchmark_suite(cases_per_family=cases_per_family, seed=seed)
    return run_automated_benchmark(cases, system, score_automated_benchmark_case, wall_seconds=wall_seconds)


def run_typed_execution_benchmark(*, cases_per_family: int = 8, seed: int = 20260914,
                                  wall_seconds: float = 600.0) -> dict[str, Any]:
    """Run the deterministic suite against the built-in typed backend baseline."""
    return run_automated_benchmark_suite(typed_execution_baseline,
                                          cases_per_family=cases_per_family,
                                          seed=seed, wall_seconds=wall_seconds)


__all__ = ["build_automated_benchmark_suite", "score_automated_benchmark_case",
           "run_automated_benchmark_suite", "typed_execution_baseline",
           "run_typed_execution_benchmark"]
