"""Optional mature genetic-programming baseline for numeric relation tasks."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np


GPLEARN_SOURCE_REVISION = "0390aea8639ce5f6c0b388400e07b58c05acad6a"
GPLEARN_FUNCTION_SET = ("add", "sub", "mul", "div", "sqrt", "sin", "cos")


class GPLearnBaselineError(ValueError):
    pass


def _serialize_program(program: Any) -> list[dict[str, Any]]:
    result = []
    for node in program.program:
        if hasattr(node, "name") and hasattr(node, "arity"):
            result.append({"function": str(node.name), "arity": int(node.arity)})
        elif type(node) in (int, np.int32, np.int64):
            result.append({"feature_index": int(node)})
        elif type(node) in (float, np.float32, np.float64):
            result.append({"constant": float(node)})
        else:
            raise GPLearnBaselineError("gplearn_program_node_unsupported")
    return result


def fit_gplearn_baseline(
    payload: Mapping[str, Any], *, maximum_program_evaluations: int = 20_000,
) -> dict[str, Any]:
    """Fit official gplearn under a deterministic, bounded configuration."""
    if not isinstance(payload, Mapping):
        raise GPLearnBaselineError("gplearn_payload_invalid")
    attachments = payload.get("attachments")
    if not isinstance(attachments, list) or len(attachments) != 1:
        raise GPLearnBaselineError("gplearn_single_attachment_required")
    rows = attachments[0].get("rows") if isinstance(attachments[0], Mapping) else None
    if not isinstance(rows, list) or len(rows) < 32 or any(not isinstance(row, Mapping) for row in rows):
        raise GPLearnBaselineError("gplearn_observations_invalid")
    columns = set(rows[0])
    if "response" not in columns or any(set(row) != columns for row in rows):
        raise GPLearnBaselineError("gplearn_response_binding_required")
    inputs = sorted(columns - {"response"})
    if not 1 <= len(inputs) <= 6:
        raise GPLearnBaselineError("gplearn_input_count_unsupported")
    x = np.asarray([[row[name] for name in inputs] for row in rows], dtype=float)
    y = np.asarray([row["response"] for row in rows], dtype=float)
    queries = np.asarray(payload.get("query_inputs"), dtype=float)
    if (x.shape != (len(rows), len(inputs)) or queries.ndim != 2
            or queries.shape[1] != len(inputs)
            or not np.isfinite(x).all() or not np.isfinite(y).all()
            or not np.isfinite(queries).all()):
        raise GPLearnBaselineError("gplearn_numeric_data_invalid")
    population_size = int(payload.get("gplearn_population_size", 1000))
    generations = int(payload.get("gplearn_generations", 20))
    seed = int(payload.get("gplearn_seed", 20261010))
    if (not 100 <= population_size <= 5000 or not 1 <= generations <= 100
            or population_size * generations > maximum_program_evaluations
            or not 0 <= seed <= 2**32 - 1):
        raise GPLearnBaselineError("gplearn_search_budget_invalid")
    try:
        import gplearn
        from gplearn.genetic import SymbolicRegressor
    except ImportError as exc:
        raise GPLearnBaselineError("gplearn_dependency_unavailable") from exc
    if getattr(gplearn, "__version__", None) != "0.5.dev0":
        raise GPLearnBaselineError("gplearn_version_unexpected")

    # Preserve the official SRSD numeric representation. Per-column affine
    # normalization can turn a one-node product into a product plus multiple
    # linear and constant terms, changing the symbolic search problem itself.
    x_center, x_scale = np.zeros(x.shape[1]), np.ones(x.shape[1])
    y_center, y_scale = 0.0, 1.0
    estimator = SymbolicRegressor(
        population_size=population_size, generations=generations,
        function_set=GPLEARN_FUNCTION_SET, metric="mse",
        parsimony_coefficient=0.001, stopping_criteria=0.0,
        random_state=seed, n_jobs=1, verbose=0, low_memory=True,
    )
    estimator.fit(x, y)
    standardized_prediction = estimator.predict((queries - x_center) / x_scale)
    prediction = y_center + y_scale * np.asarray(standardized_prediction, dtype=float)
    if prediction.shape != (len(queries),) or not np.isfinite(prediction).all():
        raise GPLearnBaselineError("gplearn_prediction_invalid")
    completed_generations = len(estimator.run_details_.get("generation", []))
    model = {
        "family": "symbolic_regression_baseline",
        "structure": "gplearn_prefix_program",
        "input_variables": inputs,
        "response_variable": "response",
        "program": _serialize_program(estimator._program),
        "x_center": x_center.tolist(), "x_scale": x_scale.tolist(),
        "y_center": y_center, "y_scale": y_scale,
        "function_set": list(GPLEARN_FUNCTION_SET),
        "source_repository": "trevorstephens/gplearn",
        "source_revision": GPLEARN_SOURCE_REVISION,
        "package_version": str(gplearn.__version__),
        "population_size": population_size, "generation_limit": generations,
        "completed_generations": completed_generations,
        "program_evaluation_upper_bound": population_size * max(completed_generations, 1),
        "random_seed": seed,
    }
    return {
        "status": "completed", "family": "modeling_algebra", "model": model,
        "predictions": prediction.tolist(),
        "usage": {"model_api_calls": 0, "numerical_solver_calls": 1,
                  "manual_interventions": 0,
                  "program_evaluation_upper_bound": model["program_evaluation_upper_bound"]},
        "policy": "official_gplearn_fixed_revision;deterministic_seed;bounded_program_budget",
    }


def fit_gplearn_baseline_isolated(
    payload: Mapping[str, Any], *, wall_seconds: float = 30.0, memory_mb: int = 1024,
    maximum_program_evaluations: int = 20_000,
) -> dict[str, Any]:
    from .solver_runtime import SolverLimits, SolverProcessRunner, SolverRuntimeError
    try:
        return SolverProcessRunner().execute(
            "gplearn_symbolic_regression/v1", dict(payload),
            limits=SolverLimits(wall_seconds=wall_seconds, memory_mb=memory_mb,
                                max_evaluations=maximum_program_evaluations),
        )
    except SolverRuntimeError as exc:
        return {
            "status": "not_assessed", "reason": exc.code,
            "usage": {"model_api_calls": 0, "numerical_solver_calls": 0,
                      "manual_interventions": 0},
            "execution_supervision": exc.metadata,
            "policy": "resource_or_dependency_failure_is_not_a_mathematical_verdict",
        }


__all__ = ["GPLEARN_FUNCTION_SET", "GPLEARN_SOURCE_REVISION", "GPLearnBaselineError",
           "fit_gplearn_baseline", "fit_gplearn_baseline_isolated"]
