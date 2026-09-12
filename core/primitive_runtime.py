"""Small execution runtime for domain-neutral mathematical primitives."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


class PrimitiveRuntimeError(ValueError):
    pass


class PrimitiveRuntime:
    """Execute bounded primitive contracts; callers still own evidence checks."""

    _OPS = {
        "linear_system", "least_squares", "polynomial_roots", "integral", "expectation",
        "linear_ode", "threshold_event", "quadratic_program", "distance",
        "interval_union", "region_membership", "segment_intersection",
        "line_of_sight",
        "normal_log_likelihood", "bootstrap_mean", "permutation_test",
    }

    def execute(self, operation: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        operation = str(operation)
        if operation not in self._OPS or not isinstance(payload, Mapping):
            raise PrimitiveRuntimeError("unsupported_or_invalid_primitive")
        try:
            if operation == "linear_system":
                matrix = np.asarray(payload["matrix"], dtype=float)
                rhs = np.asarray(payload["rhs"], dtype=float)
                if (matrix.ndim != 2 or rhs.ndim != 1 or matrix.shape[0] != matrix.shape[1]
                        or len(rhs) != matrix.shape[0] or not 1 <= matrix.shape[0] <= 512
                        or not np.isfinite(matrix).all() or not np.isfinite(rhs).all()):
                    raise PrimitiveRuntimeError("linear_system_shape_limit")
                solution = np.linalg.solve(matrix, rhs)
                residual = float(np.max(np.abs(matrix @ solution - rhs)))
                return {"operation": operation, "status": "executed", "solution": solution.tolist(),
                        "residual_inf": residual, "checks": {"finite": bool(np.isfinite(solution).all())}}
            if operation == "least_squares":
                design = np.asarray(payload["design"], dtype=float)
                target = np.asarray(payload["target"], dtype=float)
                if (design.ndim != 2 or target.ndim != 1 or design.shape[0] != len(target)
                        or not 1 <= design.shape[0] or not 1 <= design.shape[1]
                        or design.size > 2_000_000 or not np.isfinite(design).all()
                        or not np.isfinite(target).all()):
                    raise PrimitiveRuntimeError("least_squares_shape_limit")
                coefficients, _, rank, singular = np.linalg.lstsq(design, target, rcond=None)
                residual = float(np.linalg.norm(design @ coefficients - target))
                return {"operation": operation, "status": "executed", "coefficients": coefficients.tolist(),
                        "residual_l2": residual, "rank": int(rank), "singular_values": singular.tolist()}
            if operation == "polynomial_roots":
                coefficients = np.asarray(payload["coefficients"], dtype=float)
                if coefficients.ndim != 1 or not 1 <= len(coefficients) <= 129 or not np.isfinite(coefficients).all():
                    raise PrimitiveRuntimeError("polynomial_coefficients_invalid")
                roots = np.roots(coefficients)
                if not np.isfinite(roots).all():
                    raise PrimitiveRuntimeError("nonfinite_roots")
                return {"operation": operation, "status": "executed",
                        "roots": [{"real": float(root.real), "imag": float(root.imag)} for root in roots]}
            if operation == "expectation":
                values = np.asarray(payload["values"], dtype=float)
                weights = np.asarray(payload.get("weights", np.ones(len(values))), dtype=float)
                if values.ndim != 1 or weights.ndim != 1 or len(values) != len(weights) or not len(values) or len(values) > 1_000_000 or not np.isfinite(values).all() or not np.isfinite(weights).all() or np.any(weights < 0) or float(weights.sum()) <= 0:
                    raise PrimitiveRuntimeError("expectation_input_invalid")
                normalized = weights / weights.sum()
                value = float(np.dot(values, normalized))
                return {"operation": operation, "status": "executed", "expectation": value,
                        "weight_sum": float(weights.sum())}
            if operation == "linear_ode":
                matrix = np.asarray(payload["matrix"], dtype=float)
                initial = np.asarray(payload["initial"], dtype=float)
                times = np.asarray(payload["times"], dtype=float)
                if (
                    matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]
                    or initial.ndim != 1 or len(initial) != matrix.shape[0]
                    or times.ndim != 1 or len(times) < 2 or len(times) > 10_000
                    or matrix.shape[0] > 64 or matrix.shape[0] ** 2 * len(times) > 2_000_000
                    or not np.isfinite(matrix).all() or not np.isfinite(initial).all()
                    or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
                ):
                    raise PrimitiveRuntimeError("linear_ode_input_invalid")
                from scipy.linalg import expm

                start = float(times[0])
                states = np.vstack([expm(matrix * float(time - start)) @ initial for time in times])
                if not np.isfinite(states).all():
                    raise PrimitiveRuntimeError("linear_ode_nonfinite")
                return {
                    "operation": operation, "status": "executed", "method": "matrix_exponential",
                    "times": times.tolist(), "states": states.tolist(),
                    "checks": {"finite": True, "state_dimension": int(matrix.shape[0])},
                }
            if operation == "threshold_event":
                times = np.asarray(payload["times"], dtype=float)
                values = np.asarray(payload["values"], dtype=float)
                threshold = float(payload["threshold"])
                direction = str(payload.get("direction", "either"))
                if (
                    times.ndim != 1 or values.ndim != 1 or len(times) != len(values)
                    or len(times) < 2 or len(times) > 100_000 or direction not in {"rise", "fall", "either"}
                    or not np.isfinite(times).all() or not np.isfinite(values).all()
                    or not math.isfinite(threshold) or np.any(np.diff(times) <= 0)
                ):
                    raise PrimitiveRuntimeError("threshold_event_input_invalid")
                for index in range(1, len(times)):
                    left, right = float(values[index - 1]), float(values[index])
                    rising = left < threshold <= right
                    falling = left > threshold >= right
                    if (direction == "rise" and not rising) or (direction == "fall" and not falling):
                        if direction != "either":
                            continue
                    if not (rising or falling):
                        continue
                    span = right - left
                    fraction = 0.0 if span == 0 else (threshold - left) / span
                    event_time = float(times[index - 1] + fraction * (times[index] - times[index - 1]))
                    return {
                        "operation": operation, "status": "executed", "crossing_found": True,
                        "event_time": event_time, "segment_index": index - 1,
                        "direction": "rise" if rising else "fall",
                    }
                return {"operation": operation, "status": "executed", "crossing_found": False,
                        "event_time": None, "direction": direction}
            if operation == "quadratic_program":
                quadratic = np.asarray(payload["quadratic_matrix"], dtype=float)
                linear = np.asarray(payload["linear"], dtype=float)
                if (
                    quadratic.ndim != 2 or quadratic.shape[0] != quadratic.shape[1]
                    or quadratic.shape[0] < 1
                    or linear.ndim != 1 or len(linear) != quadratic.shape[0]
                    or quadratic.shape[0] > 128 or not np.isfinite(quadratic).all()
                    or not np.isfinite(linear).all()
                ):
                    raise PrimitiveRuntimeError("quadratic_program_shape_invalid")
                if not np.allclose(quadratic, quadratic.T, rtol=1e-10, atol=1e-12):
                    raise PrimitiveRuntimeError("quadratic_program_matrix_not_symmetric")
                eigenvalues = np.linalg.eigvalsh(quadratic)
                if float(np.min(eigenvalues)) < -1e-8:
                    return {
                        "operation": operation, "status": "not_assessed",
                        "reason": "nonconvex_quadratic_objective",
                        "minimum_eigenvalue": float(np.min(eigenvalues)),
                    }
                raw_a = payload.get("inequality_matrix")
                raw_b = payload.get("inequality_rhs")
                if raw_a is None:
                    inequality = np.empty((0, quadratic.shape[0]), dtype=float)
                    rhs = np.empty(0, dtype=float)
                else:
                    inequality = np.asarray(raw_a, dtype=float)
                    rhs = np.asarray(raw_b, dtype=float)
                    if inequality.ndim != 2 or inequality.shape[1] != quadratic.shape[0] or rhs.ndim != 1 or len(rhs) != inequality.shape[0]:
                        raise PrimitiveRuntimeError("quadratic_program_constraint_shape_invalid")
                    if not np.isfinite(inequality).all() or not np.isfinite(rhs).all():
                        raise PrimitiveRuntimeError("quadratic_program_constraint_nonfinite")
                raw_bounds = payload.get("bounds")
                if not isinstance(raw_bounds, list) or len(raw_bounds) != quadratic.shape[0]:
                    raise PrimitiveRuntimeError("quadratic_program_bounds_invalid")
                bounds = []
                for bound in raw_bounds:
                    if not isinstance(bound, (list, tuple)) or len(bound) != 2:
                        raise PrimitiveRuntimeError("quadratic_program_bounds_invalid")
                    lower = -np.inf if bound[0] is None else float(bound[0])
                    upper = np.inf if bound[1] is None else float(bound[1])
                    # Only the explicit open-bound markers may produce an
                    # infinity.  Reject +inf lower and -inf upper bounds:
                    # they describe an empty/undefined interval.
                    invalid_lower = not math.isfinite(lower) and lower != -np.inf
                    invalid_upper = not math.isfinite(upper) and upper != np.inf
                    if invalid_lower or invalid_upper or lower == np.inf or upper == -np.inf or lower > upper:
                        raise PrimitiveRuntimeError("quadratic_program_bounds_invalid")
                    bounds.append((lower, upper))
                x0 = payload.get("initial")
                if x0 is None:
                    x0_array = np.asarray([
                        0.0 if not np.isfinite(lower + upper) else (lower + upper) / 2.0
                        for lower, upper in bounds
                    ], dtype=float)
                else:
                    x0_array = np.asarray(x0, dtype=float)
                    if x0_array.ndim != 1 or len(x0_array) != len(bounds) or not np.isfinite(x0_array).all():
                        raise PrimitiveRuntimeError("quadratic_program_initial_invalid")
                x0_array = np.asarray([
                    min(max(value, lower), upper) for value, (lower, upper) in zip(x0_array, bounds)
                ], dtype=float)
                max_iterations = payload.get("max_iterations", 300)
                if type(max_iterations) is not int or not 1 <= max_iterations <= 2_000:
                    raise PrimitiveRuntimeError("quadratic_program_iteration_limit")
                from scipy.optimize import minimize

                objective = lambda vector: float(0.5 * vector @ quadratic @ vector + linear @ vector)
                constraints = [] if not len(inequality) else [{
                    "type": "ineq", "fun": lambda vector, row=row, limit=limit: float(limit - row @ vector)
                } for row, limit in zip(inequality, rhs)]
                result = minimize(objective, x0_array, method="SLSQP", bounds=bounds,
                                  constraints=constraints, options={"maxiter": max_iterations, "ftol": 1e-9})
                if result.x is None:
                    return {
                        "operation": operation, "status": "not_converged",
                        "solution": None, "objective": None,
                        "maximum_constraint_violation": None,
                        "iterations": int(getattr(result, "nit", 0) or 0),
                        "convexity": {"minimum_eigenvalue": float(np.min(eigenvalues)), "status": "pass"},
                        "solver_message": str(getattr(result, "message", ""))[:160],
                    }
                solution = np.asarray(result.x, dtype=float)
                violation = float(max(
                    np.max(inequality @ solution - rhs) if len(inequality) else 0.0,
                    max((max(lower - value, value - upper, 0.0) for value, (lower, upper) in zip(solution, bounds)), default=0.0),
                ))
                status = "executed" if bool(result.success) and np.isfinite(solution).all() and violation <= 1e-6 else "not_converged"
                return {
                    "operation": operation, "status": status, "solution": solution.tolist(),
                    "objective": objective(solution), "maximum_constraint_violation": violation,
                    "iterations": int(getattr(result, "nit", 0) or 0),
                    "convexity": {"minimum_eigenvalue": float(np.min(eigenvalues)), "status": "pass"},
                    "solver_message": str(getattr(result, "message", ""))[:160],
                }
            if operation == "distance":
                left = np.asarray(payload["left"], dtype=float)
                right = np.asarray(payload["right"], dtype=float)
                metric = str(payload.get("metric", "euclidean"))
                if (left.ndim != 1 or right.ndim != 1 or left.shape != right.shape
                        or not 1 <= len(left) <= 128 or not np.isfinite(left).all()
                        or not np.isfinite(right).all()):
                    raise PrimitiveRuntimeError("distance_input_invalid")
                if metric == "euclidean":
                    value = float(np.linalg.norm(left - right))
                elif metric == "manhattan":
                    value = float(np.abs(left - right).sum())
                elif metric == "haversine":
                    if len(left) != 2:
                        raise PrimitiveRuntimeError("haversine_requires_latitude_longitude")
                    lat1, lon1, lat2, lon2 = np.radians([left[0], left[1], right[0], right[1]])
                    dlat, dlon = lat2 - lat1, lon2 - lon1
                    a = float(np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2)
                    value = float(2.0 * 6371008.8 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))
                else:
                    raise PrimitiveRuntimeError("distance_metric_unsupported")
                return {"operation": operation, "status": "executed", "metric": metric,
                        "distance": value, "checks": {"finite": math.isfinite(value)}}
            if operation == "interval_union":
                raw_intervals = payload["intervals"]
                if (not isinstance(raw_intervals, (list, tuple)) or not raw_intervals
                        or len(raw_intervals) > 100_000):
                    raise PrimitiveRuntimeError("interval_union_input_invalid")
                intervals: list[tuple[float, float]] = []
                for item in raw_intervals:
                    if not isinstance(item, (list, tuple)) or len(item) != 2:
                        raise PrimitiveRuntimeError("interval_union_input_invalid")
                    start, end = float(item[0]), float(item[1])
                    if not math.isfinite(start) or not math.isfinite(end) or start > end:
                        raise PrimitiveRuntimeError("interval_union_interval_invalid")
                    intervals.append((start, end))
                intervals.sort(key=lambda pair: (pair[0], pair[1]))
                merged: list[list[float]] = []
                for start, end in intervals:
                    if not merged or start > merged[-1][1]:
                        merged.append([start, end])
                    else:
                        merged[-1][1] = max(merged[-1][1], end)
                measure = float(sum(end - start for start, end in merged))
                return {"operation": operation, "status": "executed", "intervals": merged,
                        "measure": measure, "input_count": len(intervals)}
            if operation == "region_membership":
                point = np.asarray(payload["point"], dtype=float)
                if point.ndim != 1 or not 1 <= len(point) <= 16 or not np.isfinite(point).all():
                    raise PrimitiveRuntimeError("region_point_invalid")
                region = payload.get("region")
                if isinstance(region, Mapping) and "lower" in region and "upper" in region:
                    lower = np.asarray(region["lower"], dtype=float)
                    upper = np.asarray(region["upper"], dtype=float)
                    if (lower.shape != point.shape or upper.shape != point.shape
                            or not np.isfinite(lower).all() or not np.isfinite(upper).all()
                            or np.any(lower > upper)):
                        raise PrimitiveRuntimeError("region_bounds_invalid")
                    inside = bool(np.all(point >= lower) and np.all(point <= upper))
                    kind = "box"
                elif isinstance(region, (list, tuple)):
                    polygon = np.asarray(region, dtype=float)
                    if (point.shape != (2,) or polygon.ndim != 2 or polygon.shape[1] != 2
                            or not 3 <= len(polygon) <= 10_000 or not np.isfinite(polygon).all()):
                        raise PrimitiveRuntimeError("region_polygon_invalid")
                    inside = False
                    x, y = float(point[0]), float(point[1])
                    # Ray casting; points on an edge count as inside.
                    for index in range(len(polygon)):
                        ax, ay = polygon[index - 1]
                        bx, by = polygon[index]
                        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
                        dot = (x - ax) * (x - bx) + (y - ay) * (y - by)
                        if abs(float(cross)) <= 1e-12 and dot <= 1e-12:
                            inside = True
                            break
                        if (ay > y) != (by > y):
                            x_cross = (bx - ax) * (y - ay) / (by - ay) + ax
                            if x < x_cross:
                                inside = not inside
                    kind = "polygon"
                else:
                    raise PrimitiveRuntimeError("region_definition_invalid")
                return {"operation": operation, "status": "executed", "inside": inside,
                        "region_type": kind}
            if operation == "segment_intersection":
                first = np.asarray(payload["first"], dtype=float)
                second = np.asarray(payload["second"], dtype=float)
                if (first.shape != (2, 2) or second.shape != (2, 2)
                        or not np.isfinite(first).all() or not np.isfinite(second).all()):
                    raise PrimitiveRuntimeError("segment_input_invalid")
                p, p2 = first
                q, q2 = second

                def orientation(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
                    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

                def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
                    return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
                            and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)

                o1, o2, o3, o4 = orientation(p, p2, q), orientation(p, p2, q2), orientation(q, q2, p), orientation(q, q2, p2)
                intersects = False
                if ((o1 > 1e-12 and o2 < -1e-12) or (o1 < -1e-12 and o2 > 1e-12)) and ((o3 > 1e-12 and o4 < -1e-12) or (o3 < -1e-12 and o4 > 1e-12)):
                    intersects = True
                elif ((abs(o1) <= 1e-12 and on_segment(p, p2, q))
                      or (abs(o2) <= 1e-12 and on_segment(p, p2, q2))
                      or (abs(o3) <= 1e-12 and on_segment(q, q2, p))
                      or (abs(o4) <= 1e-12 and on_segment(q, q2, p2))):
                    intersects = True
                point = None
                denominator = float((p[0] - p2[0]) * (q[1] - q2[1]) - (p[1] - p2[1]) * (q[0] - q2[0]))
                if intersects and abs(denominator) > 1e-12:
                    t = float(((p[0] - q[0]) * (q[1] - q2[1]) - (p[1] - q[1]) * (q[0] - q2[0])) / denominator)
                    point = (p + t * (p2 - p)).tolist()
                return {"operation": operation, "status": "executed", "intersects": intersects,
                        "intersection_point": point, "parallel_or_collinear": abs(denominator) <= 1e-12}
            if operation == "line_of_sight":
                source = np.asarray(payload["source"], dtype=float)
                target = np.asarray(payload["target"], dtype=float)
                obstacles = payload["obstacles"]
                if (source.shape != (2,) or target.shape != (2,) or not np.isfinite(source).all()
                        or not np.isfinite(target).all() or not isinstance(obstacles, list)
                        or len(obstacles) > 1_000):
                    raise PrimitiveRuntimeError("line_of_sight_input_invalid")
                if np.array_equal(source, target):
                    raise PrimitiveRuntimeError("line_of_sight_degenerate_segment")

                def orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
                    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))

                def on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
                    return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
                            and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)

                def crosses(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
                    values = (orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b))
                    return (((values[0] > 1e-12 and values[1] < -1e-12) or (values[0] < -1e-12 and values[1] > 1e-12))
                            and ((values[2] > 1e-12 and values[3] < -1e-12) or (values[2] < -1e-12 and values[3] > 1e-12))) or any(
                                abs(value) <= 1e-12 and on_segment(left, right, point)
                                for value, left, right, point in (
                                    (values[0], a, b, c), (values[1], a, b, d),
                                    (values[2], c, d, a), (values[3], c, d, b)))

                blocked_by = []
                for obstacle_index, raw_polygon in enumerate(obstacles):
                    polygon = np.asarray(raw_polygon, dtype=float)
                    if (polygon.ndim != 2 or polygon.shape[1] != 2 or not 3 <= len(polygon) <= 10_000
                            or not np.isfinite(polygon).all()):
                        raise PrimitiveRuntimeError("line_of_sight_obstacle_invalid")
                    for vertex_index in range(len(polygon)):
                        if crosses(source, target, polygon[vertex_index - 1], polygon[vertex_index]):
                            blocked_by.append(obstacle_index)
                            break
                return {"operation": operation, "status": "executed", "visible": not blocked_by,
                        "blocked_by": blocked_by, "obstacle_count": len(obstacles)}
            if operation == "normal_log_likelihood":
                observations = np.asarray(payload["observations"], dtype=float)
                mean = float(payload["mean"])
                standard_deviation = float(payload["standard_deviation"])
                if (observations.ndim != 1 or not 1 <= len(observations) <= 1_000_000
                        or not np.isfinite(observations).all() or not math.isfinite(mean)
                        or not math.isfinite(standard_deviation) or standard_deviation <= 0):
                    raise PrimitiveRuntimeError("normal_likelihood_input_invalid")
                standardized = (observations - mean) / standard_deviation
                with np.errstate(over="ignore", invalid="ignore"):
                    terms = standardized ** 2 + math.log(2 * math.pi * standard_deviation ** 2)
                    value = float(-0.5 * np.sum(terms))
                if not math.isfinite(value):
                    raise PrimitiveRuntimeError("normal_likelihood_nonfinite")
                return {"operation": operation, "status": "executed", "log_likelihood": value,
                        "sample_count": len(observations), "distribution": "normal"}
            if operation == "bootstrap_mean":
                values = np.asarray(payload["values"], dtype=float)
                resamples = payload.get("resamples", 2_000)
                seed = payload.get("seed", 0)
                if (values.ndim != 1 or not 2 <= len(values) <= 1_000_000
                        or not np.isfinite(values).all() or type(resamples) is not int
                        or not 100 <= resamples <= 20_000 or resamples * len(values) > 5_000_000
                        or type(seed) is not int):
                    raise PrimitiveRuntimeError("bootstrap_input_invalid")
                generator = np.random.default_rng(seed)
                draws = generator.integers(0, len(values), size=(resamples, len(values)), endpoint=False)
                means = values[draws].mean(axis=1)
                interval = np.quantile(means, [0.025, 0.975])
                if not np.isfinite(means).all() or not np.isfinite(interval).all():
                    raise PrimitiveRuntimeError("bootstrap_nonfinite")
                return {"operation": operation, "status": "executed", "mean": float(values.mean()),
                        "standard_error": float(means.std(ddof=1)),
                        "confidence_interval_95": [float(interval[0]), float(interval[1])],
                        "resamples": resamples, "seed": seed}
            if operation == "permutation_test":
                first = np.asarray(payload["first"], dtype=float)
                second = np.asarray(payload["second"], dtype=float)
                permutations = payload.get("permutations", 2_000)
                seed = payload.get("seed", 0)
                if (first.ndim != 1 or second.ndim != 1 or len(first) < 2 or len(second) < 2
                        or len(first) + len(second) > 200_000 or not np.isfinite(first).all()
                        or not np.isfinite(second).all() or type(permutations) is not int
                        or not 100 <= permutations <= 20_000 or permutations * (len(first) + len(second)) > 10_000_000
                        or type(seed) is not int):
                    raise PrimitiveRuntimeError("permutation_input_invalid")
                observed = float(first.mean() - second.mean())
                pooled = np.concatenate([first, second])
                generator = np.random.default_rng(seed)
                extreme = 0
                for _ in range(permutations):
                    shuffled = generator.permutation(pooled)
                    difference = float(shuffled[:len(first)].mean() - shuffled[len(first):].mean())
                    if not math.isfinite(difference):
                        raise PrimitiveRuntimeError("permutation_nonfinite")
                    extreme += abs(difference) >= abs(observed) - 1e-15
                p_value = float((extreme + 1) / (permutations + 1))
                return {"operation": operation, "status": "executed", "observed_difference": observed,
                        "p_value_two_sided": p_value, "permutations": permutations, "seed": seed}
            # Integral is intentionally a fixed Gauss-Legendre rule: no code
            # strings or arbitrary callbacks enter this runtime.
            coefficients = np.asarray(payload["polynomial_coefficients"], dtype=float)
            lower, upper = float(payload["lower"]), float(payload["upper"])
            if (coefficients.ndim != 1 or not 1 <= len(coefficients) <= 65
                    or not np.isfinite(coefficients).all() or not math.isfinite(lower)
                    or not math.isfinite(upper) or lower >= upper):
                raise PrimitiveRuntimeError("integral_contract_invalid")
            nodes, weights = np.polynomial.legendre.leggauss(min(128, max(8, len(coefficients) * 2)))
            x = (upper - lower) * (nodes + 1) / 2 + lower
            # Coefficients use the stable ascending convention [c0, c1, ...].
            value = float((upper - lower) / 2 * np.dot(weights, np.polynomial.polynomial.polyval(x, coefficients)))
            return {"operation": operation, "status": "executed", "integral": value,
                    "quadrature_order": int(len(nodes)), "method": "bounded_gauss_legendre"}
        except (KeyError, TypeError, ValueError, OverflowError, np.linalg.LinAlgError) as exc:
            if isinstance(exc, PrimitiveRuntimeError):
                raise
            raise PrimitiveRuntimeError("primitive_execution_failed") from exc


__all__ = ["PrimitiveRuntimeError", "PrimitiveRuntime"]
