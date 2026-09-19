"""Bounded induction from prose-adjacent raw record attachments.

This module is deliberately narrow and deterministic.  It infers a supported
model family from attachment schemas, fits several explicit candidates, and
returns a model plus an independently recomputable answer.  Unsupported or
ambiguous inputs return ``needs_input`` instead of manufacturing a contract.
"""

from __future__ import annotations

import math
import re
from itertools import combinations, product
from typing import Any, Mapping, Sequence

import numpy as np


class AutomaticModelingError(ValueError):
    pass


_RESPONSE_ALIASES = {"response", "observed_value", "target", "outcome", "output", "measurement", "y"}
_TIME_ALIASES = {"time", "timestamp", "datetime", "date", "t", "时间", "时刻"}
_STATE_ALIASES = {"state", "observed_state", "state_value", "population", "level", "y", "状态"}
_ITEM_ALIASES = {"item", "product", "option", "decision", "方案", "产品"}
_RESOURCE_ALIASES = {"resource", "resource_name", "constraint", "资源"}
_CAPACITY_ALIASES = {"capacity", "limit", "available", "budget", "容量", "上限"}
_MAXIMIZE_ALIASES = {"profit", "benefit", "revenue", "value", "收益", "利润"}
_MINIMIZE_ALIASES = {"cost", "expense", "price", "成本", "费用"}
_FIXED_COST_ALIASES = {"fixed_cost", "setup_cost", "activation_cost", "固定成本", "启动成本"}
_MINIMUM_LOT_ALIASES = {"minimum_lot", "min_lot", "minimum_batch", "最小批量"}
_ACTIVATION_GROUP_ALIASES = {"activation_group", "exclusive_group", "互斥组"}


def _normalized_name(value: Any) -> str:
    return re.sub(r"[^\w]+", "_", str(value).strip().lower(), flags=re.UNICODE).strip("_")


def _alias_column(columns: Sequence[Any], aliases: set[str], *, explicit: Any = None) -> str | None:
    names = [str(item) for item in columns]
    if explicit is not None:
        exact = [name for name in names if name == str(explicit)]
        if len(exact) != 1:
            raise AutomaticModelingError("explicit_column_binding_invalid")
        return exact[0]
    matches = [name for name in names if _normalized_name(name) in aliases]
    return matches[0] if len(matches) == 1 else None


def _number_assignment(problem: str, names: Sequence[str]) -> float | None:
    """Read only an explicit ``name = number`` assignment; never guess from prose numbers."""
    for name in sorted({_normalized_name(item) for item in names if str(item).strip()}, key=len, reverse=True):
        display = re.escape(name).replace("_", r"[\s_-]*")
        match = re.search(rf"(?<![\w]){display}\s*(?:=|＝|为|是)\s*"
                          r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
                          problem, flags=re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def _bind_queries_from_problem(payload: Mapping[str, Any], family: str,
                               tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    bound = dict(payload)
    problem = bound.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        return bound
    if family == "modeling_algebra" and "query_inputs" not in bound:
        for rows in tables.values():
            target = _alias_column(list(rows[0]), _RESPONSE_ALIASES,
                                   explicit=bound.get("response_column"))
            if target is None:
                continue
            inputs = sorted(str(column) for column in rows[0] if str(column) != target)
            assignments = [_number_assignment(problem, [name, f"x{index + 1}"])
                           for index, name in enumerate(inputs)]
            if len(inputs) == 1 and assignments[0] is None:
                assignments[0] = _number_assignment(problem, ["x", "input", "输入"])
            if assignments and all(value is not None for value in assignments):
                bound["query_inputs"] = [[float(value) for value in assignments]]
                bound["query_binding"] = "explicit_problem_assignments"
            break
    elif family == "modeling_ode" and "query_times" not in bound:
        query = _number_assignment(problem, ["time", "t", "时间", "时刻"])
        if query is not None:
            bound["query_times"] = [query]
            bound["query_binding"] = "explicit_problem_assignments"
    return bound


def _attachments(payload: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    raw = payload.get("attachments")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)) or not 1 <= len(raw) <= 8:
        raise AutomaticModelingError("record_attachments_required")
    result: dict[str, list[dict[str, Any]]] = {}
    total_rows = 0
    for attachment in raw:
        if not isinstance(attachment, Mapping) or attachment.get("format") != "records":
            raise AutomaticModelingError("record_attachment_invalid")
        name, rows = attachment.get("name"), attachment.get("rows")
        if not isinstance(name, str) or not name or name in result:
            raise AutomaticModelingError("attachment_name_invalid")
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
            raise AutomaticModelingError("attachment_rows_invalid")
        if any(not isinstance(row, Mapping) for row in rows):
            raise AutomaticModelingError("attachment_row_invalid")
        total_rows += len(rows)
        if total_rows > 10_000:
            raise AutomaticModelingError("attachment_row_budget_exceeded")
        result[name] = [dict(row) for row in rows]
    return result


def _infer_family(payload: Mapping[str, Any], tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    columns = [set(table[0]) for table in tables.values()]
    if "query_times" in payload:
        return "modeling_ode"
    if "query_inputs" in payload:
        return "modeling_algebra"
    if any(_alias_column(list(item), _TIME_ALIASES) is not None and
           (_alias_column(list(item), _STATE_ALIASES) is not None or
            _alias_column(list(item), _RESPONSE_ALIASES) is not None or
            any(_normalized_name(name).startswith("state_") for name in item)) for item in columns):
        return "modeling_ode"
    explicit_response = payload.get("response_column")
    if explicit_response is not None:
        matches = [item for item in columns if str(explicit_response) in item]
        if len(matches) != 1:
            raise AutomaticModelingError("explicit_column_binding_invalid")
        return "modeling_algebra"
    if any(_alias_column(list(item), _RESPONSE_ALIASES) is not None for item in columns):
        return "modeling_algebra"
    if len(columns) == 2 and any(_alias_column(list(item), _CAPACITY_ALIASES) is not None
                                 and _alias_column(list(item), _RESOURCE_ALIASES) is not None
                                 for item in columns):
        return "modeling_optimization"
    if len(columns) >= 2:
        return "modeling_multi_table"
    raise AutomaticModelingError("supported_model_family_not_identified")


def _fit_univariate_expression_grammar(x: np.ndarray, y: np.ndarray, *, max_depth: int = 2,
                                       strategy: str = "exhaustive", seed: int = 0) -> dict[str, Any] | None:
    """Fit a bounded family generated from operators, including depth-two compositions."""
    from scipy.optimize import least_squares

    if type(max_depth) is not int or not 1 <= max_depth <= 3:
        raise AutomaticModelingError("expression_depth_budget_invalid")
    if strategy not in {"exhaustive", "random_equal_budget", "beam_equal_budget",
                        "beam_reverse_equal_budget", "beam_bidirectional_equal_budget"} or type(seed) is not int:
        raise AutomaticModelingError("expression_search_strategy_invalid")

    scale = max(float(np.std(y)), 1e-6)
    operators = {
        "sin": lambda z: np.sin(z), "cos": lambda z: np.cos(z),
        "tanh": lambda z: np.tanh(z),
        "exp": lambda z: np.exp(np.clip(z, -40.0, 40.0)),
    }
    candidates: list[dict[str, Any]] = []

    def attempt(signature: list[str], function, starts: Sequence[Sequence[float]], bounds) -> None:
        best = None
        for start in starts:
            try:
                fit = least_squares(lambda theta: function(theta, x) - y, start,
                                    bounds=bounds, max_nfev=1200)
                prediction = np.asarray(function(fit.x, x), dtype=float)
                if not np.isfinite(prediction).all():
                    continue
                rss = float(np.sum((prediction - y) ** 2))
                item = (rss, np.asarray(fit.x, dtype=float))
                best = item if best is None or item[0] < best[0] else best
            except (ValueError, FloatingPointError, OverflowError, RuntimeError):
                continue
        if best is not None:
            rss, parameters = best
            count = len(parameters)
            bic = len(y) * math.log(max(rss / len(y), 1e-24)) + count * math.log(len(y))
            candidates.append({"bic": bic, "rss": rss, "operator_signature": signature,
                               "parameters": parameters, "function": function,
                               "complexity": len(signature) + count})

    amplitude = max(2.0 * scale, 1.0)
    signatures = [tuple(names) for depth in range(1, max_depth + 1)
                  for names in product(operators, repeat=depth)]
    include_rational = True
    search_trace: dict[str, Any] | None = None
    if strategy == "random_equal_budget":
        choices: list[tuple[str, ...]] = [*signatures, ("divide", "affine", "affine")]
        rng = np.random.default_rng(seed)
        selected = rng.choice(len(choices), size=min(21, len(choices)), replace=False)
        chosen = [choices[int(index)] for index in selected]
        signatures = [item for item in chosen if item != ("divide", "affine", "affine")]
        include_rational = ("divide", "affine", "affine") in chosen
    def fit_signatures(items: Sequence[tuple[str, ...]]) -> None:
        for signature in items:
            def function(theta, values, names=signature):
                transformed = theta[2] * values + theta[3]
                for name in reversed(names):
                    transformed = operators[name](transformed)
                return theta[0] + theta[1] * transformed
            starts = [[float(np.mean(y)), scale, frequency, phase]
                      for frequency in (0.5, 1.0, 2.0) for phase in (0.0,)]
            attempt(list(signature), function, starts,
                    ([-10 * amplitude, -10 * amplitude, -8.0, -8.0],
                     [10 * amplitude, 10 * amplitude, 8.0, 8.0]))

    topology_budget = len(signatures) + int(include_rational)
    if strategy in {"beam_equal_budget", "beam_reverse_equal_budget",
                    "beam_bidirectional_equal_budget"}:
        if max_depth != 3:
            raise AutomaticModelingError("beam_search_requires_depth_three")
        unary = [(name,) for name in operators]
        fit_signatures(unary)
        unary_beam = [tuple(item["operator_signature"]) for item in sorted(
            candidates, key=lambda item: (item["bic"], item["operator_signature"]),
        )[:2]]
        search_trace = {"unary_beam": [list(item) for item in unary_beam]}
        if strategy == "beam_bidirectional_equal_budget":
            anchor = unary_beam[0]
            forward = [(*anchor, name) for name in operators]
            reverse = [(name, *anchor) for name in operators]
            depth_two = list(dict.fromkeys([*forward, *reverse]))
            depth_two += [item for item in product(operators, repeat=2)
                          if item not in depth_two][:8-len(depth_two)]
        elif strategy == "beam_equal_budget":
            depth_two = [(*prefix, name) for prefix in unary_beam for name in operators]
        else:
            depth_two = [(name, *suffix) for suffix in unary_beam for name in operators]
        fit_signatures(depth_two)
        fitted_depth_two = [item for item in candidates if len(item["operator_signature"]) == 2]
        if strategy == "beam_bidirectional_equal_budget":
            best_forward = tuple(min(
                (item for item in fitted_depth_two
                 if tuple(item["operator_signature"]) in forward),
                key=lambda item: (item["bic"], item["operator_signature"]),
            )["operator_signature"])
            best_reverse = tuple(min(
                (item for item in fitted_depth_two
                 if tuple(item["operator_signature"]) in reverse),
                key=lambda item: (item["bic"], item["operator_signature"]),
            )["operator_signature"])
            depth_three = [(*best_forward, name) for name in operators]
            depth_three += [(name, *best_reverse) for name in operators]
            depth_three = list(dict.fromkeys(depth_three))
            depth_three += [item for item in product(operators, repeat=3)
                            if item not in depth_three][:8-len(depth_three)]
            search_trace["depth_two_beam"] = [list(best_forward), list(best_reverse)]
        else:
            depth_two_beam = [tuple(item["operator_signature"]) for item in sorted(
                fitted_depth_two, key=lambda item: (item["bic"], item["operator_signature"]),
            )[:2]]
            search_trace["depth_two_beam"] = [list(item) for item in depth_two_beam]
        if strategy == "beam_equal_budget":
            depth_three = [(*prefix, name) for prefix in depth_two_beam for name in operators]
        elif strategy == "beam_reverse_equal_budget":
            depth_three = [(name, *suffix) for suffix in depth_two_beam for name in operators]
        fit_signatures(depth_three)
        signatures = [*unary, *depth_two, *depth_three]
        include_rational = True
        topology_budget = 21
    else:
        fit_signatures(signatures)
    rational = lambda theta, values: (theta[0] + theta[1] * values) / (1.0 + theta[2] * values)
    if include_rational:
        attempt(["divide", "affine", "affine"], rational,
                [[float(np.mean(y)), scale, value] for value in (-0.2, -0.05, 0.05, 0.2)],
                ([-10 * amplitude, -10 * amplitude, -1.0], [10 * amplitude, 10 * amplitude, 1.0]))
    if not candidates:
        return None
    best = min(candidates, key=lambda item: (item["bic"], item["complexity"], item["operator_signature"]))
    best["search_evaluations"] = len(candidates)
    best["max_depth"] = max_depth
    best["strategy"] = strategy
    best["topology_budget"] = topology_budget
    best["beam_width"] = (2 if strategy in {"beam_equal_budget", "beam_reverse_equal_budget"}
                          else 1 if strategy == "beam_bidirectional_equal_budget" else None)
    best["beam_direction"] = ({"beam_equal_budget": "outer_to_inner",
                               "beam_reverse_equal_budget": "inner_to_outer",
                               "beam_bidirectional_equal_budget": "bidirectional"}.get(strategy))
    best["search_trace"] = search_trace
    return best


def _fit_algebra(payload: Mapping[str, Any], tables: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    located = None
    for candidate_rows in tables.values():
        try:
            candidate_response = _alias_column(
                list(candidate_rows[0]), _RESPONSE_ALIASES,
                explicit=payload.get("response_column"),
            )
        except AutomaticModelingError:
            continue
        if candidate_response is not None:
            located = (candidate_rows, candidate_response)
            break
    rows, response = located if located is not None else (None, None)
    if rows is None:
        raise AutomaticModelingError("algebra_observations_missing")
    inputs = sorted(column for column in rows[0] if column != response)
    if not inputs or len(inputs) > 4:
        raise AutomaticModelingError("algebra_input_count_unsupported")
    x = np.asarray([[row[name] for name in inputs] for row in rows], dtype=float)
    y = np.asarray([row[response] for row in rows], dtype=float)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or len(y) < 5:
        raise AutomaticModelingError("algebra_observations_invalid")
    raw_queries = payload.get("query_inputs")
    queries = None if raw_queries is None else np.asarray(raw_queries, dtype=float)
    if len(inputs) == 1:
        candidates = []
        for degree, structure in ((1, "affine"), (2, "quadratic"), (3, "cubic")):
            design = np.vander(x[:, 0], degree + 1)
            coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
            residual = float(np.mean((design @ coefficients - y) ** 2))
            bic = len(y) * math.log(max(residual, 1e-24)) + len(coefficients) * math.log(len(y))
            candidates.append((bic, structure, coefficients))
        polynomial_bic, structure, coefficients = min(candidates, key=lambda item: item[0])
        polynomial_residual = float(np.mean((np.polyval(coefficients, x[:, 0]) - y) ** 2))
        exact_scale = max(float(np.var(y)), 1.0)
        # A numerically exact low-degree polynomial cannot be improved in
        # predictive fit by the more expensive nonlinear grammar. Avoiding
        # those 21 nonlinear optimizations is a search dominance shortcut,
        # not a relaxed acceptance threshold.
        grammar = None if polynomial_residual <= exact_scale * 1e-20 else \
            _fit_univariate_expression_grammar(
                x[:, 0], y, max_depth=int(payload.get("expression_max_depth", 2)),
                strategy=str(payload.get("expression_search_strategy", "exhaustive")),
                seed=int(payload.get("expression_search_seed", 0)),
            )
        if grammar is not None and float(grammar["bic"]) + 2.0 < polynomial_bic:
            model = {"family": "modeling_algebra", "structure": "compositional_symbolic",
                     "input_variables": inputs, "response_variable": response,
                     "operator_signature": grammar["operator_signature"],
                     "parameters": grammar["parameters"].tolist(),
                     "complexity": int(grammar["complexity"]),
                     "search_strategy": ("deterministic_bounded_enumeration"
                                         if grammar["strategy"] == "exhaustive"
                                         else grammar["strategy"]),
                     "search_evaluations": int(grammar["search_evaluations"]),
                     "topology_budget": int(grammar["topology_budget"]),
                     "beam_width": grammar.get("beam_width"),
                     "beam_direction": grammar.get("beam_direction"),
                     "search_trace": grammar.get("search_trace"),
                     "expression_max_depth": int(grammar["max_depth"]),
                     "selection": "bounded_operator_grammar_bic"}
            if queries is None:
                return {"status": "needs_input", "model": model, "missing": ["query_inputs"],
                        "reason": "model_identified_but_prediction_query_missing"}
            queries = queries.reshape(-1, 1)
            predictions = grammar["function"](grammar["parameters"], queries[:, 0])
            return {"model": model, "predictions": np.asarray(predictions, dtype=float).tolist()}
        model = {"family": "modeling_algebra", "structure": structure,
                 "input_variables": inputs, "coefficients": coefficients.tolist(),
                 "response_variable": response,
                 "selection": "bic_over_affine_quadratic_cubic"}
        if queries is None:
            return {"status": "needs_input", "model": model, "missing": ["query_inputs"],
                    "reason": "model_identified_but_prediction_query_missing"}
        queries = queries.reshape(-1, 1)
        predictions = np.polyval(coefficients, queries[:, 0])
        return {"model": model,
                "predictions": predictions.tolist()}
    if queries is None:
        query_missing = True
        queries = np.empty((0, len(inputs)), dtype=float)
    else:
        query_missing = False
    if queries.ndim != 2 or queries.shape[1] != len(inputs):
        raise AutomaticModelingError("algebra_query_shape_invalid")
    # Main effects plus bounded pairwise and three-way interactions.  A sparse
    # exact least-squares fit identifies active terms without using field names.
    terms = [("intercept", None), *[("linear", i) for i in range(len(inputs))],
             *[("interaction", pair) for pair in combinations(range(len(inputs)), 2)],
             *[("three_way_interaction", triple) for triple in combinations(range(len(inputs)), 3)]]
    def design(values: np.ndarray) -> np.ndarray:
        columns = [np.ones(len(values))]
        columns.extend(values[:, i] for i in range(values.shape[1]))
        columns.extend(np.prod(values[:, indices], axis=1)
                       for degree in (2, 3) for indices in combinations(range(values.shape[1]), degree))
        return np.column_stack(columns)
    matrix = design(x)
    coefficients, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    polynomial_prediction = matrix @ coefficients
    polynomial_rss = float(np.sum((polynomial_prediction - y) ** 2))
    polynomial_bic = len(y) * math.log(max(polynomial_rss / len(y), 1e-24)) \
        + len(coefficients) * math.log(len(y))
    # A signed absolute power law covers scale-free multivariate ratios such as
    # c*x0*x1**-1 and sqrt(x0/x1).  It is considered only when the bounded
    # polynomial library is not already numerically exact.
    power_candidate = None
    rational_candidate = None
    root_form_candidate = None
    angular_projection_candidate = None
    product_angle_candidate = None
    y_scale = max(float(np.std(y)), float(np.mean(np.abs(y))), 1e-12)
    enable_multivariate_nonlinear = payload.get("enable_multivariate_nonlinear", True)
    if type(enable_multivariate_nonlinear) is not bool:
        raise AutomaticModelingError("multivariate_nonlinear_flag_invalid")
    enable_trigonometric_rational = payload.get("enable_trigonometric_rational_features", True)
    if type(enable_trigonometric_rational) is not bool:
        raise AutomaticModelingError("trigonometric_rational_flag_invalid")
    enable_extended_composition = payload.get("enable_extended_multivariate_composition", True)
    if type(enable_extended_composition) is not bool:
        raise AutomaticModelingError("extended_multivariate_composition_flag_invalid")
    enable_transformed_power_laurent = payload.get("enable_transformed_power_laurent", True)
    if type(enable_transformed_power_laurent) is not bool:
        raise AutomaticModelingError("transformed_power_laurent_flag_invalid")
    enable_harmonic_trigonometric = payload.get("enable_harmonic_trigonometric_features", True)
    if type(enable_harmonic_trigonometric) is not bool:
        raise AutomaticModelingError("harmonic_trigonometric_flag_invalid")
    enable_angular_projection = payload.get("enable_angular_projection_composition", True)
    if type(enable_angular_projection) is not bool:
        raise AutomaticModelingError("angular_projection_flag_invalid")
    enable_product_angle_rational = payload.get("enable_product_angle_rational_composition", True)
    if type(enable_product_angle_rational) is not bool:
        raise AutomaticModelingError("product_angle_rational_flag_invalid")
    if (enable_multivariate_nonlinear
            and math.sqrt(polynomial_rss / len(y)) / y_scale > 1e-8
            and np.all(np.abs(x) > 1e-12) and np.all(np.abs(y) > 1e-12)
            and (np.all(y > 0) or np.all(y < 0))):
        power_feature_grammar = ["raw"] * len(inputs)
        power_columns = [x[:, index] for index in range(x.shape[1])]
        if enable_extended_composition and enable_transformed_power_laurent:
            power_columns.extend(np.sin(x[:, index]) for index in range(x.shape[1]))
            power_columns.extend(np.cos(x[:, index]) for index in range(x.shape[1]))
            power_feature_grammar.extend(["sin"] * len(inputs))
            power_feature_grammar.extend(["cos"] * len(inputs))
        if any(np.any(np.abs(column) <= 1e-12) for column in power_columns):
            power_columns = []
        log_design = (np.column_stack([np.ones(len(x)),
                                      *[np.log(np.abs(column)) for column in power_columns]])
                      if power_columns else None)
        if log_design is not None:
            log_parameters, *_ = np.linalg.lstsq(log_design, np.log(np.abs(y)), rcond=None)
            sign = 1.0 if np.all(y > 0) else -1.0
            power_prediction = sign * np.exp(np.clip(log_design @ log_parameters, -700.0, 700.0))
            power_rss = float(np.sum((power_prediction - y) ** 2))
            power_bic = len(y) * math.log(max(power_rss / len(y), 1e-24)) \
                + len(log_parameters) * math.log(len(y))
            if np.isfinite(power_prediction).all() and power_bic + 2.0 < polynomial_bic:
                power_candidate = (sign, log_parameters, power_bic, power_feature_grammar)
    if (enable_multivariate_nonlinear and enable_extended_composition
            and math.sqrt(polynomial_rss / len(y)) / y_scale > 1e-8
            and np.all(np.abs(y) > 1e-12)
            and (np.all(y > 0) or np.all(y < 0))):
        sign = 1.0 if np.all(y > 0) else -1.0
        root_feature_specs: list[tuple[Any, ...]] = [("constant",)]
        root_feature_specs.extend(("raw_square", index) for index in range(x.shape[1]))
        root_feature_specs.extend(("ratio_square", left, right)
                                  for left in range(x.shape[1]) for right in range(x.shape[1])
                                  if left != right and np.all(np.abs(x[:, right]) > 1e-12))
        if enable_transformed_power_laurent:
            root_feature_specs.extend(("laurent", numerator, denominator, squared_denominator)
                                      for numerator in range(x.shape[1])
                                      for denominator in range(x.shape[1])
                                      for squared_denominator in range(x.shape[1])
                                      if np.all(np.abs(x[:, denominator]) > 1e-12)
                                      and np.all(np.abs(x[:, squared_denominator]) > 1e-12))
        def root_design(values: np.ndarray) -> np.ndarray:
            columns = []
            for spec in root_feature_specs:
                if spec[0] == "constant":
                    columns.append(np.ones(len(values)))
                elif spec[0] == "raw_square":
                    columns.append(values[:, spec[1]] ** 2)
                elif spec[0] == "ratio_square":
                    columns.append((values[:, spec[1]] / values[:, spec[2]]) ** 2)
                else:
                    columns.append(values[:, spec[1]] /
                                   (values[:, spec[2]] * values[:, spec[3]] ** 2))
            return np.column_stack(columns)
        square_design = root_design(x)
        for mode, target in (("sqrt_quadratic_form", y ** 2),
                             ("inverse_sqrt_quadratic_form", 1.0 / (y ** 2))):
            root_parameters, *_ = np.linalg.lstsq(square_design, target, rcond=None)
            radicand = square_design @ root_parameters
            if np.any(radicand <= 1e-12):
                continue
            root_prediction = (sign * np.sqrt(radicand) if mode == "sqrt_quadratic_form"
                               else sign / np.sqrt(radicand))
            root_rss = float(np.sum((root_prediction - y) ** 2))
            root_bic = len(y) * math.log(max(root_rss / len(y), 1e-24)) \
                + len(root_parameters) * math.log(len(y))
            item = (mode, root_parameters, root_bic)
            if root_bic + 2.0 < polynomial_bic and (
                    root_form_candidate is None or root_bic < root_form_candidate[2]):
                root_form_candidate = item
    if (enable_multivariate_nonlinear and enable_extended_composition
            and enable_angular_projection and x.shape[1] >= 4
            and math.sqrt(polynomial_rss / len(y)) / y_scale > 1e-8):
        for projected in range(x.shape[1]):
            for root_length in range(x.shape[1]):
                if root_length == projected:
                    continue
                remaining = [index for index in range(x.shape[1])
                             if index not in {projected, root_length}]
                for angle_left, angle_right in combinations(remaining, 2):
                    angle = x[:, angle_left] - x[:, angle_right]
                    radicand = (x[:, root_length] ** 2
                                - x[:, projected] ** 2 * np.sin(angle) ** 2)
                    if np.any(radicand <= 1e-12):
                        continue
                    angular_design = np.column_stack([
                        np.ones(len(x)),
                        x[:, projected] * np.cos(angle),
                        np.sqrt(radicand),
                    ])
                    angular_parameters, *_ = np.linalg.lstsq(angular_design, y, rcond=None)
                    angular_prediction = angular_design @ angular_parameters
                    angular_rss = float(np.sum((angular_prediction - y) ** 2))
                    angular_bic = len(y) * math.log(max(angular_rss / len(y), 1e-24)) \
                        + len(angular_parameters) * math.log(len(y))
                    item = (angular_parameters,
                            (projected, root_length, angle_left, angle_right), angular_bic)
                    if angular_bic + 2.0 < polynomial_bic and (
                            angular_projection_candidate is None
                            or angular_bic < angular_projection_candidate[2]):
                        angular_projection_candidate = item
    if enable_multivariate_nonlinear and math.sqrt(polynomial_rss / len(y)) / y_scale > 1e-8:
        centers = np.mean(x, axis=0)
        scales = np.std(x, axis=0)
        scales[scales < 1e-12] = 1.0
        standardized = (x - centers) / scales
        denominator_pairs = list(combinations(range(len(inputs)), 2))
        denominator_triples = (list(combinations(range(len(inputs)), 3))
                               if enable_extended_composition else [])
        numerator_combinations = ([*denominator_pairs, *denominator_triples]
                                  if enable_extended_composition else [])
        powered_numerator_triples = (
            [(left, right, powered, degree)
             for powered in range(len(inputs))
             for left, right in combinations(
                 [index for index in range(len(inputs)) if index != powered], 2)
             for degree in (2, 3)]
            if enable_extended_composition and enable_harmonic_trigonometric else []
        )
        powered_numerator_pairs = (
            [(other, powered, degree)
             for powered in range(len(inputs))
             for other in range(len(inputs)) if other != powered
             for degree in (2, 3)]
            if enable_extended_composition and enable_harmonic_trigonometric else []
        )
        def rational_design(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            raw_values = values * scales + centers
            pair_cosines = [np.cos(raw_values[:, left] * raw_values[:, right])
                            for left, right in denominator_pairs]
            half_angle_squares = [np.sin(0.5 * raw_values[:, index]) ** 2
                                  for index in range(values.shape[1])]
            product_angle_numerator = []
            if (enable_trigonometric_rational and enable_extended_composition
                    and enable_product_angle_rational):
                product_angle_numerator.extend(half_angle_squares)
                product_angle_numerator.extend(
                    raw_values[:, raw_index] * half_angle_squares[angle_index]
                    for raw_index in range(values.shape[1])
                    for angle_index in range(values.shape[1]))
                product_angle_numerator.extend(pair_cosines)
                product_angle_numerator.extend(
                    raw_values[:, raw_index] * pair_cosine
                    for pair_cosine in pair_cosines
                    for raw_index in range(values.shape[1]))
            numerator = np.column_stack([
                np.ones(len(values)),
                *[values[:, index] for index in range(values.shape[1])],
                *[np.prod(values[:, indices], axis=1) for indices in numerator_combinations],
                *([values[:, index] ** degree
                   for index in range(values.shape[1]) for degree in (2, 3)]
                  if powered_numerator_triples else []),
                *[values[:, other] * values[:, powered] ** degree
                  for other, powered, degree in powered_numerator_pairs],
                *[values[:, left] * values[:, right] * values[:, powered] ** degree
                  for left, right, powered, degree in powered_numerator_triples],
                *product_angle_numerator,
            ])
            cosine = np.cos(raw_values)
            sine = np.sin(raw_values)
            denominator_columns = [values[:, index] for index in range(values.shape[1])]
            denominator_columns.extend(values[:, left] * values[:, right]
                                       for left, right in denominator_pairs)
            denominator_columns.extend(np.prod(values[:, indices], axis=1)
                                       for indices in denominator_triples)
            if enable_trigonometric_rational:
                denominator_columns.extend(cosine[:, index] for index in range(values.shape[1]))
                denominator_columns.extend(sine[:, index] for index in range(values.shape[1]))
                if enable_harmonic_trigonometric:
                    denominator_columns.extend(np.cos(harmonic * raw_values[:, index])
                                               for harmonic in (2, 3)
                                               for index in range(values.shape[1]))
                    denominator_columns.extend(np.sin(harmonic * raw_values[:, index])
                                               for harmonic in (2, 3)
                                               for index in range(values.shape[1]))
                denominator_columns.extend(values[:, left] * cosine[:, right]
                                           for left in range(values.shape[1])
                                           for right in range(values.shape[1]))
                if enable_extended_composition:
                    denominator_columns.extend(pair_cosines)
                    denominator_columns.extend(np.sin(raw_values[:, left] * raw_values[:, right])
                                               for left, right in denominator_pairs)
                    if enable_product_angle_rational:
                        denominator_columns.extend(pair_cosine ** 2
                                                   for pair_cosine in pair_cosines)
                        denominator_columns.extend(
                            raw_values[:, raw_index] * pair_cosine
                            for pair_cosine in pair_cosines
                            for raw_index in range(values.shape[1]))
                        denominator_columns.extend(
                            raw_values[:, raw_index] * pair_cosine ** 2
                            for pair_cosine in pair_cosines
                            for raw_index in range(values.shape[1]))
            denominator = np.column_stack(denominator_columns)
            return numerator, denominator
        numerator_design, denominator_design = rational_design(standardized)
        cross_multiplication = np.column_stack([numerator_design, -y[:, None] * denominator_design])
        linearized_parameters, *_ = np.linalg.lstsq(cross_multiplication, y, rcond=None)
        parameter_count = numerator_design.shape[1] + denominator_design.shape[1]
        def rational_prediction(parameters: np.ndarray, values: np.ndarray) -> np.ndarray:
            numerator, denominator_terms = rational_design(values)
            numerator_parameters = parameters[:numerator.shape[1]]
            denominator_parameters = parameters[numerator.shape[1]:]
            denominator = np.column_stack([np.ones(len(values)), denominator_terms]) @ denominator_parameters
            return (numerator @ numerator_parameters) / np.where(
                np.abs(denominator) < 1e-9, np.nan, denominator,
            )
        fixed_intercept_parameters = np.concatenate([
            linearized_parameters[:numerator_design.shape[1]],
            [1.0],
            linearized_parameters[numerator_design.shape[1]:],
        ])
        homogeneous_design = np.column_stack([
            numerator_design, -y[:, None], -y[:, None] * denominator_design,
        ])
        _, _, right_vectors = np.linalg.svd(homogeneous_design, full_matrices=False)
        homogeneous_parameters = right_vectors[-1]
        best_rational = None
        for parameters in (fixed_intercept_parameters, homogeneous_parameters):
            prediction = rational_prediction(parameters, standardized)
            if not np.isfinite(prediction).all():
                continue
            rss = float(np.sum((prediction - y) ** 2))
            if best_rational is None or rss < best_rational[0]:
                best_rational = (rss, parameters)
        # Also enumerate which single grammar atom occupies the denominator.
        # This admits zero-intercept denominators such as sin(2*x), which the
        # conventional ``1 + D b`` normalization cannot represent.
        for denominator_index in range(denominator_design.shape[1]):
            denominator_column = denominator_design[:, denominator_index]
            numerator_parameters, *_ = np.linalg.lstsq(
                numerator_design, y * denominator_column, rcond=None,
            )
            denominator_parameters = np.zeros(denominator_design.shape[1] + 1)
            denominator_parameters[denominator_index + 1] = 1.0
            parameters = np.concatenate([numerator_parameters, denominator_parameters])
            prediction = rational_prediction(parameters, standardized)
            if not np.isfinite(prediction).all():
                continue
            rss = float(np.sum((prediction - y) ** 2))
            if best_rational is None or rss < best_rational[0]:
                best_rational = (rss, parameters)
        if best_rational is not None:
            rational_rss, rational_parameters = best_rational
            rational_bic = len(y) * math.log(max(rational_rss / len(y), 1e-24)) \
                + parameter_count * math.log(len(y))
            if rational_bic + 2.0 < polynomial_bic:
                rational_candidate = (rational_parameters, centers, scales, rational_bic,
                                      rational_prediction)
    if (enable_multivariate_nonlinear and enable_extended_composition
            and enable_product_angle_rational
            and math.sqrt(polynomial_rss / len(y)) / y_scale > 1e-8):
        def consider_product_angle(kind: str, parameters: np.ndarray,
                                   indices: tuple[int, ...], prediction: np.ndarray) -> None:
            nonlocal product_angle_candidate
            if not np.isfinite(prediction).all():
                return
            rss = float(np.sum((prediction - y) ** 2))
            bic = len(y) * math.log(max(rss / len(y), 1e-24)) \
                + len(parameters) * math.log(len(y))
            item = (kind, parameters, indices, bic)
            if bic + 2.0 < polynomial_bic and (
                    product_angle_candidate is None or bic < product_angle_candidate[3]):
                product_angle_candidate = item

        for angle_left, angle_right in combinations(range(x.shape[1]), 2):
            cosine = np.cos(x[:, angle_left] * x[:, angle_right])
            cosine_square = cosine ** 2
            for numerator_index in range(x.shape[1]):
                for scale_index in range(x.shape[1]):
                    denominator = x[:, scale_index] * cosine_square
                    if np.all(np.abs(denominator) > 1e-9):
                        local_design = np.column_stack([
                            x[:, numerator_index], x[:, scale_index] * cosine,
                        ])
                        parameters, *_ = np.linalg.lstsq(local_design, y * denominator, rcond=None)
                        consider_product_angle(
                            "quotient_difference", parameters,
                            (angle_left, angle_right, numerator_index, scale_index),
                            (local_design @ parameters) / denominator,
                        )
                    local_design = np.column_stack([
                        x[:, numerator_index], -y * x[:, scale_index] * cosine_square,
                    ])
                    parameters, *_ = np.linalg.lstsq(local_design, y * cosine, rcond=None)
                    denominator = cosine + parameters[1] * x[:, scale_index] * cosine_square
                    if np.all(np.abs(denominator) > 1e-9):
                        consider_product_angle(
                            "affine_cosine_square_denominator", parameters,
                            (angle_left, angle_right, numerator_index, scale_index),
                            parameters[0] * x[:, numerator_index] / denominator,
                        )
        for half_angle_index in range(x.shape[1]):
            half_square = np.sin(0.5 * x[:, half_angle_index]) ** 2
            for multiplier_index in range(x.shape[1]):
                for product_index in range(x.shape[1]):
                    if product_index == half_angle_index:
                        continue
                    cosine = np.cos(x[:, half_angle_index] * x[:, product_index])
                    numerator_atom = x[:, multiplier_index] * half_square
                    local_design = np.column_stack([numerator_atom, -y * cosine])
                    parameters, *_ = np.linalg.lstsq(local_design, y, rcond=None)
                    denominator = 1.0 + parameters[1] * cosine
                    if np.all(np.abs(denominator) > 1e-9):
                        consider_product_angle(
                            "half_angle_over_product_angle", parameters,
                            (half_angle_index, multiplier_index, product_index),
                            parameters[0] * numerator_atom / denominator,
                        )

    def product_angle_prediction(candidate: tuple[Any, ...], values: np.ndarray) -> np.ndarray:
        kind, parameters, indices, _bic = candidate
        if kind == "half_angle_over_product_angle":
            half_angle_index, multiplier_index, product_index = indices
            cosine = np.cos(values[:, half_angle_index] * values[:, product_index])
            numerator_atom = (values[:, multiplier_index]
                              * np.sin(0.5 * values[:, half_angle_index]) ** 2)
            denominator = 1.0 + parameters[1] * cosine
            return parameters[0] * numerator_atom / np.where(
                np.abs(denominator) < 1e-9, np.nan, denominator)
        angle_left, angle_right, numerator_index, scale_index = indices
        cosine = np.cos(values[:, angle_left] * values[:, angle_right])
        if kind == "quotient_difference":
            denominator = values[:, scale_index] * cosine ** 2
            numerator = (parameters[0] * values[:, numerator_index]
                         + parameters[1] * values[:, scale_index] * cosine)
        else:
            denominator = cosine + parameters[1] * values[:, scale_index] * cosine ** 2
            numerator = parameters[0] * values[:, numerator_index]
        return numerator / np.where(np.abs(denominator) < 1e-9, np.nan, denominator)

    product_angle_exact = False
    if product_angle_candidate is not None:
        product_training_prediction = product_angle_prediction(product_angle_candidate, x)
        if np.isfinite(product_training_prediction).all():
            relative_training_error = np.abs(product_training_prediction - y) / np.maximum(
                np.abs(y), y_scale * 1e-10,
            )
            product_angle_exact = bool(np.max(relative_training_error) <= 1e-3)
    if product_angle_candidate is not None and (product_angle_exact or (
            (rational_candidate is None or product_angle_candidate[3] < rational_candidate[3]) and
            (power_candidate is None or product_angle_candidate[3] < power_candidate[2]) and
            (root_form_candidate is None or product_angle_candidate[3] < root_form_candidate[2]) and
            (angular_projection_candidate is None
             or product_angle_candidate[3] < angular_projection_candidate[2]))):
        kind, parameters, indices, _bic = product_angle_candidate
        model = {"family": "modeling_algebra", "structure": "product_angle_rational",
                 "substructure": kind, "input_variables": inputs,
                 "response_variable": response, "parameters": parameters.tolist(),
                 "position_indices": list(indices),
                 "feature_grammar": "bounded_product_angle_half_angle_unknown_position",
                 "training_relative_error_max": float(np.max(relative_training_error)),
                 "selection": "bounded_position_enumeration_bic"}
        if query_missing:
            return {"status": "needs_input", "model": model, "missing": ["query_inputs"],
                    "reason": "model_identified_but_prediction_query_missing"}
        predictions = product_angle_prediction(product_angle_candidate, queries)
        if not np.isfinite(predictions).all():
            raise AutomaticModelingError("product_angle_query_singularity")
        return {"model": model, "predictions": predictions.tolist()}
    if rational_candidate is not None and (power_candidate is None
            or rational_candidate[3] < power_candidate[2]) and (root_form_candidate is None
            or rational_candidate[3] < root_form_candidate[2]) and (
                angular_projection_candidate is None
                or rational_candidate[3] < angular_projection_candidate[2]):
        parameters, centers, scales, _rational_bic, rational_prediction = rational_candidate
        model = {"family": "modeling_algebra", "structure": "multivariate_rational",
                 "input_variables": inputs, "response_variable": response,
                 "parameters": parameters.tolist(), "centers": centers.tolist(),
                 "scales": scales.tolist(), "denominator_pairs": [list(pair) for pair in denominator_pairs],
                 "rational_feature_configuration": {
                     "extended_composition": bool(enable_extended_composition),
                     "harmonic_trigonometric": bool(enable_harmonic_trigonometric),
                     "trigonometric_rational": bool(enable_trigonometric_rational),
                     "product_angle_rational": bool(enable_product_angle_rational),
                 },
                 "numerator_feature_grammar": (
                     ("raw;pairwise;three_way;unary_pairwise_three_way_power_2_3;"
                      "half_angle_square;raw_times_half_angle_square;"
                      "product_angle_cos;raw_times_product_angle_cos")
                     if powered_numerator_triples and enable_product_angle_rational else
                     "raw;pairwise;three_way;unary_pairwise_three_way_power_2_3"
                     if powered_numerator_triples else
                     "raw;pairwise;three_way" if enable_extended_composition else "raw"),
                 "denominator_feature_grammar": (
                     (("raw;pairwise;three_way;sin;cos;integer_harmonic_2_3;raw_times_cos;"
                       "sin_pair_argument;cos_pair_argument;product_angle_square;"
                       "raw_times_product_angle;raw_times_product_angle_square")
                      if enable_extended_composition and enable_harmonic_trigonometric
                      and enable_product_angle_rational else
                      ("raw;pairwise;three_way;sin;cos;integer_harmonic_2_3;raw_times_cos;"
                       "sin_pair_argument;cos_pair_argument")
                      if enable_extended_composition and enable_harmonic_trigonometric else
                      "raw;pairwise;three_way;sin;cos;raw_times_cos;sin_pair_argument;cos_pair_argument"
                      if enable_extended_composition else
                      "raw;pairwise;sin;cos;integer_harmonic_2_3;raw_times_cos"
                      if enable_harmonic_trigonometric else
                      "raw;pairwise;sin;cos;raw_times_cos")
                     if enable_trigonometric_rational else (
                         "raw;pairwise;three_way" if enable_extended_composition else "raw;pairwise")),
                 "selection": "bounded_polynomial_power_law_rational_bic"}
        if query_missing:
            return {"status": "needs_input", "model": model, "missing": ["query_inputs"],
                    "reason": "model_identified_but_prediction_query_missing"}
        predictions = rational_prediction(parameters, (queries - centers) / scales)
        if not np.isfinite(predictions).all():
            raise AutomaticModelingError("rational_query_singularity")
        return {"model": model, "predictions": predictions.tolist()}
    if root_form_candidate is not None and (power_candidate is None
            or root_form_candidate[2] < power_candidate[2]) and (
                angular_projection_candidate is None
                or root_form_candidate[2] < angular_projection_candidate[2]):
        mode, root_parameters, _root_bic = root_form_candidate
        model = {"family": "modeling_algebra", "structure": mode,
                 "input_variables": inputs, "response_variable": response,
                 "parameters": root_parameters.tolist(),
                 "feature_grammar": "constant;raw_square;ordered_ratio_square;bounded_laurent",
                 "feature_count": len(root_feature_specs),
                 "feature_specs": [list(spec) for spec in root_feature_specs],
                 "output_sign": 1.0 if np.all(y > 0) else -1.0,
                 "selection": "bounded_polynomial_power_law_rational_root_form_bic"}
        if query_missing:
            return {"status": "needs_input", "model": model, "missing": ["query_inputs"],
                    "reason": "model_identified_but_prediction_query_missing"}
        radicand = root_design(queries) @ root_parameters
        if np.any(radicand <= 1e-12):
            raise AutomaticModelingError("root_form_query_domain_error")
        sign = 1.0 if np.all(y > 0) else -1.0
        predictions = (sign * np.sqrt(radicand) if mode == "sqrt_quadratic_form"
                       else sign / np.sqrt(radicand))
        return {"model": model, "predictions": predictions.tolist()}
    if power_candidate is not None and (angular_projection_candidate is None
            or power_candidate[2] < angular_projection_candidate[2]):
        sign, log_parameters, _power_bic, power_feature_grammar = power_candidate
        model = {"family": "modeling_algebra", "structure": "signed_absolute_power_law",
                 "input_variables": inputs, "response_variable": response,
                 "amplitude": float(sign * math.exp(float(log_parameters[0]))),
                 "exponents": log_parameters[1:].tolist(),
                 "power_feature_grammar": power_feature_grammar,
                 "selection": "bounded_polynomial_power_law_rational_bic"}
        if query_missing:
            return {"status": "needs_input", "model": model, "missing": ["query_inputs"],
                    "reason": "model_identified_but_prediction_query_missing"}
        if np.any(np.abs(queries) <= 1e-12):
            raise AutomaticModelingError("power_law_query_at_zero_unsupported")
        query_power_columns = [queries[:, index] for index in range(queries.shape[1])]
        if enable_extended_composition and enable_transformed_power_laurent:
            query_power_columns.extend(np.sin(queries[:, index])
                                       for index in range(queries.shape[1]))
            query_power_columns.extend(np.cos(queries[:, index])
                                       for index in range(queries.shape[1]))
        if any(np.any(np.abs(column) <= 1e-12) for column in query_power_columns):
            raise AutomaticModelingError("power_law_query_at_zero_unsupported")
        query_design = np.column_stack([
            np.ones(len(queries)), *[np.log(np.abs(column)) for column in query_power_columns],
        ])
        predictions = sign * np.exp(np.clip(query_design @ log_parameters, -700.0, 700.0))
        return {"model": model, "predictions": predictions.tolist()}
    if angular_projection_candidate is not None:
        angular_parameters, indices, _angular_bic = angular_projection_candidate
        projected, root_length, angle_left, angle_right = indices
        model = {"family": "modeling_algebra", "structure": "angular_projection_root",
                 "input_variables": inputs, "response_variable": response,
                 "parameters": angular_parameters.tolist(),
                 "projected_index": projected, "root_length_index": root_length,
                 "angle_indices": [angle_left, angle_right],
                 "feature_grammar": "affine_of_projected_cos_difference_and_geometric_root",
                 "selection": "bounded_position_enumeration_bic"}
        if query_missing:
            return {"status": "needs_input", "model": model, "missing": ["query_inputs"],
                    "reason": "model_identified_but_prediction_query_missing"}
        query_angle = queries[:, angle_left] - queries[:, angle_right]
        query_radicand = (queries[:, root_length] ** 2
                           - queries[:, projected] ** 2 * np.sin(query_angle) ** 2)
        if np.any(query_radicand <= 1e-12):
            raise AutomaticModelingError("angular_projection_query_domain_error")
        query_design = np.column_stack([
            np.ones(len(queries)),
            queries[:, projected] * np.cos(query_angle),
            np.sqrt(query_radicand),
        ])
        return {"model": model,
                "predictions": (query_design @ angular_parameters).tolist()}
    active_interactions = [terms[index][1] for index, value in enumerate(coefficients)
                           if terms[index][0] == "interaction" and abs(float(value)) > 1e-9]
    active_three_way = [terms[index][1] for index, value in enumerate(coefficients)
                        if terms[index][0] == "three_way_interaction" and abs(float(value)) > 1e-9]
    structure = ("trilinear" if active_three_way else
                 "bilinear" if active_interactions else "multivariate_affine")
    model = {"family": "modeling_algebra", "structure": structure,
             "input_variables": inputs, "coefficients": coefficients.tolist(),
             "response_variable": response,
             "terms": [{"kind": kind, "indices": None if indices is None else (
                 list(indices) if isinstance(indices, tuple) else [indices]
             )} for kind, indices in terms],
             "selection": "bounded_main_effect_pairwise_three_way_library"}
    if query_missing:
        return {"status": "needs_input", "model": model, "missing": ["query_inputs"],
                "reason": "model_identified_but_prediction_query_missing"}
    return {"model": model,
            "predictions": (design(queries) @ coefficients).tolist()}


def _fit_coupled_ode(payload: Mapping[str, Any], rows: list[dict[str, Any]], *,
                     time_column: str, state_names: Sequence[str]) -> dict[str, Any]:
    from scipy.linalg import expm, logm

    state_names = sorted(str(name) for name in state_names)
    if not 2 <= len(state_names) <= 4:
        raise AutomaticModelingError("coupled_ode_state_count_unsupported")
    times = np.asarray([row[time_column] for row in rows], dtype=float)
    states = np.asarray([[row[name] for name in state_names] for row in rows], dtype=float)
    order = np.argsort(times, kind="stable")
    times, states = times[order], states[order]
    steps = np.diff(times)
    if (len(times) < 6 or not np.isfinite(times).all() or not np.isfinite(states).all()
            or np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-5, atol=1e-8)):
        raise AutomaticModelingError("coupled_ode_observations_invalid")
    transition, *_ = np.linalg.lstsq(states[:-1], states[1:], rcond=None)
    matrix = np.real_if_close(logm(transition).T / steps[0], tol=1000)
    if np.iscomplexobj(matrix) or not np.isfinite(matrix).all():
        raise AutomaticModelingError("coupled_ode_candidate_fit_failed")
    queries = payload.get("query_times")
    decay = -float(np.trace(matrix) / len(state_names))
    parameters: dict[str, Any] = {"matrix": np.asarray(matrix, dtype=float).tolist(), "decay": decay}
    if len(state_names) == 2:
        parameters["coupling"] = float((matrix[1, 0] - matrix[0, 1]) / 2.0)
    model = {"family": "modeling_ode", "structure": "coupled_linear",
             "state_variables": state_names, "parameters": parameters,
             "time_variable": time_column,
             "selection": "uniform_step_transition_matrix_logarithm"}
    if queries is None:
        return {"status": "needs_input", "model": model, "missing": ["query_times"],
                "reason": "model_identified_but_prediction_query_missing"}
    origin, initial = float(times[0]), states[0]
    trajectory = [(expm(matrix * (float(query) - origin)) @ initial).tolist() for query in queries]
    return {"model": model, "trajectory": trajectory}


def _fit_delay_differential_candidate(times: np.ndarray, states: np.ndarray,
                                      queries: np.ndarray | None) -> dict[str, Any] | None:
    """Search sparse delayed polynomial derivative terms on a uniform grid."""
    steps = np.diff(times)
    if len(times) < 18 or not np.allclose(steps, steps[0], rtol=1e-6, atol=1e-9):
        return None
    dt = float(steps[0])
    best_delay = None
    best_nondelay = None
    for lag in range(2, min(8, len(states) // 4) + 1):
        indices = np.arange(lag, len(states) - 1)
        target = (states[indices + 1] - states[indices]) / dt
        current, delayed = states[indices], states[indices - lag]
        features = np.column_stack([np.ones(len(indices)), current, delayed,
                                    current * current, current * delayed])
        for size in (1, 2, 3):
            for selected in combinations(range(1, 5), size):
                matrix = features[:, (0, *selected)]
                coefficients, *_ = np.linalg.lstsq(matrix, target, rcond=None)
                rss = float(np.sum((matrix @ coefficients - target) ** 2))
                bic = len(target) * math.log(max(rss / len(target), 1e-24)) + len(coefficients) * math.log(len(target))
                item = {"bic": bic, "lag_steps": lag, "terms": list(selected),
                        "coefficients": coefficients.tolist()}
                if any(term in {2, 4} for term in selected):
                    best_delay = item if best_delay is None or bic < best_delay["bic"] else best_delay
                else:
                    best_nondelay = item if best_nondelay is None or bic < best_nondelay["bic"] else best_nondelay
    if best_delay is None or best_nondelay is None or best_delay["bic"] + 8.0 >= best_nondelay["bic"]:
        return None
    model = {"family": "modeling_ode", "structure": "delay_differential_polynomial",
             "lag_steps": best_delay["lag_steps"], "lag_time": best_delay["lag_steps"] * dt,
             "term_ids": best_delay["terms"], "coefficients": best_delay["coefficients"],
             "selection": "sparse_lag_feature_grammar_bic",
             "search_strategy": "deterministic_bounded_enumeration"}
    if queries is None:
        return {"status": "needs_input", "model": model, "missing": ["query_times"],
                "reason": "model_identified_but_prediction_query_missing"}
    offsets = np.rint((queries - times[-1]) / dt).astype(int)
    if np.any(offsets < 1) or not np.allclose(times[-1] + offsets * dt, queries, atol=1e-7):
        return {"status": "needs_input", "model": model, "missing": ["uniform_future_query_times"],
                "reason": "delay_model_requires_grid_aligned_future_queries"}
    history = states.tolist(); lag = int(best_delay["lag_steps"]); terms = best_delay["terms"]
    coefficients = np.asarray(best_delay["coefficients"], dtype=float)
    predictions = []
    for step_index in range(1, int(np.max(offsets)) + 1):
        current, delayed = history[-1], history[-lag]
        available = [1.0, current, delayed, current * current, current * delayed]
        derivative = float(coefficients @ np.asarray([available[0], *[available[i] for i in terms]]))
        history.append(current + dt * derivative)
        if step_index in set(offsets.tolist()):
            predictions.append(history[-1])
    lookup = {step: value for step, value in zip(sorted(set(offsets.tolist())), predictions)}
    return {"model": model, "trajectory": [lookup[int(step)] for step in offsets]}


def _fit_ode(payload: Mapping[str, Any], tables: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from scipy.optimize import curve_fit
    located = None
    for candidate_rows in tables.values():
        columns = list(candidate_rows[0])
        time_column = _alias_column(columns, _TIME_ALIASES, explicit=payload.get("time_column"))
        state_column = _alias_column(columns, _STATE_ALIASES, explicit=payload.get("state_column"))
        state_columns = [str(name) for name in columns
                         if _normalized_name(name).startswith("state_")]
        if time_column is not None and (state_column is not None or state_columns):
            located = (candidate_rows, time_column, state_column, state_columns)
            break
    rows, time_column, state_column, state_columns = located if located is not None else (None, None, None, [])
    if rows is None:
        raise AutomaticModelingError("ode_observations_missing")
    if state_column is None:
        return _fit_coupled_ode(payload, rows, time_column=time_column, state_names=state_columns)
    times = np.asarray([row[time_column] for row in rows], dtype=float)
    states = np.asarray([row[state_column] for row in rows], dtype=float)
    order = np.argsort(times, kind="stable")
    times, states = times[order], states[order]
    raw_queries = payload.get("query_times")
    queries = None if raw_queries is None else np.asarray(raw_queries, dtype=float)
    if (len(times) < 6 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or np.any(states <= 0) or not np.isfinite(states).all()):
        raise AutomaticModelingError("ode_observations_invalid")
    delayed = _fit_delay_differential_candidate(times, states, queries)
    if delayed is not None:
        delayed["model"].update({"time_variable": time_column, "state_variable": state_column})
        return delayed
    origin, initial = float(times[0]), float(states[0])
    shifted = times - origin
    query_shifted = None if queries is None else queries - origin

    decay_rate = -float(np.polyfit(shifted, np.log(states), 1)[0])
    decay_prediction = initial * np.exp(-decay_rate * shifted)
    decay_rss = float(np.sum((decay_prediction - states) ** 2))
    candidates: list[tuple[float, str, Any]] = []
    if decay_rate > 0:
        candidates.append((len(states) * math.log(max(decay_rss / len(states), 1e-24)) + math.log(len(states)),
                           "exponential_decay", decay_rate))

    def logistic(t, capacity, rate):
        return capacity / (1.0 + ((capacity - initial) / initial) * np.exp(-rate * t))
    try:
        params, _ = curve_fit(logistic, shifted, states, p0=[max(states) * 1.2, 0.5],
                              bounds=([max(states), 1e-8], [1e8, 20.0]), maxfev=5000)
        logistic_rss = float(np.sum((logistic(shifted, *params) - states) ** 2))
        candidates.append((len(states) * math.log(max(logistic_rss / len(states), 1e-24)) + 2 * math.log(len(states)),
                           "logistic_growth", params))
    except (RuntimeError, ValueError, FloatingPointError):
        pass
    def equilibrium_approach(t, equilibrium, rate):
        return equilibrium + (initial - equilibrium) * np.exp(-rate * t)
    try:
        equilibrium_guess = max(float(states[-1]), float(np.max(states))) * 1.1
        params, _ = curve_fit(equilibrium_approach, shifted, states,
                              p0=[equilibrium_guess, 0.5],
                              bounds=([0.0, 1e-8], [1e8, 20.0]), maxfev=5000)
        equilibrium_rss = float(np.sum((equilibrium_approach(shifted, *params) - states) ** 2))
        candidates.append((len(states) * math.log(max(equilibrium_rss / len(states), 1e-24))
                           + 2 * math.log(len(states)), "externally_driven_linear", params))
    except (RuntimeError, ValueError, FloatingPointError):
        pass
    if not candidates:
        raise AutomaticModelingError("ode_candidate_fit_failed")
    _criterion, structure, parameters = min(candidates, key=lambda item: item[0])
    if structure == "exponential_decay":
        rate = float(parameters)
        model = {"family": "modeling_ode", "structure": structure, "parameter": rate,
                 "time_variable": time_column, "state_variable": state_column,
                 "selection": "bic_over_decay_logistic"}
        if query_shifted is None:
            return {"status": "needs_input", "model": model, "missing": ["query_times"],
                    "reason": "model_identified_but_prediction_query_missing"}
        return {"model": model,
                "trajectory": (initial * np.exp(-rate * query_shifted)).tolist()}
    if structure == "externally_driven_linear":
        equilibrium, rate = (float(parameters[0]), float(parameters[1]))
        model = {"family": "modeling_ode", "structure": structure,
                 "time_variable": time_column, "state_variable": state_column,
                 "parameters": {"equilibrium": equilibrium, "rate": rate,
                                "forcing": equilibrium * rate},
                 "selection": "bic_over_decay_logistic_external_drive"}
        if query_shifted is None:
            return {"status": "needs_input", "model": model, "missing": ["query_times"],
                    "reason": "model_identified_but_prediction_query_missing"}
        return {"model": model,
                "trajectory": equilibrium_approach(query_shifted, equilibrium, rate).tolist()}
    capacity, rate = (float(parameters[0]), float(parameters[1]))
    model = {"family": "modeling_ode", "structure": structure,
             "time_variable": time_column, "state_variable": state_column,
             "parameters": {"capacity": capacity, "rate": rate},
             "selection": "bic_over_decay_logistic"}
    if query_shifted is None:
        return {"status": "needs_input", "model": model, "missing": ["query_times"],
                "reason": "model_identified_but_prediction_query_missing"}
    return {"model": model,
            "trajectory": logistic(query_shifted, capacity, rate).tolist()}


def _fit_optimization(payload: Mapping[str, Any], tables: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    from scipy.optimize import Bounds, LinearConstraint, linprog, milp
    capacity_binding = None
    item_binding = None
    for rows in tables.values():
        columns = list(rows[0])
        resource_field = _alias_column(columns, _RESOURCE_ALIASES)
        capacity_field = _alias_column(columns, _CAPACITY_ALIASES)
        if resource_field is not None and capacity_field is not None:
            capacity_binding = (rows, resource_field, capacity_field)
        item_field = _alias_column(columns, _ITEM_ALIASES)
        maximize_field = _alias_column(columns, _MAXIMIZE_ALIASES)
        minimize_field = _alias_column(columns, _MINIMIZE_ALIASES)
        if item_field is not None and (maximize_field is not None) != (minimize_field is not None):
            item_binding = (rows, item_field, maximize_field or minimize_field,
                            maximize_field is not None)
    if capacity_binding is None or item_binding is None:
        raise AutomaticModelingError("optimization_tables_missing")
    capacities, resource_field, capacity_field = capacity_binding
    items, item_field, objective_field, maximize = item_binding
    names = [str(row[item_field]) for row in items]
    resources = [str(row[resource_field]) for row in capacities]
    if len(set(names)) != len(names) or len(set(resources)) != len(resources):
        raise AutomaticModelingError("optimization_identifiers_not_unique")
    normalized_item_columns = {_normalized_name(column): str(column) for column in items[0]}
    resource_columns = [normalized_item_columns.get(_normalized_name(resource)) for resource in resources]
    for resource_column in resource_columns:
        if resource_column is None or any(resource_column not in row for row in items):
            raise AutomaticModelingError("optimization_resource_column_missing")
    objective = np.asarray([row[objective_field] for row in items], dtype=float)
    matrix = np.asarray([[row[column] for row in items] for column in resource_columns], dtype=float)
    rhs = np.asarray([row[capacity_field] for row in capacities], dtype=float)
    constraint_ids = list(resources)
    fixed_field = _alias_column(list(items[0]), _FIXED_COST_ALIASES)
    minimum_lot_field = _alias_column(list(items[0]), _MINIMUM_LOT_ALIASES)
    group_field = _alias_column(list(items[0]), _ACTIVATION_GROUP_ALIASES)
    maximum_active = payload.get("maximum_active_items")
    if maximum_active is None and isinstance(payload.get("problem"), str):
        match = re.search(r"(?:最多启用|at\s+most)\s*(\d+)", payload["problem"], re.IGNORECASE)
        maximum_active = int(match.group(1)) if match else None
    activation_budget = payload.get("activation_budget")
    logical = any(value is not None for value in (
        fixed_field, minimum_lot_field, group_field, maximum_active, activation_budget,
    ))
    if logical:
        count = len(items)
        upper = []
        for column in range(count):
            positive = [rhs[row] / matrix[row, column] for row in range(len(resources)) if matrix[row, column] > 0]
            if not positive or not math.isfinite(min(positive)):
                raise AutomaticModelingError("logical_optimization_requires_finite_quantity_bounds")
            upper.append(float(min(positive)))
        fixed = np.asarray([float(row.get(fixed_field, 0.0)) if fixed_field else 0.0 for row in items])
        objective_all = np.concatenate([objective, -fixed if maximize else fixed])
        logical_rows = [np.concatenate([row, np.zeros(count)]) for row in matrix]
        logical_rhs = rhs.tolist()
        for index, name in enumerate(names):
            row = np.zeros(2 * count); row[index] = 1.0; row[count + index] = -upper[index]
            logical_rows.append(row); logical_rhs.append(0.0); constraint_ids.append(f"link:{name}")
            if minimum_lot_field is not None:
                row = np.zeros(2 * count); row[index] = -1.0
                row[count + index] = float(items[index][minimum_lot_field])
                logical_rows.append(row); logical_rhs.append(0.0); constraint_ids.append(f"minimum_lot:{name}")
        if maximum_active is not None:
            maximum_active = int(maximum_active)
            if not 1 <= maximum_active <= count: raise AutomaticModelingError("maximum_active_items_invalid")
            logical_rows.append(np.concatenate([np.zeros(count), np.ones(count)]))
            logical_rhs.append(float(maximum_active)); constraint_ids.append("maximum_active_items")
        if group_field is not None:
            groups = sorted({str(row[group_field]) for row in items})
            for group in groups:
                row = np.zeros(2 * count)
                for index, item in enumerate(items):
                    if str(item[group_field]) == group: row[count + index] = 1.0
                logical_rows.append(row); logical_rhs.append(1.0); constraint_ids.append(f"exclusive_group:{group}")
        if activation_budget is not None:
            logical_rows.append(np.concatenate([np.zeros(count), fixed]))
            logical_rhs.append(float(activation_budget)); constraint_ids.append("activation_budget")
        integrality = np.concatenate([np.ones(count) if payload.get("integer_decisions") is True else np.zeros(count),
                                     np.ones(count)])
        result = milp(-objective_all if maximize else objective_all, integrality=integrality,
                      bounds=Bounds(np.zeros(2 * count), np.asarray([*upper, *([1.0] * count)])),
                      constraints=LinearConstraint(np.asarray(logical_rows), -np.inf, np.asarray(logical_rhs)),
                      options={"time_limit": 10.0, "node_limit": 100_000})
        if not result.success: raise AutomaticModelingError("optimization_no_feasible_solution")
        decision_names = [*[f"{name}_qty" for name in names], *[f"{name}_open" for name in names]]
        units = {**{f"{name}_qty": "item" for name in names}, **{f"{name}_open": "binary" for name in names}}
        return {"model": {"family": "modeling_optimization", "structure": "logical_resource_allocation_milp",
                           "decision_variables": decision_names, "constraint_ids": constraint_ids,
                           "decision_units": units, "objective_unit": "currency",
                           "direction": "maximize" if maximize else "minimize",
                           "selection": "compositional_linear_constraint_grammar"},
                "solution": {name: float(result.x[index]) for index, name in enumerate(decision_names)},
                "objective": float(objective_all @ result.x)}
    if "minimum_total" in payload:
        minimum = float(payload["minimum_total"])
        matrix = np.vstack([matrix, -np.ones(len(items))])
        rhs = np.append(rhs, -minimum)
        constraint_ids.append("minimum_total")
    required_total = payload.get("required_total")
    if required_total is not None:
        required = float(required_total)
        matrix = np.vstack([matrix, -np.ones(len(items)), np.ones(len(items))])
        rhs = np.append(rhs, [-required, required])
        constraint_ids.extend(["balance_lower", "balance_upper"])
    integer_decisions = payload.get("integer_decisions") is True
    if integer_decisions:
        upper = []
        for column in range(len(items)):
            positive = [rhs[row] / matrix[row, column] for row in range(len(resources))
                        if matrix[row, column] > 0]
            upper.append(min(positive) if positive else math.inf)
        result = milp(-objective if maximize else objective,
                      integrality=np.ones(len(items), dtype=int),
                      bounds=Bounds(np.zeros(len(items)), np.asarray(upper)),
                      constraints=LinearConstraint(matrix, -np.inf, rhs),
                      options={"time_limit": 10.0, "node_limit": 100_000})
    else:
        result = linprog(-objective if maximize else objective, A_ub=matrix, b_ub=rhs,
                         bounds=[(0.0, None)] * len(items), method="highs")
    if not result.success:
        raise AutomaticModelingError("optimization_no_feasible_solution")
    structure = ("integer_resource_allocation_milp" if integer_decisions else
                 "equality_constrained_allocation_lp" if required_total is not None else
                 "resource_allocation_lp" if maximize else "minimum_cost_allocation_lp")
    return {"model": {"family": "modeling_optimization",
                      "structure": structure,
                      "decision_variables": names, "constraint_ids": constraint_ids,
                      "field_binding": {"item": item_field, "objective": objective_field,
                                        "resource": resource_field, "capacity": capacity_field,
                                        "resource_columns": resource_columns},
                      "decision_units": {name: "item" for name in names}, "objective_unit": "currency",
                      "direction": "maximize" if maximize else "minimize"},
            "solution": {name: float(result.x[index]) for index, name in enumerate(names)},
            "objective": float(objective @ result.x)}


def _fit_three_tables(tables: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    import pandas as pd

    frames = [pd.DataFrame(rows) for rows in tables.values()]
    weight_aliases = {"weight", "allocation_weight", "share", "fraction", "权重", "分摊比例"}
    for bridge_index, bridge in enumerate(frames):
        weight_columns = [column for column in bridge.columns if _normalized_name(column) in weight_aliases]
        if len(weight_columns) != 1:
            continue
        weight = weight_columns[0]
        remaining = [index for index in range(3) if index != bridge_index]
        for fact_index, dimension_index in (remaining, remaining[::-1]):
            fact, dimension = frames[fact_index], frames[dimension_index]
            key1 = sorted(set(fact.columns) & set(bridge.columns))
            key2 = sorted(set(bridge.columns) & set(dimension.columns))
            numeric_values = [column for column in fact.columns if column not in key1
                              and pd.api.types.is_numeric_dtype(fact[column])]
            groups = [column for column in dimension.columns if column not in key2]
            if len(key1) != 1 or len(key2) != 1 or key1[0] == key2[0] or len(numeric_values) != 1 or len(groups) != 1:
                continue
            if dimension.duplicated(key2).any() or not np.allclose(
                bridge.groupby(key1[0])[weight].sum().to_numpy(dtype=float), 1.0, atol=1e-8,
            ):
                return {"status": "needs_input", "missing": ["valid_bridge_allocation_weights"],
                        "reason": "bridge_weights_invalid"}
            joined = fact.merge(bridge, on=key1[0], how="inner", validate="one_to_many")
            joined = joined.merge(dimension, on=key2[0], how="left", validate="many_to_one")
            if joined[groups[0]].isna().any():
                return {"status": "needs_input", "missing": ["complete_dimension_mapping"],
                        "reason": "dimension_mapping_incomplete"}
            joined["_weighted_value"] = joined[numeric_values[0]] * joined[weight]
            totals = joined.groupby(groups[0], dropna=False)["_weighted_value"].sum()
            return {"model": {"family": "modeling_multi_table", "structure": "weighted_bridge_group_sum",
                              "join_keys": [key1[0], key2[0]], "aggregation": "weighted_sum",
                              "point_in_time": False, "value_column": numeric_values[0],
                              "group_column": groups[0], "weight_column": weight},
                    "rows": int(len(joined)),
                    "group_totals": {str(key): float(value) for key, value in totals.items()}}
    numeric = [[column for column in frame.columns if pd.api.types.is_numeric_dtype(frame[column])]
               for frame in frames]
    fact_candidates = [index for index, columns in enumerate(numeric) if len(columns) == 1]
    paths = []
    for fact_index in fact_candidates:
        remaining = [index for index in range(3) if index != fact_index]
        for middle_index, last_index in (remaining, remaining[::-1]):
            key1 = sorted(set(frames[fact_index].columns) & set(frames[middle_index].columns))
            key2 = sorted(set(frames[middle_index].columns) & set(frames[last_index].columns))
            if len(key1) == len(key2) == 1 and key1[0] != key2[0]:
                paths.append((fact_index, middle_index, last_index, key1[0], key2[0]))
    if len(paths) != 1:
        return {"status": "needs_input", "missing": ["unique_three_table_join_chain"],
                "reason": "three_table_join_chain_ambiguous"}
    fact_index, middle_index, last_index, key1, key2 = paths[0]
    fact, middle, last = frames[fact_index], frames[middle_index], frames[last_index]
    if middle.duplicated([key1]).any():
        return {"status": "needs_input", "missing": ["many_to_many_allocation_rule"],
                "reason": "bridge_table_duplicates_fact_key"}
    if last.duplicated([key2]).any():
        return {"status": "needs_input", "missing": ["dimension_deduplication_or_time_rule"],
                "reason": "terminal_dimension_not_unique"}
    value_columns = numeric[fact_index]
    group_columns = [column for column in last.columns if column != key2]
    if len(value_columns) != 1 or len(group_columns) != 1:
        return {"status": "needs_input", "missing": ["aggregation_value_or_group_binding"],
                "reason": "aggregation_binding_ambiguous"}
    joined = fact.merge(middle, on=key1, how="left", validate="many_to_one")
    joined = joined.merge(last, on=key2, how="left", validate="many_to_one")
    value, group = value_columns[0], group_columns[0]
    if joined[group].isna().any():
        return {"status": "needs_input", "missing": ["complete_dimension_mapping"],
                "reason": "dimension_mapping_incomplete"}
    totals = joined.groupby(group, dropna=False)[value].sum()
    return {"model": {"family": "modeling_multi_table", "structure": "three_table_chain_group_sum",
                      "join_keys": [key1, key2], "aggregation": "sum", "point_in_time": False,
                      "value_column": value, "group_column": group},
            "rows": int(len(joined)),
            "group_totals": {str(key): float(value) for key, value in totals.items()}}


def _fit_multitable(payload: Mapping[str, Any], tables: Mapping[str, list[dict[str, Any]]]) -> dict[str, Any]:
    import pandas as pd
    if len(tables) == 3:
        return _fit_three_tables(tables)
    if len(tables) != 2:
        raise AutomaticModelingError("two_or_three_tables_required_for_bounded_join")
    names = list(tables)
    frames = [pd.DataFrame(tables[name]) for name in names]
    if "event_time" in frames[0].columns and "effective_from" in frames[1].columns:
        left, right = frames
    elif "event_time" in frames[1].columns and "effective_from" in frames[0].columns:
        left, right = frames[1], frames[0]
    else:
        numeric_counts = [sum(pd.api.types.is_numeric_dtype(frame[column]) for column in frame.columns)
                          for frame in frames]
        if numeric_counts[0] == numeric_counts[1]:
            return {"status": "needs_input", "missing": ["fact_and_dimension_roles"],
                    "reason": "table_roles_ambiguous"}
        fact_index = int(numeric_counts[1] > numeric_counts[0])
        left, right = frames[fact_index], frames[1 - fact_index]
    common = sorted(set(left.columns) & set(right.columns))
    if len(common) != 1:
        return {"status": "needs_input", "missing": ["unique_join_key"],
                "reason": "join_key_ambiguous"}
    key = common[0]
    point_in_time = "event_time" in left.columns and "effective_from" in right.columns
    if right.duplicated([key]).any() and not point_in_time:
        return {"status": "needs_input", "missing": ["dimension_deduplication_or_time_rule"],
                "reason": "many_to_many_or_duplicate_dimension"}
    interval = point_in_time and "effective_to" in right.columns
    if interval:
        left["event_time"] = pd.to_datetime(left["event_time"], errors="raise")
        right["effective_from"] = pd.to_datetime(right["effective_from"], errors="raise")
        right["effective_to"] = pd.to_datetime(right["effective_to"], errors="raise")
        fact = left.reset_index(drop=True).reset_index(names="_fact_row")
        joined = fact.merge(right, on=key, how="inner", validate="many_to_many")
        joined = joined[(joined["effective_from"] <= joined["event_time"])
                        & (joined["event_time"] < joined["effective_to"])]
        if joined.duplicated("_fact_row").any():
            return {"status": "needs_input", "missing": ["nonoverlapping_validity_intervals"],
                    "reason": "validity_intervals_overlap"}
        structure = "validity_interval_group_sum"
    elif point_in_time:
        left["event_time"] = pd.to_datetime(left["event_time"], errors="raise")
        right["effective_from"] = pd.to_datetime(right["effective_from"], errors="raise")
        joined = pd.merge_asof(left.sort_values("event_time"), right.sort_values("effective_from"),
                               left_on="event_time", right_on="effective_from", by=key,
                               direction="backward", allow_exact_matches=True)
        structure = "point_in_time_group_sum"
    else:
        joined = left.merge(right, on=key, how="left", validate="many_to_one")
        structure = "many_to_one_group_sum"
    numeric = [column for column in left.columns
               if column != key and pd.api.types.is_numeric_dtype(left[column])]
    categorical = [column for column in right.columns if column not in {key, "effective_from", "effective_to"}]
    if len(numeric) != 1 or len(categorical) != 1 or joined[categorical[0]].isna().any():
        return {"status": "needs_input", "missing": ["aggregation_value_or_group_binding"],
                "reason": "aggregation_binding_ambiguous"}
    value, group = numeric[0], categorical[0]
    totals = joined.groupby(group, dropna=False)[value].sum()
    return {"model": {"family": "modeling_multi_table", "structure": structure,
                      "join_keys": [key], "aggregation": "sum", "point_in_time": point_in_time,
                      "value_column": value, "group_column": group},
            "rows": int(len(joined)), "group_totals": {str(key): float(value) for key, value in totals.items()}}


def bind_modeling_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Bind family, response, and explicit queries without fitting a model."""
    if not isinstance(payload, Mapping):
        raise AutomaticModelingError("modeling_payload_required")
    tables = _attachments(payload)
    family = _infer_family(payload, tables)
    bound = _bind_queries_from_problem(payload, family, tables)
    result: dict[str, Any] = {"status": "bound", "family": family}
    if family == "modeling_algebra":
        matches = []
        for table_name, rows in tables.items():
            response = _alias_column(
                list(rows[0]), _RESPONSE_ALIASES, explicit=bound.get("response_column"))
            if response is not None:
                matches.append((table_name, response, sorted(str(column) for column in rows[0]
                                                              if str(column) != response)))
        if len(matches) != 1:
            raise AutomaticModelingError("algebra_response_binding_ambiguous")
        table_name, response, inputs = matches[0]
        result.update(table_name=table_name, response_variable=response, input_variables=inputs,
                      bound_query_inputs=bound.get("query_inputs"),
                      query_binding=bound.get("query_binding"))
    elif family == "modeling_ode":
        result.update(bound_query_times=bound.get("query_times"),
                      query_binding=bound.get("query_binding"))
    return result


def induce_and_solve_modeling_task(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Infer, fit, and execute one bounded raw-record modeling task."""
    if not isinstance(payload, Mapping):
        raise AutomaticModelingError("modeling_payload_required")
    tables = _attachments(payload)
    family = _infer_family(payload, tables)
    payload = _bind_queries_from_problem(payload, family, tables)
    handler = {"modeling_algebra": _fit_algebra, "modeling_ode": _fit_ode,
               "modeling_optimization": _fit_optimization,
               "modeling_multi_table": _fit_multitable}[family]
    result = handler(payload, tables)
    if family == "modeling_algebra" and payload.get("query_inputs") is not None:
        result["bound_query_inputs"] = payload["query_inputs"]
    if result.get("status") == "needs_input":
        return {**result, "family": family,
                "usage": {"model_api_calls": 0, "numerical_solver_calls": 0,
                          "manual_interventions": 0},
                "policy": "bounded_schema_induction;ambiguous_inputs_are_not_guessed"}
    solver_calls = {"modeling_algebra": 2, "modeling_ode": 2,
                    "modeling_optimization": 1, "modeling_multi_table": 1}[family]
    return {"status": "completed", "family": family, **result,
            "usage": {"model_api_calls": 0, "numerical_solver_calls": solver_calls,
                      "manual_interventions": 0},
            "policy": "bounded_schema_induction;candidate_fit;typed_solution;no_hidden_reference"}


def induce_and_solve_modeling_task_isolated(
    payload: Mapping[str, Any], *, wall_seconds: float = 30.0, memory_mb: int = 1024,
) -> dict[str, Any]:
    """Run bounded induction in a disposable, OS-memory-limited worker."""
    from .solver_runtime import SolverLimits, SolverProcessRunner, SolverRuntimeError

    try:
        return SolverProcessRunner().execute(
            "automatic_modeling/v1", dict(payload),
            limits=SolverLimits(wall_seconds=wall_seconds, memory_mb=memory_mb),
        )
    except SolverRuntimeError as exc:
        return {
            "status": "not_assessed", "reason": exc.code,
            "usage": {"model_api_calls": 0, "numerical_solver_calls": 0,
                      "manual_interventions": 0},
            "execution_supervision": exc.metadata,
            "policy": "resource_failure_is_not_a_mathematical_verdict;no_in_process_fallback",
        }


__all__ = ["AutomaticModelingError", "bind_modeling_task", "induce_and_solve_modeling_task",
           "induce_and_solve_modeling_task_isolated"]
