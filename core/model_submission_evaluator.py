"""Independent numeric execution of submitted algebra model descriptions."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np


class ModelReexecutionError(ValueError):
    """Raised when a submitted model is incomplete or cannot be evaluated."""


def _array(value: Any, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ModelReexecutionError(f"{name}_invalid") from exc
    if not np.isfinite(result).all():
        raise ModelReexecutionError(f"{name}_nonfinite")
    return result


def _rational_design(model: Mapping[str, Any], queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers = _array(model.get("centers"), name="centers")
    scales = _array(model.get("scales"), name="scales")
    if centers.shape != (queries.shape[1],) or scales.shape != centers.shape or np.any(scales == 0):
        raise ModelReexecutionError("rational_scaling_invalid")
    values = (queries - centers) / scales
    pairs = list(combinations(range(queries.shape[1]), 2))
    triples = list(combinations(range(queries.shape[1]), 3))
    numerator_grammar = str(model.get("numerator_feature_grammar", ""))
    denominator_grammar = str(model.get("denominator_feature_grammar", ""))
    configuration = model.get("rational_feature_configuration")
    if configuration is not None:
        if (not isinstance(configuration, Mapping)
                or any(type(configuration.get(name)) is not bool for name in (
                    "extended_composition", "harmonic_trigonometric",
                    "trigonometric_rational", "product_angle_rational"))):
            raise ModelReexecutionError("rational_feature_configuration_invalid")
        extended = configuration["extended_composition"]
        harmonic = configuration["harmonic_trigonometric"]
        trigonometric = configuration["trigonometric_rational"]
        product_angle = configuration["product_angle_rational"]
    else:
        # Compatibility for reports serialized before the explicit configuration
        # was added. New submissions must not depend on these string inferences.
        extended = "three_way" in numerator_grammar
        harmonic = "integer_harmonic_2_3" in denominator_grammar
        trigonometric = "sin" in denominator_grammar
        product_angle = "half_angle_square" in numerator_grammar
    powered = extended and harmonic
    numerator_columns = [np.ones(len(values)), *[values[:, index]
                                                  for index in range(values.shape[1])]]
    numerator_columns.extend(values[:, left] * values[:, right] for left, right in pairs)
    if extended:
        numerator_columns.extend(np.prod(values[:, indices], axis=1) for indices in triples)
    if powered:
        # The generating grammar includes pure powers only when a three-variable
        # powered interaction is possible, while powered pairs exist for n >= 2.
        if triples:
            numerator_columns.extend(values[:, index] ** degree
                                     for index in range(values.shape[1]) for degree in (2, 3))
        numerator_columns.extend(values[:, other] * values[:, power_index] ** degree
                                 for power_index in range(values.shape[1])
                                 for other in range(values.shape[1]) if other != power_index
                                 for degree in (2, 3))
        numerator_columns.extend(
            values[:, left] * values[:, right] * values[:, power_index] ** degree
            for power_index in range(values.shape[1])
            for left, right in combinations(
                [index for index in range(values.shape[1]) if index != power_index], 2)
            for degree in (2, 3))
    pair_cosines = [np.cos(queries[:, left] * queries[:, right]) for left, right in pairs]
    if product_angle:
        half_squares = [np.sin(0.5 * queries[:, index]) ** 2
                        for index in range(queries.shape[1])]
        numerator_columns.extend(half_squares)
        numerator_columns.extend(queries[:, raw_index] * half_squares[angle_index]
                                 for raw_index in range(queries.shape[1])
                                 for angle_index in range(queries.shape[1]))
        numerator_columns.extend(pair_cosines)
        numerator_columns.extend(queries[:, raw_index] * pair_cosine
                                 for pair_cosine in pair_cosines
                                 for raw_index in range(queries.shape[1]))
    denominator_columns = [values[:, index] for index in range(values.shape[1])]
    denominator_columns.extend(values[:, left] * values[:, right] for left, right in pairs)
    if "three_way" in denominator_grammar:
        denominator_columns.extend(np.prod(values[:, indices], axis=1) for indices in triples)
    if trigonometric:
        denominator_columns.extend(np.cos(queries[:, index]) for index in range(queries.shape[1]))
        denominator_columns.extend(np.sin(queries[:, index]) for index in range(queries.shape[1]))
        if harmonic:
            denominator_columns.extend(np.cos(harmonic * queries[:, index])
                                       for harmonic in (2, 3)
                                       for index in range(queries.shape[1]))
            denominator_columns.extend(np.sin(harmonic * queries[:, index])
                                       for harmonic in (2, 3)
                                       for index in range(queries.shape[1]))
        denominator_columns.extend(values[:, left] * np.cos(queries[:, right])
                                   for left in range(values.shape[1])
                                   for right in range(values.shape[1]))
        if extended:
            denominator_columns.extend(pair_cosines)
            denominator_columns.extend(np.sin(queries[:, left] * queries[:, right])
                                       for left, right in pairs)
        if product_angle and extended:
            denominator_columns.extend(pair_cosine ** 2 for pair_cosine in pair_cosines)
            denominator_columns.extend(queries[:, raw_index] * pair_cosine
                                       for pair_cosine in pair_cosines
                                       for raw_index in range(queries.shape[1]))
            denominator_columns.extend(queries[:, raw_index] * pair_cosine ** 2
                                       for pair_cosine in pair_cosines
                                       for raw_index in range(queries.shape[1]))
    return np.column_stack(numerator_columns), np.column_stack(denominator_columns)


def reexecute_submitted_model(model: Mapping[str, Any], query_inputs: Sequence[Sequence[float]]) -> np.ndarray:
    """Execute a serialized model without using its solver's prediction function."""
    if not isinstance(model, Mapping):
        raise ModelReexecutionError("model_missing")
    inputs = model.get("input_variables")
    if not isinstance(inputs, list) or not inputs or any(type(item) is not str for item in inputs):
        raise ModelReexecutionError("input_variables_invalid")
    queries = _array(query_inputs, name="query_inputs")
    if queries.ndim != 2 or queries.shape[1] != len(inputs):
        raise ModelReexecutionError("query_shape_invalid")
    structure = model.get("structure")
    if structure == "gplearn_prefix_program":
        x_center = _array(model.get("x_center"), name="x_center")
        x_scale = _array(model.get("x_scale"), name="x_scale")
        if (x_center.shape != (queries.shape[1],) or x_scale.shape != x_center.shape
                or np.any(x_scale == 0)):
            raise ModelReexecutionError("gplearn_scaling_invalid")
        program = model.get("program")
        if not isinstance(program, list) or not program or len(program) > 10_000:
            raise ModelReexecutionError("gplearn_program_invalid")
        standardized = (queries - x_center) / x_scale
        supported = {"add": 2, "sub": 2, "mul": 2, "div": 2,
                     "sqrt": 1, "sin": 1, "cos": 1}
        def evaluate(index: int) -> tuple[np.ndarray, int]:
            if index >= len(program) or not isinstance(program[index], Mapping):
                raise ModelReexecutionError("gplearn_program_invalid")
            node = program[index]
            if set(node) == {"feature_index"} and type(node["feature_index"]) is int:
                feature = node["feature_index"]
                if not 0 <= feature < standardized.shape[1]:
                    raise ModelReexecutionError("gplearn_feature_invalid")
                return standardized[:, feature], index + 1
            if set(node) == {"constant"} and type(node["constant"]) in (int, float):
                value = float(node["constant"])
                if not np.isfinite(value):
                    raise ModelReexecutionError("gplearn_constant_invalid")
                return np.full(len(queries), value), index + 1
            if set(node) != {"function", "arity"}:
                raise ModelReexecutionError("gplearn_node_invalid")
            name, arity = node["function"], node["arity"]
            if name not in supported or arity != supported[name]:
                raise ModelReexecutionError("gplearn_function_invalid")
            arguments, cursor = [], index + 1
            for _ in range(arity):
                argument, cursor = evaluate(cursor)
                arguments.append(argument)
            if name == "add":
                value = arguments[0] + arguments[1]
            elif name == "sub":
                value = arguments[0] - arguments[1]
            elif name == "mul":
                value = arguments[0] * arguments[1]
            elif name == "div":
                # np.where evaluates both branches first; protected division
                # must not divide by zero even when that branch is discarded.
                value = np.divide(arguments[0], arguments[1],
                                  out=np.ones_like(arguments[0], dtype=float),
                                  where=np.abs(arguments[1]) > 0.001)
            elif name == "sqrt":
                value = np.sqrt(np.abs(arguments[0]))
            elif name == "sin":
                value = np.sin(arguments[0])
            else:
                value = np.cos(arguments[0])
            return value, cursor
        standardized_prediction, consumed = evaluate(0)
        if consumed != len(program):
            raise ModelReexecutionError("gplearn_program_trailing_nodes")
        y_center, y_scale = float(model.get("y_center")), float(model.get("y_scale"))
        if not np.isfinite(y_center) or not np.isfinite(y_scale) or y_scale <= 0:
            raise ModelReexecutionError("gplearn_output_scaling_invalid")
        prediction = y_center + y_scale * standardized_prediction
    elif structure == "sparse_laurent_outer":
        raw_exponents = model.get("feature_exponents")
        if (not 1 <= queries.shape[1] <= 4 or not isinstance(raw_exponents, list)
                or not 1 <= len(raw_exponents) <= 4
                or any(not isinstance(row, list) or len(row) != queries.shape[1]
                       or any(type(value) is not int or abs(value) > 2 for value in row)
                       or sum(abs(value) for value in row) > 4
                       or sum(value != 0 for value in row) > 3 for row in raw_exponents)):
            raise ModelReexecutionError("sparse_laurent_exponents_invalid")
        exponents = np.asarray(raw_exponents, dtype=int)
        coefficients = _array(model.get("coefficients"), name="coefficients")
        if coefficients.shape != (len(exponents) + 1,):
            raise ModelReexecutionError("sparse_laurent_coefficients_invalid")
        if np.any((queries == 0)[:, None, :] & (exponents[None, :, :] < 0)):
            raise ModelReexecutionError("sparse_laurent_zero_denominator")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            features = np.prod(queries[:, None, :] ** exponents[None, :, :], axis=2)
            inner = coefficients[0] + features @ coefficients[1:]
        if not np.isfinite(inner).all():
            raise ModelReexecutionError("sparse_laurent_prediction_invalid")
        outer = model.get("outer_transform")
        if outer == "identity":
            prediction = inner
        elif outer == "signed_sqrt":
            if (np.any(inner <= 1e-12) or type(model.get("output_sign")) is not int
                    or model["output_sign"] not in (-1, 1)):
                raise ModelReexecutionError("sparse_laurent_sqrt_domain_invalid")
            prediction = float(model["output_sign"]) * np.sqrt(inner)
        elif outer == "arctan":
            prediction = np.arctan(inner)
        else:
            raise ModelReexecutionError("sparse_laurent_outer_invalid")
    elif structure in {"affine", "quadratic", "cubic"}:
        coefficients = _array(model.get("coefficients"), name="coefficients")
        prediction = np.polyval(coefficients, queries[:, 0])
    elif structure == "compositional_symbolic":
        signature = model.get("operator_signature")
        parameters = _array(model.get("parameters"), name="parameters")
        if signature == ["divide", "affine", "affine"] and parameters.shape == (3,):
            denominator = 1.0 + parameters[2] * queries[:, 0]
            prediction = (parameters[0] + parameters[1] * queries[:, 0]) / denominator
        elif (isinstance(signature, list) and 1 <= len(signature) <= 3
              and parameters.shape == (4,)
              and all(item in {"sin", "cos", "tanh", "exp"} for item in signature)):
            transformed = parameters[2] * queries[:, 0] + parameters[3]
            for operator in reversed(signature):
                if operator == "sin":
                    transformed = np.sin(transformed)
                elif operator == "cos":
                    transformed = np.cos(transformed)
                elif operator == "tanh":
                    transformed = np.tanh(transformed)
                else:
                    transformed = np.exp(np.clip(transformed, -40.0, 40.0))
            prediction = parameters[0] + parameters[1] * transformed
        else:
            raise ModelReexecutionError("operator_signature_invalid")
    elif structure in {"multivariate_affine", "bilinear", "trilinear"}:
        terms, coefficients = model.get("terms"), _array(model.get("coefficients"), name="coefficients")
        if not isinstance(terms, list) or len(terms) != len(coefficients):
            raise ModelReexecutionError("polynomial_terms_invalid")
        columns = []
        for term in terms:
            kind, indices = term.get("kind"), term.get("indices")
            if kind == "intercept":
                columns.append(np.ones(len(queries)))
            elif kind == "linear" and isinstance(indices, list) and len(indices) == 1:
                columns.append(queries[:, indices[0]])
            elif kind in {"interaction", "three_way_interaction"} and isinstance(indices, list):
                columns.append(np.prod(queries[:, indices], axis=1))
            else:
                raise ModelReexecutionError("polynomial_term_invalid")
        prediction = np.column_stack(columns) @ coefficients
    elif structure == "signed_absolute_power_law":
        amplitude = float(model.get("amplitude"))
        exponents = _array(model.get("exponents"), name="exponents")
        grammar = model.get("power_feature_grammar")
        if not np.isfinite(amplitude) or not isinstance(grammar, list):
            raise ModelReexecutionError("power_model_invalid")
        columns = [queries[:, index] for index in range(queries.shape[1])]
        if grammar.count("sin"):
            columns.extend(np.sin(queries[:, index]) for index in range(queries.shape[1]))
            columns.extend(np.cos(queries[:, index]) for index in range(queries.shape[1]))
        if len(columns) != len(exponents) or any(np.any(np.abs(column) <= 1e-12) for column in columns):
            raise ModelReexecutionError("power_features_invalid")
        log_value = sum(exponent * np.log(np.abs(column))
                        for exponent, column in zip(exponents, columns))
        prediction = amplitude * np.exp(np.clip(log_value, -700.0, 700.0))
    elif structure == "multivariate_rational":
        numerator, denominator_terms = _rational_design(model, queries)
        parameters = _array(model.get("parameters"), name="parameters")
        if len(parameters) != numerator.shape[1] + denominator_terms.shape[1] + 1:
            raise ModelReexecutionError("rational_parameter_count_invalid")
        numerator_parameters = parameters[:numerator.shape[1]]
        denominator_parameters = parameters[numerator.shape[1]:]
        denominator = np.column_stack([np.ones(len(queries)), denominator_terms]) @ denominator_parameters
        prediction = numerator @ numerator_parameters / denominator
    elif structure in {"sqrt_quadratic_form", "inverse_sqrt_quadratic_form"}:
        specs = model.get("feature_specs")
        parameters = _array(model.get("parameters"), name="parameters")
        if not isinstance(specs, list) or len(specs) != len(parameters):
            raise ModelReexecutionError("root_feature_specs_invalid")
        columns = []
        for spec in specs:
            if spec[0] == "constant":
                columns.append(np.ones(len(queries)))
            elif spec[0] == "raw_square":
                columns.append(queries[:, spec[1]] ** 2)
            elif spec[0] == "ratio_square":
                columns.append((queries[:, spec[1]] / queries[:, spec[2]]) ** 2)
            elif spec[0] == "laurent":
                columns.append(queries[:, spec[1]] /
                               (queries[:, spec[2]] * queries[:, spec[3]] ** 2))
            else:
                raise ModelReexecutionError("root_feature_spec_invalid")
        radicand = np.column_stack(columns) @ parameters
        if np.any(radicand <= 1e-12):
            raise ModelReexecutionError("root_domain_invalid")
        sign = float(model.get("output_sign"))
        prediction = sign * np.sqrt(radicand) if structure == "sqrt_quadratic_form" \
            else sign / np.sqrt(radicand)
    elif structure == "angular_projection_root":
        parameters = _array(model.get("parameters"), name="parameters")
        projected, root_index = int(model["projected_index"]), int(model["root_length_index"])
        angle_left, angle_right = model["angle_indices"]
        angle = queries[:, angle_left] - queries[:, angle_right]
        radicand = queries[:, root_index] ** 2 - queries[:, projected] ** 2 * np.sin(angle) ** 2
        if parameters.shape != (3,) or np.any(radicand <= 1e-12):
            raise ModelReexecutionError("angular_projection_invalid")
        design = np.column_stack([np.ones(len(queries)),
                                  queries[:, projected] * np.cos(angle), np.sqrt(radicand)])
        prediction = design @ parameters
    elif structure == "product_angle_rational":
        parameters = _array(model.get("parameters"), name="parameters")
        indices, substructure = model.get("position_indices"), model.get("substructure")
        if parameters.shape != (2,) or not isinstance(indices, list):
            raise ModelReexecutionError("product_angle_model_invalid")
        if substructure == "half_angle_over_product_angle":
            half_index, multiplier_index, product_index = indices
            cosine = np.cos(queries[:, half_index] * queries[:, product_index])
            numerator = parameters[0] * queries[:, multiplier_index] * np.sin(0.5 * queries[:, half_index]) ** 2
            denominator = 1.0 + parameters[1] * cosine
        else:
            angle_left, angle_right, numerator_index, scale_index = indices
            cosine = np.cos(queries[:, angle_left] * queries[:, angle_right])
            if substructure == "quotient_difference":
                numerator = (parameters[0] * queries[:, numerator_index]
                             + parameters[1] * queries[:, scale_index] * cosine)
                denominator = queries[:, scale_index] * cosine ** 2
            elif substructure == "affine_cosine_square_denominator":
                numerator = parameters[0] * queries[:, numerator_index]
                denominator = cosine + parameters[1] * queries[:, scale_index] * cosine ** 2
            else:
                raise ModelReexecutionError("product_angle_substructure_invalid")
        prediction = numerator / denominator
    else:
        raise ModelReexecutionError("model_structure_unsupported")
    prediction = np.asarray(prediction, dtype=float)
    if prediction.shape != (len(queries),) or not np.isfinite(prediction).all():
        raise ModelReexecutionError("model_prediction_invalid")
    return prediction


__all__ = ["ModelReexecutionError", "reexecute_submitted_model"]
