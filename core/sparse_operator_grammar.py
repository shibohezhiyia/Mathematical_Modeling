"""Bounded sparse Laurent composition for exact multivariate relations.

The grammar enumerates exponent vectors, not named benchmark equations. Its
output transforms are identity, signed square root, and arctangent. Search is
training-only, limited to four selected terms, and accepts only near-exact
training reconstructions; downstream validation remains mandatory.
"""
from __future__ import annotations

from itertools import product
import math
from typing import Any

import numpy as np


def _terms(values: np.ndarray, exponents: np.ndarray) -> np.ndarray:
    if np.any((values == 0)[:, None, :] & (exponents[None, :, :] < 0)):
        raise ValueError('laurent_zero_denominator')
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        return np.prod(values[:, None, :] ** exponents[None, :, :], axis=2)


def evaluate_sparse_laurent_outer(model: dict[str, Any], values: np.ndarray) -> np.ndarray:
    exponents = np.asarray(model.get('feature_exponents'), dtype=int)
    coefficients = np.asarray(model.get('coefficients'), dtype=float)
    if (values.ndim != 2 or exponents.ndim != 2 or exponents.shape[1] != values.shape[1]
            or not 1 <= len(exponents) <= 4 or coefficients.shape != (len(exponents) + 1,)
            or np.any(np.abs(exponents) > 2) or np.any(np.sum(np.abs(exponents), axis=1) > 4)
            or not np.isfinite(coefficients).all()):
        raise ValueError('sparse_laurent_model_invalid')
    features = _terms(values, exponents)
    inner = coefficients[0] + features @ coefficients[1:]
    if not np.isfinite(inner).all():
        raise ValueError('sparse_laurent_prediction_invalid')
    outer = model.get('outer_transform')
    if outer == 'identity':
        result = inner
    elif outer == 'signed_sqrt':
        if np.any(inner <= 1e-12):
            raise ValueError('sparse_laurent_sqrt_domain_invalid')
        sign = model.get('output_sign')
        if sign not in (-1, 1):
            raise ValueError('sparse_laurent_sign_invalid')
        result = sign * np.sqrt(inner)
    elif outer == 'arctan':
        result = np.arctan(inner)
    else:
        raise ValueError('sparse_laurent_outer_invalid')
    if not np.isfinite(result).all():
        raise ValueError('sparse_laurent_prediction_invalid')
    return result


def fit_sparse_laurent_outer(x: np.ndarray, y: np.ndarray, *, max_terms: int = 4,
                             maximum_normalized_mse: float = 1e-8,
                             selection_criterion: str = 'nmse',
                             search_starts: int = 1,
                             start_strategy: str = 'top_correlation',
                             seed: int = 0) -> dict[str, Any] | None:
    """Return a budgeted candidate below the caller's training-fit gate.

    The default gate is near-exact. A relaxed gate must be paired with a
    separate held-out validation decision before the model is recommended.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (x.ndim != 2 or y.shape != (len(x),) or not 1 <= x.shape[1] <= 4
            or len(x) < 32 or not np.isfinite(x).all() or not np.isfinite(y).all()
            or type(max_terms) is not int or not 1 <= max_terms <= 4
            or type(selection_criterion) is not str or selection_criterion not in {'bic', 'nmse'}
            or type(search_starts) is not int or not 1 <= search_starts <= 16
            or type(start_strategy) is not str
            or start_strategy not in {'top_correlation', 'random'}
            or type(seed) is not int or not 0 <= seed <= 2**32 - 1
            or not 0 < maximum_normalized_mse < 1):
        return None
    candidates = np.asarray([
        powers for powers in product(range(-2, 3), repeat=x.shape[1])
        if 0 < sum(abs(value) for value in powers) <= 4
        and sum(value != 0 for value in powers) <= 3
    ], dtype=int)
    zero_columns = np.any(np.abs(x) <= 1e-12, axis=0)
    candidates = candidates[~np.any((candidates < 0) & zero_columns, axis=1)]
    if not len(candidates):
        return None
    with np.errstate(over='ignore', divide='ignore', invalid='ignore'):
        features = _terms(x, candidates)
    means = np.mean(features, axis=0)
    centered = features - means
    norms = np.linalg.norm(centered, axis=0)
    finite = np.isfinite(features).all(axis=0) & np.isfinite(norms) & (norms > 1e-14)
    candidates = candidates[finite]
    features = features[:, finite]
    means = means[finite]
    centered = centered[:, finite]
    norms = norms[finite]
    if not len(candidates):
        return None
    standardized = centered / norms
    transformations = [('identity', y, 1)]
    if np.all(y > 0) or np.all(y < 0):
        transformations.append(('signed_sqrt', y ** 2, 1 if np.all(y > 0) else -1))
    if np.all(np.abs(y) < np.pi / 2 - 1e-4):
        tangent = np.tan(y)
        if np.isfinite(tangent).all() and np.max(np.abs(tangent)) < 1e8:
            transformations.append(('arctan', tangent, 1))
    best: tuple[float, dict[str, Any]] | None = None
    variance = max(float(np.var(y)), 1e-24)
    rng = np.random.default_rng(seed)
    search_evaluations = 0
    for outer, target, sign in transformations:
        if not np.isfinite(target).all():
            continue
        centered_target = target - np.mean(target)
        first_correlations = np.abs(standardized.T @ centered_target)
        admissible = np.flatnonzero(first_correlations > 1e-12)
        if not len(admissible):
            continue
        if start_strategy == 'top_correlation':
            first_indices = admissible[np.argsort(-first_correlations[admissible], kind='stable')
                                       [:search_starts]]
        else:
            first_indices = rng.choice(admissible, size=min(search_starts, len(admissible)),
                                       replace=False)
        for first_index in first_indices:
            residual = centered_target.copy()
            selected: list[int] = []
            for step in range(max_terms):
                if step == 0:
                    index = int(first_index)
                else:
                    correlations = np.abs(standardized.T @ residual)
                    correlations[selected] = -np.inf
                    index = int(np.argmax(correlations))
                    if not np.isfinite(correlations[index]) or correlations[index] <= 1e-12:
                        break
                selected.append(index)
                coefficients_scaled, *_ = np.linalg.lstsq(standardized[:, selected],
                                                          centered_target, rcond=None)
                search_evaluations += 1
                residual = centered_target - standardized[:, selected] @ coefficients_scaled
                coefficients = coefficients_scaled / norms[selected]
                intercept = float(np.mean(target) - means[selected] @ coefficients)
                model = {
                    'structure': 'sparse_laurent_outer', 'outer_transform': outer,
                    'feature_exponents': candidates[selected].tolist(),
                    'coefficients': [intercept, *coefficients.tolist()],
                    'output_sign': sign,
                    'selection': 'bounded_greedy_sparse_laurent_outer',
                    'selection_criterion': ('response_scale_bic' if selection_criterion == 'bic'
                                            else 'minimum_training_nmse'),
                    'search_start_strategy': start_strategy,
                    'search_starts': search_starts,
                }
                try:
                    predicted = evaluate_sparse_laurent_outer(model, x)
                except ValueError:
                    continue
                normalized_mse = float(np.mean((predicted - y) ** 2) / variance)
                if normalized_mse <= maximum_normalized_mse:
                    if selection_criterion == 'bic':
                        parameter_count = 1 + len(selected) + int(outer != 'identity')
                        score = len(y) * math.log(max(normalized_mse, 1e-24)) \
                            + parameter_count * math.log(len(y))
                    else:
                        score = normalized_mse + 1e-12 * (len(selected) + int(outer != 'identity'))
                    if best is None or score < best[0]:
                        best = (score, model)
    if best is None:
        return None
    best[1]['search_evaluations'] = search_evaluations
    return best[1]


__all__ = ['evaluate_sparse_laurent_outer', 'fit_sparse_laurent_outer']
