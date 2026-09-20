"""Bounded composition tests; Friedman rows here are consumed development data."""
import numpy as np
import pytest

from core.automatic_modeling import induce_and_solve_modeling_task
from core.model_submission_evaluator import ModelReexecutionError, reexecute_submitted_model
from core.sparse_operator_grammar import fit_sparse_laurent_outer
from scripts.run_independent_generator_decision_probe import _cases


@pytest.mark.parametrize('outer', ['signed_sqrt', 'arctan'])
def test_sparse_grammar_recovers_permuted_compositions_without_named_template(outer):
    rng = np.random.default_rng(131)
    x = np.exp(rng.uniform(-2.0, 2.0, size=(220, 4)))
    if outer == 'signed_sqrt':
        y = np.sqrt(0.7 + 1.3 * x[:, 2] * x[:, 0] ** 2 + 0.4 * x[:, 1] / x[:, 3])
    else:
        y = np.arctan(0.7 * x[:, 2] * x[:, 0] / x[:, 1] + 0.4 / x[:, 3])
    model = fit_sparse_laurent_outer(x[:160], y[:160])
    assert model is not None
    assert model['outer_transform'] == outer
    serialized = {**model, 'input_variables': ['a', 'b', 'c', 'd']}
    prediction = reexecute_submitted_model(serialized, x[160:].tolist())
    assert np.mean((prediction - y[160:]) ** 2) / np.var(y[160:]) < 1e-8


def test_sparse_grammar_refuses_random_response_and_invalid_model():
    rng = np.random.default_rng(132)
    x = rng.uniform(0.2, 2.0, size=(120, 2))
    y = rng.normal(size=120)
    assert fit_sparse_laurent_outer(x, y) is None
    model = {'structure': 'sparse_laurent_outer', 'input_variables': ['x0'],
             'feature_exponents': [[-1]], 'coefficients': [0.0, 1.0],
             'outer_transform': 'identity'}
    with pytest.raises(ModelReexecutionError, match='zero_denominator'):
        reexecute_submitted_model(model, [[0.0]])
    with pytest.raises(ModelReexecutionError, match='exponents_invalid'):
        reexecute_submitted_model({**model, 'feature_exponents': [[-1.5]]}, [[1.0]])


def test_noisy_sparse_candidate_requires_opt_in_and_is_not_near_exact():
    rng = np.random.default_rng(807)
    x = np.exp(rng.uniform(-1.0, 1.0, size=(160, 4)))
    clean = np.sqrt(0.4 + 0.8 * x[:, 0] ** 2 / x[:, 1]
                    + 0.6 * x[:, 2] * x[:, 3])
    noisy = clean + rng.normal(0.0, 0.005 * np.std(clean), len(clean))
    assert fit_sparse_laurent_outer(x, noisy) is None
    approximate = fit_sparse_laurent_outer(x, noisy, maximum_normalized_mse=0.01)
    assert approximate is not None
    prediction = reexecute_submitted_model(
        {**approximate, 'input_variables': ['x0', 'x1', 'x2', 'x3']}, x.tolist())
    training_nmse = np.mean((prediction - noisy) ** 2) / np.var(noisy)
    assert 1e-8 < training_nmse <= 0.01


def test_multistart_sparse_search_is_bounded_and_deterministic():
    rng = np.random.default_rng(937)
    x = np.exp(rng.uniform(-1.1, 1.1, size=(96, 4)))
    y = np.sqrt(0.5 + 0.7 * x[:, 0] ** 2 / x[:, 1]
                + 0.4 * x[:, 2] / x[:, 3])
    first = fit_sparse_laurent_outer(x, y, search_starts=16, seed=937)
    second = fit_sparse_laurent_outer(x, y, search_starts=16, seed=937)
    assert first is not None and first == second
    assert first['search_evaluations'] <= 3 * 16 * 4
    assert first['search_evaluations'] > 4
    random_first = fit_sparse_laurent_outer(
        x, y, search_starts=16, start_strategy='random', seed=937)
    random_second = fit_sparse_laurent_outer(
        x, y, search_starts=16, start_strategy='random', seed=937)
    assert random_first == random_second
    assert fit_sparse_laurent_outer(x, y, search_starts=17) is None
    assert fit_sparse_laurent_outer(x, y, start_strategy='unbounded') is None


def test_consumed_friedman_development_regression_uses_composition_not_interpolation():
    for name, payload, truth in _cases(20260920):
        if name not in {'sklearn_friedman2', 'sklearn_friedman3'}:
            continue
        output = induce_and_solve_modeling_task(payload)
        assert output['status'] == 'completed'
        assert output['model']['structure'] == 'sparse_laurent_outer'
        assert output['result_grade'] == 'exploratory'
        assert output['recommended_action'] == 'run_independent_validation_before_using_prediction'
        prediction = reexecute_submitted_model(output['model'], payload['query_inputs'])
        assert np.mean((prediction - truth) ** 2) / np.var(truth) < 1e-8
