"""Scoring-contract tests; v36 confirmation itself is consumed exactly once."""
import numpy as np

from scripts.run_yacht_group_holdout_confirmation import _score


def test_yacht_scorer_requires_validated_grade_and_correct_hidden_predictions():
    hidden = np.asarray([1., 2., 3.])
    queries = [[1.], [2.], [3.]]
    model = {'structure': 'affine', 'input_variables': ['x'], 'coefficients': [1., 0.]}
    assert _score({'status': 'completed', 'result_grade': 'exploratory',
                   'model': model, 'predictions': hidden.tolist()}, hidden, queries) == (False, False, None)
    accepted, correct, error = _score({'status': 'completed',
                                       'result_grade': 'validated_candidate',
                                       'model': model, 'predictions': hidden.tolist()}, hidden, queries)
    assert (accepted, correct, error) == (True, True, 0.0)
    accepted, correct, error = _score({'status': 'completed',
                                       'result_grade': 'validated_candidate',
                                       'model': model, 'predictions': [3., 3., 3.]}, hidden, queries)
    assert accepted and not correct and error is None
    wrong_model = {'structure': 'affine', 'input_variables': ['x'], 'coefficients': [1., 2.]}
    accepted, correct, error = _score({'status': 'completed',
                                       'result_grade': 'validated_candidate',
                                       'model': wrong_model, 'predictions': [3., 4., 5.]}, hidden, queries)
    assert accepted and not correct and error is not None and error > 0.1


def test_yacht_scorer_counts_malformed_validated_predictions_as_wrong_accept():
    hidden = np.asarray([1., 2., 3.])
    queries = [[1.], [2.], [3.]]
    model = {'structure': 'affine', 'input_variables': ['x'], 'coefficients': [1., 0.]}
    assert _score({'status': 'completed', 'result_grade': 'validated_candidate',
                   'model': model, 'predictions': [1., 2.]}, hidden, queries) == (True, False, None)
    assert _score({'status': 'completed', 'result_grade': 'validated_candidate',
                   'model': model, 'predictions': [1., float('nan'), 3.]}, hidden, queries) == (True, False, None)
    assert _score({'status': 'completed', 'result_grade': 'validated_candidate',
                   'predictions': [1., 2., 3.]}, hidden, queries) == (True, False, None)
    assert _score({'status': 'completed', 'result_grade': 'validated_candidate',
                   'model': model, 'predictions': ['bad']}, hidden, queries) == (True, False, None)
