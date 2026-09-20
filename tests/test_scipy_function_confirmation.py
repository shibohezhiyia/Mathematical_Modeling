"""Consumed v29 generator integrity tests; these are not a second confirmation."""
import numpy as np

from scripts.run_scipy_function_confirmation import _generate


def test_scipy_function_cases_hide_labels_and_keep_train_test_disjoint():
    cases = _generate(20260920)
    assert len(cases) == 7
    assert len({case['structure_group'] for case in cases}) == 6
    assert [case['case_id'] for case in cases] == [
        'rosen_2d', 'rosen_3d', 'normal_density', 'logistic_interaction',
        'weighted_log', 'erf_interaction', 'log_sum_exp',
    ]
    for case in cases:
        public, hidden = case['public'], case['hidden']
        rows = public['attachments'][0]['rows']
        queries = public['query_inputs']
        assert len(rows) == 192
        assert len(queries) == len(hidden['test_output']) == 64
        assert np.isfinite(hidden['test_output']).all()
        assert 'test_output' not in str(public)
        variables = sorted(set(rows[0]) - {'response'})
        observed = {tuple(row[name] for name in variables) for row in rows}
        assert observed.isdisjoint(map(tuple, queries))
