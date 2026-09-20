"""Post-hoc Airfoil validation diagnosis; never a fresh confirmation.

Uses the already-consumed v32 split. Predictive baselines are diagnostics,
not symbolic models and not independently executable submissions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.automatic_modeling import induce_and_solve_modeling_task
from core.model_submission_evaluator import reexecute_submitted_model
from core.symbolic_portfolio import _split_training_validation, _validation_metrics
from scripts.run_airfoil_raw_source_confirmation import _case


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--output', default='artifacts/modeling-pilot/airfoil-failure-diagnosis-20260920.json')
    args = parser.parse_args()
    public, _unused_hidden_test, source = _case(args.seed)
    fit_payload, validation_queries, validation_response, validation_count = (
        _split_training_validation(public, seed=args.seed, validation_fraction=0.2))
    names = sorted(set(fit_payload['attachments'][0]['rows'][0]) - {'response'})
    rows = fit_payload['attachments'][0]['rows']
    x = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    y = np.asarray([row['response'] for row in rows], dtype=float)
    validation_x = np.asarray(validation_queries, dtype=float)

    current = induce_and_solve_modeling_task(fit_payload)
    current_prediction = reexecute_submitted_model(current['model'], validation_queries)
    results = {
        'current_bounded_grammar': {
            'selected_structure': current['model']['structure'],
            'search_scope': current['model'].get('search_scope'),
            'validation': _validation_metrics(current_prediction, validation_response),
        },
    }
    for name, estimator in (
        ('standardized_ridge', make_pipeline(StandardScaler(), Ridge(alpha=1.0))),
        ('hist_gradient_boosting', HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, random_state=args.seed)),
    ):
        estimator.fit(x, y)
        prediction = np.asarray(estimator.predict(validation_x), dtype=float)
        results[name] = {'validation': _validation_metrics(prediction, validation_response)}
    report = {
        'schema_version': 'mathmodel.airfoil-failure-diagnosis/v1',
        'status': 'posthoc_development_diagnostic',
        'source': source, 'seed': args.seed,
        'training_partition_rows': len(rows), 'validation_rows': validation_count,
        'results': results,
        'limitations': ('same already-consumed v32 split; baselines are predictive regressors, '
                        'not symbolic equation recovery; no hidden-test score used; '
                        'no threshold changed from these results'),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'results': results,
                      'output': str(destination)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
