"""Post-hoc, non-confirmatory predictive diagnostics on consumed v36 data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from scripts.run_yacht_group_holdout_confirmation import _case, _FEATURES


def _nmse(reference: np.ndarray, prediction: np.ndarray) -> float:
    variance = float(np.var(reference))
    return float(np.mean((prediction - reference) ** 2) / variance) if variance > 0 else float('inf')


def _relative_acc(reference: np.ndarray, prediction: np.ndarray) -> float:
    error = np.abs(prediction - reference) / np.maximum(np.abs(reference), 1e-12)
    return float(np.mean(error <= 0.1))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirmation', default='artifacts/modeling-confirmation/yacht-group-holdout-20260921.json')
    parser.add_argument('--output', default='artifacts/modeling-pilot/yacht-group-holdout-diagnosis-20260921.json')
    args = parser.parse_args()
    confirmation = json.loads(Path(args.confirmation).read_text(encoding='utf-8'))
    if confirmation.get('schema_version') != 'mathmodel.yacht-group-holdout-confirmation/v1' or \
            confirmation.get('status') != 'completed_without_source_change':
        raise ValueError('consumed_yacht_confirmation_required')
    seed = confirmation['seed']
    public, hidden, source = _case(seed)
    if source['local_file_sha256'] != confirmation['source']['local_file_sha256']:
        raise ValueError('source_changed_since_confirmation')
    rows = public['attachments'][0]['rows']
    fields = sorted(_FEATURES)
    x = np.asarray([[row[field] for field in fields] for row in rows], dtype=float)
    y = np.asarray([row['response'] for row in rows], dtype=float)
    query = np.asarray(public['query_inputs'], dtype=float)
    order = np.random.default_rng(seed).permutation(len(rows))
    validation_count = max(16, int(round(len(rows) * 0.2)))
    validation, training = order[:validation_count], order[validation_count:]
    baselines = {
        'constant_mean': DummyRegressor(strategy='mean'),
        'standardized_ridge_alpha_1': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        'hist_gradient_boosting_default': HistGradientBoostingRegressor(random_state=seed),
        'random_forest_100_default': RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1),
        'standardized_rbf_svr_default': make_pipeline(StandardScaler(), SVR()),
    }
    results = []
    for name, estimator in baselines.items():
        started = perf_counter()
        estimator.fit(x[training], y[training])
        validation_prediction = np.asarray(estimator.predict(x[validation]), dtype=float)
        hidden_prediction = np.asarray(estimator.predict(query), dtype=float)
        results.append({'method': name, 'validation_nmse': _nmse(y[validation], validation_prediction),
                        'hidden_hull_nmse': _nmse(hidden, hidden_prediction),
                        'validation_acc_0.1': _relative_acc(y[validation], validation_prediction),
                        'hidden_hull_acc_0.1': _relative_acc(hidden, hidden_prediction),
                        'validation_rows': len(validation), 'fit_rows': len(training),
                        'hidden_rows': len(hidden),
                        'elapsed_wall_seconds': round(perf_counter() - started, 3)})
    report = {
        'schema_version': 'mathmodel.yacht-posthoc-diagnosis/v1',
        'status': 'posthoc_development_diagnostic_not_confirmation',
        'confirmation_report': args.confirmation,
        'source_file_sha256': source['local_file_sha256'],
        'seed': seed,
        'scope': ('same consumed hull-form split and training-internal row validation as v36; '
                  'predictive baselines are not symbolic models and do not meet model-reexecution criterion'),
        'interpretation_limit': ('selected after inspecting confirmation failure; cannot revise v36 or establish '
                                 'irreducible noise, physical law, or general-source performance'),
        'results': results,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'results': results, 'output': str(destination)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
