"""Posthoc check of the rejected gplearn model on the consumed Friedman2 case."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.symbolic_portfolio import run_same_split_portfolio_arm
from scripts.run_independent_generator_decision_probe import _cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--output', default='artifacts/product-workflow-pilot/friedman2-near-miss.json')
    args = parser.parse_args()
    name, payload, truth = next(item for item in _cases(args.seed) if item[0] == 'sklearn_friedman2')
    result = run_same_split_portfolio_arm(payload, arm='official_gplearn', seed=args.seed)
    prediction = np.asarray(result['predictions'], dtype=float)
    relative = np.abs(prediction - truth) / np.maximum(np.abs(truth), 1e-12)
    test_nmse = float(np.mean((prediction - truth) ** 2) / np.var(truth))
    report = {
        'schema_version': 'mathmodel.friedman2-near-miss/v1',
        'status': 'posthoc_development_diagnostic_not_confirmation',
        'case_id': name, 'seed': args.seed,
        'validation_metrics': result['model']['same_split_attribution']['validation_metrics'],
        'hidden_test_metrics_diagnostic_only': {
            'nmse': test_nmse,
            'acc_0_1': float(np.mean(relative <= 0.1)),
            'maximum_relative_error': float(np.max(relative)),
            'relative_error_quantiles': np.quantile(relative, [0.5, 0.9, 0.95, 1.0]).tolist(),
        },
        'interpretation': 'Test labels inspected after rejection; must not be used to tune this case and then claim fresh confirmation.',
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'validation': report['validation_metrics'],
                      'test': report['hidden_test_metrics_diagnostic_only'],
                      'output': str(destination)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
