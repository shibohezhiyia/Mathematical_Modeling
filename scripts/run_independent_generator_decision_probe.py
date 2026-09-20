"""Development-only transfer probe on scikit-learn's bundled generators.

This intentionally does not consume a confirmation marker. The cases are
public and small; results may guide later preregistration but are not an
independent estimate of deployment accuracy.
"""
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
import sklearn
from sklearn.datasets import make_friedman2, make_friedman3, make_regression

from core.symbolic_portfolio import run_validation_routed_portfolio


def _cases(seed: int):
    generators = (
        ('sklearn_friedman2', lambda: make_friedman2(n_samples=256, noise=0.0, random_state=seed)),
        ('sklearn_friedman3', lambda: make_friedman3(n_samples=256, noise=0.0, random_state=seed + 1)),
        ('sklearn_linear3', lambda: make_regression(
            n_samples=256, n_features=3, n_informative=3, noise=0.0,
            random_state=seed + 2,
        )),
    )
    for name, generate in generators:
        x, y = generate()
        rng = np.random.default_rng(seed + len(name))
        order = rng.permutation(len(y))
        train, test = order[:192], order[192:]
        columns = [f'x{index}' for index in range(x.shape[1])]
        rows = [{**{column: float(x[index, position]) for position, column in enumerate(columns)},
                 'response': float(y[index])} for index in train]
        yield name, {
            'problem': 'Infer the numeric relationship from observations and predict the held-out inputs.',
            'attachments': [{'name': name, 'format': 'records', 'rows': rows}],
            'query_inputs': x[test].astype(float).tolist(),
        }, y[test].astype(float)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--output', default='artifacts/product-workflow-pilot/independent-generator-decision-comparison.json')
    args = parser.parse_args()
    rows = []
    for name, payload, reference in _cases(args.seed):
        for strategy, policy in (('conditional_two', 'current_then_fallback'),
                                 ('always_two', 'evaluate_both')):
            started = perf_counter()
            try:
                output = run_validation_routed_portfolio(
                    payload, seed=args.seed, routing_policy=policy, solver_arm_budget=2,
                )
                row = {
                    'case_id': name, 'strategy': strategy, 'status': output.get('status'),
                    'result_grade': output.get('result_grade'),
                    'reason': output.get('reason'),
                    'recommended_action': output.get('recommended_action'),
                    'solver_arm_launches': (output.get('usage') or {}).get('numerical_solver_calls'),
                    'selected_arm': (output.get('routing_evidence') or {}).get('selected_arm'),
                    'elapsed_wall_seconds': round(perf_counter() - started, 3),
                }
                if output.get('status') == 'completed':
                    prediction = np.asarray(output['predictions'], dtype=float)
                    row['test_nmse'] = float(np.mean((prediction - reference) ** 2) / np.var(reference))
                    row['accepted_correct_at_nmse_0_1'] = bool(row['test_nmse'] <= 0.1)
                else:
                    row['test_nmse'] = None
                    row['accepted_correct_at_nmse_0_1'] = False
                rows.append(row)
            except Exception as exc:
                rows.append({'case_id': name, 'strategy': strategy, 'status': 'error',
                             'reason': type(exc).__name__,
                             'elapsed_wall_seconds': round(perf_counter() - started, 3),
                             'accepted_correct_at_nmse_0_1': False})
    summaries = {}
    for strategy in ('conditional_two', 'always_two'):
        selected = [row for row in rows if row['strategy'] == strategy]
        summaries[strategy] = {
            'task_count': len(selected),
            'accepted': sum(row['status'] == 'completed' for row in selected),
            'accepted_correct': sum(row['accepted_correct_at_nmse_0_1'] for row in selected),
            'abstained': sum(row['status'] == 'needs_input' for row in selected),
            'errors': sum(row['status'] == 'error' for row in selected),
            'solver_arm_launches': sum(int(row.get('solver_arm_launches') or 0) for row in selected),
            'elapsed_wall_seconds': round(sum(row['elapsed_wall_seconds'] for row in selected), 3),
        }
    report = {
        'schema_version': 'mathmodel.independent-generator-decision-comparison/v1',
        'status': 'development_only_not_frozen_confirmation',
        'source': {'library': 'scikit-learn', 'version': sklearn.__version__,
                   'generators': ['make_friedman2', 'make_friedman3', 'make_regression']},
        'seed': args.seed,
        'protocol': '192 train observations; 64 hidden test observations; portfolio makes its own 80/20 training validation split; no test labels in solver payload',
        'success_rule': 'accepted and hidden-test NMSE <= 0.1; abstention remains in denominator',
        'scope': 'three public generator structures; posthoc development comparison after single-policy probe; no confidence interval or product-population inference',
        'rows': rows,
        'summary': summaries,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'summary': report['summary'], 'output': str(destination)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
