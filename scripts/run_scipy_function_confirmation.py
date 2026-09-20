"""One-shot independent-implementation function transfer check.

Truth comes from installed SciPy functions, not from this project's Laurent
generator. Cases are generated after the freeze. This is still a small public
synthetic-function sample, not ordinary user modeling accuracy.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from scipy.optimize import rosen
from scipy.special import erf, expit, logsumexp, xlogy
from scipy.stats import norm

from core.evaluation_freeze import create_evaluation_freeze, verify_evaluation_freeze
from core.modeling_confirmation_suite import (
    ModelingConfirmationAlreadyConsumed, complete_modeling_confirmation,
    reserve_modeling_confirmation,
)
from core.symbolic_portfolio import run_validation_routed_portfolio


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _generate(seed: int):
    rng = np.random.default_rng(seed)
    cases = []
    # These independent library operators were declared before first execution.
    definitions = (
        ('rosen_2d', 'rosenbrock', 2, (-2.0, 2.0), lambda x: np.apply_along_axis(rosen, 1, x)),
        ('rosen_3d', 'rosenbrock', 3, (-2.0, 2.0), lambda x: np.apply_along_axis(rosen, 1, x)),
        ('normal_density', 'normal_pdf', 3, (-2.0, 2.0),
         lambda x: norm.pdf(x[:, 0], loc=x[:, 1], scale=x[:, 2])),
        ('logistic_interaction', 'expit', 3, (-2.0, 2.0),
         lambda x: expit(0.7 * x[:, 0] * x[:, 1] - 0.6 * x[:, 2])),
        ('weighted_log', 'xlogy', 3, (-2.0, 2.0),
         lambda x: xlogy(x[:, 0], x[:, 1]) + 0.4 * x[:, 2]),
        ('erf_interaction', 'erf', 3, (-2.0, 2.0),
         lambda x: erf(x[:, 0] * x[:, 1]) + 0.3 * x[:, 2]),
        ('log_sum_exp', 'logsumexp', 3, (-2.0, 2.0),
         lambda x: logsumexp(x[:, :2], axis=1) + 0.2 * x[:, 2]),
    )
    for case_id, group, dimensions, bounds, generate in definitions:
        x = rng.uniform(*bounds, size=(256, dimensions))
        if case_id == 'normal_density':
            x[:, 2] = rng.uniform(0.4, 2.0, size=256)
        if case_id == 'weighted_log':
            x[:, 0] = rng.uniform(0.3, 2.0, size=256)
            x[:, 1] = rng.uniform(0.3, 2.0, size=256)
        y = np.asarray(generate(x), dtype=float)
        if y.shape != (256,) or not np.isfinite(y).all():
            raise ValueError('independent_function_values_invalid')
        order = rng.permutation(256)
        train, test = order[:192], order[192:]
        rows = [{**{f'x{column}': float(x[index, column]) for column in range(dimensions)},
                 'response': float(y[index])} for index in train]
        cases.append({
            'case_id': case_id, 'structure_group': group,
            'public': {'problem': 'Infer the numeric relation and predict the supplied inputs.',
                       'attachments': [{'name': 'measurements', 'format': 'records', 'rows': rows}],
                       'query_inputs': x[test].tolist()},
            'hidden': {'test_output': y[test].tolist()},
        })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--output', default='artifacts/modeling-confirmation/scipy-function-20260920.json')
    parser.add_argument('--registry-dir', default='artifacts/modeling-confirmation/consumed')
    args = parser.parse_args()
    protocol = 'scipy-function-confirmation-v29'
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={'case_count': 7, 'maximum_solver_arms': 2,
                'max_wall_seconds_per_arm': 30, 'memory_mb': 1024, 'model_api_calls': 0},
        methods=('sparse_laurent_outer_enabled', 'same_system_sparse_laurent_outer_disabled'),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = _generate(args.seed)
    commitment = sha256(_canonical(cases).encode('utf-8')).hexdigest()
    rows = []
    for arm, enabled in (('enabled', True), ('disabled', False)):
        for case in cases:
            payload = {**case['public'], 'enable_sparse_laurent_outer': enabled}
            started = perf_counter()
            try:
                output = run_validation_routed_portfolio(
                    payload, seed=args.seed, routing_policy='current_then_fallback',
                    solver_arm_budget=2,
                )
                accepted = output.get('status') == 'completed'
                nmse = None
                if accepted:
                    reference = np.asarray(case['hidden']['test_output'], dtype=float)
                    prediction = np.asarray(output['predictions'], dtype=float)
                    nmse = float(np.mean((prediction - reference) ** 2) / np.var(reference))
                correct = bool(accepted and nmse is not None and nmse <= 0.1)
                rows.append({
                    'case_id': case['case_id'], 'structure_group': case['structure_group'],
                    'arm': arm, 'status': output.get('status'), 'reason': output.get('reason'),
                    'selected_arm': (output.get('routing_evidence') or {}).get('selected_arm'),
                    'selected_structure': (output.get('model') or {}).get('structure'),
                    'validation_metrics': (output.get('routing_evidence') or {}).get('validation_metrics'),
                    'test_nmse': nmse, 'accepted': accepted, 'correct': correct,
                    'wrong_accept': bool(accepted and not correct),
                    'solver_arm_launches': (output.get('usage') or {}).get('numerical_solver_calls'),
                    'elapsed_wall_seconds': round(perf_counter() - started, 3),
                })
            except Exception as exc:
                rows.append({'case_id': case['case_id'], 'structure_group': case['structure_group'],
                             'arm': arm, 'status': 'error', 'reason': type(exc).__name__,
                             'accepted': False, 'correct': False, 'wrong_accept': False,
                             'elapsed_wall_seconds': round(perf_counter() - started, 3)})
    summary = {}
    for arm in ('enabled', 'disabled'):
        selected = [row for row in rows if row['arm'] == arm]
        summary[arm] = {
            'correct': sum(row['correct'] for row in selected),
            'accepted': sum(row['accepted'] for row in selected),
            'wrong_accept': sum(row['wrong_accept'] for row in selected),
            'abstained': sum(row['status'] == 'needs_input' for row in selected),
            'errors': sum(row['status'] == 'error' for row in selected),
            'solver_arm_launches': sum(int(row.get('solver_arm_launches') or 0) for row in selected),
            'elapsed_wall_seconds': round(sum(row['elapsed_wall_seconds'] for row in selected), 3),
        }
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ('completed_without_source_change'
              if before['status'] == after['status'] == 'verified'
              else 'invalidated_by_source_change')
    report = {
        'schema_version': 'mathmodel.scipy-function-confirmation/v1',
        'status': status, 'seed': args.seed, 'case_commitment': commitment,
        'source': {'library': 'scipy', 'version': scipy.__version__,
                   'relationship': 'independent_installed_function_implementations'},
        'scope': 'seven public mathematical function tasks; six related structure groups; not real-world attachment understanding',
        'success_rule': 'accepted and hidden-test NMSE <= 0.1; abstentions remain in denominator',
        'summary': summary, 'rows': rows,
        'freeze': freeze, 'freeze_before': before, 'freeze_after': after,
        'policy': 'reserve_and_freeze_before_case_generation;all_failures_retained;no_posthoc_threshold_change',
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    complete_modeling_confirmation(marker, outcome='completed' if status.startswith('completed') else 'failed',
                                   report_path=str(destination))
    print(json.dumps({'status': status, 'summary': summary, 'output': str(destination)}, ensure_ascii=False))
    return 0 if status.startswith('completed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
