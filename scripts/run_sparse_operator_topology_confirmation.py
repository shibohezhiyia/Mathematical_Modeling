"""One-shot topology holdout for the bounded sparse outer-operator grammar.

All eight structures are generated only after reserving the seed and freezing
source. They are synthetic grammar-scope tests, not external-source tasks.
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

from core.evaluation_freeze import create_evaluation_freeze, verify_evaluation_freeze
from core.modeling_confirmation_suite import (
    ModelingConfirmationAlreadyConsumed, complete_modeling_confirmation,
    reserve_modeling_confirmation,
)
from core.symbolic_portfolio import run_validation_routed_portfolio


# Deliberately not the Friedman2/3 topology or the development unit-test cases.
# Each case has two latent Laurent terms and an unknown variable placement.
_TOPOLOGIES = (
    ('sqrt_a', 'sqrt', ((2, -1, 0, 0), (0, 0, 1, 1))),
    ('sqrt_b', 'sqrt', ((1, 0, -1, 0), (0, 2, 0, 2))),
    ('sqrt_c', 'sqrt', ((2, 0, 0, -2), (0, 1, 1, 0))),
    ('sqrt_d', 'sqrt', ((0, 2, -1, 0), (1, 0, 0, 2))),
    ('atan_a', 'atan', ((0, -1, 1, 1), (1, 0, -2, 0))),
    ('atan_b', 'atan', ((1, 2, 0, -1), (-1, 0, 1, 0))),
    ('atan_c', 'atan', ((1, 0, 1, -2), (-1, 1, 0, 0))),
    ('atan_d', 'atan', ((0, 1, -2, 1), (2, -1, 0, 0))),
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _generate(seed: int):
    rng = np.random.default_rng(seed)
    cases = []
    for case_id, outer, powers in _TOPOLOGIES:
        x = np.exp(rng.uniform(-1.5, 1.5, size=(256, 4)))
        coefficients = rng.uniform(0.35, 1.15, size=2)
        terms = np.column_stack([
            np.prod(x ** np.asarray(exponent), axis=1) for exponent in powers
        ])
        inside = 0.4 + terms @ coefficients
        y = np.sqrt(inside) if outer == 'sqrt' else np.arctan(inside)
        order = rng.permutation(256)
        train, test = order[:192], order[192:]
        rows = [{**{f'x{column}': float(x[index, column]) for column in range(4)},
                 'response': float(y[index])} for index in train]
        cases.append({
            'case_id': case_id, 'structure_group': case_id,
            'public': {'problem': 'Infer the numeric relation and predict the supplied inputs.',
                       'attachments': [{'name': 'observations', 'format': 'records', 'rows': rows}],
                       'query_inputs': x[test].tolist()},
            'hidden': {'test_output': y[test].tolist(), 'outer': outer,
                       'exponents': powers, 'coefficients': coefficients.tolist()},
        })
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--output', default='artifacts/modeling-confirmation/sparse-operator-topology-20260920.json')
    parser.add_argument('--registry-dir', default='artifacts/modeling-confirmation/consumed')
    args = parser.parse_args()
    protocol = 'sparse-operator-topology-confirmation-v28'
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={'case_count': 8, 'maximum_solver_arms': 2,
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
                prediction = output.get('predictions')
                nmse = None
                if output.get('status') == 'completed':
                    reference = np.asarray(case['hidden']['test_output'], dtype=float)
                    values = np.asarray(prediction, dtype=float)
                    nmse = float(np.mean((values - reference) ** 2) / np.var(reference))
                accepted = output.get('status') == 'completed'
                correct = bool(accepted and nmse is not None and nmse <= 1e-4)
                rows.append({
                    'case_id': case['case_id'], 'structure_group': case['structure_group'],
                    'arm': arm, 'status': output.get('status'), 'reason': output.get('reason'),
                    'selected_arm': (output.get('routing_evidence') or {}).get('selected_arm'),
                    'selected_structure': (output.get('model') or {}).get('structure'),
                    'solver_arm_launches': (output.get('usage') or {}).get('numerical_solver_calls'),
                    'test_nmse': nmse, 'accepted': accepted, 'correct': correct,
                    'wrong_accept': bool(accepted and not correct),
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
        'schema_version': 'mathmodel.sparse-operator-topology-confirmation/v1',
        'status': status, 'seed': args.seed, 'case_commitment': commitment,
        'scope': 'eight synthetic topologies inside declared Laurent/outer grammar;not external-source or general modeling accuracy',
        'success_rule': 'accepted and independent hidden-test NMSE <= 1e-4; abstentions remain in denominator',
        'summary': summary, 'rows': rows,
        'freeze': freeze, 'freeze_before': before, 'freeze_after': after,
        'policy': 'reserve_and_freeze_before_generation;all_failures_retained;no_posthoc_threshold_change',
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
