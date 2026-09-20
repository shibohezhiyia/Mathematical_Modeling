"""One-shot irregular-noise check on known v31 topology families.

New observations are generated after reservation/freeze. The topology names
and generator family are already known; this is perturbation transfer only.
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
from scripts.run_validation_gated_sparse_confirmation import _cases, _near_exact_solver


_VARIANTS = ('heteroscedastic_gaussian', 'student_t3', 'missing_response_5pct')


def _perturb(x: np.ndarray, y: np.ndarray, *, variant: str, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    response = y.copy()
    scale = max(float(np.std(y)), 1e-12)
    if variant == 'heteroscedastic_gaussian':
        normalized_x = (x[:, 0] - np.min(x[:, 0])) / max(float(np.ptp(x[:, 0])), 1e-12)
        response += rng.normal(0.0, (0.005 + 0.015 * normalized_x) * scale)
    elif variant == 'student_t3':
        response += rng.standard_t(3.0, size=len(y)) * (0.01 / np.sqrt(3.0)) * scale
    elif variant == 'missing_response_5pct':
        indices = rng.choice(len(y), max(1, round(0.05 * len(y))), replace=False)
        response[indices] = np.nan
    else:
        raise ValueError('perturbation_invalid')
    return response


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--output', default='artifacts/modeling-confirmation/sparse-irregular-noise-20260920.json')
    parser.add_argument('--registry-dir', default='artifacts/modeling-confirmation/consumed')
    args = parser.parse_args()
    protocol = 'sparse-irregular-noise-confirmation-v33'
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={'structure_groups': 6, 'perturbations': len(_VARIANTS),
                'solver_arms_per_case': 1, 'wall_seconds_per_arm': 30,
                'memory_mb': 1024, 'model_api_calls': 0},
        methods=('validation_gated_sparse_fit_0_01', 'near_exact_sparse_fit_1e_8'),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = _cases(args.seed)
    commitment = sha256(json.dumps(
        [(case['case_id'], case['training_x'].tolist(), case['training_y'].tolist(),
          case['query_inputs'], case['test_y'].tolist()) for case in cases],
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    rows = []
    for ordinal, case in enumerate(cases):
        for variant_index, variant in enumerate(_VARIANTS):
            x = case['training_x']
            y = _perturb(x, case['training_y'], variant=variant,
                         seed=args.seed + 100 * ordinal + variant_index)
            payload = {
                'problem': 'Infer a numeric relation and predict the supplied inputs.',
                'attachments': [{'name': 'observations', 'format': 'records',
                                 'rows': [{**{f'x{column}': float(x[index, column])
                                            for column in range(4)},
                                           'response': float(y[index])}
                                          for index in range(len(y))]}],
                'query_inputs': case['query_inputs'],
            }
            for method, solver in (('validation_gated', None), ('near_exact', _near_exact_solver)):
                started = perf_counter()
                try:
                    output = run_validation_routed_portfolio(
                        payload, seed=args.seed, routing_policy='current_then_fallback',
                        solver_arm_budget=1, current_solver=solver)
                    accepted = output.get('status') == 'completed'
                    nmse = None
                    if accepted:
                        prediction = np.asarray(output['predictions'], dtype=float)
                        nmse = float(np.mean((prediction - case['test_y']) ** 2)
                                     / np.var(case['test_y']))
                    correct = bool(accepted and nmse is not None and nmse <= 0.1)
                    rows.append({'case_id': case['case_id'], 'variant': variant,
                                 'in_declared_grammar': case['in_declared_grammar'],
                                 'method': method, 'status': output.get('status'),
                                 'reason': output.get('reason'), 'accepted': accepted,
                                 'correct_prediction': correct,
                                 'wrong_accept': bool(accepted and not correct),
                                 'test_nmse': nmse,
                                 'selected_structure': (output.get('model') or {}).get('structure'),
                                 'solver_arm_launches': (output.get('usage') or {}).get('numerical_solver_calls'),
                                 'elapsed_wall_seconds': round(perf_counter() - started, 3)})
                except Exception as exc:
                    rows.append({'case_id': case['case_id'], 'variant': variant,
                                 'in_declared_grammar': case['in_declared_grammar'],
                                 'method': method, 'status': 'error', 'reason': type(exc).__name__,
                                 'accepted': False, 'correct_prediction': False,
                                 'wrong_accept': False,
                                 'elapsed_wall_seconds': round(perf_counter() - started, 3)})
    summary = {}
    for method in ('validation_gated', 'near_exact'):
        for variant in _VARIANTS:
            selected = [row for row in rows if row['method'] == method and row['variant'] == variant]
            summary[f'{method}:{variant}'] = {
                'correct_predictions': sum(row['correct_prediction'] for row in selected),
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
        'schema_version': 'mathmodel.sparse-irregular-noise-confirmation/v1',
        'status': status, 'seed': args.seed, 'case_commitment': commitment,
        'scope': 'known_v31_topologies;new_observations;four_in_grammar_two_controls;not_real_measurement_source',
        'perturbations': list(_VARIANTS),
        'acceptance_rule': 'heldout validation NMSE <= 0.01 and ACC0.1 >= 0.9',
        'hidden_success_rule': 'accepted and clean hidden-test NMSE <= 0.1',
        'summary': summary, 'rows': rows,
        'freeze': freeze, 'freeze_before': before, 'freeze_after': after,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    complete_modeling_confirmation(marker, outcome='completed' if status.startswith('completed') else 'failed',
                                   report_path=str(destination))
    print(json.dumps({'status': status, 'summary': summary, 'output': str(destination)},
                     ensure_ascii=False))
    return 0 if status.startswith('completed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
