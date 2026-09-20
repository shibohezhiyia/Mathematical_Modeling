"""One-shot validation-gated sparse-fit check on new topologies and controls.

The two one-arm routes share observations, split, solver, and validation rule;
only the sparse training-fit gate differs. This is internal synthetic evidence,
not an external-source or full two-arm portfolio evaluation.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from core.automatic_modeling import induce_and_solve_modeling_task_isolated
from core.evaluation_freeze import create_evaluation_freeze, verify_evaluation_freeze
from core.modeling_confirmation_suite import (
    ModelingConfirmationAlreadyConsumed, complete_modeling_confirmation,
    reserve_modeling_confirmation,
)
from core.symbolic_portfolio import run_validation_routed_portfolio


_GRAMMAR_TOPOLOGIES = (
    ('sqrt_e', 'sqrt', ((-1, 1, 1, 0), (0, -2, 0, 2))),
    ('sqrt_f', 'sqrt', ((0, 1, -1, 2), (2, 0, 0, -1))),
    ('atan_e', 'atan', ((-2, 0, 1, 1), (0, 2, -1, 0))),
    ('atan_f', 'atan', ((1, -1, 0, 2), (0, 0, 2, -1))),
)
_VARIANTS = ('response_noise_0_5pct', 'input_round_2')


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _cases(seed: int):
    rng = np.random.default_rng(seed)
    cases = []
    specifications = [(*entry, True) for entry in _GRAMMAR_TOPOLOGIES]
    specifications += [('exp_control', 'exp', (), False),
                       ('sin_control', 'sin', (), False)]
    for name, outer, exponents, in_grammar in specifications:
        x = np.exp(rng.uniform(-1.2, 1.2, size=(224, 4)))
        if in_grammar:
            coefficients = rng.uniform(0.35, 0.9, size=2)
            latent = 0.5 + sum(float(coefficient) * np.prod(x ** np.asarray(powers), axis=1)
                               for coefficient, powers in zip(coefficients, exponents))
            y = np.sqrt(latent) if outer == 'sqrt' else np.arctan(latent)
        elif outer == 'exp':
            y = np.exp(0.35 * x[:, 0] + 0.15 * x[:, 1] * x[:, 2])
        else:
            y = np.sin(0.7 * x[:, 0] * x[:, 1] + 0.4 * x[:, 2])
        order = rng.permutation(len(x))
        train, test = order[:160], order[160:]
        cases.append({'case_id': name, 'in_declared_grammar': in_grammar,
                      'reference_outer': outer, 'reference_exponents': exponents,
                      'training_x': x[train], 'training_y': y[train],
                      'query_inputs': x[test].tolist(), 'test_y': y[test]})
    return cases


def _perturb(x: np.ndarray, y: np.ndarray, *, variant: str, seed: int):
    rng = np.random.default_rng(seed)
    if variant == 'response_noise_0_5pct':
        return x.copy(), y + rng.normal(0, 0.005 * np.std(y), size=len(y))
    if variant == 'input_round_2':
        return np.round(x, 2), y.copy()
    raise ValueError('perturbation_invalid')


def _near_exact_solver(payload):
    strict_payload = deepcopy(dict(payload))
    strict_payload['validation_gated_sparse_fit'] = False
    return induce_and_solve_modeling_task_isolated(strict_payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260922)
    parser.add_argument('--output', default='artifacts/modeling-confirmation/validation-gated-sparse-20260920.json')
    parser.add_argument('--registry-dir', default='artifacts/modeling-confirmation/consumed')
    args = parser.parse_args()
    protocol = 'validation-gated-sparse-confirmation-v31'
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
    commitment = sha256(_canonical([{'case_id': case['case_id'],
                                     'in_declared_grammar': case['in_declared_grammar'],
                                     'training_x': case['training_x'].tolist(),
                                     'training_y': case['training_y'].tolist(),
                                     'query_inputs': case['query_inputs'],
                                     'test_y': case['test_y'].tolist()}
                                    for case in cases]).encode('utf-8')).hexdigest()
    rows = []
    for ordinal, case in enumerate(cases):
        for variant_index, variant in enumerate(_VARIANTS):
            x, y = _perturb(case['training_x'], case['training_y'], variant=variant,
                            seed=args.seed + 100 * ordinal + variant_index)
            payload = {'problem': 'Infer a numeric relation and predict the supplied inputs.',
                       'attachments': [{'name': 'observations', 'format': 'records',
                                        'rows': [{**{f'x{column}': float(x[index, column])
                                                   for column in range(4)},
                                                  'response': float(y[index])}
                                                 for index in range(len(y))]}],
                       'query_inputs': case['query_inputs']}
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
                    model = output.get('model') or {}
                    rows.append({'case_id': case['case_id'], 'variant': variant,
                                 'in_declared_grammar': case['in_declared_grammar'],
                                 'method': method, 'status': output.get('status'),
                                 'reason': output.get('reason'), 'accepted': accepted,
                                 'correct_prediction': correct,
                                 'wrong_accept': bool(accepted and not correct),
                                 'test_nmse': nmse, 'selected_structure': model.get('structure'),
                                 'selected_outer': model.get('outer_transform'),
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
                'elapsed_wall_seconds': round(sum(row['elapsed_wall_seconds'] for row in selected), 3),
            }
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ('completed_without_source_change'
              if before['status'] == after['status'] == 'verified'
              else 'invalidated_by_source_change')
    report = {'schema_version': 'mathmodel.validation-gated-sparse-confirmation/v1',
              'status': status, 'seed': args.seed, 'case_commitment': commitment,
              'scope': 'internal_new_topologies_and_out_of_grammar_controls;one_arm_only;not_external_source',
              'perturbations': list(_VARIANTS),
              'acceptance_rule': 'heldout validation NMSE <= 0.01 and ACC0.1 >= 0.9',
              'hidden_success_rule': 'accepted and independent clean-test NMSE <= 0.1',
              'summary': summary, 'rows': rows,
              'freeze': freeze, 'freeze_before': before, 'freeze_after': after}
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
