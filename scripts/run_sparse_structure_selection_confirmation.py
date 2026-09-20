"""One-shot equal-grammar BIC versus training-NMSE selection confirmation.

Four new in-grammar topologies and two out-of-grammar controls are generated
after reservation and source freeze. Both methods have the same feature pool,
greedy path, term budget, split, and solver-arm budget; only final selection
among fitted candidates differs. Prediction and exact support are separate.
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


_TOPOLOGIES = (
    ('sqrt_g', 'sqrt', ((-1, 2, 0, 1), (0, -1, 2, 0))),
    ('sqrt_h', 'sqrt', ((2, 0, -1, 1), (0, 2, 0, -2))),
    ('atan_g', 'atan', ((0, -1, 2, 1), (1, 0, -1, 1))),
    ('atan_h', 'atan', ((-2, 1, 0, 1), (0, 1, 2, -1))),
)
_VARIANTS = ('heteroscedastic_gaussian', 'student_t3')


def _generate(seed: int):
    rng = np.random.default_rng(seed)
    cases = []
    specs = [(*row, True) for row in _TOPOLOGIES]
    specs += [('exp_control_b', 'exp', (), False), ('sin_control_b', 'sin', (), False)]
    for name, outer, exponents, in_grammar in specs:
        x = np.exp(rng.uniform(-1.2, 1.2, size=(224, 4)))
        if in_grammar:
            coefficients = rng.uniform(0.35, 0.9, size=2)
            latent = 0.5 + sum(float(coefficient) * np.prod(x ** np.asarray(powers), axis=1)
                               for coefficient, powers in zip(coefficients, exponents))
            y = np.sqrt(latent) if outer == 'sqrt' else np.arctan(latent)
        elif outer == 'exp':
            y = np.exp(0.25 * x[:, 0] * x[:, 2] + 0.2 * x[:, 1])
        else:
            y = np.sin(0.6 * x[:, 0] * x[:, 3] + 0.35 * x[:, 1])
        order = rng.permutation(len(x))
        train, test = order[:160], order[160:]
        cases.append({'case_id': name, 'in_grammar': in_grammar,
                      'reference_outer': outer, 'reference_exponents': exponents,
                      'x_train': x[train], 'y_train': y[train],
                      'x_test': x[test].tolist(), 'y_test': y[test]})
    return cases


def _perturb(x: np.ndarray, y: np.ndarray, *, variant: str, seed: int):
    rng = np.random.default_rng(seed)
    response = y.copy()
    scale = max(float(np.std(y)), 1e-12)
    if variant == 'heteroscedastic_gaussian':
        local = (x[:, 0] - np.min(x[:, 0])) / max(float(np.ptp(x[:, 0])), 1e-12)
        response += rng.normal(0.0, (0.005 + 0.015 * local) * scale)
    elif variant == 'student_t3':
        response += rng.standard_t(3.0, len(y)) * (0.01 / np.sqrt(3.0)) * scale
    else:
        raise ValueError('perturbation_invalid')
    return response


def _minimum_nmse_solver(payload):
    selected = deepcopy(dict(payload))
    selected['sparse_laurent_selection_criterion'] = 'nmse'
    return induce_and_solve_modeling_task_isolated(selected)


def _bic_solver(payload):
    selected = deepcopy(dict(payload))
    selected['sparse_laurent_selection_criterion'] = 'bic'
    return induce_and_solve_modeling_task_isolated(selected)


def _structure_match(model, case):
    if not case['in_grammar']:
        return {'exact_support': None, 'material_support': None, 'outer_match': None}
    if model.get('structure') != 'sparse_laurent_outer':
        return {'exact_support': False, 'material_support': False, 'outer_match': False}
    expected_outer = 'signed_sqrt' if case['reference_outer'] == 'sqrt' else 'arctan'
    target = {tuple(row) for row in case['reference_exponents']}
    exponents = [tuple(row) for row in model.get('feature_exponents', [])]
    coefficients = np.asarray(model.get('coefficients', [])[1:], dtype=float)
    if len(coefficients) != len(exponents):
        return {'exact_support': False, 'material_support': False, 'outer_match': False}
    threshold = 0.01 * max(float(np.max(np.abs(coefficients))) if len(coefficients) else 0.0, 1e-12)
    material = {exponent for exponent, coefficient in zip(exponents, coefficients)
                if abs(coefficient) >= threshold}
    return {'exact_support': set(exponents) == target,
            'material_support': material == target,
            'outer_match': model.get('outer_transform') == expected_outer}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--output', default='artifacts/modeling-confirmation/sparse-structure-selection-20260920.json')
    parser.add_argument('--registry-dir', default='artifacts/modeling-confirmation/consumed')
    args = parser.parse_args()
    protocol = 'sparse-structure-selection-confirmation-v34'
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={'structure_groups': 6, 'noise_variants': len(_VARIANTS),
                'solver_arms_per_case': 1, 'maximum_terms': 4,
                'wall_seconds_per_arm': 30, 'memory_mb': 1024, 'model_api_calls': 0},
        methods=('response_scale_bic', 'minimum_training_nmse'),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = _generate(args.seed)
    commitment = sha256(json.dumps(
        [(case['case_id'], case['x_train'].tolist(), case['y_train'].tolist(),
          case['x_test'], case['y_test'].tolist()) for case in cases],
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    rows = []
    for ordinal, case in enumerate(cases):
        for variant_index, variant in enumerate(_VARIANTS):
            x = case['x_train']
            y = _perturb(x, case['y_train'], variant=variant,
                         seed=args.seed + 100 * ordinal + variant_index)
            payload = {'problem': 'Infer a numeric relation and predict the supplied inputs.',
                       'attachments': [{'name': 'observations', 'format': 'records',
                                        'rows': [{**{f'x{column}': float(x[index, column])
                                                   for column in range(4)},
                                                  'response': float(y[index])}
                                                 for index in range(len(y))]}],
                       'query_inputs': case['x_test']}
            for method, solver in (('bic', _bic_solver), ('minimum_nmse', None)):
                started = perf_counter()
                try:
                    output = run_validation_routed_portfolio(
                        payload, seed=args.seed, routing_policy='current_then_fallback',
                        solver_arm_budget=1, current_solver=solver)
                    accepted = output.get('status') == 'completed'
                    test_nmse = None
                    if accepted:
                        prediction = np.asarray(output['predictions'], dtype=float)
                        test_nmse = float(np.mean((prediction - case['y_test']) ** 2)
                                          / np.var(case['y_test']))
                    correct = bool(accepted and test_nmse is not None and test_nmse <= 0.1)
                    model = output.get('model') or {}
                    rows.append({'case_id': case['case_id'], 'variant': variant,
                                 'in_declared_grammar': case['in_grammar'], 'method': method,
                                 'status': output.get('status'), 'reason': output.get('reason'),
                                 'accepted': accepted, 'correct_prediction': correct,
                                 'wrong_accept': bool(accepted and not correct),
                                 'test_nmse': test_nmse, 'selected_structure': model.get('structure'),
                                 'selected_terms': len(model.get('feature_exponents', [])),
                                 **_structure_match(model, case),
                                 'elapsed_wall_seconds': round(perf_counter() - started, 3)})
                except Exception as exc:
                    rows.append({'case_id': case['case_id'], 'variant': variant,
                                 'in_declared_grammar': case['in_grammar'], 'method': method,
                                 'status': 'error', 'reason': type(exc).__name__,
                                 'accepted': False, 'correct_prediction': False,
                                 'wrong_accept': False, 'exact_support': False,
                                 'material_support': False, 'outer_match': False,
                                 'elapsed_wall_seconds': round(perf_counter() - started, 3)})
    summary = {}
    for method in ('bic', 'minimum_nmse'):
        for variant in _VARIANTS:
            selected = [row for row in rows if row['method'] == method and row['variant'] == variant]
            summary[f'{method}:{variant}'] = {
                'correct_predictions': sum(row['correct_prediction'] for row in selected),
                'accepted': sum(row['accepted'] for row in selected),
                'wrong_accept': sum(row['wrong_accept'] for row in selected),
                'exact_support': sum(row['exact_support'] is True for row in selected),
                'material_support': sum(row['material_support'] is True for row in selected),
                'outer_match': sum(row['outer_match'] is True for row in selected),
                'abstained': sum(row['status'] == 'needs_input' for row in selected),
                'errors': sum(row['status'] == 'error' for row in selected),
                'elapsed_wall_seconds': round(sum(row['elapsed_wall_seconds'] for row in selected), 3),
            }
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ('completed_without_source_change'
              if before['status'] == after['status'] == 'verified'
              else 'invalidated_by_source_change')
    report = {'schema_version': 'mathmodel.sparse-structure-selection-confirmation/v1',
              'status': status, 'seed': args.seed, 'case_commitment': commitment,
              'scope': 'new_internal_topologies;equal_grammar_and_one_arm_budget;not_external_source',
              'support_rule': 'exact serialized exponent set; material set ignores coefficients below 1 percent of largest non-intercept magnitude',
              'acceptance_rule': 'heldout validation NMSE <= 0.01 and ACC0.1 >= 0.9',
              'hidden_success_rule': 'accepted and clean hidden-test NMSE <= 0.1',
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
