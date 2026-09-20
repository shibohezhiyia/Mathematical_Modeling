"""One-shot equal-start-budget sparse search comparison on new topologies."""
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
from scripts.run_sparse_structure_selection_confirmation import _structure_match


_TOPOLOGIES = (
    ('sqrt_i', 'sqrt', ((1, -2, 1, 0), (0, 1, 0, 2))),
    ('sqrt_j', 'sqrt', ((0, 2, -2, 0), (-1, 0, 1, 2))),
    ('atan_i', 'atan', ((2, 0, 0, -2), (0, -1, 1, 2))),
    ('atan_j', 'atan', ((-1, 0, 2, 1), (1, 1, -2, 0))),
)
_VARIANTS = ('heteroscedastic_gaussian', 'student_t3')


def _generate(seed: int):
    rng = np.random.default_rng(seed)
    cases = []
    specs = [(*row, True) for row in _TOPOLOGIES]
    specs += [('exp_control_c', 'exp', (), False), ('sin_control_c', 'sin', (), False)]
    for name, outer, exponents, in_grammar in specs:
        x = np.exp(rng.uniform(-1.2, 1.2, size=(224, 4)))
        if in_grammar:
            coefficients = rng.uniform(0.35, 0.9, size=2)
            latent = 0.5 + sum(float(coefficient) * np.prod(x ** np.asarray(powers), axis=1)
                               for coefficient, powers in zip(coefficients, exponents))
            y = np.sqrt(latent) if outer == 'sqrt' else np.arctan(latent)
        elif outer == 'exp':
            y = np.exp(0.28 * x[:, 0] * x[:, 1] + 0.18 * x[:, 3])
        else:
            y = np.sin(0.5 * x[:, 0] * x[:, 2] + 0.4 * x[:, 3])
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


def _solver(strategy: str, seed: int):
    def fit(payload):
        selected = deepcopy(dict(payload))
        selected['sparse_laurent_search_starts'] = 16
        selected['sparse_laurent_start_strategy'] = strategy
        selected['sparse_laurent_search_seed'] = seed
        return induce_and_solve_modeling_task_isolated(selected)
    return fit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260927)
    parser.add_argument('--output', default='artifacts/modeling-confirmation/sparse-multistart-search-20260920.json')
    parser.add_argument('--registry-dir', default='artifacts/modeling-confirmation/consumed')
    args = parser.parse_args()
    protocol = 'sparse-multistart-search-confirmation-v35'
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={'structure_groups': 6, 'noise_variants': len(_VARIANTS),
                'maximum_terms': 4, 'multistart_width': 16,
                'solver_arms_per_case': 1, 'wall_seconds_per_arm': 30,
                'memory_mb': 1024, 'model_api_calls': 0},
        methods=('top_correlation_16_starts', 'random_16_starts', 'original_greedy_one_start'),
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
            methods = (('top_16', _solver('top_correlation', args.seed)),
                       ('random_16', _solver('random', args.seed)),
                       ('greedy_1', None))
            for method, solver in methods:
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
                                 'search_evaluations': model.get('search_evaluations'),
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
    for method in ('top_16', 'random_16', 'greedy_1'):
        for variant in _VARIANTS:
            selected = [row for row in rows if row['method'] == method and row['variant'] == variant]
            summary[f'{method}:{variant}'] = {
                'correct_predictions': sum(row['correct_prediction'] for row in selected),
                'accepted': sum(row['accepted'] for row in selected),
                'wrong_accept': sum(row['wrong_accept'] for row in selected),
                'exact_support': sum(row['exact_support'] is True for row in selected),
                'material_support': sum(row['material_support'] is True for row in selected),
                'abstained': sum(row['status'] == 'needs_input' for row in selected),
                'errors': sum(row['status'] == 'error' for row in selected),
                'elapsed_wall_seconds': round(sum(row['elapsed_wall_seconds'] for row in selected), 3),
            }
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ('completed_without_source_change'
              if before['status'] == after['status'] == 'verified'
              else 'invalidated_by_source_change')
    report = {'schema_version': 'mathmodel.sparse-multistart-search-confirmation/v1',
              'status': status, 'seed': args.seed, 'case_commitment': commitment,
              'scope': 'new_internal_topologies;top16_vs_random16_same_evaluation_budget;one_start_lower_budget',
              'search_budget_note': ('same maximum of 16 starts per available outer transform and four '
                                     'terms for top/random; exact operation counts on rejected cases unavailable'),
              'support_rule': 'exact serialized exponents; material support ignores coefficients below 1 percent of largest non-intercept magnitude',
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
