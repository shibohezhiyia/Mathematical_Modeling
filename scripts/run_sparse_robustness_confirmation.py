"""Freeze-first mechanism robustness and equal-grammar OMP comparison.

Uses new observations of known v28 topologies. This tests the sparse branch,
not the full two-solver product workflow or cross-topology transfer.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
from itertools import product
import json
from pathlib import Path
import sys
from time import perf_counter
import warnings

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit

from core.evaluation_freeze import create_evaluation_freeze, verify_evaluation_freeze
from core.model_submission_evaluator import reexecute_submitted_model
from core.modeling_confirmation_suite import (
    ModelingConfirmationAlreadyConsumed, complete_modeling_confirmation,
    reserve_modeling_confirmation,
)
from core.sparse_operator_grammar import fit_sparse_laurent_outer
from scripts.run_sparse_operator_topology_confirmation import _generate


_VARIANTS = ('clean', 'noise_0_5pct', 'noise_2pct', 'response_round_2',
             'input_round_2', 'drop_rows_20pct', 'missing_response_5pct')


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'), allow_nan=False)


def _perturb(x: np.ndarray, y: np.ndarray, *, variant: str, seed: int):
    rng = np.random.default_rng(seed)
    features, response = x.copy(), y.copy()
    if variant == 'noise_0_5pct':
        response += rng.normal(0, 0.005 * np.std(y), size=len(y))
    elif variant == 'noise_2pct':
        response += rng.normal(0, 0.02 * np.std(y), size=len(y))
    elif variant == 'response_round_2':
        response = np.round(response, 2)
    elif variant == 'input_round_2':
        features = np.round(features, 2)
    elif variant == 'drop_rows_20pct':
        indices = np.sort(rng.choice(len(y), int(round(0.8 * len(y))), replace=False))
        features, response = features[indices], response[indices]
    elif variant == 'missing_response_5pct':
        indices = rng.choice(len(y), max(1, int(round(0.05 * len(y)))), replace=False)
        response[indices] = np.nan
    elif variant != 'clean':
        raise ValueError('perturbation_invalid')
    return features, response


def _omp_same_grammar(x: np.ndarray, y: np.ndarray):
    """Reference OMP search with the same basis, links, and four-term limit."""
    if (x.ndim != 2 or y.shape != (len(x),) or len(x) < 32
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        return None
    exponents = np.asarray([
        powers for powers in product(range(-2, 3), repeat=x.shape[1])
        if 0 < sum(abs(value) for value in powers) <= 4
        and sum(value != 0 for value in powers) <= 3
        and all(value >= 0 or np.all(np.abs(x[:, index]) > 1e-12)
                for index, value in enumerate(powers))
    ], dtype=int)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        features = np.prod(x[:, None, :] ** exponents[None, :, :], axis=2)
    means = np.mean(features, axis=0)
    centered = features - means
    norms = np.linalg.norm(centered, axis=0)
    keep = np.isfinite(features).all(axis=0) & np.isfinite(norms) & (norms > 1e-14)
    exponents, means, norms = exponents[keep], means[keep], norms[keep]
    design = centered[:, keep] / norms
    if not len(exponents):
        return None
    transformations = [('identity', y, 1)]
    if np.all(y > 0) or np.all(y < 0):
        transformations.append(('signed_sqrt', y ** 2, 1 if np.all(y > 0) else -1))
    if np.all(np.abs(y) < np.pi / 2 - 1e-4):
        tangent = np.tan(y)
        if np.isfinite(tangent).all() and np.max(np.abs(tangent)) < 1e8:
            transformations.append(('arctan', tangent, 1))
    best = None
    variance = max(float(np.var(y)), 1e-24)
    for outer, target, sign in transformations:
        if not np.isfinite(target).all():
            continue
        for terms in range(1, 5):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                estimator = OrthogonalMatchingPursuit(n_nonzero_coefs=terms, fit_intercept=False)
                estimator.fit(design, target - np.mean(target))
            selected = np.flatnonzero(np.abs(estimator.coef_) > 1e-14)
            if not len(selected):
                continue
            raw_coefficients = estimator.coef_[selected] / norms[selected]
            intercept = float(np.mean(target) - means[selected] @ raw_coefficients)
            model = {
                'structure': 'sparse_laurent_outer', 'outer_transform': outer,
                'feature_exponents': exponents[selected].tolist(),
                'coefficients': [intercept, *raw_coefficients.tolist()],
                'output_sign': sign,
            }
            try:
                prediction = reexecute_submitted_model(
                    {**model, 'input_variables': [f'x{index}' for index in range(x.shape[1])]},
                    x.tolist(),
                )
            except ValueError:
                continue
            nmse = float(np.mean((prediction - y) ** 2) / variance)
            if nmse <= 1e-8:
                score = nmse + 1e-12 * (len(selected) + (outer != 'identity'))
                if best is None or score < best[0]:
                    best = (score, model)
    return best[1] if best else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--output', default='artifacts/modeling-confirmation/sparse-robustness-20260920.json')
    parser.add_argument('--registry-dir', default='artifacts/modeling-confirmation/consumed')
    args = parser.parse_args()
    protocol = 'sparse-robustness-confirmation-v30'
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={'structure_groups': 8, 'perturbations': len(_VARIANTS),
                'maximum_terms': 4, 'maximum_outer_links': 3, 'model_api_calls': 0},
        methods=('bounded_greedy_sparse_laurent', 'sklearn_orthogonal_matching_pursuit_same_grammar'),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = _generate(args.seed)
    commitment = sha256(_canonical(cases).encode('utf-8')).hexdigest()
    results = []
    for ordinal, case in enumerate(cases):
        source_rows = case['public']['attachments'][0]['rows']
        x = np.asarray([[row[f'x{index}'] for index in range(4)] for row in source_rows], dtype=float)
        y = np.asarray([row['response'] for row in source_rows], dtype=float)
        queries = case['public']['query_inputs']
        reference = np.asarray(case['hidden']['test_output'], dtype=float)
        for variant_index, variant in enumerate(_VARIANTS):
            perturbed_x, perturbed_y = _perturb(
                x, y, variant=variant, seed=args.seed + 100 * ordinal + variant_index)
            for method, fit in (('greedy', fit_sparse_laurent_outer),
                                ('omp_reference', _omp_same_grammar)):
                started = perf_counter()
                try:
                    model = fit(perturbed_x, perturbed_y)
                    if model is None:
                        results.append({'case_id': case['case_id'], 'structure_group': case['structure_group'],
                                        'variant': variant, 'method': method, 'status': 'abstained',
                                        'accepted': False, 'correct': False, 'wrong_accept': False,
                                        'elapsed_wall_seconds': round(perf_counter() - started, 4)})
                        continue
                    prediction = reexecute_submitted_model(
                        {**model, 'input_variables': ['x0', 'x1', 'x2', 'x3']}, queries)
                    nmse = float(np.mean((prediction - reference) ** 2) / np.var(reference))
                    correct = bool(nmse <= 0.1)
                    results.append({'case_id': case['case_id'], 'structure_group': case['structure_group'],
                                    'variant': variant, 'method': method, 'status': 'accepted',
                                    'accepted': True, 'correct': correct, 'wrong_accept': not correct,
                                    'test_nmse': nmse, 'outer_transform': model['outer_transform'],
                                    'selected_terms': len(model['feature_exponents']),
                                    'elapsed_wall_seconds': round(perf_counter() - started, 4)})
                except Exception as exc:
                    results.append({'case_id': case['case_id'], 'structure_group': case['structure_group'],
                                    'variant': variant, 'method': method, 'status': 'error',
                                    'reason': type(exc).__name__, 'accepted': False,
                                    'correct': False, 'wrong_accept': False,
                                    'elapsed_wall_seconds': round(perf_counter() - started, 4)})
    summaries = {}
    for variant in _VARIANTS:
        summaries[variant] = {}
        for method in ('greedy', 'omp_reference'):
            selected = [row for row in results if row['variant'] == variant and row['method'] == method]
            summaries[variant][method] = {
                'correct': sum(row['correct'] for row in selected),
                'accepted': sum(row['accepted'] for row in selected),
                'wrong_accept': sum(row['wrong_accept'] for row in selected),
                'abstained': sum(row['status'] == 'abstained' for row in selected),
                'errors': sum(row['status'] == 'error' for row in selected),
                'elapsed_wall_seconds': round(sum(row['elapsed_wall_seconds'] for row in selected), 4),
            }
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ('completed_without_source_change'
              if before['status'] == after['status'] == 'verified'
              else 'invalidated_by_source_change')
    report = {
        'schema_version': 'mathmodel.sparse-robustness-confirmation/v1',
        'status': status, 'seed': args.seed, 'case_commitment': commitment,
        'scope': 'mechanism_only;known_v28_topologies;new_observations;not_full_product_workflow_or_cross_source',
        'perturbations': list(_VARIANTS),
        'training_acceptance_rule': 'train NMSE <= 1e-8; no threshold tuning after inspection',
        'test_success_rule': 'accepted and clean hidden-test NMSE <= 0.1; abstention remains in denominator',
        'search_comparison': 'same Laurent exponent pool, three output links, one-to-four terms; implementation and wall-time may differ',
        'summaries': summaries, 'rows': results,
        'freeze': freeze, 'freeze_before': before, 'freeze_after': after,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    complete_modeling_confirmation(marker, outcome='completed' if status.startswith('completed') else 'failed',
                                   report_path=str(destination))
    print(json.dumps({'status': status, 'summaries': summaries, 'output': str(destination)}, ensure_ascii=False))
    return 0 if status.startswith('completed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
