"""One-shot real-source check with unseen yacht hull forms.

UCI metadata supplies column roles; the system still receives the raw measured
records rather than a known equation. This is not a test of arbitrary-file
parsing, and a low prediction error would not prove physical-law recovery.
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
from core.model_submission_evaluator import reexecute_submitted_model
from core.symbolic_portfolio import run_validation_routed_portfolio


_DATA = ROOT / 'data' / 'external' / 'uci-yacht-hydrodynamics' / 'yacht_hydrodynamics.data'
_ARCHIVE = ROOT / 'data' / 'external' / 'uci-yacht-hydrodynamics.zip'
_FEATURES = ('longitudinal_center_buoyancy', 'prismatic_coefficient',
             'length_displacement_ratio', 'beam_draught_ratio',
             'length_beam_ratio', 'froude_number')
_FILE_SHA256 = '00dfecc0fc01ddd4c90b558a3ac11b246df8ebcfea130724223475a9a67f0ea1'
_ARCHIVE_SHA256 = 'aa52b68f88c4bb552187a53ef4c5753fa178f6a36035a3771c5bc04e078487ac'
_SOURCE = 'https://archive.ics.uci.edu/dataset/243/yacht%2Bhydrodynamics'


def _case(seed: int):
    file_bytes = _DATA.read_bytes()
    archive_bytes = _ARCHIVE.read_bytes()
    if sha256(file_bytes).hexdigest() != _FILE_SHA256 or sha256(archive_bytes).hexdigest() != _ARCHIVE_SHA256:
        raise ValueError('yacht_local_source_hash_mismatch')
    values = np.loadtxt(_DATA)
    if values.shape != (308, 7) or not np.isfinite(values).all():
        raise ValueError('yacht_local_source_invalid')
    hulls, hull_id, counts = np.unique(values[:, :5], axis=0, return_inverse=True,
                                        return_counts=True)
    if hulls.shape != (22, 5) or not np.all(counts == 14):
        raise ValueError('yacht_hull_grouping_invalid')
    hull_order = np.random.default_rng(seed).permutation(len(hulls))
    training_hulls, heldout_hulls = hull_order[:16], hull_order[16:]
    train = np.flatnonzero(np.isin(hull_id, training_hulls))
    test = np.flatnonzero(np.isin(hull_id, heldout_hulls))
    if len(train) != 224 or len(test) != 84:
        raise ValueError('yacht_group_split_invalid')
    rows = [{**{name: float(values[index, column])
                 for column, name in enumerate(_FEATURES)},
             'response': float(values[index, 6])} for index in train]
    query_order = [_FEATURES.index(name) for name in sorted(_FEATURES)]
    public = {
        'problem': ('Predict measured residuary resistance per unit weight of displacement '
                    'from six hull-shape and Froude-number measurements.'),
        'attachments': [{'name': 'yacht_hydrodynamics.data', 'format': 'records', 'rows': rows}],
        'query_inputs': values[test][:, query_order].tolist(),
        'gplearn_population_size': 300, 'gplearn_generations': 10,
    }
    source = {
        'source_url': _SOURCE, 'archive_sha256': _ARCHIVE_SHA256,
        'local_file_sha256': _FILE_SHA256, 'source_rows': len(values),
        'training_rows': len(train), 'hidden_test_rows': len(test),
        'training_hull_groups': len(training_hulls), 'hidden_hull_groups': len(heldout_hulls),
        'feature_names': list(_FEATURES),
        'schema_role_source': 'UCI variable table; mapped into records by evaluation adapter',
        'split_rule': 'seeded permutation of unique first-five-column hull forms; 16 train / 6 holdout',
    }
    return public, values[test, 6], source


def _score(output: dict, hidden: np.ndarray,
           query_inputs: list[list[float]]) -> tuple[bool, bool, float | None]:
    accepted = output.get('status') == 'completed' and output.get('result_grade') == 'validated_candidate'
    if not accepted:
        return False, False, None
    try:
        predictions = np.asarray(output.get('predictions'), dtype=float)
        independently_computed = reexecute_submitted_model(output.get('model'), query_inputs)
    except (TypeError, ValueError, FloatingPointError):
        return True, False, None
    if (predictions.shape != hidden.shape or independently_computed.shape != hidden.shape
            or not np.isfinite(predictions).all()
            or not np.allclose(predictions, independently_computed, rtol=1e-8, atol=1e-10)):
        return True, False, None
    variance = float(np.var(hidden))
    nmse = float(np.mean((predictions - hidden) ** 2) / variance) if variance > 0 else float('inf')
    return True, nmse <= 0.1, nmse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260925)
    parser.add_argument('--output', default='artifacts/modeling-confirmation/yacht-group-holdout-20260921.json')
    parser.add_argument('--registry-dir', default='artifacts/modeling-confirmation/consumed')
    args = parser.parse_args()
    protocol = 'yacht-group-holdout-confirmation-v36'
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={'source_groups': 1, 'training_hull_groups': 16, 'hidden_hull_groups': 6,
                'training_rows': 224, 'hidden_test_rows': 84,
                'maximum_solver_arms': 2, 'program_evaluations': 3000,
                'wall_seconds_per_arm': 30, 'memory_mb': 1024, 'model_api_calls': 0},
        methods=('current_one_arm', 'current_then_gplearn_two_arms'),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    public, hidden, source = _case(args.seed)
    commitment = sha256(json.dumps(hidden.tolist(), separators=(',', ':'),
                                   allow_nan=False).encode('utf-8')).hexdigest()
    rows = []
    for name, budget in (('current_one_arm', 1), ('current_then_gplearn_two_arms', 2)):
        started = perf_counter()
        try:
            output = run_validation_routed_portfolio(
                public, seed=args.seed, routing_policy='current_then_fallback',
                solver_arm_budget=budget)
            accepted, correct, nmse = _score(output, hidden, public['query_inputs'])
            route = output.get('routing_evidence') or {}
            rows.append({'method': name, 'status': output.get('status'),
                         'reason': output.get('reason'), 'result_grade': output.get('result_grade'),
                         'accepted': accepted, 'correct_prediction': correct,
                         'wrong_accept': bool(accepted and not correct), 'test_nmse': nmse,
                         'selected_structure': (output.get('model') or {}).get('structure'),
                         'selected_arm': route.get('selected_arm'),
                         'validation_metrics': route.get('validation_metrics'),
                         'solver_arm_launches': (output.get('usage') or {}).get('numerical_solver_calls'),
                         'worker_elapsed_seconds': (output.get('execution_supervision') or {}).get('elapsed_seconds'),
                         'elapsed_wall_seconds': round(perf_counter() - started, 3)})
        except Exception as exc:
            rows.append({'method': name, 'status': 'error', 'reason': type(exc).__name__,
                         'accepted': False, 'correct_prediction': False, 'wrong_accept': False,
                         'elapsed_wall_seconds': round(perf_counter() - started, 3)})
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ('completed_without_source_change'
              if before['status'] == after['status'] == 'verified'
              else 'invalidated_by_source_change')
    report = {
        'schema_version': 'mathmodel.yacht-group-holdout-confirmation/v1',
        'status': status, 'seed': args.seed, 'source': source,
        'hidden_test_commitment': commitment,
        'scope': ('one measured source, two correlated solver-budget variants, hull-form group holdout; '
                  'UCI supplies schema roles; not arbitrary-file binding or physical-law recovery'),
        'success_rule': 'validated acceptance and hidden-hull NMSE <= 0.1; abstentions remain in denominator',
        'rows': rows, 'freeze': freeze, 'freeze_before': before, 'freeze_after': after,
        'policy': 'reserve_and_freeze_before_group_split;all_failures_retained;no_threshold_tuning',
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    complete_modeling_confirmation(marker, outcome='completed' if status.startswith('completed') else 'failed',
                                   report_path=str(destination))
    print(json.dumps({'status': status, 'rows': rows, 'output': str(destination)},
                     ensure_ascii=True))
    return 0 if status.startswith('completed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
