"""One-shot product-route check on the locally cached UCI Airfoil data.

The original five predictors and measured response are retained. Metadata
provides column roles; this does not test automatic parsing of arbitrary files
or recovery of a unique physical equation from noisy observations.
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


_DATA = ROOT / 'data' / 'external' / 'uci-airfoil-self-noise' / 'airfoil_self_noise.dat'
_FEATURES = ('frequency', 'attack_angle', 'chord_length',
             'free_stream_velocity', 'suction_side_displacement_thickness')
_SOURCE = 'https://archive.ics.uci.edu/dataset/291/airfoil%2Bself%2Bnoise'


def _case(seed: int):
    raw = _DATA.read_bytes()
    values = np.loadtxt(_DATA)
    if values.shape != (1503, 6) or not np.isfinite(values).all():
        raise ValueError('airfoil_local_source_invalid')
    order = np.random.default_rng(seed).permutation(len(values))
    train, test = order[:512], order[512:640]
    rows = [{**{name: float(values[index, column])
                 for column, name in enumerate(_FEATURES)},
             'response': float(values[index, 5])} for index in train]
    query_order = [_FEATURES.index(name) for name in sorted(_FEATURES)]
    public = {
        'problem': ('Predict measured scaled sound pressure level from frequency, angle of '
                    'attack, chord length, free-stream velocity, and suction-side '
                    'displacement thickness.'),
        'attachments': [{'name': 'airfoil_self_noise.dat', 'format': 'records', 'rows': rows}],
        'query_inputs': values[test][:, query_order].tolist(),
        'gplearn_population_size': 300, 'gplearn_generations': 10,
    }
    return public, values[test, 5], {
        'source_url': _SOURCE, 'local_file_sha256': sha256(raw).hexdigest(),
        'source_rows': len(values), 'training_rows': len(train), 'hidden_test_rows': len(test),
        'feature_names': list(_FEATURES),
        'schema_role_source': 'UCI variable table; mapped into records by evaluation adapter',
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260924)
    parser.add_argument('--output', default='artifacts/modeling-confirmation/airfoil-raw-source-20260920.json')
    parser.add_argument('--registry-dir', default='artifacts/modeling-confirmation/consumed')
    args = parser.parse_args()
    protocol = 'airfoil-raw-source-confirmation-v32'
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({'status': 'refused', 'reason': str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={'source_groups': 1, 'training_rows': 512, 'hidden_test_rows': 128,
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
            accepted = output.get('status') == 'completed'
            nmse = None
            if accepted:
                prediction = np.asarray(output['predictions'], dtype=float)
                nmse = float(np.mean((prediction - hidden) ** 2) / np.var(hidden))
            correct = bool(accepted and nmse is not None and nmse <= 0.1)
            route = output.get('routing_evidence') or {}
            rows.append({'method': name, 'status': output.get('status'),
                         'reason': output.get('reason'), 'result_grade': output.get('result_grade'),
                         'accepted': accepted, 'correct_prediction': correct,
                         'wrong_accept': bool(accepted and not correct),
                         'test_nmse': nmse,
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
        'schema_version': 'mathmodel.airfoil-raw-source-confirmation/v1',
        'status': status, 'seed': args.seed, 'source': source,
        'hidden_test_commitment': commitment,
        'scope': ('one real measured source, two correlated budget variants; schema roles supplied '
                  'from UCI metadata; not automatic file/semantic binding or source-level success rate'),
        'success_rule': 'accepted and hidden-test NMSE <= 0.1; abstentions remain in denominator',
        'rows': rows, 'freeze': freeze, 'freeze_before': before, 'freeze_after': after,
        'policy': 'reserve_and_freeze_before_sampling;all_failures_retained;no_threshold_tuning',
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    complete_modeling_confirmation(marker, outcome='completed' if status.startswith('completed') else 'failed',
                                   report_path=str(destination))
    print(json.dumps({'status': status, 'rows': rows, 'output': str(destination)},
                     ensure_ascii=False))
    return 0 if status.startswith('completed') else 2


if __name__ == '__main__':
    raise SystemExit(main())
