"""Development-only raw-vs-positive-rescaled diagnosis of failed generators."""
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

from core.symbolic_portfolio import run_validation_routed_portfolio
from scripts.run_independent_generator_decision_probe import _cases


def _positive_rescale(payload):
    rows = payload['attachments'][0]['rows']
    names = sorted(set(rows[0]) - {'response'})
    training = np.asarray([[row[name] for name in names] for row in rows], dtype=float)
    scales = np.max(np.abs(training), axis=0)
    scales[scales == 0] = 1.0
    changed = {
        **payload,
        'attachments': [{**payload['attachments'][0], 'rows': [
            {**row, **{name: float(row[name] / scales[index])
                      for index, name in enumerate(names)}} for row in rows
        ]}],
        'query_inputs': (np.asarray(payload['query_inputs'], dtype=float) / scales).tolist(),
    }
    return changed, dict(zip(names, scales.tolist()))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260920)
    parser.add_argument('--output', default='artifacts/product-workflow-pilot/independent-generator-failure-diagnosis.json')
    args = parser.parse_args()
    rows = []
    for name, original, _reference in _cases(args.seed):
        if name not in {'sklearn_friedman2', 'sklearn_friedman3'}:
            continue
        scaled, factors = _positive_rescale(original)
        for representation, payload in (('raw', original), ('positive_unit_scale', scaled)):
            start = perf_counter()
            try:
                output = run_validation_routed_portfolio(
                    payload, seed=args.seed, routing_policy='evaluate_both', solver_arm_budget=2,
                )
                evidence = output.get('routing_evidence') or {}
                rows.append({
                    'case_id': name, 'representation': representation,
                    'status': output.get('status'), 'reason': output.get('reason'),
                    'validation_metrics': evidence.get('validation_metrics', {}),
                    'selected_arm': evidence.get('selected_arm'),
                    'input_scales': factors if representation == 'positive_unit_scale' else None,
                    'elapsed_wall_seconds': round(perf_counter() - start, 3),
                })
            except Exception as exc:
                rows.append({'case_id': name, 'representation': representation,
                             'status': 'error', 'reason': type(exc).__name__,
                             'elapsed_wall_seconds': round(perf_counter() - start, 3)})
    report = {
        'schema_version': 'mathmodel.independent-generator-failure-diagnosis/v1',
        'status': 'posthoc_development_diagnostic_not_confirmation',
        'seed': args.seed,
        'transformation': 'Each input column divided by training-only maximum absolute value; output and validation thresholds unchanged.',
        'scope': 'Two already-consumed Friedman generator structures; no new generalization evidence.',
        'rows': rows,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'output': str(destination)}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
