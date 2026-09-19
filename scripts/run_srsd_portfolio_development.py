"""Evaluate the training-only symbolic router on consumed v19 cases."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_srbench_adapter import run_llm_srbench_comparison
from core.srsd_adapter import load_srsd_cases
from core.symbolic_portfolio import run_validation_routed_portfolio
from scripts.run_srsd_gplearn_confirmation import V19_IDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--output", default="artifacts/srsd-pilot/portfolio-development-20261012.json")
    args = parser.parse_args()
    cases = load_srsd_cases(args.dataset_dir, instance_ids=V19_IDS,
                            source_revision=args.source_revision, seed=args.seed)
    report = run_llm_srbench_comparison(
        cases, treatment_name="training_validation_portfolio",
        baseline_name="current_bounded_grammar",
        treatment_solver=lambda payload: run_validation_routed_portfolio(payload, seed=args.seed),
        schema_version="mathmodel.srsd-portfolio-development/v1",
        report_status="development_on_consumed_v19_cases",
        selection_policy="consumed_v19_ids;not_confirmation",
        source_policy="official_srsd_snapshot;training_only_router;double_solver_cost",
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "effect": report["paired_effect"],
                      "output": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
