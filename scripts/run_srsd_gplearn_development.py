"""Development comparison with the fixed-revision official gplearn baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gplearn_baseline import fit_gplearn_baseline_isolated
from core.llm_srbench_adapter import run_llm_srbench_comparison
from core.srsd_adapter import load_srsd_cases
from scripts.run_srsd_pilot import DEVELOPMENT_IDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--output", default="artifacts/srsd-pilot/gplearn-development-20261010.json")
    args = parser.parse_args()
    cases = load_srsd_cases(
        args.dataset_dir, instance_ids=DEVELOPMENT_IDS, source_revision=args.source_revision,
        train_limit=512, test_limit=256, seed=args.seed,
    )
    report = run_llm_srbench_comparison(
        cases, treatment_name="current_bounded_grammar", baseline_name="official_gplearn",
        treatment_payload={}, baseline_payload={"gplearn_population_size": 1000,
                                                "gplearn_generations": 20,
                                                "gplearn_seed": args.seed},
        treatment_solver=None, baseline_solver=fit_gplearn_baseline_isolated,
        schema_version="mathmodel.srsd-gplearn-development/v1",
        report_status="development_only_official_srsd_gplearn",
        selection_policy="fixed_consumed_development_ids;not_confirmation",
        source_policy="official_srsd_snapshot;official_gplearn_fixed_revision",
    )
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "output": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
