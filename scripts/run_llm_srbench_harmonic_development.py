"""Re-run the consumed v14 cases after the harmonic rational extension."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_srbench_adapter import load_llm_srbench_cases, run_llm_srbench_comparison
from scripts.run_llm_srbench_transformed_power_confirmation import V14_IDS


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", default=(
        "artifacts/llm-srbench-pilot/v14-post-harmonic-rational-development.json"))
    args = parser.parse_args()
    cases = load_llm_srbench_cases(
        args.dataset_dir, instance_ids=V14_IDS, source_revision=args.source_revision,
        train_limit=512, test_limit=256, seed=20261004,
    )
    report = run_llm_srbench_comparison(
        cases, treatment_name="harmonic_rational_composition",
        baseline_name="without_harmonic_rational_composition",
        treatment_payload={"enable_multivariate_nonlinear": True,
                           "enable_trigonometric_rational_features": True,
                           "enable_extended_multivariate_composition": True,
                           "enable_transformed_power_laurent": True,
                           "enable_harmonic_trigonometric_features": True},
        baseline_payload={"enable_multivariate_nonlinear": True,
                          "enable_trigonometric_rational_features": True,
                          "enable_extended_multivariate_composition": True,
                          "enable_transformed_power_laurent": True,
                          "enable_harmonic_trigonometric_features": False},
        schema_version="mathmodel.llm-srbench-harmonic-development/v1",
    )
    report["status"] = "development_only_after_v14_truth_inspection"
    report["policy"] += ";consumed_v14_cases;not_confirmation"
    report["source"] = cases[0]["source"]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "effect": report["paired_effect"], "output": str(destination)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
