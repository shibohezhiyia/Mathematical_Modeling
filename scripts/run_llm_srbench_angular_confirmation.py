"""Freeze a new LLM-SRBench grid for angular-position composition ablation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation_freeze import create_evaluation_freeze, verify_evaluation_freeze
from core.llm_srbench_adapter import (
    load_llm_srbench_cases, run_llm_srbench_comparison, select_llm_srbench_instances,
)
from core.modeling_confirmation_suite import reserve_modeling_confirmation, complete_modeling_confirmation
from scripts.run_llm_srbench_pilot import DEFAULT_IDS as DEVELOPMENT_IDS
from scripts.run_llm_srbench_trig_confirmation import V11_IDS
from scripts.run_llm_srbench_extended_confirmation import V12_IDS, V13_IDS
from scripts.run_llm_srbench_transformed_power_confirmation import V14_IDS
from scripts.run_llm_srbench_harmonic_confirmation import V15_IDS

V16_IDS = (
    "lsr_transform_i.50.26_3_0", "lsr_transform_i.50.26_0_0",
    "lsr_transform_iii.4.33_3_0", "lsr_transform_ii.15.4_1_0",
    "lsr_transform_ii.24.17_1_1", "lsr_transform_i.30.3_0_0",
    "lsr_transform_i.24.6_1_1", "lsr_transform_iii.13.18_1_1",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--output", default=(
        "artifacts/llm-srbench-confirmation/community-mirror-angular-20261006.json"))
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    args = parser.parse_args()
    protocol = "llm-srbench-angular-confirmation-v16"
    marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    budget = {"case_count": args.case_count, "train_rows_per_case": 512,
              "test_rows_per_case": 256, "max_wall_seconds_per_case": 30,
              "memory_mb": 1024, "model_api_calls": 0}
    methods = ("angular-position-composition/v16",
               "without-angular-position-composition/v16")
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed, budget=budget, methods=methods,
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    selected = select_llm_srbench_instances(
        args.dataset_dir, count=args.case_count, seed=args.seed,
        exclude=(*DEVELOPMENT_IDS, *V11_IDS, *V12_IDS, *V13_IDS, *V14_IDS, *V15_IDS),
        maximum_arity=4,
    )
    cases = load_llm_srbench_cases(
        args.dataset_dir, instance_ids=selected, source_revision=args.source_revision,
        train_limit=512, test_limit=256, seed=args.seed,
    )
    commitment = sha256(json.dumps(
        [{"instance_id": case["instance_id"], "source": case["source"]} for case in cases],
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    common = {"enable_multivariate_nonlinear": True,
              "enable_trigonometric_rational_features": True,
              "enable_extended_multivariate_composition": True,
              "enable_transformed_power_laurent": True,
              "enable_harmonic_trigonometric_features": True}
    report = run_llm_srbench_comparison(
        cases, treatment_name="angular_position_composition",
        baseline_name="without_angular_position_composition",
        treatment_payload={**common, "enable_angular_projection_composition": True},
        baseline_payload={**common, "enable_angular_projection_composition": False},
        schema_version="mathmodel.llm-srbench-angular-comparison/v1",
    )
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change"
              if before["status"] == after["status"] == "verified" else "invalidated")
    report["source"] = cases[0]["source"]
    report["confirmation_protocol"] = {
        "status": status, "seed": args.seed, "case_commitment": commitment,
        "selected_instance_ids": list(selected), "freeze": freeze,
        "freeze_before": before, "freeze_after": after,
        "scope": "community_mirror;hash_selection_before_truth_load;prior_45_cases_excluded;angular_position_composition_ablation",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(
        marker, outcome="completed" if status.startswith("completed") else "failed",
        report_path=str(destination),
    )
    print(json.dumps({"status": status, "selected": selected,
                      "summary": report["summary"], "effect": report["paired_effect"],
                      "output": str(destination)}, ensure_ascii=False))
    return 0 if status.startswith("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
