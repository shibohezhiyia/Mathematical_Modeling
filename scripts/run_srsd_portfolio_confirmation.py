"""Freeze the training-only two-solver portfolio on remaining SRSD tasks."""
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
from core.llm_srbench_adapter import run_llm_srbench_comparison
from core.modeling_confirmation_suite import reserve_modeling_confirmation, complete_modeling_confirmation
from core.srsd_adapter import load_srsd_cases, select_srsd_instances
from core.symbolic_portfolio import run_validation_routed_portfolio
from scripts.run_srsd_final_confirmation import V18_IDS
from scripts.run_srsd_gplearn_confirmation import V19_IDS
from scripts.run_srsd_pilot import DEVELOPMENT_IDS


V20_IDS = (
    "feynman-iii.15.27", "feynman-i.30.5", "feynman-ii.27.18", "feynman-ii.8.31",
    "feynman-i.43.16", "feynman-ii.34.29b", "feynman-i.27.6", "feynman-i.18.16",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--output", default="artifacts/srsd-confirmation/official-portfolio-20261013.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    args = parser.parse_args()
    protocol = "srsd-portfolio-confirmation-v20"
    marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    budget = {"case_count": args.case_count, "train_rows_per_case": 512,
              "test_rows_per_case": 256, "current_wall_seconds_per_case": 30,
              "portfolio_wall_seconds_per_case": 60, "peak_memory_mb": 1024,
              "model_api_calls": 0, "gplearn_program_evaluation_upper_bound": 20_000}
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed, budget=budget,
        methods=("training-validation-symbolic-portfolio/v20", "current-bounded-grammar/v20"),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    excluded = (*DEVELOPMENT_IDS, *V18_IDS, *V19_IDS)
    selected = select_srsd_instances(args.dataset_dir, count=args.case_count, seed=args.seed,
                                     exclude=excluded, maximum_arity=4)
    cases = load_srsd_cases(args.dataset_dir, instance_ids=selected,
                            source_revision=args.source_revision, seed=args.seed)
    commitment = sha256(json.dumps(
        [{"instance_id": case["instance_id"], "source": case["source"]} for case in cases],
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    report = run_llm_srbench_comparison(
        cases, treatment_name="training_validation_portfolio",
        baseline_name="current_bounded_grammar",
        treatment_solver=lambda payload: run_validation_routed_portfolio(payload, seed=args.seed),
        schema_version="mathmodel.srsd-portfolio-confirmation/v1",
        report_status="frozen_official_srsd_portfolio_confirmation",
        selection_policy="hash_selected_before_truth_load;prior_20_srsd_ids_excluded;frozen_source_code",
        source_policy="official_srsd_snapshot;training_only_router;all_failures_retained",
    )
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change"
              if before["status"] == after["status"] == "verified" else "invalidated")
    report["comparison_scope"] = {
        "portfolio_training_fraction": 0.8, "portfolio_validation_fraction": 0.2,
        "hidden_test_reference_used_for_routing": False,
        "portfolio_solver_runs_per_case": 2, "baseline_solver_runs_per_case": 1,
        "peak_memory_limit_mb": 1024, "portfolio_wall_limit_seconds": 60,
        "baseline_wall_limit_seconds": 30,
        "interpretation": (
            "The portfolio is a quality-versus-cost comparison. Any success gain cannot be "
            "reported as equal-compute search efficiency."),
    }
    report["source"] = {"repository": "yoshitomo-matsubara/srsd-feynman_easy",
                        "revision": args.source_revision, "license": "CC-BY-4.0"}
    report["confirmation_protocol"] = {
        "status": status, "seed": args.seed, "case_commitment": commitment,
        "selected_instance_ids": list(selected), "excluded_prior_ids": list(excluded),
        "freeze": freeze, "freeze_before": before, "freeze_after": after,
        "scope": "first_execution_after_freeze;training_only_routing;serialized_models_retained",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(marker, outcome="completed" if status.startswith("completed") else "failed",
                                   report_path=str(destination))
    print(json.dumps({"status": status, "selected": selected,
                      "summary": report["summary"], "effect": report["paired_effect"],
                      "output": str(destination)}, ensure_ascii=False))
    return 0 if status.startswith("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
