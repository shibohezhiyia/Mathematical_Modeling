"""Freeze and run the final official-SRSD confirmation once."""
from __future__ import annotations

import argparse
from hashlib import sha256
import importlib.util
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
from scripts.run_srsd_pilot import DEVELOPMENT_IDS


V18_IDS = (
    "feynman-ii.3.24", "feynman-ii.38.14", "feynman-i.26.2", "feynman-ii.13.17",
    "feynman-ii.38.3", "feynman-ii.34.11", "feynman-i.25.13", "feynman-i.14.3",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--output", default="artifacts/srsd-confirmation/official-easy-20261009.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    args = parser.parse_args()

    protocol = "srsd-official-final-confirmation-v18"
    marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    budget = {"case_count": args.case_count, "train_rows_per_case": 512,
              "test_rows_per_case": 256, "max_wall_seconds_per_case": 30,
              "memory_mb": 1024, "model_api_calls": 0}
    methods = ("current-bounded-grammar/v18", "polynomial-diagnostic-ablation/v18")
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed, budget=budget, methods=methods,
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    selected = select_srsd_instances(
        args.dataset_dir, count=args.case_count, seed=args.seed,
        exclude=DEVELOPMENT_IDS, maximum_arity=4,
    )
    cases = load_srsd_cases(
        args.dataset_dir, instance_ids=selected, source_revision=args.source_revision,
        train_limit=512, test_limit=256, seed=args.seed,
    )
    commitment = sha256(json.dumps(
        [{"instance_id": case["instance_id"], "source": case["source"]} for case in cases],
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    report = run_llm_srbench_comparison(
        cases,
        treatment_name="current_bounded_grammar",
        baseline_name="polynomial_diagnostic_ablation",
        treatment_payload={"enable_multivariate_nonlinear": True},
        baseline_payload={"enable_multivariate_nonlinear": False},
        schema_version="mathmodel.srsd-final-comparison/v1",
        report_status="frozen_official_srsd_confirmation",
        selection_policy="hash_selected_before_truth_load;development_ids_excluded;frozen_source_code",
        source_policy="official_srsd_snapshot;cc_by_4_0;public_pretraining_exposure_unknown",
    )
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change"
              if before["status"] == after["status"] == "verified" else "invalidated")
    external_packages = {name: bool(importlib.util.find_spec(name))
                         for name in ("pysr", "gplearn", "operon", "symbolicregression")}
    report["source"] = {
        "repository": "yoshitomo-matsubara/srsd-feynman_easy",
        "revision": args.source_revision, "license": "CC-BY-4.0",
        "case_manifests": [{"instance_id": case["instance_id"],
                            "files": case["source"]["files"]} for case in cases],
    }
    report["baseline_scope"] = {
        "polynomial_diagnostic_ablation": (
            "same process/time/memory budget;not a strong mature symbolic-regression baseline;"
            "univariate branch remains the shared bounded implementation"),
        "strong_mature_baseline": "not_executed",
        "availability_probe": external_packages,
        "interpretation": (
            "No claim of search-method superiority is permitted from this report. "
            "A separately installed and preregistered mature baseline remains required."),
    }
    report["cost_and_intervention_policy"] = {
        "model_api_calls": 0, "manual_interventions_during_run": 0,
        "peak_memory_measurement": "unavailable;hard limit and backend recorded per row",
        "elapsed_time_source": "isolated worker supervision metadata",
    }
    report["confirmation_protocol"] = {
        "status": status, "seed": args.seed, "case_commitment": commitment,
        "selected_instance_ids": list(selected), "excluded_development_ids": list(DEVELOPMENT_IDS),
        "freeze": freeze, "freeze_before": before, "freeze_after": after,
        "scope": (
            "first_execution_after_freeze;official_srsd_easy;hash_selection_before_truth_load;"
            "all_failures_retained;submitted_models_independently_reexecuted"),
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
