"""Freeze a new official-SRSD comparison against fixed-revision gplearn."""
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
from core.gplearn_baseline import GPLEARN_SOURCE_REVISION, fit_gplearn_baseline_isolated
from core.llm_srbench_adapter import run_llm_srbench_comparison
from core.modeling_confirmation_suite import reserve_modeling_confirmation, complete_modeling_confirmation
from core.srsd_adapter import load_srsd_cases, select_srsd_instances
from scripts.run_srsd_final_confirmation import V18_IDS
from scripts.run_srsd_pilot import DEVELOPMENT_IDS


V19_IDS = (
    "feynman-iii.7.38", "feynman-i.18.12", "feynman-ii.15.5", "feynman-ii.4.23",
    "feynman-i.12.5", "feynman-i.14.4", "feynman-i.12.4", "feynman-i.47.23",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261011)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--output", default="artifacts/srsd-confirmation/official-gplearn-20261011.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    args = parser.parse_args()

    protocol = "srsd-gplearn-confirmation-v19"
    marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    budget = {"case_count": args.case_count, "train_rows_per_case": 512,
              "test_rows_per_case": 256, "max_wall_seconds_per_case": 30,
              "memory_mb": 1024, "model_api_calls": 0,
              "gplearn_program_evaluation_upper_bound": 20_000}
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed, budget=budget,
        methods=("current-bounded-grammar/v19", f"official-gplearn/{GPLEARN_SOURCE_REVISION}"),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    selected = select_srsd_instances(
        args.dataset_dir, count=args.case_count, seed=args.seed,
        exclude=(*DEVELOPMENT_IDS, *V18_IDS), maximum_arity=4,
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
        cases, treatment_name="current_bounded_grammar", baseline_name="official_gplearn",
        treatment_payload={}, baseline_payload={"gplearn_population_size": 1000,
                                                "gplearn_generations": 20,
                                                "gplearn_seed": args.seed},
        baseline_solver=fit_gplearn_baseline_isolated,
        schema_version="mathmodel.srsd-gplearn-confirmation/v1",
        report_status="frozen_official_srsd_gplearn_confirmation",
        selection_policy="hash_selected_before_truth_load;prior_srsd_ids_excluded;frozen_source_code",
        source_policy="official_srsd_snapshot;official_gplearn_fixed_revision;public_pretraining_exposure_unknown",
    )
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change"
              if before["status"] == after["status"] == "verified" else "invalidated")
    report["comparison_scope"] = {
        "same": ["SRSD cases", "sampled train/test rows", "30 second wall limit",
                 "1024 MB memory limit", "zero model API calls", "zero run-time manual intervention"],
        "shared_core_operators": ["add", "sub", "multiply", "protected_divide", "sqrt", "sin", "cos"],
        "different": ["candidate representation", "search algorithm", "mathematical priors",
                      "candidate-evaluation accounting"],
        "gplearn_program_evaluation_upper_bound": 20_000,
        "interpretation": (
            "This is a mature-system capability comparison, not isolated proof of search superiority. "
            "Existing equal-topology-budget search ablations remain the search-attribution evidence."),
    }
    report["source"] = {"srsd_repository": "yoshitomo-matsubara/srsd-feynman_easy",
                        "srsd_revision": args.source_revision, "srsd_license": "CC-BY-4.0",
                        "gplearn_repository": "trevorstephens/gplearn",
                        "gplearn_revision": GPLEARN_SOURCE_REVISION,
                        "gplearn_license": "BSD-3-Clause"}
    report["confirmation_protocol"] = {
        "status": status, "seed": args.seed, "case_commitment": commitment,
        "selected_instance_ids": list(selected),
        "excluded_prior_ids": [*DEVELOPMENT_IDS, *V18_IDS],
        "freeze": freeze, "freeze_before": before, "freeze_after": after,
        "scope": "first_execution_after_freeze;all_failures_retained;models_independently_reexecuted",
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
