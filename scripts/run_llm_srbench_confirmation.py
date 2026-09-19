"""Freeze, hash-select, and run an external-format community-mirror confirmation."""
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument("--output", default="artifacts/llm-srbench-confirmation/community-mirror-20261001.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    args = parser.parse_args()
    protocol = "llm-srbench-mirror-confirmation-v11"
    marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    budget = {"case_count": args.case_count, "train_rows_per_case": 512,
              "test_rows_per_case": 256, "max_wall_seconds_per_case": 30,
              "memory_mb": 1024, "model_api_calls": 0}
    methods = ("current-multivariate-grammar/v11", "polynomial-only-ablation/v11")
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed, budget=budget, methods=methods,
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    selected = select_llm_srbench_instances(
        args.dataset_dir, count=args.case_count, seed=args.seed,
        exclude=DEVELOPMENT_IDS, maximum_arity=4,
    )
    cases = load_llm_srbench_cases(
        args.dataset_dir, instance_ids=selected, source_revision=args.source_revision,
        train_limit=512, test_limit=256, seed=args.seed,
    )
    commitment = sha256(json.dumps(
        [{"instance_id": case["instance_id"], "source": case["source"]} for case in cases],
        sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    report = run_llm_srbench_comparison(cases)
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change"
              if before["status"] == after["status"] == "verified" else "invalidated")
    report["source"] = cases[0]["source"]
    report["confirmation_protocol"] = {
        "status": status, "seed": args.seed, "case_commitment": commitment,
        "selected_instance_ids": list(selected), "freeze": freeze,
        "freeze_before": before, "freeze_after": after,
        "scope": "community_mirror;hash_selection_before_truth_load;public_pretraining_exposure_unknown;not_official_dataset_snapshot",
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
