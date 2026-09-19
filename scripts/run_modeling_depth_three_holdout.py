"""Freeze and run the depth-three versus depth-two grammar comparison once."""
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
from core.modeling_benchmark_suite import run_compositional_depth_three_comparison
from core.modeling_confirmation_suite import reserve_modeling_confirmation, complete_modeling_confirmation
from core.modeling_depth_three_holdout import build_depth_three_holdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--output", default="artifacts/modeling-depth-three-holdout/confirmation-20260929.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    args = parser.parse_args()
    protocol = "modeling-depth-three-holdout-v10"
    marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    budget = {"max_wall_seconds_per_case": 30, "memory_mb": 1024,
              "depth_three_topology_budget": 85, "depth_two_topology_budget": 21,
              "parameter_max_nfev": 1200}
    methods = ("exhaustive-depth-three/v10", "exhaustive-depth-two/v10")
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed, budget=budget, methods=methods,
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = build_depth_three_holdout(seed=args.seed)
    commitment = sha256(json.dumps(
        [case.public_metadata() for case in cases], sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    report = run_compositional_depth_three_comparison(
        cases=cases, wall_seconds=30.0, memory_mb=1024,
    )
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change"
              if before["status"] == after["status"] == "verified" else "invalidated")
    report["confirmation_protocol"] = {
        "status": status, "seed": args.seed, "suite_commitment": commitment,
        "freeze": freeze, "freeze_before": before, "freeze_after": after,
        "case_count": len(cases),
        "structure_group_count": len({case.structure_group for case in cases}),
        "scope": "first_execution_after_freeze;new_depth_three_topologies;isolated_resource_supervision;depth_budgets_intentionally_differ",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(
        marker, outcome="completed" if status.startswith("completed") else "failed",
        report_path=str(destination),
    )
    print(json.dumps({"status": status, "summary": report["summary"],
                      "effect": report["paired_effect"], "output": str(destination)},
                     ensure_ascii=False))
    return 0 if status.startswith("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
