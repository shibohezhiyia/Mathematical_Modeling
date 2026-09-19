"""Freeze once, then compare on-demand routing with an always-run-both ablation."""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmark_statistics import paired_benchmark_effect
from core.evaluation_freeze import create_evaluation_freeze, verify_evaluation_freeze
from core.modeling_confirmation_suite import (
    ModelingConfirmationAlreadyConsumed, complete_modeling_confirmation, reserve_modeling_confirmation,
)
from core.product_routing_confirmation import build_product_routing_confirmation
from core.product_workflow_confirmation import run_product_workflow_case, score_product_workflow_output


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen product routing comparison")
    parser.add_argument("--output", default="artifacts/product-workflow-confirmation/routing-20261016.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    protocol = "product-routing-confirmation-v23"
    budget = {"case_count": 8, "solver_call_ceiling_per_case": 2,
              "model_api_calls": 0, "manual_interventions": 0}
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({"status": "refused", "reason": str(exc), "seed": args.seed}))
        return 3
    freeze = create_evaluation_freeze(ROOT, protocol_id=protocol, seed=args.seed, budget=budget,
                                      methods=("current-then-fallback/v1", "always-run-both/v1"))
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = build_product_routing_confirmation(seed=args.seed)
    commitment = sha256(_canonical([
        {"case_id": case.case_id, "group": case.structure_group,
         "public": case.public_payload(), "reference": case.hidden_reference()}
        for case in cases
    ]).encode("utf-8")).hexdigest()
    rows = []
    for arm, policy in (("on_demand", "current_then_fallback"), ("always_both", "evaluate_both")):
        for case in cases:
            output = run_product_workflow_case(case.public_payload(), portfolio=True,
                                               routing_policy=policy)
            scored = score_product_workflow_output(output, case.hidden_reference())
            if case.expected_status == "needs_input":
                scored["valid"] = bool(scored["valid"] and output.get("result_grade") == "abstain"
                                       and output.get("reason") == "portfolio_transition_region_underobserved")
            rows.append({"case_id": case.case_id, "structure_group": case.structure_group,
                         "expected_status": case.expected_status, "arm": arm, **output, **scored})

    def summary(arm):
        selected = [row for row in rows if row["arm"] == arm]
        return {"case_count": len(selected), "valid_count": sum(row["valid"] for row in selected),
                "covered_count": sum(row["covered"] for row in selected),
                "abstain_count": sum(row["abstained"] for row in selected),
                "unsafe_accept_count": sum(row["unsafe_accept"] for row in selected),
                "solver_call_count": sum(int((row.get("usage") or {}).get("numerical_solver_calls", 0))
                                         for row in selected),
                "duration_seconds": sum(float(row["duration_seconds"]) for row in selected)}
    valid = {(row["arm"], row["case_id"]): bool(row["valid"]) for row in rows}
    by_id = {case.case_id: case for case in cases}
    paired = paired_benchmark_effect([
        {"structure_group": by_id[case_id].structure_group,
         "baseline_valid": valid[("always_both", case_id)],
         "candidate_valid": valid[("on_demand", case_id)]}
        for case_id in sorted(by_id)
    ], baseline="baseline_valid", treatment="candidate_valid", cluster="structure_group",
       seed=args.seed, bootstrap_replicates=1000)
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change" if before["status"] == after["status"] == "verified"
              else "invalidated_by_source_change")
    report = {"schema_version": "mathmodel.product-routing-confirmation/v1", "status": status,
              "scope": "single_table_transition_uncertainty_and_on_demand_cost;not_general_accuracy",
              "seed": args.seed, "case_commitment": commitment,
              "summary": {"on_demand": summary("on_demand"), "always_both": summary("always_both")},
              "paired_effect": paired, "rows": rows,
              "freeze": freeze, "freeze_before": before, "freeze_after": after,
              "policy": "freeze_before_case_construction;same_two_solver_ceiling;abstentions_retained"}
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(marker, outcome="completed" if status == "completed_without_source_change" else "failed",
                                   report_path=str(destination))
    print(json.dumps({"status": status, "summary": report["summary"],
                      "paired_effect": paired, "output": str(destination)}, ensure_ascii=False))
    return 0 if status == "completed_without_source_change" else 2


if __name__ == "__main__":
    raise SystemExit(main())
