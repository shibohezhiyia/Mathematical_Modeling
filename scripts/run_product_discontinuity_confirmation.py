"""Freeze and compare the discontinuity gate with its pre-gate ablation."""
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
    ModelingConfirmationAlreadyConsumed, complete_modeling_confirmation,
    reserve_modeling_confirmation,
)
from core.product_discontinuity_confirmation import build_product_discontinuity_confirmation
from core.product_workflow_confirmation import run_product_workflow_case, score_product_workflow_output


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen discontinuity-gate confirmation")
    parser.add_argument("--output", default="artifacts/product-workflow-confirmation/discontinuity-20261015.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    parser.add_argument("--seed", type=int, default=20261015)
    args = parser.parse_args()
    protocol_id = "product-discontinuity-confirmation-v22"
    budget = {"case_count": 8, "solver_call_ceiling_per_case": 2,
              "model_api_calls": 0, "manual_interventions": 0}
    methods = ("ordinary-entry-discontinuity-gate/v1", "ordinary-entry-gate-disabled/v1")
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol_id, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({"status": "refused", "reason": str(exc), "seed": args.seed}))
        return 3

    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol_id, seed=args.seed, budget=budget, methods=methods)
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = build_product_discontinuity_confirmation(seed=args.seed)
    commitment = sha256(_canonical([
        {"case_id": case.case_id, "group": case.structure_group,
         "public": case.public_payload(), "reference": case.hidden_reference()}
        for case in cases
    ]).encode("utf-8")).hexdigest()
    rows = []
    for arm, gate in (("discontinuity_gate", True), ("gate_disabled", False)):
        for case in cases:
            output = run_product_workflow_case(
                case.public_payload(), portfolio=True, discontinuity_gate=gate)
            scored = score_product_workflow_output(output, case.hidden_reference())
            rows.append({"case_id": case.case_id, "structure_group": case.structure_group,
                         "expected_status": case.expected_status, "arm": arm, **output, **scored})

    def summary(arm):
        selected = [row for row in rows if row["arm"] == arm]
        return {"case_count": len(selected), "valid_count": sum(row["valid"] for row in selected),
                "covered_count": sum(row["covered"] for row in selected),
                "abstain_count": sum(row["abstained"] for row in selected),
                "unsafe_accept_count": sum(row["unsafe_accept"] for row in selected),
                "false_abstain_count": sum(row["abstained"] and row["expected_status"] == "completed"
                                           for row in selected),
                "duration_seconds": sum(float(row["duration_seconds"]) for row in selected)}
    by_id = {case.case_id: case for case in cases}
    valid = {(row["arm"], row["case_id"]): bool(row["valid"]) for row in rows}
    paired = paired_benchmark_effect([
        {"structure_group": by_id[case_id].structure_group,
         "baseline_valid": valid[("gate_disabled", case_id)],
         "candidate_valid": valid[("discontinuity_gate", case_id)]}
        for case_id in sorted(by_id)
    ], baseline="baseline_valid", treatment="candidate_valid",
       cluster="structure_group", seed=args.seed, bootstrap_replicates=1000)
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change"
              if before["status"] == after["status"] == "verified"
              else "invalidated_by_source_change")
    report = {
        "schema_version": "mathmodel.product-discontinuity-confirmation/v1",
        "status": status, "seed": args.seed,
        "scope": "exact_piecewise_constant_jump_gate_and_smooth_controls;not_general_discontinuity_detection",
        "case_commitment": commitment,
        "summary": {"discontinuity_gate": summary("discontinuity_gate"),
                    "gate_disabled": summary("gate_disabled")},
        "paired_effect": paired, "rows": rows,
        "freeze": freeze, "freeze_before": before, "freeze_after": after,
        "policy": "freeze_before_case_construction;same_budget_ceiling;all_failures_retained",
    }
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
