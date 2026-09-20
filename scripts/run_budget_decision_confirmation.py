"""Freeze once, then compare one-arm, conditional-two-arm, and always-two-arm decisions."""
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
from core.product_rescue_confirmation import build_product_rescue_confirmation
from core.product_routing_confirmation import build_product_routing_confirmation
from core.product_workflow_confirmation import (
    run_product_workflow_case, score_product_workflow_output, summarize_product_workflow_scores,
)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--output", default="artifacts/product-workflow-confirmation/budget-decision-20261021.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    args = parser.parse_args()
    protocol = "budget-decision-confirmation-v27"
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={"case_count": 10, "maximum_solver_arms": 2,
                "model_api_calls": 0, "manual_interventions": 0},
        methods=("validation-budget-one/v1", "validation-conditional-two/v1",
                 "validation-always-two/v1"),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = (*build_product_routing_confirmation(seed=args.seed),
             *build_product_rescue_confirmation(seed=args.seed)[:2])
    commitment = sha256(_canonical([
        {"case_id": case.case_id, "group": case.structure_group,
         "public": case.public_payload(), "reference": case.hidden_reference()}
        for case in cases
    ]).encode("utf-8")).hexdigest()
    rows = []
    for arm, policy, budget in (
        ("one_arm", "current_then_fallback", 1),
        ("conditional_two", "current_then_fallback", 2),
        ("always_two", "evaluate_both", 2),
    ):
        for case in cases:
            try:
                output = run_product_workflow_case(case.public_payload(), portfolio=True,
                                                   routing_policy=policy, solver_arm_budget=budget)
                scored = score_product_workflow_output(output, case.hidden_reference())
                rows.append({"case_id": case.case_id, "structure_group": case.structure_group,
                             "expected_status": case.expected_status, "arm": arm, **output, **scored})
            except Exception as exc:
                rows.append({"case_id": case.case_id, "structure_group": case.structure_group,
                             "expected_status": case.expected_status, "arm": arm,
                             "status": "error", "reason": type(exc).__name__, "valid": False,
                             "covered": False, "accepted_correct": False, "abstained": False,
                             "validated_accepted": False, "out_of_scope_accept": False,
                             "incorrect_prediction_accept": False, "false_abstain": False,
                             "unsafe_accept": False, "duration_seconds": 0.0})
    summary = {}
    for arm in ("one_arm", "conditional_two", "always_two"):
        selected = [row for row in rows if row["arm"] == arm]
        summary[arm] = {
            **summarize_product_workflow_scores(selected),
            "structure_group_count": len({row["structure_group"] for row in selected}),
            "solver_arm_launch_count": sum(int((row.get("usage") or {}).get("numerical_solver_calls", 0))
                                           for row in selected),
            "duration_seconds": sum(float(row["duration_seconds"]) for row in selected),
        }
    valid = {(row["arm"], row["case_id"]): bool(row["valid"]) for row in rows}
    by_id = {case.case_id: case for case in cases}
    paired = paired_benchmark_effect([
        {"structure_group": by_id[case_id].structure_group,
         "baseline_valid": valid[("one_arm", case_id)],
         "candidate_valid": valid[("conditional_two", case_id)]}
        for case_id in sorted(by_id)
    ], baseline="baseline_valid", treatment="candidate_valid", cluster="structure_group",
       seed=args.seed, bootstrap_replicates=1000)
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change" if before["status"] == after["status"] == "verified"
              else "invalidated_by_source_change")
    report = {
        "schema_version": "mathmodel.budget-decision-confirmation/v1", "status": status,
        "scope": "known_structure_product_tasks;budget_coverage_tradeoff_not_general_method_superiority",
        "seed": args.seed, "case_commitment": commitment,
        "summary": summary, "paired_effect": paired, "rows": rows,
        "freeze": freeze, "freeze_before": before, "freeze_after": after,
        "policy": "reserve_and_freeze_before_generation;all_failures_and_abstentions_retained",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(marker, outcome="completed" if status.startswith("completed") else "failed",
                                   report_path=str(destination))
    print(json.dumps({"status": status, "summary": summary, "paired_effect": paired,
                      "output": str(destination)}, ensure_ascii=False))
    return 0 if status.startswith("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
