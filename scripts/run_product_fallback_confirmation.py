"""One-shot frozen comparison of on-demand and always-both product routing."""
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
from core.product_fallback_confirmation import build_product_fallback_confirmation
from core.product_rescue_confirmation import build_product_rescue_confirmation
from core.product_workflow_confirmation import run_product_workflow_case, score_product_workflow_output


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main(*, suite: str = "fallback") -> int:
    if suite not in {"fallback", "rescue"}:
        raise ValueError("product_confirmation_suite_invalid")
    parser = argparse.ArgumentParser(description="Run prospective second-solver rescue confirmation")
    parser.add_argument("--output", default=("artifacts/product-workflow-confirmation/rescue-20261018.json"
                                             if suite == "rescue" else
                                             "artifacts/product-workflow-confirmation/fallback-20261017.json"))
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    parser.add_argument("--seed", type=int, default=20261018 if suite == "rescue" else 20261017)
    args = parser.parse_args()
    protocol = ("product-rescue-confirmation-v25" if suite == "rescue"
                else "product-fallback-confirmation-v24")
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({"status": "refused", "reason": str(exc), "seed": args.seed}))
        return 3
    budget = {"case_count": 4, "solver_call_ceiling_per_case": 2,
              "model_api_calls": 0, "manual_interventions": 0}
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed, budget=budget,
        methods=("current-then-fallback/v1", "always-run-both/v1"),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = (build_product_rescue_confirmation(seed=args.seed) if suite == "rescue"
             else build_product_fallback_confirmation(seed=args.seed))
    commitment = sha256(_canonical([
        {"case_id": case.case_id, "group": case.structure_group,
         "public": case.public_payload(), "reference": case.hidden_reference()}
        for case in cases
    ]).encode("utf-8")).hexdigest()
    rows = []
    for arm, policy in (("on_demand", "current_then_fallback"),
                        ("always_both", "evaluate_both")):
        for case in cases:
            output = run_product_workflow_case(case.public_payload(), portfolio=True,
                                               routing_policy=policy)
            scored = score_product_workflow_output(output, case.hidden_reference())
            route = output.get("routing_evidence") or {}
            metrics = route.get("validation_metrics") or {}
            first = metrics.get("current_bounded_grammar") or {}
            second = metrics.get("official_gplearn") or {}
            rows.append({"case_id": case.case_id, "structure_group": case.structure_group,
                         "arm": arm, "first_passed": first.get("passes_primary_rule") is True,
                         "second_passed": second.get("passes_primary_rule") is True,
                         "selected_arm": route.get("selected_arm"), **output, **scored})

    def summary(arm):
        selected = [row for row in rows if row["arm"] == arm]
        return {
            "case_count": len(selected), "valid_count": sum(bool(row["valid"]) for row in selected),
            "covered_count": sum(bool(row["covered"]) for row in selected),
            "abstain_count": sum(bool(row["abstained"]) for row in selected),
            "unsafe_accept_count": sum(bool(row["unsafe_accept"]) for row in selected),
            "second_arm_started_count": sum(int((row.get("usage") or {}).get("numerical_solver_calls", 0)) == 2
                                            for row in selected),
            "second_arm_rescue_count": sum(not row["first_passed"] and row["second_passed"]
                                           and row["selected_arm"] == "official_gplearn" and row["valid"]
                                           for row in selected),
            "solver_arm_launch_count": sum(int((row.get("usage") or {}).get("numerical_solver_calls", 0))
                                           for row in selected),
            "duration_seconds": sum(float(row["duration_seconds"]) for row in selected),
        }
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
    report = {
        "schema_version": "mathmodel.product-fallback-confirmation/v1", "status": status,
        "scope": ("fresh_domain_and_record_holdout_for_development_selected_sqrt_absolute_topology;"
                  "not_unseen_topology" if suite == "rescue" else
                  "fresh_parameter_and_record_holdout_for_known_product_sine_topology;not_unseen_topology"),
        "seed": args.seed, "case_commitment": commitment,
        "summary": {"on_demand": summary("on_demand"), "always_both": summary("always_both")},
        "paired_effect": paired, "rows": rows, "freeze": freeze,
        "freeze_before": before, "freeze_after": after,
        "policy": "reserve_and_freeze_before_case_construction;same_two_arm_ceiling;failures_retained",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(marker,
                                   outcome="completed" if status == "completed_without_source_change" else "failed",
                                   report_path=str(destination))
    print(json.dumps({"status": status, "summary": report["summary"],
                      "paired_effect": paired, "output": str(destination)}, ensure_ascii=False))
    return 0 if status == "completed_without_source_change" else 2


if __name__ == "__main__":
    raise SystemExit(main())
