"""One-shot, fixed-version risk and cost audit across declared task strata.

The 39 compiler tasks and 10 product-entry tasks have different interfaces and
are reported separately. All generating structures were known before freeze;
this is a mixed regression/risk audit, not a population accuracy estimate.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation_freeze import create_evaluation_freeze, verify_evaluation_freeze
from core.modeling_benchmark_suite import current_problem_compiler_adapter, score_modeling_benchmark_case
from core.modeling_confirmation_suite import (
    ModelingConfirmationAlreadyConsumed, build_modeling_confirmation_suite,
    complete_modeling_confirmation, reserve_modeling_confirmation,
)
from core.product_rescue_confirmation import build_product_rescue_confirmation
from core.product_routing_confirmation import build_product_routing_confirmation
from core.product_workflow_confirmation import (
    run_product_workflow_case, score_product_workflow_output, summarize_product_workflow_scores,
)


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-version mixed modeling risk audit")
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", default="artifacts/product-workflow-confirmation/mixed-risk-20261020.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    args = parser.parse_args()
    protocol = "mixed-risk-confirmation-v26"
    try:
        marker = reserve_modeling_confirmation(args.registry_dir, protocol_id=protocol, seed=args.seed)
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({"status": "refused", "reason": str(exc)}))
        return 3
    freeze = create_evaluation_freeze(
        ROOT, protocol_id=protocol, seed=args.seed,
        budget={"compiler_case_count": 39, "product_case_count": 10,
                "product_solver_arm_ceiling": 2, "model_api_calls": 0, "manual_interventions": 0},
        methods=("current-raw-compiler/v1", "product-on-demand/v1",
                 "product-always-both/v1", "product-single-solver/v1"),
    )
    before = verify_evaluation_freeze(ROOT, freeze)
    compiler_cases = build_modeling_confirmation_suite(seed=args.seed)
    product_cases = (*build_product_routing_confirmation(seed=args.seed),
                     *build_product_rescue_confirmation(seed=args.seed)[:2])
    commitment = sha256(_canonical({
        "compiler": [{"id": case.case_id, "public": case.public_input,
                      "reference": case.hidden_reference} for case in compiler_cases],
        "product": [{"id": case.case_id, "public": case.public_payload(),
                     "reference": case.hidden_reference()} for case in product_cases],
    }).encode("utf-8")).hexdigest()

    compiler_rows = []
    for case in compiler_cases:
        public = {"id": case.case_id, "family": case.family,
                  "statement": case.statement, "input": case.public_input}
        started = time.monotonic()
        try:
            output = current_problem_compiler_adapter(public, {})
            scored = score_modeling_benchmark_case(output, case.hidden_reference)
            covered = isinstance(output.get("model"), dict) and any(
                key in output for key in ("predictions", "trajectory", "solution", "group_totals"))
            compiler_rows.append({
                "case_id": case.case_id, "family": case.family,
                "structure_group": case.structure_group, "status": output.get("status", "completed"),
                "valid": bool(scored["valid"]), "score": scored.get("score"),
                "reason": scored.get("reason"), "covered": covered,
                "invalid_accepted": covered and not scored["valid"],
                "false_abstain": output.get("status") == "needs_input" and not scored["valid"],
                "duration_seconds": time.monotonic() - started,
            })
        except Exception as exc:
            compiler_rows.append({
                "case_id": case.case_id, "family": case.family,
                "structure_group": case.structure_group, "status": "error",
                "valid": False, "covered": False, "invalid_accepted": False,
                "false_abstain": False, "reason": type(exc).__name__,
                "duration_seconds": time.monotonic() - started,
            })

    product_rows = []
    for arm, portfolio, policy in (
        ("on_demand", True, "current_then_fallback"),
        ("always_both", True, "evaluate_both"),
        ("single_solver", False, "current_then_fallback"),
    ):
        for case in product_cases:
            try:
                output = run_product_workflow_case(case.public_payload(), portfolio=portfolio,
                                                   routing_policy=policy)
                scored = score_product_workflow_output(output, case.hidden_reference())
                product_rows.append({"case_id": case.case_id, "structure_group": case.structure_group,
                                     "arm": arm, "expected_status": case.expected_status,
                                     **output, **scored})
            except Exception as exc:
                product_rows.append({"case_id": case.case_id, "structure_group": case.structure_group,
                                     "arm": arm, "expected_status": case.expected_status,
                                     "status": "error", "reason": type(exc).__name__,
                                     "duration_seconds": 0.0, "valid": False, "covered": False,
                                     "accepted_correct": False, "abstained": False,
                                     "validated_accepted": False, "out_of_scope_accept": False,
                                     "incorrect_prediction_accept": False, "false_abstain": False,
                                     "unsafe_accept": False})

    by_family = defaultdict(list)
    for row in compiler_rows:
        by_family[row["family"]].append(row)
    compiler_summary = {
        family: {"task_count": len(rows), "structure_group_count": len({r["structure_group"] for r in rows}),
                 "valid_count": sum(bool(r["valid"]) for r in rows),
                 "accepted_count": sum(bool(r["covered"]) for r in rows),
                 "invalid_accepted_count": sum(bool(r["invalid_accepted"]) for r in rows),
                 "false_abstain_count": sum(bool(r["false_abstain"]) for r in rows),
                 "duration_seconds": sum(float(r["duration_seconds"]) for r in rows)}
        for family, rows in sorted(by_family.items())
    }
    product_summary = {}
    for arm in ("on_demand", "always_both", "single_solver"):
        rows = [row for row in product_rows if row["arm"] == arm]
        product_summary[arm] = {
            **summarize_product_workflow_scores(rows),
            "structure_group_count": len({row["structure_group"] for row in rows}),
            "solver_arm_launch_count": (sum(int((row.get("usage") or {}).get("numerical_solver_calls", 0))
                                            for row in rows) if arm != "single_solver" else None),
            "reported_numerical_solver_calls": sum(
                int((row.get("usage") or {}).get("numerical_solver_calls", 0)) for row in rows),
            "duration_seconds": sum(float(row["duration_seconds"]) for row in rows),
        }
    after = verify_evaluation_freeze(ROOT, freeze)
    status = ("completed_without_source_change" if before["status"] == after["status"] == "verified"
              else "invalidated_by_source_change")
    report = {
        "schema_version": "mathmodel.mixed-risk-confirmation/v1", "status": status,
        "seed": args.seed, "case_commitment": commitment,
        "scope": "known_structure_mixed_regression;compiler_and_product_strata_not_pooled_as_population_accuracy",
        "compiler_summary": compiler_summary, "product_summary": product_summary,
        "compiler_rows": compiler_rows, "product_rows": product_rows,
        "freeze": freeze, "freeze_before": before, "freeze_after": after,
        "policy": "reserve_and_freeze_before_case_generation;all_failures_retained;historical_reports_not_rewritten",
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(marker, outcome="completed" if status.startswith("completed") else "failed",
                                   report_path=str(destination))
    print(json.dumps({"status": status, "compiler_summary": compiler_summary,
                      "product_summary": product_summary, "output": str(destination)}, ensure_ascii=False))
    return 0 if status.startswith("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
