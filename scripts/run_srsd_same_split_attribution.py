"""Separate training-split, solver, and routing effects on consumed v20 cases."""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.benchmark_statistics import paired_benchmark_effect
from core.llm_srbench_adapter import run_llm_srbench_pilot
from core.srsd_adapter import load_srsd_cases
from core.symbolic_portfolio import run_same_split_portfolio_arm
from scripts.run_srsd_portfolio_confirmation import V20_IDS


def _valid(row):
    return bool(row["nmse"] is not None and row["nmse"] <= 0.01
                and row["acc_0.1"] is not None and row["acc_0.1"] >= 0.9
                and row["model_reexecution_status"] == "verified"
                and row["model_prediction_consistent"])


def _rank(row):
    metrics = row["submitted_model"]["same_split_attribution"]["validation_metrics"]
    return (0 if metrics["passes_primary_rule"] else 1,
            float(metrics["nmse"]), -float(metrics["acc_0.1"]), row["arm"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", default="artifacts/srsd-pilot/same-split-attribution-v20.json")
    args = parser.parse_args()
    cases = load_srsd_cases(args.dataset_dir, instance_ids=V20_IDS,
                            source_revision=args.source_revision, seed=args.seed)
    rows = []
    for case in cases:
        case_rows = []
        for arm in ("current_80pct", "gplearn_80pct"):
            solver_arm = "current_bounded_grammar" if arm == "current_80pct" else "official_gplearn"
            report = run_llm_srbench_pilot(
                [case], solver=lambda payload, selected=solver_arm: run_same_split_portfolio_arm(
                    payload, arm=selected, seed=args.seed),
                status="development_on_consumed_v20_cases",
                selection_policy="same_fixed_80_20_training_split;not_confirmation",
                source_policy="official_srsd_snapshot;consumed_v20_cases",
            )
            row = {**report["rows"][0], "arm": arm}
            row["valid"] = _valid(row)
            case_rows.append(row)
            rows.append(row)
        selected = min(case_rows, key=_rank)
        selected_metrics = selected["submitted_model"]["same_split_attribution"]["validation_metrics"]
        if selected_metrics["passes_primary_rule"]:
            routed = deepcopy(selected)
            routed.update(arm="validation_router_80pct", result_grade="validated_candidate",
                          recommended_action="use_with_stated_scope",
                          selected_source_arm=selected["arm"],
                          execution_elapsed_seconds=float(sum(
                              row["execution_elapsed_seconds"] or 0 for row in case_rows)))
        else:
            routed = {
                "instance_id": case["instance_id"], "subset": case["subset"],
                "arm": "validation_router_80pct", "status": "failed",
                "reason": "portfolio_no_validated_candidate", "nmse": None, "acc_0.1": None,
                "valid": False, "model_present": False, "model_structure": None,
                "model_digest": None, "submitted_model": None,
                "model_reexecution_status": "not_assessed",
                "model_prediction_consistent": False,
                "model_prediction_max_abs_error": None, "resource_supervised": True,
                "execution_elapsed_seconds": float(sum(row["execution_elapsed_seconds"] or 0
                                                       for row in case_rows)),
                "memory_limit_mb": 1024, "memory_backend": case_rows[0]["memory_backend"],
                "model_api_calls": 0, "manual_interventions": 0,
                "output_status": "needs_input", "result_grade": "abstain",
                "recommended_action": "collect_more_observations_or_expand_declared_model_scope",
                "ground_truth_expression": case["hidden_reference"]["ground_truth_expression"],
                "selected_source_arm": None,
            }
        rows.append(routed)

    arms = ("current_80pct", "gplearn_80pct", "validation_router_80pct")
    summary = {}
    for arm in arms:
        arm_rows = [row for row in rows if row["arm"] == arm]
        numeric = [row for row in arm_rows if row["nmse"] is not None]
        accepted = [row for row in arm_rows if row.get("result_grade") != "abstain"]
        summary[arm] = {
            "case_count": len(arm_rows), "valid_count": sum(row["valid"] for row in arm_rows),
            "abstain_count": sum(row.get("result_grade") == "abstain" for row in arm_rows),
            "coverage_rate": len(accepted) / len(arm_rows),
            "conditional_valid_rate": (sum(row["valid"] for row in accepted) / len(accepted)
                                       if accepted else None),
            "mean_nmse_on_covered": (float(np.mean([row["nmse"] for row in numeric]))
                                      if numeric else None),
            "elapsed_seconds": float(sum(row["execution_elapsed_seconds"] or 0 for row in arm_rows)),
        }
    def effect(treatment, baseline):
        samples = []
        for case in cases:
            indexed = {row["arm"]: row for row in rows if row["instance_id"] == case["instance_id"]}
            samples.append({"baseline": float(indexed[baseline]["valid"]),
                            "treatment": float(indexed[treatment]["valid"]),
                            "structure_group": case["instance_id"]})
        return paired_benchmark_effect(samples, cluster="structure_group", min_samples=5)
    report = {
        "schema_version": "mathmodel.srsd-same-split-attribution/v1",
        "status": "development_on_consumed_v20_cases", "rows": rows, "summary": summary,
        "paired_effects": {
            "router_minus_current80": effect("validation_router_80pct", "current_80pct"),
            "router_minus_gplearn80": effect("validation_router_80pct", "gplearn_80pct"),
            "gplearn80_minus_current80": effect("gplearn_80pct", "current_80pct"),
        },
        "policy": (
            "identical_training_and_validation_indices_across_arms;hidden_test_not_used_for_routing;"
            "consumed_v20_cases;development_attribution_only;abstentions_remain_in_denominator"),
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "effects": report["paired_effects"],
                      "output": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
