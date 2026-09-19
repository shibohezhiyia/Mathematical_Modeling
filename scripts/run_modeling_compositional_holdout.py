"""Freeze then execute unseen operator topologies without a pre-run."""
from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from core.evaluation_freeze import create_evaluation_freeze, verify_evaluation_freeze
from core.modeling_benchmark_suite import current_problem_compiler_adapter, isolated_baseline_adapter, run_three_arm_modeling_comparison
from core.modeling_compositional_holdout import build_compositional_holdout
from core.modeling_confirmation_suite import complete_modeling_confirmation, reserve_modeling_confirmation

def _digest(paths):
    value = sha256()
    for path in paths: value.update(path.read_bytes())
    return value.hexdigest()

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--output", default="artifacts/modeling-compositional-holdout/confirmation-20260922.json")
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed"); args = parser.parse_args()
    marker = reserve_modeling_confirmation(args.registry_dir, protocol_id="modeling-compositional-holdout-v4", seed=args.seed)
    budget = {"per_case_wall_seconds": 30, "memory_mb": 1024, "max_model_api_calls": 0,
              "max_numerical_solver_calls": 2, "max_manual_interventions": 1}
    methods = ("typed-interface-probe/v6", "operator-grammar-induction/v6", "transparent-tools/v6")
    freeze = create_evaluation_freeze(ROOT, protocol_id="modeling-compositional-holdout-v4", seed=args.seed,
                                      budget=budget, methods=methods)
    before = verify_evaluation_freeze(ROOT, freeze); cases = build_compositional_holdout(seed=args.seed)
    commitment = sha256(json.dumps([c.public_metadata() for c in cases], sort_keys=True,
                                   separators=(",", ":")).encode()).hexdigest()
    suite = ROOT / "core" / "modeling_benchmark_suite.py"
    supervision = [ROOT / "core" / n for n in ("solver_runtime.py", "solver_worker.py", "solver_process_limits.py")]
    report = run_three_arm_modeling_comparison({"frozen_old": isolated_baseline_adapter("frozen_old"),
        "candidate_new": current_problem_compiler_adapter,
        "simple_tool_baseline": isolated_baseline_adapter("simple_tool_baseline")}, cases=cases,
        fixed_budget={**budget, "seed": args.seed}, system_versions={
        "frozen_old": {"version_id": methods[0], "source_digest": _digest([suite, *supervision])},
        "candidate_new": {"version_id": methods[1], "source_digest": _digest([suite, ROOT / "core" / "dynamic_model_compiler.py", ROOT / "core" / "automatic_modeling.py", *supervision])},
        "simple_tool_baseline": {"version_id": methods[2], "source_digest": _digest([suite, *supervision])}})
    after = verify_evaluation_freeze(ROOT, freeze)
    status = "completed_without_source_change" if before["status"] == after["status"] == "verified" else "invalidated"
    report["confirmation_protocol"] = {"status": status, "seed": args.seed, "case_count": len(cases),
        "structure_group_count": len({c.structure_group for c in cases}), "suite_commitment": commitment,
        "freeze": freeze, "freeze_before": before, "freeze_after": after,
        "scope": "unseen_operator_topologies_within_predeclared_bounded_grammar;internal_generator"}
    destination = Path(args.output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(marker, outcome="completed" if status.startswith("completed") else "failed", report_path=str(destination))
    print(json.dumps({"status": status, "arm_summary": report["arm_summary"], "output": str(destination)}, ensure_ascii=False))
    return 0 if status.startswith("completed") else 2
if __name__ == "__main__": raise SystemExit(main())
