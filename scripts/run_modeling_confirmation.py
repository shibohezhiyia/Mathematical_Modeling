"""Freeze code first, then run the internal modeling confirmation pack once."""

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
from core.modeling_benchmark_suite import (
    current_problem_compiler_adapter,
    legacy_typed_executor_adapter,
    isolated_baseline_adapter,
    run_three_arm_modeling_comparison,
    simple_tool_modeling_adapter,
)
from core.modeling_confirmation_suite import (
    ModelingConfirmationAlreadyConsumed,
    build_modeling_confirmation_suite,
    complete_modeling_confirmation,
    reserve_modeling_confirmation,
)


def _digest(paths: list[Path]) -> str:
    value = sha256()
    for path in paths:
        value.update(path.read_bytes())
    return value.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one frozen internal modeling confirmation")
    parser.add_argument("--output", default="artifacts/modeling-confirmation/latest.json")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--registry-dir", default="artifacts/modeling-confirmation/consumed")
    args = parser.parse_args()
    budget = {"per_case_wall_seconds": 30, "max_model_api_calls": 0,
              "max_numerical_solver_calls": 2, "max_manual_interventions": 1}
    methods = ("typed-only-proxy/v1", "bounded-raw-induction/v1", "transparent-tools/v1")

    try:
        marker = reserve_modeling_confirmation(
            args.registry_dir, protocol_id="modeling-internal-confirmation-v1", seed=args.seed,
        )
    except ModelingConfirmationAlreadyConsumed as exc:
        print(json.dumps({"status": "refused", "reason": str(exc), "seed": args.seed}))
        return 3

    # This call must precede construction of the confirmation cases.
    freeze = create_evaluation_freeze(
        ROOT, protocol_id="modeling-internal-confirmation-v1", seed=args.seed,
        budget=budget, methods=methods,
    )
    freeze_before = verify_evaluation_freeze(ROOT, freeze)
    cases = build_modeling_confirmation_suite(seed=args.seed)
    suite_metadata = [case.public_metadata() for case in cases]
    suite_commitment = sha256(json.dumps(
        suite_metadata, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()

    suite_path = ROOT / "core" / "modeling_benchmark_suite.py"
    compiler_path = ROOT / "core" / "dynamic_model_compiler.py"
    induction_path = ROOT / "core" / "automatic_modeling.py"
    supervision_paths = [ROOT / "core" / name for name in (
        "solver_runtime.py", "solver_worker.py", "solver_process_limits.py",
    )]
    report = run_three_arm_modeling_comparison({
        "frozen_old": isolated_baseline_adapter("frozen_old"),
        "candidate_new": current_problem_compiler_adapter,
        "simple_tool_baseline": isolated_baseline_adapter("simple_tool_baseline"),
    }, cases=cases, fixed_budget={**budget, "seed": args.seed}, system_versions={
        "frozen_old": {"version_id": methods[0],
                       "source_digest": _digest([suite_path, *supervision_paths])},
        "candidate_new": {"version_id": methods[1],
                          "source_digest": _digest([suite_path, compiler_path, induction_path,
                                                    *supervision_paths])},
        "simple_tool_baseline": {"version_id": methods[2],
                                 "source_digest": _digest([suite_path, *supervision_paths])},
    })
    freeze_after = verify_evaluation_freeze(ROOT, freeze)
    report["confirmation_protocol"] = {
        "status": "completed_without_source_change" if freeze_before["status"] == freeze_after["status"] == "verified"
                  else "invalidated_by_source_change",
        "scope": "internal_parameter_and_representation_confirmation;not_unseen_structure_or_external_domain",
        "seed": args.seed,
        "case_count": len(cases),
        "structure_group_count": len({case.structure_group for case in cases}),
        "suite_commitment": suite_commitment,
        "freeze": freeze,
        "freeze_before": freeze_before,
        "freeze_after": freeze_after,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(
        marker, outcome="completed" if report["confirmation_protocol"]["status"] == "completed_without_source_change"
        else "failed", report_path=str(destination),
    )
    print(json.dumps({"status": report["status"], "confirmation": report["confirmation_protocol"],
                      "arm_summary": report["arm_summary"], "output": str(destination)}, ensure_ascii=False))
    return 0 if report["confirmation_protocol"]["status"] == "completed_without_source_change" else 2


if __name__ == "__main__":
    raise SystemExit(main())
