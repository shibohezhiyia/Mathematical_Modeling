"""One-shot frozen confirmation for newly developed bounded structures."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation_freeze import create_evaluation_freeze, verify_evaluation_freeze
from core.modeling_benchmark_suite import current_problem_compiler_adapter, isolated_baseline_adapter, run_three_arm_modeling_comparison
from core.modeling_confirmation_suite import complete_modeling_confirmation, reserve_modeling_confirmation
from core.modeling_extension_development import build_modeling_extension_confirmation_suite


def _digest(paths):
    value = sha256()
    for path in paths:
        value.update(path.read_bytes())
    return value.hexdigest()


def main() -> int:
    protocol, seed = "modeling-extension-confirmation-v1", 20260919
    marker = reserve_modeling_confirmation(
        ROOT / "artifacts" / "modeling-confirmation" / "consumed", protocol_id=protocol, seed=seed,
    )
    budget = {"per_case_wall_seconds": 30, "memory_mb": 1024, "max_model_api_calls": 0,
              "max_numerical_solver_calls": 2, "max_manual_interventions": 1}
    methods = ("typed-only-proxy/v4", "bounded-raw-induction/v4", "transparent-tools/v4")
    freeze = create_evaluation_freeze(ROOT, protocol_id=protocol, seed=seed, budget=budget, methods=methods)
    before = verify_evaluation_freeze(ROOT, freeze)
    cases = build_modeling_extension_confirmation_suite(seed=seed)
    metadata = [case.public_metadata() for case in cases]
    commitment = sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    suite = ROOT / "core" / "modeling_benchmark_suite.py"
    supervision = [ROOT / "core" / name for name in ("solver_runtime.py", "solver_worker.py", "solver_process_limits.py")]
    report = run_three_arm_modeling_comparison({
        "frozen_old": isolated_baseline_adapter("frozen_old"),
        "candidate_new": current_problem_compiler_adapter,
        "simple_tool_baseline": isolated_baseline_adapter("simple_tool_baseline"),
    }, cases=cases, fixed_budget={**budget, "seed": seed}, system_versions={
        "frozen_old": {"version_id": methods[0], "source_digest": _digest([suite, *supervision])},
        "candidate_new": {"version_id": methods[1], "source_digest": _digest([
            suite, ROOT / "core" / "automatic_modeling.py", ROOT / "core" / "dynamic_model_compiler.py", *supervision])},
        "simple_tool_baseline": {"version_id": methods[2], "source_digest": _digest([suite, *supervision])},
    })
    after = verify_evaluation_freeze(ROOT, freeze)
    status = "completed_without_source_change" if before["status"] == after["status"] == "verified" else "invalidated"
    report["confirmation_protocol"] = {"status": status, "seed": seed, "case_count": len(cases),
                                       "structure_group_count": len({case.structure_group for case in cases}),
                                       "suite_commitment": commitment, "freeze": freeze,
                                       "freeze_before": before, "freeze_after": after,
                                       "scope": "new-parameter-and-representation-confirmation-for-developed-extensions;not-newer-structure-generalization"}
    destination = ROOT / "artifacts" / "modeling-extension-confirmation" / "confirmation-20260919.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    complete_modeling_confirmation(marker, outcome="completed" if status.startswith("completed") else "failed",
                                   report_path=str(destination))
    print(json.dumps({"status": status, "arm_summary": report["arm_summary"],
                      "resource_comparison_eligible": report["resource_comparison_eligible"],
                      "output": str(destination)}, ensure_ascii=False))
    return 0 if status.startswith("completed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
