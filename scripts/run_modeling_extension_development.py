"""Run the post-challenge structure-extension development comparison."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.modeling_benchmark_suite import (
    current_problem_compiler_adapter, isolated_baseline_adapter, run_three_arm_modeling_comparison,
)
from core.modeling_extension_development import (
    build_modeling_extension_development_suite, run_modeling_extension_ablation,
)


def _digest(paths):
    value = sha256()
    for path in paths:
        value.update(path.read_bytes())
    return value.hexdigest()


def main() -> int:
    cases = build_modeling_extension_development_suite()
    suite = ROOT / "core" / "modeling_benchmark_suite.py"
    supervision = [ROOT / "core" / name for name in (
        "solver_runtime.py", "solver_worker.py", "solver_process_limits.py",
    )]
    report = run_three_arm_modeling_comparison({
        "frozen_old": isolated_baseline_adapter("frozen_old"),
        "candidate_new": current_problem_compiler_adapter,
        "simple_tool_baseline": isolated_baseline_adapter("simple_tool_baseline"),
    }, cases=cases, fixed_budget={"per_case_wall_seconds": 30, "memory_mb": 1024,
                                  "max_model_api_calls": 0, "max_numerical_solver_calls": 2,
                                  "max_manual_interventions": 1, "seed": 20260918},
       system_versions={
           "frozen_old": {"version_id": "typed-only-proxy/v3", "source_digest": _digest([suite, *supervision])},
           "candidate_new": {"version_id": "bounded-raw-induction/v3", "source_digest": _digest([
               suite, ROOT / "core" / "automatic_modeling.py", ROOT / "core" / "dynamic_model_compiler.py",
               *supervision])},
           "simple_tool_baseline": {"version_id": "transparent-tools/v3", "source_digest": _digest([suite, *supervision])},
       })
    report["development_ablation"] = run_modeling_extension_ablation(cases=cases)
    destination = ROOT / "artifacts" / "modeling-extension-development" / "latest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "arm_summary": report["arm_summary"],
                      "ablation": report["development_ablation"]["summary"],
                      "resource_comparison_eligible": report["resource_comparison_eligible"],
                      "output": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
