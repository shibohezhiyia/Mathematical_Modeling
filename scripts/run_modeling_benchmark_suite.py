"""Run the bundled raw-input modeling benchmark against three fixed proxies."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.modeling_benchmark_suite import (
    current_problem_compiler_adapter,
    legacy_typed_executor_adapter,
    isolated_baseline_adapter,
    run_three_arm_modeling_comparison,
    run_automatic_modeling_ablation,
    run_modeling_scorer_attack_audit,
    simple_tool_modeling_adapter,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run prose/raw-attachment modeling comparison")
    parser.add_argument("--output", default="artifacts/modeling-benchmark/latest.json")
    args = parser.parse_args()
    def digest(paths):
        value = sha256()
        for path in paths:
            value.update(path.read_bytes())
        return value.hexdigest()

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
    }, system_versions={
        "frozen_old": {"version_id": "typed-only-proxy/v1",
                       "source_digest": digest([suite_path, *supervision_paths])},
        "candidate_new": {"version_id": "bounded-raw-induction/v1",
                          "source_digest": digest([suite_path, compiler_path, induction_path,
                                                   *supervision_paths])},
        "simple_tool_baseline": {"version_id": "transparent-tools/v1",
                                 "source_digest": digest([suite_path, *supervision_paths])},
    })
    report["development_ablation"] = run_automatic_modeling_ablation()
    report["scorer_attack_audit"] = run_modeling_scorer_attack_audit()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "arm_summary": report["arm_summary"],
                      "ablation_summary": report["development_ablation"]["summary"],
                      "scorer_attack_audit": {
                          key: report["scorer_attack_audit"][key] for key in (
                              "status", "control_count", "control_failure_count",
                              "attack_count", "false_accept_count", "attack_detection_rate"
                          )
                      },
                      "output": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
