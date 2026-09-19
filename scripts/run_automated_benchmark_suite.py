"""Run the repository's deterministic typed execution benchmark.

The output is an engineering regression artifact.  It is deliberately not
reported as real-world accuracy or as evidence of open-world generalisation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.automated_benchmark_suite import run_typed_execution_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the separated generator/system/scorer benchmark")
    parser.add_argument("--cases-per-family", type=int, default=8, choices=range(1, 33))
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--wall-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/automated-benchmark/latest.json"))
    args = parser.parse_args()
    report = run_typed_execution_benchmark(cases_per_family=args.cases_per_family,
                                            seed=args.seed, wall_seconds=args.wall_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "case_count", "completed_count", "valid_count", "valid_rate", "policy")},
                     ensure_ascii=False))
    return 0 if report["status"] == "completed" and report["valid_count"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
