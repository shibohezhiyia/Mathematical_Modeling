"""Run the reproducible public external-data benchmark."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.external_data_benchmark import run_external_data_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description="Run external data benchmark with fixed splits")
    parser.add_argument("--catalog", type=Path, default=ROOT / "examples" / "external_dataset_catalog.json")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "external")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports" / "external_data_benchmark.json")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--include-current-system", action="store_true",
                        help="在同一冻结切分上运行当前 ModelingEngine（较慢）")
    args = parser.parse_args()
    result = run_external_data_benchmark(args.catalog, args.data_root, seed=args.seed,
                                         include_current_system=args.include_current_system)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "case_count": result["case_count"], "completed_count": result["completed_count"],
                      "failed_count": result["failed_count"],
                      "evaluation_status": result["evaluation_status"],
                      "paired_relative_rmse_effect": result["paired_relative_rmse_effect_treatment_minus_baseline"],
                      "permutation_p_two_sided": result["paired_relative_rmse_statistics"]["permutation_p_two_sided"]}, ensure_ascii=False))
    return 0 if result["completed_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
