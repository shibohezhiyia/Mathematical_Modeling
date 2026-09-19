"""Run the bounded LLM-SRBench community-mirror development pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.llm_srbench_adapter import load_llm_srbench_cases, run_llm_srbench_pilot


DEFAULT_IDS = (
    "lsr_transform_ii.27.18_1_0", "lsr_transform_iii.12.43_0_0",
    "lsr_transform_ii.11.28_0_0", "lsr_transform_ii.8.31_1_0",
    "lsr_transform_ii.3.24_1_0",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", default="artifacts/llm-srbench-pilot/community-mirror-development.json")
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--instance-id", action="append", dest="instance_ids")
    args = parser.parse_args()
    cases = load_llm_srbench_cases(
        args.dataset_dir, instance_ids=tuple(args.instance_ids or DEFAULT_IDS),
        source_revision=args.source_revision,
        seed=args.seed,
    )
    report = run_llm_srbench_pilot(cases)
    report["source"] = cases[0]["source"]
    report["seed"] = args.seed
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "summary": report["summary"],
                      "output": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
