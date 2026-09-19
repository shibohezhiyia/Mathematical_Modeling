"""Run a development-only pilot on the official SRSD-Feynman easy split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.srsd_adapter import load_srsd_cases, run_srsd_pilot, select_srsd_instances


DEVELOPMENT_IDS = (
    "feynman-i.12.1", "feynman-ii.15.4", "feynman-ii.10.9", "feynman-iii.12.43",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--seed", type=int, default=20261008)
    parser.add_argument("--case-count", type=int, default=4)
    parser.add_argument("--output", default="artifacts/srsd-pilot/official-easy-development.json")
    args = parser.parse_args()
    selected = select_srsd_instances(args.dataset_dir, count=args.case_count, seed=args.seed)
    cases = load_srsd_cases(args.dataset_dir, instance_ids=selected,
                            source_revision=args.source_revision, seed=args.seed)
    report = run_srsd_pilot(cases)
    report["selected_instance_ids"] = list(selected)
    report["source"] = cases[0]["source"]
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"selected": selected, "summary": report["summary"],
                      "output": str(destination)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
