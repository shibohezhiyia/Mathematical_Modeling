"""List the external benchmark sources approved by the repository policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from core.benchmark_sources import BenchmarkSourceRegistry


def main() -> int:
    parser = argparse.ArgumentParser(description="List registered modeling benchmark sources")
    parser.add_argument("--role", choices=(
        "task_input", "rubric_material", "method_comparison", "reference_only",
        "policy_reference",
    ))
    parser.add_argument("--file", type=Path,
                        help="源目录 JSON；默认使用 examples/benchmark_sources.json")
    args = parser.parse_args()
    registry = BenchmarkSourceRegistry.load(args.file or (ROOT / "examples" / "benchmark_sources.json"))
    rows = registry.sources if args.role is None else registry.by_role(args.role)
    print(json.dumps({
        "schema_version": "mathmodel.benchmark-source-list/v1",
        "role": args.role,
        "sources": [
            {"id": row.source_id, "title": row.title, "url": row.url,
             "authority": row.authority, "access": row.access,
             "license_status": row.license_status, "role": row.role,
             "notes": row.notes}
            for row in rows
        ],
        "gold_truth_policy": "public_sources_are_not_gold_truth",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
