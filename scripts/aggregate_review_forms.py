"""Aggregate completed anonymous review forms using a private mapping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.review_packet import ReviewPacketError, aggregate_review_forms


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="汇总匿名建模盲审表")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True, help="仓外私有映射")
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--forms", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        read = lambda path: json.loads(path.read_text(encoding="utf-8"))
        result = aggregate_review_forms(read(args.packet), read(args.mapping), read(args.rubric), [read(path) for path in args.forms])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "case_count": result["case_count"], "output": str(args.output)}, ensure_ascii=False))
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, ReviewPacketError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
