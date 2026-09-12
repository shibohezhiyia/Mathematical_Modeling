"""Generate anonymised blind-review forms for a frozen run set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.review_packet import ReviewPacketError, build_review_form, build_review_packet


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成匿名冻结留出集盲审包")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True, help="包含 run records 数组的 JSON")
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, help="运行结果根目录；按 case_id/result.json 查找，仅写入私有映射")
    parser.add_argument("--reviewer", nargs="+", default=["reviewer-1", "reviewer-2"])
    args = parser.parse_args(argv)
    try:
        manifest = _read(args.manifest)
        runs = _read(args.runs)
        rubric = _read(args.rubric)
        if not isinstance(runs, list):
            raise ReviewPacketError("runs_file_must_be_array")
        output_paths = None
        if args.result_root:
            output_paths = {row.get("case_id"): str(args.result_root / str(row.get("case_id")) / "result.json") for row in runs if isinstance(row, dict) and row.get("case_id")}
        result = build_review_packet(manifest, runs, rubric, output_paths=output_paths)
        _write(args.output_dir / "packet.json", result["packet"])
        # The mapping contains original case IDs and must not be committed.
        _write(args.output_dir / "private_mapping.json", result["private_mapping"])
        for reviewer in args.reviewer:
            _write(args.output_dir / f"form_{reviewer}.json", build_review_form(result["packet"], rubric, reviewer_id=reviewer))
        print(json.dumps({"status": "created", "case_count": result["packet"]["case_count"], "output_dir": str(args.output_dir), "private_mapping": "keep_private"}, ensure_ascii=False))
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, ReviewPacketError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
