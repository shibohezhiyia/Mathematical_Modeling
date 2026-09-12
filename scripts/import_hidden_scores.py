"""Import an external hidden-label score export and write a safe summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hidden_label_evaluation import HiddenLabelEvaluationError, summarize_hidden_label_scores


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="导入隐藏标签平台分数并生成描述性摘要")
    parser.add_argument("--scores", type=Path, required=True, help="平台导出的 JSON 数组")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tasks", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        rows = json.loads(args.scores.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise HiddenLabelEvaluationError("scores_file_must_be_array")
        result = summarize_hidden_label_scores(rows, min_tasks=args.min_tasks)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "summarized", "output": str(args.output), "task_count": result["task_count"]}, ensure_ascii=False))
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, HiddenLabelEvaluationError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
