"""Write a public metadata freeze before an unseen evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation_freeze import EvaluationFreezeError, create_evaluation_freeze


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a content-addressed evaluation freeze")
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "reports" / "evaluation_freeze.json")
    parser.add_argument("--method", action="append", dest="methods", required=True)
    parser.add_argument("--max-seconds", type=int, default=1800)
    parser.add_argument("--max-memory-mb", type=int, default=4096)
    parser.add_argument("--max-api-calls", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=32)
    args = parser.parse_args()
    try:
        result = create_evaluation_freeze(
            ROOT, protocol_id=args.protocol_id, seed=args.seed,
            budget={"max_seconds": args.max_seconds, "max_memory_mb": args.max_memory_mb,
                    "max_api_calls": args.max_api_calls, "max_candidates": args.max_candidates},
            methods=args.methods,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"status": result["status"], "freeze_digest": result["freeze_digest"],
                          "file_count": len(result["files"]), "output": str(args.output)}, ensure_ascii=False))
        return 0
    except (OSError, EvaluationFreezeError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
