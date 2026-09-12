"""Validate metadata for a project-external unseen modeling holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.holdout_intake import HoldoutIntakeError, validate_holdout_intake


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate metadata-only unseen holdout intake")
    parser.add_argument("input", type=Path)
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--min-families", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        result = validate_holdout_intake(payload, min_cases=args.min_cases, min_families=args.min_families)
    except (OSError, UnicodeError, json.JSONDecodeError, HoldoutIntakeError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] == "ready_for_sealing" else 2


if __name__ == "__main__":
    raise SystemExit(main())
