"""Verify that a sealed blind manifest matches an intake commitment file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.holdout_binding import HoldoutBindingError, bind_holdout_intake


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Bind metadata-only intake to a sealed blind manifest")
    parser.add_argument("intake", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--min-cases", type=int, default=20)
    parser.add_argument("--min-families", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        intake = json.loads(args.intake.read_text(encoding="utf-8"))
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = bind_holdout_intake(intake, manifest, min_cases=args.min_cases, min_families=args.min_families)
    except (OSError, UnicodeError, json.JSONDecodeError, HoldoutBindingError) as exc:
        print(json.dumps({"status": "error", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
