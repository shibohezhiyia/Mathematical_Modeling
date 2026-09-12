"""Validate the conservative audit of candidate unseen-evaluation sources."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.evaluation_source_audit import EvaluationSourceRegistry


def main() -> int:
    registry = EvaluationSourceRegistry.load(ROOT / "examples" / "evaluation_sources.json")
    print(json.dumps(registry.public_metadata(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
