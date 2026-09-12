"""Validate the tracked public experiment configuration offline."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.experiment_config import load_experiment_config  # noqa: E402


def main() -> int:
    config = load_experiment_config(ROOT / "examples" / "public_experiment_config.json")
    print(json.dumps({"status": "pass", "config_digest": config.digest,
                      "catalog_digest": config.catalog_digest,
                      "methods": list(config.methods),
                      "selection_splits": list(config.selection_splits),
                      "evaluation_splits": list(config.evaluation_splits)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
