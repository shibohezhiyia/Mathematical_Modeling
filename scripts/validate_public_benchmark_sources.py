"""Validate the tracked public source catalog without fetching problem text."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from core.public_benchmark_catalog import load_public_catalog  # noqa: E402


def main() -> int:
    catalog = load_public_catalog(ROOT / "examples" / "public_benchmark_sources.json")
    counts = {}
    for source in catalog.sources():
        counts[source.provider] = counts.get(source.provider, 0) + 1
    print(json.dumps({"status": "pass", "source_count": len(catalog.sources()),
                      "catalog_digest": catalog.digest, "providers": counts}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
