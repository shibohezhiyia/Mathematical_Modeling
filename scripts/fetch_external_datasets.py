"""Fetch pinned public datasets for an external-data evaluation arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.external_dataset_catalog import ExternalDatasetError, dataset, download_dataset, load_catalog


def main() -> int:
    parser = argparse.ArgumentParser(description="Download verified public modelling datasets")
    parser.add_argument("--catalog", type=Path, default=ROOT / "examples" / "external_dataset_catalog.json")
    parser.add_argument("--dataset", action="append", dest="dataset_ids", help="dataset id; repeatable")
    parser.add_argument("--output-root", type=Path, default=Path("data/external"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extract", action="store_true", help="safely extract ZIP members into <output>/<id>")
    args = parser.parse_args()
    try:
        catalog = load_catalog(args.catalog)
        selected = catalog["datasets"] if not args.dataset_ids else [dataset(catalog, x) for x in args.dataset_ids]
        receipts = []
        for item in selected:
            if args.dry_run:
                receipts.append({"id": item["id"], "url": item["url"], "bytes": item.get("bytes"), "sha256": item.get("sha256")})
            else:
                receipts.append(download_dataset(item, args.output_root, extract=args.extract))
        print(json.dumps({"schema_version": "mathmodel.external-dataset-fetch/v1", "count": len(receipts), "datasets": receipts}, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, ExternalDatasetError) as exc:
        print(json.dumps({"status": "rejected", "reason": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
