"""Validate an OpenModelBench manifest without executing any model."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.open_model_bench import OpenModelBench, BenchmarkValidationError


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate an OpenModelBench manifest")
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)
    try:
        payload = json.loads(args.manifest.read_text(encoding="utf-8"))
        bench = OpenModelBench.from_payload(payload)
    except (OSError, UnicodeError, json.JSONDecodeError, BenchmarkValidationError) as exc:
        print(json.dumps({"status": "invalid", "error": str(exc)}, ensure_ascii=False))
        return 2
    summary = {
        "status": "valid",
        "schema_version": payload["schema_version"],
        "name": payload["name"],
        "revision": payload["revision"],
        "digest": bench.digest,
        "case_count": len(bench.cases()),
        "splits": {split: len(bench.cases(split=split)) for split in
                   ("development", "unseen", "structure_transform", "adversarial")},
        "families": {family: len(bench.cases(family=family)) for family in
                     ("data", "pure_mechanistic", "multi_table", "optimization", "dynamics")},
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
