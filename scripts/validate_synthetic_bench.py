"""Validate the five bounded synthetic OpenModelBench fixtures."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.synthetic_bench_cases import build_minimal_case_catalog, validate_case_catalog


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cases = build_minimal_case_catalog()
    result = validate_case_catalog(cases)
    result["cases"] = [case.public() for case in cases]
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "valid" else 2)
