"""Derive a non-destructive v26 metric erratum from immutable row-level output."""
from __future__ import annotations

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


def derive_erratum(report: Mapping[str, Any], *, source: str, source_sha256: str) -> dict[str, Any]:
    if report.get("schema_version") != "mathmodel.mixed-risk-confirmation/v1":
        raise ValueError("mixed_risk_report_schema_invalid")
    compiler = deepcopy(report["compiler_summary"])
    for family, summary in compiler.items():
        rows = [row for row in report["compiler_rows"] if row["family"] == family]
        summary["false_abstain_count"] = sum(
            row.get("status") == "needs_input" and not bool(row.get("valid")) for row in rows
        )
    product = deepcopy(report["product_summary"])
    product["single_solver"]["reported_numerical_solver_calls"] = product["single_solver"].pop(
        "solver_arm_launch_count"
    )
    product["single_solver"]["solver_arm_launch_count"] = None
    return {
        "schema_version": "mathmodel.mixed-risk-erratum/v1",
        "source_report": source, "source_sha256": source_sha256,
        "source_freeze_digest": report["freeze"]["freeze_digest"],
        "source_freeze_status": [report["freeze_before"]["status"], report["freeze_after"]["status"]],
        "corrected_compiler_summary": compiler, "corrected_product_summary": product,
        "corrections": [
            "A correct required-ambiguity refusal is not a false abstention.",
            "Single-solver reported numerical_solver_calls count internal work, not portfolio arm launches.",
        ],
        "policy": "row_level_recalculation_only;original_frozen_report_not_overwritten",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="artifacts/product-workflow-confirmation/mixed-risk-20261020.json")
    parser.add_argument("--output", default="artifacts/product-workflow-confirmation/mixed-risk-20261020-erratum.json")
    args = parser.parse_args()
    source = Path(args.input)
    raw = source.read_bytes()
    report = json.loads(raw)
    audit = derive_erratum(report, source=str(source), source_sha256=sha256(raw).hexdigest())
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(destination), "corrections": audit["corrections"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
