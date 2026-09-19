"""Bounded multi-table join candidate and CEGIS adapter.

This adapter treats table grain, keys and aggregation as mathematical
structure.  It never joins by column position and it refuses unbounded
many-to-many expansion.  The evaluator checks explicit row/target contracts;
it is not a substitute for causal identification.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence
import pandas as pd

from .cegis_controller import CEGISConfig
from .model_family_adapters import ModelFamilyAdapter, run_model_family_cegis


class MultiTableCEGISError(ValueError):
    pass


def compile_multitable_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise MultiTableCEGISError("multitable_candidate_must_be_object")
    tables = candidate.get("tables")
    joins = candidate.get("joins", [])
    if not isinstance(tables, Mapping) or not 1 <= len(tables) <= 8:
        raise MultiTableCEGISError("tables_count_invalid")
    if not isinstance(joins, list) or len(joins) > 8:
        raise MultiTableCEGISError("joins_count_invalid")
    normalized = {}
    for name, rows in tables.items():
        if not isinstance(name, str) or not name.strip() or not isinstance(rows, list) or not rows or len(rows) > 20_000:
            raise MultiTableCEGISError("table_rows_invalid")
        if not all(isinstance(row, Mapping) for row in rows):
            raise MultiTableCEGISError("table_row_must_be_object")
        frame = pd.DataFrame.from_records(rows)
        if frame.empty or frame.shape[1] > 256:
            raise MultiTableCEGISError("table_shape_invalid")
        normalized[name] = frame
    clean_joins = []
    for join in joins:
        if not isinstance(join, Mapping):
            raise MultiTableCEGISError("join_must_be_object")
        required = {"left_table", "right_table", "left_key", "right_key"}
        if not required.issubset(join) or set(join) - required - {"how", "aggregate", "max_rows"}:
            raise MultiTableCEGISError("join_fields_invalid")
        if join["left_table"] not in normalized or join["right_table"] not in normalized:
            raise MultiTableCEGISError("join_table_missing")
        how = str(join.get("how", "left"))
        if how not in {"left", "inner"}:
            raise MultiTableCEGISError("join_how_invalid")
        aggregate = str(join.get("aggregate", "mean"))
        if aggregate not in {"mean", "sum", "first", "count"}:
            raise MultiTableCEGISError("join_aggregate_invalid")
        clean_joins.append({"left_table": str(join["left_table"]), "right_table": str(join["right_table"]),
                            "left_key": str(join["left_key"]), "right_key": str(join["right_key"]),
                            "how": how, "aggregate": aggregate,
                            "max_rows": int(join.get("max_rows", 100_000))})
    return {"id": str(candidate.get("id", "multitable_candidate"))[:120], "tables": normalized,
            "joins": clean_joins, "target": candidate.get("target"), "max_output_rows": int(candidate.get("max_output_rows", 100_000))}


def materialize_multitable_candidate(compiled: Mapping[str, Any]) -> pd.DataFrame:
    if isinstance(compiled.get("tables") if isinstance(compiled, Mapping) else None, Mapping):
        sample = next(iter(compiled["tables"].values()), None)
        if isinstance(sample, list):
            compiled = compile_multitable_candidate(compiled)
    tables = compiled["tables"]
    if not tables:
        raise MultiTableCEGISError("no_tables")
    result = next(iter(tables.values())).copy()
    used = {next(iter(tables))}
    for join in compiled["joins"]:
        left, right = join["left_table"], join["right_table"]
        if left not in used or right in used:
            if right in used:
                continue
            raise MultiTableCEGISError("join_order_not_connected")
        other = tables[right].copy()
        lk, rk = join["left_key"], join["right_key"]
        if lk not in result.columns or rk not in other.columns:
            raise MultiTableCEGISError("join_key_missing")
        # Aggregate the right table before joining.  This is the safe default
        # for one-to-many data and prevents accidental Cartesian products.
        value_cols = [c for c in other.columns if c != rk]
        numeric = [c for c in value_cols if pd.api.types.is_numeric_dtype(other[c])]
        if numeric:
            grouped = other.groupby(rk, dropna=False, sort=False)[numeric]
            aggregate = join["aggregate"]
            if aggregate == "count":
                other = grouped.size().rename("__row_count").reset_index()
            elif aggregate == "first":
                other = grouped.first().reset_index()
            else:
                other = getattr(grouped, aggregate)().reset_index()
        if len(other) > join["max_rows"]:
            raise MultiTableCEGISError("join_output_budget_exceeded")
        suffix = f"__{right}"
        result = result.merge(other, how=join["how"], left_on=lk, right_on=rk, suffixes=("", suffix), validate="many_to_one")
        if len(result) > compiled["max_output_rows"]:
            raise MultiTableCEGISError("merged_rows_exceed_budget")
        used.add(right)
    result.attrs["multitable_join_audit"] = {"table_count": len(tables), "join_count": len(compiled["joins"]), "output_rows": len(result)}
    return result


def evaluate_multitable_candidate(compiled: Mapping[str, Any], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases or len(cases) > 32:
        raise MultiTableCEGISError("cases_invalid")
    violations = []
    outputs = []
    row_errors: list[float] = []
    for idx, case in enumerate(cases):
        frame = materialize_multitable_candidate(compiled)
        expected_rows = case.get("expected_rows")
        if expected_rows is not None and len(frame) != int(expected_rows):
            violations.append({"case": idx, "reason": "row_count_mismatch", "actual": len(frame), "expected": int(expected_rows)})
            row_errors.append(abs(float(len(frame) - int(expected_rows))))
        elif expected_rows is not None:
            row_errors.append(0.0)
        target = case.get("target", compiled.get("target"))
        if target and target not in frame.columns:
            violations.append({"case": idx, "reason": "target_missing", "target": target})
        outputs.append({"rows": len(frame), "columns": list(frame.columns)[:256]})
    metrics: dict[str, float] = {
        "complexity": float(len(compiled.get("tables", {})) + len(compiled.get("joins", []))),
        "constraint_violation": float(max(row_errors, default=0.0)),
        # This measures observed join-contract violations only; it is not a
        # statistical stability or causal validity certificate.
        "instability": float(max(row_errors, default=0.0)),
    }
    if row_errors:
        metrics["validation_loss"] = float(sum(row_errors) / len(row_errors))
    return {"status": "fail" if violations else "pass", "violations": violations,
            "predictions": [float(item["rows"]) for item in outputs], "metrics": metrics,
            "outputs": outputs, "cost_units": len(cases),
            "policy": "grain_and_key_checked;_many_to_many_aggregated_before_join;validation_loss_requires_expected_rows"}


def run_multitable_cegis(candidates: Iterable[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]], *, config: CEGISConfig | None = None) -> dict[str, Any]:
    def compile_(candidate): return compile_multitable_candidate(candidate)
    def evaluate(compiled, frozen): return evaluate_multitable_candidate(compiled, frozen)
    def diagnose(feedback): return {"violations": feedback.get("violations", [])}
    def patch(candidate, diagnostic):
        for join_index, join in enumerate(candidate.get("joins", [])):
            if join.get("aggregate") != "count":
                item = dict(candidate); joins = [dict(j) for j in candidate.get("joins", [])]
                joins[join_index]["aggregate"] = "count" if diagnostic.get("violations") else "first"
                item["joins"] = joins
                yield item
                break
    adapter = ModelFamilyAdapter("multitable", compile_, evaluate, diagnose, patch)
    return run_model_family_cegis(adapter, candidates, cases, config=config)


__all__ = ["MultiTableCEGISError", "compile_multitable_candidate", "materialize_multitable_candidate", "evaluate_multitable_candidate", "run_multitable_cegis"]
