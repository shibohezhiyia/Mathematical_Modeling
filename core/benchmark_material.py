"""Bind registered external sources to a benchmark without importing content.

This is the missing link between a URL catalog and the blind benchmark
protocol.  A material plan records which sources are allowed for task input,
rubric construction, method comparison and human-only reference inspection.
It never treats a public solution as a score-bearing answer.
"""

from __future__ import annotations

from hashlib import sha256
import json
from typing import Any, Iterable, Mapping

from .benchmark_sources import BenchmarkSourceError, BenchmarkSourceRegistry


MATERIAL_SCHEMA = "mathmodel.benchmark-material-plan/v1"


class BenchmarkMaterialError(ValueError):
    pass


def _ids(values: Iterable[str], code: str) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise BenchmarkMaterialError(code)
    try:
        rows = list(values)
    except TypeError as exc:
        raise BenchmarkMaterialError(code) from exc
    if not rows or any(not isinstance(item, str) or not item.strip() for item in rows):
        raise BenchmarkMaterialError(code)
    rows = [item.strip() for item in rows]
    if len(rows) != len(set(rows)):
        raise BenchmarkMaterialError("duplicate_material_source")
    return rows


def _case_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 160 or any(ch.isspace() for ch in value):
        raise BenchmarkMaterialError("material_case_id_invalid")
    return value.strip()


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return sha256(encoded.encode("utf-8")).hexdigest()


def build_material_plan(registry: BenchmarkSourceRegistry, *, case_id: str,
                        input_source_ids: Iterable[str], rubric_source_ids: Iterable[str],
                        comparison_source_ids: Iterable[str] = (),
                        reference_source_ids: Iterable[str] = ()) -> dict[str, Any]:
    if not isinstance(registry, BenchmarkSourceRegistry):
        raise BenchmarkMaterialError("source_registry_required")
    case_id = _case_id(case_id)
    groups = {
        "input_source_ids": _ids(input_source_ids, "input_sources_required"),
        "rubric_source_ids": _ids(rubric_source_ids, "rubric_sources_required"),
        "comparison_source_ids": list(comparison_source_ids) if not isinstance(comparison_source_ids, (str, bytes)) else None,
        "reference_source_ids": list(reference_source_ids) if not isinstance(reference_source_ids, (str, bytes)) else None,
    }
    if groups["comparison_source_ids"] is None or groups["reference_source_ids"] is None:
        raise BenchmarkMaterialError("optional_sources_invalid")
    for key in ("comparison_source_ids", "reference_source_ids"):
        values = groups[key]
        if any(not isinstance(item, str) or not item.strip() for item in values):
            raise BenchmarkMaterialError("optional_sources_invalid")
        groups[key] = [item.strip() for item in values]
        if len(groups[key]) != len(set(groups[key])):
            raise BenchmarkMaterialError("duplicate_material_source")
    all_ids = [item for values in groups.values() for item in values]
    if len(all_ids) != len(set(all_ids)):
        raise BenchmarkMaterialError("source_reused_across_material_roles")
    expected = {
        "input_source_ids": "task_input", "rubric_source_ids": "rubric_material",
        "comparison_source_ids": "method_comparison", "reference_source_ids": "reference_only",
    }
    resolved = {}
    for key, role in expected.items():
        resolved[key] = []
        for source_id in groups[key]:
            try:
                source = registry.get(source_id)
            except BenchmarkSourceError as exc:
                raise BenchmarkMaterialError("unknown_material_source") from exc
            if source.role != role:
                raise BenchmarkMaterialError("source_role_mismatch")
            resolved[key].append(source_id)
    body = {
        "schema_version": MATERIAL_SCHEMA, "case_id": case_id,
        "input_source_ids": resolved["input_source_ids"],
        "rubric_source_ids": resolved["rubric_source_ids"],
        "comparison_source_ids": resolved["comparison_source_ids"],
        "reference_source_ids": resolved["reference_source_ids"],
        "score_policy": "public_sources_are_not_gold_truth; independent_rubric_and_sealed_reference_required",
    }
    return {**body, "plan_digest": _digest(body)}


def validate_material_plan(registry: BenchmarkSourceRegistry, plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping) or plan.get("schema_version") != MATERIAL_SCHEMA:
        raise BenchmarkMaterialError("material_plan_schema_invalid")
    required = {"schema_version", "case_id", "input_source_ids", "rubric_source_ids",
                "comparison_source_ids", "reference_source_ids", "score_policy", "plan_digest"}
    if set(plan) != required:
        raise BenchmarkMaterialError("material_plan_fields_invalid")
    body = {key: plan[key] for key in required if key not in {"plan_digest"}}
    if plan.get("plan_digest") != _digest(body):
        raise BenchmarkMaterialError("material_plan_digest_mismatch")
    rebuilt = build_material_plan(registry, case_id=plan["case_id"],
                                  input_source_ids=plan["input_source_ids"],
                                  rubric_source_ids=plan["rubric_source_ids"],
                                  comparison_source_ids=plan["comparison_source_ids"],
                                  reference_source_ids=plan["reference_source_ids"])
    if rebuilt != dict(plan):
        raise BenchmarkMaterialError("material_plan_not_canonical")
    return {"status": "pass", "case_id": plan["case_id"], "plan_digest": plan["plan_digest"],
            "score_policy": plan["score_policy"]}


__all__ = ["BenchmarkMaterialError", "MATERIAL_SCHEMA", "build_material_plan", "validate_material_plan"]
