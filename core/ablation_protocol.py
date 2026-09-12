"""Pre-registered comparison and ablation manifests.

The protocol makes missing, timed-out, rejected, and incomplete runs explicit
denominator entries.  It does not manufacture improvement or significance
without observed paired outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


class AblationProtocolError(ValueError):
    pass


_STATUSES = frozenset({"completed", "timeout", "rejected", "incomplete", "error"})


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def build_ablation_manifest(
    arms: Sequence[Mapping[str, Any]],
    *,
    task_ids: Sequence[str],
    fixed_budget: Mapping[str, Any],
    final_test_fingerprint: str,
    scoring_criteria: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(arms, Sequence) or isinstance(arms, (str, bytes)) or not arms:
        raise AblationProtocolError("arms_required")
    if (not isinstance(task_ids, Sequence) or isinstance(task_ids, (str, bytes)) or not task_ids or
            any(type(item) is not str or not item.strip() for item in task_ids) or
            len(set(task_ids)) != len(task_ids)):
        raise AblationProtocolError("task_ids_required")
    if not isinstance(fixed_budget, Mapping) or not fixed_budget:
        raise AblationProtocolError("fixed_budget_required")
    if not isinstance(final_test_fingerprint, str) or not final_test_fingerprint.strip():
        raise AblationProtocolError("final_test_fingerprint_required")
    if not isinstance(scoring_criteria, Sequence) or isinstance(scoring_criteria, (str, bytes)) or not scoring_criteria:
        raise AblationProtocolError("scoring_criteria_required")
    normalized = []
    seen = set()
    for arm in arms:
        if not isinstance(arm, Mapping) or not isinstance(arm.get("id"), str) or not arm["id"].strip():
            raise AblationProtocolError("arm_id_required")
        identifier = arm["id"].strip()
        if identifier in seen:
            raise AblationProtocolError("duplicate_arm_id")
        seen.add(identifier)
        components = arm.get("components", {})
        if not isinstance(components, Mapping):
            raise AblationProtocolError("arm_components_required")
        baseline = arm.get("baseline", False)
        if type(baseline) is not bool:
            raise AblationProtocolError("baseline_must_be_boolean")
        normalized.append({"id": identifier, "label": str(arm.get("label", identifier))[:160],
                           "components": dict(components), "baseline": baseline})
    if sum(item["baseline"] for item in normalized) != 1:
        raise AblationProtocolError("exactly_one_baseline_required")
    payload = {"arms": normalized, "task_ids": list(task_ids), "fixed_budget": dict(fixed_budget),
               "final_test_fingerprint": final_test_fingerprint.strip(), "scoring_criteria": [str(item) for item in scoring_criteria]}
    return {"schema_version": "mathmodel.ablation-manifest/v1", "status": "preregistered",
            **payload, "manifest_digest": _digest(payload),
            "policy": "same_tasks_budget_backend_and_locked_final_test; no_success_only_denominator"}


def record_ablation_results(
    manifest: Mapping[str, Any],
    results: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "mathmodel.ablation-manifest/v1":
        raise AblationProtocolError("manifest_required")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise AblationProtocolError("results_required")
    arms = {item["id"] for item in manifest.get("arms", [])}
    tasks = set(manifest.get("task_ids", []))
    rows = []
    for row in results:
        if not isinstance(row, Mapping) or row.get("arm_id") not in arms or row.get("task_id") not in tasks:
            raise AblationProtocolError("result_arm_or_task_unknown")
        status = row.get("status")
        if status not in _STATUSES:
            raise AblationProtocolError("result_status_invalid")
        valid = row.get("valid", False)
        if type(valid) is not bool:
            raise AblationProtocolError("valid_must_be_boolean")
        score = row.get("score") if status == "completed" else None
        if score is not None and (type(score) not in (int, float) or isinstance(score, bool) or
                                   not math.isfinite(float(score))):
            raise AblationProtocolError("score_invalid")
        rows.append({"arm_id": row["arm_id"], "task_id": row["task_id"], "status": status,
                     "valid": valid if status == "completed" else False,
                     "score": score,
                     "error_code": row.get("error_code")})
    expected = len(arms) * len(tasks)
    complete_grid = len(rows) == expected and len({(row["arm_id"], row["task_id"]) for row in rows}) == expected
    by_status = {status: sum(row["status"] == status for row in rows) for status in sorted(_STATUSES)}
    return {"schema_version": "mathmodel.ablation-results/v1", "status": "assessed" if complete_grid else "incomplete",
            "expected_rows": expected, "observed_rows": len(rows), "complete_grid": complete_grid,
            "status_counts": by_status, "rows": rows,
            "policy": "timeouts_rejections_and_incomplete_runs_remain_in_denominator; no significance_without_paired_data"}


__all__ = ["AblationProtocolError", "build_ablation_manifest", "record_ablation_results"]
