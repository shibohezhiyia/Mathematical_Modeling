"""Independent, automatically scorable modeling-task benchmark protocol.

The runner separates three roles that are often accidentally coupled in
synthetic evaluations:

* a case generator/fixture owns the hidden reference;
* the system under test receives only the public statement and inputs;
* an independent scorer receives the system output and the hidden reference.

This is an evaluation harness, not a solver and not evidence of real-world
generalization.  Cases are grouped by ``structure_group`` so parameter
variants cannot be counted as independent mathematical tasks.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from collections import defaultdict
from statistics import mean, median
import time
from typing import Any, Callable, Mapping, Sequence


class AutomatedBenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class AutomatedBenchmarkCase:
    case_id: str
    family: str
    statement: str
    public_input: Mapping[str, Any]
    hidden_reference: Mapping[str, Any]
    structure_group: str
    source_group: str = "synthetic"

    def validate(self) -> "AutomatedBenchmarkCase":
        for name in ("case_id", "family", "statement", "structure_group", "source_group"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 160:
                raise AutomatedBenchmarkError(f"{name}_invalid")
        if not isinstance(self.public_input, Mapping) or not isinstance(self.hidden_reference, Mapping):
            raise AutomatedBenchmarkError("case_payloads_must_be_mappings")
        return self

    @staticmethod
    def _digest(value: Any) -> str:
        payload = json.dumps(value, ensure_ascii=False, sort_keys=True,
                             separators=(",", ":"), allow_nan=False, default=str)
        return sha256(payload.encode("utf-8")).hexdigest()

    def public_metadata(self) -> dict[str, Any]:
        self.validate()
        return {
            "id": self.case_id, "family": self.family,
            "structure_group": self.structure_group,
            "source_group": self.source_group,
            "input_digest": self._digest(self.public_input),
            "reference_digest": self._digest(self.hidden_reference),
        }


def _validate_score(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise AutomatedBenchmarkError("score_must_be_finite_numeric")
    return float(value)


def run_automated_benchmark(
    cases: Sequence[AutomatedBenchmarkCase],
    system: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    scorer: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
    *, max_cases: int = 256, wall_seconds: float = 600.0,
) -> dict[str, Any]:
    """Run fixed cases without leaking hidden references to the system.

    ``system`` receives a fresh copy containing only the public fields.  The
    scorer is the sole component that sees the hidden reference.  A scorer's
    ``valid`` field is required to be boolean; malformed results become a
    failed row instead of silently counting as success.
    """
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases:
        raise AutomatedBenchmarkError("cases_required")
    if type(max_cases) is not int or not 1 <= max_cases <= 10_000 or len(cases) > max_cases:
        raise AutomatedBenchmarkError("case_budget_invalid")
    if type(wall_seconds) not in (int, float) or not math.isfinite(float(wall_seconds)) or not 0.01 <= float(wall_seconds) <= 86_400:
        raise AutomatedBenchmarkError("wall_budget_invalid")
    if not callable(system) or not callable(scorer):
        raise AutomatedBenchmarkError("system_and_scorer_required")
    normalized: list[AutomatedBenchmarkCase] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, AutomatedBenchmarkCase):
            raise AutomatedBenchmarkError("case_type_invalid")
        case.validate()
        if case.case_id in seen:
            raise AutomatedBenchmarkError("duplicate_case_id")
        seen.add(case.case_id)
        normalized.append(case)

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    for case in normalized:
        if time.monotonic() - started >= float(wall_seconds):
            rows.append({"id": case.case_id, "family": case.family,
                         "structure_group": case.structure_group,
                         "source_group": case.source_group,
                         "status": "timeout", "valid": None})
            continue
        tick = time.monotonic()
        public = {"id": case.case_id, "family": case.family,
                  "statement": case.statement,
                  "input": deepcopy(dict(case.public_input)),
                  "structure_group": case.structure_group}
        try:
            output = system(deepcopy(public))
            if not isinstance(output, Mapping):
                raise AutomatedBenchmarkError("system_output_must_be_mapping")
            scored = scorer(deepcopy(dict(output)), deepcopy(dict(case.hidden_reference)))
            if not isinstance(scored, Mapping) or type(scored.get("valid")) is not bool:
                raise AutomatedBenchmarkError("scorer_valid_boolean_required")
            score = scored.get("score")
            row = {"id": case.case_id, "family": case.family,
                   "structure_group": case.structure_group,
                   "source_group": case.source_group, "status": "completed",
                   "valid": bool(scored["valid"]),
                   "score": _validate_score(score) if score is not None else None,
                   "duration_seconds": float(time.monotonic() - tick)}
            if scored.get("reason") is not None:
                row["reason"] = str(scored["reason"])[:240]
        except Exception as exc:
            row = {"id": case.case_id, "family": case.family,
                   "structure_group": case.structure_group,
                   "source_group": case.source_group, "status": "failed",
                   "valid": None, "error_code": type(exc).__name__,
                   "duration_seconds": float(time.monotonic() - tick)}
        rows.append(row)

    groups = {str(row["structure_group"]) for row in rows}
    completed = [row for row in rows if row["status"] == "completed"]
    valid = [row for row in completed if row.get("valid") is True]
    def _summary(key: str) -> dict[str, Any]:
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            buckets[str(row.get(key, "unknown"))].append(row)
        result: dict[str, Any] = {}
        for bucket, items in sorted(buckets.items()):
            finished = [item for item in items if item.get("status") == "completed"]
            accepted = [item for item in finished if item.get("valid") is True]
            scores = [float(item["score"]) for item in finished
                      if isinstance(item.get("score"), (int, float)) and math.isfinite(float(item["score"]))]
            durations = [float(item["duration_seconds"]) for item in items
                         if isinstance(item.get("duration_seconds"), (int, float))]
            result[bucket] = {
                "case_count": len(items),
                "completed_count": len(finished),
                "valid_count": len(accepted),
                "failed_or_timeout_count": len(items) - len(finished),
                "valid_rate": (len(accepted) / len(finished)) if finished else None,
                "score_mean": mean(scores) if scores else None,
                "score_median": median(scores) if scores else None,
                "duration_seconds_sum": sum(durations),
            }
        return result
    return {
        "schema_version": "mathmodel.automated-benchmark/v1",
        "status": "completed" if len(rows) == len(normalized) else "partial",
        "rows": rows, "case_count": len(normalized),
        "completed_count": len(completed), "valid_count": len(valid),
        "valid_rate": (len(valid) / len(completed)) if completed else None,
        "structure_group_count": len(groups),
        "structure_groups": sorted(groups),
        "family_summary": _summary("family"),
        "structure_group_summary": _summary("structure_group"),
        "wall_seconds": float(time.monotonic() - started),
        "policy": "generator_system_scorer_separated;_structure_group_is_unit_of_independence;not_real_world_accuracy",
    }


__all__ = ["AutomatedBenchmarkError", "AutomatedBenchmarkCase", "run_automated_benchmark"]
