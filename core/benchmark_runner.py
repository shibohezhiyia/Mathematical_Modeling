"""Bounded fixed-task benchmark execution and latency summaries."""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Mapping, Sequence


class BenchmarkRunnerError(ValueError):
    pass


_STATUSES = frozenset({"completed", "failed", "timeout", "rejected", "incomplete"})


@dataclass(frozen=True)
class BenchmarkBudget:
    max_cases: int = 256
    max_repeats: int = 3
    wall_seconds: float = 120.0

    def validate(self) -> "BenchmarkBudget":
        if type(self.max_cases) is not int or not 1 <= self.max_cases <= 10_000:
            raise BenchmarkRunnerError("invalid_max_cases")
        if type(self.max_repeats) is not int or not 1 <= self.max_repeats <= 32:
            raise BenchmarkRunnerError("invalid_max_repeats")
        if type(self.wall_seconds) not in (int, float) or not math.isfinite(float(self.wall_seconds)) or not 0.01 <= float(self.wall_seconds) <= 86_400:
            raise BenchmarkRunnerError("invalid_wall_budget")
        return self


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(probability * len(ordered))) - 1))
    return float(ordered[index])


def run_fixed_benchmark(
    cases: Sequence[Mapping[str, Any]], evaluate: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    *, budget: BenchmarkBudget | None = None,
) -> dict[str, Any]:
    """Run a fixed case list and retain timeouts/failures as first-class rows."""
    config = (budget or BenchmarkBudget()).validate()
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not cases or len(cases) > config.max_cases:
        raise BenchmarkRunnerError("cases_out_of_bounds")
    if not callable(evaluate):
        raise BenchmarkRunnerError("evaluator_required")
    started = time.monotonic()
    rows = []
    seen = set()
    normalized_cases = []
    for case in cases:
        if (not isinstance(case, Mapping) or type(case.get("id")) is not str or
                not case["id"].strip() or len(case["id"].strip()) > 128):
            raise BenchmarkRunnerError("invalid_case")
        identifier = case["id"].strip()
        if identifier in seen:
            raise BenchmarkRunnerError("duplicate_case_id")
        seen.add(identifier)
        normalized_cases.append({**dict(case), "id": identifier})
    cases = normalized_cases
    for index, case in enumerate(cases):
        for repeat in range(config.max_repeats):
            elapsed = time.monotonic() - started
            if elapsed >= config.wall_seconds:
                rows.append({"id": case["id"], "repeat": repeat, "status": "timeout", "duration_seconds": 0.0})
                break
            tick = time.monotonic()
            try:
                result = evaluate(case)
                if not isinstance(result, Mapping):
                    raise BenchmarkRunnerError("evaluator_must_return_mapping")
                status = result.get("status", "completed")
                if not isinstance(status, str) or not status.strip() or len(status) > 80:
                    raise BenchmarkRunnerError("invalid_evaluator_status")
                status = status.strip()
                if status not in _STATUSES:
                    raise BenchmarkRunnerError("invalid_evaluator_status")
                score = result.get("score")
                if score is not None and (type(score) not in (int, float) or not math.isfinite(float(score))):
                    raise BenchmarkRunnerError("invalid_evaluator_score")
                memory = result.get("memory_mb")
                if memory is not None and (type(memory) not in (int, float) or not math.isfinite(float(memory)) or float(memory) < 0):
                    raise BenchmarkRunnerError("invalid_evaluator_memory")
                row = {"id": case["id"], "repeat": repeat, "status": status,
                       "duration_seconds": float(time.monotonic() - tick),
                       "score": score, "memory_mb": memory}
            except BenchmarkRunnerError as exc:
                # A malformed evaluator response is a failed case, not a
                # reason to discard all later cases in the fixed benchmark.
                row = {"id": case["id"], "repeat": repeat, "status": "failed",
                       "duration_seconds": float(time.monotonic() - tick),
                       "error_code": str(exc)[:80] or "evaluator_contract_error"}
            except Exception as exc:
                row = {"id": case["id"], "repeat": repeat, "status": "failed",
                       "duration_seconds": float(time.monotonic() - tick), "error_code": type(exc).__name__}
            rows.append(row)
        if time.monotonic() - started >= config.wall_seconds and index + 1 < len(cases):
            for remaining in cases[index + 1:]:
                rows.append({"id": remaining["id"], "repeat": 0, "status": "not_run_budget", "duration_seconds": 0.0})
            break
    durations = [row["duration_seconds"] for row in rows if row["status"] not in {"timeout", "not_run_budget"}]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {"schema_version": "mathmodel.fixed-benchmark/v1", "rows": rows, "status_counts": counts,
            "case_count": len(cases), "row_count": len(rows),
            "latency_seconds": {"p50": _percentile(durations, 0.50), "p95": _percentile(durations, 0.95),
                                 "max": max(durations, default=None)},
            "wall_seconds": float(time.monotonic() - started),
            "policy": "fixed_cases_and_unfinished_rows_are_reported; no success-rate claim without truth_labels"}


__all__ = ["BenchmarkRunnerError", "BenchmarkBudget", "run_fixed_benchmark"]
