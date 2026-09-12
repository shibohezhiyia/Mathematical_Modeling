"""Same-budget paired evaluation for competing modeling pipelines.

Both methods receive the same case order, seed and per-case budget.  Timeouts,
rejections and incomplete rows remain in the denominator; the report never
turns a missing result into a success or a speedup claim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import statistics
import time
import tracemalloc
from typing import Any, Callable, Mapping, Sequence


class PairedBudgetBenchmarkError(ValueError):
    pass


@dataclass(frozen=True)
class PairedBudget:
    wall_seconds_per_method: float = 30.0
    max_cases: int = 256
    max_repeats: int = 3
    seed: int = 0

    def validate(self) -> "PairedBudget":
        if (type(self.wall_seconds_per_method) not in (int, float)
                or not math.isfinite(float(self.wall_seconds_per_method))
                or not 0.01 <= float(self.wall_seconds_per_method) <= 86_400):
            raise PairedBudgetBenchmarkError("invalid_wall_budget")
        if type(self.max_cases) is not int or not 1 <= self.max_cases <= 10_000:
            raise PairedBudgetBenchmarkError("invalid_case_budget")
        if type(self.max_repeats) is not int or not 1 <= self.max_repeats <= 32:
            raise PairedBudgetBenchmarkError("invalid_repeat_budget")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise PairedBudgetBenchmarkError("invalid_seed")
        return self


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(sorted(values)[min(len(values) - 1, max(0, math.ceil(probability * len(values)) - 1))])


def _run_one(evaluate: Callable[..., Mapping[str, Any]], case: Mapping[str, Any],
             *, method: str, seed: int, budget: PairedBudget) -> dict[str, Any]:
    started = time.perf_counter()
    tracemalloc.start()
    row: dict[str, Any] | None = None
    try:
        result = evaluate(case, method=method, seed=seed,
                          wall_seconds=budget.wall_seconds_per_method)
        if not isinstance(result, Mapping):
            raise PairedBudgetBenchmarkError("evaluator_must_return_mapping")
        status = result.get("status", "completed")
        if not isinstance(status, str) or not status.strip() or len(status) > 80:
            raise PairedBudgetBenchmarkError("invalid_status")
        score = result.get("score")
        if score is not None and (type(score) not in (int, float) or not math.isfinite(float(score))):
            raise PairedBudgetBenchmarkError("invalid_score")
        valid = result.get("valid", status in {"completed", "accepted", "passed"})
        if type(valid) is not bool:
            raise PairedBudgetBenchmarkError("invalid_valid_flag")
        api_calls = result.get("api_calls", 0)
        cost = result.get("cost", 0.0)
        if type(api_calls) is not int or api_calls < 0 or type(cost) not in (int, float) or not math.isfinite(float(cost)) or float(cost) < 0:
            raise PairedBudgetBenchmarkError("invalid_resource_metrics")
        row = {"status": status.strip(), "score": score, "valid": valid,
                "api_calls": api_calls, "cost": float(cost),
                "duration_seconds": float(time.perf_counter() - started),
                "error_accept": bool(result.get("error_accept", False))}
    except Exception as exc:
        row = {"status": "failed", "score": None, "valid": False, "api_calls": 0,
                "cost": 0.0, "duration_seconds": float(time.perf_counter() - started),
                "error_code": type(exc).__name__, "error_accept": False}
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if row is not None:
            row["peak_memory_mb"] = float(peak / (1024 * 1024))
    return row or {"status": "failed", "score": None, "valid": False,
                   "api_calls": 0, "cost": 0.0,
                   "duration_seconds": float(time.perf_counter() - started),
                   "peak_memory_mb": 0.0, "error_accept": False}


def run_paired_budget_benchmark(
    cases: Sequence[Mapping[str, Any]], evaluate: Callable[..., Mapping[str, Any]], *,
    budget: PairedBudget | None = None,
) -> dict[str, Any]:
    """Run baseline and treatment with paired seeds and identical budgets."""
    config = (budget or PairedBudget()).validate()
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not 1 <= len(cases) <= config.max_cases:
        raise PairedBudgetBenchmarkError("cases_out_of_bounds")
    if not callable(evaluate):
        raise PairedBudgetBenchmarkError("evaluator_required")
    identifiers = []
    for item in cases:
        if not isinstance(item, Mapping) or type(item.get("id")) is not str or not item["id"].strip():
            raise PairedBudgetBenchmarkError("invalid_case")
        identifiers.append(item["id"].strip())
    if len(set(identifiers)) != len(identifiers):
        raise PairedBudgetBenchmarkError("duplicate_case_id")
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for repeat in range(config.max_repeats):
        for case, case_id in zip(cases, identifiers):
            seed = (config.seed + repeat * 1_000_003 + sum(map(ord, case_id))) % 2**32
            baseline = _run_one(evaluate, case, method="baseline", seed=seed, budget=config)
            treatment = _run_one(evaluate, case, method="treatment", seed=seed, budget=config)
            rows.append({"id": case_id, "repeat": repeat, "seed": seed,
                         "baseline": baseline, "treatment": treatment})

    def summary(method: str) -> dict[str, Any]:
        values = [row[method] for row in rows]
        durations = [float(item["duration_seconds"]) for item in values]
        valid = [item for item in values if item["valid"]]
        return {"rows": len(values), "valid_count": len(valid),
                "valid_rate": len(valid) / len(values),
                "error_accept_count": sum(item["error_accept"] for item in values),
                "api_calls": sum(item["api_calls"] for item in values),
                "cost": sum(item["cost"] for item in values),
                "peak_memory_mb": max((item.get("peak_memory_mb", 0.0) for item in values), default=0.0),
                "latency_seconds": {"p50": _percentile(durations, .5), "p95": _percentile(durations, .95),
                                    "max": max(durations, default=None)},
                "scores": [item["score"] for item in valid if item["score"] is not None]}

    result = {"schema_version": "mathmodel.paired-budget-benchmark/v1",
              "status": "completed", "case_count": len(cases),
              "repeat_count": config.max_repeats, "budget": config.__dict__.copy(),
              "baseline": summary("baseline"), "treatment": summary("treatment"),
              "rows": rows, "wall_seconds": float(time.perf_counter() - started),
              "policy": "paired_same_seed_and_budget;_unfinished_and_error_accept_rows_remain_in_denominator"}
    return result


__all__ = ["PairedBudgetBenchmarkError", "PairedBudget", "run_paired_budget_benchmark"]
