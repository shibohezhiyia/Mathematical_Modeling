"""Cold/warm/incremental cache benchmark with output consistency checks."""

from __future__ import annotations

from hashlib import sha256
import json
import math
import time
import tracemalloc
from typing import Any, Callable, Mapping, Sequence

from .cache_consistency import compare_cached_uncached


class CachePerformanceBenchmarkError(ValueError):
    pass


def _digest(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                         separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise CachePerformanceBenchmarkError("result_must_be_finite_json") from exc
    return sha256(raw).hexdigest()


def _measure(run: Callable[[], Any]) -> tuple[Any, float, float]:
    tracemalloc.start()
    started = time.perf_counter()
    result = run()
    duration = time.perf_counter() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, float(duration), float(peak / (1024 * 1024))


def run_cache_performance_benchmark(
    cases: Sequence[Mapping[str, Any]], runner: Callable[..., Any], *,
    max_cases: int = 256, seed: int = 0,
) -> dict[str, Any]:
    """Measure cold, warm and incremental calls without treating timing as proof."""
    if not callable(runner):
        raise CachePerformanceBenchmarkError("runner_required")
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)) or not 1 <= len(cases) <= max_cases:
        raise CachePerformanceBenchmarkError("cases_out_of_bounds")
    if type(seed) is not int:
        raise CachePerformanceBenchmarkError("seed_must_be_integer")
    rows = []
    for case in cases:
        if not isinstance(case, Mapping) or type(case.get("id")) is not str or not case["id"].strip():
            raise CachePerformanceBenchmarkError("invalid_case")
        case_id = case["id"].strip()
        try:
            cold, cold_time, cold_memory = _measure(
                lambda: runner(case, cache_enabled=False, incremental=False, seed=seed))
            warm_first, warm_first_time, warm_first_memory = _measure(
                lambda: runner(case, cache_enabled=True, incremental=False, seed=seed))
            warm_second, warm_second_time, warm_second_memory = _measure(
                lambda: runner(case, cache_enabled=True, incremental=False, seed=seed))
            incremental, incremental_time, incremental_memory = _measure(
                lambda: runner(case, cache_enabled=True, incremental=True, seed=seed))
            consistency = compare_cached_uncached(lambda: cold, lambda: warm_second)
            rows.append({"id": case_id, "status": "completed", "cold": {
                "duration_seconds": cold_time, "peak_memory_mb": cold_memory, "digest": _digest(cold)},
                "warm_first": {"duration_seconds": warm_first_time, "peak_memory_mb": warm_first_memory,
                               "digest": _digest(warm_first)},
                "warm_second": {"duration_seconds": warm_second_time, "peak_memory_mb": warm_second_memory,
                                "digest": _digest(warm_second)},
                "incremental": {"duration_seconds": incremental_time, "peak_memory_mb": incremental_memory,
                                "digest": _digest(incremental)},
                "consistency": consistency})
        except Exception as exc:
            rows.append({"id": case_id, "status": "not_assessed", "error_code": type(exc).__name__})
    assessed = [row for row in rows if row["status"] == "completed"]
    consistent = [row for row in assessed if row["consistency"].get("status") == "tested_not_falsified"]
    return {
        "schema_version": "mathmodel.cache-performance-benchmark/v1",
        "status": "assessed" if len(assessed) == len(rows) else "partial",
        "case_count": len(rows), "assessed_count": len(assessed),
        "consistent_count": len(consistent), "rows": rows,
        "policy": "cold_warm_incremental_measurement_only;timing_is_not_speedup_proof_and_output_consistency_is_finite",
    }


__all__ = ["CachePerformanceBenchmarkError", "run_cache_performance_benchmark"]
