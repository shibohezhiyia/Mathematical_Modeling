"""Bounded scheduler stress protocol for cancellation and resource pressure."""

from __future__ import annotations

from concurrent.futures import Future
import threading
import time
from typing import Any, Callable, Sequence

from .work_queue import BoundedWorkQueue, WorkQueueFull, WorkQueueLimits


class StressProtocolError(ValueError):
    pass


def run_scheduler_stress(
    jobs: Sequence[Callable[[], Any]],
    *,
    limits: WorkQueueLimits | None = None,
    cancel_indices: Sequence[int] = (),
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    """Exercise admission, cancellation and recovery without unbounded work."""
    if not isinstance(jobs, Sequence) or isinstance(jobs, (str, bytes)) or not jobs or any(not callable(job) for job in jobs):
        raise StressProtocolError("jobs_required")
    if not isinstance(cancel_indices, Sequence) or isinstance(cancel_indices, (str, bytes)) or any(type(index) is not int or not 0 <= index < len(jobs) for index in cancel_indices):
        raise StressProtocolError("cancel_indices_invalid")
    if type(timeout_seconds) not in (int, float) or not 0.05 <= float(timeout_seconds) <= 120:
        raise StressProtocolError("invalid_timeout")
    queue = BoundedWorkQueue(limits or WorkQueueLimits())
    futures: list[Future] = []
    statuses: list[str] = []
    try:
        for index, job in enumerate(jobs):
            try:
                future = queue.submit(job, estimated_memory_mb=1)
                if index in set(cancel_indices):
                    future.cancel()
                futures.append(future)
            except WorkQueueFull:
                statuses.append("admission_rejected")
        deadline = time.monotonic() + float(timeout_seconds)
        for future in futures:
            remaining = max(0.0, deadline - time.monotonic())
            try:
                future.result(timeout=remaining)
                statuses.append("completed")
            except Exception as exc:
                statuses.append("cancelled" if future.cancelled() else ("timeout" if isinstance(exc, TimeoutError) else "failed"))
        snapshot = queue.snapshot()
    finally:
        queue.shutdown(wait=False, cancel_pending=True)
    return {
        "schema_version": "mathmodel.stress-protocol/v1",
        "status": "assessed",
        "job_count": len(jobs),
        "statuses": statuses,
        "queue_snapshot": snapshot,
        "policy": "stress_result_is_scheduler_evidence_only; service_status_and_math_validity_require_separate_checks",
    }


__all__ = ["StressProtocolError", "run_scheduler_stress"]
