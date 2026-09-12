"""Deterministic multi-task scheduler stress protocol.

The protocol exercises admission, cancellation, timeout accounting and queue
recovery with trusted callables.  It is not an OS memory test and must not be
reported as a main-service or solver sandbox certification.
"""

from __future__ import annotations

from concurrent.futures import TimeoutError as FutureTimeout
import math
import time
from typing import Any, Mapping, Sequence

from .work_queue import BoundedWorkQueue, WorkQueueFull, WorkQueueLimits


class MultitaskStressError(ValueError):
    pass


def run_multitask_stress(
    tasks: Sequence[Mapping[str, Any]],
    *,
    limits: WorkQueueLimits | None = None,
    cancel_ids: Sequence[str] = (),
    timeout_seconds: float = 1.0,
) -> dict[str, Any]:
    if not isinstance(tasks, Sequence) or isinstance(tasks, (str, bytes)) or not 1 <= len(tasks) <= 256:
        raise MultitaskStressError("tasks_must_be_bounded_sequence")
    if not isinstance(cancel_ids, Sequence) or isinstance(cancel_ids, (str, bytes)):
        raise MultitaskStressError("cancel_ids_must_be_sequence")
    if type(timeout_seconds) not in (int, float) or not math.isfinite(float(timeout_seconds)) or not 0.01 <= float(timeout_seconds) <= 60:
        raise MultitaskStressError("invalid_timeout_seconds")
    normalized = []
    seen = set()
    for task in tasks:
        if not isinstance(task, Mapping) or not isinstance(task.get("id"), str) or not task["id"].strip():
            raise MultitaskStressError("task_id_required")
        identifier = task["id"].strip()
        if identifier in seen:
            raise MultitaskStressError("duplicate_task_id")
        if not callable(task.get("fn")):
            raise MultitaskStressError("task_fn_must_be_callable")
        seen.add(identifier)
        normalized.append((identifier, task))
    unknown_cancel = {str(item) for item in cancel_ids} - seen
    if unknown_cancel:
        raise MultitaskStressError("cancel_id_not_found")
    queue = BoundedWorkQueue(limits or WorkQueueLimits())
    futures = {}
    records = []
    try:
        for identifier, task in normalized:
            try:
                futures[identifier] = queue.submit(
                    task["fn"],
                    estimated_memory_mb=task.get("estimated_memory_mb", 1),
                    thread_budget=task.get("thread_budget", 1),
                    api_calls=task.get("api_calls", 0),
                )
            except WorkQueueFull as exc:
                records.append({"id": identifier, "status": "admission_rejected", "reason": str(exc)})
        for identifier, _task in normalized:
            future = futures.get(identifier)
            if future is None:
                continue
            if identifier in cancel_ids and future.cancel():
                records.append({"id": identifier, "status": "cancelled"})
                continue
            try:
                value = future.result(timeout=float(timeout_seconds))
                records.append({"id": identifier, "status": "completed", "result_type": type(value).__name__})
            except FutureTimeout:
                records.append({"id": identifier, "status": "timeout"})
            except Exception as exc:
                records.append({"id": identifier, "status": "failed", "error_type": type(exc).__name__})
    finally:
        queue.shutdown(wait=False, cancel_pending=True)
    snapshot = queue.snapshot()
    return {
        "schema_version": "mathmodel.multitask-stress/v1",
        "records": records,
        "status_counts": {status: sum(item["status"] == status for item in records)
                           for status in {item["status"] for item in records}},
        "recovered_reservations": snapshot["reserved_memory_mb"] == 0 and snapshot["reserved_native_threads"] == 0,
        "queue_snapshot": snapshot,
        "policy": "scheduler_protocol_only;_not_os_memory_or_main_service_certification",
    }


__all__ = ["MultitaskStressError", "run_multitask_stress"]
