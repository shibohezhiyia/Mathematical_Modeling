"""Unified bounded asynchronous execution service for typed model contracts.

The service is intentionally small: it owns admission, lifecycle state and
cooperative cancellation, while numerical correctness remains the backend's
responsibility.  It never accepts source code or a callable from HTTP input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
import uuid
import inspect
from typing import Any, Callable, Mapping

from .work_queue import BoundedWorkQueue, WorkQueueError, WorkQueueFull, WorkQueueLimits


class ExecutionServiceError(ValueError):
    pass


@dataclass
class _Task:
    task_id: str
    owner: str
    created_at: float
    status: str = "queued"
    result: Any = None
    error: str | None = None
    cancellation_requested: bool = False
    started_at: float | None = None
    finished_at: float | None = None
    future: Any = None
    cancel: threading.Event = field(default_factory=threading.Event)


class ExecutionService:
    """Bounded task registry with owner-scoped status and cancellation."""

    def __init__(self, limits: WorkQueueLimits | None = None) -> None:
        self._queue = BoundedWorkQueue(limits or WorkQueueLimits(max_workers=2, max_pending=16))
        self._lock = threading.RLock()
        self._tasks: dict[str, _Task] = {}

    def submit(self, owner: str, fn: Callable[..., Any], *args: Any,
               estimated_memory_mb: int = 256, thread_budget: int = 1,
               api_calls: int = 0, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(owner, str) or not owner or len(owner) > 160:
            raise ExecutionServiceError("execution_owner_invalid")
        if not callable(fn):
            raise ExecutionServiceError("execution_callable_invalid")
        task = _Task(uuid.uuid4().hex, owner, time.time())
        with self._lock:
            self._tasks[task.task_id] = task
        def invoke() -> None:
            with self._lock:
                if task.cancel.is_set():
                    task.status = "cancelled"
                    task.finished_at = time.time()
                    return
                task.status = "running"
                task.started_at = time.time()
            try:
                # Pass cancellation only when the callable explicitly opts in;
                # all model backends remain typed and deterministic.
                accepts_cancel = "cancel" in inspect.signature(fn).parameters
                value = fn(*args, cancel=task.cancel, **kwargs) if accepts_cancel else fn(*args, **kwargs)
                with self._lock:
                    task.result = value
                    task.status = "cancelled" if task.cancel.is_set() else "completed"
                    task.finished_at = time.time()
            except Exception as exc:
                with self._lock:
                    task.error = type(exc).__name__
                    task.status = "cancelled" if task.cancel.is_set() else "failed"
                    task.finished_at = time.time()
        try:
            task.future = self._queue.submit(invoke, estimated_memory_mb=estimated_memory_mb,
                                             thread_budget=thread_budget, api_calls=api_calls)
        except (WorkQueueFull, WorkQueueError):
            with self._lock:
                self._tasks.pop(task.task_id, None)
            raise ExecutionServiceError("execution_queue_full")
        return self.status(owner, task.task_id)

    def status(self, owner: str, task_id: str) -> dict[str, Any]:
        task = self._get(owner, task_id)
        with self._lock:
            return self._public(task)

    def cancel(self, owner: str, task_id: str) -> dict[str, Any]:
        task = self._get(owner, task_id)
        with self._lock:
            if task.status in {"completed", "failed", "cancelled"}:
                return self._public(task)
            task.cancellation_requested = True
            task.cancel.set()
            if task.future is not None and task.future.cancel():
                task.status = "cancelled"
                task.finished_at = time.time()
            elif task.status == "queued":
                task.status = "cancelling"
            else:
                task.status = "cancelling"
            return self._public(task)

    def _get(self, owner: str, task_id: str) -> _Task:
        if not isinstance(task_id, str) or len(task_id) != 32:
            raise ExecutionServiceError("execution_task_id_invalid")
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.owner != owner:
                raise ExecutionServiceError("execution_task_not_found")
            return task

    @staticmethod
    def _public(task: _Task) -> dict[str, Any]:
        return {
            "schema_version": "mathmodel.execution-task/v1",
            "task_id": task.task_id, "status": task.status,
            "cancellation_requested": task.cancellation_requested,
            "created_at": task.created_at, "started_at": task.started_at,
            "finished_at": task.finished_at,
            "result": task.result if task.status == "completed" else None,
            "error": task.error if task.status == "failed" else None,
            "policy": "owner_scoped_bounded_queue_cooperative_cancel",
        }

    def shutdown(self) -> None:
        self._queue.shutdown(wait=False, cancel_pending=True)


__all__ = ["ExecutionServiceError", "ExecutionService"]
