"""Bounded in-process work admission for expensive modeling tasks.

This is deliberately a scheduler, not a process sandbox.  Numerical workers
still need :mod:`solver_runtime` supervision.  The queue only guarantees that
the server does not accept more pending work or declared memory than its
configured budget, and that reservations are released on every completion
path.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from collections import deque
import threading
import time
from typing import Any, Callable


class WorkQueueError(ValueError):
    pass


class WorkQueueFull(WorkQueueError):
    pass


@dataclass(frozen=True)
class WorkQueueLimits:
    max_workers: int = 2
    max_pending: int = 16
    max_memory_mb: int = 2048
    max_native_threads: int = 2
    max_api_calls_per_window: int = 64
    api_window_seconds: int = 60

    def __post_init__(self) -> None:
        for name in ("max_workers", "max_pending", "max_memory_mb", "max_native_threads",
                     "max_api_calls_per_window", "api_window_seconds"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise WorkQueueError(f"invalid_{name}")
        if (self.max_workers > 32 or self.max_pending > 10_000 or self.max_memory_mb > 1_048_576
                or self.max_native_threads > 256 or self.max_api_calls_per_window > 1_000_000
                or self.api_window_seconds > 86_400):
            raise WorkQueueError("work_queue_limit_too_large")


class BoundedWorkQueue:
    """Thread-backed admission queue with explicit memory reservations."""

    def __init__(self, limits: WorkQueueLimits | None = None) -> None:
        self.limits = limits or WorkQueueLimits()
        self._executor = ThreadPoolExecutor(max_workers=self.limits.max_workers,
                                            thread_name_prefix="mathmodel-worker")
        self._lock = threading.RLock()
        self._pending = 0
        self._reserved_memory = 0
        self._reserved_native_threads = 0
        self._api_call_times: deque[float] = deque()
        self._submitted = 0
        self._completed = 0
        self._closed = False

    def submit(self, fn: Callable[..., Any], *args: Any,
               estimated_memory_mb: int = 1, thread_budget: int = 1,
               api_calls: int = 0, **kwargs: Any) -> Future:
        if not callable(fn):
            raise WorkQueueError("work_must_be_callable")
        if (type(estimated_memory_mb) is not int
                or not 1 <= estimated_memory_mb <= self.limits.max_memory_mb):
            raise WorkQueueError("invalid_estimated_memory")
        if (type(thread_budget) is not int or not 1 <= thread_budget <= self.limits.max_native_threads):
            raise WorkQueueError("invalid_thread_budget")
        if type(api_calls) is not int or not 0 <= api_calls <= self.limits.max_api_calls_per_window:
            raise WorkQueueError("invalid_api_calls")
        with self._lock:
            if self._closed:
                raise WorkQueueError("work_queue_closed")
            if self._pending >= self.limits.max_pending:
                raise WorkQueueFull("pending_work_limit_reached")
            if self._reserved_memory + estimated_memory_mb > self.limits.max_memory_mb:
                raise WorkQueueFull("memory_reservation_limit_reached")
            if self._reserved_native_threads + thread_budget > self.limits.max_native_threads:
                raise WorkQueueFull("native_thread_limit_reached")
            now = time.monotonic()
            cutoff = now - self.limits.api_window_seconds
            while self._api_call_times and self._api_call_times[0] < cutoff:
                self._api_call_times.popleft()
            if len(self._api_call_times) + api_calls > self.limits.max_api_calls_per_window:
                raise WorkQueueFull("api_rate_limit_reached")
            self._pending += 1
            self._reserved_memory += estimated_memory_mb
            self._reserved_native_threads += thread_budget
            self._api_call_times.extend([now] * api_calls)
            self._submitted += 1
        try:
            future = self._executor.submit(fn, *args, **kwargs)
        except BaseException:
            with self._lock:
                # Roll back only the calls just appended.  Submission holds
                # the lock while reserving; a failed executor submission is
                # therefore the only path that must undo the rate window.
                for _ in range(api_calls):
                    if self._api_call_times:
                        self._api_call_times.pop()
            self._release(estimated_memory_mb, thread_budget, api_calls, completed=False)
            raise
        future.add_done_callback(lambda _: self._release(estimated_memory_mb, thread_budget, api_calls))
        return future

    def _release(self, memory_mb: int, thread_budget: int, api_calls: int, *, completed: bool = True) -> None:
        with self._lock:
            self._pending = max(0, self._pending - 1)
            self._reserved_memory = max(0, self._reserved_memory - memory_mb)
            self._reserved_native_threads = max(0, self._reserved_native_threads - thread_budget)
            if completed:
                self._completed += 1

    def snapshot(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "pending": self._pending,
                "reserved_memory_mb": self._reserved_memory,
                "reserved_native_threads": self._reserved_native_threads,
                "api_calls_in_window": len(self._api_call_times),
                "submitted": self._submitted,
                "completed": self._completed,
                "closed": self._closed,
            }

    def shutdown(self, *, wait: bool = True, cancel_pending: bool = False) -> None:
        if type(wait) is not bool or type(cancel_pending) is not bool:
            raise WorkQueueError("shutdown_flags_must_be_boolean")
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=cancel_pending)


__all__ = ["WorkQueueError", "WorkQueueFull", "WorkQueueLimits", "BoundedWorkQueue"]
