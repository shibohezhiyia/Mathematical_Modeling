import threading
import time

import pytest

from core.work_queue import BoundedWorkQueue, WorkQueueError, WorkQueueFull, WorkQueueLimits


def test_work_queue_releases_memory_and_pending_reservations_on_success_and_failure():
    queue = BoundedWorkQueue(WorkQueueLimits(max_workers=1, max_pending=2, max_memory_mb=4))
    try:
        good = queue.submit(lambda: 3, estimated_memory_mb=3)
        bad = queue.submit(lambda: (_ for _ in ()).throw(RuntimeError("boom")), estimated_memory_mb=1)
        assert good.result() == 3
        with pytest.raises(RuntimeError):
            bad.result()
        assert queue.snapshot()["pending"] == 0
        assert queue.snapshot()["reserved_memory_mb"] == 0
    finally:
        queue.shutdown()


def test_work_queue_rejects_pending_and_memory_overcommit_without_blocking():
    gate = threading.Event()
    queue = BoundedWorkQueue(WorkQueueLimits(max_workers=1, max_pending=1, max_memory_mb=2))
    try:
        first = queue.submit(gate.wait, estimated_memory_mb=2)
        with pytest.raises(WorkQueueFull, match="pending"):
            queue.submit(lambda: None, estimated_memory_mb=1)
        gate.set()
        first.result(timeout=2)
    finally:
        queue.shutdown()


def test_work_queue_rejects_coercion_and_submit_after_shutdown():
    with pytest.raises(WorkQueueError):
        WorkQueueLimits(max_workers="2")
    queue = BoundedWorkQueue()
    queue.shutdown()
    with pytest.raises(WorkQueueError, match="closed"):
        queue.submit(lambda: None)


def test_work_queue_limits_native_threads_and_api_calls():
    gate = threading.Event()
    queue = BoundedWorkQueue(WorkQueueLimits(
        max_workers=2, max_pending=3, max_memory_mb=4,
        max_native_threads=1, max_api_calls_per_window=1,
    ))
    try:
        first = queue.submit(gate.wait, estimated_memory_mb=1, api_calls=1)
        with pytest.raises(WorkQueueFull, match="native_thread"):
            queue.submit(lambda: None, estimated_memory_mb=1)
        gate.set()
        first.result(timeout=2)
        # API calls remain rate-limited for the configured window even after
        # the worker finishes; this prevents bursty nested LLM calls.
        with pytest.raises(WorkQueueFull, match="api_rate"):
            queue.submit(lambda: None, estimated_memory_mb=1, api_calls=1)
    finally:
        queue.shutdown()
