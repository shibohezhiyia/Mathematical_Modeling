import time

from core.multitask_stress import run_multitask_stress
from core.work_queue import WorkQueueLimits


def test_multitask_stress_records_admission_cancel_failure_and_recovery():
    result = run_multitask_stress(
        [
            {"id": "ok", "fn": lambda: 1},
            {"id": "bad", "fn": lambda: (_ for _ in ()).throw(RuntimeError("x"))},
            {"id": "queued", "fn": lambda: 3, "estimated_memory_mb": 1},
        ],
        limits=WorkQueueLimits(max_workers=1, max_pending=2, max_memory_mb=2,
                               max_native_threads=1, max_api_calls_per_window=8),
        cancel_ids=["queued"],
        timeout_seconds=0.5,
    )
    assert result["status_counts"]["completed"] >= 1
    assert result["status_counts"]["failed"] == 1
    assert result["recovered_reservations"] is True
