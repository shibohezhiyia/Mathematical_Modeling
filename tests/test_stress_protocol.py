from core.stress_protocol import run_scheduler_stress
from core.work_queue import WorkQueueLimits


def test_scheduler_stress_records_admission_and_cancelled_jobs():
    result = run_scheduler_stress([lambda: 1, lambda: 1], limits=WorkQueueLimits(max_workers=1, max_pending=2, max_memory_mb=2), cancel_indices=[1])
    assert result["status"] == "assessed"
    assert "completed" in result["statuses"]
    assert result["queue_snapshot"]["closed"] is False
