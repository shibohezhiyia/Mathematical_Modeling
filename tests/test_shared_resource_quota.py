import multiprocessing
import time

import pytest

from core.shared_resource_quota import SharedQuotaError, SharedQuotaLimits, SharedResourceQuota


def _acquire_in_child(database, ready):
    quota = SharedResourceQuota(database, limits=SharedQuotaLimits(max_memory_mb=100, max_slots=1))
    lease = quota.acquire("child", memory_mb=80)
    ready.put(lease is not None)
    if lease:
        time.sleep(0.2)
        quota.release(lease["token"])


def test_shared_quota_is_atomic_across_instances_and_releases(tmp_path):
    db = tmp_path / "quota.sqlite3"
    left = SharedResourceQuota(db, limits=SharedQuotaLimits(max_memory_mb=100, max_slots=1))
    right = SharedResourceQuota(db, limits=SharedQuotaLimits(max_memory_mb=100, max_slots=1))
    lease = left.acquire("worker-a", memory_mb=80)
    assert lease and right.acquire("worker-b", memory_mb=30) is None
    assert right.snapshot()["reserved_memory_mb"] == 80
    assert right.release(lease["token"]) is True
    assert right.acquire("worker-b", memory_mb=30)


def test_shared_quota_reclaims_expired_leases_and_validates_requests(tmp_path):
    quota = SharedResourceQuota(tmp_path / "quota.sqlite3", limits=SharedQuotaLimits(max_memory_mb=10, max_slots=1))
    lease = quota.acquire("short", memory_mb=5, ttl_seconds=1)
    assert lease
    time.sleep(1.05)
    assert quota.acquire("new", memory_mb=10)
    with pytest.raises(SharedQuotaError, match="owner_required"):
        quota.acquire("", memory_mb=1)


def test_shared_quota_rejects_inconsistent_limits_across_workers(tmp_path):
    db = tmp_path / "quota.sqlite3"
    SharedResourceQuota(db, limits=SharedQuotaLimits(max_memory_mb=100, max_slots=2))
    with pytest.raises(SharedQuotaError, match="shared_quota_limits_mismatch"):
        SharedResourceQuota(db, limits=SharedQuotaLimits(max_memory_mb=200, max_slots=2))


def test_shared_quota_is_atomic_across_real_processes(tmp_path):
    db = str(tmp_path / "quota.sqlite3")
    SharedResourceQuota(db, limits=SharedQuotaLimits(max_memory_mb=100, max_slots=1))
    context = multiprocessing.get_context("spawn")
    ready = context.Queue()
    workers = [context.Process(target=_acquire_in_child, args=(db, ready)) for _ in range(2)]
    for worker in workers:
        worker.start()
    outcomes = [ready.get(timeout=15) for _ in workers]
    for worker in workers:
        worker.join(timeout=15)
    assert sorted(outcomes) == [False, True]
    assert all(worker.exitcode == 0 for worker in workers)
