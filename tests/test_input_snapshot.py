import pytest

from core.input_snapshot import InputSnapshotCoordinator, InputSnapshotError


def test_new_input_invalidates_old_run_and_prevents_stale_commit():
    coordinator = InputSnapshotCoordinator()
    old = coordinator.begin({"x": [1, 2]}, version="v1")
    assert coordinator.can_commit(old)
    new = coordinator.begin({"x": [1, 3]}, version="v1")
    assert not coordinator.can_commit(old)
    assert coordinator.commit(new, {"ok": True}) == {"ok": True}
    with pytest.raises(InputSnapshotError, match="stale_or_cancelled"):
        coordinator.commit(old, {"stale": True})


def test_cancelled_run_is_not_reused_and_payload_is_bounded():
    coordinator = InputSnapshotCoordinator()
    snapshot = coordinator.begin({"x": 1}, version="v1")
    coordinator.cancel(snapshot)
    assert not coordinator.can_commit(snapshot)
    with pytest.raises(InputSnapshotError, match="stale_or_cancelled"):
        coordinator.commit(snapshot, 1)
    with pytest.raises(InputSnapshotError, match="too_large"):
        coordinator.begin("x" * 8_000_001, version="v1")
