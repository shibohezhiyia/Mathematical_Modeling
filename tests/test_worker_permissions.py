from pathlib import Path
import os

import pytest

from core.worker_permissions import (
    WorkerPermissionError,
    WorkerPermissionPolicy,
    check_audit_event,
    directory_usage_bytes,
    enforce_write_quota,
)


def test_permission_policy_allows_declared_root_but_denies_network_and_subprocess(tmp_path):
    policy = WorkerPermissionPolicy((str(tmp_path),)).validate()
    check_audit_event(policy, "open", (str(tmp_path / "result.json"), "w"))
    with pytest.raises(WorkerPermissionError, match="network"):
        check_audit_event(policy, "socket.connect", ("example.org", 443))
    with pytest.raises(WorkerPermissionError, match="subprocess"):
        check_audit_event(policy, "subprocess.Popen", ("cmd",))
        with pytest.raises(WorkerPermissionError, match="filesystem"):
            check_audit_event(policy, "open", (str(Path(tmp_path).parent / "outside"), "w"))


def test_permission_policy_denies_socket_creation_before_connect(tmp_path):
    policy = WorkerPermissionPolicy((str(tmp_path),)).validate()
    with pytest.raises(WorkerPermissionError, match="network"):
        check_audit_event(policy, "socket.__new__", (object(), 2, 1, 0))


def test_permission_policy_rejects_broad_root_and_bad_quota(tmp_path):
    with pytest.raises(WorkerPermissionError, match="broad"):
        WorkerPermissionPolicy((Path(tmp_path).anchor,)).validate()
    with pytest.raises(WorkerPermissionError, match="quota"):
        WorkerPermissionPolicy((str(tmp_path),), max_write_bytes=0).validate()


def test_permission_policy_allows_existing_descriptors_and_devnull():
    policy = WorkerPermissionPolicy((str(Path.cwd()),)).validate()
    check_audit_event(policy, "open", (os.devnull, "w"))
    check_audit_event(policy, "open", (1, "w"))


def test_permission_policy_checks_destructive_and_two_path_filesystem_events(tmp_path):
    policy = WorkerPermissionPolicy((str(tmp_path),)).validate()
    check_audit_event(policy, "os.remove", (str(tmp_path / "result.json"),))
    check_audit_event(policy, "os.rename", (str(tmp_path / "a"), str(tmp_path / "b")))
    with pytest.raises(WorkerPermissionError, match="filesystem"):
        check_audit_event(policy, "os.remove", (str(Path(tmp_path).parent / "outside"),))
    with pytest.raises(WorkerPermissionError, match="filesystem"):
        check_audit_event(policy, "os.rename", (str(tmp_path / "a"), str(Path(tmp_path).parent / "outside")))


def test_worker_write_quota_reports_usage_and_blocks_new_writes(tmp_path):
    payload = tmp_path / "existing.bin"
    payload.write_bytes(b"1234567890")
    policy = WorkerPermissionPolicy((str(tmp_path),), max_write_bytes=10).validate()
    assert directory_usage_bytes(tmp_path) == 10
    assert enforce_write_quota(policy)["remaining_bytes"] == 0
    with pytest.raises(WorkerPermissionError, match="write_quota"):
        check_audit_event(policy, "open", (str(tmp_path / "new.bin"), "w"))
