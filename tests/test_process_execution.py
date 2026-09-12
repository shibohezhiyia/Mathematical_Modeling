import time
from pathlib import Path

import pytest

from core.process_execution import ProcessExecutionService
from core.task_store import TaskStore


def _wait(service, owner, task_id, terminal={"completed", "failed", "cancelled", "timeout"}):
    for _ in range(200):
        task = service.status(owner, task_id)
        if task["status"] in terminal:
            return task
        time.sleep(0.01)
    raise AssertionError(task)


def _dynamic_payload():
    return {
        "nodes": [
            {"id": "x", "op": "variable", "inputs": [], "kind": "quantity", "dimensions": {}, "attributes": {"name": "x"}},
            {"id": "two", "op": "constant", "inputs": [], "kind": "quantity", "dimensions": {}, "attributes": {"value": 2}},
            {"id": "y", "op": "multiply", "inputs": ["x", "two"], "kind": "quantity", "dimensions": {}, "attributes": {}},
        ], "bindings": {"x": 3}, "output_ids": ["y"],
    }


def _hang_worker(kind, contract, output):
    time.sleep(30)


def test_process_service_persists_and_executes_typed_backend(tmp_path):
    service = ProcessExecutionService(TaskStore(tmp_path / "tasks.sqlite3"), max_workers=1)
    try:
        created = service.submit_dynamic("owner", "primitive_graph", _dynamic_payload(), wall_seconds=20)
        task = _wait(service, "owner", created["task_id"])
        assert task["status"] == "completed"
        assert task["result"]["result"]["outputs"]["y"] == 6
        restarted = TaskStore(tmp_path / "tasks.sqlite3")
        assert restarted.get(created["task_id"], "owner")["status"] == "completed"
    finally:
        for runtime in list(service._runtime.values()):
            service._terminate(runtime.process)


def test_process_service_timeout_terminates_worker_and_persists(tmp_path, monkeypatch):
    # Replace the internal typed worker only for a bounded lifecycle test.
    import core.process_execution as module
    original = module._dynamic_worker
    monkeypatch.setattr(module, "_dynamic_worker", _hang_worker)
    service = ProcessExecutionService(TaskStore(tmp_path / "tasks.sqlite3"), max_workers=1)
    try:
        created = service.submit_dynamic("owner", "primitive_graph", _dynamic_payload(), wall_seconds=0.2)
        task = _wait(service, "owner", created["task_id"])
        assert task["status"] == "timeout"
        assert service._runtime == {}
    finally:
        monkeypatch.setattr(module, "_dynamic_worker", original)


def test_process_service_cancel_terminates_running_worker(tmp_path, monkeypatch):
    import core.process_execution as module
    monkeypatch.setattr(module, "_dynamic_worker", _hang_worker)
    service = ProcessExecutionService(TaskStore(tmp_path / "tasks.sqlite3"), max_workers=1)
    try:
        created = service.submit_dynamic("owner", "primitive_graph", _dynamic_payload(), wall_seconds=20)
        time.sleep(0.1)
        service.cancel("owner", created["task_id"])
        task = _wait(service, "owner", created["task_id"])
        assert task["status"] == "cancelled"
        assert task["cancellation_requested"] is True
    finally:
        for runtime in list(service._runtime.values()):
            service._terminate(runtime.process)


def test_task_store_marks_inflight_rows_interrupted_after_restart(tmp_path):
    store = TaskStore(tmp_path / "tasks.sqlite3")
    store.create("a" * 32, "owner")
    store.update("a" * 32, status="running", started_at=time.time())
    assert TaskStore(tmp_path / "tasks.sqlite3").mark_orphans() == 1
    assert store.get("a" * 32, "owner")["status"] == "interrupted"
