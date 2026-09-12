"""Hard-terminable process execution for the typed dynamic compiler."""
from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
import queue
import threading
import time
import uuid
from typing import Any

from .task_store import TaskStore
from .execution_capabilities import get_capabilities, validate_budget


def _dynamic_worker(kind: str, contract: dict[str, Any], output) -> None:
    """Spawn target; only the internal typed compiler is reachable."""
    try:
        from .dynamic_model_compiler import compile_and_execute_model
        output.put(("ok", compile_and_execute_model(kind, contract)))
    except Exception as exc:
        output.put(("error", type(exc).__name__))


def _research_worker(payload: dict[str, Any], output) -> None:
    try:
        from .research_process import run_research_payload
        output.put(("ok", run_research_payload(payload)))
    except Exception as exc:
        output.put(("error", type(exc).__name__))

def _training_worker(payload: dict[str, Any], output) -> None:
    try:
        from .training_process import run_training_payload
        output.put(("ok", run_training_payload(payload)))
    except Exception as exc:
        output.put(("error", type(exc).__name__ + ": " + str(exc)))


@dataclass
class _Runtime:
    owner: str
    process: Any
    output: Any
    cancel: threading.Event
    timeout: float


class ProcessExecutionService:
    """Durable owner-scoped tasks with cancellation that kills the worker."""
    def __init__(self, store: TaskStore | None = None, *, max_workers: int = 2):
        if type(max_workers) is not int or not 1 <= max_workers <= 16:
            raise ValueError("process_worker_limit_invalid")
        self.store = store
        if self.store is not None:
            # A process cannot survive a service restart with a valid handle;
            # make persisted in-flight rows explicitly recoverable instead of
            # leaving them in a misleading "running" state.
            self.store.mark_orphans(max_age_seconds=3600)
        self.context = mp.get_context("spawn")
        self.max_workers = max_workers
        self._lock = threading.RLock()
        self._runtime: dict[str, _Runtime] = {}
        self._semaphore = threading.BoundedSemaphore(max_workers)

    def submit_dynamic(self, owner: str, kind: str, contract: dict[str, Any], *, wall_seconds: float = 120) -> dict[str, Any]:
        if not isinstance(owner, str) or not owner or len(owner) > 160:
            raise ValueError("execution_owner_invalid")
        if not isinstance(kind, str) or not isinstance(contract, dict):
            raise ValueError("execution_contract_invalid")
        wall_seconds = validate_budget(kind, wall_seconds)
        task_id = uuid.uuid4().hex
        if self.store:
            self.store.create(task_id, owner)
            if not self.store.admit(task_id, self.max_workers):
                self.store.update(task_id, status="rejected", finished_at=time.time(), error="queue_full")
                raise ValueError("execution_queue_full")
        if not self._semaphore.acquire(blocking=False):
            if self.store:
                self.store.update(task_id, status="rejected", finished_at=time.time(), error="queue_full")
            raise ValueError("execution_queue_full")
        output = self.context.Queue(maxsize=1)
        process = self.context.Process(target=_dynamic_worker, args=(kind, contract, output), name=f"mathmodel-{task_id[:12]}", daemon=False)
        runtime = _Runtime(owner, process, output, threading.Event(), float(wall_seconds))
        with self._lock:
            self._runtime[task_id] = runtime
        try:
            process.start()
        except Exception:
            self._finish(task_id, "failed", error="worker_start_failed")
            raise ValueError("worker_start_failed")
        if self.store:
            self.store.update(task_id, status="running", started_at=time.time())
        threading.Thread(target=self._watch, args=(task_id,), daemon=True).start()
        return self.status(owner, task_id)

    def submit_research(self, owner: str, payload: dict[str, Any], *, wall_seconds: float = 600) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("research_payload_invalid")
        validate_budget("research", wall_seconds)
        return self._submit_process(owner, _research_worker, (payload,), wall_seconds=wall_seconds)

    def submit_training(self, owner: str, payload: dict[str, Any], *, wall_seconds: float = 1800) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("training_payload_invalid")
        validate_budget("training", wall_seconds)
        return self._submit_process(owner, _training_worker, (payload,), wall_seconds=wall_seconds)

    @staticmethod
    def capabilities() -> dict[str, dict[str, Any]]:
        return get_capabilities()

    def _submit_process(self, owner: str, target, args: tuple, *, wall_seconds: float):
        if not isinstance(owner, str) or not owner:
            raise ValueError("execution_owner_invalid")
        if type(wall_seconds) not in (int, float) or not 0 < float(wall_seconds) <= 1800:
            raise ValueError("execution_wall_limit_invalid")
        task_id = uuid.uuid4().hex
        if self.store:
            self.store.create(task_id, owner)
            if not self.store.admit(task_id, self.max_workers):
                self.store.update(task_id, status="rejected", finished_at=time.time(), error="queue_full")
                raise ValueError("execution_queue_full")
        if not self._semaphore.acquire(blocking=False):
            if self.store:
                self.store.update(task_id, status="rejected", finished_at=time.time(), error="queue_full")
            raise ValueError("execution_queue_full")
        output = self.context.Queue(maxsize=1)
        # Non-daemon is intentional: model backends may use their own worker
        # pools. The supervisor still owns the PID and can terminate/kill it.
        process = self.context.Process(target=target, args=(*args, output), name=f"mathmodel-{task_id[:12]}", daemon=False)
        runtime = _Runtime(owner, process, output, threading.Event(), float(wall_seconds))
        with self._lock:
            self._runtime[task_id] = runtime
        try:
            process.start()
        except Exception:
            self._finish(task_id, "failed", error="worker_start_failed")
            raise ValueError("worker_start_failed")
        if self.store:
            self.store.update(task_id, status="running", started_at=time.time())
        threading.Thread(target=self._watch, args=(task_id,), daemon=True).start()
        return self.status(owner, task_id)

    def _watch(self, task_id: str) -> None:
        with self._lock:
            runtime = self._runtime.get(task_id)
        if runtime is None:
            return
        deadline = time.monotonic() + runtime.timeout
        message = None
        while runtime.process.is_alive():
            if runtime.cancel.is_set():
                self._terminate(runtime.process)
                self._finish(task_id, "cancelled")
                return
            if time.monotonic() >= deadline:
                self._terminate(runtime.process)
                self._finish(task_id, "timeout", error="wall_timeout")
                return
            try:
                message = runtime.output.get(timeout=0.05)
                break
            except queue.Empty:
                continue
        runtime.process.join(timeout=0.5)
        if message is None:
            try:
                message = runtime.output.get_nowait()
            except queue.Empty:
                message = ("error", "worker_exit")
        if runtime.cancel.is_set():
            self._finish(task_id, "cancelled")
        elif message[0] == "ok":
            self._finish(task_id, "completed", result=message[1])
        else:
            self._finish(task_id, "failed", error=str(message[1]))

    @staticmethod
    def _terminate(process) -> None:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(timeout=1)

    def _finish(self, task_id: str, status: str, *, result=None, error=None) -> None:
        with self._lock:
            runtime = self._runtime.get(task_id)
            if runtime is None:
                return
        if self.store:
            fields = {"status": status, "finished_at": time.time()}
            if result is not None:
                import json
                fields["result_json"] = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
            if error is not None:
                fields["error"] = error
            self.store.update(task_id, **fields)
        with self._lock:
            self._runtime.pop(task_id, None)
        self._semaphore.release()
        if runtime is not None:
            try:
                runtime.output.close()
                runtime.output.join_thread()
            except (OSError, AttributeError):
                pass

    def status(self, owner: str, task_id: str) -> dict[str, Any]:
        if self.store:
            task = self.store.get(task_id, owner)
            if task is None:
                raise ValueError("execution_task_not_found")
            return task
        raise ValueError("execution_store_required")

    def wait(self, owner: str, task_id: str, *, timeout: float | None = None) -> dict[str, Any]:
        """Wait for a task while the child watcher retains hard timeout control."""
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            task = self.status(owner, task_id)
            if task["status"] in {"completed", "failed", "cancelled", "timeout", "interrupted", "rejected"}:
                return task
            if deadline is not None and time.monotonic() >= deadline:
                self.cancel(owner, task_id)
                return self.status(owner, task_id)
            time.sleep(0.02)

    def cancel(self, owner: str, task_id: str) -> dict[str, Any]:
        task = self.status(owner, task_id)
        if task["status"] in {"completed", "failed", "cancelled", "timeout", "interrupted", "rejected"}:
            return task
        with self._lock:
            runtime = self._runtime.get(task_id)
            if runtime is not None:
                runtime.cancel.set()
        if self.store:
            self.store.update(task_id, status="cancelling", cancellation_requested=1)
        return self.status(owner, task_id)

    def shutdown(self) -> None:
        with self._lock:
            running = list(self._runtime.items())
        for task_id, runtime in running:
            runtime.cancel.set()
            self._terminate(runtime.process)
            self._finish(task_id, "cancelled")


__all__ = ["ProcessExecutionService"]
