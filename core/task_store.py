"""Small durable task ledger used by supervised execution services."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any


class TaskStore:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        if self.path.anchor == self.path:
            raise ValueError("task_store_root_invalid")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE TABLE IF NOT EXISTS execution_tasks (
                task_id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL,
                created_at REAL NOT NULL, started_at REAL, finished_at REAL,
                cancellation_requested INTEGER NOT NULL DEFAULT 0,
                result_json TEXT, error TEXT
            )""")

    def _connect(self):
        db = sqlite3.connect(str(self.path), timeout=5, check_same_thread=False)
        db.row_factory = sqlite3.Row
        return db

    def create(self, task_id: str, owner: str) -> None:
        with self._lock, self._connect() as db:
            db.execute("INSERT INTO execution_tasks(task_id,owner,status,created_at) VALUES(?,?,?,?)",
                       (task_id, owner, "queued", time.time()))

    def update(self, task_id: str, **fields: Any) -> None:
        allowed = {"status", "started_at", "finished_at", "cancellation_requested", "result_json", "error"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError("task_store_field_invalid")
        if not fields:
            return
        assignments = ",".join(f"{key}=?" for key in fields)
        values = [fields[key] for key in fields] + [task_id]
        with self._lock, self._connect() as db:
            db.execute(f"UPDATE execution_tasks SET {assignments} WHERE task_id=?", values)

    def get(self, task_id: str, owner: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as db:
            row = db.execute("SELECT * FROM execution_tasks WHERE task_id=? AND owner=?", (task_id, owner)).fetchone()
        if row is None:
            return None
        result = None
        if row["result_json"]:
            try:
                result = json.loads(row["result_json"])
            except (TypeError, ValueError, json.JSONDecodeError):
                result = None
        return {"schema_version": "mathmodel.execution-task/v1", "task_id": row["task_id"],
                "status": row["status"], "cancellation_requested": bool(row["cancellation_requested"]),
                "created_at": row["created_at"], "started_at": row["started_at"],
                "finished_at": row["finished_at"], "result": result,
                "error": row["error"], "policy": "durable_owner_scoped_task_ledger"}

    def mark_orphans(self, *, max_age_seconds: float = 0) -> int:
        """Recover stale rows after a service restart.

        Fresh queued rows may belong to another web worker, so only rows whose
        last activity is older than the recovery lease are interrupted.
        """
        cutoff = time.time() - max(0.0, float(max_age_seconds))
        with self._lock, self._connect() as db:
            cursor = db.execute("""UPDATE execution_tasks SET status='interrupted', finished_at=?
                WHERE status IN ('queued','running','cancelling')
                AND COALESCE(started_at, created_at) < ?""", (time.time(), cutoff))
            return int(cursor.rowcount)

    def admit(self, task_id: str, max_workers: int) -> bool:
        """Atomically enforce a cross-process active-task limit."""
        if type(max_workers) is not int or max_workers < 1:
            raise ValueError("task_store_worker_limit_invalid")
        with self._lock, self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM execution_tasks WHERE task_id=?", (task_id,)).fetchone()
            active = db.execute("SELECT COUNT(*) FROM execution_tasks WHERE status IN ('queued','running','cancelling')").fetchone()[0]
            if row is None or row[0] != "queued" or active > max_workers:
                return False
            return True


__all__ = ["TaskStore"]
