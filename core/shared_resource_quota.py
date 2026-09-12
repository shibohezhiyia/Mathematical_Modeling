"""SQLite-backed cross-process resource leases.

`BoundedWorkQueue` and the solver semaphore are process-local.  This module is
an opt-in coordinator for deployments with several service workers: a short
SQLite transaction atomically reserves memory/slots, expired leases are
reclaimed, and no numerical result is stored in the quota database.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Any


class SharedQuotaError(ValueError):
    pass


@dataclass(frozen=True)
class SharedQuotaLimits:
    max_memory_mb: int = 4096
    max_slots: int = 8
    lease_seconds: float = 120.0

    def validate(self) -> "SharedQuotaLimits":
        if type(self.max_memory_mb) is not int or not 1 <= self.max_memory_mb <= 1_048_576:
            raise SharedQuotaError("invalid_shared_memory_limit")
        if type(self.max_slots) is not int or not 1 <= self.max_slots <= 4096:
            raise SharedQuotaError("invalid_shared_slot_limit")
        if type(self.lease_seconds) not in (int, float) or not 1 <= float(self.lease_seconds) <= 86_400:
            raise SharedQuotaError("invalid_shared_lease_seconds")
        return self


class SharedResourceQuota:
    """Atomic, bounded lease store safe for multiple Python processes."""

    def __init__(self, database: str | os.PathLike[str], *, limits: SharedQuotaLimits | None = None):
        path = Path(database).expanduser().resolve()
        if path.name in {"", ".", ".."} or path.parent == path:
            raise SharedQuotaError("quota_database_path_invalid")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.database = path
        self.limits = (limits or SharedQuotaLimits()).validate()
        self._local = threading.local()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.database), timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as db:
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("CREATE TABLE IF NOT EXISTS leases (token TEXT PRIMARY KEY, owner TEXT NOT NULL, memory_mb INTEGER NOT NULL, slots INTEGER NOT NULL, expires_at REAL NOT NULL)")
            db.execute("CREATE INDEX IF NOT EXISTS leases_expiry ON leases(expires_at)")
            db.execute("CREATE TABLE IF NOT EXISTS quota_config (id INTEGER PRIMARY KEY CHECK(id = 1), max_memory_mb INTEGER NOT NULL, max_slots INTEGER NOT NULL)")
            row = db.execute("SELECT max_memory_mb, max_slots FROM quota_config WHERE id = 1").fetchone()
            expected = (self.limits.max_memory_mb, self.limits.max_slots)
            if row is None:
                db.execute("INSERT INTO quota_config(id, max_memory_mb, max_slots) VALUES (1, ?, ?)", expected)
            elif (int(row[0]), int(row[1])) != expected:
                raise SharedQuotaError("shared_quota_limits_mismatch")

    @staticmethod
    def _validate_request(owner: str, memory_mb: int, slots: int, ttl_seconds: float) -> None:
        if not isinstance(owner, str) or not 1 <= len(owner.strip()) <= 160:
            raise SharedQuotaError("owner_required")
        if type(memory_mb) is not int or memory_mb < 1:
            raise SharedQuotaError("invalid_requested_memory")
        if type(slots) is not int or slots < 1:
            raise SharedQuotaError("invalid_requested_slots")
        if type(ttl_seconds) not in (int, float) or not 0 < float(ttl_seconds) <= 86_400:
            raise SharedQuotaError("invalid_requested_ttl")

    def acquire(self, owner: str, *, memory_mb: int, slots: int = 1,
                ttl_seconds: float | None = None) -> dict[str, Any] | None:
        self._validate_request(owner, memory_mb, slots, self.limits.lease_seconds if ttl_seconds is None else ttl_seconds)
        ttl = self.limits.lease_seconds if ttl_seconds is None else float(ttl_seconds)
        if memory_mb > self.limits.max_memory_mb or slots > self.limits.max_slots:
            return None
        now = time.time()
        token = secrets.token_hex(24)
        with self._connect() as db:
            db.execute("PRAGMA busy_timeout=5000")
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM leases WHERE expires_at <= ?", (now,))
            totals = db.execute("SELECT COALESCE(SUM(memory_mb),0), COALESCE(SUM(slots),0) FROM leases").fetchone()
            if int(totals[0]) + memory_mb > self.limits.max_memory_mb or int(totals[1]) + slots > self.limits.max_slots:
                db.rollback()
                return None
            expires = now + ttl
            db.execute("INSERT INTO leases(token, owner, memory_mb, slots, expires_at) VALUES (?,?,?,?,?)",
                       (token, owner.strip(), memory_mb, slots, expires))
            db.commit()
        return {"token": token, "owner": owner.strip(), "memory_mb": memory_mb, "slots": slots,
                "expires_at": expires, "database": str(self.database)}

    def release(self, token: str) -> bool:
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise SharedQuotaError("token_required")
        with self._connect() as db:
            cursor = db.execute("DELETE FROM leases WHERE token = ?", (token,))
            return cursor.rowcount == 1

    def renew(self, token: str, *, ttl_seconds: float | None = None) -> bool:
        if not isinstance(token, str) or not 1 <= len(token) <= 128:
            raise SharedQuotaError("token_required")
        ttl = self.limits.lease_seconds if ttl_seconds is None else ttl_seconds
        if type(ttl) not in (int, float) or not 0 < float(ttl) <= 86_400:
            raise SharedQuotaError("invalid_requested_ttl")
        with self._connect() as db:
            cursor = db.execute("UPDATE leases SET expires_at = ? WHERE token = ? AND expires_at > ?",
                                (time.time() + float(ttl), token, time.time()))
            return cursor.rowcount == 1

    def snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._connect() as db:
            db.execute("DELETE FROM leases WHERE expires_at <= ?", (now,))
            row = db.execute("SELECT COUNT(*), COALESCE(SUM(memory_mb),0), COALESCE(SUM(slots),0) FROM leases").fetchone()
        return {"active_leases": int(row[0]), "reserved_memory_mb": int(row[1]), "reserved_slots": int(row[2]),
                "limits": {"max_memory_mb": self.limits.max_memory_mb, "max_slots": self.limits.max_slots},
                "status": "assessed", "policy": "cross_process_lease_metadata_only_no_result_storage"}


__all__ = ["SharedQuotaError", "SharedQuotaLimits", "SharedResourceQuota"]
