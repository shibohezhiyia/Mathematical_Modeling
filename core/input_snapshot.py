"""Immutable input snapshots and compare-and-swap run ownership."""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass
from typing import Any


class InputSnapshotError(ValueError):
    pass


def _fingerprint(payload: Any, version: str) -> str:
    if not isinstance(version, str) or not 1 <= len(version) <= 160:
        raise InputSnapshotError("input_version_invalid")
    try:
        raw = json.dumps({"version": version, "payload": payload}, ensure_ascii=False,
                         sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    except (TypeError, ValueError, OverflowError) as exc:
        raise InputSnapshotError("input_payload_not_serializable") from exc
    if len(raw) > 8_000_000:
        raise InputSnapshotError("input_snapshot_too_large")
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class InputSnapshot:
    generation: int
    version: str
    digest: str


class InputSnapshotCoordinator:
    """Prevent stale jobs from committing results after an input update."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._generation = 0
        self._digest: str | None = None
        self._cancelled: set[int] = set()

    def begin(self, payload: Any, *, version: str) -> InputSnapshot:
        digest = _fingerprint(payload, version)
        with self._lock:
            self._generation += 1
            snapshot = InputSnapshot(self._generation, version, digest)
            self._digest = digest
            self._cancelled.discard(snapshot.generation)
            return snapshot

    def cancel(self, snapshot: InputSnapshot) -> None:
        self._validate_snapshot(snapshot)
        with self._lock:
            self._cancelled.add(snapshot.generation)

    def can_commit(self, snapshot: InputSnapshot) -> bool:
        self._validate_snapshot(snapshot)
        with self._lock:
            return (snapshot.generation == self._generation
                    and snapshot.digest == self._digest
                    and snapshot.generation not in self._cancelled)

    def commit(self, snapshot: InputSnapshot, result: Any) -> Any:
        if not self.can_commit(snapshot):
            raise InputSnapshotError("stale_or_cancelled_run")
        return result

    @staticmethod
    def _validate_snapshot(snapshot: InputSnapshot) -> None:
        if not isinstance(snapshot, InputSnapshot) or type(snapshot.generation) is not int or snapshot.generation < 1:
            raise InputSnapshotError("snapshot_token_invalid")
        if not isinstance(snapshot.version, str) or not snapshot.version or not isinstance(snapshot.digest, str):
            raise InputSnapshotError("snapshot_token_invalid")


__all__ = ["InputSnapshot", "InputSnapshotCoordinator", "InputSnapshotError"]
