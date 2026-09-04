"""Versioned, auditable storage for one mathematical-modeling run.

The manager deliberately separates durable evidence from disposable runtime
state.  Every public artifact is addressed by a path relative to a single run
root and recorded with a checksum in ``artifact_manifest.json``.  Cleanup is
restricted to the exact ``cache`` and ``temp`` directories; reports, evidence
and charts cannot be deleted through this API.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


ARTIFACT_SCHEMA_VERSION = "mathmodel.run-artifacts/v1"
CACHE_SCHEMA_VERSION = "mathmodel.run-cache/v1"
_DIRECTORIES = {
    "evidence": "evidence",
    "reports": "reports",
    "charts": "charts",
    "cache": "cache",
    "temp": "temp",
    "logs": "logs",
}
_DISPOSABLE_CATEGORIES = frozenset({"cache", "temp"})
_ARTIFACT_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_CACHE_NAMESPACE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def create_run_id() -> str:
    """Return a sortable, collision-resistant identifier for a local run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{uuid.uuid4().hex[:8]}"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")


class RunArtifactManager:
    """Manage the strict directory contract for exactly one completed run."""

    manifest_name = "artifact_manifest.json"
    cache_manifest_name = "cache_manifest.json"

    def __init__(self, root: os.PathLike[str] | str, run_id: Optional[str] = None) -> None:
        self.root = Path(root).resolve()
        self.run_id = str(run_id or self.root.name or create_run_id())
        self.created_at = _utc_now()
        self._artifacts: Dict[str, Dict[str, Any]] = {}
        self._cleanup_history: list[Dict[str, Any]] = []
        self._cache_entries: Dict[str, Dict[str, Any]] = {}
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in _DIRECTORIES.values():
            (self.root / directory).mkdir(parents=True, exist_ok=True)
        self._write_cache_manifest()
        self._write_manifest(status="running")

    @classmethod
    def open_existing(cls, root: os.PathLike[str] | str) -> "RunArtifactManager":
        """Open an existing run without recreating or resetting its manifests."""
        instance = cls.__new__(cls)
        instance.root = Path(root).resolve()
        manifest_path = instance.root / cls.manifest_name
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("不支持的运行产物清单版本")
        instance.run_id = str(payload.get("run_id", instance.root.name))
        instance.created_at = str(payload.get("created_at", _utc_now()))
        instance._artifacts = {
            str(item["id"]): dict(item) for item in payload.get("artifacts", [])
            if isinstance(item, dict) and item.get("id")
        }
        instance._cleanup_history = list(payload.get("cleanup_history", []))
        instance._cache_entries = {}
        cache_manifest = instance.root / _DIRECTORIES["cache"] / cls.cache_manifest_name
        if cache_manifest.is_file():
            try:
                cache_payload = json.loads(cache_manifest.read_text(encoding="utf-8"))
                if cache_payload.get("schema_version") == CACHE_SCHEMA_VERSION:
                    instance._cache_entries = {
                        str(item["key"]): dict(item)
                        for item in cache_payload.get("entries", [])
                        if isinstance(item, dict) and item.get("key")
                    }
            except (OSError, ValueError, json.JSONDecodeError):
                instance._cache_entries = {}
        return instance

    @property
    def manifest_path(self) -> Path:
        return self.root / self.manifest_name

    @property
    def directories(self) -> Dict[str, str]:
        return dict(_DIRECTORIES)

    def path(self, category: str, relative_name: os.PathLike[str] | str) -> Path:
        """Resolve a path inside a declared category and reject traversal."""
        if category not in _DIRECTORIES:
            raise ValueError(f"未知产物类别: {category}")
        relative = Path(relative_name)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("产物路径必须是非空、无上级跳转的相对路径")
        base = (self.root / _DIRECTORIES[category]).resolve()
        target = (base / relative).resolve()
        try:
            target.relative_to(base)
        except ValueError as exc:
            raise ValueError("产物路径越出运行目录") from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def relative_path(self, path: os.PathLike[str] | str) -> str:
        target = Path(path).resolve()
        try:
            return target.relative_to(self.root).as_posix()
        except ValueError as exc:
            raise ValueError("文件不属于当前运行目录") from exc

    def write_json(
        self,
        artifact_id: str,
        category: str,
        relative_name: str,
        payload: Any,
        *,
        format_version: str = "1.0",
        required: bool = True,
        disposable: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        target = self.path(category, relative_name)
        content = json.dumps(
            payload, ensure_ascii=False, indent=2, sort_keys=True, default=str
        ).encode("utf-8") + b"\n"
        self._atomic_write(target, content)
        self.register_existing(
            artifact_id, category, target, media_type="application/json",
            format_version=format_version, required=required, disposable=disposable,
            metadata=metadata,
        )
        return target

    def write_text(
        self,
        artifact_id: str,
        category: str,
        relative_name: str,
        content: str,
        *,
        media_type: str = "text/plain; charset=utf-8",
        format_version: str = "1.0",
        required: bool = True,
        disposable: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Path:
        target = self.path(category, relative_name)
        self._atomic_write(target, str(content).encode("utf-8"))
        self.register_existing(
            artifact_id, category, target, media_type=media_type,
            format_version=format_version, required=required, disposable=disposable,
            metadata=metadata,
        )
        return target

    def register_existing(
        self,
        artifact_id: str,
        category: str,
        path: os.PathLike[str] | str,
        *,
        media_type: str,
        format_version: str = "1.0",
        required: bool = False,
        disposable: bool = False,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not _ARTIFACT_ID.fullmatch(str(artifact_id)):
            raise ValueError(f"非法 artifact_id: {artifact_id!r}")
        if category not in _DIRECTORIES:
            raise ValueError(f"未知产物类别: {category}")
        if bool(disposable) != (category in _DISPOSABLE_CATEGORIES):
            if disposable:
                raise ValueError("只有 cache/temp 产物可以标记为可删除")
            if category in _DISPOSABLE_CATEGORIES:
                raise ValueError("cache/temp 产物必须标记为可删除")
        target = Path(path).resolve()
        expected_base = (self.root / _DIRECTORIES[category]).resolve()
        try:
            target.relative_to(expected_base)
        except ValueError as exc:
            raise ValueError("产物文件不在声明的类别目录中") from exc
        if not target.is_file():
            raise FileNotFoundError(target)
        record: Dict[str, Any] = {
            "id": str(artifact_id),
            "category": category,
            "relative_path": self.relative_path(target),
            "media_type": str(media_type),
            "format_version": str(format_version),
            "sha256": self._sha256(target),
            "size_bytes": target.stat().st_size,
            "required": bool(required),
            "disposable": bool(disposable),
            "created_at": _utc_now(),
        }
        if metadata:
            record["metadata"] = dict(metadata)
        self._artifacts[str(artifact_id)] = record
        return dict(record)

    def cache_key(self, namespace: str, inputs: Any, version: str = "1") -> str:
        self._validate_namespace(namespace)
        envelope = {"namespace": namespace, "version": str(version), "inputs": inputs}
        return hashlib.sha256(_canonical_json(envelope)).hexdigest()

    def write_cache(
        self,
        namespace: str,
        inputs: Any,
        payload: Any,
        *,
        version: str = "1",
        ttl_seconds: Optional[int] = None,
    ) -> Path:
        """Write a disposable JSON cache entry with deterministic naming."""
        self._validate_namespace(namespace)
        key = self.cache_key(namespace, inputs, version)
        relative_name = f"{namespace}/{key[:2]}/{key}.json"
        target = self.path("cache", relative_name)
        content = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
        self._atomic_write(target, content.encode("utf-8"))
        now = _utc_now()
        self._cache_entries[key] = {
            "key": key,
            "namespace": namespace,
            "relative_path": self.relative_path(target),
            "format_version": str(version),
            "sha256": self._sha256(target),
            "size_bytes": target.stat().st_size,
            "created_at": now,
            "ttl_seconds": None if ttl_seconds is None else max(0, int(ttl_seconds)),
            "disposable": True,
        }
        self._write_cache_manifest()
        return target

    def cleanup_disposable(
        self,
        categories: Sequence[str] = ("cache", "temp"),
        *,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """Delete only files below exact disposable directories.

        Directories remain in place, so the layout is stable after cleanup.
        """
        requested = tuple(dict.fromkeys(str(item) for item in categories))
        invalid = [item for item in requested if item not in _DISPOSABLE_CATEGORIES]
        if invalid:
            raise ValueError(f"禁止清理非缓存产物: {', '.join(invalid)}")
        files: list[Path] = []
        for category in requested:
            base = (self.root / _DIRECTORIES[category]).resolve()
            self._assert_exact_disposable_root(base, category)
            control_file = (
                (base / self.cache_manifest_name).resolve()
                if category == "cache" else None
            )
            files.extend(
                path for path in base.rglob("*")
                if path.is_file() and path.resolve() != control_file
            )
        unique_files = sorted(set(files), key=lambda item: item.as_posix())
        size_bytes = sum(path.stat().st_size for path in unique_files if path.exists())
        deleted_paths = [self.relative_path(path) for path in unique_files]
        if not dry_run:
            for path in unique_files:
                path.unlink(missing_ok=True)
            for category in requested:
                base = (self.root / _DIRECTORIES[category]).resolve()
                directories = sorted(
                    (path for path in base.rglob("*") if path.is_dir()),
                    key=lambda item: len(item.parts), reverse=True,
                )
                for directory in directories:
                    try:
                        directory.rmdir()
                    except OSError:
                        pass
                base.mkdir(parents=True, exist_ok=True)
            removed = {
                artifact_id for artifact_id, record in self._artifacts.items()
                if record["category"] in requested
            }
            for artifact_id in removed:
                self._artifacts.pop(artifact_id, None)
            if "cache" in requested:
                self._cache_entries.clear()
                self._write_cache_manifest()
            event = {
                "at": _utc_now(), "categories": list(requested),
                "deleted_files": len(unique_files), "deleted_bytes": size_bytes,
            }
            self._cleanup_history.append(event)
            self._write_manifest(status=self._current_status())
        return {
            "dry_run": bool(dry_run),
            "categories": list(requested),
            "deleted_files": len(unique_files),
            "deleted_bytes": size_bytes,
            "relative_paths": deleted_paths,
        }

    def clear_cache(self, *, dry_run: bool = False) -> Dict[str, Any]:
        return self.cleanup_disposable(("cache",), dry_run=dry_run)

    def finalize(self, status: str = "complete") -> Path:
        if status not in {"complete", "failed", "incomplete"}:
            raise ValueError("最终状态必须是 complete/failed/incomplete")
        missing = [
            record["id"] for record in self._artifacts.values()
            if record["required"] and not (self.root / record["relative_path"]).is_file()
        ]
        final_status = "incomplete" if missing and status == "complete" else status
        self._write_manifest(status=final_status, missing_required=missing)
        return self.manifest_path

    def read_manifest(self) -> Dict[str, Any]:
        if not self.manifest_path.is_file():
            raise FileNotFoundError(self.manifest_path)
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _manifest_payload(
        self, status: str, missing_required: Optional[Iterable[str]] = None
    ) -> Dict[str, Any]:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": _utc_now(),
            "status": status,
            "directories": dict(_DIRECTORIES),
            "artifacts": sorted(self._artifacts.values(), key=lambda item: item["relative_path"]),
            "integrity": {
                "hash_algorithm": "sha256",
                "missing_required_artifact_ids": sorted(missing_required or []),
            },
            "cleanup_policy": {
                "safe_delete_categories": sorted(_DISPOSABLE_CATEGORIES),
                "preserve_categories": ["evidence", "reports", "charts", "logs"],
                "delete_whole_run": "Delete this run root only when the entire run is no longer needed.",
            },
            "cleanup_history": list(self._cleanup_history),
        }

    def _write_manifest(
        self, status: str, missing_required: Optional[Iterable[str]] = None
    ) -> None:
        payload = self._manifest_payload(status, missing_required)
        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        self._atomic_write(self.manifest_path, content)

    def _write_cache_manifest(self) -> None:
        target = self.path("cache", self.cache_manifest_name)
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "updated_at": _utc_now(),
            "disposable": True,
            "entries": sorted(self._cache_entries.values(), key=lambda item: item["key"]),
        }
        content = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        self._atomic_write(target, content)

    def _atomic_write(self, target: Path, content: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = (self.root / _DIRECTORIES["temp"]).resolve()
        temp_dir.mkdir(parents=True, exist_ok=True)
        temporary = temp_dir / f".{target.name}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    def _current_status(self) -> str:
        try:
            return str(self.read_manifest().get("status", "running"))
        except (OSError, ValueError, json.JSONDecodeError):
            return "running"

    def _assert_exact_disposable_root(self, base: Path, category: str) -> None:
        expected = (self.root / _DIRECTORIES[category]).resolve()
        if base != expected or base == self.root or category not in _DISPOSABLE_CATEGORIES:
            raise ValueError("清理目标未通过运行目录边界校验")
        base.relative_to(self.root)

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _validate_namespace(namespace: str) -> None:
        if not _CACHE_NAMESPACE.fullmatch(str(namespace)):
            raise ValueError("缓存命名空间只能包含小写字母、数字、下划线和连字符")


__all__ = [
    "ARTIFACT_SCHEMA_VERSION", "CACHE_SCHEMA_VERSION", "RunArtifactManager",
    "create_run_id",
]
