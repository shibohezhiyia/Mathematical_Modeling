"""JSON-only persistence for intermediate compilation artifacts.

The persistent layer is deliberately narrower than a general object cache:
only JSON-compatible intermediate artifacts are written, never predictions,
verdicts, tracebacks, code, or pickled objects.  A corrupt or incompatible
file is ignored and compilation falls back to the in-memory path.
"""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .incremental_compile_cache import IncrementalCompileCache, IncrementalCompileCacheError, SCHEMA_VERSION


class PersistentCompileCacheError(ValueError):
    pass


def _json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))
    except (TypeError, ValueError, OverflowError) as exc:
        raise PersistentCompileCacheError("artifact_must_be_json_compatible") from exc


class PersistentCompileCache:
    """Bounded persistent wrapper around :class:`IncrementalCompileCache`."""

    def __init__(self, directory: str | Path, *, max_entries: int = 256,
                 max_artifact_bytes: int = 8_000_000) -> None:
        try:
            root = Path(directory).expanduser().resolve()
        except (TypeError, ValueError, OSError) as exc:
            raise PersistentCompileCacheError("invalid_cache_directory") from exc
        if root == Path(root.anchor):
            raise PersistentCompileCacheError("cache_directory_must_not_be_filesystem_root")
        self.directory = root
        self.path = root / "intermediate_cache.json"
        self._memory = IncrementalCompileCache(
            max_entries=max_entries, max_artifact_bytes=max_artifact_bytes
        )
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if (not isinstance(payload, Mapping)
                    or payload.get("schema_version") != SCHEMA_VERSION
                    or not isinstance(payload.get("entries"), list)):
                return
            entries = OrderedDict()
            for item in payload["entries"][: self._memory.max_entries]:
                if not isinstance(item, Mapping) or not isinstance(item.get("key"), str):
                    return
                artifact = item.get("artifact")
                size = _json_size(artifact)
                if size > self._memory.max_artifact_bytes:
                    continue
                entries[item["key"]] = {
                    "artifact": artifact, "bytes": size,
                    "source_signature": str(item.get("source_signature", "")),
                    "domain_signature": str(item.get("domain_signature", "")),
                }
            self._memory._entries = entries
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError, PersistentCompileCacheError):
            # Fail closed: a bad disk cache simply causes a cold compile.
            self._memory._entries.clear()

    def _save(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        entries = []
        for key, item in self._memory._entries.items():
            artifact = item.get("artifact")
            try:
                size = _json_size(artifact)
            except PersistentCompileCacheError:
                continue
            if size <= self._memory.max_artifact_bytes:
                entries.append({"key": key, "artifact": artifact, "bytes": size,
                                "source_signature": item.get("source_signature", ""),
                                "domain_signature": item.get("domain_signature", "")})
        payload = {"schema_version": SCHEMA_VERSION, "entries": entries}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def compile(self, nodes: Sequence[Mapping[str, Any]], compile_node: Callable[[Mapping[str, Any], Mapping[str, Any]], Any], **kwargs: Any) -> dict[str, Any]:
        self._load()
        before = set(self._memory._entries)
        try:
            result = self._memory.compile(nodes, compile_node, **kwargs)
        except IncrementalCompileCacheError:
            raise
        new_artifacts = [item for item in result.get("compiled_nodes", [])]
        if new_artifacts or set(self._memory._entries) != before:
            self._save()
        result["persistent_cache"] = {"path": str(self.path), "loaded": True,
                                       "json_only": True, "verdicts_cached": False}
        return result

    def clear(self) -> bool:
        """Delete only this cache file; return whether it existed."""
        existed = self.path.is_file()
        if existed:
            self.path.unlink()
        self._memory._entries.clear()
        self._loaded = True
        return existed


__all__ = ["PersistentCompileCacheError", "PersistentCompileCache"]
