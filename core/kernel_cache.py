import os
import json
import threading
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Optional

try:
    import diskcache
    _HAS_DISKCACHE = True
except Exception:
    _HAS_DISKCACHE = False


class KernelCache:
    """Kernel cache with optional disk-backed storage.

    - If `diskcache` is installed, use it with an LRU eviction based on `size_limit_bytes`.
    - Otherwise fallback to an in-memory LRU limited by `max_items`.

    The cache directory is created under the project workspace to avoid using other drives.
    """

    def __init__(self, cache_dir: Optional[str] = None, size_limit_bytes: int = 2 * 1024 ** 3, max_items: int = 64):
        self.lock = threading.RLock()
        if cache_dir is None:
            cache_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'data', 'cache', 'kernel_cache')
        self.cache_dir = os.path.abspath(cache_dir)
        self.data_dir = os.path.join(self.cache_dir, 'entries')
        self.temp_dir = os.path.join(self.cache_dir, 'temp')
        self.manifest_path = os.path.join(self.cache_dir, 'cache_manifest.json')
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.size_limit = size_limit_bytes
        self.max_items = max_items

        if _HAS_DISKCACHE:
            try:
                self._cache = diskcache.Cache(directory=self.data_dir, size_limit=self.size_limit)
                self._use_disk = True
            except Exception:
                self._cache = OrderedDict()
                self._use_disk = False
        else:
            self._cache = OrderedDict()
            self._use_disk = False
        self._write_manifest()

    def _write_manifest(self) -> None:
        payload = {
            'schema_version': 'mathmodel.kernel-cache/v1',
            'updated_at': datetime.now(timezone.utc).isoformat(),
            'disposable': True,
            'backend': 'diskcache' if self._use_disk else 'memory',
            'layout': {'entries': 'entries/', 'atomic_staging': 'temp/'},
            'size_limit_bytes': self.size_limit,
            'max_memory_items': self.max_items,
            'cleanup': 'Call clear() or delete this kernel_cache directory.',
        }
        temporary = os.path.join(
            self.temp_dir, f'.cache_manifest.{uuid.uuid4().hex}.tmp'
        )
        try:
            with open(temporary, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.manifest_path)
        finally:
            try:
                if os.path.exists(temporary):
                    os.remove(temporary)
            except OSError:
                pass

    def get(self, key: Any) -> Optional[Any]:
        with self.lock:
            try:
                if self._use_disk:
                    return self._cache.get(key)
                else:
                    return self._cache.get(key)
            except Exception:
                return None

    def set(self, key: Any, value: Any) -> None:
        with self.lock:
            try:
                if self._use_disk:
                    # diskcache handles eviction
                    self._cache.set(key, value)
                else:
                    # simple in-memory LRU
                    if key in self._cache:
                        self._cache.pop(key)
                    self._cache[key] = value
                    while len(self._cache) > self.max_items:
                        self._cache.popitem(last=False)
            except Exception:
                pass

    def contains(self, key: Any) -> bool:
        with self.lock:
            try:
                return key in self._cache
            except Exception:
                return False

    def clear(self) -> int:
        """Clear kernel entries while preserving the documented cache layout."""
        with self.lock:
            try:
                count = len(self._cache)
                self._cache.clear()
                self._write_manifest()
                return int(count)
            except Exception:
                return 0

    def stats(self) -> dict:
        with self.lock:
            try:
                entries = len(self._cache)
                volume = int(self._cache.volume()) if self._use_disk else 0
            except Exception:
                entries, volume = 0, 0
            return {
                'schema_version': 'mathmodel.kernel-cache/v1',
                'backend': 'diskcache' if self._use_disk else 'memory',
                'entries': int(entries),
                'size_bytes': volume,
                'cache_dir': self.cache_dir,
                'manifest_path': self.manifest_path,
                'disposable': True,
            }

    def close(self) -> None:
        """Release the optional disk backend without deleting cache files."""
        with self.lock:
            if self._use_disk:
                try:
                    self._cache.close()
                except Exception:
                    pass
