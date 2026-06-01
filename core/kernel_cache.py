import os
import threading
from collections import OrderedDict
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
        os.makedirs(self.cache_dir, exist_ok=True)

        self.size_limit = size_limit_bytes
        self.max_items = max_items

        if _HAS_DISKCACHE:
            try:
                self._cache = diskcache.Cache(directory=self.cache_dir, size_limit=self.size_limit)
                self._use_disk = True
            except Exception:
                self._cache = OrderedDict()
                self._use_disk = False
        else:
            self._cache = OrderedDict()
            self._use_disk = False

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
