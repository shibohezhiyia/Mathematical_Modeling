import json

from core.kernel_cache import KernelCache


def test_kernel_cache_has_versioned_disposable_layout_and_clear(tmp_path):
    cache = KernelCache(cache_dir=str(tmp_path), max_items=4)
    try:
        cache.set("kernel-a", [[1.0, 0.0], [0.0, 1.0]])
        assert cache.contains("kernel-a")
        manifest = json.loads((tmp_path / "cache_manifest.json").read_text("utf-8"))
        assert manifest["schema_version"] == "mathmodel.kernel-cache/v1"
        assert manifest["disposable"] is True
        assert (tmp_path / "entries").is_dir()
        assert cache.stats()["entries"] == 1
        assert cache.clear() == 1
        assert not cache.contains("kernel-a")
        assert (tmp_path / "cache_manifest.json").is_file()
    finally:
        cache.close()
