from core.persistent_compile_cache import PersistentCompileCache


def test_persistent_compile_cache_reuses_json_intermediate_across_instances(tmp_path):
    nodes = [
        {"id": "x", "kind": "input", "params": {"value": 2}},
        {"id": "y", "kind": "double", "inputs": ["x"]},
    ]
    calls = {"count": 0}

    def compile_node(node, upstream):
        calls["count"] += 1
        return node.get("params", {}).get("value", upstream["x"] * 2 if upstream else 0)

    first = PersistentCompileCache(tmp_path / "cache")
    first.compile(nodes, compile_node, compiler_version="v1", source_signature="s", domain_signature="d")
    second = PersistentCompileCache(tmp_path / "cache")
    result = second.compile(nodes, compile_node, compiler_version="v1", source_signature="s", domain_signature="d")
    assert calls["count"] == 2
    assert result["reused_nodes"] == ["x", "y"]
    assert result["persistent_cache"]["json_only"] is True


def test_persistent_compile_cache_clear_is_scoped(tmp_path):
    cache = PersistentCompileCache(tmp_path / "cache")
    cache.compile([{"id": "x", "kind": "input", "params": {"value": 1}}],
                  lambda node, upstream: node["params"]["value"])
    assert cache.path.is_file()
    assert cache.clear() is True
    assert not cache.path.exists()


def test_persistent_compile_cache_corrupt_json_falls_back_to_cold_compile(tmp_path):
    cache = PersistentCompileCache(tmp_path / "cache")
    cache.directory.mkdir(parents=True)
    cache.path.write_text("[]", encoding="utf-8")
    calls = {"count": 0}
    result = cache.compile([{"id": "x", "kind": "input", "params": {"value": 4}}],
                           lambda node, upstream: calls.__setitem__("count", calls["count"] + 1) or 4)
    assert calls["count"] == 1
    assert result["reused_nodes"] == []
