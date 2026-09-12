import pytest

from core.incremental_compile_cache import (
    IncrementalCompileCache, IncrementalCompileCacheError, contract_source_facts,
)
from core.model_hypotheses import ProblemContract


def test_unchanged_subgraph_is_reused_and_descendants_recompile():
    cache = IncrementalCompileCache()
    calls = []

    def compile_node(node, upstream):
        calls.append(node["id"])
        return {"id": node["id"], "value": node.get("params", {}).get("value", 0) + sum(
            item["value"] for item in upstream.values())}

    graph = [
        {"id": "a", "kind": "constant", "params": {"value": 2}},
        {"id": "b", "kind": "add", "inputs": ["a"], "params": {"value": 3}},
        {"id": "c", "kind": "add", "inputs": ["b"], "params": {"value": 4}},
    ]
    first = cache.compile(graph, compile_node, source_signature="data-1")
    second = cache.compile(graph, compile_node, source_signature="data-1")
    assert first["compiled_nodes"] == ["a", "b", "c"]
    assert second["reused_nodes"] == ["a", "b", "c"]
    changed = [dict(node) for node in graph]
    changed[0] = {**changed[0], "params": {"value": 5}}
    third = cache.compile(changed, compile_node, source_signature="data-1")
    assert third["compiled_nodes"] == ["a", "b", "c"]
    assert calls == ["a", "b", "c", "a", "b", "c"]


def test_source_and_domain_signatures_prevent_stale_reuse():
    cache = IncrementalCompileCache()
    graph = [{"id": "a", "kind": "constant", "params": {"value": 1}}]
    calls = []
    compile_node = lambda node, upstream: calls.append(node["id"]) or 1
    cache.compile(graph, compile_node, source_signature="data-1", domain_signature="unit-m")
    cache.compile(graph, compile_node, source_signature="data-2", domain_signature="unit-m")
    cache.compile(graph, compile_node, source_signature="data-2", domain_signature="unit-s")
    assert calls == ["a", "a", "a"]


def test_cycles_and_missing_inputs_are_rejected_and_clear_is_recoverable():
    cache = IncrementalCompileCache()
    with pytest.raises(IncrementalCompileCacheError, match="cycle"):
        cache.compile([{"id": "a", "inputs": ["b"]}, {"id": "b", "inputs": ["a"]}], lambda *_: 1)
    with pytest.raises(IncrementalCompileCacheError, match="missing"):
        cache.compile([{"id": "a", "inputs": ["missing"]}], lambda *_: 1)
    cache.compile([{"id": "a"}], lambda *_: 1)
    assert cache.clear() == 1
    assert cache.stats()["entries"] == 0


def test_contract_revision_reuses_only_nodes_with_unchanged_explicit_fact_dependencies():
    cache = IncrementalCompileCache()
    original = ProblemContract.create("研究系统。")
    revised = original.record_confirmation("边界是否闭合？", "是，使用闭区间。",
                                           hard_constraint=True)
    calls = []

    def compile_node(node, upstream):
        calls.append(node["id"])
        return node["id"]

    original_graph = [
        {"id": "statement_node", "kind": "parse", "inputs": [],
         "source_fact_ids": ["statement"]},
        {"id": "derived", "kind": "compile", "inputs": ["statement_node"],
         "source_fact_ids": []},
    ]
    first = cache.compile(
        original_graph, compile_node, source_signature=original.digest,
        source_facts=contract_source_facts(original),
    )
    second = cache.compile(
        original_graph, compile_node, source_signature=revised.digest,
        source_facts=contract_source_facts(revised),
    )
    assert first["compiled_nodes"] == ["statement_node", "derived"]
    assert second["reused_nodes"] == ["statement_node", "derived"]
    assert second["validation_skipped"] is False

    revised_graph = [
        original_graph[0],
        {**original_graph[1], "source_fact_ids": ["user_confirmation_1"]},
    ]
    third = cache.compile(
        revised_graph, compile_node, source_signature=revised.digest,
        source_facts=contract_source_facts(revised),
    )
    assert third["reused_nodes"] == ["statement_node"]
    assert third["compiled_nodes"] == ["derived"]
    assert calls == ["statement_node", "derived", "derived"]


def test_unknown_or_duplicate_fact_dependencies_are_rejected():
    cache = IncrementalCompileCache()
    facts = {"given": {"text": "x"}}
    with pytest.raises(IncrementalCompileCacheError, match="unknown_source_fact_dependency"):
        cache.compile([{"id": "a", "source_fact_ids": ["missing"]}], lambda *_: 1,
                      source_facts=facts)
    with pytest.raises(IncrementalCompileCacheError, match="duplicate_source_fact_dependency"):
        cache.compile([{"id": "a", "source_fact_ids": ["given", "given"]}], lambda *_: 1,
                      source_facts=facts)


def test_nodes_without_explicit_fact_dependencies_remain_conservatively_contract_scoped():
    cache = IncrementalCompileCache()
    calls = []
    node = [{"id": "a", "kind": "parse"}]
    cache.compile(node, lambda *_: calls.append("a") or 1,
                  source_signature="contract-v1", source_facts={"a": 1})
    result = cache.compile(node, lambda *_: calls.append("a") or 1,
                           source_signature="contract-v2", source_facts={"a": 1, "b": 2})
    assert result["compiled_nodes"] == ["a"]
    assert result["node_source_dependencies"]["a"] is None
    assert calls == ["a", "a"]


def test_compile_with_consistency_compares_stable_artifacts_and_preserves_cache_hits():
    cache = IncrementalCompileCache()
    graph = [
        {"id": "a", "kind": "constant", "params": {"value": 2}},
        {"id": "b", "kind": "add", "inputs": ["a"], "params": {"value": 3}},
    ]
    calls = []

    def compile_node(node, upstream):
        calls.append(node["id"])
        return node.get("params", {}).get("value", 0) + sum(upstream.values())

    first = cache.compile_with_consistency(graph, compile_node, source_signature="snapshot-1")
    assert first["status"] == "tested_not_falsified"
    assert first["consistency"]["mismatch_count"] == 0
    assert first["cached"]["reused_nodes"] == []
    second = cache.compile_with_consistency(graph, compile_node, source_signature="snapshot-1")
    assert second["status"] == "tested_not_falsified"
    assert second["cached"]["reused_nodes"] == ["a", "b"]
    assert second["verdicts_cached"] is False
    assert calls == ["a", "b", "a", "b", "a", "b"]


def test_equivalence_key_requires_an_explicit_proof_certificate():
    cache = IncrementalCompileCache()
    with pytest.raises(IncrementalCompileCacheError, match="proof_certificate"):
        cache.compile([{"id": "a", "equivalence_key": "same"}], lambda *_: 1)
    result = cache.compile([{"id": "a", "equivalence_key": "same",
                             "equivalence_certificate": {"status": "proved", "rule_id": "commute"}}],
                           lambda *_: 1)
    assert result["compiled_nodes"] == ["a"]


def test_compile_with_consistency_reports_intermediate_drift_as_counterexample():
    cache = IncrementalCompileCache()
    graph = [{"id": "a", "kind": "constant"}]
    counter = {"value": 0}

    def compile_node(node, upstream):
        counter["value"] += 1
        return counter["value"]

    result = cache.compile_with_consistency(graph, compile_node)
    assert result["status"] == "counterexample"
    assert result["consistency"]["status"] == "counterexample"
    assert result["consistency"]["mismatch_count"] == 1
