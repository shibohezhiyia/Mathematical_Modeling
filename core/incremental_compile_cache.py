"""Versioned, bounded cache for deterministic intermediate graph compilation.

The cache deliberately stores intermediate artifacts only.  A key includes the
node definition, upstream keys, compiler version, source snapshot and domain
contract, so changing a patch invalidates all affected descendants naturally.
It is not a result/verdict cache and cannot bypass independent validation.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math
import sys
from typing import Any, Callable, Mapping, Sequence


SCHEMA_VERSION = "mathmodel.incremental-compile-cache/v1"


class IncrementalCompileCacheError(ValueError):
    """Raised when a graph or cache contract is invalid."""


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                          allow_nan=False, default=str)
    except (TypeError, ValueError) as exc:
        raise IncrementalCompileCacheError("value_is_not_canonicalizable") from exc


def _hash_payload(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def contract_source_facts(contract) -> dict[str, dict[str, Any]]:
    """Convert an exact ``ProblemContract`` into node-addressable source facts."""
    from .model_hypotheses import ProblemContract

    if not isinstance(contract, ProblemContract):
        raise IncrementalCompileCacheError("problem_contract_required")
    payload = contract.public()
    hard = set(payload["hard_constraint_ids"])
    return {
        fact["id"]: {
            "start": fact["start"], "end": fact["end"], "text": fact["text"],
            "hard_constraint": fact["id"] in hard,
        }
        for fact in payload["facts"]
    }


def _node_id(node: Mapping[str, Any]) -> str:
    identifier = node.get("id")
    if not isinstance(identifier, (str, int)) or isinstance(identifier, bool) or not str(identifier).strip():
        raise IncrementalCompileCacheError("node_id_required")
    return str(identifier).strip()


class IncrementalCompileCache:
    """Compile a small typed DAG while reusing unchanged intermediate nodes.

    ``compile_node`` receives ``(node, upstream_artifacts)``.  Nodes must list
    upstream ids in ``inputs``.  A node may provide ``equivalence_key`` when
    the compiler has an explicit, documented reason that two ids are
    semantically equivalent; otherwise ids remain part of the cache key.
    """

    def __init__(self, *, max_entries: int = 256, max_artifact_bytes: int = 8_000_000) -> None:
        if type(max_entries) is not int or not 1 <= max_entries <= 100_000:
            raise IncrementalCompileCacheError("invalid_max_entries")
        if type(max_artifact_bytes) is not int or not 1 <= max_artifact_bytes <= 1_000_000_000:
            raise IncrementalCompileCacheError("invalid_max_artifact_bytes")
        self.max_entries = max_entries
        self.max_artifact_bytes = max_artifact_bytes
        self._entries: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @staticmethod
    def _normalise_nodes(nodes: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)) or not nodes:
            raise IncrementalCompileCacheError("nodes_must_be_nonempty_sequence")
        normalised: list[dict[str, Any]] = []
        by_id: dict[str, dict[str, Any]] = {}
        for original in nodes:
            if not isinstance(original, Mapping):
                raise IncrementalCompileCacheError("node_must_be_object")
            node = dict(original)
            identifier = _node_id(node)
            if identifier in by_id:
                raise IncrementalCompileCacheError("duplicate_node_id")
            inputs = node.get("inputs", [])
            if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
                raise IncrementalCompileCacheError("node_inputs_must_be_sequence")
            node["inputs"] = [str(item).strip() for item in inputs]
            if any(not item for item in node["inputs"]):
                raise IncrementalCompileCacheError("node_input_id_must_be_nonempty")
            if "equivalence_key" in node:
                certificate = node.get("equivalence_certificate")
                if (not isinstance(certificate, Mapping)
                        or certificate.get("status") != "proved"
                        or not isinstance(certificate.get("rule_id"), str)
                        or not certificate["rule_id"].strip()):
                    raise IncrementalCompileCacheError("equivalence_key_requires_proof_certificate")
            normalised.append(node)
            by_id[identifier] = node
        for node in normalised:
            if any(input_id not in by_id for input_id in node["inputs"]):
                raise IncrementalCompileCacheError("node_input_reference_missing")
        return normalised, by_id

    @staticmethod
    def _topological(nodes: Sequence[Mapping[str, Any]], by_id: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
        state: dict[str, int] = {}
        ordered: list[dict[str, Any]] = []

        def visit(identifier: str) -> None:
            marker = state.get(identifier, 0)
            if marker == 1:
                raise IncrementalCompileCacheError("graph_contains_cycle")
            if marker == 2:
                return
            state[identifier] = 1
            node = by_id[identifier]
            for upstream in node["inputs"]:
                visit(upstream)
            state[identifier] = 2
            ordered.append(dict(node))

        for node in nodes:
            visit(_node_id(node))
        return ordered

    def _key(self, node: Mapping[str, Any], upstream_keys: Sequence[str], *,
             compiler_version: str, source_signature: str, domain_signature: str) -> str:
        identifier = _node_id(node)
        identity = node.get("equivalence_key", identifier)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "compiler_version": compiler_version,
            "source_signature": source_signature,
            "domain_signature": domain_signature,
            "identity": identity,
            "kind": node.get("kind"),
            "params": node.get("params", node.get("payload", {})),
            "upstream_keys": list(upstream_keys),
        }
        return _hash_payload(payload)

    def compile(
        self,
        nodes: Sequence[Mapping[str, Any]],
        compile_node: Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
        *,
        compiler_version: str = "1",
        source_signature: str = "source-unknown",
        domain_signature: str = "domain-unknown",
        source_facts: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not callable(compile_node):
            raise IncrementalCompileCacheError("compile_node_must_be_callable")
        for value, error in ((compiler_version, "invalid_compiler_version"),
                             (source_signature, "invalid_source_signature"),
                             (domain_signature, "invalid_domain_signature")):
            if not isinstance(value, str) or not 1 <= len(value) <= 512:
                raise IncrementalCompileCacheError(error)
        normalised, by_id = self._normalise_nodes(nodes)
        ordered = self._topological(normalised, by_id)
        fact_values = None
        if source_facts is not None:
            if not isinstance(source_facts, Mapping) or len(source_facts) > 1024:
                raise IncrementalCompileCacheError("invalid_source_facts")
            fact_values = dict(source_facts)
            if any(not isinstance(key, str) or not key or len(key) > 128 for key in fact_values):
                raise IncrementalCompileCacheError("invalid_source_fact_id")
            _canonical(fact_values)
        artifacts: dict[str, Any] = {}
        keys: dict[str, str] = {}
        dependencies: dict[str, list[str] | None] = {}
        reused: list[str] = []
        compiled: list[str] = []
        for node in ordered:
            identifier = _node_id(node)
            upstream_keys = [keys[input_id] for input_id in node["inputs"]]
            node_source_signature = source_signature
            dependencies[identifier] = None
            if fact_values is not None and "source_fact_ids" in node:
                raw_dependencies = node["source_fact_ids"]
                if (not isinstance(raw_dependencies, Sequence)
                        or isinstance(raw_dependencies, (str, bytes))
                        or len(raw_dependencies) > 128):
                    raise IncrementalCompileCacheError("invalid_source_fact_dependencies")
                source_ids = [str(value) for value in raw_dependencies]
                if len(source_ids) != len(set(source_ids)):
                    raise IncrementalCompileCacheError("duplicate_source_fact_dependency")
                if any(value not in fact_values for value in source_ids):
                    raise IncrementalCompileCacheError("unknown_source_fact_dependency")
                dependencies[identifier] = source_ids
                node_source_signature = "facts:" + _hash_payload({
                    value: fact_values[value] for value in source_ids
                })
            key = self._key(node, upstream_keys, compiler_version=compiler_version,
                             source_signature=node_source_signature,
                             domain_signature=domain_signature)
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                artifacts[identifier] = entry["artifact"]
                keys[identifier] = key
                reused.append(identifier)
                continue
            upstream = {input_id: artifacts[input_id] for input_id in node["inputs"]}
            artifact = compile_node(node, upstream)
            estimated = max(1, sys.getsizeof(artifact))
            if estimated <= self.max_artifact_bytes:
                self._entries[key] = {"artifact": artifact, "bytes": estimated,
                                      "source_signature": source_signature,
                                      "domain_signature": domain_signature}
                self._entries.move_to_end(key)
                while len(self._entries) > self.max_entries or self._cache_bytes() > self.max_artifact_bytes:
                    self._entries.popitem(last=False)
            artifacts[identifier] = artifact
            keys[identifier] = key
            compiled.append(identifier)
        return {
            "schema_version": SCHEMA_VERSION,
            "artifacts": artifacts,
            "node_keys": keys,
            "node_source_dependencies": dependencies,
            "reused_nodes": reused,
            "compiled_nodes": compiled,
            "cache": self.stats(),
            "verdicts_cached": False,
            "validation_skipped": False,
        }

    def _cache_bytes(self) -> int:
        return sum(int(item.get("bytes", 0)) for item in self._entries.values())

    def compile_with_consistency(
        self,
        nodes: Sequence[Mapping[str, Any]],
        compile_node: Callable[[Mapping[str, Any], Mapping[str, Any]], Any],
        *,
        compiler_version: str = "1",
        source_signature: str = "source-unknown",
        domain_signature: str = "domain-unknown",
        source_facts: Mapping[str, Any] | None = None,
        tolerance: float = 1e-9,
    ) -> dict[str, Any]:
        """Compile cold and cached paths and compare only stable artifacts.

        The cold path uses a separate cache with the same limits.  The current
        cache is then compiled normally, so callers can observe reuse without
        treating hit counts as mathematical evidence.  A mismatch is returned
        as a cache-consistency counterexample; it never silently falls back to
        the cached answer.
        """
        if type(tolerance) not in (int, float) or not math.isfinite(float(tolerance)) or tolerance < 0:
            raise IncrementalCompileCacheError("invalid_consistency_tolerance")
        cold_cache = IncrementalCompileCache(
            max_entries=self.max_entries,
            max_artifact_bytes=self.max_artifact_bytes,
        )
        cold = cold_cache.compile(
            nodes, compile_node, compiler_version=compiler_version,
            source_signature=source_signature, domain_signature=domain_signature,
            source_facts=source_facts,
        )
        cached = self.compile(
            nodes, compile_node, compiler_version=compiler_version,
            source_signature=source_signature, domain_signature=domain_signature,
            source_facts=source_facts,
        )
        # Cache hit/miss counters and node ordering are expected to differ;
        # compare only the values that a downstream compiler consumes.
        cold_stable = {
            "artifacts": cold["artifacts"],
            "node_keys": cold["node_keys"],
            "node_source_dependencies": cold["node_source_dependencies"],
        }
        cached_stable = {
            "artifacts": cached["artifacts"],
            "node_keys": cached["node_keys"],
            "node_source_dependencies": cached["node_source_dependencies"],
        }
        from .cache_consistency import compare_cached_uncached

        consistency = compare_cached_uncached(
            lambda: cold_stable, lambda: cached_stable, tolerance=float(tolerance),
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "tested_not_falsified" if consistency["status"] == "tested_not_falsified" else "counterexample",
            "cold": cold,
            "cached": cached,
            "consistency": consistency,
            "verdicts_cached": False,
            "validation_skipped": False,
            "policy": "cache_consistency_compares_stable_intermediates_only; cache_hits_are_not_evidence",
        }

    def stats(self) -> dict[str, Any]:
        return {"entries": len(self._entries), "max_entries": self.max_entries,
                "artifact_bytes": self._cache_bytes(),
                "max_artifact_bytes": self.max_artifact_bytes,
                "schema_version": SCHEMA_VERSION}

    def clear(self) -> int:
        count = len(self._entries)
        self._entries.clear()
        return count


__all__ = [
    "SCHEMA_VERSION", "IncrementalCompileCacheError", "IncrementalCompileCache",
    "contract_source_facts",
]
