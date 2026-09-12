"""Validated, failure-aware modeling concepts built on :mod:`experience_store`.

A concept is a reusable *search hint* such as "saturation may be plausible
under a capacity bound".  It is not a solved recipe: implementations,
applicability, and counterexamples are kept together and every retrieval is
marked as requiring validation for the current task.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from .experience_store import ExperienceStore, ExperienceStoreError


SCHEMA_VERSION = "mathmodel.modeling-concept/v1"
_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_.:-]{0,127}$")
_FIELDS = {"name", "statement", "mechanism_tags", "implementations",
           "applicability", "failure_conditions", "counterexamples", "evidence_refs"}


class ConceptLibraryError(ValueError):
    """Raised when a reusable concept is malformed or cannot be stored."""


def _text(value: Any, name: str, limit: int) -> str:
    if not isinstance(value, str) or not 1 <= len(value.strip()) <= limit:
        raise ConceptLibraryError(f"{name}_must_be_nonempty_text")
    return value.strip()


def _text_list(value: Any, name: str, limit: int, item_limit: int = 400) -> list[str]:
    if not isinstance(value, list) or len(value) > limit:
        raise ConceptLibraryError(f"{name}_must_be_a_bounded_list")
    return [_text(item, name, item_limit).strip() for item in value]


def _normalize(concept: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(concept, Mapping):
        raise ConceptLibraryError("concept_must_be_an_object")
    if set(concept) - _FIELDS:
        raise ConceptLibraryError("concept_contains_unknown_fields")
    result = {
        "schema_version": SCHEMA_VERSION,
        "name": _text(concept.get("name"), "name", 200),
        "statement": _text(concept.get("statement"), "statement", 4000),
        "mechanism_tags": _text_list(concept.get("mechanism_tags", []), "mechanism_tags", 32, 80),
        "applicability": _text_list(concept.get("applicability", []), "applicability", 32),
        "failure_conditions": _text_list(concept.get("failure_conditions", []), "failure_conditions", 64),
        "counterexamples": [], "evidence_refs": [], "implementations": [],
    }
    if not result["mechanism_tags"]:
        raise ConceptLibraryError("mechanism_tags_must_not_be_empty")
    if len(set(result["mechanism_tags"])) != len(result["mechanism_tags"]):
        raise ConceptLibraryError("duplicate_mechanism_tag")
    implementations = concept.get("implementations", [])
    if not isinstance(implementations, list) or not 1 <= len(implementations) <= 16:
        raise ConceptLibraryError("implementations_must_have_1_to_16_items")
    for item in implementations:
        if not isinstance(item, Mapping) or set(item) - {"primitive_ids", "description", "unit_signatures"}:
            raise ConceptLibraryError("invalid_concept_implementation")
        primitive_ids = _text_list(item.get("primitive_ids", []), "primitive_ids", 64, 120)
        if not primitive_ids:
            raise ConceptLibraryError("implementation_requires_primitive_ids")
        units = _text_list(item.get("unit_signatures", []), "unit_signatures", 64, 120)
        result["implementations"].append({
            "primitive_ids": list(dict.fromkeys(primitive_ids)),
            "description": _text(item.get("description"), "implementation_description", 2000),
            "unit_signatures": list(dict.fromkeys(units)),
        })
    counterexamples = concept.get("counterexamples", [])
    if not isinstance(counterexamples, list) or len(counterexamples) > 128:
        raise ConceptLibraryError("counterexamples_must_be_bounded")
    counterexample_ids: set[str] = set()
    for item in counterexamples:
        if not isinstance(item, Mapping) or set(item) - {"id", "reason", "scope"}:
            raise ConceptLibraryError("invalid_concept_counterexample")
        identifier = _text(item.get("id"), "counterexample_id", 160)
        if identifier in counterexample_ids:
            raise ConceptLibraryError("duplicate_concept_counterexample")
        counterexample_ids.add(identifier)
        result["counterexamples"].append({"id": identifier,
                                           "reason": _text(item.get("reason"), "counterexample_reason", 1000),
                                           "scope": _text(item.get("scope"), "counterexample_scope", 500)})
    evidence_refs = concept.get("evidence_refs", [])
    if not isinstance(evidence_refs, list) or len(evidence_refs) > 64:
        raise ConceptLibraryError("evidence_refs_must_be_bounded")
    for item in evidence_refs:
        if not isinstance(item, Mapping) or set(item) - {"run_id", "method", "scope"}:
            raise ConceptLibraryError("invalid_concept_evidence_ref")
        run_id = _text(item.get("run_id"), "evidence_run_id", 128)
        if not _ID.fullmatch(run_id):
            raise ConceptLibraryError("evidence_run_id_must_be_safe")
        result["evidence_refs"].append({"run_id": run_id,
                                         "method": _text(item.get("method"), "evidence_method", 300),
                                         "scope": _text(item.get("scope"), "evidence_scope", 500)})
    return result


class ConceptLibrary:
    """Concept facade that never bypasses :class:`ExperienceStore` gates."""

    def __init__(self, store: ExperienceStore):
        if not isinstance(store, ExperienceStore):
            raise ConceptLibraryError("store_must_be_an_experience_store")
        self.store = store

    def put(self, concept: Mapping[str, Any], *, run_id: str | None = None,
            topic_key: str | None = None) -> dict[str, Any]:
        normalized = _normalize(concept)
        unit_signatures = sorted({unit for item in normalized["implementations"]
                                  for unit in item["unit_signatures"]})
        return self.store.put(
            {"kind": "modeling_concept", "concept": normalized},
            graph_signature=normalized["mechanism_tags"],
            unit_signature=unit_signatures,
            objective_signature=[normalized["name"]],
            run_id=run_id, topic_key=topic_key,
        )

    def promote_verified(self, experience_id: str, evidence: Mapping[str, Any]) -> dict[str, Any]:
        try:
            record = self.store.get(experience_id, include_recipe=True)
        except ExperienceStoreError as exc:
            raise ConceptLibraryError(str(exc)) from exc
        recipe = record.get("payload", {}).get("recipe", {})
        if recipe.get("kind") != "modeling_concept":
            raise ConceptLibraryError("experience_is_not_a_modeling_concept")
        return self.store.promote_verified(experience_id, evidence)

    def search(self, *, mechanism_tags: Sequence[str] = (), unit_signatures: Sequence[str] = (),
               objective: Sequence[str] = (), limit: int = 20,
               include_provisional: bool = True) -> list[dict[str, Any]]:
        if any(isinstance(value, (str, bytes)) for value in (mechanism_tags, unit_signatures, objective)):
            raise ConceptLibraryError("search_signatures_must_be_sequences")
        tags = _text_list(list(mechanism_tags), "mechanism_tags", 32, 80)
        units = _text_list(list(unit_signatures), "unit_signatures", 64, 120)
        objectives = _text_list(list(objective), "objective", 8, 200)
        matches = self.store.search(graph_signature=tags, unit_signature=units,
                                    objective_signature=objectives, limit=limit,
                                    include_provisional=include_provisional)
        output = []
        for record in matches:
            try:
                full = self.store.get(record["experience_id"], include_recipe=True)
            except ExperienceStoreError:
                continue
            recipe = full.get("payload", {}).get("recipe", {})
            if recipe.get("kind") != "modeling_concept":
                continue
            record = dict(record)
            record["concept"] = deepcopy(recipe.get("concept"))
            record["requires_current_validation"] = True
            record["seed_only"] = True
            output.append(record)
        return output

    def revise_with_counterexample(self, experience_id: str, *, counterexample_id: str,
                                   reason: str, scope: str) -> dict[str, Any]:
        """Write a new provisional version; verified objects remain immutable."""
        if not isinstance(counterexample_id, str) or not _ID.fullmatch(counterexample_id):
            raise ConceptLibraryError("counterexample_id_must_be_safe")
        current = self.store.get(experience_id, include_recipe=True)
        concept = current.get("payload", {}).get("recipe", {}).get("concept")
        if not isinstance(concept, Mapping):
            raise ConceptLibraryError("experience_is_not_a_modeling_concept")
        updated = deepcopy(dict(concept))
        counterexamples = list(updated.get("counterexamples", []))
        if any(item.get("id") == counterexample_id for item in counterexamples):
            raise ConceptLibraryError("duplicate_concept_counterexample")
        counterexamples.append({"id": str(counterexample_id), "reason": reason, "scope": scope})
        updated["counterexamples"] = counterexamples
        updated.pop("schema_version", None)
        return self.put(updated, topic_key=current.get("topic_key"))


__all__ = ["SCHEMA_VERSION", "ConceptLibraryError", "ConceptLibrary"]
