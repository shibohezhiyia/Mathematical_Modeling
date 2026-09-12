import pytest

from core.concept_library import ConceptLibrary, ConceptLibraryError
from core.experience_store import ExperienceStore


def concept():
    return {
        "name": "capacity_saturation",
        "statement": "容量约束下，响应可能随负载增加而趋于饱和。",
        "mechanism_tags": ["saturation", "capacity"],
        "implementations": [{"primitive_ids": ["minimum", "multiply"],
                              "description": "使用有界响应原语，并在当前题目上重新拟合参数。",
                              "unit_signatures": ["dimensionless"]}],
        "applicability": ["存在明确容量上界", "高负载观测足够"],
        "failure_conditions": ["高负载区持续下降", "单位绑定不一致"],
        "counterexamples": [], "evidence_refs": [],
    }


def test_concept_is_stored_as_seed_and_searchable(tmp_path):
    library = ConceptLibrary(ExperienceStore(tmp_path / "store"))
    stored = library.put(concept(), run_id="run-1", topic_key="topic-1")
    assert stored["status"] == "provisional"
    matches = library.search(mechanism_tags=["saturation"])
    assert len(matches) == 1
    assert matches[0]["concept"]["name"] == "capacity_saturation"
    assert matches[0]["seed_only"] is True
    assert matches[0]["requires_current_validation"] is True


def test_verified_concept_requires_replay_scope_and_is_protected(tmp_path):
    library = ConceptLibrary(ExperienceStore(tmp_path / "store"))
    stored = library.put(concept())
    with pytest.raises(Exception, match="复现"):
        library.promote_verified(stored["experience_id"], {"validation_scope": {}})
    evidence = {"replay_passed": True, "counterexamples_checked": True,
                "security_checked": True, "validation_scope": {"domain": "bounded"}}
    verified = library.promote_verified(stored["experience_id"], evidence)
    assert verified["status"] == "verified"
    with pytest.raises(Exception):
        library.store.purge(statuses=("verified",), dry_run=False)


def test_counterexample_revision_is_new_provisional_version(tmp_path):
    library = ConceptLibrary(ExperienceStore(tmp_path / "store"))
    old = library.put(concept())
    revised = library.revise_with_counterexample(old["experience_id"], counterexample_id="cx-1",
                                                 reason="高负载点越界", scope="declared_domain")
    assert revised["experience_id"] != old["experience_id"]
    assert revised["status"] == "provisional"
    assert len(library.store.get(revised["experience_id"])["payload"]["recipe"]["concept"]["counterexamples"]) == 1


def test_concept_validation_rejects_empty_implementation_and_unknown_fields(tmp_path):
    library = ConceptLibrary(ExperienceStore(tmp_path / "store"))
    bad = concept(); bad["implementations"] = []
    with pytest.raises(ConceptLibraryError, match="implementations"):
        library.put(bad)
    bad = concept(); bad["secret"] = "api-key"
    with pytest.raises(ConceptLibraryError, match="unknown_fields"):
        library.put(bad)


def test_concept_library_rejects_ambiguous_revisions_and_string_search_coercion(tmp_path):
    library = ConceptLibrary(ExperienceStore(tmp_path / "store"))
    bad = concept()
    bad["counterexamples"] = [
        {"id": "same", "reason": "r1", "scope": "s"},
        {"id": "same", "reason": "r2", "scope": "s"},
    ]
    with pytest.raises(ConceptLibraryError, match="duplicate_concept_counterexample"):
        library.put(bad)
    with pytest.raises(ConceptLibraryError, match="search_signatures_must_be_sequences"):
        library.search(mechanism_tags="saturation")
    stored = library.put(concept())
    with pytest.raises(ConceptLibraryError, match="counterexample_id_must_be_safe"):
        library.revise_with_counterexample(stored["experience_id"], counterexample_id=1,
                                            reason="r", scope="s")
