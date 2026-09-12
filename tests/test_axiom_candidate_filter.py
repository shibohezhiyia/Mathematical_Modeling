import pytest

from core.axiom_candidate_filter import AxiomFilterError, filter_candidates_by_axioms


def test_axiom_filter_requires_explicit_supported_data_and_background_axioms():
    report = filter_candidates_by_axioms(
        [{"id": "m1", "data_evidence_ids": ["d1"], "axiom_ids": ["a1"]}],
        data_evidence={"d1": {"status": "supports"}},
        background_axioms={"a1": {"status": "supports"}},
    )
    assert report["eligible_ids"] == ["m1"]
    assert "explicit_data" in report["policy"]


def test_axiom_filter_rejects_extrapolation_counterexample_without_hiding_candidate():
    report = filter_candidates_by_axioms(
        [{"id": "m1", "data_evidence_ids": ["d1"], "axiom_ids": ["a1"],
          "domain_extrapolation": {"status": "counterexample_found"}}],
        data_evidence={"d1": {"status": "supports"}},
        background_axioms={"a1": {"status": "supports"}},
    )
    assert report["candidates"][0]["status"] == "rejected"
    assert report["eligible_ids"] == []


def test_axiom_filter_rejects_bad_candidate_collection():
    with pytest.raises(AxiomFilterError, match="candidates"):
        filter_candidates_by_axioms([], data_evidence={}, background_axioms={})
