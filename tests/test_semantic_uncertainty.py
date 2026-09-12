import pytest

from core.semantic_uncertainty import SemanticUncertaintyError, assess_semantic_hypotheses


def test_semantic_uncertainty_keeps_declared_weights_explicit():
    result = assess_semantic_hypotheses([
        {"id": "a", "evidence_weight": 3, "evidence_refs": ["f1"]},
        {"id": "b", "evidence_weight": 1, "evidence_refs": ["f2"]},
    ])
    assert result["status"] == "assessed"
    assert result["policy"].endswith("posterior_probabilities")
    assert sum(item["normalized_weight"] for item in result["hypotheses"]) == pytest.approx(1)


def test_zero_evidence_is_not_assessed():
    assert assess_semantic_hypotheses([{"id": "a", "weight": 0}])["status"] == "not_assessed"


def test_duplicate_hypotheses_and_string_evidence_refs_are_rejected():
    with pytest.raises(SemanticUncertaintyError, match="hypothesis_id_must_be_unique"):
        assess_semantic_hypotheses([
            {"id": "same", "weight": 1}, {"id": " same ", "weight": 1},
        ])
    with pytest.raises(SemanticUncertaintyError, match="evidence_refs_must_be_a_sequence"):
        assess_semantic_hypotheses([{"id": "a", "weight": 1, "evidence_refs": "evidence/a"}])
