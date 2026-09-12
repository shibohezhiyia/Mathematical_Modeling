import pytest

from core.evaluation_source_audit import (
    EvaluationSourceAuditError,
    EvaluationSourceRegistry,
)


def _registry():
    return EvaluationSourceRegistry.load("examples/evaluation_sources.json")


def test_public_modeling_sources_are_not_claim_eligible():
    registry = _registry()
    assert registry.eligible() == ()
    assert registry.get("modelingagent-modelingbench").suitability == "public_or_insufficient_evidence"
    assert registry.get("comap-public-problems").accuracy_eligible is False


def test_private_math_benchmarks_are_methodology_only_when_domain_is_not_modeling():
    registry = _registry()
    assert registry.get("improofbench").suitability == "methodology_reference_only"
    assert registry.get("frontiermath").accuracy_eligible is False


def test_source_audit_never_claims_statistical_significance_at_source_level():
    payload = _registry().public_metadata()
    assert payload["eligible_source_count"] == 0
    assert all(row["statistical_claim_eligible"] is False for row in payload["sources"])


def test_conflicting_public_private_metadata_is_rejected():
    base = {
        "id": "bad", "title": "bad", "url": "https://example.com/bad",
        "domain_match": "high", "access": "public", "private_holdout": True,
        "independent_reference": True, "expert_review": True, "open_ended": True,
        "separate_from_development": True,
    }
    with pytest.raises(EvaluationSourceAuditError, match="private_holdout_public_access_conflict"):
        EvaluationSourceRegistry.from_payload({
            "schema_version": "mathmodel.evaluation-source-audit/v1",
            "sources": [base],
        })
