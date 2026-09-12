import pytest

from core.cache_policy import CachePolicyError, classify_cache_failure, validate_cache_hit, warm_start_gate


def test_cache_policy_never_persists_transient_failures_and_checks_identity():
    assert classify_cache_failure("timeout")["cacheable"] is False
    assert classify_cache_failure("deterministic_type_error")["cacheable"] is True
    expected = {"source_signature": "s", "compiler_version": "1", "domain_signature": "m", "data_view_digest": "d", "preprocessing_digest": "p"}
    assert validate_cache_hit(expected, dict(expected))["status"] == "reusable"
    assert validate_cache_hit({"source_signature": "s"}, expected, conclusion=True)["status"] == "not_reusable"


def test_warm_start_requires_mapping_structure_and_cold_comparison():
    assert warm_start_gate({"a": "a0"}, structure_unchanged=True, cold_start_available=True)["status"] == "warm_start_allowed"
    assert warm_start_gate({}, structure_unchanged=True, cold_start_available=True)["status"] == "cold_start_required"
    with pytest.raises(CachePolicyError):
        warm_start_gate(None, structure_unchanged="yes", cold_start_available=True)
