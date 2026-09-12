import pytest

from core.verdict_export import VerdictExportError, approved_writing_projection, build_trace_bundle


def _verdict():
    return {
        "schema_version": "mathmodel.model-verdict/v1", "run_id": "r1",
        "approved_candidate_ids": ["m1"],
        "candidates": [{"id": "m1", "state": "approved", "evidence_refs": ["e1"]}],
        "evidence_refs": ["e1"], "assumptions": ["a"], "counterexamples": [],
    }


def test_trace_bundle_reports_missing_refs():
    bundle = build_trace_bundle(verdict=_verdict(), evidence=[])
    assert bundle["traceability"]["status"] == "incomplete"
    assert bundle["traceability"]["missing_evidence_refs"] == ["e1"]


def test_trace_bundle_can_carry_finite_conclusion_certificate():
    from core.conclusion_certificate import build_conclusion_certificate
    certificate = build_conclusion_certificate(applicability_boundary="declared domain",
                                                residuals=[{"value": 0.1}],
                                                constraint_checks=[{"violation": 0.0}])
    bundle = build_trace_bundle(verdict=_verdict(), evidence=[], certificate=certificate)
    assert bundle["certificate"]["schema_version"] == "mathmodel.conclusion-certificate/v1"


def test_writing_projection_only_allows_approved_candidate():
    projection = approved_writing_projection(_verdict())
    assert projection["approved_candidates"][0]["id"] == "m1"
    projection["assumptions"].append("local mutation")
    assert "local mutation" not in _verdict()["assumptions"]


def test_writing_projection_rejects_unresolved_output():
    verdict = {**_verdict(), "approved_candidate_ids": []}
    with pytest.raises(VerdictExportError, match="no_approved"):
        approved_writing_projection(verdict)
