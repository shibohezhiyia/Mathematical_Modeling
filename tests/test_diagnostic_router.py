import numpy as np
import pytest

from core.diagnostic_router import DiagnosticRouterError, route_model_diagnostics, route_structure_diagnostics
from core.structure_diagnostics import diagnose_series_structure


def test_structure_signals_become_reversible_search_hints():
    times = np.arange(32, dtype=float)
    values = np.sin(times * 2 * np.pi / 8)
    diagnostics = diagnose_series_structure(times, values, series_names=["temperature"], max_lag=16)
    routed = route_structure_diagnostics(diagnostics)
    assert routed["hint_count"] >= 1
    assert any("sinusoidal_basis" in hint["candidate_primitives"] for hint in routed["hints"])
    assert all(hint["status"] == "proposal_not_executed" for hint in routed["hints"])
    assert all(hint["requires_current_validation"] for hint in routed["hints"])
    assert routed["may_modify_ir"] is False


def test_change_point_and_monotone_routes_preserve_evidence():
    payload = {
        "schema_version": "mathmodel.structure-diagnostics/v1",
        "trajectory_sha256": "abc",
        "diagnostics": {
            "x": {"status": "signal", "signals": ["change_point_candidate", "monotone_candidate"],
                  "change_point": {"index": 10}},
            "noise": {"status": "no_signal", "signals": []},
        },
    }
    result = route_structure_diagnostics(payload)
    assert [hint["family"] for hint in result["hints"]] == ["monotone_response", "piecewise_or_event"]
    assert result["hints"][1]["evidence"]["signal_record"]["change_point"]["index"] == 10


def test_residual_router_ignores_locked_test_records():
    result = route_model_diagnostics([
        {"id": "dev", "code": "search_residual_memory",
         "context": {"phase": "development", "may_feed_search": True},
         "state": "observed", "evidence": {"lag": 2}},
        {"id": "test", "code": "search_residual_memory",
         "context": {"phase": "final_test", "may_feed_search": False},
         "state": "observed", "evidence": {"lag": 9}},
    ])
    assert result["hint_count"] == 1
    assert result["hints"][0]["source"] == "residual"
    assert result["hints"][0]["subject"] == "dev"


def test_prediction_and_cluster_diagnostics_route_as_soft_hints():
    result = route_model_diagnostics([
        {"id": "prediction", "code": "prediction_residual_structure",
         "context": {"phase": "development", "may_feed_search": True},
         "state": "suspected", "evidence": {"correlation": 0.3}},
        {"id": "cluster", "code": "clustering_seed_instability",
         "context": {"phase": "development", "may_feed_search": True},
         "state": "suspected", "evidence": {"status": "fail"}},
        {"id": "periodic", "code": "periodic_structure_signal",
         "context": {"phase": "development", "may_feed_search": True},
         "state": "suspected", "evidence": {"signal": "periodic_candidate"}},
        {"id": "optimization", "code": "optimization_audit_warning",
         "context": {"phase": "execution", "may_feed_search": True},
         "state": "suspected", "evidence": {"status": "fail"}},
    ])
    assert result["hint_count"] == 4
    assert all(hint["status"] == "proposal_not_executed" for hint in result["hints"])
    assert all(hint["hard_constraint"] is False for hint in result["hints"])


def test_router_rejects_malformed_or_unknown_signals():
    with pytest.raises(DiagnosticRouterError, match="invalid_structure_diagnostics_schema"):
        route_structure_diagnostics({"diagnostics": {}})
    payload = {"schema_version": "mathmodel.structure-diagnostics/v1", "diagnostics": {
        "x": {"status": "signal", "signals": ["not_a_route"]}
    }}
    result = route_structure_diagnostics(payload)
    assert result["hint_count"] == 0


def test_router_rejects_nonfinite_or_ambiguous_diagnostic_payloads():
    payload = {"schema_version": "mathmodel.structure-diagnostics/v1", "diagnostics": {
        "x": {"status": "signal", "signals": ["monotone_candidate"], "value": float("nan")}
    }}
    with pytest.raises(DiagnosticRouterError, match="diagnostic_evidence_must_be_finite_json"):
        route_structure_diagnostics(payload)
    with pytest.raises(DiagnosticRouterError, match="invalid_structure_signals"):
        route_structure_diagnostics({
            "schema_version": "mathmodel.structure-diagnostics/v1",
            "diagnostics": {"x": {"status": "signal", "signals": [None]}},
        })
