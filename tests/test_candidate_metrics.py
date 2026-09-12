import pytest

from core.candidate_metrics import compare_unified_candidate_metrics
from core.candidate_gate import CandidateGateError


def _candidate(identifier, fit, complexity):
    return {"id": identifier, "hard_checks": {"units": "pass"}, "metrics": {
        "fit": fit, "complexity": complexity, "dimension_violation": 0,
        "constraint_violation": 0, "stability": 0.1, "theory_support": 1,
        "cost": 1,
    }}


def test_unified_metrics_preserve_pareto_objectives_and_directions():
    result = compare_unified_candidate_metrics([_candidate("a", 1, 5), _candidate("b", 2, 4)])
    assert result["schema_version"] == "mathmodel.candidate-metrics/v1"
    assert set(result["pareto_ids"]) == {"a", "b"}
    assert result["metric_directions"]["theory_support"] == "max"
    assert "scalar" in result["policy"]


def test_unified_metrics_reject_missing_or_invalid_objectives():
    candidate = _candidate("a", 1, 5)
    candidate["metrics"].pop("cost")
    with pytest.raises(CandidateGateError, match="unified_metric_schema"):
        compare_unified_candidate_metrics([candidate])
    candidate = _candidate("a", 1, 5)
    candidate["metrics"]["fit"] = float("nan")
    with pytest.raises(CandidateGateError, match="metric_not_finite"):
        compare_unified_candidate_metrics([candidate])
