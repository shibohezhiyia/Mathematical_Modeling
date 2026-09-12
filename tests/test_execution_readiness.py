import pytest

from core.execution_readiness import ExecutionReadinessError, assess_execution_readiness
from core.candidate_gate import gate_candidates


def _checks(**updates):
    result = {name: {"status": "verified", "evidence": [name + "-evidence"]}
              for name in ("type", "unit", "source", "resource", "security")}
    result.update(updates)
    return result


def test_execution_readiness_requires_all_explicit_checks():
    result = assess_execution_readiness(_checks())
    assert result["status"] == "ready"
    assert set(result["checks"]) == {"type", "unit", "source", "resource", "security"}
    assert assess_execution_readiness(_checks(unit={"status": "pending"}))["status"] == "not_assessed"
    assert assess_execution_readiness(_checks(security={"status": "denied"}))["status"] == "blocked"


@pytest.mark.parametrize("payload, code", [
    ({}, "required_check_missing"),
    ({"type": "verified", "unit": "verified", "source": "verified", "resource": "verified",
      "security": "made_up"}, "unknown_check_status"),
])
def test_execution_readiness_rejects_missing_or_unknown_status(payload, code):
    with pytest.raises(ExecutionReadinessError, match=code):
        assess_execution_readiness(payload)


def test_candidate_gate_applies_readiness_when_present():
    result = gate_candidates([{
        "id": "candidate",
        "hard_checks": {},
        "execution_readiness": _checks(source={"status": "pending"}),
        "metrics": {},
    }])
    row = result["candidates"][0]
    assert row["status"] == "not_assessed"
    assert "execution_readiness" in row["pending_hard_checks"]


def test_candidate_gate_rejects_malformed_readiness_contract():
    result = gate_candidates([{"id": "candidate", "hard_checks": {},
                               "execution_readiness": {"type": "verified"}}])
    row = result["candidates"][0]
    assert row["status"] == "hard_failure"
    assert "execution_readiness_contract" in row["failed_hard_checks"]
