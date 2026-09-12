import pytest

from core.repair_router import RepairRouterError, route_repair


def test_deterministic_repairs_do_not_call_model():
    result = route_repair("unit_mismatch", {"field": "speed", "expected": "m/s"})
    assert result["route"] == "rule_solver"
    assert result["model_calls_allowed"] is False
    assert result["execution_authorized"] is False


def test_semantic_repairs_are_proposals_only():
    result = route_repair("unknown_mechanism", {"residual": 1.2})
    assert result["route"] == "semantic_model"
    assert result["model_calls_allowed"] is True
    assert result["execution_authorized"] is False


def test_unknown_repairs_require_input_and_context_is_stable():
    first = route_repair("something_new", {"a": 1})
    second = route_repair("something_new", {"a": 1})
    assert first["route"] == "needs_input"
    assert first["context_sha256"] == second["context_sha256"]
    with pytest.raises(RepairRouterError):
        route_repair("unit_mismatch", {"x": float("nan")})
