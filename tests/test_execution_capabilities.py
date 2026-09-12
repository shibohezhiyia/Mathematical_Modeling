from core.execution_capabilities import get_capabilities, validate_budget


def test_capabilities_are_bounded_and_copy_safe():
    caps = get_capabilities()
    assert {"research", "training", "optimization"}.issubset(caps)
    caps["research"]["max_wall_seconds"] = 1
    assert get_capabilities()["research"]["max_wall_seconds"] > 1
    assert validate_budget("optimization", 1) == 1.0


def test_unknown_backend_and_budget_are_rejected():
    import pytest
    with pytest.raises(ValueError, match="unsupported_execution_backend"):
        validate_budget("made_up_backend", 1)
    with pytest.raises(ValueError, match="execution_wall_limit_exceeded"):
        validate_budget("optimization", 601)
