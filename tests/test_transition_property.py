import pytest

from core.property_testing import PropertyTestError, check_transition_property


def test_transition_property_checks_conservation_and_reversibility():
    states = [{"x": 1.0, "y": 2.0}, {"x": -2.0, "y": 3.0}]
    conservation = check_transition_property(
        lambda state: {"x": state["x"] + 1, "y": state["y"] - 1}, states,
        kind="conservation", invariant=lambda state: state["x"] + state["y"],
    )
    assert conservation["status"] == "pass"
    reversible = check_transition_property(
        lambda state: {"x": state["x"] + 1}, [{"x": 1.0}, {"x": -2.0}],
        kind="reversible", inverse=lambda state: {"x": state["x"] - 1},
    )
    assert reversible["status"] == "pass"


def test_transition_property_preserves_counterexamples_and_contract_errors():
    failed = check_transition_property(
        lambda state: {"x": state["x"] + 1}, [{"x": 1.0}],
        kind="conservation", invariant=lambda state: state["x"],
    )
    assert failed["status"] == "fail"
    with pytest.raises(PropertyTestError):
        check_transition_property(lambda state: state, [{"x": 1}], kind="reversible")
