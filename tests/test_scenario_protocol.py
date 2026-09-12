import pytest

from core.scenario_protocol import ScenarioProtocolError, build_scenario_contract


def test_scenario_protocol_distinguishes_noncausal_what_if_from_causal_claims():
    perturbation = build_scenario_contract("parameter_perturbation", {"beta": 1.1})
    assert perturbation["status"] == "scenario_proposal"
    assert perturbation["causal_claim_authorized"] is False
    missing = build_scenario_contract("causal_intervention", {"treatment": 1})
    assert missing["status"] == "not_assessed"
    identified = build_scenario_contract("causal_intervention", {"treatment": 1},
                                         identification_assumptions=["no unmeasured confounding"],
                                         design="randomized assignment")
    assert identified["status"] == "identified_proposal"
    assert identified["causal_claim_authorized"] is False


def test_scenario_protocol_rejects_invalid_or_nonfinite_values():
    with pytest.raises(ScenarioProtocolError):
        build_scenario_contract("unknown", {"x": 1})
    with pytest.raises(ScenarioProtocolError):
        build_scenario_contract("scenario_analysis", {"x": float("nan")})
