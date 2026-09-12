from core.research_claims import operationalize_claim


def test_research_claims_require_operational_evidence():
    result = operationalize_claim("automatic_repair", {
        "cases_tested": 4, "budget": {"rounds": 3}, "failure_policy": "bounded",
        "independent_confirmation": True,
    })
    assert result["status"] == "operationalized"


def test_research_claims_do_not_accept_truthy_but_untyped_evidence():
    result = operationalize_claim("open_world", {
        "cases_tested": "many", "budget": {}, "failure_policy": " ",
        "independent_confirmation": "yes",
    })
    assert result["status"] == "not_assessed"
    assert set(result["invalid"]) == {
        "cases_tested", "budget", "failure_policy", "independent_confirmation",
    }
