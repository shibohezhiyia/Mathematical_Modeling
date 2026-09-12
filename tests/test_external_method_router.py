import pytest

from core.external_method_router import ExternalMethodRoutingError, plan_external_methods


def test_router_plans_multiple_methods_from_evidence_without_assuming_dependencies():
    result = plan_external_methods({
        "numeric_data": True,
        "time_series": True,
        "spatial_grid": True,
        "known_dynamics": True,
        "noise_expected": True,
        "interaction_graph": False,
    })
    assert result["status"] == "planned"
    methods = {row["method"] for row in result["candidates"]}
    assert {"sindy", "weak_sindy", "pde_find", "ude", "llm_sr"} <= methods
    assert all(row["dependency_status"] == "not_assessed" for row in result["candidates"])
    assert all(row["policy"].startswith("method_plan_is_not_execution") for row in result["candidates"])


def test_router_marks_external_llm_code_as_proposal_only_by_default():
    result = plan_external_methods({"numeric_data": True}, enabled_methods=["llm_sr"])
    assert result["candidates"][0]["status"] == "proposal_only"
    assert result["candidates"][0]["mode"] == "proposal_only"


def test_router_uses_dependency_evidence_when_explicitly_provided():
    result = plan_external_methods(
        {"numeric_data": True, "time_series": True, "known_dynamics": True},
        enabled_methods=["sindy", "ude"],
        installed_backends={"sindy": True, "ude": False},
    )
    assert [row["status"] for row in result["candidates"]] == ["ready", "unavailable"]


@pytest.mark.parametrize("profile", [
    {"numeric_data": "true"},
    {"unknown_flag": True},
    {"numeric_data": True, "allow_external_repositories": 1},
])
def test_router_rejects_ambiguous_profiles(profile):
    with pytest.raises(ExternalMethodRoutingError):
        plan_external_methods(profile)


def test_router_rejects_duplicate_or_unknown_method_requests():
    with pytest.raises(ExternalMethodRoutingError, match="enabled_methods_unknown_or_duplicate"):
        plan_external_methods({"numeric_data": True}, enabled_methods=["llm_sr", "llm_sr"])
    with pytest.raises(ExternalMethodRoutingError, match="enabled_methods_unknown_or_duplicate"):
        plan_external_methods({"numeric_data": True}, enabled_methods=["made_up"])


def test_router_reports_no_eligible_method_instead_of_fabricating_a_solver():
    result = plan_external_methods({"numeric_data": False})
    assert result["status"] == "no_eligible_method"
    assert result["candidates"] == []
