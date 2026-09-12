import pytest

from core.stream_plan import StreamPlanError, estimate_join_cardinality, plan_external_state_operation


def test_join_plan_estimates_many_to_many_explosion_without_aggregating():
    result = estimate_join_cardinality({"a": 100, "b": 2}, {"a": 200, "b": 3}, max_output_rows=1000)
    assert result["estimated_output_rows"] == 20006
    assert result["status"] == "needs_semantic_aggregation"
    assert "semantic" in result["policy"]


def test_external_state_plan_requires_protocol_and_projects_columns():
    assert plan_external_state_operation("sort", projected_columns=["time"]).get("status") == "needs_external_state_protocol"
    result = plan_external_state_operation("join", projected_columns=["id", "value"], state_protocol="spill-to-sqlite")
    assert result["status"] == "admitted"
    with pytest.raises(StreamPlanError):
        plan_external_state_operation("window", projected_columns=[])
