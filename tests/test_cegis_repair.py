import pytest

from core.cegis_repair import CEGISRepairError, build_repair_directives, prioritized_operators


def test_known_counterexample_becomes_closed_repair_preference():
    result = build_repair_directives({"violations": [
        {"witness_id": "w1", "reason": "output_out_of_bounds"},
        {"witness_id": "w2", "reason": "numeric_domain_at_bound_input"},
    ]})
    assert result["directive_count"] == 2
    assert result["unknown_reason_count"] == 0
    assert prioritized_operators({"violations": [{"reason": "output_out_of_bounds"}]})[:2] == ("minimum", "maximum")
    assert all(item["requires_revalidation"] for item in result["directives"])


def test_unknown_reason_is_preserved_without_inventing_an_operator():
    result = build_repair_directives({"violations": [{"reason": "unseen_mechanism"}]})
    assert result["unknown_reason_count"] == 1
    assert result["directives"][0]["candidate_binary_primitives"] == []
    assert result["directives"][0]["candidate_unary_primitives"] == []


@pytest.mark.parametrize("feedback", [None, {"violations": "bad"}, {"violations": [{}]}])
def test_repair_feedback_contract_is_strict(feedback):
    if feedback == {"violations": [{}]}:
        # An omitted reason is retained as an unknown, not silently dropped.
        assert build_repair_directives(feedback)["unknown_reason_count"] == 1
    else:
        with pytest.raises(CEGISRepairError):
            build_repair_directives(feedback)

