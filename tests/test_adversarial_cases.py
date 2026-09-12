import pytest

from core.adversarial_cases import AdversarialCaseError, build_adversarial_cases


def test_adversarial_case_builder_orders_boundaries_and_is_reproducible():
    first = build_adversarial_cases({"x": [0, 1], "y": [-2, 2]}, max_cases=12, seed=3)
    second = build_adversarial_cases({"y": [-2, 2], "x": [0, 1]}, max_cases=12, seed=3)
    assert first == second
    assert first[0]["kind"] == "endpoint"
    assert any(item["kind"] == "optimized" for item in first)
    assert all("violation" not in item for item in first)


def test_adversarial_case_builder_rejects_unbounded_or_bad_budget():
    with pytest.raises(AdversarialCaseError, match="bound"):
        build_adversarial_cases({"x": [1, 0]})
    with pytest.raises(AdversarialCaseError, match="budget"):
        build_adversarial_cases({"x": [0, 1]}, max_cases=0)
