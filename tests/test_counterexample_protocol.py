import pytest

from core.counterexample_protocol import (
    CounterexampleProtocolError,
    normalize_counterexample,
    replay_counterexamples,
    run_counterexample_suite,
)


def _norm(category="normative_violation", **extra):
    base = {"witness_id": "w1", "category": category, "contract_version": "ignored",
            "in_domain": True, "property_id": "nonnegative", "values": {"x": 0.0}}
    if category == "empirical_mismatch":
        base.update(observed=2.0, predicted=1.0)
    if category == "numerical_failure":
        base.update(error="solver_nonfinite")
    if category == "out_of_domain":
        base.update(in_domain=False, domain="x outside [0, 1]", property_id=None)
    base.update(extra)
    return base


def test_normalize_requires_explicit_category_and_evidence():
    result = normalize_counterexample(_norm(), contract_version="contract/v1")
    assert result["category"] == "normative_violation"
    with pytest.raises(CounterexampleProtocolError, match="category_required"):
        normalize_counterexample({"witness_id": "w1"}, contract_version="contract/v1")
    with pytest.raises(CounterexampleProtocolError, match="in_domain_evidence"):
        normalize_counterexample(_norm(in_domain=None), contract_version="contract/v1")
    with pytest.raises(CounterexampleProtocolError, match="replay_attempts"):
        normalize_counterexample(_norm(replay={"attempts": "1"}), contract_version="contract/v1")


def test_replay_keeps_callback_errors_unresolved_and_counts_categories():
    records = [_norm(), _norm("empirical_mismatch", witness_id="w2"), _norm("numerical_failure", witness_id="w3")]
    result = replay_counterexamples(records, lambda item: item["witness_id"] == "w1", contract_version="contract/v1")
    assert result["reproduced"] == 1 and result["not_reproduced"] == 2
    result = replay_counterexamples(records, lambda item: (_ for _ in ()).throw(RuntimeError()), contract_version="contract/v1")
    assert result["status"] == "partial" and result["not_assessed"] == 3


def test_suite_runs_boundary_cases_first_and_never_calls_clean_search_a_proof():
    seen = []
    cases = [
        {"case_id": "r", "kind": "random", "values": {"x": 1}},
        {"case_id": "e", "kind": "endpoint", "values": {"x": 0}},
    ]
    result = run_counterexample_suite(
        cases, lambda case: (seen.append(case["kind"]) or {"violation": False}),
        contract_version="contract/v1")
    assert seen == ["endpoint", "random"]
    assert result["status"] == "tested_not_falsified"


def test_suite_keeps_invalid_witness_as_unassessed_and_accepts_normative_one():
    cases = [{"case_id": "a", "kind": "endpoint"}, {"case_id": "b", "kind": "degenerate"}]
    def evaluate(case):
        if case["case_id"] == "a":
            return {"violation": True, "category": "normative_violation", "in_domain": True,
                    "property_id": "positive"}
        return {"violation": True, "category": "unknown", "in_domain": True}
    result = run_counterexample_suite(cases, evaluate, contract_version="contract/v1")
    assert result["status"] == "counterexample_found"
    assert len(result["counterexamples"]) == 1 and result["not_assessed"] == 1
