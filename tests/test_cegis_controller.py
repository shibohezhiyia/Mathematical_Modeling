from core.cegis_controller import CEGISConfig, CEGISControllerError, run_cegis


def test_cegis_rejects_counterexample_then_accepts_mutation():
    seen = []
    archive_seen = []

    def evaluate(candidate):
        seen.append(candidate["value"])
        if candidate["value"] == 0:
            return {"status": "fail", "failure_code": "residual_bias",
                    "violations": [{"reason": "residual_bias", "witness_id": "w1"}]}
        return {"status": "pass", "score": 0.2, "violations": []}

    def mutate(candidate, feedback):
        assert feedback["violations"][0]["witness_id"] == "w1"
        archive_seen.append(feedback["counterexample_archive"])
        return [{"value": 1}]

    result = run_cegis([{"value": 0}], evaluate, mutate)
    assert result["status"] == "accepted_candidates"
    assert result["rounds"] == 2
    assert seen == [0, 1]
    assert result["policy"]["accepted_is_tested_not_falsified"] is True
    assert archive_seen == [[{"candidate_hash": result["records"][0]["candidate_hash"],
                              "round": 1, "failure_code": "residual_bias",
                              "reason": "residual_bias", "witness_id": "w1"}]]
    assert result["records"][1]["parent_hash"] == result["records"][0]["candidate_hash"]
    assert result["lineage"][result["records"][1]["candidate_hash"]] == result["records"][0]["candidate_hash"]


def test_cegis_deduplicates_and_never_promotes_unchecked_candidate():
    calls = []

    def evaluate(candidate):
        calls.append(candidate)
        return {"status": "fail", "violations": [{"reason": "bad"}]}

    result = run_cegis([{"x": 1}, {"x": 1}], evaluate,
                       lambda candidate, feedback: [{"x": 1}],
                       config=CEGISConfig(max_rounds=3, max_repairs=2))
    assert result["candidate_count"] == 1
    assert result["accepted_candidate_hashes"] == []
    assert result["status"] == "candidate_set_inadequate"
    assert len(calls) == 1


def test_cegis_budget_and_callback_failures_are_explicit():
    result = run_cegis([{"x": 1}], lambda candidate: {"status": "fail", "violations": [{}]},
                       lambda candidate, feedback: [{"x": 2}, {"x": 3}],
                       config=CEGISConfig(max_rounds=1, max_candidates=2, max_repairs=2))
    assert result["status"] == "budget_exhausted"
    assert result["accepted_candidate_hashes"] == []

    invalid = run_cegis([{"x": 1}], lambda candidate: None,
                        lambda candidate, feedback: [], config=CEGISConfig(max_rounds=1))
    assert invalid["records"][0]["status"] == "not_assessed"
    try:
        run_cegis([{"x": 1}], None, lambda candidate, feedback: [])
    except CEGISControllerError as exc:
        assert exc.code == "callbacks_must_be_callable"
    else:
        raise AssertionError("non-callable evaluator must be rejected")


def test_cegis_bounds_counterexample_archive_without_losing_run_status():
    observed = []

    def evaluate(candidate):
        return {"status": "fail", "failure_code": "bad", "violations": [
            {"reason": f"w{candidate['x']}", "witness_id": f"w{candidate['x']}"},
        ]}

    def mutate(candidate, feedback):
        observed.append(feedback["counterexample_archive_count"])
        return [{"x": candidate["x"] + 1}]

    result = run_cegis(
        [{"x": 0}], evaluate, mutate,
        config=CEGISConfig(max_rounds=4, max_candidates=8, max_repairs=4, max_counterexamples=2),
    )
    assert result["status"] == "budget_exhausted"
    assert result["counterexample_archive_count"] == 2
    assert result["dropped_counterexamples"] == 2
    assert observed == [1, 2, 2, 2]


def test_cegis_enforces_declared_cost_budget_without_promoting_over_budget_result():
    calls = []

    def evaluate(candidate):
        calls.append(candidate)
        return {"status": "pass", "violations": [], "cost_units": 3}

    result = run_cegis([{"x": 1}, {"x": 2}], evaluate, lambda candidate, feedback: [],
                       config=CEGISConfig(max_cost_units=2))
    assert result["status"] == "budget_exhausted"
    assert result["budget_reason"] == "cost_budget_exhausted"
    assert result["accepted_candidate_hashes"] == []
    assert result["cost_units_used"] == 3
    assert len(calls) == 1


def test_cegis_rejects_invalid_budget_and_evaluator_cost():
    try:
        CEGISConfig(max_cost_units=-1)
    except CEGISControllerError as exc:
        assert exc.code == "invalid_cost_budget"
    else:
        raise AssertionError("negative cost budget must be rejected")
    result = run_cegis([{"x": 1}], lambda candidate: {"status": "pass", "violations": [], "cost_units": "1"}, lambda candidate, feedback: [])
    assert result["records"][0]["status"] == "not_assessed"
