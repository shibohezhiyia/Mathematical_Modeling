import pytest

from core.automated_benchmark import (
    AutomatedBenchmarkCase, AutomatedBenchmarkError, run_automated_benchmark,
)


def _case(case_id="c1", group="g1"):
    return AutomatedBenchmarkCase(
        case_id, "ode", "x'=-kx", {"x": [1.0, 0.5]},
        {"trajectory": [1.0, 0.5]}, group,
    )


def test_automated_benchmark_keeps_reference_out_of_system_and_groups_variants():
    seen = []

    def system(public):
        seen.append(public)
        assert "reference" not in public and "hidden_reference" not in public
        return {"trajectory": [1.0, 0.5]}

    def scorer(output, reference):
        assert reference["trajectory"] == [1.0, 0.5]
        return {"valid": output["trajectory"] == reference["trajectory"], "score": 0.0}

    result = run_automated_benchmark([_case(), _case("c2", "g1"), _case("c3", "g2")], system, scorer)
    assert result["status"] == "completed"
    assert result["valid_count"] == 3
    assert result["structure_group_count"] == 2
    assert len(seen) == 3


def test_automated_benchmark_malformed_scorer_is_failed_not_success():
    result = run_automated_benchmark([_case()], lambda _: {}, lambda _o, _r: {"score": 1.0})
    assert result["rows"][0]["status"] == "failed"
    assert result["valid_count"] == 0


def test_automated_benchmark_rejects_duplicate_cases_and_invalid_valid_flag():
    with pytest.raises(AutomatedBenchmarkError):
        run_automated_benchmark([_case(), _case()], lambda _: {}, lambda _o, _r: {"valid": True})
    result = run_automated_benchmark([_case()], lambda _: {}, lambda _o, _r: {"valid": 1})
    assert result["rows"][0]["status"] == "failed"
