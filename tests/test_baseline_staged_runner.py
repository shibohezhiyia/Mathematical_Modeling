import pytest

from core.baseline_staged_runner import BaselineStagedRunnerError, run_baseline_then_search
from core.staged_evaluation import StageSpec


def test_baseline_is_recorded_before_search_and_exploration_stays_nonfinal():
    calls = []

    def evaluate(candidate, stage):
        calls.append(candidate["id"])
        return {"status": "pass", "score": candidate["value"]}

    report = run_baseline_then_search("regression", 2.0, [{"id": "a", "value": 1.0}], evaluate,
                                     [StageSpec("cheap", 1)])
    assert report["status"] == "exploration_complete"
    assert report["baseline"]["baseline"]["value"] == 2.0
    assert calls == ["a"]


def test_strict_mode_requires_confirmation():
    def evaluate(candidate, stage):
        return {"status": "pass", "score": candidate["value"]}

    with pytest.raises(BaselineStagedRunnerError, match="confirmation"):
        run_baseline_then_search("regression", 2.0, [{"id": "a", "value": 1.0}], evaluate,
                                 [StageSpec("cheap", 1)], mode="strict_validation")
    result = run_baseline_then_search(
        "regression", 2.0, [{"id": "a", "value": 1.0}], evaluate,
        [StageSpec("cheap", 1), StageSpec("confirm", 1, confirmation=True)],
        mode="strict_validation")
    assert result["status"] == "strict_validation_complete"
    assert result["search"]["accepted_ids"] == ["a"]


def test_empty_search_is_explicitly_baseline_only():
    result = run_baseline_then_search("regression", 2.0, [], lambda *_: {}, [StageSpec("cheap", 1)])
    assert result["status"] == "baseline_only"
    assert result["search"] is None
