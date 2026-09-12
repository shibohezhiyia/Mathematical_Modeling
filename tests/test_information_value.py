import pytest

from core.information_value import InformationValueError, estimate_question_information_gain


def test_information_gain_prefers_question_that_separates_hypotheses():
    result = estimate_question_information_gain(
        {"linear": 1.0, "saturating": 1.0, "delayed": 1.0},
        [
            {"question": "mechanism?", "answers": {"A": ["linear"], "B": ["saturating", "delayed"]}},
            {"question": "weak?", "answers": {"yes": ["linear", "saturating"], "no": ["delayed"]}},
        ],
    )
    assert result["questions"][0]["expected_information_gain_bits"] == pytest.approx(
        result["questions"][1]["expected_information_gain_bits"]
    )
    assert result["questions"][0]["expected_information_gain_bits"] > 0
    assert result["policy"].startswith("conditional_on_declared")


def test_information_gain_rejects_overlapping_or_unknown_partition():
    with pytest.raises(InformationValueError, match="overlapping_answer_partition"):
        estimate_question_information_gain({"a": 1, "b": 1}, [{"question": "q", "answers": {"x": ["a"], "y": ["a", "b"]}}])
    with pytest.raises(InformationValueError, match="unknown_answer_hypothesis"):
        estimate_question_information_gain({"a": 1, "b": 1}, [{"question": "q", "answers": {"x": ["c"], "y": ["a"]}}])


def test_information_value_rejects_unbounded_or_unhashable_question_payloads():
    with pytest.raises(InformationValueError, match="invalid_question_budget"):
        estimate_question_information_gain({"a": 1}, [{"question": "q", "answers": {"x": ["a"], "y": []}}], max_questions=0)
    with pytest.raises(InformationValueError, match="invalid_answer_branch"):
        estimate_question_information_gain({"a": 1}, [{"question": "q", "answers": {"x": [["a"]], "y": []}}])
    with pytest.raises(InformationValueError, match="invalid_hypothesis_weight"):
        estimate_question_information_gain({"a": 1, " a ": 2}, [{"question": "q", "answers": {"x": ["a"], "y": ["a"]}}])


def test_information_gain_ranks_by_declared_cost_but_retains_raw_gain():
    result = estimate_question_information_gain(
        {"a": 1.0, "b": 1.0, "c": 1.0, "d": 1.0},
        [
            {"question": "expensive balanced", "cost": 4,
             "answers": {"yes": ["a", "b"], "no": ["c", "d"]}},
            {"question": "cheap partial", "cost": 1,
             "answers": {"yes": ["a"], "no": ["b"]}},
        ],
    )
    assert result["questions"][0]["question"] == "cheap partial"
    assert result["questions"][0]["expected_information_gain_bits"] > result["questions"][1]["expected_information_gain_bits"]
    assert result["questions"][0]["gain_per_cost"] > result["questions"][1]["gain_per_cost"]
    assert result["ranking"].endswith("declared_cost")


def test_information_gain_rejects_invalid_question_cost():
    with pytest.raises(InformationValueError, match="invalid_question_cost"):
        estimate_question_information_gain(
            {"a": 1, "b": 1},
            [{"question": "q", "cost": 0, "answers": {"x": ["a"], "y": ["b"]}}],
        )
