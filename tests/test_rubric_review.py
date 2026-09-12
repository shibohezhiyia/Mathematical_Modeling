import pytest

from core.rubric_review import RubricReviewError, aggregate_reviews, build_rubric, record_review


def test_independent_reviews_require_two_reviewers_before_aggregation_is_ready():
    rubric = build_rubric(["contract", "numerical", "evidence"], scale_max=4)
    first = record_review(rubric, case_id="case-a", evaluator_id="r1", scores={"contract": 4, "numerical": 3, "evidence": 2})
    incomplete = aggregate_reviews(rubric, [first])
    assert incomplete["status"] == "incomplete"
    second = record_review(rubric, case_id="case-a", evaluator_id="r2", scores={"contract": 3, "numerical": 3, "evidence": 1})
    ready = aggregate_reviews(rubric, [first, second])
    assert ready["status"] == "ready_for_adjudication"
    assert ready["rows"][0]["mean_scores"]["contract"] == 3.5


def test_review_contract_rejects_duplicate_or_out_of_range_scores():
    rubric = build_rubric(["contract"])
    first = record_review(rubric, case_id="case-a", evaluator_id="r1", scores={"contract": 4})
    with pytest.raises(RubricReviewError, match="duplicate_review"):
        aggregate_reviews(rubric, [first, first])
    with pytest.raises(RubricReviewError, match="score_out_of_range"):
        record_review(rubric, case_id="case-a", evaluator_id="r2", scores={"contract": 5})
