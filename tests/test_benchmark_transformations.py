import pytest

from core.benchmark_transformations import transform_case, transform_statement
from core.open_model_bench import BenchmarkCase, BenchmarkValidationError


def _case():
    return BenchmarkCase.from_payload({
        "id": "cooling_dev",
        "split": "development",
        "family": "dynamics",
        "statement": "Let x be the temperature. Predict x after time t for a device.",
        "tags": ["toy"],
    })


def test_transform_statement_is_word_aware_and_deterministic():
    text = transform_statement("x and xx; temperature at m/s", symbol_map={"x": "T"},
                               story_map={"temperature": "salinity"}, unit_map={"m/s": "km/h"})
    assert text == "T and xx; salinity at km/h"


def test_transform_case_drops_original_fingerprint_and_marks_split():
    result = transform_case(_case(), transform_id="rename", symbol_map={"x": "state"})
    assert result.split == "structure_transform"
    assert result.data_fingerprints == ()
    assert result.has_ground_truth is False
    assert "transform:rename" in result.tags
    assert "state" in result.statement


def test_transform_ground_truth_requires_new_fixture_fingerprint():
    with pytest.raises(BenchmarkValidationError, match="transformed_ground_truth_requires_new_data"):
        transform_case(_case(), transform_id="truth", has_ground_truth=True)


def test_transform_rejects_reused_fingerprint_and_invalid_truth_flag():
    case = BenchmarkCase.from_payload({
        "id": "with_data", "split": "development", "family": "data",
        "statement": "x in m/s", "data_fingerprints": ["a" * 64],
    })
    with pytest.raises(BenchmarkValidationError, match="transformed_data_reuses_original_fingerprint"):
        transform_case(case, transform_id="reuse", transformed_data_fingerprints=("a" * 64,))
    with pytest.raises(BenchmarkValidationError, match="invalid_transformed_ground_truth_flag"):
        transform_case(_case(), transform_id="bad", has_ground_truth=1)


def test_transform_replacements_are_literal_not_regex_backreferences():
    assert transform_statement("x has unit m/s", symbol_map={"x": r"\1"}, unit_map={"m/s": r"\2"}) == r"\1 has unit \2"
