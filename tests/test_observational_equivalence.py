import pytest

from core.model_set_assessment import (
    ModelSetAssessmentError,
    identify_observationally_equivalent_models,
)


def test_equivalent_observation_set_preserves_non_unique_mechanisms():
    report = identify_observationally_equivalent_models([
        {"id": "m1", "predictions": [1, 2, 3]},
        {"id": "m2", "predictions": [1 + 1e-9, 2, 3]},
        {"id": "m3", "predictions": [2, 2, 3]},
    ])
    assert report["status"] == "equivalent_set_found"
    assert ["m1", "m2"] in report["groups"]
    assert "not_algebraic" in report["policy"]


def test_equivalent_observation_set_rejects_negative_tolerance():
    with pytest.raises(ModelSetAssessmentError, match="tolerance"):
        identify_observationally_equivalent_models([{"id": "m", "predictions": [1]}], prediction_tolerance=-1)
