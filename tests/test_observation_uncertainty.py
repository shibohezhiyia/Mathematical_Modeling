import pytest

from core.observation_uncertainty import ObservationUncertaintyError, assess_observation_uncertainty


def test_observation_uncertainty_separates_between_and_within_variance():
    result = assess_observation_uncertainty([
        {"observation_id": "a", "processing_id": "raw", "value": 1.0, "unit_signature": "m"},
        {"observation_id": "a", "processing_id": "smooth", "value": 1.2, "unit_signature": "m"},
        {"observation_id": "b", "processing_id": "raw", "value": 3.0, "unit_signature": "m"},
        {"observation_id": "b", "processing_id": "smooth", "value": 2.8, "unit_signature": "m"},
    ])
    assert result["status"] == "assessed"
    assert result["between_observation_variance"] > result["within_observation_variance"]
    assert result["variance_reconstruction_residual"] == pytest.approx(0.0)


def test_observation_uncertainty_rejects_mixed_units_and_marks_singletons_partial():
    partial = assess_observation_uncertainty([
        {"observation_id": "a", "processing_id": "raw", "value": 1.0, "unit_signature": "m"},
        {"observation_id": "b", "processing_id": "raw", "value": 2.0, "unit_signature": "m"},
    ])
    assert partial["status"] == "partial"
    with pytest.raises(ObservationUncertaintyError, match="unit_signatures_must_match"):
        assess_observation_uncertainty([
            {"observation_id": "a", "processing_id": "raw", "value": 1.0, "unit_signature": "m"},
            {"observation_id": "a", "processing_id": "smooth", "value": 100.0, "unit_signature": "cm"},
        ])


def test_observation_uncertainty_keeps_processing_and_noise_components_separate():
    result = assess_observation_uncertainty([
        {"observation_id": "a", "processing_id": "raw", "value": 1.0,
         "observation_noise_variance": 0.04, "unit_signature": "m"},
        {"observation_id": "a", "processing_id": "raw", "value": 1.2,
         "observation_noise_variance": 0.04, "unit_signature": "m"},
        {"observation_id": "a", "processing_id": "smooth", "value": 1.1,
         "observation_noise_variance": 0.01, "unit_signature": "m"},
        {"observation_id": "b", "processing_id": "raw", "value": 3.0,
         "observation_noise_variance": 0.04, "unit_signature": "m"},
        {"observation_id": "b", "processing_id": "smooth", "value": 2.8,
         "observation_noise_variance": 0.01, "unit_signature": "m"},
    ])
    assert result["declared_observation_noise_variance"] == pytest.approx(0.028)
    assert result["declared_noise_record_count"] == 5
    assert result["within_processing_variance"] >= 0
    assert result["decomposition"]


def test_observation_uncertainty_rejects_invalid_declared_noise():
    with pytest.raises(ObservationUncertaintyError, match="observation_noise_variance"):
        assess_observation_uncertainty([
            {"observation_id": "a", "processing_id": "raw", "value": 1.0,
             "observation_noise_variance": -1, "unit_signature": "m"},
            {"observation_id": "b", "processing_id": "raw", "value": 2.0,
             "unit_signature": "m"},
        ])
