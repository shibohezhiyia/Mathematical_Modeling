import numpy as np
import pytest

from core.unclosed_state_competition import (
    UnclosedStateCompetitionError,
    compete_unclosed_state_explanations,
)


def test_high_noise_does_not_become_an_automatic_hidden_state_verdict():
    rng = np.random.default_rng(7)
    observations = np.column_stack((np.linspace(0, 1, 80), np.linspace(1, 2, 80)))
    residuals = rng.normal(0, 0.2, size=observations.shape)
    result = compete_unclosed_state_explanations(residuals, observations, state_names=["x", "y"])
    assert result["status"] == "screening_only"
    assert result["external_input_status"] == "not_assessed"
    assert result["policy"]["latent_state_not_proven"] is True
    assert all(item["id"] != "external_input" for item in result["candidate_explanations"])


def test_external_input_and_competing_signals_are_reported_without_causal_claims():
    t = np.linspace(0, 8, 80)
    external = np.column_stack((np.sin(t), np.cos(t)))
    residuals = np.column_stack((external[:, 0] + 0.05 * np.sin(5 * t), np.diff(np.r_[0, external[:, 1]])))
    observations = np.column_stack((1 + t, 2 + t))
    result = compete_unclosed_state_explanations(residuals, observations, external)
    assert result["external_input_status"] == "assessed"
    assert any(item["id"] == "external_input" for item in result["candidate_explanations"])
    assert result["policy"]["scores_are_not_probabilities"] is True
    assert result["near_tied_explanations"]


def test_competition_rejects_misaligned_or_nonfinite_inputs():
    with pytest.raises(UnclosedStateCompetitionError, match="must_align"):
        compete_unclosed_state_explanations(np.zeros((20, 1)), np.zeros((20, 2)))
    bad = np.zeros((20, 1))
    bad[0, 0] = np.nan
    with pytest.raises(UnclosedStateCompetitionError, match="must_be_finite"):
        compete_unclosed_state_explanations(bad, np.zeros((20, 1)))
