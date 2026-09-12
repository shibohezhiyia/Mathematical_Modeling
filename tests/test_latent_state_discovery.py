import numpy as np
import pytest

from core.latent_state_discovery import LatentStateDiscoveryError, discover_delay_latent_states


def test_delay_embedding_returns_reproducible_latent_candidate_with_holdout():
    times = np.linspace(0, 20, 80)
    observations = np.column_stack((np.sin(times), np.cos(times), np.sin(times) + 0.02 * np.cos(3 * times)))
    result = discover_delay_latent_states(times, observations, ["x", "y", "z"], latent_dim=2)

    assert result["status"] == "candidate_generated"
    assert result["latent_dim"] == 2
    assert result["train_rows"] + result["validation_rows"] == result["embedded_rows"]
    assert np.isfinite(result["validation_reconstruction_rmse"])
    assert result["policy"].endswith("physical_hidden_variable")


def test_delay_embedding_rejects_invalid_window_and_names():
    times = np.arange(16, dtype=float)
    observations = np.column_stack((times, times**2))
    with pytest.raises(LatentStateDiscoveryError, match="delay_window"):
        discover_delay_latent_states(times, observations, delay_count=10, delay_stride=2)
    with pytest.raises(LatentStateDiscoveryError, match="observation_names"):
        discover_delay_latent_states(times, observations, ["x", "x"])
