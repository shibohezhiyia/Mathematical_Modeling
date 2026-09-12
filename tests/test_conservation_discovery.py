import numpy as np
import pytest

from core.conservation_discovery import ConservationDiscoveryError, discover_linear_conservation


def test_discover_linear_invariant_and_keep_proof_boundary_explicit():
    times = np.linspace(0.0, 12.0, 40)
    states = np.column_stack((times, 2.0 * times + 4.0, np.sin(times)))
    result = discover_linear_conservation(times, states, ["mass_a", "mass_b", "oscillator"])

    assert result["status"] == "candidate_found"
    assert result["nullity"] >= 1
    assert any(item["status"] == "candidate" for item in result["candidates"])
    assert result["policy"].endswith("not_conservation_proof")
    assert result["trajectory_sha256"]


def test_complete_svd_keeps_nullspace_when_state_count_exceeds_derivative_rows():
    times = np.linspace(0.0, 4.0, 8)
    states = np.column_stack([times, 2.0 * times, 3.0 * times, np.sin(times), np.cos(times), times**2])
    result = discover_linear_conservation(times, states, max_candidates=2)

    # The response contains a full six-dimensional right-singular spectrum;
    # this guards against silently dropping null directions in short series.
    assert len(result["singular_values"]) == states.shape[1]
    assert result["nullity"] >= 1


def test_conservation_discovery_rejects_non_monotone_time_and_bad_names():
    times = [0, 1, 1, 2, 3, 4]
    states = np.column_stack((times, np.asarray(times) ** 2))
    with pytest.raises(ConservationDiscoveryError, match="strictly_increasing"):
        discover_linear_conservation(times, states)

    with pytest.raises(ConservationDiscoveryError, match="state_names"):
        discover_linear_conservation(np.arange(6), np.column_stack((np.arange(6), np.arange(6))), ["x", "x"])
