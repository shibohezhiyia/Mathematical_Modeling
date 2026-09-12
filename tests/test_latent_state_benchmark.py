import numpy as np
import pytest

from core.latent_state_benchmark import LatentBenchmarkError, assess_latent_false_positive_control


def _independent_noise(seed, replicate):
    rng = np.random.default_rng(seed + replicate)
    return np.arange(80, dtype=float), rng.normal(size=(80, 3))


def test_no_latent_control_reports_false_positive_operating_characteristic():
    report = assess_latent_false_positive_control(_independent_noise, replicates=4, seed=11)
    assert report["status"] == "assessed"
    assert report["assessed_replicates"] == 4
    assert 0 <= report["false_positive_rate"] <= 1
    assert "not_physical_latent" in report["policy"]


def test_latent_control_rejects_invalid_budget():
    with pytest.raises(LatentBenchmarkError, match="replicates"):
        assess_latent_false_positive_control(_independent_noise, replicates=1)
