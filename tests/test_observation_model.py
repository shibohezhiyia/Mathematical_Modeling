import pytest

from core.observation_model import ObservationModelError, ObservationModelSpec, validate_observation_competition


def test_process_and_observation_models_are_explicit_competitors():
    process = ObservationModelSpec(("x",), ("z",), "x = z + bias", noise="gaussian")
    biased = ObservationModelSpec(("x",), ("z",), "x = z + b[group]", grouping=("group",), bias_terms=("b",))
    result = validate_observation_competition([process, biased])
    assert result["count"] == 2
    assert result["policy"].startswith("process")


def test_invalid_delay_and_missingness_are_rejected():
    with pytest.raises(ObservationModelError, match="delay"):
        ObservationModelSpec(("x",), (), "x=z", delay=-1)
    with pytest.raises(ObservationModelError, match="missingness"):
        ObservationModelSpec(("x",), (), "x=z", missingness="unknown")
