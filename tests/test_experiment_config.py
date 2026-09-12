import json
from pathlib import Path

import pytest

from core.experiment_config import ExperimentConfigError, PublicExperimentConfig, load_experiment_config


def test_public_experiment_config_is_reproducible_and_binds_catalog():
    path = Path(__file__).resolve().parents[1] / "examples" / "public_experiment_config.json"
    config = load_experiment_config(path)
    assert config.max_api_calls == 24
    assert config.selection_splits == ("development", "structure_transform")
    assert config.evaluation_splits == ("unseen", "adversarial")
    assert len(config.digest) == 64
    assert PublicExperimentConfig.from_payload(json.loads(json.dumps(config.public(), ensure_ascii=False))).digest == config.digest


def test_config_rejects_evaluation_leakage_and_budget_overflow():
    path = Path(__file__).resolve().parents[1] / "examples" / "public_experiment_config.json"
    payload = load_experiment_config(path).public()
    with pytest.raises(ExperimentConfigError, match="selection_split_leakage"):
        PublicExperimentConfig.from_payload({**payload, "selection_splits": ["unseen"]})
    with pytest.raises(ExperimentConfigError, match="invalid_budget"):
        PublicExperimentConfig.from_payload({**payload, "budget": {**payload["budget"], "max_cases": 0}})


def test_config_rejects_problem_material_and_duplicate_methods():
    path = Path(__file__).resolve().parents[1] / "examples" / "public_experiment_config.json"
    payload = load_experiment_config(path).public()
    with pytest.raises(ExperimentConfigError, match="invalid_methods"):
        PublicExperimentConfig.from_payload({**payload, "methods": ["baseline", "baseline"]})
    with pytest.raises(ExperimentConfigError, match="invalid_policy"):
        PublicExperimentConfig.from_payload({**payload, "policy": {"no_problem_text_in_config": False}})
