import math

import pytest

from core.baseline_registry import (
    BaselineRegistry,
    BaselineRegistryError,
    BaselineSpec,
    default_baseline_registry,
)


def test_default_registry_is_stable_and_exposes_all_minimal_families():
    first, second = default_baseline_registry(), default_baseline_registry()
    assert first.digest == second.digest
    assert {item["family"] for item in first.public()["baselines"]} >= {
        "data", "dynamics", "optimization",
    }
    assert first.for_task("regression").baseline_id == "regression_mean"


def test_baseline_comparison_preserves_metric_direction_and_scope():
    registry = default_baseline_registry()
    better = registry.compare("regression", 1.0, 1.5)
    worse = registry.compare("classification", 0.6, 0.7)
    assert better["status"] == "candidate_better"
    assert better["margin"] == pytest.approx(0.5)
    assert worse["status"] == "baseline_not_beaten"
    assert registry.compare("unknown", 1, 2)["status"] == "not_assessed"
    assert registry.compare("regression", math.nan, 2)["status"] == "not_assessed"


def test_registry_rejects_duplicate_or_unsafe_specs():
    spec = BaselineSpec("one", "data", "x", "rmse", "minimize", "description")
    with pytest.raises(BaselineRegistryError, match="duplicate"):
        BaselineRegistry([spec, spec])
