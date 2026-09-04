"""Tests that public ModelingEngine configuration is applied as requested."""

import pandas as pd
import pytest
import numpy as np

from core.modeling_engine import (
    AutoEncoder,
    EncodingType,
    EnsembleMethod,
    FeatureSelectionStrategy,
    ModelingEngine,
    TaskType,
    TaskTypeDetector,
)


def test_documented_string_options_are_parsed_without_silent_fallback():
    engine = ModelingEngine(
        encoding="target",
        feature_selection="rfe",
        ensemble="best_single",
        verbose=False,
    )

    assert engine.encoding is EncodingType.TARGET
    assert engine.feature_selection is FeatureSelectionStrategy.RFE
    assert engine.ensemble is EnsembleMethod.BEST_SINGLE


def test_explicit_encoding_strategy_overrides_auto_heuristic():
    X = pd.DataFrame({"city": ["A", "B", "A"]})
    encoded = AutoEncoder(strategy=EncodingType.LABEL).fit_transform(X)

    assert list(encoded.columns) == ["city"]
    assert encoded["city"].tolist() == [0, 1, 0]


def test_invalid_public_configuration_fails_fast():
    with pytest.raises(ValueError, match="融合策略"):
        ModelingEngine(ensemble="not-a-method", verbose=False)


def test_mape_excludes_zero_actuals_instead_of_exploding():
    mape = TaskTypeDetector.get_metrics_dict(TaskType.REGRESSION)["mape"]

    assert mape(np.array([0.0, 100.0]), np.array([5.0, 110.0])) == 10.0
    assert mape(np.array([0.0]), np.array([5.0])) == 0.0


def test_selected_model_is_refit_on_all_training_rows_after_cv():
    X = pd.DataFrame({
        "x1": np.linspace(-2, 2, 60),
        "x2": np.linspace(2, -2, 60),
    })
    y = pd.Series(3 * X["x1"] - X["x2"])
    engine = ModelingEngine(
        task_type="regression",
        model_keys=["ridge"],
        n_splits=3,
        feature_selection="none",
        ensemble="best_single",
        auto_sample=False,
        n_jobs=1,
        verbose=False,
    )

    result = engine.fit(X, y)

    assert result.preprocessing_info["final_refit_samples"] == len(X)
    assert len(result.best_cv_result.fitted_models) == 4
    assert engine.predict(X.iloc[:5]).shape == (5,)


def test_target_encoding_is_downgraded_to_fold_safe_frequency_encoding():
    X = pd.DataFrame({"category": [f"c{i}" for i in range(60)] * 2})
    y = pd.Series(np.tile([0, 1], 60))
    encoder = AutoEncoder(strategy=EncodingType.TARGET).fit(X, y)

    report = encoder.get_encoding_report()
    assert report.loc[0, "strategy"] == "frequency"
    assert pd.api.types.is_numeric_dtype(encoder.transform(X)["category"])


def test_group_metadata_is_isolated_before_preprocessing_and_prediction():
    groups = np.repeat(np.arange(12), 5)
    X = pd.DataFrame({
        "entity_id": groups,
        "x": np.linspace(-2, 2, len(groups)),
    })
    y = pd.Series(3 * X["x"] + np.sin(groups))
    engine = ModelingEngine(
        task_type="regression",
        model_keys=["ridge"],
        n_splits=3,
        feature_selection="none",
        ensemble="best_single",
        fold_type="group",
        group_col="entity_id",
        auto_sample=False,
        n_jobs=1,
        verbose=False,
    )

    result = engine.fit(X, y)

    assert result.preprocessing_info["original_features"] == 1
    assert len(result.best_cv_result.oof_pred) == len(X)
    assert engine.predict(X.iloc[:3]).shape == (3,)
