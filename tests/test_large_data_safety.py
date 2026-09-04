"""Resource-budget regressions for large and wide datasets."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor

import core.permutation_importance as pi_module
from core.drift_detection import DriftDetector
from core.modeling_engine import (
    AutoEncoder,
    EncodingType,
    ModelingEngine,
    TaskType,
)
from core.integrated_pipeline import IntegratedPipeline
from core.permutation_importance import compute_permutation_importance


def test_high_cardinality_dense_onehot_falls_back_to_frequency():
    X = pd.DataFrame({"user_id": [f"user-{i}" for i in range(500)]})

    encoder = AutoEncoder(strategy=EncodingType.ONEHOT, max_dense_onehot_categories=100)
    encoded = encoder.fit_transform(X)

    assert encoded.shape == (500, 1)
    assert encoder.get_encoding_report().iloc[0]["strategy"] == "frequency"


def test_large_workload_prunes_default_models_and_serializes_cv():
    engine = ModelingEngine(model_keys=None, n_jobs=16, verbose=False)
    engine._original_shape = (500_000, 100)
    engine._large_workload = True
    candidates = {
        key: object()
        for key in ["svm", "knn", "mlp", "hist_gb", "sgd", "lr", "rf", "et"]
    }

    selected = engine._apply_large_data_model_guards(
        candidates, TaskType.CLASSIFICATION, n_samples=20_000
    )

    assert list(selected) == ["hist_gb", "sgd", "lr"]
    assert engine._get_effective_cv_jobs(n_splits=3) == 1
    assert engine._get_effective_cv_folds(n_samples=20_000) == 2


def test_large_regression_sgd_does_not_receive_classifier_only_n_jobs():
    engine = ModelingEngine(verbose=False)
    engine._large_workload = True

    model = engine._create_runtime_model("sgd", TaskType.REGRESSION)

    assert model.__class__.__name__ == "SGDRegressor"


def test_wide_table_adaptive_sample_limit_uses_cell_budget():
    engine = ModelingEngine(max_samples=50_000, verbose=False)
    X = pd.DataFrame(np.zeros((200, 20_000), dtype=np.float32))

    limit = engine._get_adaptive_sample_limit(X)

    assert 100 <= limit <= 400


def test_permutation_importance_caps_rows_and_columns(monkeypatch):
    captured = {}

    def fake_pi(model, X, y, **kwargs):
        captured["shape"] = X.shape
        return SimpleNamespace(
            importances_mean=np.zeros(X.shape[1]),
            importances_std=np.zeros(X.shape[1]),
        )

    monkeypatch.setattr(pi_module, "permutation_importance", fake_pi)
    X = pd.DataFrame(np.random.RandomState(42).normal(size=(6000, 150)))
    y = pd.Series(np.random.RandomState(43).normal(size=6000))

    result = compute_permutation_importance(
        DummyRegressor(), X, y, max_samples=5000, max_features=100
    )

    assert captured["shape"][0] <= 1000
    assert captured["shape"][1] == 100
    assert len(result) == 100


def test_drift_detector_caps_reference_copy():
    X = pd.DataFrame({"x": np.arange(25_000), "y": np.arange(25_000) % 3})
    detector = DriftDetector(max_samples=2000).fit_reference(X)

    assert detector._reference_n_rows == 25_000
    assert len(detector._reference) == 2000


def test_integrated_auto_encoding_does_not_force_dense_onehot():
    n_train, n_test = 300, 50
    df = pd.DataFrame({
        "user_id": [f"user-{i}" for i in range(n_train + n_test)],
        "value": np.arange(n_train + n_test, dtype=float),
        "target": np.r_[np.arange(n_train) % 2, [np.nan] * n_test],
    })
    pipeline = IntegratedPipeline(
        strategy_preference="fast",
        target_col="target",
        task_type="classification",
        model_keys=["lr"],
        feature_selection="none",
        ensemble="best_single",
        n_splits=2,
        visualization=False,
    )

    result = pipeline.run(df)
    report = result.modeling_result.encoding_report

    user_strategy = report.loc[report["column"] == "user_id", "strategy"].iloc[0]
    assert user_strategy != "onehot"
