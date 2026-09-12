"""End-to-end comparison between a frozen baseline and the current system.

The public-data benchmark historically compared two sklearn estimators.  This
module adds a real treatment arm: the same train/test split is passed through
``ModelingEngine`` with preprocessing, cross-validation and model competition.
It deliberately returns ``not_assessed`` on any execution failure instead of
turning a partial run into an accuracy claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCHEMA_VERSION = "mathmodel.system-benchmark/v1"


def _digest(values: Sequence[float]) -> str:
    payload = json.dumps([float(x) for x in values], separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _score(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if y_true.ndim != 1 or prediction.ndim != 1 or y_true.shape != prediction.shape:
        raise ValueError("prediction_shape_invalid")
    if not np.isfinite(y_true).all() or not np.isfinite(prediction).all():
        raise ValueError("prediction_non_finite")
    return {
        "mae": float(mean_absolute_error(y_true, prediction)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, prediction))),
        "r2": float(r2_score(y_true, prediction)) if np.unique(y_true).size > 1 else None,
        "prediction_digest": _digest(prediction.tolist()),
    }


def _fit_baseline(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
    model.fit(X_train, y_train)
    return np.asarray(model.predict(X_test), dtype=float)


def _fit_current_system(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    *,
    random_state: int,
    model_keys: Sequence[str] | None,
    n_splits: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    # Import lazily: importing the benchmark must remain cheap and must not
    # pull optional deep-learning dependencies into the public-data scorer.
    from .modeling_engine import ModelingEngine

    selected = list(model_keys) if model_keys else ["ridge", "hist_gb", "et", "residual_stack"]
    engine = ModelingEngine(
        task_type="regression", model_keys=selected,
        n_splits=max(2, min(int(n_splits), 5)), encoding="auto", feature_selection="none",
        ensemble="best_single", optimize_hyperparams=False, auto_sample=False,
        max_samples=max(128, len(X_train)), use_meta_learning=False,
        random_state=int(random_state), n_jobs=1, verbose=False,
    )
    result = engine.fit(X_train.reset_index(drop=True), y_train.reset_index(drop=True), X_test.reset_index(drop=True))
    prediction = None
    if result.ensemble_result:
        prediction = result.ensemble_result.get("test")
    if prediction is None:
        prediction = engine.predict(X_test.reset_index(drop=True))
    diagnostics = {
        "best_model_key": result.best_model_key,
        "trained_model_count": len(result.cv_results or []),
        "leaderboard_rows": int(len(result.leaderboard)) if result.leaderboard is not None else 0,
    }
    return np.asarray(prediction, dtype=float), diagnostics


def compare_frozen_systems(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    *,
    random_state: int = 42,
    model_keys: Sequence[str] | None = None,
    n_splits: int = 3,
) -> dict[str, Any]:
    """Run baseline and current system on exactly the same frozen split."""
    started = time.perf_counter()
    if not isinstance(X_train, pd.DataFrame) or not isinstance(X_test, pd.DataFrame):
        raise TypeError("features_must_be_dataframes")
    ytr, yte = pd.to_numeric(pd.Series(y_train), errors="coerce"), pd.to_numeric(pd.Series(y_test), errors="coerce")
    if len(X_train) != len(ytr) or len(X_test) != len(yte) or len(yte) == 0:
        raise ValueError("split_lengths_invalid")
    try:
        baseline_prediction = _fit_baseline(X_train, ytr, X_test)
        baseline = _score(yte.to_numpy(dtype=float), baseline_prediction)
    except Exception as exc:
        return {"schema_version": SCHEMA_VERSION, "status": "not_assessed",
                "failure_code": "baseline_failed", "error_type": type(exc).__name__}
    try:
        treatment_prediction, diagnostics = _fit_current_system(
            X_train, ytr, X_test, random_state=random_state,
            model_keys=model_keys, n_splits=n_splits,
        )
        treatment = _score(yte.to_numpy(dtype=float), treatment_prediction)
    except Exception as exc:
        return {"schema_version": SCHEMA_VERSION, "status": "not_assessed",
                "baseline": baseline, "failure_code": "current_system_failed",
                "error_type": type(exc).__name__}
    effect = (treatment["rmse"] / baseline["rmse"] - 1.0) if baseline["rmse"] > 0 else None
    return {
        "schema_version": SCHEMA_VERSION, "status": "completed",
        "baseline": baseline, "current_system": treatment,
        "current_system_diagnostics": diagnostics,
        "relative_rmse_effect_current_minus_baseline": effect,
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "feature_count": int(X_train.shape[1]),
        "random_state": int(random_state), "n_splits": int(n_splits),
        "elapsed_seconds": float(time.perf_counter() - started),
        "policy": "paired_frozen_split; current_system_is_not_independent_expert_review",
    }


__all__ = ["SCHEMA_VERSION", "compare_frozen_systems"]
