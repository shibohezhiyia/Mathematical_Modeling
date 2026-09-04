"""Ensure the forecasting wrapper preserves temporal validation semantics."""

from types import SimpleNamespace

import numpy as np
import pandas as pd

import core.time_series_forecaster as ts_module
from core.time_series_forecaster import TSConfig, TimeSeriesForecaster


def test_forecaster_uses_time_cv_and_reports_rmse_mean(monkeypatch, tmp_path):
    captured = {}

    class FakeEngine:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def fit(self, X, y):
            return SimpleNamespace(
                leaderboard=pd.DataFrame({"rmse_mean": [1.25]}),
                best_cv_result=SimpleNamespace(fitted_models=[object()]),
            )

    monkeypatch.setattr(ts_module, "ModelingEngine", FakeEngine)
    config = TSConfig(
        date_col="date",
        target_col="target",
        id_cols=["series"],
        model_keys=["ridge"],
        n_splits=3,
        output_dir=str(tmp_path),
        verbose=False,
    )
    data = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=50, freq="D"),
        "target": np.arange(50, dtype=float),
        "series": "one",
    })

    _, _, scores = TimeSeriesForecaster(config)._train_single(data)

    assert captured["fold_type"] == "time"
    assert scores["rmse_cv"] == 1.25
