"""
并行建模扩展测试
覆盖 Metrics、HyperparameterSearch、ParallelModelingEngine、quick_model
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.parallel_modeling import (
    HyperparameterSearch,
    Metrics,
    ModelConfig,
    ModelRegistry,
    ModelResult,
    ParallelModelingEngine,
    quick_model,
)
from core.performance_scheduler import ExecutionPlan, StrategyLevel


@pytest.fixture(autouse=True)
def reset_registry():
    ModelRegistry._initialized = False
    ModelRegistry._models = {}
    ModelRegistry._init()
    yield
    ModelRegistry._initialized = False
    ModelRegistry._models = {}


# =============================================================================
# Metrics
# =============================================================================

class TestMetrics:
    def test_get_metrics_classification(self):
        metrics = Metrics.get_metrics("classification")
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "f1_macro" in metrics
        assert "f1_weighted" in metrics
        assert all(callable(v) for v in metrics.values())

    def test_get_metrics_regression(self):
        metrics = Metrics.get_metrics("regression")
        assert "rmse" in metrics
        assert "mae" in metrics
        assert "r2" in metrics
        assert "mse" in metrics
        assert all(callable(v) for v in metrics.values())

    def test_get_metrics_invalid_returns_regression(self):
        metrics = Metrics.get_metrics("unknown")
        assert "rmse" in metrics

    def test_accuracy_metric(self):
        y_true = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 1])
        score = Metrics.CLASSIFICATION["accuracy"](y_true, y_pred)
        assert score == pytest.approx(0.8)

    def test_auc_metric(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.4, 0.35, 0.8])
        score = Metrics.CLASSIFICATION["auc"](y_true, y_pred)
        assert 0.0 <= score <= 1.0

    def test_f1_macro_metric(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 1])
        score = Metrics.CLASSIFICATION["f1_macro"](y_true, y_pred)
        assert 0.0 <= score <= 1.0

    def test_f1_weighted_metric(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        score = Metrics.CLASSIFICATION["f1_weighted"](y_true, y_pred)
        assert 0.0 <= score <= 1.0

    def test_rmse_metric(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        score = Metrics.REGRESSION["rmse"](y_true, y_pred)
        assert score > 0

    def test_mae_metric(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        score = Metrics.REGRESSION["mae"](y_true, y_pred)
        assert score > 0

    def test_r2_metric(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        score = Metrics.REGRESSION["r2"](y_true, y_pred)
        assert score <= 1.0

    def test_mse_metric(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        score = Metrics.REGRESSION["mse"](y_true, y_pred)
        assert score > 0

    def test_evaluate_classification(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1])
        proba = np.array([0.2, 0.8, 0.6, 0.9])
        result = Metrics.evaluate(y_true, y_pred, "classification", proba)
        assert "accuracy" in result
        assert "auc" in result
        assert result["accuracy"] is not None

    def test_evaluate_regression(self):
        y_true = np.array([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        result = Metrics.evaluate(y_true, y_pred, "regression")
        assert "rmse" in result
        assert "r2" in result

    def test_evaluate_multiclass_auc(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 1, 2])
        proba = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.1, 0.8, 0.1],
                [0.1, 0.2, 0.7],
                [0.6, 0.3, 0.1],
                [0.2, 0.7, 0.1],
                [0.1, 0.3, 0.6],
            ]
        )
        result = Metrics.evaluate(y_true, y_pred, "classification", proba)
        assert "auc" in result
        assert result["auc"] is not None


# =============================================================================
# HyperparameterSearch
# =============================================================================

class TestHyperparameterSearch:
    def test_init_sets_params(self):
        hs = HyperparameterSearch(n_trials=15, random_state=123)
        assert hs.n_trials == 15
        assert hs.random_state == 123

    def test_search_small_grid_classification(self):
        config = ModelConfig(
            name="Linear",
            model_class={"classification": LogisticRegression},
            default_params={"max_iter": 1000, "random_state": 42},
            param_distributions={"C": [0.1, 1.0]},
            task_type="classification",
        )
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        hs = HyperparameterSearch(n_trials=3, random_state=42)
        hs._use_optuna = False
        best_params, best_score = hs.search(config, X, y, "classification", cv_folds=2)
        assert isinstance(best_params, dict)
        assert isinstance(best_score, float)

    def test_search_random_strategy_regression(self):
        config = ModelConfig(
            name="Linear",
            model_class={"regression": Ridge},
            default_params={"random_state": 42},
            param_distributions={"alpha": [0.1, 1.0]},
            task_type="regression",
        )
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(40, 3))
        y = np.random.randn(40)
        hs = HyperparameterSearch(n_trials=3, random_state=42)
        hs._use_optuna = False
        best_params, best_score = hs.search(config, X, y, "regression", cv_folds=2)
        assert isinstance(best_params, dict)
        assert isinstance(best_score, (float, int))

    def test_search_empty_param_distributions(self):
        config = ModelConfig(
            name="Linear",
            model_class={"classification": LogisticRegression},
            default_params={},
            param_distributions={},
            task_type="classification",
        )
        X = pd.DataFrame(np.random.randn(30, 2))
        y = np.random.choice([0, 1], 30)
        hs = HyperparameterSearch(n_trials=3, random_state=42)
        best_params, best_score = hs.search(config, X, y, "classification", cv_folds=2)
        assert best_params == {}
        assert best_score == 0.0


# =============================================================================
# ParallelModelingEngine
# =============================================================================

class TestParallelModelingEngine:
    @staticmethod
    def _make_engine(task_type="classification"):
        plan = ExecutionPlan(
            strategy=StrategyLevel.STANDARD,
            n_jobs=1,
            cv_folds=2,
            hyperparameter_trials=0,
            use_gpu=False,
        )
        return ParallelModelingEngine(task_type=task_type, plan=plan, random_state=42)

    def test_fit_classification(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 4))
        y = np.random.choice([0, 1], 60)
        engine = self._make_engine("classification")
        engine.fit(X, y, model_keys=["linear"])
        assert len(engine.results) > 0
        assert len(engine.leaderboard) > 0
        assert "linear" in engine.results

    def test_fit_regression(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 4))
        y = np.random.randn(60)
        engine = self._make_engine("regression")
        engine.fit(X, y, model_keys=["linear"])
        assert len(engine.results) > 0
        assert len(engine.leaderboard) > 0
        assert "linear" in engine.results

    def test_fit_with_test_set(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 3))
        y = np.random.choice([0, 1], 60)
        X_test = pd.DataFrame(np.random.randn(20, 3))
        engine = self._make_engine("classification")
        engine.fit(X, y, X_test=X_test, model_keys=["linear"])
        result = engine.results["linear"]
        assert result.test_predictions is not None
        assert len(result.test_predictions) == 20

    def test_extract_feature_importance_tree(self):
        engine = self._make_engine("classification")
        model = DecisionTreeClassifier(random_state=42)
        X = pd.DataFrame(np.random.randn(50, 4), columns=["a", "b", "c", "d"])
        y = np.random.choice([0, 1], 50)
        model.fit(X, y)
        fi = engine._extract_feature_importance(model, feature_names=list(X.columns))
        assert fi is not None
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) == 4

    def test_extract_feature_importance_linear(self):
        engine = self._make_engine("classification")
        model = LogisticRegression(random_state=42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        model.fit(X, y)
        fi = engine._extract_feature_importance(model, feature_names=["f1", "f2", "f3"])
        assert fi is not None
        assert len(fi) == 3

    def test_extract_feature_importance_no_attr(self):
        engine = self._make_engine("classification")
        fi = engine._extract_feature_importance(object(), None)
        assert fi is None

    def test_blend_weighted(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(40, 3))
        y = np.random.choice([0, 1], 40)
        m1 = LogisticRegression(random_state=42).fit(X, y)
        m2 = LogisticRegression(random_state=43).fit(X, y)
        r1 = ModelResult("m1", "M1", m1, {}, 0.9, {}, 0.1, None, None, None)
        r2 = ModelResult("m2", "M2", m2, {}, 0.8, {}, 0.1, None, None, None)
        engine = self._make_engine("classification")
        engine.leaderboard = [r1, r2]
        X_test = pd.DataFrame(np.random.randn(10, 3))
        preds = engine._blend_weighted(engine.leaderboard, X_test)
        assert len(preds) == 10
        assert isinstance(preds, np.ndarray)

    def test_blend_average(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(40, 3))
        y = np.random.choice([0, 1], 40)
        m1 = LogisticRegression(random_state=42).fit(X, y)
        m2 = LogisticRegression(random_state=43).fit(X, y)
        r1 = ModelResult("m1", "M1", m1, {}, 0.9, {}, 0.1, None, None, None)
        r2 = ModelResult("m2", "M2", m2, {}, 0.8, {}, 0.1, None, None, None)
        engine = self._make_engine("classification")
        engine.leaderboard = [r1, r2]
        X_test = pd.DataFrame(np.random.randn(10, 3))
        preds = engine._blend_average(engine.leaderboard, X_test)
        assert len(preds) == 10
        assert isinstance(preds, np.ndarray)

    def test_get_leaderboard(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        engine = self._make_engine("classification")
        engine.fit(X, y, model_keys=["linear"])
        lb = engine.get_leaderboard()
        assert isinstance(lb, pd.DataFrame)
        assert "rank" in lb.columns
        assert "model" in lb.columns
        assert "cv_score" in lb.columns

    def test_get_leaderboard_empty(self):
        engine = self._make_engine("classification")
        lb = engine.get_leaderboard()
        assert lb.empty

    def test_predict_average(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        X_test = pd.DataFrame(np.random.randn(10, 3))
        engine = self._make_engine("classification")
        engine.fit(X, y, model_keys=["linear"])
        preds = engine.predict(X_test, blend_method="average")
        assert len(preds) == 10

    def test_predict_weighted(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        X_test = pd.DataFrame(np.random.randn(10, 3))
        engine = self._make_engine("classification")
        engine.fit(X, y, model_keys=["linear"])
        preds = engine.predict(X_test, blend_method="weighted")
        assert len(preds) == 10

    def test_predict_stacking(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        X_test = pd.DataFrame(np.random.randn(10, 3))
        engine = self._make_engine("classification")
        engine.fit(X, y, model_keys=["linear"])
        preds = engine.predict(X_test, blend_method="stacking")
        assert len(preds) == 10

    def test_predict_top_k(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        X_test = pd.DataFrame(np.random.randn(10, 3))
        engine = self._make_engine("classification")
        engine.fit(X, y, model_keys=["linear"])
        preds = engine.predict(X_test, top_k=1)
        assert len(preds) == 10

    def test_predict_not_trained(self):
        engine = self._make_engine("classification")
        with pytest.raises(ValueError, match="尚未训练"):
            engine.predict(pd.DataFrame(np.random.randn(5, 2)))


# =============================================================================
# quick_model
# =============================================================================

class TestQuickModel:
    def test_quick_model_classification(self, capsys):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        X_test = pd.DataFrame(np.random.randn(10, 3))
        plan = ExecutionPlan(hyperparameter_trials=0, cv_folds=2, n_jobs=1)
        preds = quick_model(X, y, X_test, task_type="classification", plan=plan)
        assert preds is not None
        assert len(preds) == 10
        captured = capsys.readouterr()
        assert "并行建模结果摘要" in captured.out

    def test_quick_model_regression(self, capsys):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.randn(50)
        X_test = pd.DataFrame(np.random.randn(10, 3))
        plan = ExecutionPlan(hyperparameter_trials=0, cv_folds=2, n_jobs=1)
        preds = quick_model(X, y, X_test, task_type="regression", plan=plan)
        assert preds is not None
        assert len(preds) == 10

    def test_quick_model_return_engine(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        plan = ExecutionPlan(hyperparameter_trials=0, cv_folds=2, n_jobs=1)
        preds, engine = quick_model(
            X, y, task_type="classification", plan=plan, return_engine=True
        )
        assert isinstance(engine, ParallelModelingEngine)

    def test_quick_model_no_test_set(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = np.random.choice([0, 1], 50)
        plan = ExecutionPlan(hyperparameter_trials=0, cv_folds=2, n_jobs=1)
        preds = quick_model(X, y, task_type="classification", plan=plan)
        assert preds is None
