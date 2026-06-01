"""
可视化扩展测试 — 覆盖边界条件与未测函数

补充 test_visualization.py 的覆盖缺口，重点测试：
- 无有效数据时的降级行为
- 便捷函数 plot_reward_curve / plot_optimization_history / plot_autoencoder_results
- ModelingResult 对象传入 plot_feature_importance
- 最小化输入的 plot_modeling_summary / plot_data_profile
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.visualization import (
    DataVisualizer,
    EvaluationVisualizer,
    ModelVisualizer,
    plot_autoencoder_results,
    plot_data_profile,
    plot_modeling_summary,
    plot_optimization_history,
    plot_reward_curve,
    _init_matplotlib,
    _MPL_AVAILABLE,
    _save_or_show,
)
from core.evaluation_engine import DecisionReport, DecisionMode, ModelScore, RiskLevel

skip_if_no_mpl = pytest.mark.skipif(not _MPL_AVAILABLE, reason="matplotlib 未安装")


@pytest.fixture(autouse=True)
def close_figs():
    yield
    if _MPL_AVAILABLE:
        import matplotlib.pyplot as plt
        plt.close("all")


@pytest.fixture
def sample_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "num_a": np.random.randn(100),
            "num_b": np.random.randn(100) * 2 + 5,
            "num_c": np.random.randint(0, 100, 100),
            "cat_x": np.random.choice(["A", "B", "C"], 100),
        }
    )


@pytest.fixture
def temp_save_path():
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    viz_dir = os.path.join(proj_root, "workspace", "reports", "test_viz_ext")
    os.makedirs(viz_dir, exist_ok=True)
    return os.path.join(viz_dir, "test_viz.png")


# =============================================================================
# DataVisualizer 边界
# =============================================================================

@skip_if_no_mpl
class TestDataVisualizerExtended:
    def test_plot_correlation_heatmap_exact_two_cols(self, temp_save_path):
        dv = DataVisualizer()
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})
        fig = dv.plot_correlation_heatmap(df, save_path=temp_save_path)
        assert fig is not None

    def test_plot_target_distribution_numpy_array(self, temp_save_path):
        dv = DataVisualizer()
        y = np.random.choice([0, 1, 2], 100)
        fig = dv.plot_target_distribution(y, task_type="classification", save_path=temp_save_path)
        assert fig is not None

    def test_plot_pairplot_default_columns(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_pairplot(sample_df, save_path=temp_save_path)
        assert fig is not None

    def test_plot_categorical_counts_top_n(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_categorical_counts(sample_df, "cat_x", top_n=2, save_path=temp_save_path)
        assert fig is not None

    def test_plot_distribution_without_hue(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_distribution(sample_df, "num_a", save_path=temp_save_path)
        assert fig is not None


# =============================================================================
# ModelVisualizer 边界
# =============================================================================

@skip_if_no_mpl
class TestModelVisualizerExtended:
    def test_plot_feature_importance_with_result_object(self, temp_save_path):
        mv = ModelVisualizer()
        fi = pd.DataFrame(
            {"feature": ["a", "b", "c"], "importance": [0.5, 0.3, 0.2]}
        )
        result = MagicMock()
        result.feature_importance = fi
        fig = mv.plot_feature_importance(result, save_path=temp_save_path)
        assert fig is not None

    def test_plot_confusion_matrix_with_labels(self, temp_save_path):
        mv = ModelVisualizer()
        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 2, 1, 0, 1, 2]
        fig = mv.plot_confusion_matrix(
            y_true, y_pred, labels=[0, 1, 2], normalize=True, save_path=temp_save_path
        )
        assert fig is not None

    def test_plot_cv_boxplot_single_model(self, temp_save_path):
        mv = ModelVisualizer()
        from core.modeling_engine import CVResult

        cv = CVResult(
            model_key="lr",
            model_name="LR",
            fold_scores={"f1": [0.7, 0.71, 0.72]},
            mean_scores={"f1": 0.71},
            std_scores={"f1": 0.01},
        )
        fig = mv.plot_cv_boxplot([cv], save_path=temp_save_path)
        assert fig is not None

    def test_plot_roc_curves_no_valid_data(self, temp_save_path):
        mv = ModelVisualizer()
        from core.modeling_engine import CVResult

        cv = CVResult(
            model_key="lr",
            model_name="LR",
            oof_proba=None,
            fold_scores={},
            mean_scores={},
            std_scores={},
        )
        fig = mv.plot_roc_curves([cv], y_true=np.array([0, 1, 0, 1]), save_path=temp_save_path)
        assert fig is None

    def test_plot_leaderboard_no_metric_found(self, temp_save_path):
        mv = ModelVisualizer()
        lb = pd.DataFrame({"model": ["A", "B"], "train_time": [1.0, 2.0]})
        fig = mv.plot_leaderboard(lb, save_path=temp_save_path)
        assert fig is None


# =============================================================================
# EvaluationVisualizer 边界
# =============================================================================

@skip_if_no_mpl
class TestEvaluationVisualizerExtended:
    def test_plot_radar_single_model(self, temp_save_path):
        ev = EvaluationVisualizer()
        scores = [
            ModelScore(
                model_key="lr",
                model_name="LogisticRegression",
                primary_metric="f1_weighted",
                primary_score=0.85,
                primary_std=0.02,
                train_time=2.0,
                n_parameters=1000,
                accuracy_score=90,
                speed_score=30,
                stability_score=85,
                simplicity_score=20,
                generalization_score=80,
                composite_score=61,
                rank=1,
                overfit_risk=RiskLevel.LOW,
                underfit_risk=RiskLevel.LOW,
            )
        ]
        report = DecisionReport(
            mode=DecisionMode.BALANCED,
            mode_description="平衡模式",
            recommended_model="lr",
            recommended_name="LogisticRegression",
            recommendation_reason="测试",
            confidence=0.75,
            scores=scores,
        )
        fig = ev.plot_radar_comparison(report, save_path=temp_save_path)
        assert fig is not None

    def test_plot_score_breakdown_single(self, temp_save_path):
        ev = EvaluationVisualizer()
        scores = [
            ModelScore(
                model_key="lr",
                model_name="LogisticRegression",
                primary_metric="f1_weighted",
                primary_score=0.85,
                primary_std=0.02,
                train_time=2.0,
                n_parameters=1000,
                accuracy_score=90,
                speed_score=30,
                stability_score=85,
                simplicity_score=20,
                generalization_score=80,
                composite_score=61,
                rank=1,
                overfit_risk=RiskLevel.LOW,
                underfit_risk=RiskLevel.LOW,
            )
        ]
        report = DecisionReport(
            mode=DecisionMode.BALANCED,
            mode_description="平衡模式",
            recommended_model="lr",
            recommended_name="LogisticRegression",
            recommendation_reason="测试",
            confidence=0.75,
            scores=scores,
        )
        fig = ev.plot_score_breakdown(report, save_path=temp_save_path)
        assert fig is not None

    def test_plot_risk_summary_single(self, temp_save_path):
        ev = EvaluationVisualizer()
        scores = [
            ModelScore(
                model_key="lr",
                model_name="LogisticRegression",
                primary_metric="f1_weighted",
                primary_score=0.85,
                primary_std=0.02,
                train_time=2.0,
                n_parameters=1000,
                accuracy_score=90,
                speed_score=30,
                stability_score=85,
                simplicity_score=20,
                generalization_score=80,
                composite_score=61,
                rank=1,
                overfit_risk=RiskLevel.LOW,
                underfit_risk=RiskLevel.LOW,
            )
        ]
        report = DecisionReport(
            mode=DecisionMode.BALANCED,
            mode_description="平衡模式",
            recommended_model="lr",
            recommended_name="LogisticRegression",
            recommendation_reason="测试",
            confidence=0.75,
            scores=scores,
        )
        fig = ev.plot_risk_summary(report, save_path=temp_save_path)
        assert fig is not None

    def test_plot_mode_comparison_empty(self, temp_save_path):
        ev = EvaluationVisualizer()
        fig = ev.plot_mode_comparison([], "classification", save_path=temp_save_path)
        # The implementation does not early-return for empty cv_results;
        # it returns a figure with no lines.
        assert fig is not None


# =============================================================================
# 便捷函数扩展
# =============================================================================

@skip_if_no_mpl
class TestConvenienceFunctionsExtended:
    def test_plot_modeling_summary_classification_minimal(self, tmp_path):
        from core.modeling_engine import CVResult, ModelingResult, TaskType

        cv = CVResult(
            model_key="lr",
            model_name="LR",
            fold_scores={},
            mean_scores={},
            std_scores={},
        )
        result = ModelingResult(
            task_type=TaskType.CLASSIFICATION,
            cv_results=[cv],
            best_model_key="lr",
            best_cv_result=cv,
            leaderboard=pd.DataFrame(),
        )
        paths = plot_modeling_summary(
            result,
            save_dir=str(tmp_path / "min_cls"),
            task_type="classification",
        )
        assert isinstance(paths, dict)

    def test_plot_modeling_summary_regression_minimal(self, tmp_path):
        from core.modeling_engine import CVResult, ModelingResult, TaskType

        cv = CVResult(
            model_key="ridge",
            model_name="Ridge",
            fold_scores={},
            mean_scores={},
            std_scores={},
        )
        result = ModelingResult(
            task_type=TaskType.REGRESSION,
            cv_results=[cv],
            best_model_key="ridge",
            best_cv_result=cv,
            leaderboard=pd.DataFrame(),
        )
        paths = plot_modeling_summary(
            result,
            save_dir=str(tmp_path / "min_reg"),
            task_type="regression",
        )
        assert isinstance(paths, dict)

    def test_plot_data_profile_no_target(self, tmp_path):
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "a": np.random.randn(50),
                "b": np.random.randn(50),
                "c": np.random.choice(["X", "Y"], 50),
            }
        )
        paths = plot_data_profile(df, save_dir=str(tmp_path / "profile_no_target"))
        assert isinstance(paths, dict)
        assert len(paths) > 0

    def test_plot_reward_curve(self, tmp_path):
        history = [
            {"trial": 1, "reward": 1.0, "epsilon": 0.5},
            {"trial": 2, "reward": 2.0, "epsilon": 0.4},
            {"trial": 3, "reward": 1.5, "epsilon": 0.3},
        ]
        path = str(tmp_path / "rl.png")
        result = plot_reward_curve(history, save_path=path)
        assert result is not None
        assert os.path.exists(result)

    def test_plot_reward_curve_empty(self):
        result = plot_reward_curve([])
        assert result is None

    def test_plot_optimization_history(self, tmp_path):
        history = {
            "model_a": [
                {"trial": 1, "score": 0.5},
                {"trial": 2, "score": 0.7},
            ],
            "model_b": [
                {"trial": 1, "score": 0.6},
            ],
        }
        path = str(tmp_path / "opt.png")
        result = plot_optimization_history(history, save_path=path)
        assert result is not None
        assert os.path.exists(result)

    def test_plot_optimization_history_empty(self):
        result = plot_optimization_history({})
        assert result is None

    def test_plot_autoencoder_results(self, tmp_path):
        np.random.seed(42)
        X_orig = np.random.randn(100, 10)
        X_reconstructed = X_orig + np.random.randn(100, 10) * 0.1
        encoded = np.random.randn(100, 4)
        path = str(tmp_path / "ae.png")
        result = plot_autoencoder_results(
            X_orig, X_reconstructed, encoded=encoded, save_path=path
        )
        assert result is not None
        assert os.path.exists(result)


# =============================================================================
# 内部工具函数
# =============================================================================

@skip_if_no_mpl
class TestInternalUtilities:
    def test_save_or_show_without_path(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        result = _save_or_show(fig, save_path=None)
        assert result is None
        plt.close(fig)

    def test_init_matplotlib_idempotent(self):
        assert _init_matplotlib() is True
