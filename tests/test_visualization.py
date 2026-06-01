"""
可视化模块测试

覆盖：
1. DataVisualizer（分布、相关性、缺失值、目标分布、类别计数）
2. ModelVisualizer（特征重要性、排行榜、混淆矩阵、残差、预测散点、ROC、CV箱线）
3. EvaluationVisualizer（雷达图、得分分解、模式对比、风险摘要）
4. 便捷函数（plot_modeling_summary, plot_data_profile）
5. 与 IntegratedPipeline 集成
"""

import os
import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from core.visualization import (
    DataVisualizer, ModelVisualizer, EvaluationVisualizer,
    plot_modeling_summary, plot_data_profile,
    _init_matplotlib, _MPL_AVAILABLE
)
from core.evaluation_engine import ModelScore, DecisionReport, DecisionMode, RiskLevel
from core.workspace_manager import get_workspace_manager


# =============================================================================
# 跳过条件
# =============================================================================

skip_if_no_mpl = pytest.mark.skipif(not _MPL_AVAILABLE, reason="matplotlib 未安装")


# =============================================================================
# 测试数据
# =============================================================================

@pytest.fixture
def sample_df():
    """样本数据框"""
    np.random.seed(42)
    return pd.DataFrame({
        'num_a': np.random.randn(100),
        'num_b': np.random.randn(100) * 2 + 5,
        'num_c': np.random.randint(0, 100, 100),
        'cat_x': np.random.choice(['A', 'B', 'C'], 100),
        'cat_y': np.random.choice(['X', 'Y', 'Z', 'W'], 100),
        'missing_col': np.where(np.random.rand(100) > 0.8, np.nan, np.random.randn(100)),
    })


@pytest.fixture
def classification_y():
    return pd.Series(np.random.choice([0, 1], 100))


@pytest.fixture
def regression_y():
    return pd.Series(np.random.randn(100))


@pytest.fixture
def mock_decision_report():
    """模拟决策报告"""
    scores = [
        ModelScore(
            model_key='xgb', model_name='XGBoost',
            primary_metric='f1_weighted', primary_score=0.85, primary_std=0.02,
            train_time=2.0, n_parameters=1000,
            accuracy_score=90, speed_score=30, stability_score=85,
            simplicity_score=20, generalization_score=80,
            composite_score=61, rank=1,
            overfit_risk=RiskLevel.LOW, underfit_risk=RiskLevel.LOW
        ),
        ModelScore(
            model_key='lr', model_name='LogisticRegression',
            primary_metric='f1_weighted', primary_score=0.75, primary_std=0.01,
            train_time=0.5, n_parameters=100,
            accuracy_score=50, speed_score=95, stability_score=95,
            simplicity_score=90, generalization_score=85,
            composite_score=83, rank=2,
            overfit_risk=RiskLevel.LOW, underfit_risk=RiskLevel.LOW
        ),
    ]
    report = DecisionReport(
        mode=DecisionMode.BALANCED,
        mode_description="平衡模式：综合考虑精度、速度、稳定性",
        recommended_model='lr',
        recommended_name='LogisticRegression',
        recommendation_reason="测试推荐",
        confidence=0.75,
        scores=scores
    )
    report.comparison_table = pd.DataFrame([
        {'模型': 'LogisticRegression', '排名': 1},
        {'模型': 'XGBoost', '排名': 2},
    ])
    return report


@pytest.fixture
def temp_save_path(tmp_path):
    """临时保存路径（使用项目目录避免C盘重定向）"""
    # 使用项目workspace目录，避免C盘路径被重定向导致断言失败
    import os
    proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    viz_dir = os.path.join(proj_root, 'workspace', 'reports', 'test_viz_temp')
    os.makedirs(viz_dir, exist_ok=True)
    return os.path.join(viz_dir, "test_viz.png")


# =============================================================================
# DataVisualizer 测试
# =============================================================================

@skip_if_no_mpl
class TestDataVisualizer:
    
    def test_plot_distribution(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_distribution(sample_df, 'num_a', save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_distribution_with_hue(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_distribution(sample_df, 'num_a', hue='cat_x', save_path=temp_save_path)
        assert fig is not None
    
    def test_plot_distribution_missing_col(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_distribution(sample_df, 'nonexistent', save_path=temp_save_path)
        assert fig is None
    
    def test_plot_correlation_heatmap(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_correlation_heatmap(sample_df, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_correlation_heatmap_insufficient_cols(self, temp_save_path):
        dv = DataVisualizer()
        df = pd.DataFrame({'a': [1, 2, 3]})
        fig = dv.plot_correlation_heatmap(df, save_path=temp_save_path)
        assert fig is None
    
    def test_plot_missing_values(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_missing_values(sample_df, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_missing_values_no_missing(self, temp_save_path):
        dv = DataVisualizer()
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        fig = dv.plot_missing_values(df, save_path=temp_save_path)
        assert fig is not None
    
    def test_plot_target_distribution_classification(self, classification_y, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_target_distribution(classification_y, task_type='classification', save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_target_distribution_regression(self, regression_y, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_target_distribution(regression_y, task_type='regression', save_path=temp_save_path)
        assert fig is not None
    
    def test_plot_categorical_counts(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_categorical_counts(sample_df, 'cat_x', save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_categorical_counts_missing_col(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_categorical_counts(sample_df, 'nonexistent', save_path=temp_save_path)
        assert fig is None
    
    def test_plot_pairplot(self, sample_df, temp_save_path):
        dv = DataVisualizer()
        fig = dv.plot_pairplot(sample_df, columns=['num_a', 'num_b'], hue='cat_x', save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)


# =============================================================================
# ModelVisualizer 测试
# =============================================================================

@skip_if_no_mpl
class TestModelVisualizer:
    
    def test_plot_feature_importance(self, temp_save_path):
        mv = ModelVisualizer()
        fi = pd.DataFrame({
            'feature': ['a', 'b', 'c', 'd'],
            'importance': [0.4, 0.3, 0.2, 0.1]
        })
        fig = mv.plot_feature_importance(fi, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_feature_importance_empty(self, temp_save_path):
        mv = ModelVisualizer()
        fig = mv.plot_feature_importance(pd.DataFrame(), save_path=temp_save_path)
        assert fig is None
    
    def test_plot_leaderboard(self, temp_save_path):
        mv = ModelVisualizer()
        lb = pd.DataFrame({
            'model': ['XGBoost', 'LR', 'RF'],
            'mean_f1_weighted': [0.85, 0.75, 0.80],
            'std_f1_weighted': [0.02, 0.01, 0.03]
        })
        fig = mv.plot_leaderboard(lb, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_leaderboard_none(self, temp_save_path):
        mv = ModelVisualizer()
        fig = mv.plot_leaderboard(None, save_path=temp_save_path)
        assert fig is None
    
    def test_plot_confusion_matrix(self, temp_save_path):
        mv = ModelVisualizer()
        y_true = [0, 1, 0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 0, 0, 0, 1, 1, 1]
        fig = mv.plot_confusion_matrix(y_true, y_pred, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_confusion_matrix_normalized(self, temp_save_path):
        mv = ModelVisualizer()
        y_true = [0, 1, 0, 1, 0, 1, 1, 0]
        y_pred = [0, 1, 0, 0, 0, 1, 1, 1]
        fig = mv.plot_confusion_matrix(y_true, y_pred, normalize=True, save_path=temp_save_path)
        assert fig is not None
    
    def test_plot_residuals(self, temp_save_path):
        mv = ModelVisualizer()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        fig = mv.plot_residuals(y_true, y_pred, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_prediction_scatter(self, temp_save_path):
        mv = ModelVisualizer()
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        fig = mv.plot_prediction_scatter(y_true, y_pred, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_cv_boxplot(self, temp_save_path):
        mv = ModelVisualizer()
        
        # 模拟 CVResult
        from core.modeling_engine import CVResult
        cv1 = CVResult(
            model_key='lr', model_name='LR',
            fold_scores={'f1': [0.7, 0.75, 0.73, 0.72, 0.74]},
            mean_scores={'f1': 0.73}, std_scores={'f1': 0.02}
        )
        cv2 = CVResult(
            model_key='xgb', model_name='XGB',
            fold_scores={'f1': [0.8, 0.85, 0.83, 0.82, 0.84]},
            mean_scores={'f1': 0.83}, std_scores={'f1': 0.02}
        )
        
        fig = mv.plot_cv_boxplot([cv1, cv2], save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_roc_curves(self, temp_save_path):
        mv = ModelVisualizer()
        
        from core.modeling_engine import CVResult
        y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1])
        proba = np.array([[0.8, 0.2], [0.7, 0.3], [0.3, 0.7], [0.2, 0.8],
                         [0.9, 0.1], [0.4, 0.6], [0.6, 0.4], [0.1, 0.9]])
        
        cv = CVResult(
            model_key='lr', model_name='LR',
            oof_proba=proba,
            fold_scores={}, mean_scores={}, std_scores={}
        )
        
        fig = mv.plot_roc_curves([cv], y_true=y_true, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)


# =============================================================================
# EvaluationVisualizer 测试
# =============================================================================

@skip_if_no_mpl
class TestEvaluationVisualizer:
    
    def test_plot_radar_comparison(self, mock_decision_report, temp_save_path):
        ev = EvaluationVisualizer()
        fig = ev.plot_radar_comparison(mock_decision_report, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_radar_empty(self, temp_save_path):
        ev = EvaluationVisualizer()
        report = DecisionReport(mode=DecisionMode.BALANCED, scores=[])
        fig = ev.plot_radar_comparison(report, save_path=temp_save_path)
        assert fig is None
    
    def test_plot_score_breakdown(self, mock_decision_report, temp_save_path):
        ev = EvaluationVisualizer()
        fig = ev.plot_score_breakdown(mock_decision_report, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_mode_comparison(self, temp_save_path):
        ev = EvaluationVisualizer()
        
        from core.modeling_engine import CVResult
        cv_results = [
            CVResult(model_key='lr', model_name='LR',
                    fold_scores={'f1': [0.7, 0.71, 0.72]},
                    mean_scores={'f1': 0.71}, std_scores={'f1': 0.01}, train_time=0.5),
            CVResult(model_key='xgb', model_name='XGB',
                    fold_scores={'f1': [0.8, 0.85, 0.82]},
                    mean_scores={'f1': 0.82}, std_scores={'f1': 0.03}, train_time=2.0),
        ]
        
        fig = ev.plot_mode_comparison(cv_results, 'classification', save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)
    
    def test_plot_risk_summary(self, mock_decision_report, temp_save_path):
        ev = EvaluationVisualizer()
        fig = ev.plot_risk_summary(mock_decision_report, save_path=temp_save_path)
        assert fig is not None
        assert os.path.exists(temp_save_path)


# =============================================================================
# 便捷函数测试
# =============================================================================

@skip_if_no_mpl
class TestConvenienceFunctions:
    
    def test_plot_data_profile(self, sample_df, tmp_path):
        paths = plot_data_profile(sample_df, target='cat_x', task_type='classification',
                                   save_dir=str(tmp_path / 'data_profile'))
        assert isinstance(paths, dict)
        assert len(paths) > 0
        for p in paths.values():
            assert os.path.exists(p)
    
    def test_plot_modeling_summary_classification(self, tmp_path):
        from core.modeling_engine import ModelingResult, CVResult, TaskType
        
        cv = CVResult(
            model_key='lr', model_name='LR',
            fold_scores={'f1': [0.7, 0.71, 0.72]},
            mean_scores={'f1': 0.71}, std_scores={'f1': 0.01},
            oof_pred=np.array([0, 1, 0, 1]),
            oof_proba=np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1], [0.2, 0.8]])
        )
        
        result = ModelingResult(
            task_type=TaskType.CLASSIFICATION,
            cv_results=[cv],
            best_model_key='lr',
            best_cv_result=cv,
            leaderboard=pd.DataFrame({'model': ['LR'], 'mean_f1': [0.71]})
        )
        
        paths = plot_modeling_summary(result, y_train=np.array([0, 1, 0, 1]),
                                       save_dir=str(tmp_path / 'model_summary'),
                                       task_type='classification')
        assert isinstance(paths, dict)
        assert len(paths) > 0
    
    def test_plot_modeling_summary_regression(self, tmp_path):
        from core.modeling_engine import ModelingResult, CVResult, TaskType
        
        cv = CVResult(
            model_key='lr', model_name='LR',
            fold_scores={'rmse': [0.5, 0.6, 0.55]},
            mean_scores={'rmse': 0.55}, std_scores={'rmse': 0.05},
            oof_pred=np.array([1.1, 2.0, 3.2])
        )
        
        result = ModelingResult(
            task_type=TaskType.REGRESSION,
            cv_results=[cv],
            best_model_key='lr',
            best_cv_result=cv,
            leaderboard=pd.DataFrame({'model': ['LR'], 'mean_rmse': [0.55]})
        )
        
        paths = plot_modeling_summary(result, y_train=np.array([1, 2, 3]),
                                       save_dir=str(tmp_path / 'reg_summary'),
                                       task_type='regression')
        assert isinstance(paths, dict)
        assert len(paths) > 0


# =============================================================================
# 与 IntegratedPipeline 集成测试
# =============================================================================

@skip_if_no_mpl
class TestPipelineIntegration:
    
    def test_pipeline_with_visualization(self, tmp_path):
        from core.integrated_pipeline import IntegratedPipeline
        
        np.random.seed(42)
        df = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'target': np.random.choice([0, 1], 100)
        })
        
        pipeline = IntegratedPipeline(
            target_col='target',
            task_type='classification',
            model_keys=['lr', 'nb'],
            n_splits=3,
            visualization=True,
            allow_disk_write=True
        )
        
        result = pipeline.run(df)
        
        assert result.visualization_paths is not None
        assert len(result.visualization_paths) > 0
        for p in result.visualization_paths.values():
            assert os.path.exists(p)
    
    def test_pipeline_without_visualization(self):
        from core.integrated_pipeline import IntegratedPipeline
        
        np.random.seed(42)
        df = pd.DataFrame({
            'a': np.random.randn(50),
            'target': np.random.choice([0, 1], 50)
        })
        
        pipeline = IntegratedPipeline(
            target_col='target',
            task_type='classification',
            model_keys=['lr'],
            n_splits=3,
            visualization=False
        )
        
        result = pipeline.run(df)
        assert result.visualization_paths is None


# =============================================================================
# 边界条件测试
# =============================================================================

@skip_if_no_mpl
class TestEdgeCases:
    
    def test_mpl_not_available(self, monkeypatch):
        """模拟 matplotlib 不可用时优雅降级"""
        import core.visualization as viz_mod
        # Mock _init_matplotlib 使其不设置 _MPL_AVAILABLE
        monkeypatch.setattr(viz_mod, '_MPL_AVAILABLE', False)
        monkeypatch.setattr(viz_mod, '_SNS_AVAILABLE', False)
        # Mock plt 为 None
        monkeypatch.setattr(viz_mod, '_init_matplotlib', lambda: False)
        
        dv = DataVisualizer()
        # _MPL_AVAILABLE 已经是 False，但 __init__ 会调用 _init_matplotlib（已被mock）
        # 所以 _MPL_AVAILABLE 保持 False
        fig = dv.plot_distribution(pd.DataFrame({'a': [1, 2]}), 'a')
        assert fig is None
    
    def test_no_permission_save(self, monkeypatch, sample_df, tmp_path):
        """磁盘写入禁用时不应报错"""
        wm = get_workspace_manager()
        old = wm.allow_disk_write
        wm.set_allow_disk_write(False)
        
        try:
            dv = DataVisualizer()
            fig = dv.plot_distribution(sample_df, 'num_a')
            # 应该返回 fig 但不保存
            assert fig is not None
        finally:
            wm.set_allow_disk_write(old)
