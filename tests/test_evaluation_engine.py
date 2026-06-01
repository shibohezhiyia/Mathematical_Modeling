"""
模型自动评估与决策引擎测试

覆盖：
1. ModelEvaluator 多维度评估
2. AutoDecisionEngine 自动决策（各种模式）
3. 用户覆盖机制
4. 与 ModelingEngine 集成
5. 报告生成
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

from core.evaluation_engine import (
    ModelEvaluator, AutoDecisionEngine, DecisionReport,
    DecisionMode, RiskLevel, ModelScore,
    auto_select, print_decision_report
)
from core.modeling_engine import CVResult, TaskType


# =============================================================================
# 辅助函数：生成模拟 CVResult
# =============================================================================

def make_cv_result(model_key, model_name,
                   mean_f1=0.8, std_f1=0.02,
                   train_time=1.0, n_models=1):
    """生成模拟 CVResult"""
    r = CVResult(
        model_key=model_key,
        model_name=model_name,
        fold_scores={
            'f1_weighted': [mean_f1 - std_f1, mean_f1, mean_f1 + std_f1] * (n_models // 3 + 1)
        },
        mean_scores={'f1_weighted': mean_f1, 'accuracy': mean_f1},
        std_scores={'f1_weighted': std_f1, 'accuracy': std_f1},
        train_time=train_time,
        fitted_models=[MagicMock()] * n_models
    )
    return r


def make_regression_cv_result(model_key, model_name,
                              mean_rmse=0.5, std_rmse=0.05,
                              train_time=1.0):
    """生成模拟回归 CVResult"""
    r = CVResult(
        model_key=model_key,
        model_name=model_name,
        fold_scores={
            'rmse': [mean_rmse - std_rmse, mean_rmse, mean_rmse + std_rmse]
        },
        mean_scores={'rmse': mean_rmse, 'r2': 0.8},
        std_scores={'rmse': std_rmse, 'r2': 0.02},
        train_time=train_time,
        fitted_models=[MagicMock()]
    )
    return r


# =============================================================================
# ModelEvaluator 测试
# =============================================================================

class TestModelEvaluator:
    
    def test_evaluate_all_basic(self):
        """基础评估：多个模型应产生不同分数"""
        cv_results = [
            make_cv_result('lr', 'LogisticRegression', mean_f1=0.75, std_f1=0.03, train_time=0.5),
            make_cv_result('xgb', 'XGBoost', mean_f1=0.85, std_f1=0.02, train_time=2.0),
            make_cv_result('rf', 'RandomForest', mean_f1=0.82, std_f1=0.04, train_time=1.5),
        ]
        
        evaluator = ModelEvaluator()
        scores = evaluator.evaluate_all(cv_results, TaskType.CLASSIFICATION)
        
        assert len(scores) == 3
        # XGBoost 精度最高，应该排在前面
        assert scores[0].model_key == 'xgb'
        assert scores[0].accuracy_score > scores[1].accuracy_score
        
        # LogisticRegression 训练最快，速度分应该最高
        lr_score = next(s for s in scores if s.model_key == 'lr')
        assert lr_score.speed_score > 80  # 相对最快
    
    def test_empty_cv_results(self):
        """空输入应返回空列表"""
        evaluator = ModelEvaluator()
        scores = evaluator.evaluate_all([], TaskType.CLASSIFICATION)
        assert scores == []
    
    def test_regression_evaluation(self):
        """回归任务：RMSE越小越好"""
        cv_results = [
            make_regression_cv_result('lr', 'LinearRegression', mean_rmse=0.6, std_rmse=0.05),
            make_regression_cv_result('xgb', 'XGBoost', mean_rmse=0.4, std_rmse=0.03),
        ]
        
        evaluator = ModelEvaluator()
        scores = evaluator.evaluate_all(cv_results, TaskType.REGRESSION)
        
        # XGBoost RMSE更低，应该排在前面
        assert scores[0].model_key == 'xgb'
        # 原始 primary_score 应该显示正的 RMSE
        assert scores[0].primary_score == pytest.approx(0.4, abs=0.01)
        assert scores[1].primary_score == pytest.approx(0.6, abs=0.01)
    
    def test_risk_assessment(self):
        """风险等级判断"""
        # 高方差 = 过拟合风险
        high_var = make_cv_result('overfit', 'OverfitModel', mean_f1=0.9, std_f1=0.15, train_time=1.0)
        low_var = make_cv_result('stable', 'StableModel', mean_f1=0.85, std_f1=0.01, train_time=1.0)
        
        evaluator = ModelEvaluator()
        scores = evaluator.evaluate_all([high_var, low_var], TaskType.CLASSIFICATION)
        
        overfit = next(s for s in scores if s.model_key == 'overfit')
        stable = next(s for s in scores if s.model_key == 'stable')
        
        assert overfit.overfit_risk in (RiskLevel.HIGH, RiskLevel.MEDIUM)
        assert stable.overfit_risk == RiskLevel.LOW
    
    def test_stability_score(self):
        """稳定性分数：方差越小分越高"""
        cv_results = [
            make_cv_result('a', 'ModelA', mean_f1=0.8, std_f1=0.01),
            make_cv_result('b', 'ModelB', mean_f1=0.8, std_f1=0.10),
        ]
        
        evaluator = ModelEvaluator()
        scores = evaluator.evaluate_all(cv_results, TaskType.CLASSIFICATION)
        
        a = next(s for s in scores if s.model_key == 'a')
        b = next(s for s in scores if s.model_key == 'b')
        
        assert a.stability_score > b.stability_score


# =============================================================================
# AutoDecisionEngine 测试
# =============================================================================

class TestAutoDecisionEngine:
    
    def _make_scores(self):
        """生成测试用的 ModelScore 列表"""
        return [
            ModelScore(
                model_key='xgb', model_name='XGBoost',
                primary_metric='f1_weighted', primary_score=0.85, primary_std=0.02,
                train_time=2.0, n_parameters=10000,
                accuracy_score=90, speed_score=30, stability_score=85,
                simplicity_score=20, generalization_score=80,
                composite_score=61
            ),
            ModelScore(
                model_key='lr', model_name='LogisticRegression',
                primary_metric='f1_weighted', primary_score=0.75, primary_std=0.01,
                train_time=0.5, n_parameters=100,
                accuracy_score=50, speed_score=95, stability_score=95,
                simplicity_score=90, generalization_score=85,
                composite_score=83
            ),
            ModelScore(
                model_key='rf', model_name='RandomForest',
                primary_metric='f1_weighted', primary_score=0.82, primary_std=0.04,
                train_time=1.5, n_parameters=5000,
                accuracy_score=80, speed_score=50, stability_score=60,
                simplicity_score=50, generalization_score=70,
                composite_score=62
            ),
        ]
    
    def test_accuracy_first_mode(self):
        """精度优先模式应推荐精度最高的模型"""
        engine = AutoDecisionEngine(mode=DecisionMode.ACCURACY_FIRST)
        scores = self._make_scores()
        report = engine.decide(scores)
        
        assert report.recommended_model == 'xgb'
        assert report.confidence > 0
        assert '精度' in report.recommendation_reason or 'XGBoost' in report.recommendation_reason
    
    def test_speed_first_mode(self):
        """速度优先模式应推荐最快的模型"""
        engine = AutoDecisionEngine(mode=DecisionMode.SPEED_FIRST)
        scores = self._make_scores()
        report = engine.decide(scores)
        
        assert report.recommended_model == 'lr'
    
    def test_stability_first_mode(self):
        """稳定性优先模式应推荐最稳定的模型"""
        engine = AutoDecisionEngine(mode=DecisionMode.STABILITY_FIRST)
        scores = self._make_scores()
        report = engine.decide(scores)
        
        assert report.recommended_model == 'lr'
    
    def test_simplicity_first_mode(self):
        """简单优先模式应推荐最简单的模型"""
        engine = AutoDecisionEngine(mode=DecisionMode.SIMPLICITY_FIRST)
        scores = self._make_scores()
        report = engine.decide(scores)
        
        assert report.recommended_model == 'lr'
    
    def test_balanced_mode(self):
        """平衡模式应综合考虑"""
        engine = AutoDecisionEngine(mode=DecisionMode.BALANCED)
        scores = self._make_scores()
        report = engine.decide(scores)
        
        # LogisticRegression 综合得分最高 (83)
        assert report.recommended_model == 'lr'
    
    def test_user_override(self):
        """用户覆盖应优先于自动决策"""
        engine = AutoDecisionEngine(mode=DecisionMode.ACCURACY_FIRST)
        scores = self._make_scores()
        
        # 即使精度优先，用户指定 lr
        report = engine.decide(scores, user_override='lr')
        assert report.recommended_model == 'lr'
        assert report.confidence == 1.0
        assert '用户指定' in report.recommendation_reason
    
    def test_user_override_invalid(self):
        """用户覆盖无效模型时应回退到自动推荐"""
        engine = AutoDecisionEngine(mode=DecisionMode.ACCURACY_FIRST)
        scores = self._make_scores()
        
        report = engine.decide(scores, user_override='nonexistent')
        assert report.recommended_model == 'xgb'  # 自动推荐
        assert report.confidence < 1.0
    
    def test_comparison_table(self):
        """对比表应包含所有维度"""
        engine = AutoDecisionEngine(mode=DecisionMode.BALANCED)
        scores = self._make_scores()
        report = engine.decide(scores)
        
        assert report.comparison_table is not None
        assert len(report.comparison_table) == 3
        assert '模型' in report.comparison_table.columns
        assert '模式得分' in report.comparison_table.columns
    
    def test_risk_analysis(self):
        """风险分析应包含过拟合/欠拟合提示"""
        scores = [
            ModelScore(
                model_key='bad', model_name='BadModel',
                primary_metric='f1_weighted', primary_score=0.40, primary_std=0.15,
                train_time=120.0, n_parameters=100000,
                accuracy_score=10, speed_score=10, stability_score=10,
                simplicity_score=10, generalization_score=10,
                composite_score=10,
                overfit_risk=RiskLevel.HIGH,
                underfit_risk=RiskLevel.HIGH
            )
        ]
        engine = AutoDecisionEngine()
        report = engine.decide(scores)
        
        assert len(report.risks) > 0
        assert any('过拟合' in r for r in report.risks)
        assert any('欠拟合' in r for r in report.risks)
    
    def test_custom_mode(self):
        """自定义权重模式"""
        engine = AutoDecisionEngine(
            mode=DecisionMode.CUSTOM,
            custom_weights={'accuracy': 0.0, 'speed': 1.0, 'stability': 0.0, 'simplicity': 0.0, 'generalization': 0.0}
        )
        scores = self._make_scores()
        report = engine.decide(scores)
        
        # 只看速度，应该选 lr
        assert report.recommended_model == 'lr'
    
    def test_empty_scores(self):
        """空分数列表应返回空报告"""
        engine = AutoDecisionEngine()
        report = engine.decide([])
        assert report.recommended_model == ""
        assert report.confidence == 0.0


# =============================================================================
# auto_select 便捷函数测试
# =============================================================================

class TestAutoSelect:
    
    def test_auto_select_classification(self):
        """一键自动选择：分类任务（精度优先应推荐XGBoost）"""
        cv_results = [
            make_cv_result('lr', 'LogisticRegression', mean_f1=0.75, std_f1=0.01, train_time=0.5),
            make_cv_result('xgb', 'XGBoost', mean_f1=0.88, std_f1=0.02, train_time=2.0),
        ]
        
        # 精度优先模式
        report = auto_select(cv_results, 'classification', mode='accuracy_first')
        assert report.recommended_model == 'xgb'
        assert report.comparison_table is not None
        
        # 速度优先模式应推荐 lr
        report_speed = auto_select(cv_results, 'classification', mode='speed_first')
        assert report_speed.recommended_model == 'lr'
    
    def test_auto_select_regression(self):
        """一键自动选择：回归任务"""
        cv_results = [
            make_regression_cv_result('lr', 'LinearRegression', mean_rmse=0.6),
            make_regression_cv_result('xgb', 'XGBoost', mean_rmse=0.4),
        ]
        
        report = auto_select(cv_results, 'regression', mode='accuracy_first')
        
        # XGBoost RMSE 更低，更好
        assert report.recommended_model == 'xgb'
    
    def test_auto_select_with_override(self):
        """一键自动选择 + 用户覆盖"""
        cv_results = [
            make_cv_result('lr', 'LogisticRegression', mean_f1=0.75),
            make_cv_result('xgb', 'XGBoost', mean_f1=0.88),
        ]
        
        report = auto_select(cv_results, 'classification', mode='accuracy_first', user_override='lr')
        assert report.recommended_model == 'lr'


# =============================================================================
# 与 ModelingEngine 集成测试
# =============================================================================

class TestModelingEngineIntegration:
    
    def test_auto_decision_in_modeling_engine(self):
        """ModelingEngine 应自动产生决策报告"""
        from core.modeling_engine import ModelingEngine
        
        X = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.choice(['x', 'y', 'z'], 100)
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        engine = ModelingEngine(
            model_keys=['lr', 'nb'],  # 只跑少量模型加速测试
            auto_decision_mode='balanced',
            n_splits=3
        )
        result = engine.fit(X, y)
        
        assert result.decision_report is not None
        assert result.auto_recommended_model is not None
        assert result.best_model_key is not None
    
    def test_user_override_in_modeling_engine(self):
        """ModelingEngine 应支持用户覆盖自动推荐"""
        from core.modeling_engine import ModelingEngine
        
        X = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        engine = ModelingEngine(
            model_keys=['lr', 'nb'],
            auto_decision_mode='balanced',
            user_override_model='nb',
            n_splits=3
        )
        result = engine.fit(X, y)
        
        assert result.best_model_key == 'nb'
        assert result.decision_report is not None
        assert result.decision_report.recommended_model == 'nb'
    
    def test_accuracy_first_mode_in_engine(self):
        """ModelingEngine 精度优先模式"""
        from core.modeling_engine import ModelingEngine
        
        X = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        
        engine = ModelingEngine(
            model_keys=['lr', 'nb'],
            auto_decision_mode='accuracy_first',
            n_splits=3
        )
        result = engine.fit(X, y)
        
        assert result.decision_report is not None
        assert result.decision_report.mode == DecisionMode.ACCURACY_FIRST


# =============================================================================
# 报告打印测试
# =============================================================================

class TestReportPrinting:
    
    def test_print_decision_report(self, capsys):
        """打印决策报告不应报错"""
        scores = [
            ModelScore(
                model_key='xgb', model_name='XGBoost',
                primary_metric='f1_weighted', primary_score=0.85, primary_std=0.02,
                train_time=1.0, n_parameters=1000,
                accuracy_score=90, speed_score=50, stability_score=80,
                simplicity_score=40, generalization_score=75,
                composite_score=67
            )
        ]
        engine = AutoDecisionEngine(mode=DecisionMode.BALANCED)
        report = engine.decide(scores)
        
        print_decision_report(report)
        captured = capsys.readouterr()
        
        assert '自动评估与决策报告' in captured.out
        assert 'XGBoost' in captured.out
        assert '推荐模型' in captured.out


# =============================================================================
# DecisionMode 字符串解析测试
# =============================================================================

class TestDecisionModeParsing:
    
    def test_string_mode(self):
        """字符串模式应正确解析"""
        engine = AutoDecisionEngine(mode='speed_first')
        assert engine.mode == DecisionMode.SPEED_FIRST
    
    def test_enum_mode(self):
        """枚举模式应直接使用"""
        engine = AutoDecisionEngine(mode=DecisionMode.ACCURACY_FIRST)
        assert engine.mode == DecisionMode.ACCURACY_FIRST
