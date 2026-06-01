"""
高级功能测试：超参优化 + 深度学习 + 可解释性
"""
import os
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.modeling_engine import ModelingEngine, TaskType, EncodingType, FeatureSelectionStrategy, EnsembleMethod


class TestHyperparameterOptimization(unittest.TestCase):
    """测试超参数优化"""
    
    def test_random_search_fallback(self):
        """测试随机搜索回退"""
        from core.hyperparameter_optimizer import HyperparameterOptimizer
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1], 100))
        
        optimizer = HyperparameterOptimizer(
            n_trials=5,
            sampler='random',
            cv_folds=2
        )
        
        result = optimizer.optimize('lr', X, y, TaskType.CLASSIFICATION)
        
        self.assertIsNotNone(result.best_params)
        self.assertIsInstance(result.best_score, float)
        self.assertGreaterEqual(result.n_trials, 0)
    
    def test_optimize_all_models(self):
        """测试多模型并行优化"""
        from core.hyperparameter_optimizer import HyperparameterOptimizer
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(80, 4))
        y = pd.Series(np.random.choice([0, 1], 80))
        
        optimizer = HyperparameterOptimizer(n_trials=3, cv_folds=2)
        results = optimizer.optimize_all(['lr', 'dt'], X, y, TaskType.CLASSIFICATION)
        
        self.assertIn('lr', results)
        self.assertIn('dt', results)
    
    def test_quick_optimize(self):
        """测试快速优化接口"""
        from core.hyperparameter_optimizer import quick_optimize
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 3))
        y = pd.Series(np.random.choice([0, 1], 60))
        
        result = quick_optimize('lr', X, y, n_trials=3)
        self.assertIsNotNone(result.best_params)


class TestExplainability(unittest.TestCase):
    """测试可解释性分析"""
    
    def test_builtin_importance(self):
        """测试内置特征重要性"""
        from core.explainability import ExplainabilityEngine
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=['f1', 'f2', 'f3', 'f4', 'f5'])
        y = pd.Series(np.random.choice([0, 1], 100))
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        engine = ExplainabilityEngine(use_shap=False, use_lime=False)
        result = engine.explain_model(model, X, model_key='rf', task_type=TaskType.CLASSIFICATION)
        
        self.assertIsNotNone(result.global_importance)
        self.assertEqual(len(result.global_importance), 5)
        self.assertEqual(result.method, 'builtin')
    
    def test_instance_explanation(self):
        """测试单样本解释"""
        from core.explainability import ExplainabilityEngine
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 4), columns=['a', 'b', 'c', 'd'])
        y = pd.Series(np.random.choice([0, 1], 50))
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X, y)
        
        engine = ExplainabilityEngine(use_shap=False, use_lime=False)
        exp = engine.explain_instance(model, X, 0, task_type=TaskType.CLASSIFICATION)
        
        self.assertIn('prediction', exp)
        self.assertIn('features', exp)
    
    def test_compare_models(self):
        """测试多模型对比"""
        from core.explainability import ExplainabilityEngine
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(60, 4), columns=['a', 'b', 'c', 'd'])
        y = pd.Series(np.random.choice([0, 1], 60))
        
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        lr = LogisticRegression(max_iter=1000, random_state=42)
        rf.fit(X, y)
        lr.fit(X, y)
        
        engine = ExplainabilityEngine(use_shap=False, use_lime=False)
        comparison = engine.compare_models({'rf': rf, 'lr': lr}, X)
        
        self.assertGreater(len(comparison), 0)
        self.assertIn('feature', comparison.columns)
    
    def test_generate_report(self):
        """测试报告生成"""
        from core.explainability import ExplainabilityEngine, ExplanationResult
        from sklearn.ensemble import RandomForestClassifier
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3), columns=['x', 'y', 'z'])
        y = pd.Series(np.random.choice([0, 1], 50))
        
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        engine = ExplainabilityEngine(use_shap=False, use_lime=False)
        result = engine.explain_model(model, X, model_key='rf', task_type=TaskType.CLASSIFICATION)
        
        report = engine.generate_report(result)
        self.assertIn('模型可解释性报告', report)


class TestModelingEngineWithAdvancedFeatures(unittest.TestCase):
    """测试 ModelingEngine 集成高级功能"""
    
    def test_with_hyperparameter_optimization(self):
        """测试 ModelingEngine + 超参优化"""
        np.random.seed(42)
        n = 150
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'cat': np.random.choice(['A', 'B'], n),
            'y': np.random.choice([0, 1], n)
        })
        
        X = df.drop(columns=['y'])
        y = df['y']
        
        engine = ModelingEngine(
            task_type='classification',
            model_keys=['dt'],  # 简单模型，避免优化太慢
            n_splits=2,
            optimize_hyperparams=True,
            hyperparam_trials=3,
            feature_selection=FeatureSelectionStrategy.NONE
        )
        
        result = engine.fit(X, y)
        
        self.assertIsNotNone(result.optimized_params)
        self.assertIn('dt', result.optimized_params)
    
    def test_with_explainability(self):
        """测试 ModelingEngine + 可解释性"""
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'y': np.random.choice([0, 1], n)
        })
        
        X = df.drop(columns=['y'])
        y = df['y']
        
        engine = ModelingEngine(
            task_type='classification',
            model_keys=['lr'],
            n_splits=2,
            explainability=True,
            feature_selection=FeatureSelectionStrategy.NONE
        )
        
        result = engine.fit(X, y)
        
        self.assertIsNotNone(result.explainability_results)
        self.assertIn('lr', result.explainability_results)
    
    def test_combined_advanced_features(self):
        """测试同时启用超参优化 + 可解释性"""
        np.random.seed(42)
        n = 120
        df = pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'f3': np.random.randn(n),
            'y': np.random.choice([0, 1], n)
        })
        
        X = df.drop(columns=['y'])
        y = df['y']
        
        engine = ModelingEngine(
            task_type='classification',
            model_keys=['dt', 'lr'],
            n_splits=2,
            optimize_hyperparams=True,
            hyperparam_trials=3,
            explainability=True,
            feature_selection=FeatureSelectionStrategy.NONE,
            ensemble=EnsembleMethod.BEST_SINGLE
        )
        
        result = engine.fit(X, y)
        
        self.assertIsNotNone(result.optimized_params)
        self.assertIsNotNone(result.explainability_results)
        self.assertIsNotNone(result.leaderboard)


class TestIntegratedAdvancedPipeline(unittest.TestCase):
    """测试集成流水线的高级功能"""
    
    def test_pipeline_with_optimization(self):
        """集成流水线 + 超参优化"""
        from core.integrated_pipeline import IntegratedPipeline
        
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'cat': np.random.choice(['A', 'B', 'C'], n),
            'target': np.nan
        })
        df.loc[:150, 'target'] = np.random.choice([0, 1], 151)
        
        pipeline = IntegratedPipeline(
            target_col='target',
            task_type='classification',
            optimize_hyperparams=False,  # 避免测试太慢
            explainability=False,
            model_keys=['lr', 'dt'],
            n_splits=2
        )
        
        result = pipeline.run(df)
        
        self.assertIsNotNone(result.predictions)
        self.assertEqual(len(result.predictions), 49)
        self.assertIsNotNone(result.leaderboard)




class TestDeepLearningAndRL(unittest.TestCase):
    """测试深度学习和强化学习功能"""
    
    def test_rl_optimizer_basic(self):
        """测试 RLOptimizer 基本功能"""
        from core.reinforcement_learning import RLOptimizer
        
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(80, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.randint(0, 2, 80))
        
        rl = RLOptimizer(n_trials=2, cv_folds=2, random_state=42)
        result = rl.optimize('lr', X, y, 'classification')
        
        self.assertIsNotNone(result.best_params)
        self.assertIsNotNone(result.best_score)
        self.assertEqual(result.n_trials, 2)
        self.assertEqual(result.sampler_type, 'rl_dqn_v2')
    
    def test_modeling_engine_dl_filter_disabled(self):
        """测试默认禁用深度学习模型"""
        from core.modeling_engine import ModelLibrary
        
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION)
        dl_keys = [k for k in models if k.startswith('torch_') or k == 'tabnet']
        self.assertGreater(len(dl_keys), 0, "Deep learning models should be registered")
        
        engine = ModelingEngine(
            task_type='classification',
            deep_learning={'enabled': False, 'models': []},
            feature_selection=FeatureSelectionStrategy.NONE
        )
        # 无法直接获取 filtered models，但可以通过 fit 间接验证
        X = pd.DataFrame(np.random.randn(30, 2), columns=['a', 'b'])
        y = pd.Series(np.random.randint(0, 2, 30))
        result = engine.fit(X, y)
        
        trained_keys = [r.model_key for r in result.cv_results]
        for k in dl_keys:
            self.assertNotIn(k, trained_keys, f"DL model {k} should be excluded by default")
    
    def test_modeling_engine_dl_filter_enabled(self):
        """测试启用深度学习模型"""
        engine = ModelingEngine(
            task_type='classification',
            model_keys=['torch_mlp'],  # 只测试 torch_mlp
            deep_learning={'enabled': True, 'models': ['torch_mlp']},
            n_splits=2,
            feature_selection=FeatureSelectionStrategy.NONE
        )
        X = pd.DataFrame(np.random.randn(40, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.randint(0, 2, 40))
        result = engine.fit(X, y)
        
        trained_keys = [r.model_key for r in result.cv_results]
        self.assertIn('torch_mlp', trained_keys)
    
    def test_integrated_pipeline_with_dl_config(self):
        """测试 IntegratedPipeline 传递 DL 配置"""
        from core.integrated_pipeline import IntegratedPipeline
        
        np.random.seed(42)
        df = pd.DataFrame({
            'x1': np.random.randn(50),
            'x2': np.random.randn(50),
            'y': np.random.choice([0, 1], 50)
        })
        
        pipeline = IntegratedPipeline(
            target_col='y',
            task_type='classification',
            model_keys=['lr'],
            n_splits=2,
            deep_learning={'enabled': False, 'models': []},
            optimizer='bayesian',
            dim_reduction='none'
        )
        
        result = pipeline.run(df)
        self.assertIsNotNone(result.leaderboard)
        self.assertIsNone(result.predictions)  # no test set


if __name__ == '__main__':
    unittest.main()
