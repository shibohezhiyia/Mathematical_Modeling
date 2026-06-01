"""
ModelingEngine 扩展测试 - 提升覆盖率
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch
from dataclasses import field, dataclass
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.modeling_engine import (
    AutoEncoder, AutoFeatureSelector, CrossValidator, EnsembleBuilder,
    ModelLibrary, ModelingEngine, TaskType, TaskTypeDetector,
    EncodingType, FeatureSelectionStrategy, EnsembleMethod, CVResult
)


class TestAutoEncoderExtended(unittest.TestCase):
    """测试编码器扩展场景"""
    
    def setUp(self):
        self.df = pd.DataFrame({
            'num': [1.0, 2.0, 3.0, 4.0, 5.0],
            'cat': ['A', 'B', 'A', 'C', 'B'],
            'target': [0, 1, 0, 1, 0]
        })
    
    def test_fit_transform_basic(self):
        enc = AutoEncoder(onehot_max_categories=10)
        result = enc.fit_transform(self.df[['cat', 'num']], self.df['target'])
        self.assertGreater(len(result.columns), 0)
        self.assertEqual(len(result), 5)
    
    def test_transform_consistency(self):
        enc = AutoEncoder()
        train = self.df[['cat']]
        enc.fit(train, self.df['target'])
        result1 = enc.transform(train)
        result2 = enc.transform(train)
        pd.testing.assert_frame_equal(result1, result2)
    
    def test_unknown_category_transform(self):
        enc = AutoEncoder()
        train = pd.DataFrame({'cat': ['A', 'B']})
        enc.fit(train, pd.Series([0, 1]))
        test = pd.DataFrame({'cat': ['A', 'C']})
        result = enc.transform(test)
        self.assertEqual(len(result), 2)
    
    def test_transform_without_fit(self):
        enc = AutoEncoder()
        with self.assertRaises(Exception):
            enc.transform(self.df)
    
    def test_high_cardinality(self):
        df = pd.DataFrame({'cat': [f'val_{i}' for i in range(100)]})
        y = pd.Series([0] * 100)
        enc = AutoEncoder(onehot_max_categories=5)
        result = enc.fit_transform(df, y)
        self.assertEqual(len(result.columns), 1)
    
    def test_no_categorical_columns(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        enc = AutoEncoder()
        result = enc.fit_transform(df, pd.Series([0, 1, 0]))
        self.assertEqual(list(result.columns), ['a', 'b'])
    
    def test_binary_category(self):
        df = pd.DataFrame({'cat': ['A', 'B'] * 20})
        enc = AutoEncoder()
        result = enc.fit_transform(df, pd.Series([0] * 40))
        self.assertGreater(len(result.columns), 0)


class TestAutoFeatureSelectorExtended(unittest.TestCase):
    """测试特征选择器扩展场景"""
    
    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(100, 10), columns=[f'f{i}' for i in range(10)])
        self.y = pd.Series(np.random.randint(0, 2, 100))
    
    def test_variance_threshold(self):
        sel = AutoFeatureSelector(strategy=FeatureSelectionStrategy.VARIANCE)
        result = sel.fit_transform(self.X, self.y, TaskType.CLASSIFICATION)
        self.assertLessEqual(result.shape[1], self.X.shape[1])
    
    def test_correlation_strategy(self):
        sel = AutoFeatureSelector(strategy=FeatureSelectionStrategy.CORRELATION)
        result = sel.fit_transform(self.X, self.y, TaskType.CLASSIFICATION)
        self.assertLessEqual(result.shape[1], self.X.shape[1])
    
    def test_rfe_fallback(self):
        # RFE strategy is not fully implemented, should fallback gracefully
        sel = AutoFeatureSelector(strategy=FeatureSelectionStrategy.RFE, n_features=5)
        result = sel.fit_transform(self.X, self.y, TaskType.CLASSIFICATION)
        # Should return data without error (likely falls through to no selection)
        self.assertEqual(len(result), 100)
    
    def test_model_based_strategy(self):
        sel = AutoFeatureSelector(strategy=FeatureSelectionStrategy.MODEL_BASED)
        result = sel.fit_transform(self.X, self.y, TaskType.CLASSIFICATION)
        self.assertLessEqual(result.shape[1], self.X.shape[1])
    
    def test_pca_strategy(self):
        sel = AutoFeatureSelector(strategy=FeatureSelectionStrategy.PCA_DIM, n_features=5)
        result = sel.fit_transform(self.X, self.y, TaskType.CLASSIFICATION)
        self.assertEqual(result.shape[1], 5)
    
    def test_none_strategy(self):
        sel = AutoFeatureSelector(strategy=FeatureSelectionStrategy.NONE)
        result = sel.fit_transform(self.X, self.y, TaskType.CLASSIFICATION)
        self.assertEqual(result.shape, self.X.shape)
    
    def test_regression_task(self):
        y_reg = pd.Series(np.random.randn(100))
        sel = AutoFeatureSelector(strategy=FeatureSelectionStrategy.MI)
        result = sel.fit_transform(self.X, y_reg, TaskType.REGRESSION)
        self.assertLessEqual(result.shape[1], self.X.shape[1])


class TestCrossValidatorExtended(unittest.TestCase):
    """测试交叉验证器扩展场景"""
    
    def test_classification_cv(self):
        cv = CrossValidator(n_splits=3)
        X = pd.DataFrame(np.random.randn(30, 3))
        y = pd.Series([0, 1] * 15)
        from sklearn.dummy import DummyClassifier
        model = DummyClassifier(strategy='most_frequent')
        result = cv.cross_validate(model, X, y, task_type=TaskType.CLASSIFICATION)
        self.assertIsInstance(result.mean_scores, dict)
        self.assertIn('accuracy', result.mean_scores)
    
    def test_regression_cv(self):
        cv = CrossValidator(n_splits=3)
        X = pd.DataFrame(np.random.randn(30, 3))
        y = pd.Series(np.random.randn(30))
        from sklearn.dummy import DummyRegressor
        model = DummyRegressor()
        result = cv.cross_validate(model, X, y, task_type=TaskType.REGRESSION)
        self.assertIsInstance(result.mean_scores, dict)
        self.assertIn('rmse', result.mean_scores)


class TestEnsembleBuilderExtended(unittest.TestCase):
    """测试集成构建器扩展场景"""
    
    def _make_cv_result(self, key, oof):
        return CVResult(
            model_key=key,
            model_name=key,
            mean_scores={'accuracy': 0.8, 'f1_weighted': 0.79},
            oof_pred=np.array(oof),
        )
    
    def test_voting_hard(self):
        builder = EnsembleBuilder(method=EnsembleMethod.VOTING_HARD)
        cv_results = [
            self._make_cv_result('m1', [0, 1, 0, 1]),
            self._make_cv_result('m2', [0, 1, 1, 1]),
        ]
        result = builder.blend(cv_results, task_type=TaskType.CLASSIFICATION)
        self.assertIn('oof', result)
        self.assertIn('weights', result)
    
    def test_voting_soft(self):
        builder = EnsembleBuilder(method=EnsembleMethod.VOTING_SOFT)
        cv_results = [
            self._make_cv_result('m1', [0.2, 0.8, 0.3, 0.9]),
            self._make_cv_result('m2', [0.3, 0.7, 0.4, 0.8]),
        ]
        result = builder.blend(cv_results, task_type=TaskType.CLASSIFICATION)
        self.assertIn('oof', result)
    
    def test_weighted_blend(self):
        builder = EnsembleBuilder(method=EnsembleMethod.WEIGHTED)
        cv_results = [
            self._make_cv_result('m1', [1.0, 2.0, 3.0]),
            self._make_cv_result('m2', [1.1, 2.1, 2.9]),
        ]
        result = builder.blend(cv_results, task_type=TaskType.REGRESSION)
        self.assertIn('oof', result)
    
    def test_stacking(self):
        builder = EnsembleBuilder(method=EnsembleMethod.STACKING)
        cv_results = [
            self._make_cv_result('m1', [0.2, 0.8, 0.3, 0.9]),
            self._make_cv_result('m2', [0.3, 0.7, 0.4, 0.8]),
        ]
        result = builder.blend(cv_results, task_type=TaskType.CLASSIFICATION)
        self.assertIn('oof', result)
    
    def test_best_single(self):
        builder = EnsembleBuilder(method=EnsembleMethod.BEST_SINGLE)
        cv_results = [self._make_cv_result('m1', [0, 1, 0, 1])]
        result = builder.blend(cv_results, task_type=TaskType.CLASSIFICATION)
        self.assertIn('oof', result)
        self.assertEqual(result['weights']['m1'], 1.0)


class TestModelLibraryExtended(unittest.TestCase):
    """测试模型库扩展场景"""
    
    def test_list_models_returns_df(self):
        df = ModelLibrary.list_models(TaskType.CLASSIFICATION)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
    
    def test_get_models_empty(self):
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION, ['nonexistent'])
        self.assertEqual(len(models), 0)
    
    def test_create_model_invalid_key(self):
        with self.assertRaises(ValueError):
            ModelLibrary.create_model('nonexistent', TaskType.CLASSIFICATION)
    
    def test_get_models_with_categories(self):
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION, categories=['tree'])
        self.assertGreater(len(models), 0)
    
    def test_create_model_override_params(self):
        model = ModelLibrary.create_model('lr', TaskType.CLASSIFICATION, C=0.5)
        self.assertEqual(model.C, 0.5)


class TestModelingEngineExtended(unittest.TestCase):
    """测试建模引擎扩展场景"""
    
    def setUp(self):
        np.random.seed(42)
        self.df = pd.DataFrame({
            'a': np.random.randn(60),
            'b': np.random.randn(60),
            'c': np.random.choice(['X', 'Y', 'Z'], 60),
            'target': np.random.randint(0, 2, 60)
        })
        self.X = self.df.drop(columns=['target'])
        self.y = self.df['target']
    
    def test_with_label_encoding(self):
        engine = ModelingEngine(
            encoding=EncodingType.LABEL, n_splits=3, auto_sample=False
        )
        result = engine.fit(self.X, self.y)
        self.assertIsNotNone(result.leaderboard)
    
    def test_with_target_encoding(self):
        engine = ModelingEngine(
            encoding=EncodingType.TARGET, n_splits=3, auto_sample=False
        )
        result = engine.fit(self.X, self.y)
        self.assertIsNotNone(result.leaderboard)
    
    def test_with_pca_feature_selection(self):
        engine = ModelingEngine(
            feature_selection=FeatureSelectionStrategy.PCA_DIM, n_splits=3, auto_sample=False
        )
        result = engine.fit(self.X, self.y)
        self.assertIsNotNone(result.leaderboard)
    
    def test_with_voting_hard_ensemble(self):
        engine = ModelingEngine(
            ensemble=EnsembleMethod.VOTING_HARD, n_splits=3, auto_sample=False
        )
        result = engine.fit(self.X, self.y)
        self.assertIsNotNone(result.leaderboard)
    
    def test_with_stacking_ensemble(self):
        engine = ModelingEngine(
            ensemble=EnsembleMethod.STACKING, n_splits=3, auto_sample=False
        )
        result = engine.fit(self.X, self.y)
        self.assertIsNotNone(result.leaderboard)
    
    def test_clustering_task(self):
        engine = ModelingEngine(
            task_type='clustering', n_splits=3, auto_sample=False
        )
        result = engine.fit(self.X)
        self.assertIsNotNone(result.leaderboard)
    
    def test_with_test_set(self):
        X_train = self.X.iloc[:40]
        y_train = self.y.iloc[:40]
        X_test = self.X.iloc[40:]
        engine = ModelingEngine(n_splits=3, auto_sample=False)
        result = engine.fit(X_train, y_train, X_test)
        self.assertIsNotNone(result.leaderboard)
    
    def test_print_report(self):
        engine = ModelingEngine(n_splits=3, auto_sample=False)
        result = engine.fit(self.X, self.y)
        engine.print_report()  # Should not raise
    
    def test_user_override_model(self):
        engine = ModelingEngine(
            user_override_model='lr',
            n_splits=3, auto_sample=False
        )
        result = engine.fit(self.X, self.y)
        self.assertIsNotNone(result.leaderboard)
    
    def test_single_class(self):
        engine = ModelingEngine(n_splits=3, auto_sample=False)
        y = pd.Series([0] * 60)
        result = engine.fit(self.X, y)
        self.assertIsNotNone(result)
    
    def test_regression_task(self):
        engine = ModelingEngine(task_type='regression', n_splits=3, auto_sample=False)
        y = pd.Series(np.random.randn(60))
        result = engine.fit(self.X, y)
        self.assertIsNotNone(result.leaderboard)


if __name__ == '__main__':
    unittest.main()
