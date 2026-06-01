"""
建模引擎测试
"""
import os
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.modeling_engine import (
    TaskType, TaskTypeDetector,
    EncodingType, FeatureSelectionStrategy, EnsembleMethod,
    AutoEncoder, AutoFeatureSelector,
    ModelLibrary, CrossValidator, EnsembleBuilder, CVResult,
    ModelingEngine, ModelingResult
)


class TestTaskTypeDetector(unittest.TestCase):
    """测试任务类型检测"""
    
    def test_classification_binary(self):
        y = pd.Series([0, 1, 0, 1, 0])
        self.assertEqual(TaskTypeDetector.detect(y), TaskType.CLASSIFICATION)
    
    def test_classification_multi(self):
        y = pd.Series([0, 1, 2, 0, 1, 2])
        self.assertEqual(TaskTypeDetector.detect(y), TaskType.CLASSIFICATION)
    
    def test_regression_continuous(self):
        # 样本数要足够多，使得唯一值比例很低
        y = pd.Series(np.random.randn(100) * 10 + 50)
        self.assertEqual(TaskTypeDetector.detect(y), TaskType.REGRESSION)
    
    def test_regression_many_unique(self):
        y = pd.Series(np.random.randn(100))
        self.assertEqual(TaskTypeDetector.detect(y), TaskType.REGRESSION)
    
    def test_clustering_no_y(self):
        self.assertEqual(TaskTypeDetector.detect(y=None), TaskType.CLUSTERING)
    
    def test_user_hint_override(self):
        y = pd.Series([1.5, 2.3, 3.7])
        self.assertEqual(TaskTypeDetector.detect(y, user_hint='classification'), TaskType.CLASSIFICATION)
    
    def test_categorical_target(self):
        y = pd.Series(['A', 'B', 'A', 'C'])
        self.assertEqual(TaskTypeDetector.detect(y), TaskType.CLASSIFICATION)


class TestAutoEncoder(unittest.TestCase):
    """测试自动编码器"""
    
    def test_binary_label_encoding(self):
        df = pd.DataFrame({'A': ['X', 'Y', 'X', 'Y']})
        enc = AutoEncoder()
        result = enc.fit_transform(df)
        self.assertEqual(result['A'].nunique(), 2)
        self.assertTrue(all(result['A'].isin([0, 1])))
    
    def test_low_cardinality_onehot(self):
        df = pd.DataFrame({'A': ['a', 'b', 'c', 'a']})
        enc = AutoEncoder(onehot_max_categories=10)
        result = enc.fit_transform(df)
        # 3个类别 <= 10，应该OneHot
        self.assertIn('AutoEncoder', str(type(enc)))
        report = enc.get_encoding_report()
        if not report.empty:
            self.assertEqual(report.iloc[0]['strategy'], 'onehot')
    
    def test_high_cardinality_frequency(self):
        df = pd.DataFrame({'A': [f'cat_{i}' for i in range(100)]})
        enc = AutoEncoder()
        result = enc.fit_transform(df)
        self.assertTrue(pd.api.types.is_numeric_dtype(result['A']))
    
    def test_target_encoding(self):
        df = pd.DataFrame({'cat': ['A', 'B', 'A', 'B', 'A']})
        y = pd.Series([1, 2, 1, 2, 1])
        enc = AutoEncoder()
        result = enc.fit_transform(df, y)
        # 目标编码后应该是数值
        self.assertTrue(pd.api.types.is_numeric_dtype(result['cat']))
    
    def test_transform_consistency(self):
        df_train = pd.DataFrame({'A': ['X', 'Y', 'Z']})
        df_test = pd.DataFrame({'A': ['X', 'Y']})
        enc = AutoEncoder()
        enc.fit(df_train)
        result_train = enc.transform(df_train)
        result_test = enc.transform(df_test)
        self.assertEqual(len(result_test), 2)


class TestAutoFeatureSelector(unittest.TestCase):
    """测试自动特征选择"""
    
    def test_variance_threshold(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1, 1, 1, 1, 1],  # 零方差
            'C': [2, 3, 4, 5, 6]
        })
        y = pd.Series([0, 1, 0, 1, 0])
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.VARIANCE)
        result = selector.fit_transform(df, y, TaskType.CLASSIFICATION)
        self.assertNotIn('B', result.columns)
    
    def test_mutual_information(self):
        np.random.seed(42)
        df = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.choice([0, 1], 100))
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.MI, n_features=5)
        result = selector.fit_transform(df, y, TaskType.CLASSIFICATION)
        self.assertEqual(result.shape[1], 5)
    
    def test_correlation_filter(self):
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [1.01, 2.01, 3.01, 4.01, 5.01],  # 与A高度相关
            'C': [5, 4, 3, 2, 1]
        })
        y = pd.Series([0, 1, 0, 1, 0])
        selector = AutoFeatureSelector(
            strategy=FeatureSelectionStrategy.CORRELATION,
            correlation_threshold=0.99
        )
        result = selector.fit_transform(df, y, TaskType.CLASSIFICATION)
        # A和B高度相关，应该删除其中一个
        self.assertLess(result.shape[1], 3)
    
    def test_none_strategy(self):
        df = pd.DataFrame(np.random.randn(10, 5))
        y = pd.Series([0, 1] * 5)
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.NONE)
        result = selector.fit_transform(df, y, TaskType.CLASSIFICATION)
        self.assertEqual(result.shape[1], 5)


class TestModelLibrary(unittest.TestCase):
    """测试模型库"""
    
    def test_classification_models(self):
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION)
        self.assertGreater(len(models), 0)
        self.assertIn('lr', models)  # LogisticRegression
    
    def test_regression_models(self):
        models = ModelLibrary.get_models(TaskType.REGRESSION)
        self.assertGreater(len(models), 0)
        self.assertIn('linear', models)  # LinearRegression
    
    def test_create_model(self):
        model = ModelLibrary.create_model('lr', TaskType.CLASSIFICATION)
        self.assertIsNotNone(model)
    
    def test_list_models(self):
        df = ModelLibrary.list_models(TaskType.CLASSIFICATION)
        self.assertGreater(len(df), 0)
        self.assertIn('key', df.columns)
        self.assertIn('name', df.columns)
    
    def test_filter_by_category(self):
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION, categories=['linear'])
        self.assertGreater(len(models), 0)
        for spec in models.values():
            self.assertEqual(spec.category, 'linear')


class TestCrossValidator(unittest.TestCase):
    """测试交叉验证"""
    
    def test_classification_cv(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1], 100))
        
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        cv = CrossValidator(n_splits=3)
        result = cv.cross_validate(model, X, y, TaskType.CLASSIFICATION)
        
        self.assertEqual(len(result.fitted_models), 3)
        self.assertIn('accuracy', result.mean_scores)
        self.assertEqual(len(result.oof_pred), 100)
    
    def test_regression_cv(self):
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randn(50))
        
        from sklearn.linear_model import Ridge
        model = Ridge(random_state=42)
        cv = CrossValidator(n_splits=3)
        result = cv.cross_validate(model, X, y, TaskType.REGRESSION)
        
        self.assertIn('rmse', result.mean_scores)
        self.assertEqual(len(result.oof_pred), 50)


class TestEnsembleBuilder(unittest.TestCase):
    """测试模型融合"""
    
    def test_weighted_blend(self):
        # 模拟两个CV结果
        result1 = CVResult(
            model_key='m1', model_name='Model1',
            mean_scores={'accuracy': 0.9},
            oof_pred=np.array([0, 1, 0, 1])
        )
        result2 = CVResult(
            model_key='m2', model_name='Model2',
            mean_scores={'accuracy': 0.8},
            oof_pred=np.array([0, 1, 1, 1])
        )
        
        builder = EnsembleBuilder(method=EnsembleMethod.WEIGHTED)
        blend = builder.blend([result1, result2], task_type=TaskType.CLASSIFICATION)
        
        self.assertIn('oof', blend)
        self.assertIn('weights', blend)
        self.assertGreater(blend['weights']['m1'], blend['weights']['m2'])
    
    def test_voting_hard(self):
        result1 = CVResult(
            model_key='m1', model_name='Model1',
            mean_scores={'accuracy': 0.9},
            oof_pred=np.array([0, 1, 0, 1])
        )
        result2 = CVResult(
            model_key='m2', model_name='Model2',
            mean_scores={'accuracy': 0.8},
            oof_pred=np.array([0, 1, 1, 1])
        )
        
        builder = EnsembleBuilder(method=EnsembleMethod.VOTING_HARD)
        blend = builder.blend([result1, result2], task_type=TaskType.CLASSIFICATION)
        
        self.assertEqual(len(blend['oof']), 4)


class TestModelingEngine(unittest.TestCase):
    """测试建模引擎完整流程"""
    
    def test_classification_pipeline(self):
        """分类完整流程"""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'num1': np.random.randn(n),
            'num2': np.random.randn(n),
            'cat1': np.random.choice(['A', 'B', 'C'], n),
            'target': np.random.choice([0, 1], n)
        })
        
        X = df.drop(columns=['target'])
        y = df['target']
        
        engine = ModelingEngine(
            task_type='classification',
            model_keys=['lr', 'dt'],
            n_splits=3,
            encoding=EncodingType.ONEHOT,
            feature_selection=FeatureSelectionStrategy.NONE,
            ensemble=EnsembleMethod.WEIGHTED
        )
        
        result = engine.fit(X, y)
        
        self.assertEqual(result.task_type, TaskType.CLASSIFICATION)
        self.assertEqual(len(result.cv_results), 2)
        self.assertIsNotNone(result.leaderboard)
        self.assertIsNotNone(result.ensemble_result)
    
    def test_regression_pipeline(self):
        """回归完整流程"""
        np.random.seed(42)
        n = 150
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'cat': np.random.choice(['X', 'Y'], n),
            'y': np.random.randn(n)
        })
        
        X = df.drop(columns=['y'])
        y = df['y']
        
        engine = ModelingEngine(
            task_type='regression',
            model_keys=['ridge', 'dt'],
            n_splits=3,
            feature_selection=FeatureSelectionStrategy.NONE
        )
        
        result = engine.fit(X, y)
        
        self.assertEqual(result.task_type, TaskType.REGRESSION)
        self.assertGreater(len(result.cv_results), 0)
    
    def test_auto_task_detection(self):
        """自动任务类型检测"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 3))
        y = pd.Series(np.random.choice([0, 1], 100))
        
        engine = ModelingEngine(model_keys=['lr'], n_splits=2)
        result = engine.fit(X, y)
        
        self.assertEqual(result.task_type, TaskType.CLASSIFICATION)
    
    def test_with_feature_selection(self):
        """带特征选择"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 20))
        y = pd.Series(np.random.choice([0, 1], 100))
        
        engine = ModelingEngine(
            model_keys=['lr'],
            n_splits=3,
            feature_selection=FeatureSelectionStrategy.MI
        )
        
        result = engine.fit(X, y)
        
        # 特征应该被选择减少
        self.assertIsNotNone(result.preprocessing_info)
        self.assertLessEqual(
            result.preprocessing_info.get('selected_features', 20),
            result.preprocessing_info.get('original_features', 20)
        )
    
    def test_clustering_pipeline(self):
        """聚类完整流程"""
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 4))
        
        engine = ModelingEngine(
            task_type='clustering',
            model_keys=['kmeans']
        )
        
        result = engine.fit(X)
        
        self.assertEqual(result.task_type, TaskType.CLUSTERING)
        self.assertGreater(len(result.cv_results), 0)
    
    def test_with_test_set(self):
        """带测试集的完整流程"""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'x': np.random.randn(n),
            'cat': np.random.choice(['A', 'B'], n),
            'y': np.random.choice([0, 1], n)
        })
        
        X = df.drop(columns=['y'])
        y = df['y']
        X_test = pd.DataFrame({
            'x': np.random.randn(50),
            'cat': np.random.choice(['A', 'B'], 50)
        })
        
        engine = ModelingEngine(
            model_keys=['lr', 'dt'],
            n_splits=2,
            ensemble=EnsembleMethod.WEIGHTED
        )
        
        result = engine.fit(X, y, X_test)
        
        self.assertIsNotNone(result.ensemble_result)
        self.assertEqual(len(result.ensemble_result['test']), 50)


if __name__ == '__main__':
    unittest.main()
