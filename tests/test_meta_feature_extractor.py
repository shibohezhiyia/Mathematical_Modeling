"""
MetaFeatureExtractor 单元测试

覆盖:
  - extract() 分类/回归数据
  - 特征类型统计 (numeric, categorical, datetime)
  - 缺失值、稀疏度、相关性
  - 类别分布、目标变量统计
  - _compute_complexity() 评分
  - MetaFeatures.to_dict()
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.meta_feature_extractor import MetaFeatures, MetaFeatureExtractor
from core.modeling_engine import TaskType


class TestExtractMetaFeaturesClassification(unittest.TestCase):
    """测试分类数据元特征提取"""

    def setUp(self):
        self.extractor = MetaFeatureExtractor()

    def test_basic_classification(self):
        X = pd.DataFrame({
            'num1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num2': [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        y = pd.Series([0, 1, 0, 1, 0])
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_samples, 5)
        self.assertEqual(meta.n_features, 2)
        self.assertEqual(meta.n_numeric, 2)
        self.assertEqual(meta.n_categorical, 0)
        self.assertEqual(meta.n_classes, 2)
        self.assertGreater(meta.target_entropy, 0)

    def test_multi_class(self):
        X = pd.DataFrame(np.random.randn(30, 4))
        y = pd.Series(np.random.choice([0, 1, 2], 30))
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_classes, 3)
        self.assertEqual(meta.n_samples, 30)

    def test_imbalanced_classes(self):
        X = pd.DataFrame(np.random.randn(100, 3))
        y = pd.Series([0] * 90 + [1] * 10)
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_classes, 2)
        self.assertAlmostEqual(meta.class_imbalance_ratio, 9.0, places=5)

    def test_balanced_classes(self):
        X = pd.DataFrame(np.random.randn(100, 3))
        y = pd.Series([0] * 50 + [1] * 50)
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(meta.class_imbalance_ratio, 1.0)


class TestExtractMetaFeaturesRegression(unittest.TestCase):
    """测试回归数据元特征提取"""

    def setUp(self):
        self.extractor = MetaFeatureExtractor()

    def test_basic_regression(self):
        X = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
        })
        y = pd.Series(np.random.randn(50))
        meta = self.extractor.extract(X, y, TaskType.REGRESSION)
        self.assertEqual(meta.n_samples, 50)
        self.assertEqual(meta.n_features, 2)
        self.assertEqual(meta.n_classes, 0)
        self.assertGreater(meta.target_std, 0)
        self.assertEqual(meta.class_imbalance_ratio, 1.0)

    def test_regression_constant_target(self):
        X = pd.DataFrame(np.random.randn(20, 3))
        y = pd.Series([5.0] * 20)
        meta = self.extractor.extract(X, y, TaskType.REGRESSION)
        self.assertEqual(meta.target_std, 0.0)


class TestFeatureTypeCounting(unittest.TestCase):
    """测试特征类型统计"""

    def setUp(self):
        self.extractor = MetaFeatureExtractor()

    def test_all_numeric(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4.0, 5.0, 6.0]})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_numeric, 2)
        self.assertEqual(meta.n_categorical, 0)
        self.assertEqual(meta.numeric_ratio, 1.0)
        self.assertEqual(meta.categorical_ratio, 0.0)

    def test_all_categorical(self):
        X = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_numeric, 0)
        self.assertEqual(meta.n_categorical, 2)
        self.assertEqual(meta.numeric_ratio, 0.0)
        self.assertEqual(meta.categorical_ratio, 1.0)

    def test_mixed_types(self):
        X = pd.DataFrame({
            'num1': [1, 2, 3],
            'cat1': ['a', 'b', 'c'],
            'num2': [4.0, 5.0, 6.0],
        })
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_numeric, 2)
        self.assertEqual(meta.n_categorical, 1)
        self.assertAlmostEqual(meta.numeric_ratio, 2 / 3, places=5)
        self.assertAlmostEqual(meta.categorical_ratio, 1 / 3, places=5)

    def test_datetime_treated_as_non_numeric(self):
        X = pd.DataFrame({
            'num': [1, 2, 3],
            'dt': pd.to_datetime(['2021-01-01', '2021-01-02', '2021-01-03']),
        })
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_numeric, 1)
        self.assertEqual(meta.n_categorical, 1)

    def test_sample_feature_ratio(self):
        X = pd.DataFrame(np.random.randn(100, 5))
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertAlmostEqual(meta.sample_feature_ratio, 20.0, places=5)


class TestMissingAndSparsity(unittest.TestCase):
    """测试缺失值与稀疏度"""

    def setUp(self):
        self.extractor = MetaFeatureExtractor()

    def test_no_missing(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.missing_ratio, 0.0)

    def test_with_missing(self):
        X = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4.0, 5.0, np.nan]})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertAlmostEqual(meta.missing_ratio, 2 / 6, places=5)

    def test_all_missing(self):
        X = pd.DataFrame({'a': [np.nan, np.nan], 'b': [np.nan, np.nan]})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.missing_ratio, 1.0)

    def test_no_zeros(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.sparsity, 0.0)

    def test_with_zeros(self):
        X = pd.DataFrame({'a': [0, 0, 3], 'b': [0, 5, 6]})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertAlmostEqual(meta.sparsity, 3 / 6, places=5)

    def test_zeros_only_in_numeric(self):
        X = pd.DataFrame({
            'num': [0, 1, 2],
            'cat': ['a', 'b', 'c'],
        })
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertAlmostEqual(meta.sparsity, 1 / 3, places=5)

    def test_empty_dataframe(self):
        X = pd.DataFrame({'a': pd.Series([], dtype=float)})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_samples, 0)
        self.assertEqual(meta.n_features, 1)
        self.assertEqual(meta.missing_ratio, 0.0)


class TestFeatureCorrelation(unittest.TestCase):
    """测试特征相关性提取"""

    def setUp(self):
        self.extractor = MetaFeatureExtractor()

    def test_single_numeric_no_correlation(self):
        X = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.feature_correlation_mean, 0.0)
        self.assertEqual(meta.feature_correlation_max, 0.0)

    def test_no_numeric_no_correlation(self):
        X = pd.DataFrame({'a': ['x', 'y', 'z'], 'b': ['p', 'q', 'r']})
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.feature_correlation_mean, 0.0)
        self.assertEqual(meta.feature_correlation_max, 0.0)

    def test_two_numeric_correlated(self):
        X = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [1.01, 2.01, 3.01, 4.01, 5.01],
        })
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertGreater(meta.feature_correlation_max, 0.99)
        self.assertGreater(meta.feature_correlation_mean, 0.99)

    def test_mixed_correlated_and_uncorrelated(self):
        np.random.seed(42)
        X = pd.DataFrame({
            'a': np.random.randn(100),
            'b': np.random.randn(100),
            'c': np.random.randn(100),
        })
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertGreaterEqual(meta.feature_correlation_max, 0.0)
        self.assertLess(meta.feature_correlation_max, 0.9)


class TestComplexityScore(unittest.TestCase):
    """测试复杂度评分"""

    def setUp(self):
        self.extractor = MetaFeatureExtractor()

    def test_small_data_low_complexity(self):
        X = pd.DataFrame(np.random.randn(50, 5))
        y = pd.Series(np.random.choice([0, 1], 50))
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertLess(meta.complexity_score, 50)

    def test_large_data_high_dim(self):
        X = pd.DataFrame(np.random.randn(2000, 80))
        y = pd.Series(np.random.choice([0, 1], 2000))
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertGreater(meta.complexity_score, 30)

    def test_missing_increases_complexity(self):
        X_clean = pd.DataFrame(np.random.randn(100, 5))
        X_missing = X_clean.copy()
        X_missing.iloc[0:10, 0] = np.nan
        y = pd.Series(np.random.choice([0, 1], 100))
        meta_clean = self.extractor.extract(X_clean, y, TaskType.CLASSIFICATION)
        meta_missing = self.extractor.extract(X_missing, y, TaskType.CLASSIFICATION)
        self.assertGreater(meta_missing.complexity_score, meta_clean.complexity_score)

    def test_imbalance_increases_complexity(self):
        X = pd.DataFrame(np.random.randn(100, 5))
        y_balanced = pd.Series([0] * 50 + [1] * 50)
        y_imbalanced = pd.Series([0] * 95 + [1] * 5)
        meta_balanced = self.extractor.extract(X, y_balanced, TaskType.CLASSIFICATION)
        meta_imbalanced = self.extractor.extract(X, y_imbalanced, TaskType.CLASSIFICATION)
        self.assertGreater(meta_imbalanced.complexity_score, meta_balanced.complexity_score)

    def test_high_correlation_increases_complexity(self):
        X_corr = pd.DataFrame({
            'a': [1, 2, 3, 4, 5] * 20,
            'b': [1.01, 2.01, 3.01, 4.01, 5.01] * 20,
        })
        y = pd.Series(np.random.choice([0, 1], 100))
        meta = self.extractor.extract(X_corr, y, TaskType.CLASSIFICATION)
        self.assertGreater(meta.complexity_score, 10)

    def test_sparsity_increases_complexity(self):
        X_dense = pd.DataFrame(np.random.randn(100, 5) + 5)
        # 稀疏矩阵：大部分为零，但保留少量非零值以避免相关系数全为 NaN
        X_sparse = pd.DataFrame(np.zeros((100, 5)))
        X_sparse.iloc[::10, :] = 1.0
        y = pd.Series(np.random.choice([0, 1], 100))
        meta_dense = self.extractor.extract(X_dense, y, TaskType.CLASSIFICATION)
        meta_sparse = self.extractor.extract(X_sparse, y, TaskType.CLASSIFICATION)
        self.assertGreater(meta_sparse.complexity_score, meta_dense.complexity_score)

    def test_complexity_capped_at_100(self):
        X = pd.DataFrame(np.random.randn(5000, 50))
        y = pd.Series(np.random.choice([0, 1], 5000))
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertLessEqual(meta.complexity_score, 100.0)


class TestMetaFeaturesToDict(unittest.TestCase):
    """测试 MetaFeatures.to_dict"""

    def test_to_dict_keys(self):
        meta = MetaFeatures(n_samples=100, n_features=5)
        d = meta.to_dict()
        expected_keys = [
            'n_samples', 'n_features', 'n_numeric', 'n_categorical',
            'sample_feature_ratio', 'numeric_ratio', 'categorical_ratio',
            'missing_ratio', 'sparsity', 'feature_correlation_mean',
            'feature_correlation_max', 'n_classes', 'class_imbalance_ratio',
            'target_entropy', 'target_std', 'complexity_score',
        ]
        for key in expected_keys:
            self.assertIn(key, d)

    def test_to_dict_values(self):
        meta = MetaFeatures(
            n_samples=100, n_features=5, missing_ratio=0.1234,
            complexity_score=45.678
        )
        d = meta.to_dict()
        self.assertEqual(d['n_samples'], 100)
        self.assertEqual(d['missing_ratio'], 0.1234)
        self.assertEqual(d['complexity_score'], 45.68)


class TestNoTarget(unittest.TestCase):
    """测试 y=None 的情况"""

    def setUp(self):
        self.extractor = MetaFeatureExtractor()

    def test_no_target_classification(self):
        X = pd.DataFrame(np.random.randn(20, 3))
        meta = self.extractor.extract(X, None, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_classes, 0)
        self.assertEqual(meta.target_entropy, 0.0)
        self.assertEqual(meta.target_std, 0.0)

    def test_no_target_regression(self):
        X = pd.DataFrame(np.random.randn(20, 3))
        meta = self.extractor.extract(X, None, TaskType.REGRESSION)
        self.assertEqual(meta.target_std, 0.0)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def setUp(self):
        self.extractor = MetaFeatureExtractor()

    def test_single_row(self):
        X = pd.DataFrame({'a': [1.0], 'b': [2.0]})
        y = pd.Series([0])
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_samples, 1)
        self.assertEqual(meta.n_features, 2)

    def test_single_feature(self):
        X = pd.DataFrame({'a': [1.0, 2.0, 3.0]})
        y = pd.Series([0, 1, 0])
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(meta.n_features, 1)
        self.assertEqual(meta.feature_correlation_mean, 0.0)
        self.assertEqual(meta.feature_correlation_max, 0.0)

    def test_all_same_numeric(self):
        X = pd.DataFrame({'a': [5.0, 5.0, 5.0], 'b': [5.0, 5.0, 5.0]})
        y = pd.Series([0, 1, 0])
        meta = self.extractor.extract(X, y, TaskType.CLASSIFICATION)
        self.assertTrue(np.isnan(meta.feature_correlation_mean) or meta.feature_correlation_mean == 0.0)


if __name__ == '__main__':
    unittest.main()
