"""
代码质量与健壮性测试

覆盖边缘场景和异常处理：
- TaskTypeDetector 边缘场景
- AutoEncoder 未知类别行为
- AutoFeatureSelector 异常情况
- ModelLibrary create_model / list_models / get_models
- CrossValidator 极端类别不平衡
"""
import unittest
from unittest.mock import patch, MagicMock
import warnings

import numpy as np
import pandas as pd

from core.modeling_engine import (
    TaskTypeDetector, TaskType,
    AutoEncoder, EncodingType,
    AutoFeatureSelector, FeatureSelectionStrategy,
    ModelLibrary, ModelSpec,
    CrossValidator, CVResult
)


# =============================================================================
# TaskTypeDetector
# =============================================================================

class TestTaskTypeDetectorEdgeCases(unittest.TestCase):
    """TaskTypeDetector 边缘场景测试"""
    
    def test_empty_y_returns_unknown(self):
        """空标签应返回 UNKNOWN"""
        X = pd.DataFrame(np.random.randn(10, 3))
        y = pd.Series([], dtype=float)
        result = TaskTypeDetector.detect(y, X)
        self.assertEqual(result, TaskType.UNKNOWN)
    
    def test_all_nan_y_returns_unknown(self):
        """全 NaN 标签应返回 UNKNOWN"""
        X = pd.DataFrame(np.random.randn(10, 3))
        y = pd.Series([np.nan] * 10)
        result = TaskTypeDetector.detect(y, X)
        self.assertEqual(result, TaskType.UNKNOWN)
    
    def test_unique_ratio_path_classification(self):
        """unique_ratio < 0.05 且 n_unique <= 100 应判定为分类"""
        X = pd.DataFrame(np.random.randn(1000, 3))
        # 1000 样本，20 个唯一值 → unique_ratio=0.02
        y = pd.Series(np.random.choice(range(20), 1000))
        result = TaskTypeDetector.detect(y, X)
        self.assertEqual(result, TaskType.CLASSIFICATION)
    
    def test_unique_ratio_path_regression(self):
        """unique_ratio >= 0.05 应判定为回归"""
        X = pd.DataFrame(np.random.randn(100, 3))
        # 100 样本，50 个唯一值 → unique_ratio=0.5
        y = pd.Series(np.random.choice(range(50), 100))
        result = TaskTypeDetector.detect(y, X)
        self.assertEqual(result, TaskType.REGRESSION)
    
    def test_categorical_target_classification(self):
        """字符串目标应判定为分类"""
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.choice(['A', 'B', 'C'], 50))
        result = TaskTypeDetector.detect(y, X)
        self.assertEqual(result, TaskType.CLASSIFICATION)
    
    def test_user_hint_override(self):
        """用户提示应覆盖自动判断"""
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randn(50))  # 连续值，默认回归
        result = TaskTypeDetector.detect(y, X, 'classification')
        self.assertEqual(result, TaskType.CLASSIFICATION)
    
    def test_invalid_user_hint(self):
        """无效用户提示应回退到自动判断"""
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))
        result = TaskTypeDetector.detect(y, X, 'invalid_hint')
        self.assertEqual(result, TaskType.CLASSIFICATION)
    
    def test_get_metrics_dict_clustering(self):
        """聚类任务指标字典"""
        metrics = TaskTypeDetector.get_metrics_dict(TaskType.CLUSTERING)
        self.assertIn('silhouette', metrics)
    
    def test_get_primary_metric_clustering(self):
        """聚类任务主指标"""
        metric = TaskTypeDetector.get_primary_metric(TaskType.CLUSTERING)
        self.assertEqual(metric, 'silhouette')


# =============================================================================
# AutoEncoder
# =============================================================================

class TestAutoEncoderUnknownCategories(unittest.TestCase):
    """AutoEncoder 未知类别行为测试"""
    
    def setUp(self):
        self.encoder = AutoEncoder(onehot_max_categories=10)
    
    def test_label_encoding_unknown_category(self):
        """LABEL 编码应处理未知类别"""
        X_train = pd.DataFrame({'cat': ['A', 'B', 'A', 'B']})
        y = pd.Series([0, 1, 0, 1])
        self.encoder.fit(X_train, y)
        
        X_test = pd.DataFrame({'cat': ['A', 'B', 'C']})  # C 是未知类别
        result = self.encoder.transform(X_test)
        # 未知类别应映射为 -1
        self.assertEqual(result['cat'].iloc[2], -1)
    
    def test_ordinal_encoding_unknown_category(self):
        """ORDINAL 编码应处理未知类别"""
        X_train = pd.DataFrame({'cat': ['A', 'B', 'C', 'D', 'E', 'A']})  # 6 unique -> ordinal (>10 onehot_max? no, 6<=10 -> onehot. Need >10)
        # Use more categories to trigger ordinal encoding
        X_train = pd.DataFrame({'cat': list('ABCDEFGHIJKLMNO') * 2})  # 15 unique
        y = pd.Series([0, 1] * 15)
        self.encoder.fit(X_train, y)
        
        X_test = pd.DataFrame({'cat': ['A', 'B', 'Z']})  # Z 是未知类别
        result = self.encoder.transform(X_test)
        # 未知类别应映射为 -1 (OrdinalEncoder handle_unknown='use_encoded_value')
        self.assertEqual(result['cat'].iloc[2], -1)
    
    def test_target_encoding_unknown_category(self):
        """TARGET 编码应处理未知类别"""
        # Use high cardinality to trigger target encoding
        X_train = pd.DataFrame({'cat': [f'val_{i}' for i in range(60)] * 2})
        y = pd.Series([1.0, 2.0] * 60)
        self.encoder.fit(X_train, y)
        
        X_test = pd.DataFrame({'cat': ['val_0', 'val_1', 'UNKNOWN']})  # UNKNOWN 是未知类别
        result = self.encoder.transform(X_test)
        # 未知类别应填充全局均值
        self.assertFalse(pd.isna(result['cat'].iloc[2]))
    
    def test_onehot_encoding_unknown_category(self):
        """ONEHOT 编码应处理未知类别（忽略）"""
        # 使用 3 个类别触发 ONEHOT 编码 (n_unique=3 > 2, <= 10)
        X_train = pd.DataFrame({'cat': ['A', 'B', 'C', 'A', 'B', 'C']})
        y = pd.Series([0, 1, 0, 1, 0, 1])
        self.encoder.fit(X_train, y)
        
        X_test = pd.DataFrame({'cat': ['A', 'B', 'Z']})  # Z 是未知类别
        result = self.encoder.transform(X_test)
        # OneHotEncoder handle_unknown='ignore'，应输出全 0 行
        # 结果应有 3 列 (cat_A, cat_B, cat_C)
        self.assertEqual(len(result.columns), 3)
        # 第三行 (Z) 应全为 0
        self.assertEqual(result.iloc[2].sum(), 0)
    
    def test_transform_without_fit_raises(self):
        """未 fit 就 transform 应抛出异常"""
        encoder = AutoEncoder()
        X = pd.DataFrame({'a': [1, 2, 3]})
        with self.assertRaises(ValueError):
            encoder.transform(X)
    
    def test_empty_dataframe(self):
        """空 DataFrame 应能处理"""
        X = pd.DataFrame()
        y = pd.Series([0, 1])
        self.encoder.fit(X, y)
        result = self.encoder.transform(pd.DataFrame())
        self.assertTrue(result.empty)


# =============================================================================
# AutoFeatureSelector
# =============================================================================

class TestAutoFeatureSelectorEdgeCases(unittest.TestCase):
    """AutoFeatureSelector 异常情况测试"""
    
    def setUp(self):
        np.random.seed(42)
    
    def test_high_class_ratio_fallback(self):
        """类别数接近样本数时应回退到 VarianceThreshold"""
        # 10 样本，8 个类别 → n_classes/n_samples=0.8 > 0.5
        X = pd.DataFrame(np.random.randn(10, 5))
        y = pd.Series(range(10))  # 每个样本一个类别
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.MI)
        result = selector.fit_transform(X, y, TaskType.CLASSIFICATION)
        # 不应抛出异常，应返回结果
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_single_class_fallback(self):
        """单类别应能处理"""
        X = pd.DataFrame(np.random.randn(20, 5))
        y = pd.Series([0] * 20)
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.MI)
        result = selector.fit_transform(X, y, TaskType.CLASSIFICATION)
        self.assertIsInstance(result, pd.DataFrame)
    
    def test_pca_strategy(self):
        """PCA 策略应能降维"""
        X = pd.DataFrame(np.random.randn(100, 10))
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.PCA_DIM, n_features=3)
        result = selector.fit_transform(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(result.shape[1], 3)
    
    def test_correlation_strategy(self):
        """CORRELATION 策略应能处理"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.CORRELATION, n_features=3)
        result = selector.fit_transform(X, y, TaskType.CLASSIFICATION)
        self.assertLessEqual(result.shape[1], 5)
    
    def test_none_strategy(self):
        """NONE 策略应返回原始数据"""
        X = pd.DataFrame(np.random.randn(50, 5))
        y = pd.Series(np.random.randint(0, 2, 50))
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.NONE)
        result = selector.fit_transform(X, y, TaskType.CLASSIFICATION)
        pd.testing.assert_frame_equal(result, X)
    
    def test_rfe_strategy(self):
        """RFE 策略应能处理"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.RFE, n_features=3)
        result = selector.fit_transform(X, y, TaskType.CLASSIFICATION)
        self.assertLessEqual(result.shape[1], 5)
    
    def test_model_based_strategy(self):
        """MODEL_BASED 策略应能处理"""
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.randint(0, 2, 100))
        selector = AutoFeatureSelector(strategy=FeatureSelectionStrategy.MODEL_BASED, n_features=3)
        result = selector.fit_transform(X, y, TaskType.CLASSIFICATION)
        self.assertLessEqual(result.shape[1], 5)


# =============================================================================
# ModelLibrary
# =============================================================================

class TestModelLibraryRobustness(unittest.TestCase):
    """ModelLibrary 健壮性测试"""
    
    @classmethod
    def setUpClass(cls):
        ModelLibrary._init()
    
    def test_create_model_unknown_key_raises(self):
        """未知模型 key 应抛出 ValueError"""
        with self.assertRaises(ValueError) as ctx:
            ModelLibrary.create_model('nonexistent_model', TaskType.CLASSIFICATION)
        self.assertIn('nonexistent_model', str(ctx.exception))
    
    def test_create_model_override_params(self):
        """create_model 应支持覆盖默认参数"""
        model = ModelLibrary.create_model('dt', TaskType.CLASSIFICATION, max_depth=3)
        self.assertEqual(model.max_depth, 3)
    
    def test_get_models_filter_by_keys(self):
        """get_models 应支持 key 过滤"""
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION, model_keys=['lr', 'dt'])
        self.assertEqual(set(models.keys()), {'lr', 'dt'})
    
    def test_get_models_filter_by_category(self):
        """get_models 应支持 category 过滤"""
        models = ModelLibrary.get_models(TaskType.CLASSIFICATION, categories=['linear'])
        self.assertIn('lr', models)
        # 确保非 linear 模型不在结果中
        self.assertNotIn('xgb', models)
    
    def test_list_models_returns_dataframe(self):
        """list_models 应返回 DataFrame"""
        df = ModelLibrary.list_models(TaskType.CLASSIFICATION)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('key', df.columns)
        self.assertIn('name', df.columns)
        self.assertIn('category', df.columns)
    
    def test_list_models_empty_when_no_models(self):
        """没有模型时应返回空 DataFrame"""
        df = ModelLibrary.get_models(TaskType.CLASSIFICATION, categories=['nonexistent'])
        self.assertEqual(len(df), 0)
    
    def test_model_spec_str(self):
        """ModelSpec 应有正确的字符串表示"""
        spec = ModelLibrary.get_models(TaskType.CLASSIFICATION)['lr']
        self.assertEqual(spec.key, 'lr')
        self.assertEqual(spec.name, 'LogisticRegression')
    
    def test_lazy_initialization(self):
        """_init 应是懒加载的"""
        # 重新创建引用，验证 _initialized 状态
        self.assertTrue(ModelLibrary._initialized)


# =============================================================================
# CrossValidator
# =============================================================================

class TestCrossValidatorEdgeCases(unittest.TestCase):
    """CrossValidator 极端情况测试"""
    
    def test_extreme_class_imbalance_single_sample(self):
        """某类别只有 1 个样本时应回退到 KFold，且不应崩溃"""
        X = pd.DataFrame(np.random.randn(20, 3))
        y = pd.Series([0] * 19 + [1])  # 类别 1 只有 1 个样本
        cv = CrossValidator(n_splits=5, random_state=42)
        from sklearn.dummy import DummyClassifier
        # 使用 DummyClassifier（支持单类别）避免 proba 形状问题
        model = DummyClassifier(strategy='most_frequent')
        result = cv.cross_validate(model, X, y, TaskType.CLASSIFICATION)
        self.assertIsInstance(result, CVResult)
        # KFold 回退后可能有些 fold 只有一类，指标计算会跳过
        self.assertIsNotNone(result.mean_scores)
    
    def test_extreme_class_imbalance_two_samples(self):
        """某类别只有 2 个样本时应降低折数"""
        X = pd.DataFrame(np.random.randn(20, 3))
        y = pd.Series([0] * 18 + [1, 1])  # 类别 1 只有 2 个样本
        cv = CrossValidator(n_splits=5, random_state=42)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        result = cv.cross_validate(model, X, y, TaskType.CLASSIFICATION)
        self.assertIsInstance(result, CVResult)
    
    def test_non_contiguous_labels(self):
        """非连续整数标签应正确映射"""
        X = pd.DataFrame(np.random.randn(30, 3))
        y = pd.Series([10, 20, 10, 20, 10, 20] * 5)  # 标签 10 和 20
        cv = CrossValidator(n_splits=3, random_state=42)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        result = cv.cross_validate(model, X, y, TaskType.CLASSIFICATION)
        self.assertIsInstance(result, CVResult)
        # OOF 预测应包含原始标签值
        unique_preds = set(result.oof_pred)
        self.assertTrue(unique_preds.issubset({10, 20}))
    
    def test_regression_cv(self):
        """回归 CV 应正常工作"""
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randn(50))
        cv = CrossValidator(n_splits=3, random_state=42)
        from sklearn.linear_model import Ridge
        model = Ridge(random_state=42)
        result = cv.cross_validate(model, X, y, TaskType.REGRESSION)
        self.assertIsInstance(result, CVResult)
        self.assertIn('rmse', result.mean_scores or {})
    
    def test_binary_classification_with_proba(self):
        """二分类应支持概率预测"""
        X = pd.DataFrame(np.random.randn(50, 3))
        y = pd.Series(np.random.randint(0, 2, 50))
        cv = CrossValidator(n_splits=3, random_state=42)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        result = cv.cross_validate(model, X, y, TaskType.CLASSIFICATION)
        self.assertIsNotNone(result.oof_proba)
        self.assertEqual(len(result.oof_proba), 50)
    
    def test_multiclass_probability_alignment(self):
        """多分类概率矩阵应对齐"""
        X = pd.DataFrame(np.random.randn(60, 3))
        y = pd.Series(np.random.randint(0, 3, 60))
        cv = CrossValidator(n_splits=3, random_state=42)
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=1000, random_state=42)
        result = cv.cross_validate(model, X, y, TaskType.CLASSIFICATION)
        self.assertIsNotNone(result.oof_proba)
        # 多分类概率应为 (n_samples, n_classes)
        self.assertEqual(result.oof_proba.shape, (60, 3))
    
    def test_feature_importance_extraction(self):
        """特征重要性应能正确提取"""
        X = pd.DataFrame(np.random.randn(50, 3), columns=['a', 'b', 'c'])
        y = pd.Series(np.random.randint(0, 2, 50))
        cv = CrossValidator(n_splits=3, random_state=42)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        result = cv.cross_validate(model, X, y, TaskType.CLASSIFICATION)
        self.assertIsNotNone(result.feature_importance)


if __name__ == '__main__':
    unittest.main()
