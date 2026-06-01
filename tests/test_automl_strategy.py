"""
AutoMLStrategy 单元测试

覆盖:
  - recommend() 不同数据规模和用户偏好
  - _recommend_optimizer 所有分支
  - _recommend_models 分类/回归各偏好
  - _recommend_ensemble 所有分支
  - _recommend_deep_learning 启用/禁用条件
  - _estimate_time 各时间档位
  - _build_reasoning 文本构建
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.automl_strategy import AutoMLStrategy, StrategyRecommendation
from core.meta_feature_extractor import MetaFeatures
from core.modeling_engine import TaskType


class TestAutoMLStrategyRecommend(unittest.TestCase):
    """测试 recommend 主入口"""

    def _make_meta(self, n_samples=1000, n_features=20, complexity_score=50.0,
                   missing_ratio=0.0, class_imbalance_ratio=1.0, sparsity=0.0):
        return MetaFeatures(
            n_samples=n_samples,
            n_features=n_features,
            n_numeric=n_features,
            n_categorical=0,
            sample_feature_ratio=n_samples / max(n_features, 1),
            numeric_ratio=1.0,
            categorical_ratio=0.0,
            missing_ratio=missing_ratio,
            sparsity=sparsity,
            feature_correlation_mean=0.3,
            feature_correlation_max=0.6,
            n_classes=2,
            class_imbalance_ratio=class_imbalance_ratio,
            target_entropy=0.5,
            target_std=1.0,
            complexity_score=complexity_score,
        )

    def test_recommend_returns_dataclass(self):
        meta = self._make_meta()
        rec = AutoMLStrategy.recommend(meta, TaskType.CLASSIFICATION)
        self.assertIsInstance(rec, StrategyRecommendation)
        self.assertIsInstance(rec.optimizer, str)
        self.assertIsInstance(rec.model_keys, list)
        self.assertIsInstance(rec.ensemble, str)
        self.assertIsInstance(rec.deep_learning, dict)
        self.assertIsInstance(rec.expected_time, str)
        self.assertIsInstance(rec.reasoning, str)

    def test_recommend_small_data_balanced(self):
        """小数据 + balanced => bayesian"""
        meta = self._make_meta(n_samples=500, n_features=10, complexity_score=30)
        rec = AutoMLStrategy.recommend(meta, TaskType.CLASSIFICATION, user_preference='balanced')
        self.assertEqual(rec.optimizer, 'bayesian')
        self.assertIn('lr', rec.model_keys)

    def test_recommend_large_data_speed_first(self):
        """大数据 + speed_first => hyperband"""
        meta = self._make_meta(n_samples=100000, n_features=50)
        rec = AutoMLStrategy.recommend(meta, TaskType.CLASSIFICATION, user_preference='speed_first')
        self.assertEqual(rec.optimizer, 'hyperband')
        self.assertEqual(rec.ensemble, 'best_single')

    def test_recommend_very_large_data_speed_first(self):
        """超大数据 + speed_first => hyperband (n_samples > 50000 分支)"""
        meta = self._make_meta(n_samples=80000, n_features=30)
        rec = AutoMLStrategy.recommend(meta, TaskType.CLASSIFICATION, user_preference='speed_first')
        self.assertEqual(rec.optimizer, 'hyperband')

    def test_recommend_high_complexity_accuracy_first(self):
        """高复杂度 + accuracy_first => genetic"""
        meta = self._make_meta(n_samples=5000, n_features=50, complexity_score=65)
        rec = AutoMLStrategy.recommend(meta, TaskType.CLASSIFICATION, user_preference='accuracy_first')
        self.assertEqual(rec.optimizer, 'genetic')
        # accuracy_first 分类应包含更多模型 (最多5个)
        self.assertIn('et', rec.model_keys)

    def test_recommend_exploration_large_data(self):
        """大数据高维 + exploration => rl"""
        meta = self._make_meta(n_samples=20000, n_features=60)
        rec = AutoMLStrategy.recommend(meta, TaskType.CLASSIFICATION, user_preference='exploration')
        self.assertEqual(rec.optimizer, 'rl')

    def test_recommend_exploration_small_data(self):
        """小数据 + exploration => genetic"""
        meta = self._make_meta(n_samples=500, n_features=10)
        rec = AutoMLStrategy.recommend(meta, TaskType.CLASSIFICATION, user_preference='exploration')
        self.assertEqual(rec.optimizer, 'genetic')

    def test_recommend_user_override_optimizer(self):
        meta = self._make_meta()
        rec = AutoMLStrategy.recommend(meta, TaskType.CLASSIFICATION, user_optimizer='random')
        self.assertEqual(rec.optimizer, 'random')
        self.assertIn('用户指定优化器', rec.reasoning)

    def test_recommend_user_override_models(self):
        meta = self._make_meta()
        rec = AutoMLStrategy.recommend(meta, TaskType.CLASSIFICATION, user_model_keys=['lr', 'dt'])
        self.assertEqual(rec.model_keys, ['lr', 'dt'])
        self.assertIn('用户指定模型', rec.reasoning)

    def test_recommend_both_user_overrides(self):
        meta = self._make_meta()
        rec = AutoMLStrategy.recommend(
            meta, TaskType.CLASSIFICATION,
            user_optimizer='bayesian',
            user_model_keys=['lr']
        )
        self.assertEqual(rec.optimizer, 'bayesian')
        self.assertEqual(rec.model_keys, ['lr'])


class TestRecommendOptimizer(unittest.TestCase):
    """测试 _recommend_optimizer 分支覆盖"""

    def _make_meta(self, n_samples=1000, n_features=20, complexity_score=50.0):
        return MetaFeatures(
            n_samples=n_samples, n_features=n_features,
            complexity_score=complexity_score,
        )

    def test_speed_first_small(self):
        meta = self._make_meta(n_samples=1000)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'speed_first'), 'random')

    def test_speed_first_large(self):
        meta = self._make_meta(n_samples=100000)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'speed_first'), 'hyperband')

    def test_accuracy_first_high_complexity(self):
        meta = self._make_meta(complexity_score=65)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'accuracy_first'), 'genetic')

    def test_accuracy_first_low_complexity(self):
        meta = self._make_meta(complexity_score=50)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'accuracy_first'), 'bayesian')

    def test_exploration_large(self):
        meta = self._make_meta(n_samples=20000, n_features=60)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'exploration'), 'rl')

    def test_exploration_small(self):
        meta = self._make_meta(n_samples=500, n_features=10)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'exploration'), 'genetic')

    def test_balanced_very_large(self):
        meta = self._make_meta(n_samples=200000)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'balanced'), 'hyperband')

    def test_balanced_small(self):
        meta = self._make_meta(n_samples=500)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'balanced'), 'bayesian')

    def test_balanced_high_complexity(self):
        meta = self._make_meta(n_samples=5000, complexity_score=75)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'balanced'), 'genetic')

    def test_balanced_medium(self):
        meta = self._make_meta(n_samples=5000, complexity_score=50)
        self.assertEqual(AutoMLStrategy._recommend_optimizer(meta, 'balanced'), 'bayesian')


class TestRecommendModels(unittest.TestCase):
    """测试 _recommend_models"""

    def _make_meta(self, n_samples=1000, n_features=20, class_imbalance_ratio=1.0):
        return MetaFeatures(
            n_samples=n_samples, n_features=n_features,
            class_imbalance_ratio=class_imbalance_ratio,
        )

    def test_classification_speed_first(self):
        meta = self._make_meta()
        models = AutoMLStrategy._recommend_models(meta, TaskType.CLASSIFICATION, 'speed_first')
        self.assertEqual(models, ['lr', 'dt', 'lgb'])

    def test_classification_accuracy_first(self):
        meta = self._make_meta()
        models = AutoMLStrategy._recommend_models(meta, TaskType.CLASSIFICATION, 'accuracy_first')
        self.assertIn('lr', models)
        self.assertIn('et', models)
        # 扩展后最多5个，svm/gbdt 可能在截断位置之后
        self.assertLessEqual(len(models), 5)

    def test_classification_large_data_removes_slow(self):
        meta = self._make_meta(n_samples=200000)
        models = AutoMLStrategy._recommend_models(meta, TaskType.CLASSIFICATION, 'accuracy_first')
        self.assertNotIn('svm', models)
        self.assertNotIn('knn', models)

    def test_classification_high_dim_adds_lr(self):
        meta = self._make_meta(n_features=150)
        models = AutoMLStrategy._recommend_models(meta, TaskType.CLASSIFICATION, 'balanced')
        self.assertEqual(models[0], 'lr')

    def test_classification_imbalance_adds_xgb(self):
        meta = self._make_meta(class_imbalance_ratio=6.0)
        models = AutoMLStrategy._recommend_models(meta, TaskType.CLASSIFICATION, 'balanced')
        self.assertIn('xgb', models)

    def test_classification_max_five_models(self):
        meta = self._make_meta(n_features=200, class_imbalance_ratio=10.0)
        models = AutoMLStrategy._recommend_models(meta, TaskType.CLASSIFICATION, 'accuracy_first')
        self.assertLessEqual(len(models), 5)

    def test_regression_speed_first(self):
        meta = self._make_meta()
        models = AutoMLStrategy._recommend_models(meta, TaskType.REGRESSION, 'speed_first')
        self.assertEqual(models, ['ridge', 'linear', 'lgb'])

    def test_regression_accuracy_first(self):
        meta = self._make_meta()
        models = AutoMLStrategy._recommend_models(meta, TaskType.REGRESSION, 'accuracy_first')
        self.assertIn('et', models)
        self.assertLessEqual(len(models), 5)

    def test_regression_large_data(self):
        meta = self._make_meta(n_samples=200000)
        models = AutoMLStrategy._recommend_models(meta, TaskType.REGRESSION, 'accuracy_first')
        self.assertNotIn('svr', models)
        self.assertNotIn('knn', models)


class TestRecommendEnsemble(unittest.TestCase):
    """测试 _recommend_ensemble"""

    def _make_meta(self, n_samples=1000, complexity_score=50.0):
        return MetaFeatures(n_samples=n_samples, complexity_score=complexity_score)

    def test_speed_first(self):
        meta = self._make_meta()
        self.assertEqual(AutoMLStrategy._recommend_ensemble(meta, 'speed_first'), 'best_single')

    def test_small_data(self):
        meta = self._make_meta(n_samples=300)
        self.assertEqual(AutoMLStrategy._recommend_ensemble(meta, 'balanced'), 'best_single')

    def test_high_complexity(self):
        meta = self._make_meta(complexity_score=75)
        self.assertEqual(AutoMLStrategy._recommend_ensemble(meta, 'balanced'), 'stacking')

    def test_default_weighted(self):
        meta = self._make_meta(n_samples=1000, complexity_score=50)
        self.assertEqual(AutoMLStrategy._recommend_ensemble(meta, 'balanced'), 'weighted')


class TestRecommendDeepLearning(unittest.TestCase):
    """测试 _recommend_deep_learning"""

    def _make_meta(self, n_samples=5000, n_features=20):
        return MetaFeatures(n_samples=n_samples, n_features=n_features)

    def test_disabled_by_preference(self):
        meta = self._make_meta()
        config = AutoMLStrategy._recommend_deep_learning(meta, 'balanced')
        self.assertFalse(config['enabled'])
        self.assertEqual(config['models'], [])

    def test_disabled_by_preference_speed(self):
        meta = self._make_meta()
        config = AutoMLStrategy._recommend_deep_learning(meta, 'speed_first')
        self.assertFalse(config['enabled'])

    def test_disabled_by_small_data(self):
        meta = self._make_meta(n_samples=500, n_features=20)
        config = AutoMLStrategy._recommend_deep_learning(meta, 'accuracy_first')
        self.assertFalse(config['enabled'])

    def test_disabled_by_low_dim(self):
        meta = self._make_meta(n_samples=5000, n_features=5)
        config = AutoMLStrategy._recommend_deep_learning(meta, 'accuracy_first')
        self.assertFalse(config['enabled'])

    def test_enabled_accuracy_first(self):
        meta = self._make_meta(n_samples=5000, n_features=20)
        config = AutoMLStrategy._recommend_deep_learning(meta, 'accuracy_first')
        self.assertTrue(config['enabled'])
        self.assertIn('torch_mlp', config['models'])
        self.assertIn('torch_cnn1d', config['models'])

    def test_enabled_exploration(self):
        meta = self._make_meta(n_samples=5000, n_features=15)
        config = AutoMLStrategy._recommend_deep_learning(meta, 'exploration')
        self.assertTrue(config['enabled'])
        self.assertIn('torch_mlp', config['models'])
        # n_features < 20, 不添加 cnn1d
        self.assertNotIn('torch_cnn1d', config['models'])

    def test_enabled_high_dim(self):
        meta = self._make_meta(n_samples=5000, n_features=25)
        config = AutoMLStrategy._recommend_deep_learning(meta, 'accuracy_first')
        self.assertTrue(config['enabled'])
        self.assertIn('torch_cnn1d', config['models'])


class TestEstimateTime(unittest.TestCase):
    """测试 _estimate_time"""

    def _make_meta(self, n_samples=1000):
        return MetaFeatures(n_samples=n_samples)

    def test_very_fast(self):
        meta = self._make_meta()
        time_str = AutoMLStrategy._estimate_time(meta, 'random', [])
        self.assertEqual(time_str, '很快 (< 2分钟)')

    def test_fast(self):
        meta = self._make_meta()
        time_str = AutoMLStrategy._estimate_time(meta, 'random', ['lr'])
        self.assertEqual(time_str, '较快 (2-5分钟)')

    def test_medium(self):
        meta = self._make_meta(n_samples=50000)
        time_str = AutoMLStrategy._estimate_time(meta, 'hyperband', ['lr', 'xgb'])
        self.assertEqual(time_str, '中等 (5-15分钟)')

    def test_slow(self):
        meta = self._make_meta(n_samples=50000)
        time_str = AutoMLStrategy._estimate_time(meta, 'bayesian', ['lr', 'xgb', 'lgb', 'rf'])
        self.assertEqual(time_str, '较慢 (15-30分钟)')

    def test_very_slow(self):
        meta = self._make_meta(n_samples=200000)
        time_str = AutoMLStrategy._estimate_time(meta, 'genetic', ['lr', 'xgb', 'lgb', 'rf', 'et'])
        self.assertEqual(time_str, '很慢 (> 30分钟)')

    def test_unknown_optimizer_multiplier(self):
        meta = self._make_meta()
        time_str = AutoMLStrategy._estimate_time(meta, 'unknown_optimizer', [])
        # 未知优化器使用默认 1.5 倍率，无模型时 base_time=0
        self.assertEqual(time_str, '很快 (< 2分钟)')


class TestBuildReasoning(unittest.TestCase):
    """测试 _build_reasoning"""

    def test_basic(self):
        meta = MetaFeatures(n_samples=1000, n_features=20, complexity_score=50.0)
        reasoning = AutoMLStrategy._build_reasoning(
            meta, 'bayesian', ['lr', 'xgb'], 'weighted', ['自动推荐优化器: bayesian']
        )
        self.assertIn('1000 样本', reasoning)
        self.assertIn('20 特征', reasoning)
        self.assertIn('50/100', reasoning)
        self.assertIn('bayesian', reasoning)
        self.assertIn('weighted', reasoning)

    def test_with_missing(self):
        meta = MetaFeatures(n_samples=1000, n_features=20, complexity_score=50.0, missing_ratio=0.05)
        reasoning = AutoMLStrategy._build_reasoning(
            meta, 'bayesian', ['lr'], 'weighted', []
        )
        self.assertIn('缺失率', reasoning)

    def test_with_imbalance(self):
        meta = MetaFeatures(n_samples=1000, n_features=20, complexity_score=50.0,
                            class_imbalance_ratio=4.0)
        reasoning = AutoMLStrategy._build_reasoning(
            meta, 'bayesian', ['lr'], 'weighted', []
        )
        self.assertIn('类别不平衡', reasoning)

    def test_with_sparsity(self):
        meta = MetaFeatures(n_samples=1000, n_features=20, complexity_score=50.0, sparsity=0.2)
        reasoning = AutoMLStrategy._build_reasoning(
            meta, 'bayesian', ['lr'], 'weighted', []
        )
        self.assertIn('稀疏度', reasoning)

    def test_no_extra_reasons(self):
        meta = MetaFeatures(n_samples=1000, n_features=20, complexity_score=50.0)
        reasoning = AutoMLStrategy._build_reasoning(
            meta, 'bayesian', ['lr'], 'weighted', []
        )
        self.assertNotIn('缺失率', reasoning)
        self.assertNotIn('类别不平衡', reasoning)
        self.assertNotIn('稀疏度', reasoning)


if __name__ == '__main__':
    unittest.main()
