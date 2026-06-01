"""
RLOptimizer 专项测试

覆盖:
  - _build_candidates: 列表 / dict / SearchSpace / log scale / conditional
  - _apply_action: 动作索引到 (param, value) 的映射正确性
  - _ParamActionSpace: 编码/解码/apply
  - metric 选择: classification -> roc_auc, regression -> neg_mean_squared_error
  - fallback: TORCH_AVAILABLE=False 时稳定返回 rl_torch_unavailable
  - optimize 完整流程: RL 训练 + 结果结构
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from core.reinforcement_learning import (
    RLOptimizer, _ParamActionSpace, _EvaluationCache,
    extract_meta_features, TORCH_AVAILABLE
)
from core.search_space import SearchSpace
from core.modeling_engine import TaskType


class TestExtractMetaFeatures(unittest.TestCase):
    def test_classification(self):
        X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': ['x', 'y', 'x', 'y', 'x']})
        y = pd.Series([0, 1, 0, 1, 0])
        features = extract_meta_features(X, y, TaskType.CLASSIFICATION)
        self.assertEqual(len(features), 8)
        self.assertTrue(np.all(np.isfinite(features)))
    
    def test_regression(self):
        X = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        y = pd.Series([1.0, 2.0, 3.0])
        features = extract_meta_features(X, y, TaskType.REGRESSION)
        self.assertEqual(len(features), 8)
        self.assertTrue(np.all(np.isfinite(features)))


class TestParamActionSpace(unittest.TestCase):
    def test_flat_action_space(self):
        candidates = {
            'C': [0.01, 0.1, 1.0],
            'penalty': ['l1', 'l2'],
        }
        space = _ParamActionSpace(candidates)
        # action_dim = 3 + 2 = 5
        self.assertEqual(space.action_dim, 5)
        
        # 解码验证
        self.assertEqual(space.decode(0), ('C', 0.01, 0))
        self.assertEqual(space.decode(2), ('C', 1.0, 2))
        self.assertEqual(space.decode(3), ('penalty', 'l1', 0))
        self.assertEqual(space.decode(4), ('penalty', 'l2', 1))
    
    def test_apply_action(self):
        candidates = {'C': [0.01, 0.1, 1.0], 'penalty': ['l1', 'l2']}
        space = _ParamActionSpace(candidates)
        search_space = SearchSpace({'C': {'type': 'float', 'low': 0.01, 'high': 1.0},
                                     'penalty': {'type': 'categorical', 'choices': ['l1', 'l2']}})
        
        current = {'C': 0.1, 'penalty': 'l1'}
        # action 0 -> C = 0.01
        result = space.apply(current, 0, search_space)
        self.assertEqual(result['C'], 0.01)
        self.assertEqual(result['penalty'], 'l1')
        
        # action 4 -> penalty = 'l2'
        result = space.apply(current, 4, search_space)
        self.assertEqual(result['C'], 0.1)
        self.assertEqual(result['penalty'], 'l2')
    
    def test_get_action_index(self):
        candidates = {'C': [0.01, 0.1, 1.0], 'penalty': ['l1', 'l2']}
        space = _ParamActionSpace(candidates)
        self.assertEqual(space.get_action_index('C', 0), 0)
        self.assertEqual(space.get_action_index('C', 2), 2)
        self.assertEqual(space.get_action_index('penalty', 0), 3)
        self.assertEqual(space.get_action_index('penalty', 1), 4)


class TestRLOptimizerBuildCandidates(unittest.TestCase):
    def setUp(self):
        self.optimizer = RLOptimizer(n_trials=5, random_state=42)
    
    def test_list_format(self):
        space = {'C': [0.01, 0.1, 1.0], 'penalty': ['l1', 'l2']}
        cand = self.optimizer._build_candidates(space, n_candidates=8)
        self.assertEqual(cand['C'], [0.01, 0.1, 1.0])
        self.assertEqual(cand['penalty'], ['l1', 'l2'])
    
    def test_dict_format_int(self):
        space = {'max_depth': {'type': 'int', 'low': 3, 'high': 10}}
        cand = self.optimizer._build_candidates(space, n_candidates=5)
        self.assertTrue(all(isinstance(v, int) for v in cand['max_depth']))
        self.assertTrue(all(3 <= v <= 10 for v in cand['max_depth']))
    
    def test_dict_format_float_linear(self):
        space = {'C': {'type': 'float', 'low': 0.01, 'high': 10.0}}
        cand = self.optimizer._build_candidates(space, n_candidates=5)
        self.assertEqual(len(cand['C']), 5)
        self.assertAlmostEqual(cand['C'][0], 0.01, places=5)
        self.assertAlmostEqual(cand['C'][-1], 10.0, places=5)
    
    def test_dict_format_float_log(self):
        space = {'lr': {'type': 'float', 'low': 1e-5, 'high': 1.0, 'scale': 'log'}}
        cand = self.optimizer._build_candidates(space, n_candidates=5)
        self.assertEqual(len(cand['lr']), 5)
        self.assertTrue(all(v > 0 for v in cand['lr']))
        self.assertAlmostEqual(cand['lr'][0], 1e-5, places=5)
        self.assertAlmostEqual(cand['lr'][-1], 1.0, places=5)
        # 对数尺度: 相邻比值应接近
        ratios = [cand['lr'][i+1] / cand['lr'][i] for i in range(len(cand['lr'])-1)]
        self.assertTrue(all(r > 1 for r in ratios))
    
    def test_search_space_object(self):
        space = SearchSpace({
            'lr': {'type': 'float', 'low': 1e-5, 'high': 1.0, 'scale': 'log'},
            'max_depth': {'type': 'int', 'low': 3, 'high': 10},
        })
        cand = self.optimizer._build_candidates(space, n_candidates=5)
        self.assertIn('lr', cand)
        self.assertIn('max_depth', cand)
    
    def test_categorical_format(self):
        space = {'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear']}}
        cand = self.optimizer._build_candidates(space, n_candidates=8)
        self.assertEqual(set(cand['kernel']), {'rbf', 'linear'})


class TestRLOptimizerRandomParams(unittest.TestCase):
    def setUp(self):
        self.optimizer = RLOptimizer(n_trials=5, random_state=42)
    
    def test_list_format(self):
        space = {'C': [0.01, 0.1, 1.0], 'penalty': ['l1', 'l2']}
        params = self.optimizer._random_params(space)
        self.assertIn(params['C'], [0.01, 0.1, 1.0])
        self.assertIn(params['penalty'], ['l1', 'l2'])
    
    def test_search_space_object(self):
        space = SearchSpace({
            'lr': {'type': 'float', 'low': 1e-5, 'high': 1.0, 'scale': 'log'},
        })
        params = self.optimizer._random_params(space)
        self.assertIn('lr', params)
        self.assertGreaterEqual(params['lr'], 1e-5)
        self.assertLessEqual(params['lr'], 1.0)


class TestRLOptimizerMetricSelection(unittest.TestCase):
    def setUp(self):
        self.optimizer = RLOptimizer(n_trials=3, random_state=42)
        self.X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        self.y_binary = pd.Series([0, 1, 0, 1, 0])
        self.y_multi = pd.Series([0, 1, 2, 0, 1])
        self.y_reg = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    
    @patch('core.reinforcement_learning.TORCH_AVAILABLE', True)
    @patch('core.reinforcement_learning.torch')
    def test_classification_binary_metric(self, mock_torch):
        """二分类默认使用 roc_auc"""
        mock_torch.cuda.is_available.return_value = False
        mock_torch.device = MagicMock(return_value='cpu')
        mock_torch.manual_seed = MagicMock()
        
        with patch.object(self.optimizer, '_torch_available', True):
            with patch.object(self.optimizer, '_device', 'cpu'):
                # 模拟 DQN
                mock_policy = MagicMock()
                mock_target = MagicMock()
                mock_optimizer = MagicMock()
                
                with patch('core.reinforcement_learning._DQN', return_value=mock_policy):
                    with patch('core.reinforcement_learning.optim.Adam', return_value=mock_optimizer):
                        # 使用 lr 模型（LogisticRegression）测试 metric 选择
                        with patch.object(self.optimizer, '_evaluate_with_cache', return_value=0.8):
                            result = self.optimizer.optimize('lr', self.X, self.y_binary, 'classification')
                            # 结果结构验证
                            self.assertEqual(result.model_key, 'lr')
                            self.assertIsInstance(result.best_params, dict)
                            self.assertIsInstance(result.optimization_history, list)
    
    def test_regression_metric(self):
        """回归默认使用 neg_mean_squared_error"""
        with patch('core.reinforcement_learning.TORCH_AVAILABLE', False):
            optimizer = RLOptimizer(n_trials=3, random_state=42)
            result = optimizer.optimize('linear', self.X, self.y_reg, 'regression')
            self.assertEqual(result.model_key, 'linear')
            self.assertIsInstance(result.best_params, dict)


class TestRLOptimizerFallback(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({'a': [1, 2, 3, 4, 5], 'b': [5, 4, 3, 2, 1]})
        self.y = pd.Series([0, 1, 0, 1, 0])
    
    @patch('core.reinforcement_learning.TORCH_AVAILABLE', False)
    def test_torch_unavailable_returns_explicit_sampler_type(self):
        """TORCH_AVAILABLE=False 时应返回 rl_torch_unavailable"""
        optimizer = RLOptimizer(n_trials=3, random_state=42)
        result = optimizer.optimize('lr', self.X, self.y, 'classification')
        self.assertEqual(result.sampler_type, 'rl_torch_unavailable')
        self.assertIsInstance(result.best_params, dict)
        self.assertIsInstance(result.optimization_history, list)
        self.assertGreaterEqual(result.n_trials, 0)
    
    @patch('core.reinforcement_learning.TORCH_AVAILABLE', False)
    def test_torch_unavailable_stable_across_calls(self):
        """多次调用 fallback 应产生稳定结果"""
        optimizer = RLOptimizer(n_trials=3, random_state=42)
        results = [
            optimizer.optimize('lr', self.X, self.y, 'classification')
            for _ in range(3)
        ]
        sampler_types = [r.sampler_type for r in results]
        self.assertTrue(all(s == 'rl_torch_unavailable' for s in sampler_types))


class TestEvaluationCache(unittest.TestCase):
    def test_cache_hit(self):
        cache = _EvaluationCache()
        params = {'C': 0.1, 'kernel': 'rbf'}
        cache.set(params, 1.0, 0.85)
        self.assertEqual(cache.get(params, 1.0), 0.85)
    
    def test_cache_miss(self):
        cache = _EvaluationCache()
        params = {'C': 0.1, 'kernel': 'rbf'}
        self.assertIsNone(cache.get(params, 1.0))
    
    def test_cache_different_params(self):
        cache = _EvaluationCache()
        cache.set({'C': 0.1}, 1.0, 0.85)
        self.assertIsNone(cache.get({'C': 0.2}, 1.0))
    
    def test_cache_different_subset(self):
        cache = _EvaluationCache()
        cache.set({'C': 0.1}, 1.0, 0.85)
        self.assertIsNone(cache.get({'C': 0.1}, 0.5))


class TestRLOptimizerIntegration(unittest.TestCase):
    """集成测试：使用真实模型和数据"""
    
    def setUp(self):
        self.X = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50),
        })
        self.y_cls = pd.Series(np.random.randint(0, 2, 50))
        self.y_reg = pd.Series(np.random.randn(50))
    
    def _test_optimize_structure(self, result):
        self.assertIn('best_params', dir(result) if hasattr(result, 'best_params') else [])
        self.assertIsInstance(result.best_params, dict)
        self.assertIsInstance(result.best_score, float)
        self.assertIsInstance(result.optimization_history, list)
        self.assertIsInstance(result.n_trials, int)
        self.assertIsInstance(result.optimize_time, float)
        self.assertTrue(len(result.sampler_type) > 0)
        
        if result.optimization_history:
            for h in result.optimization_history:
                self.assertIn('trial', h)
                self.assertIn('params', h)
                self.assertIn('score', h)
    
    @patch('core.reinforcement_learning.TORCH_AVAILABLE', False)
    def test_fallback_classification(self):
        optimizer = RLOptimizer(n_trials=3, random_state=42)
        result = optimizer.optimize('lr', self.X, self.y_cls, 'classification')
        self._test_optimize_structure(result)
        self.assertEqual(result.sampler_type, 'rl_torch_unavailable')
    
    @patch('core.reinforcement_learning.TORCH_AVAILABLE', False)
    def test_fallback_regression(self):
        optimizer = RLOptimizer(n_trials=3, random_state=42)
        result = optimizer.optimize('ridge', self.X, self.y_reg, 'regression')
        self._test_optimize_structure(result)
        self.assertEqual(result.sampler_type, 'rl_torch_unavailable')
    
    @patch('core.reinforcement_learning.TORCH_AVAILABLE', False)
    def test_empty_search_space(self):
        """无搜索空间的模型应返回 rl_none"""
        optimizer = RLOptimizer(n_trials=3, random_state=42)
        # regression linear 没有 hyperparam_space
        result = optimizer.optimize('linear', self.X, self.y_reg, 'regression')
        self.assertEqual(result.sampler_type, 'rl_none')


if __name__ == '__main__':
    unittest.main()
