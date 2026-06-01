"""
RL Optimizer 扩展测试
"""
import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.reinforcement_learning import (
    _ParamActionSpace, _EvaluationCache, RLOptimizer
)


class TestParamActionSpaceExtended(unittest.TestCase):
    """测试参数动作空间"""
    
    def test_empty_search_space(self):
        action_space = _ParamActionSpace({})
        self.assertEqual(len(action_space.actions), 0)
    
    def test_single_param(self):
        action_space = _ParamActionSpace({'lr': [0.01, 0.1]})
        self.assertEqual(len(action_space.actions), 2)
    
    def test_multi_param(self):
        action_space = _ParamActionSpace({'a': [1, 2], 'b': ['x', 'y']})
        self.assertEqual(len(action_space.actions), 4)
    
    def test_decode_action(self):
        action_space = _ParamActionSpace({'a': [1, 2]})
        name, value, idx = action_space.decode(0)
        self.assertEqual(name, 'a')
        self.assertIn(value, [1, 2])
    
    def test_apply_action(self):
        action_space = _ParamActionSpace({'a': [1, 2]})
        result = action_space.apply({'a': 0}, 1, {'a': [1, 2]})
        self.assertEqual(result['a'], 2)


class TestEvaluationCacheExtended(unittest.TestCase):
    """测试评估缓存"""
    
    def test_cache_hit(self):
        cache = _EvaluationCache()
        params = {'lr': 0.01}
        cache.set(params, 0.3, 0.85)
        self.assertEqual(cache.get(params, 0.3), 0.85)
    
    def test_cache_miss(self):
        cache = _EvaluationCache()
        self.assertIsNone(cache.get({'lr': 0.01}, 0.3))
    
    def test_cache_different_params(self):
        cache = _EvaluationCache()
        cache.set({'lr': 0.01}, 0.3, 0.85)
        self.assertIsNone(cache.get({'lr': 0.1}, 0.3))
    
    def test_cache_different_subset(self):
        cache = _EvaluationCache()
        cache.set({'lr': 0.01}, 0.3, 0.85)
        self.assertIsNone(cache.get({'lr': 0.01}, 0.5))


class TestRLOptimizerExtended(unittest.TestCase):
    """测试 RL 优化器扩展场景"""
    
    def setUp(self):
        np.random.seed(42)
        self.X = pd.DataFrame(np.random.randn(50, 5))
        self.y = pd.Series(np.random.randint(0, 2, 50))
    
    def test_init_default_params(self):
        opt = RLOptimizer(n_trials=10)
        self.assertEqual(opt.n_trials, 10)
        self.assertEqual(opt.cv_folds, 3)
    
    def test_init_custom_params(self):
        opt = RLOptimizer(
            n_trials=20,
            cv_folds=5,
            hidden_dim=64,
            n_parallel=2
        )
        self.assertEqual(opt.n_trials, 20)
        self.assertEqual(opt.cv_folds, 5)
        self.assertEqual(opt.hidden_dim, 64)
    
    def test_optimize_returns_valid_result(self):
        opt = RLOptimizer(n_trials=3, subset_schedule=[(0.0, 1.0)])
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertIsNotNone(result.best_params)
        self.assertIsNotNone(result.best_score)
        self.assertIn('rl', result.sampler_type)
    
    def test_optimize_classification(self):
        opt = RLOptimizer(n_trials=3, subset_schedule=[(0.0, 1.0)])
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertIsNotNone(result.best_params)
        self.assertIsNotNone(result.best_score)
    
    def test_optimize_regression(self):
        opt = RLOptimizer(n_trials=3, subset_schedule=[(0.0, 1.0)])
        y_reg = pd.Series(np.random.randn(50))
        result = opt.optimize('ridge', self.X, y_reg, 'regression')
        self.assertIsNotNone(result.best_params)
    
    def test_optimization_history(self):
        opt = RLOptimizer(n_trials=3, subset_schedule=[(0.0, 1.0)])
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertIsInstance(result.optimization_history, list)
    
    def test_sampler_type_with_torch(self):
        opt = RLOptimizer(n_trials=3)
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertIn(result.sampler_type, ['rl_dqn_v2', 'rl_torch_unavailable', 'rl_none', 'rl_no_actions'])
    
    def test_optimize_time_tracking(self):
        opt = RLOptimizer(n_trials=3, subset_schedule=[(0.0, 1.0)])
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertGreaterEqual(result.optimize_time, 0)
    
    def test_parallel_final_validation(self):
        opt = RLOptimizer(n_trials=3, n_parallel=2, subset_schedule=[(0.0, 1.0)])
        result = opt.optimize('lr', self.X, self.y, 'classification')
        self.assertIsNotNone(result.best_params)


if __name__ == '__main__':
    unittest.main()
