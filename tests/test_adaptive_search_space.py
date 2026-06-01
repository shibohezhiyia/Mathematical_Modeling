"""
自适应搜索空间测试
"""
import unittest
import numpy as np

from core.adaptive_search_space import (
    AdaptiveSearchSpace, AdaptationConfig, SearchSpaceAdapter
)
from core.search_space import SearchSpace


class TestAdaptiveSearchSpace(unittest.TestCase):
    """测试自适应搜索空间"""
    
    def test_basic_creation(self):
        """测试基本创建"""
        config = {'lr': {'type': 'float', 'low': 1e-5, 'high': 1.0, 'scale': 'log'}}
        space = AdaptiveSearchSpace(config)
        self.assertEqual(len(space.params), 1)
        self.assertEqual(space.params['lr'].low, 1e-5)
    
    def test_update_history(self):
        """测试历史记录更新"""
        config = {'alpha': {'type': 'float', 'low': 0.01, 'high': 1.0}}
        space = AdaptiveSearchSpace(config)
        space.update_history({'alpha': 0.5}, 0.8)
        self.assertEqual(len(space.history), 1)
    
    def test_should_adapt_false_before_warmup(self):
        """测试 warm-up 前不调整"""
        space = AdaptiveSearchSpace(
            {'x': {'type': 'float', 'low': 0, 'high': 10}},
            AdaptationConfig(warmup_trials=5)
        )
        for i in range(4):
            space.update_history({'x': float(i)}, float(i))
        self.assertFalse(space.should_adapt())
    
    def test_should_adapt_true_after_warmup(self):
        """测试 warm-up 后满足条件时调整"""
        space = AdaptiveSearchSpace(
            {'x': {'type': 'float', 'low': 0, 'high': 10}},
            AdaptationConfig(warmup_trials=5, adapt_every=1)
        )
        for i in range(5):
            space.update_history({'x': float(i)}, float(i))
        self.assertTrue(space.should_adapt())
    
    def test_adapt_numeric_shrink(self):
        """测试数值型参数收缩"""
        space = AdaptiveSearchSpace(
            {'x': {'type': 'float', 'low': 0, 'high': 100}},
            AdaptationConfig(warmup_trials=3, shrink_ratio=0.5, min_range_ratio=0.1)
        )
        # 高分集中在 80-90 区域
        for i in range(10):
            x = float(i * 10)
            score = 0.9 if 80 <= x <= 90 else 0.3
            space.update_history({'x': x}, score)
        
        report = space.adapt(direction='maximize')
        # 检查是否调整了参数
        self.assertIn('adapted_params', report)
        self.assertIn('importance', report)
    
    def test_adapt_categorical_prune(self):
        """测试离散值剪枝"""
        space = AdaptiveSearchSpace(
            {'cat': {'type': 'categorical', 'choices': ['a', 'b', 'c', 'd']}},
            AdaptationConfig(warmup_trials=3, prune_threshold=0.3, adapt_every=1, correlation_threshold=0.0)
        )
        # 'a' 和 'b' 表现好，'c' 和 'd' 表现差
        for _ in range(5):
            space.update_history({'cat': 'a'}, 0.9)
            space.update_history({'cat': 'b'}, 0.85)
            space.update_history({'cat': 'c'}, 0.4)
            space.update_history({'cat': 'd'}, 0.3)
        
        report = space.adapt(direction='maximize')
        pruned = report.get('pruned_values', {}).get('cat', [])
        # 至少剪掉了一些低分值
        self.assertGreaterEqual(len(pruned), 1)
    
    def test_importance_computation(self):
        """测试参数重要性计算"""
        space = AdaptiveSearchSpace(
            {'x': {'type': 'float', 'low': 0, 'high': 10}},
            AdaptationConfig(warmup_trials=3)
        )
        # x 与 score 强正相关
        for i in range(10):
            space.update_history({'x': float(i)}, float(i) / 10.0)
        
        importance = space._compute_param_importance(direction='maximize')
        self.assertIn('x', importance)
        self.assertGreater(importance['x'], 0.5)
    
    def test_reset(self):
        """测试重置"""
        space = AdaptiveSearchSpace({'x': {'type': 'float', 'low': 0, 'high': 10}})
        space.update_history({'x': 5.0}, 0.8)
        space.reset()
        self.assertEqual(len(space.history), 0)
        self.assertEqual(space.params['x'].low, 0)
    
    def test_backward_compatible_sample(self):
        """测试与 SearchSpace 兼容的采样"""
        space = AdaptiveSearchSpace({'x': [1, 2, 3]})
        params = space.sample(random_state=42)
        self.assertIn(params['x'], [1, 2, 3])


class TestSearchSpaceAdapter(unittest.TestCase):
    """测试搜索空间适配器"""
    
    def test_adapter_sample_and_report(self):
        """测试采样和报告流程"""
        adapter = SearchSpaceAdapter({'x': {'type': 'float', 'low': 0, 'high': 1}})
        params = adapter.sample(random_state=42)
        self.assertIn('x', params)
        adapter.report(0.8)
        self.assertEqual(len(adapter.space.history), 1)
    
    def test_adapter_from_search_space(self):
        """测试从 SearchSpace 构建"""
        base = SearchSpace({'y': [10, 20, 30]})
        adapter = SearchSpaceAdapter(base)
        params = adapter.sample(random_state=42)
        self.assertIn(params['y'], [10, 20, 30])


if __name__ == '__main__':
    unittest.main()
