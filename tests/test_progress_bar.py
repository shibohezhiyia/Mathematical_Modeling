"""
进度条模块测试
"""
import os
import unittest
from unittest.mock import patch

import numpy as np

from core.progress_bar import (
    DummyTqdm, get_progress_bar, progress_iter, progress_range,
    _TQDM_AVAILABLE
)


class TestDummyTqdm(unittest.TestCase):
    """测试 DummyTqdm（tqdm 不可用时或禁用时的占位符）"""
    
    def test_iterable_mode(self):
        """测试有 iterable 的迭代模式"""
        bar = DummyTqdm(iterable=[1, 2, 3])
        result = list(bar)
        self.assertEqual(result, [1, 2, 3])
        self.assertEqual(bar.n, 3)
    
    def test_total_mode(self):
        """测试只有 total 的 range 模式"""
        bar = DummyTqdm(total=5)
        result = list(bar)
        self.assertEqual(len(result), 5)
        self.assertEqual(bar.n, 5)
    
    def test_zero_total(self):
        """测试 total 为 0"""
        bar = DummyTqdm(total=0)
        result = list(bar)
        self.assertEqual(result, [])
    
    def test_context_manager(self):
        """测试上下文管理器"""
        with DummyTqdm(total=3) as bar:
            list(bar)
        # 不应抛出异常
    
    def test_update(self):
        """测试 update 方法"""
        bar = DummyTqdm()
        bar.update(3)
        self.assertEqual(bar.n, 3)
    
    def test_set_description(self):
        """测试 set_description"""
        bar = DummyTqdm(desc="old")
        bar.set_description("new")
        self.assertEqual(bar.desc, "new")
    
    def test_set_postfix(self):
        """测试 set_postfix（无实际操作）"""
        bar = DummyTqdm()
        bar.set_postfix(loss=0.5)  # 不应抛出异常
    
    def test_close(self):
        """测试 close（无实际操作）"""
        bar = DummyTqdm()
        bar.close()  # 不应抛出异常


class TestGetProgressBar(unittest.TestCase):
    """测试 get_progress_bar 工厂函数"""
    
    def test_disable_returns_dummy(self):
        """禁用时应返回 DummyTqdm"""
        bar = get_progress_bar(total=5, disable=True)
        self.assertIsInstance(bar, DummyTqdm)
    
    def test_env_disable(self):
        """环境变量 DISABLE_TQDM=1 应禁用进度条"""
        with patch.dict(os.environ, {'DISABLE_TQDM': '1'}):
            bar = get_progress_bar(total=5)
            self.assertIsInstance(bar, DummyTqdm)
    
    def test_env_disable_true(self):
        """环境变量 DISABLE_TQDM=true 应禁用进度条"""
        with patch.dict(os.environ, {'DISABLE_TQDM': 'true'}):
            bar = get_progress_bar(total=5)
            self.assertIsInstance(bar, DummyTqdm)
    
    def test_progress_iter(self):
        """测试 progress_iter 包装"""
        result = list(progress_iter([10, 20, 30], disable=True))
        self.assertEqual(result, [10, 20, 30])
    
    def test_progress_range(self):
        """测试 progress_range 包装"""
        result = list(progress_range(4, disable=True))
        self.assertEqual(result, [0, 1, 2, 3])
    
    def test_progress_range_with_total(self):
        """测试 progress_range 的 total 参数"""
        result = list(progress_range(3, desc="test", disable=True))
        self.assertEqual(result, [0, 1, 2])


class TestIntegrationWithOptimizers(unittest.TestCase):
    """测试进度条与优化器集成"""
    
    def test_optimizer_verbose_false(self):
        """verbose=False 时优化器应正常工作"""
        from core.optimizer_factory import OptimizerFactory
        
        opt = OptimizerFactory.create('random', n_trials=2, verbose=False)
        self.assertFalse(opt.verbose)
    
    def test_optimizer_verbose_true(self):
        """verbose=True 时优化器应正常工作"""
        from core.optimizer_factory import OptimizerFactory
        
        opt = OptimizerFactory.create('random', n_trials=2, verbose=True)
        self.assertTrue(opt.verbose)
    
    def test_bayesian_optimizer_verbose(self):
        """BayesianOptimizer 应支持 verbose"""
        from core.hyperparameter_optimizer import BayesianOptimizer
        
        opt = BayesianOptimizer(n_trials=2, verbose=False)
        self.assertFalse(opt.verbose)
    
    def test_rl_optimizer_verbose(self):
        """RLOptimizer 应支持 verbose"""
        from core.reinforcement_learning import RLOptimizer
        
        opt = RLOptimizer(n_trials=2, verbose=False)
        self.assertFalse(opt.verbose)
    
    def test_modeling_engine_verbose(self):
        """ModelingEngine 应支持 verbose"""
        from core.modeling_engine import ModelingEngine
        
        engine = ModelingEngine(n_splits=2, verbose=False)
        self.assertFalse(engine.verbose)
    
    def test_cross_validator_verbose(self):
        """CrossValidator 应支持 verbose"""
        from core.modeling_engine import CrossValidator
        
        cv = CrossValidator(n_splits=3, verbose=False)
        self.assertFalse(cv.verbose)
    
    def test_parallel_modeling_engine_verbose(self):
        """ParallelModelingEngine 应支持 verbose"""
        from core.parallel_modeling import ParallelModelingEngine
        
        engine = ParallelModelingEngine(verbose=False)
        self.assertFalse(engine.verbose)
    
    def test_hyperparameter_search_verbose(self):
        """HyperparameterSearch 应支持 verbose"""
        from core.parallel_modeling import HyperparameterSearch
        
        search = HyperparameterSearch(n_trials=5, verbose=False)
        self.assertFalse(search.verbose)


class TestIntegrationWithPipeline(unittest.TestCase):
    """测试进度条与流水线集成"""
    
    def test_auto_missing_pipeline_verbose(self):
        """AutoMissingPipeline 应支持 verbose"""
        from core.auto_pipeline import AutoMissingPipeline
        
        pipeline = AutoMissingPipeline(verbose=False)
        self.assertFalse(pipeline.verbose)


class TestTqdmIfAvailable(unittest.TestCase):
    """测试 tqdm 可用时的行为"""
    
    def test_returns_tqdm_when_available(self):
        """tqdm 可用时应返回 tqdm 实例"""
        if _TQDM_AVAILABLE:
            bar = get_progress_bar(total=5)
            self.assertNotIsInstance(bar, DummyTqdm)
        else:
            self.skipTest("tqdm not installed")


if __name__ == '__main__':
    unittest.main()
