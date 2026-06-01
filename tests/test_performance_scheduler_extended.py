"""
PerformanceScheduler 扩展测试
"""
import os
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.performance_scheduler import (
    PerformanceScheduler, ExecutionPlan, StrategyLevel,
    DataScaleEvaluator, HardwareDetector, HardwareProfile,
    DataScaleMetrics, auto_schedule
)


class TestHardwareDetectorExtended(unittest.TestCase):
    """测试硬件探测器"""
    
    def test_detect_returns_profile(self):
        hw = HardwareDetector.detect()
        self.assertIsInstance(hw, HardwareProfile)
        self.assertGreater(hw.cpu_count, 0)
        self.assertGreater(hw.memory_total_gb, 0)
    
    def test_detect_gpu_with_pynvml(self):
        try:
            import pynvml
            has_pynvml = True
        except ImportError:
            has_pynvml = False
        
        if has_pynvml:
            has_gpu, count, names, mems = HardwareDetector._detect_gpu()
            self.assertIsInstance(has_gpu, bool)
            self.assertIsInstance(count, int)
            self.assertIsInstance(names, list)
            self.assertIsInstance(mems, list)
    
    def test_detect_cached(self):
        # HardwareDetector.detect is @lru_cache
        hw1 = HardwareDetector.detect()
        hw2 = HardwareDetector.detect()
        self.assertEqual(hw1.cpu_count, hw2.cpu_count)


class TestDataScaleEvaluatorExtended(unittest.TestCase):
    """测试数据规模评估器"""
    
    def test_small_data_tier(self):
        df = pd.DataFrame({'a': [1, 2, 3]})
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertEqual(metrics.size_tier, 'small')
    
    def test_small_data(self):
        df = pd.DataFrame(np.random.randn(500, 5))
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertEqual(metrics.size_tier, 'small')
    
    def test_medium_data(self):
        df = pd.DataFrame(np.random.randn(50000, 20))
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertIn(metrics.size_tier, ['medium', 'large'])
    
    def test_large_data(self):
        df = pd.DataFrame(np.random.randn(500000, 50))
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertEqual(metrics.size_tier, 'large')
    
    def test_huge_data(self):
        df = pd.DataFrame(np.random.randn(10000000, 2))
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertEqual(metrics.size_tier, 'huge')
    
    def test_wide_data(self):
        df = pd.DataFrame(np.random.randn(100, 500))
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertGreater(metrics.n_cols, 400)
    
    def test_complexity_score(self):
        df = pd.DataFrame(np.random.randn(100000, 100))
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertGreater(metrics.complexity_score, 0)
        self.assertLessEqual(metrics.complexity_score, 100)
    
    def test_with_datetime_columns(self):
        df = pd.DataFrame({
            'a': range(100),
            'dt': pd.date_range('2020-01-01', periods=100)
        })
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertEqual(metrics.n_rows, 100)
    
    def test_with_text_columns(self):
        df = pd.DataFrame({
            'a': range(100),
            'txt': ['hello world'] * 100
        })
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertEqual(metrics.n_rows, 100)


class TestPerformanceSchedulerExtended(unittest.TestCase):
    """测试性能调度器"""
    
    def test_small_data_strategy(self):
        df = pd.DataFrame({'a': range(500), 'target': [0, 1] * 250})
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        self.assertIsInstance(plan, ExecutionPlan)
        self.assertIn(plan.strategy, [StrategyLevel.STANDARD, StrategyLevel.FAST])
    
    def test_medium_data_fast(self):
        df = pd.DataFrame(np.random.randn(200000, 20))
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        self.assertIn(plan.strategy, [StrategyLevel.FAST, StrategyLevel.ULTRA])
    
    def test_large_data_ultra(self):
        df = pd.DataFrame(np.random.randn(2000000, 50))
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        self.assertIn(plan.strategy, [StrategyLevel.FAST, StrategyLevel.ULTRA])
    
    def test_user_preference_speed_first(self):
        df = pd.DataFrame({'a': range(500), 'target': [0, 1] * 250})
        scheduler = PerformanceScheduler(user_preference=StrategyLevel.FAST)
        plan = scheduler.schedule(df)
        self.assertEqual(plan.strategy, StrategyLevel.FAST)
    
    def test_user_preference_accuracy_first(self):
        df = pd.DataFrame({'a': range(500), 'target': [0, 1] * 250})
        scheduler = PerformanceScheduler(user_preference=StrategyLevel.STANDARD)
        plan = scheduler.schedule(df)
        self.assertEqual(plan.strategy, StrategyLevel.STANDARD)
    
    def test_standard_plan_fields(self):
        df = pd.DataFrame({'a': range(500), 'target': [0, 1] * 250})
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        self.assertIsNotNone(plan.n_jobs)
        self.assertIsNotNone(plan.cv_folds)
        self.assertIsNotNone(plan.max_models)
        self.assertIsNotNone(plan.hyperparameter_trials)
    
    def test_fast_plan_sampling(self):
        df = pd.DataFrame(np.random.randn(600000, 10))
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        if plan.strategy == StrategyLevel.FAST:
            self.assertIsNotNone(plan.sample_size)
            self.assertGreater(plan.sample_size, 0)
    
    def test_ultra_plan_chunking(self):
        df = pd.DataFrame(np.random.randn(6000000, 10))
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        if plan.strategy == StrategyLevel.ULTRA:
            self.assertIsNotNone(plan.chunk_size)
            self.assertGreater(plan.chunk_size, 0)
    
    def test_recommendation_text(self):
        df = pd.DataFrame({'a': range(500), 'target': [0, 1] * 250})
        scheduler = PerformanceScheduler()
        scheduler.schedule(df)
        text = scheduler.get_recommendation_text()
        self.assertIn('性能调度决策报告', text)
    
    def test_recommendation_text_before_schedule(self):
        scheduler = PerformanceScheduler()
        text = scheduler.get_recommendation_text()
        self.assertIn('尚未执行调度', text)
    
    def test_auto_schedule(self):
        df = pd.DataFrame({'a': range(500), 'target': [0, 1] * 250})
        plan = auto_schedule(df)
        self.assertIsInstance(plan, ExecutionPlan)


class TestStrategyLevel(unittest.TestCase):
    """测试策略级别枚举"""
    
    def test_values(self):
        self.assertEqual(StrategyLevel.FAST.value, 'fast')
        self.assertEqual(StrategyLevel.STANDARD.value, 'standard')
        self.assertEqual(StrategyLevel.ULTRA.value, 'ultra')
    
    def test_repr(self):
        self.assertEqual(repr(StrategyLevel.FAST), "StrategyLevel.FAST")


if __name__ == '__main__':
    unittest.main()
