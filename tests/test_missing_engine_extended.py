"""
MissingEngine 扩展测试
"""
import os
import sys
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.missing_engine import (
    CacheManager, LazyExecutor, MissingPatternClassifier,
    MissingValueHandler, FastMissingClassifier,
    MissingStrategy, MissingPattern
)
from core.auto_pipeline import AutoMissingPipeline, PipelineConfig


class TestCacheManagerExtended(unittest.TestCase):
    """测试缓存管理器"""
    
    def test_basic_cache(self):
        cache = CacheManager()
        cache.set('key1', {'value': 1})
        self.assertEqual(cache.get('key1'), {'value': 1})
    
    def test_cache_miss(self):
        cache = CacheManager()
        self.assertIsNone(cache.get('nonexistent'))
    
    def test_cached_decorator(self):
        cache = CacheManager()
        call_count = [0]
        @cache.cached
        def compute(x):
            call_count[0] += 1
            return x * 2
        r1 = compute(5)
        r2 = compute(5)
        self.assertEqual(r1, 10)
        self.assertEqual(r2, 10)
        self.assertEqual(call_count[0], 1)


class TestLazyExecutorExtended(unittest.TestCase):
    """测试懒执行器"""
    
    def test_lazy_execution(self):
        called = [False]
        def fn(a, b):
            called[0] = True
            return a + b
        executor = LazyExecutor(fn, 2, 3)
        self.assertFalse(called[0])
        result = executor.result
        self.assertTrue(called[0])
        self.assertEqual(result, 5)
    
    def test_lazy_invalidate(self):
        executor = LazyExecutor(lambda: 42)
        r1 = executor.result
        executor.invalidate()
        r2 = executor.result
        self.assertEqual(r1, 42)
        self.assertEqual(r2, 42)


class TestMissingPatternClassifierExtended(unittest.TestCase):
    """测试缺失模式分类器"""
    
    def test_no_missing(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        classifier = MissingPatternClassifier()
        profile = classifier.classify(df, 'a')
        self.assertEqual(profile.pattern, MissingPattern.NONE)
    
    def test_high_missing_rate(self):
        df = pd.DataFrame({'a': [1.0, np.nan, np.nan, np.nan], 'b': [1, 2, 3, 4]})
        classifier = MissingPatternClassifier()
        profile = classifier.classify(df, 'a')
        self.assertNotEqual(profile.pattern, MissingPattern.NONE)
        self.assertGreater(profile.missing_rate, 0.5)
    
    def test_target_missing(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'target': [0, np.nan, 1]})
        classifier = MissingPatternClassifier()
        profile = classifier.classify(df, 'target', target_col='target')
        self.assertEqual(profile.pattern, MissingPattern.TARGET_MISSING)


class TestMissingValueHandlerExtended(unittest.TestCase):
    """测试缺失值处理器"""
    
    def test_handle_mean(self):
        df = pd.DataFrame({'a': [1.0, 2.0, np.nan]})
        handler = MissingValueHandler()
        result = handler.handle(df, 'a', MissingStrategy.MEAN)
        self.assertFalse(result['a'].isnull().any())
    
    def test_handle_median(self):
        df = pd.DataFrame({'a': [1.0, 2.0, np.nan]})
        handler = MissingValueHandler()
        result = handler.handle(df, 'a', MissingStrategy.MEDIAN)
        self.assertFalse(result['a'].isnull().any())
    
    def test_handle_mode(self):
        df = pd.DataFrame({'a': ['X', 'Y', 'X', np.nan]})
        handler = MissingValueHandler()
        result = handler.handle(df, 'a', MissingStrategy.MODE)
        self.assertFalse(result['a'].isnull().any())
    
    def test_handle_new_category(self):
        df = pd.DataFrame({'a': ['X', 'Y', np.nan]})
        handler = MissingValueHandler()
        result = handler.handle(df, 'a', MissingStrategy.NEW_CATEGORY)
        self.assertFalse(result['a'].isnull().any())
    
    def test_handle_flag_median(self):
        df = pd.DataFrame({'a': [1.0, 2.0, np.nan]})
        handler = MissingValueHandler()
        result = handler.handle(df, 'a', MissingStrategy.FLAG_MEDIAN)
        self.assertFalse(result['a'].isnull().any())
        self.assertIn('a_is_missing', result.columns)
    
    def test_handle_interpolate(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        handler = MissingValueHandler()
        result = handler.handle(df, 'a', MissingStrategy.INTERPOLATE)
        self.assertFalse(result['a'].isnull().any())
    
    def test_handle_none(self):
        df = pd.DataFrame({'a': [1.0, np.nan]})
        handler = MissingValueHandler()
        result = handler.handle(df, 'a', MissingStrategy.NONE)
        self.assertTrue(result['a'].isnull().any())
    
    def test_handle_unknown_strategy(self):
        df = pd.DataFrame({'a': [1.0, np.nan]})
        handler = MissingValueHandler()
        # Test with a strategy not in handler_map
        result = handler.handle(df, 'a', MissingStrategy.PREDICT)
        self.assertIsNotNone(result)


class TestFastMissingClassifierExtended(unittest.TestCase):
    """测试快速缺失分类器"""
    
    def test_classify_all_no_missing(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        classifier = FastMissingClassifier()
        profiles = classifier.classify_all(df)
        self.assertIsInstance(profiles, dict)
    
    def test_classify_all_with_missing(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [4, 5, 6]})
        classifier = FastMissingClassifier()
        profiles = classifier.classify_all(df)
        self.assertIn('a', profiles)
        self.assertNotEqual(profiles['a'].pattern, MissingPattern.NONE)


class TestAutoMissingPipelineExtended(unittest.TestCase):
    """测试自动缺失处理流水线"""
    
    def test_full_pipeline_numeric(self):
        df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan, 4.0],
            'b': [5.0, np.nan, 7.0, 8.0],
            'target': [0, 1, 0, 1]
        })
        config = PipelineConfig(target_col='target', fast_mode=True)
        pipeline = AutoMissingPipeline(config)
        train, test, report = pipeline.run(df)
        self.assertFalse(train.isnull().any().any())
        self.assertIsNotNone(report)
    
    def test_full_pipeline_categorical(self):
        df = pd.DataFrame({
            'a': ['X', 'Y', np.nan, 'X'],
            'b': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]
        })
        config = PipelineConfig(target_col='target', fast_mode=True)
        pipeline = AutoMissingPipeline(config)
        train, test, report = pipeline.run(df)
        self.assertFalse(train.isnull().any().any())
    
    def test_drop_high_missing(self):
        df = pd.DataFrame({
            'a': [1.0, np.nan, np.nan, np.nan],
            'b': [1.0, 2.0, 3.0, 4.0],
            'target': [0, 1, 0, 1]
        })
        config = PipelineConfig(target_col='target', drop_col_threshold=0.5, fast_mode=True)
        pipeline = AutoMissingPipeline(config)
        train, test, report = pipeline.run(df)
        self.assertNotIn('a', train.columns)
    
    def test_no_target(self):
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [4.0, 5.0, np.nan]
        })
        config = PipelineConfig(fast_mode=True)
        pipeline = AutoMissingPipeline(config)
        train, test, report = pipeline.run(df)
        self.assertFalse(train.isnull().any().any())
    
    def test_fast_mode(self):
        df = pd.DataFrame({
            'a': [1.0, np.nan] * 100,
            'target': [0, 1] * 100
        })
        config = PipelineConfig(target_col='target', fast_mode=True)
        pipeline = AutoMissingPipeline(config)
        train, test, report = pipeline.run(df)
        self.assertFalse(train.isnull().any().any())
    
    def test_get_train_test(self):
        df = pd.DataFrame({
            'a': [1.0, 2.0, np.nan],
            'target': [0, 1, 0]
        })
        config = PipelineConfig(target_col='target', fast_mode=True)
        pipeline = AutoMissingPipeline(config)
        pipeline.run(df)
        train, test = pipeline.get_train_test()
        self.assertIsNotNone(train)
    
    def test_print_report(self):
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'target': [0, 1, 0]
        })
        config = PipelineConfig(target_col='target', fast_mode=True)
        pipeline = AutoMissingPipeline(config)
        pipeline.run(df)
        pipeline.print_report()  # Should not raise


if __name__ == '__main__':
    unittest.main()
