"""
缺失值智能分析引擎 - 单元测试
"""
import os
import sys
import unittest
from core.workspace_manager import get_workspace_manager

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.missing_engine import (
    MissingPatternClassifier, MissingValueHandler,
    MissingPattern, MissingStrategy, StructuralRule,
    ColumnMissingProfile, MissingReport,
    FastMissingClassifier, CacheManager, LazyExecutor
)
from core.auto_pipeline import AutoMissingPipeline, PipelineConfig


class TestCacheManager(unittest.TestCase):
    """测试缓存管理器"""
    
    def test_cache_hit_miss(self):
        cache = CacheManager()
        
        @cache.cached
        def slow_func(x):
            return x * 2
        
        r1 = slow_func(5)
        r2 = slow_func(5)
        r3 = slow_func(10)
        
        self.assertEqual(r1, 10)
        self.assertEqual(r2, 10)  # 缓存命中
        self.assertEqual(r3, 20)
        
        stats = cache.stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 2)


class TestLazyExecutor(unittest.TestCase):
    """测试惰性执行器"""
    
    def test_lazy_execution(self):
        call_count = [0]
        
        def compute():
            call_count[0] += 1
            return 42
        
        lazy = LazyExecutor(compute)
        self.assertEqual(call_count[0], 0)  # 还未执行
        
        r1 = lazy.result
        self.assertEqual(r1, 42)
        self.assertEqual(call_count[0], 1)  # 第一次访问才执行
        
        r2 = lazy.result
        self.assertEqual(r2, 42)
        self.assertEqual(call_count[0], 1)  # 第二次直接返回缓存


class TestMissingPatternClassifier(unittest.TestCase):
    """测试缺失模式分类器"""
    
    def setUp(self):
        self.classifier = MissingPatternClassifier()
    
    def test_no_missing(self):
        df = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        profile = self.classifier.classify(df, 'A')
        self.assertEqual(profile.pattern, MissingPattern.NONE)
        self.assertEqual(profile.missing_count, 0)
    
    def test_target_missing(self):
        """测试目标缺失识别"""
        df = pd.DataFrame({
            'feature': [1, 2, 3, 4, 5, 6],
            'target': [0, 1, 0, np.nan, np.nan, np.nan]  # 后3个是测试集
        })
        profile = self.classifier.classify(df, 'target', target_col='target')
        self.assertEqual(profile.pattern, MissingPattern.TARGET_MISSING)
        self.assertEqual(profile.recommended_strategy, MissingStrategy.PREDICT)
    
    def test_structural_missing(self):
        """测试结构性缺失检测"""
        df = pd.DataFrame({
            'married': ['是', '是', '否', '否', '是', '否', '是', '否', '是', '否'] * 10,
            'spouse_income': [5000, 6000, np.nan, np.nan, 7000, np.nan, 5500, np.nan, 8000, np.nan] * 10
        })
        profile = self.classifier.classify(df, 'spouse_income')
        
        # 应该检测到结构性缺失
        self.assertEqual(profile.pattern, MissingPattern.STRUCTURAL)
        self.assertTrue(len(profile.structural_rules) > 0)
        
        # 主要条件列应该是 married
        rule = profile.structural_rules[0]
        self.assertEqual(rule.condition_col, 'married')
        self.assertEqual(rule.condition_value, '否')
        self.assertGreaterEqual(rule.confidence, 0.90)
    
    def test_true_missing_numeric(self):
        """测试真缺失 - 数值型"""
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.normal(0, 1, 100),
            'B': np.random.normal(0, 1, 100)
        })
        # 随机缺失
        missing_idx = np.random.choice(100, 15, replace=False)
        df.loc[missing_idx, 'B'] = np.nan
        
        profile = self.classifier.classify(df, 'B')
        self.assertEqual(profile.pattern, MissingPattern.TRUE_MISSING)
        # 数值型真缺失应该推荐中位数或均值
        self.assertIn(profile.recommended_strategy, 
                      [MissingStrategy.MEDIAN, MissingStrategy.MEAN])
    
    def test_true_missing_categorical(self):
        """测试真缺失 - 类别型"""
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.choice(['X', 'Y', 'Z'], 200),
            'B': np.random.choice(['P', 'Q', 'R', 'S'], 200)
        })
        missing_idx = np.random.choice(200, 20, replace=False)
        df.loc[missing_idx, 'B'] = np.nan
        
        profile = self.classifier.classify(df, 'B')
        self.assertEqual(profile.pattern, MissingPattern.TRUE_MISSING)
        self.assertEqual(profile.recommended_strategy, MissingStrategy.NEW_CATEGORY)
    
    def test_high_missing_rate(self):
        """测试高缺失率列"""
        df = pd.DataFrame({
            'A': [np.nan] * 85 + list(range(15))  # 85%缺失
        })
        profile = self.classifier.classify(df, 'A')
        self.assertEqual(profile.pattern, MissingPattern.TRUE_MISSING)
        # 高缺失率数值型推荐 FLAG_MEDIAN
        self.assertEqual(profile.recommended_strategy, MissingStrategy.FLAG_MEDIAN)


class TestMissingValueHandler(unittest.TestCase):
    """测试缺失值处理器"""
    
    def setUp(self):
        self.handler = MissingValueHandler()
    
    def test_fill_mean(self):
        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 4.0]})
        result = self.handler.handle(df.copy(), 'A', MissingStrategy.MEAN)
        self.assertEqual(result['A'].isnull().sum(), 0)
        self.assertAlmostEqual(result['A'].iloc[2], 7.0/3, places=5)
    
    def test_fill_median(self):
        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = self.handler.handle(df.copy(), 'A', MissingStrategy.MEDIAN)
        self.assertEqual(result['A'].isnull().sum(), 0)
        self.assertEqual(result['A'].iloc[2], 3.0)  # median of [1,2,4,5] = 3.0
    
    def test_fill_mode(self):
        df = pd.DataFrame({'A': ['X', 'Y', 'X', 'X', np.nan]})
        result = self.handler.handle(df.copy(), 'A', MissingStrategy.MODE)
        self.assertEqual(result['A'].isnull().sum(), 0)
        self.assertEqual(result['A'].iloc[-1], 'X')
    
    def test_fill_new_category(self):
        df = pd.DataFrame({'A': ['X', 'Y', np.nan, 'X']})
        result = self.handler.handle(df.copy(), 'A', MissingStrategy.NEW_CATEGORY)
        self.assertEqual(result['A'].isnull().sum(), 0)
        self.assertEqual(result['A'].iloc[2], '__MISSING__')
    
    def test_fill_flag_median(self):
        df = pd.DataFrame({'A': [1.0, 2.0, np.nan, 4.0]})
        result = self.handler.handle(df.copy(), 'A', MissingStrategy.FLAG_MEDIAN)
        self.assertEqual(result['A'].isnull().sum(), 0)
        self.assertIn('A_is_missing', result.columns)
        self.assertEqual(result['A_is_missing'].iloc[2], 1)
    
    def test_conditional_median(self):
        df = pd.DataFrame({
            'group': ['A', 'A', 'A', 'B', 'B', 'B'],
            'value': [10.0, 12.0, np.nan, 20.0, np.nan, 22.0]
        })
        rule = StructuralRule('group', 'A', 1.0, 3)
        result = self.handler.handle(df.copy(), 'value', MissingStrategy.CONDITIONAL_MEDIAN, rule=rule)
        self.assertEqual(result['value'].isnull().sum(), 0)
        # group=A 的中位数是 11.0
        self.assertAlmostEqual(result.loc[2, 'value'], 11.0, places=5)
    
    def test_fill_interpolate(self):
        df = pd.DataFrame({'A': [1.0, np.nan, 3.0, np.nan, 5.0]})
        result = self.handler.handle(df.copy(), 'A', MissingStrategy.INTERPOLATE)
        self.assertEqual(result['A'].isnull().sum(), 0)
        self.assertAlmostEqual(result['A'].iloc[1], 2.0, places=5)
    
    def test_predict_strategy(self):
        """目标缺失策略应保留NaN"""
        df = pd.DataFrame({'target': [1.0, 2.0, np.nan]})
        result = self.handler.handle(df.copy(), 'target', MissingStrategy.PREDICT)
        self.assertTrue(result['target'].isnull().iloc[-1])


class TestAutoMissingPipeline(unittest.TestCase):
    """测试自动流程"""
    
    def test_full_pipeline_with_target(self):
        """完整流程 - 有目标列"""
        np.random.seed(42)
        n = 200
        
        # 构建训练+测试合并数据
        df = pd.DataFrame({
            'age': np.random.randint(20, 60, n),
            'income': np.random.lognormal(8, 0.5, n),
            'married': np.random.choice(['是', '否'], n),
            'city': np.random.choice(['北京', '上海', '广州'], n),
            'spouse_income': np.nan,  # 先全空
            'target': np.nan
        })
        
        # 前120行是训练集
        train_idx = range(120)
        test_idx = range(120, n)
        
        df.loc[train_idx, 'target'] = np.random.choice([0, 1], 120)
        
        # 结构性缺失：已婚的有配偶收入，未婚的无
        for i in train_idx:
            if df.loc[i, 'married'] == '是':
                df.loc[i, 'spouse_income'] = np.random.lognormal(8, 0.3)
            # 否 -> 保持NaN
        
        # 真缺失：随机缺失一些年龄
        missing_age = np.random.choice(list(train_idx), 10, replace=False)
        df.loc[missing_age, 'age'] = np.nan
        
        # 运行流程
        config = PipelineConfig(
            target_col='target',
            structural_threshold=0.85
        )
        pipeline = AutoMissingPipeline(config)
        train_df, test_df, report = pipeline.run(df)
        
        # 验证
        self.assertIsNotNone(train_df)
        self.assertIsNotNone(test_df)
        self.assertEqual(len(train_df), 120)
        self.assertEqual(len(test_df), 80)
        
        # 训练集目标列应无缺失
        self.assertEqual(train_df['target'].isnull().sum(), 0)
        
        # 测试集目标列应保留NaN（待预测）
        self.assertTrue(test_df['target'].isnull().all())
        
        # 报告应包含spouse_income的结构性缺失
        spouse_profile = report.column_profiles.get('spouse_income')
        if spouse_profile:
            self.assertEqual(spouse_profile.pattern, MissingPattern.STRUCTURAL)
    
    def test_auto_detect_target(self):
        """自动识别目标列"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5, 6],
            'feature2': ['a', 'b', 'c', 'd', 'e', 'f'],
            'target': [0, 1, 0, np.nan, np.nan, np.nan]
        })
        
        pipeline = AutoMissingPipeline()
        train_df, test_df, report = pipeline.run(df)
        
        self.assertEqual(report.target_col, 'target')
        self.assertEqual(len(train_df), 3)
        self.assertEqual(len(test_df), 3)
    
    def test_fast_mode(self):
        """快速模式"""
        np.random.seed(42)
        df = pd.DataFrame({
            'A': np.random.randn(1000),
            'B': np.random.randn(1000),
            'C': np.random.choice(['X', 'Y'], 1000),
            'target': np.concatenate([np.random.choice([0, 1], 700), [np.nan] * 300])
        })
        # 随机缺失
        missing_b = np.random.choice(1000, 50, replace=False)
        df.loc[missing_b, 'B'] = np.nan
        
        config = PipelineConfig(fast_mode=True, sample_size=500)
        pipeline = AutoMissingPipeline(config)
        train_df, test_df, report = pipeline.run(df)
        
        self.assertIsNotNone(train_df)
        self.assertIsNotNone(report)
        self.assertEqual(len(train_df), 700)
    
    def test_drop_high_missing(self):
        """测试高缺失率列删除"""
        df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [np.nan] * 5,  # 100%缺失
            'target': [0, 1, 0, np.nan, np.nan]
        })
        
        config = PipelineConfig(target_col='target', drop_col_threshold=0.90)
        pipeline = AutoMissingPipeline(config)
        train_df, test_df, report = pipeline.run(df)
        
        # B列应被删除
        self.assertNotIn('B', train_df.columns)


class TestStructuralMissingScenarios(unittest.TestCase):
    """测试各种结构性缺失场景"""
    
    def test_child_parent_scenario(self):
        """有子女 → 子女教育支出；无子女 → 教育支出为空"""
        df = pd.DataFrame({
            'has_child': ['有', '有', '无', '无', '有', '无'] * 20,
            'education_expense': [10000, 15000, np.nan, np.nan, 12000, np.nan] * 20
        })
        
        classifier = MissingPatternClassifier(structural_threshold=0.85)
        profile = classifier.classify(df, 'education_expense')
        
        self.assertEqual(profile.pattern, MissingPattern.STRUCTURAL)
        self.assertTrue(any(r.condition_col == 'has_child' for r in profile.structural_rules))
    
    def test_job_type_scenario(self):
        """自由职业 → 公司名称为空；全职 → 有公司名称"""
        df = pd.DataFrame({
            'job_type': ['全职', '全职', '自由职业', '自由职业', '全职'] * 30,
            'company_name': ['A公司', 'B公司', np.nan, np.nan, 'C公司'] * 30
        })
        
        classifier = MissingPatternClassifier(structural_threshold=0.85)
        profile = classifier.classify(df, 'company_name')
        
        self.assertEqual(profile.pattern, MissingPattern.STRUCTURAL)


if __name__ == '__main__':
    unittest.main()
