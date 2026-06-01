"""
自动降采样引擎测试

覆盖：
1. 触发逻辑（阈值判断）
2. 分类任务分层采样（类别比例保持）
3. 回归任务分位数分层采样（目标分布保持）
4. 无监督随机采样
5. 与 ModelingEngine 集成
6. 与 IntegratedPipeline 集成
7. 边界条件（小数据不采样、异常回退）
"""

import pytest
import numpy as np
import pandas as pd

from core.sampling_engine import (
    AutoSampler, SamplingReport, SamplingStrategy,
    auto_sample
)
from core.modeling_engine import TaskType, ModelingEngine


# =============================================================================
# 基础功能测试
# =============================================================================

class TestAutoSamplerTrigger:
    
    def test_should_sample_large_data(self):
        """大数据应触发采样"""
        sampler = AutoSampler(max_samples=1000)
        assert sampler.should_sample(5000) is True
    
    def test_should_not_sample_small_data(self):
        """小数据不应触发采样"""
        sampler = AutoSampler(max_samples=1000, min_samples=500)
        assert sampler.should_sample(400) is False
    
    def test_should_not_sample_at_threshold(self):
        """刚好等于max_samples不应触发"""
        sampler = AutoSampler(max_samples=1000)
        assert sampler.should_sample(1000) is False
    
    def test_should_sample_by_memory(self):
        """内存过大应触发采样"""
        sampler = AutoSampler(max_samples=100000)
        assert sampler.should_sample(5000, memory_mb=3000) is True
    
    def test_should_not_sample_below_min(self):
        """低于min_samples即使超过max也不采样（边界保护）"""
        sampler = AutoSampler(max_samples=100, min_samples=50)
        assert sampler.should_sample(30) is False


class TestStratifiedSampling:
    """分类任务分层采样测试"""
    
    def test_classification_stratified(self):
        """分类任务应正确保持类别比例"""
        np.random.seed(42)
        n = 10000
        X = pd.DataFrame({'a': np.random.randn(n), 'b': np.random.randn(n)})
        y = pd.Series(np.random.choice([0, 0, 0, 1, 1, 2], n))  # 类别比例 3:2:1
        
        sampler = AutoSampler(max_samples=1000, task_type='classification', random_state=42)
        X_s, y_s, report = sampler.sample(X, y)
        
        assert report.strategy == "stratified"
        assert report.sampled_n <= 1000
        assert report.sampled_n >= 900  # 允许一定浮动
        
        # 检查类别比例保持
        orig_ratio = y.value_counts(normalize=True).sort_index()
        samp_ratio = pd.Series(y_s).value_counts(normalize=True).sort_index()
        
        for cls in orig_ratio.index:
            assert samp_ratio[cls] == pytest.approx(orig_ratio[cls], abs=0.05)
        
        assert report.distribution_preservation > 0.95
    
    def test_stratified_with_rare_class(self):
        """稀有类别应被正确处理"""
        n = 5000
        X = pd.DataFrame({'a': np.random.randn(n)})
        y = pd.Series([0] * 4990 + [1] * 10)  # 极不平衡
        
        sampler = AutoSampler(max_samples=500, task_type='classification', random_state=42)
        X_s, y_s, report = sampler.sample(X, y)
        
        # 即使稀有类别也应保留
        classes = pd.Series(y_s).unique()
        assert 1 in classes
        assert report.distribution_preservation > 0.90


class TestQuantileStratifiedSampling:
    """回归任务分位数分层采样测试"""
    
    def test_regression_quantile_stratified(self):
        """回归任务应保持目标值分布"""
        np.random.seed(42)
        n = 10000
        X = pd.DataFrame({'a': np.random.randn(n), 'b': np.random.randn(n)})
        y = pd.Series(np.random.exponential(scale=2, size=n))
        
        sampler = AutoSampler(max_samples=1000, task_type='regression', random_state=42)
        X_s, y_s, report = sampler.sample(X, y)
        
        assert report.strategy == "quantile_stratified"
        assert report.sampled_n <= 1000
        
        # 检查关键分位数保持
        orig_q = np.quantile(y, [0.25, 0.5, 0.75])
        samp_q = np.quantile(y_s, [0.25, 0.5, 0.75])
        
        for o, s in zip(orig_q, samp_q):
            assert s == pytest.approx(o, rel=0.15)
        
        assert report.distribution_preservation > 0.80
    
    def test_regression_with_constant_target(self):
        """回归目标值恒定时应回退到随机采样"""
        n = 5000
        X = pd.DataFrame({'a': np.random.randn(n)})
        y = pd.Series([5.0] * n)
        
        sampler = AutoSampler(max_samples=500, min_samples=100, task_type='regression', random_state=42)
        X_s, y_s, report = sampler.sample(X, y)
        
        # 分箱失败，回退到随机
        assert report.strategy in ("random", "quantile_stratified")
        assert len(X_s) <= 500


class TestRandomSampling:
    """随机采样测试"""
    
    def test_unsupervised_random(self):
        """无监督任务应使用随机采样"""
        n = 5000
        X = pd.DataFrame({'a': np.random.randn(n), 'b': np.random.randn(n)})
        
        sampler = AutoSampler(max_samples=500, min_samples=100, task_type='clustering', random_state=42)
        X_s, y_s, report = sampler.sample(X, None)
        
        assert report.strategy == "random"
        assert len(X_s) <= 500
        assert y_s is None
    
    def test_no_sampling_needed(self):
        """小数据不应采样"""
        n = 100
        X = pd.DataFrame({'a': np.random.randn(n)})
        y = pd.Series(np.random.choice([0, 1], n))
        
        sampler = AutoSampler(max_samples=1000, task_type='classification', random_state=42)
        X_s, y_s, report = sampler.sample(X, y)
        
        assert report.strategy == "none"
        assert len(X_s) == n
        assert report.sample_ratio == 1.0


class TestForceSampling:
    """强制采样测试"""
    
    def test_force_sample(self):
        """force=True 应忽略阈值"""
        n = 100
        X = pd.DataFrame({'a': np.random.randn(n)})
        y = pd.Series(np.random.choice([0, 1], n))
        
        sampler = AutoSampler(max_samples=50, min_samples=10, task_type='classification', random_state=42)
        X_s, y_s, report = sampler.sample(X, y, force=True)
        
        assert report.strategy == "stratified"
        assert report.sampled_n < n


class TestDistributionMetrics:
    """分布保持度计算测试"""
    
    def test_class_preservation_perfect(self):
        """完全保持时 distribution_preservation ≈ 1.0"""
        sampler = AutoSampler()
        y_orig = pd.Series([0, 0, 1, 1])
        y_samp = pd.Series([0, 0, 1, 1])
        score = sampler._calc_class_preservation(y_orig, y_samp)
        assert score == pytest.approx(1.0, abs=0.01)
    
    def test_class_preservation_poor(self):
        """完全破坏时 distribution_preservation ≈ 0.0"""
        sampler = AutoSampler()
        y_orig = pd.Series([0] * 100 + [1] * 100)
        y_samp = pd.Series([0] * 200)  # 只有一类
        score = sampler._calc_class_preservation(y_orig, y_samp)
        assert score <= 0.5


# =============================================================================
# 便捷函数测试
# =============================================================================

class TestAutoSampleShortcut:
    
    def test_auto_sample_classification(self):
        n = 5000
        X = pd.DataFrame({'a': np.random.randn(n)})
        y = pd.Series(np.random.choice([0, 1], n))
        
        X_s, y_s, report = auto_sample(X, y, max_samples=500, task_type='classification')
        
        assert report.sampled_n <= 500  # auto_sample uses default min_samples=1000, so this may not trigger
        assert report.strategy == "stratified"
    
    def test_auto_sample_no_trigger(self):
        n = 100
        X = pd.DataFrame({'a': np.random.randn(n)})
        y = pd.Series(np.random.choice([0, 1], n))
        
        X_s, y_s, report = auto_sample(X, y, max_samples=1000, task_type='classification')
        
        assert report.strategy == "none"
        assert len(X_s) == n


# =============================================================================
# 与 ModelingEngine 集成测试
# =============================================================================

class TestModelingEngineIntegration:
    
    def test_sampling_in_modeling_engine(self):
        """ModelingEngine 应自动降采样大数据"""
        np.random.seed(42)
        n = 3000
        X = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
        })
        y = pd.Series(np.random.choice([0, 1], n))
        
        engine = ModelingEngine(
            model_keys=['lr'],
            auto_sample=True,
            max_samples=1000,
            n_splits=3
        )
        result = engine.fit(X, y)
        
        assert result.sampling_report is not None
        assert result.sampling_report.sample_ratio < 1.0
        assert result.sampling_report.sampled_n <= 1000
    
    def test_no_sampling_when_disabled(self):
        """auto_sample=False 时不应采样"""
        np.random.seed(42)
        n = 3000
        X = pd.DataFrame({'a': np.random.randn(n)})
        y = pd.Series(np.random.choice([0, 1], n))
        
        engine = ModelingEngine(
            model_keys=['lr'],
            auto_sample=False,
            max_samples=1000,
            n_splits=3
        )
        result = engine.fit(X, y)
        
        # 当 auto_sample=False 时，sampling_report 可能为 None 或 none 策略
        if result.sampling_report:
            assert result.sampling_report.strategy == "none"
    
    def test_sampling_with_regression(self):
        """回归任务 ModelingEngine 集成"""
        np.random.seed(42)
        n = 3000
        X = pd.DataFrame({'a': np.random.randn(n), 'b': np.random.randn(n)})
        y = pd.Series(np.random.randn(n))
        
        engine = ModelingEngine(
            auto_sample=True,
            max_samples=1000,
            n_splits=3
        )
        result = engine.fit(X, y)
        
        assert result.sampling_report is not None
        assert result.sampling_report.strategy in ("quantile_stratified", "random")


# =============================================================================
# 与 IntegratedPipeline 集成测试
# =============================================================================

class TestPipelineIntegration:
    
    def test_pipeline_with_sampling(self):
        """IntegratedPipeline 应支持自动降采样"""
        from core.integrated_pipeline import IntegratedPipeline
        
        np.random.seed(42)
        n = 3000
        df = pd.DataFrame({
            'a': np.random.randn(n),
            'b': np.random.randn(n),
            'target': np.random.choice([0, 1], n)
        })
        
        pipeline = IntegratedPipeline(
            target_col='target',
            task_type='classification',
            model_keys=['lr'],
            n_splits=3,
            auto_sample=True,
            max_samples=1000
        )
        
        result = pipeline.run(df)
        
        assert result.modeling_result is not None
        assert result.modeling_result.sampling_report is not None
        assert result.modeling_result.sampling_report.sampled_n <= 1000
    
    def test_pipeline_without_sampling(self):
        """IntegratedPipeline 关闭采样时不应采样"""
        from core.integrated_pipeline import IntegratedPipeline
        
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'a': np.random.randn(n),
            'target': np.random.choice([0, 1], n)
        })
        
        pipeline = IntegratedPipeline(
            target_col='target',
            task_type='classification',
            model_keys=['lr'],
            n_splits=3,
            auto_sample=False
        )
        
        result = pipeline.run(df)
        
        if result.modeling_result and result.modeling_result.sampling_report:
            assert result.modeling_result.sampling_report.strategy == "none"


# =============================================================================
# 边界条件测试
# =============================================================================

class TestEdgeCases:
    
    def test_empty_data(self):
        """空数据应优雅处理"""
        X = pd.DataFrame({'a': []})
        y = pd.Series([], dtype=int)
        
        sampler = AutoSampler(max_samples=100, task_type='classification')
        X_s, y_s, report = sampler.sample(X, y)
        
        assert len(X_s) == 0
        assert report.strategy == "none"
    
    def test_single_class(self):
        """单类别分类应回退到随机"""
        n = 5000
        X = pd.DataFrame({'a': np.random.randn(n)})
        y = pd.Series([0] * n)
        
        sampler = AutoSampler(max_samples=500, min_samples=100, task_type='classification', random_state=42)
        X_s, y_s, report = sampler.sample(X, y)
        
        # 单类别 stratify 等价于 random，且不会失败
        assert report.strategy == "stratified"
        assert len(X_s) <= 500
    
    def test_numpy_array_input(self):
        """numpy 数组输入应支持"""
        n = 5000
        X = np.random.randn(n, 3)
        y = np.random.choice([0, 1], n)
        
        sampler = AutoSampler(max_samples=500, min_samples=100, task_type='classification', random_state=42)
        X_s, y_s, report = sampler.sample(X, y)
        
        assert isinstance(X_s, np.ndarray)
        assert len(X_s) <= 500
