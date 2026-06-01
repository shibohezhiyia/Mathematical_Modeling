"""
性能调度 + 并行建模 + 加速层 测试
"""
import os
import sys
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.performance_scheduler import (
    PerformanceScheduler, ExecutionPlan, StrategyLevel,
    DataScaleEvaluator, HardwareDetector, auto_schedule
)
from core.accelerators import (
    ParallelEngine, GPUManager, get_gpu_manager,
    optimize_memory, get_system_info
)
from core.parallel_modeling import (
    ParallelModelingEngine, ModelRegistry, Metrics, quick_model
)
from core.integrated_pipeline import IntegratedPipeline


class TestDataScaleEvaluator(unittest.TestCase):
    """测试数据规模评估"""
    
    def test_small_data(self):
        df = pd.DataFrame({
            'A': range(100),
            'B': range(100),
            'C': ['x'] * 100
        })
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertEqual(metrics.size_tier, 'small')
        self.assertLess(metrics.complexity_score, 30)
    
    def test_large_data(self):
        df = pd.DataFrame(np.random.randn(200000, 50))
        metrics = DataScaleEvaluator.evaluate(df)
        self.assertEqual(metrics.size_tier, 'large')
        self.assertGreater(metrics.complexity_score, 20)  # 对数缩放后约28分
    
    def test_huge_data(self):
        # 用小规模模拟大数据（避免测试机内存不足）
        df = pd.DataFrame(np.random.randn(1000, 10))
        df.attrs['simulated_rows'] = 20000000
        metrics = DataScaleEvaluator.evaluate(df)
        # 手动覆盖行数测试分级逻辑
        metrics.n_rows = 20000000
        self.assertEqual(metrics.size_tier, 'huge')


class TestHardwareDetector(unittest.TestCase):
    """测试硬件探测"""
    
    def test_detect(self):
        hw = HardwareDetector.detect()
        self.assertGreater(hw.cpu_count, 0)
        self.assertGreater(hw.memory_total_gb, 0)
        self.assertIsInstance(hw.has_gpu, bool)
    
    def test_system_info(self):
        info = get_system_info()
        self.assertIn('cpu_count', info)
        self.assertIn('gpu_available', info)


class TestPerformanceScheduler(unittest.TestCase):
    """测试性能调度器"""
    
    def test_small_standard(self):
        df = pd.DataFrame({
            'A': range(500),
            'B': range(500),
            'target': [0, 1] * 250
        })
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        # 小数据通常选 STANDARD，但如果硬件强也可能选 FAST，都是合理的
        self.assertIn(plan.strategy, [StrategyLevel.STANDARD, StrategyLevel.FAST])
    
    def test_large_fast(self):
        df = pd.DataFrame(np.random.randn(200000, 30))
        scheduler = PerformanceScheduler()
        plan = scheduler.schedule(df)
        self.assertIn(plan.strategy, [StrategyLevel.FAST, StrategyLevel.ULTRA])
        self.assertIsNotNone(plan.sample_size)
    
    def test_user_override(self):
        df = pd.DataFrame({'A': range(100)})
        scheduler = PerformanceScheduler(user_preference=StrategyLevel.ULTRA)
        plan = scheduler.schedule(df)
        self.assertEqual(plan.strategy, StrategyLevel.ULTRA)
    
    def test_auto_schedule(self):
        df = pd.DataFrame(np.random.randn(50000, 20))
        plan = auto_schedule(df)
        self.assertIsNotNone(plan.strategy)


class TestAccelerators(unittest.TestCase):
    """测试加速层"""
    
    def test_parallel_engine_map(self):
        engine = ParallelEngine(n_jobs=2, backend='thread')
        result = engine.map(lambda x: x ** 2, [1, 2, 3, 4, 5])
        self.assertEqual(result, [1, 4, 9, 16, 25])
    
    def test_optimize_memory(self):
        df = pd.DataFrame({
            'A': [1, 2, 3],  # int64 -> int8
            'B': [1.0, 2.0, 3.0],  # float64 -> float32
            'C': ['x', 'y', 'z']  # object -> category if repeated
        })
        optimized = optimize_memory(df, verbose=False)
        self.assertEqual(optimized['A'].dtype, np.uint8)
        self.assertEqual(optimized['B'].dtype, np.float32)
    
    def test_gpu_manager(self):
        gpu = get_gpu_manager()
        self.assertIsInstance(gpu.available, bool)


class TestParallelModeling(unittest.TestCase):
    """测试并行建模"""
    
    def setUp(self):
        np.random.seed(42)
        self.X_train = pd.DataFrame(np.random.randn(500, 10), columns=[f'f{i}' for i in range(10)])
        self.y_train = np.random.choice([0, 1], 500)
        self.X_test = pd.DataFrame(np.random.randn(100, 10), columns=[f'f{i}' for i in range(10)])
    
    def test_metrics(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        proba = np.array([0.1, 0.9, 0.2, 0.4])
        
        metrics = Metrics.evaluate(y_true, y_pred, 'classification', proba)
        self.assertIn('accuracy', metrics)
        self.assertIn('auc', metrics)
        self.assertGreaterEqual(metrics['accuracy'], 0)
    
    def test_model_registry(self):
        Registry = ModelRegistry
        Registry._init()
        models = Registry.get_available_models('classification')
        self.assertGreater(len(models), 0)
    
    def test_engine_fit_classification(self):
        """测试分类任务并行建模"""
        engine = ParallelModelingEngine(
            task_type='classification',
            plan=ExecutionPlan(strategy=StrategyLevel.FAST, n_jobs=2, 
                              cv_folds=3, hyperparameter_trials=0,
                              max_models=2, use_gpu=False)
        )
        engine.fit(self.X_train, self.y_train, self.X_test)
        
        self.assertGreater(len(engine.results), 0)
        self.assertIsNotNone(engine.leaderboard)
        
        lb = engine.get_leaderboard()
        self.assertGreater(len(lb), 0)
    
    def test_engine_predict_blend(self):
        """测试模型融合预测"""
        engine = ParallelModelingEngine(
            task_type='classification',
            plan=ExecutionPlan(strategy=StrategyLevel.FAST, n_jobs=2,
                              cv_folds=3, hyperparameter_trials=0,
                              max_models=2, use_gpu=False)
        )
        engine.fit(self.X_train, self.y_train, self.X_test)
        
        # 平均融合
        pred_avg = engine.predict(self.X_test, blend_method='average')
        self.assertEqual(len(pred_avg), len(self.X_test))
        
        # 加权融合
        pred_weighted = engine.predict(self.X_test, blend_method='weighted')
        self.assertEqual(len(pred_weighted), len(self.X_test))
    
    def test_engine_regression(self):
        """测试回归任务"""
        y_reg = np.random.randn(500)
        engine = ParallelModelingEngine(
            task_type='regression',
            plan=ExecutionPlan(strategy=StrategyLevel.FAST, n_jobs=2,
                              cv_folds=3, hyperparameter_trials=0,
                              max_models=2, use_gpu=False)
        )
        engine.fit(self.X_train, y_reg, self.X_test)
        
        self.assertGreater(len(engine.results), 0)
        pred = engine.predict(self.X_test)
        self.assertEqual(len(pred), len(self.X_test))
    
    def test_quick_model(self):
        """测试快速建模接口"""
        from core.performance_scheduler import ExecutionPlan, StrategyLevel
        plan = ExecutionPlan(
            strategy=StrategyLevel.FAST, n_jobs=2, cv_folds=2,
            hyperparameter_trials=0, max_models=2, use_gpu=False
        )
        pred, engine = quick_model(
            self.X_train, self.y_train, self.X_test,
            task_type='classification', plan=plan,
            return_engine=True
        )
        self.assertIsNotNone(pred)
        self.assertEqual(len(pred), len(self.X_test))


class TestIntegratedPipeline(unittest.TestCase):
    """测试集成流水线"""
    
    def test_full_pipeline_classification(self):
        """分类任务完整流程"""
        np.random.seed(42)
        n = 400
        
        df = pd.DataFrame({
            'f1': np.random.randn(n),
            'f2': np.random.randn(n),
            'f3': np.random.choice(['A', 'B', 'C'], n),
            'target': np.nan
        })
        # 前300为训练
        df.loc[:299, 'target'] = np.random.choice([0, 1], 300)
        
        pipeline = IntegratedPipeline(strategy_preference='fast', target_col='target')
        result = pipeline.run(df)
        
        self.assertIsNotNone(result.train_df)
        self.assertIsNotNone(result.test_df)
        self.assertIsNotNone(result.predictions)
        self.assertEqual(len(result.predictions), 100)
        self.assertIsNotNone(result.leaderboard)
    
    def test_full_pipeline_regression(self):
        """回归任务完整流程"""
        np.random.seed(42)
        n = 400
        
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
            'y': np.nan
        })
        df.loc[:299, 'y'] = np.random.randn(300)
        
        pipeline = IntegratedPipeline(
            strategy_preference='fast',
            target_col='y',
            task_type='regression'
        )
        result = pipeline.run(df)
        
        self.assertIsNotNone(result.predictions)
        self.assertEqual(len(result.predictions), 100)


if __name__ == '__main__':
    unittest.main()
