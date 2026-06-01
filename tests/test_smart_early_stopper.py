"""
智能早停策略测试
"""
import unittest
import time

from core.smart_early_stopper import (
    TrialEarlyStopper, TrialEarlyStopConfig,
    FoldEarlyStopper, FoldEarlyStopConfig,
    ModelEarlyStopper, ModelEarlyStopConfig,
    SmartEarlyStopper, SmartEarlyStopConfig,
    StopReason
)


class TestTrialEarlyStopper(unittest.TestCase):
    """测试 Trial 级早停"""
    
    def test_disabled(self):
        """禁用时不应触发"""
        stopper = TrialEarlyStopper(TrialEarlyStopConfig(enabled=False))
        self.assertFalse(stopper.should_stop(0.0)[0])
    
    def test_warmup_no_stop(self):
        """warm-up 期间不应触发"""
        stopper = TrialEarlyStopper(TrialEarlyStopConfig(warmup_trials=5))
        self.assertFalse(stopper.should_stop(0.0)[0])
    
    def test_median_strategy_stop(self):
        """中位数策略应正确触发"""
        config = TrialEarlyStopConfig(strategy='median', min_std_factor=0.0)
        stopper = TrialEarlyStopper(config)
        # 填充历史
        for s in [0.5, 0.6, 0.7, 0.8, 0.9]:
            stopper.report(s)
        # 低于 median=0.7 的应触发
        should_stop, reason = stopper.should_stop(0.6)
        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.TRIAL_MEDIAN)
    
    def test_median_strategy_continue(self):
        """中位数策略不应误触发"""
        config = TrialEarlyStopConfig(strategy='median', min_std_factor=0.0)
        stopper = TrialEarlyStopper(config)
        for s in [0.5, 0.6, 0.7, 0.8, 0.9]:
            stopper.report(s)
        should_stop, _ = stopper.should_stop(0.8)
        self.assertFalse(should_stop)
    
    def test_percentile_strategy(self):
        """百分位策略"""
        config = TrialEarlyStopConfig(strategy='percentile', percentile=0.2)
        stopper = TrialEarlyStopper(config)
        for s in range(10):
            stopper.report(float(s) / 10)
        # 低于 20% 分位（约 0.16）的应触发
        should_stop, reason = stopper.should_stop(0.05)
        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.TRIAL_PERCENTILE)
    
    def test_absolute_strategy(self):
        """绝对阈值策略"""
        config = TrialEarlyStopConfig(strategy='absolute', absolute_threshold=0.5)
        stopper = TrialEarlyStopper(config)
        for s in range(5):
            stopper.report(0.6)
        should_stop, reason = stopper.should_stop(0.4)
        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.ABSOLUTE_THRESHOLD)
    
    def test_get_stats(self):
        """统计信息"""
        stopper = TrialEarlyStopper(TrialEarlyStopConfig())
        for s in [0.5, 0.6, 0.7]:
            stopper.report(s)
        stats = stopper.get_stats()
        self.assertEqual(stats['n_scores'], 3)
        self.assertAlmostEqual(stats['mean'], 0.6, places=5)


class TestFoldEarlyStopper(unittest.TestCase):
    """测试 Fold 级早停"""
    
    def test_timeout(self):
        """超时检测"""
        config = FoldEarlyStopConfig(max_fold_time=0.01)
        stopper = FoldEarlyStopper(config)
        stopper.start_fold()
        time.sleep(0.02)
        should_stop, reason = stopper.check()
        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.FOLD_TIMEOUT)
    
    def test_degrade_detection(self):
        """性能下降检测"""
        config = FoldEarlyStopConfig(degrade_patience=2, degrade_threshold=0.01)
        stopper = FoldEarlyStopper(config, direction='maximize')
        stopper.start_fold()
        # 连续下降
        should_stop, _ = stopper.check(1.0)
        self.assertFalse(should_stop)
        should_stop, _ = stopper.check(0.95)
        self.assertFalse(should_stop)
        should_stop, reason = stopper.check(0.90)
        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.FOLD_DEGRADE)
    
    def test_no_degrade_when_improving(self):
        """改善时不应触发"""
        config = FoldEarlyStopConfig(degrade_patience=2, degrade_threshold=0.01)
        stopper = FoldEarlyStopper(config, direction='maximize')
        stopper.start_fold()
        for s in [0.8, 0.85, 0.9, 0.95]:
            should_stop, _ = stopper.check(s)
            self.assertFalse(should_stop)


class TestModelEarlyStopper(unittest.TestCase):
    """测试模型级早停"""
    
    def test_xgb_params(self):
        """XGBoost 参数"""
        stopper = ModelEarlyStopper(ModelEarlyStopConfig(early_stopping_rounds=50))
        params = stopper.get_params('xgb')
        self.assertEqual(params['early_stopping_rounds'], 50)
    
    def test_lgb_params(self):
        """LightGBM 参数"""
        stopper = ModelEarlyStopper(ModelEarlyStopConfig(early_stopping_rounds=30))
        params = stopper.get_params('lgb')
        self.assertEqual(params['early_stopping_rounds'], 30)
    
    def test_torch_params(self):
        """PyTorch 参数"""
        stopper = ModelEarlyStopper(ModelEarlyStopConfig(patience=5, min_delta=0.001))
        params = stopper.get_params('mlp')
        self.assertEqual(params['early_stopping_patience'], 5)
    
    def test_convergence_detection(self):
        """收敛检测"""
        stopper = ModelEarlyStopper(ModelEarlyStopConfig(min_delta=0.01, patience=2))
        stopper.check_convergence(0.5)  # 初始化
        stopper.check_convergence(0.52)  # 改善
        stopper.check_convergence(0.52)  # 无改善
        should_stop, reason, improved = stopper.check_convergence(0.52)
        self.assertTrue(should_stop)
        self.assertEqual(reason, StopReason.MODEL_PLATEAU)
        self.assertFalse(improved)
    
    def test_disabled(self):
        """禁用时不返回参数"""
        stopper = ModelEarlyStopper(ModelEarlyStopConfig(enabled=False))
        self.assertEqual(stopper.get_params('xgb'), {})


class TestSmartEarlyStopper(unittest.TestCase):
    """测试统一早停器"""
    
    def test_trial_check(self):
        """Trial 检查接口"""
        config = SmartEarlyStopConfig(trial=TrialEarlyStopConfig(strategy='median'))
        stopper = SmartEarlyStopper(config)
        for s in [0.5, 0.6, 0.7, 0.8, 0.9]:
            stopper.trial_report(s)
        should_stop, reason = stopper.trial_check(0.1)
        self.assertTrue(should_stop)
    
    def test_fold_check(self):
        """Fold 检查接口"""
        config = SmartEarlyStopConfig(fold=FoldEarlyStopConfig(degrade_patience=2))
        stopper = SmartEarlyStopper(config)
        stopper.fold_start()
        stopper.fold_check(1.0)
        stopper.fold_check(0.9)
        should_stop, reason = stopper.fold_check(0.8)
        self.assertTrue(should_stop)
    
    def test_model_params(self):
        """模型参数接口"""
        stopper = SmartEarlyStopper()
        params = stopper.get_model_params('xgb')
        self.assertIn('early_stopping_rounds', params)
    
    def test_stats(self):
        """统计接口"""
        stopper = SmartEarlyStopper()
        stats = stopper.get_stats()
        self.assertIn('trial', stats)
        self.assertIn('fold', stats)
        self.assertIn('model', stats)


if __name__ == '__main__':
    unittest.main()
