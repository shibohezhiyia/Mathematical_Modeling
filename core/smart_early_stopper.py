"""
智能早停策略

提供多层早停机制：
1. Trial 级早停：基于历史分数分布自动剪枝差 trial
2. Fold 级早停：CV 过程中提前终止表现极差的 fold
3. 模型级早停：为树模型和深度学习模型传递早停参数

所有策略可独立配置和组合使用。
"""
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from scipy import stats

# ModelEarlyStopper.get_params 支持的模型 key 集合
# 3 个 GBDT 系列（xgb/lgb/cat）共享 early_stopping_rounds；
# 5 个 PyTorch 系列（mlp/cnn/lstm/gru/torch）共享 patience/min_delta/restore_best
_GBDT_KEYS = frozenset(('xgb', 'xgboost', 'lgb', 'lightgbm', 'cat', 'catboost'))
_TORCH_KEYS = frozenset(('mlp', 'cnn', 'lstm', 'gru', 'torch'))


class StopReason(Enum):
    """早停原因"""
    NONE = "未触发"
    TRIAL_MEDIAN = "trial分数低于中位数阈值"
    TRIAL_PERCENTILE = "trial分数低于百分位阈值"
    FOLD_TIMEOUT = "fold超时"
    FOLD_DEGRADE = "fold性能持续下降"
    MODEL_CONVERGE = "模型收敛"
    MODEL_PLATEAU = "模型分数平台期"
    ABSOLUTE_THRESHOLD = "绝对阈值"


@dataclass
class TrialEarlyStopConfig:
    """Trial 级早停配置"""
    enabled: bool = True
    warmup_trials: int = 5           # 前 N 次不早停
    strategy: str = 'median'         # 'median', 'percentile', 'absolute'
    percentile: float = 0.25         # 低于此百分位则早停
    min_std_factor: float = 1.0      # 低于 median - factor * std 则早停
    absolute_threshold: Optional[float] = None  # 绝对阈值


@dataclass
class FoldEarlyStopConfig:
    """Fold 级早停配置"""
    enabled: bool = True
    max_fold_time: Optional[float] = None  # 单个 fold 最大耗时（秒）
    degrade_patience: int = 2         # 连续 N 次下降则早停
    degrade_threshold: float = 0.05   # 下降比例阈值


@dataclass
class ModelEarlyStopConfig:
    """模型级早停配置"""
    enabled: bool = True
    early_stopping_rounds: int = 30   # 树模型早停轮数
    min_delta: float = 1e-4           # 最小改进阈值
    patience: int = 10                # 耐心值
    restore_best: bool = True         # 是否恢复最佳权重


@dataclass
class SmartEarlyStopConfig:
    """完整早停配置"""
    trial: TrialEarlyStopConfig = field(default_factory=TrialEarlyStopConfig)
    fold: FoldEarlyStopConfig = field(default_factory=FoldEarlyStopConfig)
    model: ModelEarlyStopConfig = field(default_factory=ModelEarlyStopConfig)
    direction: str = 'maximize'       # 'maximize' 或 'minimize'
    # 新增：多目标早停配置（精度+速度权衡）
    multi_objective: bool = False
    time_weight: float = 0.3        # 时间在综合评分中的权重
    score_weight: float = 0.7         # 分数在综合评分中的权重
    max_time_per_trial: Optional[float] = None  # 每个 trial 最大时间（秒）


class TrialEarlyStopper:
    """
    Trial 级早停器
    
    基于历史 trial 的分数分布，判断当前 trial 是否值得继续。
    
    策略:
    - median: 当前分数低于历史 median - factor * std 时早停
    - percentile: 当前分数低于历史 P-th 百分位时早停
    - absolute: 当前分数低于绝对阈值时早停
    """
    
    def __init__(self, config: TrialEarlyStopConfig,
                 direction: str = 'maximize') -> None:
        self.config = config
        self.direction = direction
        self.scores: List[float] = []
        self._stopped_trials: int = 0
        # 新增：概率性早停的历史记录
        self._prob_history: List[float] = []
    
    def _is_better(self, current: float, threshold: float) -> bool:
        """根据 direction 判断 current 是否优于 threshold。

        maximize: current >= threshold 视为优
        minimize: current <= threshold 视为优

        TrialEarlyStopper.should_stop 3 个 strategy 分支都使用相同的
        maximize/minimize 比较模式，提取为辅助方法。
        """
        return (current >= threshold) if self.direction == 'maximize' else (current <= threshold)

    def should_stop(self, current_score: float) -> tuple:
        """
        判断当前 trial 是否应该早停

        Returns:
            (should_stop: bool, reason: StopReason)
        """
        if not self.config.enabled:
            return False, StopReason.NONE

        if len(self.scores) < self.config.warmup_trials:
            return False, StopReason.NONE

        scores = np.array(self.scores)

        if self.config.strategy == 'median':
            threshold = np.median(scores) - self.config.min_std_factor * np.std(scores)
            if not self._is_better(current_score, threshold):
                self._stopped_trials += 1
                return True, StopReason.TRIAL_MEDIAN

        elif self.config.strategy == 'percentile':
            threshold = np.percentile(scores, self.config.percentile * 100)
            if not self._is_better(current_score, threshold):
                self._stopped_trials += 1
                return True, StopReason.TRIAL_PERCENTILE

        elif self.config.strategy == 'absolute':
            threshold = self.config.absolute_threshold
            if threshold is not None and not self._is_better(current_score, threshold):
                self._stopped_trials += 1
                return True, StopReason.ABSOLUTE_THRESHOLD

        # 概率性早停：基于历史分布计算当前 trial 的 p-value
        # 如果当前分数显著差（p < 0.1），以概率方式早停，避免过度保守
        if len(scores) >= 10:
            if self.direction == 'maximize':
                percentile = stats.percentileofscore(scores, current_score, kind='rank')
            else:
                percentile = 100 - stats.percentileofscore(scores, current_score, kind='rank')
            if percentile < 10:
                prob = 0.5 + (10 - percentile) / 20  # 10% -> 0.5, 0% -> 1.0
                self._prob_history.append(prob)
                if len(self._prob_history) > 1 and np.random.random() < prob:
                    self._stopped_trials += 1
                    return True, StopReason.TRIAL_PERCENTILE

        return False, StopReason.NONE
    
    def report(self, score: float) -> None:
        """报告 trial 最终分数"""
        self.scores.append(float(score))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取早停统计"""
        if not self.scores:
            return {'n_scores': 0, 'stopped_trials': self._stopped_trials}
        return {
            'n_scores': len(self.scores),
            'mean': float(np.mean(self.scores)),
            'median': float(np.median(self.scores)),
            'std': float(np.std(self.scores)),
            'min': float(np.min(self.scores)),
            'max': float(np.max(self.scores)),
            'stopped_trials': self._stopped_trials,
            'stop_rate': self._stopped_trials / max(len(self.scores), 1)
        }


class FoldEarlyStopper:
    """
    Fold 级早停器
    
    在交叉验证过程中监控每个 fold 的表现，支持：
    - 超时检测
    - 性能持续下降检测
    """
    
    def __init__(self, config: FoldEarlyStopConfig,
                 direction: str = 'maximize') -> None:
        self.config = config
        self.direction = direction
        self._fold_start_time: Optional[float] = None
        self._fold_scores: List[float] = []
        self._stopped_folds: int = 0
    
    def start_fold(self) -> None:
        """标记 fold 开始"""
        self._fold_start_time = time.time()
        self._fold_scores.clear()
    
    def check(self, current_score: Optional[float] = None) -> tuple:
        """
        检查当前 fold 是否应该早停
        
        优化：使用向量化计算下降检测，避免 Python 级循环。
        
        Args:
            current_score: 当前验证分数（如每轮迭代后的验证分数）
            
        Returns:
            (should_stop: bool, reason: StopReason)
        """
        if not self.config.enabled:
            return False, StopReason.NONE
        
        # 超时检测
        if self._fold_start_time is not None and self.config.max_fold_time is not None:
            elapsed = time.time() - self._fold_start_time
            if elapsed > self.config.max_fold_time:
                self._stopped_folds += 1
                return True, StopReason.FOLD_TIMEOUT
        
        # 性能下降检测
        if current_score is not None:
            self._fold_scores.append(float(current_score))
            if len(self._fold_scores) >= self.config.degrade_patience + 1:
                recent = np.array(self._fold_scores[-(self.config.degrade_patience + 1):])
                # 向量化计算下降比例
                if self.direction == 'maximize':
                    drops = (recent[:-1] - recent[1:]) / (np.abs(recent[:-1]) + 1e-10)
                else:
                    drops = (recent[1:] - recent[:-1]) / (np.abs(recent[:-1]) + 1e-10)
                # 向量化统计下降次数
                degrades = int(np.sum(drops > self.config.degrade_threshold))
                if degrades >= self.config.degrade_patience:
                    self._stopped_folds += 1
                    return True, StopReason.FOLD_DEGRADE
        
        return False, StopReason.NONE
    
    def get_stats(self) -> Dict[str, Any]:
        """获取 fold 早停统计"""
        return {
            'stopped_folds': self._stopped_folds,
            'last_fold_scores': self._fold_scores.copy()
        }


class ModelEarlyStopper:
    """
    模型级早停封装
    
    为不同模型类型提供统一的早停参数生成和回调接口。
    
    支持:
    - XGBoost: early_stopping_rounds + eval_set
    - LightGBM: early_stopping_rounds + eval_set
    - CatBoost: early_stopping_rounds + eval_set
    - sklearn 深度学习: 通过 callback
    - PyTorch 模型: 通过 patience + min_delta
    """
    
    def __init__(self, config: ModelEarlyStopConfig) -> None:
        self.config = config
        self._best_score: Optional[float] = None
        self._best_weights: Optional[Any] = None
        self._patience_counter: int = 0
        self._history: List[float] = []
    
    def get_params(self, model_key: str) -> Dict[str, Any]:
        """
        获取模型特定的早停参数

        Args:
            model_key: 模型标识（如 'xgb', 'lgb', 'catboost'）

        Returns:
            可传入模型构造函数的早停参数字典
        """
        if not self.config.enabled:
            return {}

        # 支持 sklearn / XGBoost / LightGBM / CatBoost / PyTorch 五种
        if model_key in _GBDT_KEYS:
            return {'early_stopping_rounds': self.config.early_stopping_rounds}
        if model_key in _TORCH_KEYS:
            return {
                'early_stopping_patience': self.config.patience,
                'early_stopping_min_delta': self.config.min_delta,
                'early_stopping_restore_best': self.config.restore_best,
            }
        return {}
    
    def check_convergence(self, val_score: float) -> tuple:
        """
        通用收敛检测（用于自定义训练循环）
        
        Args:
            val_score: 当前验证分数
            
        Returns:
            (should_stop: bool, reason: StopReason, improved: bool)
        """
        if not self.config.enabled:
            return False, StopReason.NONE, False
        
        self._history.append(val_score)
        
        if self._best_score is None:
            self._best_score = val_score
            self._best_weights = None  # 由调用方保存权重
            return False, StopReason.NONE, True
        
        improved = val_score > self._best_score + self.config.min_delta
        
        if improved:
            self._best_score = val_score
            self._patience_counter = 0
            return False, StopReason.NONE, True
        
        self._patience_counter += 1
        if self._patience_counter >= self.config.patience:
            return True, StopReason.MODEL_PLATEAU, False
        
        return False, StopReason.NONE, False
    
    @property
    def best_score(self) -> Optional[float]:
        return self._best_score
    
    def reset(self) -> None:
        """重置状态"""
        self._best_score = None
        self._best_weights = None
        self._patience_counter = 0
        self._history.clear()


class SmartEarlyStopper:
    """
    智能早停器（统一入口）
    
    组合 Trial / Fold / Model 三层早停策略。
    
    使用方式:
        stopper = SmartEarlyStopper(config=SmartEarlyStopConfig())
        
        # 在优化器循环中
        for trial in range(n_trials):
            score = evaluate(params)
            should_stop, reason = stopper.trial_check(score)
            if should_stop:
                break
        
        # 在 CV 循环中
        for fold in range(n_folds):
            stopper.fold_start()
            for epoch in training:
                val_score = validate()
                should_stop, reason = stopper.fold_check(val_score)
                if should_stop:
                    break
        
        # 获取早停参数
        early_stop_params = stopper.get_model_params('xgb')
    """
    
    def __init__(self, config: Optional[SmartEarlyStopConfig] = None) -> None:
        self.config = config or SmartEarlyStopConfig()
        self.trial_stopper = TrialEarlyStopper(
            self.config.trial, self.config.direction
        )
        self.fold_stopper = FoldEarlyStopper(
            self.config.fold, self.config.direction
        )
        self.model_stopper = ModelEarlyStopper(self.config.model)
        
        # 多目标早停追踪
        self._trial_start_time: Optional[float] = None
        self._trial_scores: List[float] = []
        self._trial_times: List[float] = []
    
    # -------------------------------------------------------------------------
    # Trial 级接口
    # -------------------------------------------------------------------------
    
    def trial_start(self) -> None:
        """标记 trial 开始（用于多目标时间追踪）"""
        self._trial_start_time = time.time()
        self._trial_scores.clear()
        self._trial_times.clear()
    
    def trial_check(self, current_score: float) -> tuple:
        """
        检查 trial 是否应该早停
        
        支持多目标：如果启用了 multi_objective，同时考虑时间成本。
        """
        # 标准单目标早停
        should_stop, reason = self.trial_stopper.should_stop(current_score)
        if should_stop:
            return True, reason
        
        # 多目标早停：考虑时间成本
        if self.config.multi_objective and self._trial_start_time is not None:
            elapsed = time.time() - self._trial_start_time
            self._trial_scores.append(current_score)
            self._trial_times.append(elapsed)
            
            if len(self._trial_scores) >= 3 and self.config.max_time_per_trial is not None:
                # 计算边际收益：每单位时间带来的分数提升
                recent_scores = np.array(self._trial_scores[-3:])
                recent_times = np.array(self._trial_times[-3:])
                
                if self.config.direction == 'maximize':
                    score_gain = recent_scores[-1] - recent_scores[0]
                else:
                    score_gain = recent_scores[0] - recent_scores[-1]
                
                time_cost = recent_times[-1] - recent_times[0]
                
                if time_cost > 0:
                    marginal_improvement = score_gain / time_cost
                    # 如果边际收益低于阈值且已超时的 50%，概率性早停
                    if marginal_improvement < 1e-4 and elapsed > self.config.max_time_per_trial * 0.5:
                        return True, StopReason.TRIAL_MEDIAN
        
        return False, StopReason.NONE
    
    def trial_report(self, score: float) -> None:
        """报告 trial 最终分数"""
        self.trial_stopper.report(score)
    
    # -------------------------------------------------------------------------
    # Fold 级接口
    # -------------------------------------------------------------------------
    
    def fold_start(self) -> None:
        """标记 fold 开始"""
        self.fold_stopper.start_fold()
    
    def fold_check(self, current_score: Optional[float] = None) -> tuple:
        """检查 fold 是否应该早停"""
        return self.fold_stopper.check(current_score)
    
    # -------------------------------------------------------------------------
    # Model 级接口
    # -------------------------------------------------------------------------
    
    def get_model_params(self, model_key: str) -> Dict[str, Any]:
        """获取模型早停参数"""
        return self.model_stopper.get_params(model_key)
    
    def model_check(self, val_score: float) -> tuple:
        """检查模型训练是否应该早停"""
        return self.model_stopper.check_convergence(val_score)
    
    def reset_model(self) -> None:
        """重置模型早停状态"""
        self.model_stopper.reset()
    
    # -------------------------------------------------------------------------
    # 统计
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """获取所有早停统计"""
        return {
            'trial': self.trial_stopper.get_stats(),
            'fold': self.fold_stopper.get_stats(),
            'model': {
                'best_score': self.model_stopper.best_score,
                'patience_counter': self.model_stopper._patience_counter
            }
        }
