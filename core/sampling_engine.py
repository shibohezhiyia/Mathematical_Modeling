"""
自动降采样引擎

核心能力：
1. 自动判断是否需要降采样（基于数据规模/性能策略）
2. 智能分层采样，保持数据分布：
   - 分类任务 → Stratified采样（保持类别比例）
   - 回归任务 → 等频分箱后分层采样（保持目标分布）
   - 无监督/聚类 → 随机采样
3. 采样报告：记录采样比例、分布保持度

使用方式：
    sampler = AutoSampler(max_samples=50000, task_type='classification')
    if sampler.should_sample(len(df)):
        X_s, y_s, report = sampler.sample(X, y)
        print(report)  # 原始100万 → 采样5万，类别比例保持99.2%
"""

from typing import Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.modeling_engine import TaskType, TaskTypeDetector
from utils.helpers import log_info, log_warning


# =============================================================================
# 采样报告
# =============================================================================

@dataclass
class SamplingReport:
    """采样报告"""
    original_n: int = 0
    sampled_n: int = 0
    sample_ratio: float = 1.0
    strategy: str = "none"          # stratified / quantile_stratified / random / none
    task_type: str = ""
    
    # 分布保持度（各类别/分箱的原始比例 vs 采样后比例的平均绝对误差）
    distribution_preservation: float = 1.0  # 1.0=完美保持，0.0=完全破坏
    
    # 详细信息
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        if self.sample_ratio >= 1.0:
            return f"SamplingReport(未采样, n={self.original_n})"
        return (f"SamplingReport({self.original_n} → {self.sampled_n}, "
                f"比例={self.sample_ratio:.1%}, 策略={self.strategy}, "
                f"分布保持={self.distribution_preservation:.1%})")


# =============================================================================
# 采样策略枚举
# =============================================================================

class SamplingStrategy(Enum):
    AUTO = "auto"
    STRATIFIED = "stratified"           # 分层采样（分类）
    QUANTILE_STRATIFIED = "quantile"    # 分位数分层（回归）
    RANDOM = "random"                   # 纯随机
    NONE = "none"                       # 不采样


# =============================================================================
# 自动降采样引擎
# =============================================================================

class AutoSampler:
    """
    自动降采样引擎
    
    触发逻辑：
    - 数据行数 > max_samples → 采样到 max_samples
    - 或用户显式要求采样
    
    策略选择：
    - 分类 → StratifiedShuffleSplit 保持类别比例
    - 回归 → 等频分箱后分层采样保持目标分布
    - 无监督 → 随机采样
    """
    
    # 默认阈值
    DEFAULT_MAX_SAMPLES = 50_000
    DEFAULT_MIN_SAMPLES = 1_000       # 低于此值不采样（防止样本过少）
    
    def __init__(self,
                 max_samples: int = DEFAULT_MAX_SAMPLES,
                 min_samples: int = DEFAULT_MIN_SAMPLES,
                 task_type: Optional[Union[str, TaskType]] = None,
                 strategy: Union[str, SamplingStrategy] = SamplingStrategy.AUTO,
                 random_state: int = 42) -> None:
        """
        Args:
            max_samples: 最大样本数，超过则触发采样
            min_samples: 最小样本数，低于此值不采样
            task_type: 任务类型（None=自动推断）
            strategy: 采样策略
            random_state: 随机种子
        """
        self.max_samples = max_samples
        self.min_samples = min_samples
        self.random_state = random_state
        
        if isinstance(task_type, TaskType):
            self.task_type = task_type
        elif task_type:
            self.task_type = TaskType(task_type)
        else:
            self.task_type = None
        
        if isinstance(strategy, str):
            self.strategy = SamplingStrategy(strategy)
        else:
            self.strategy = strategy
    
    def should_sample(self, n_rows: int, memory_mb: Optional[float] = None) -> bool:
        """
        判断是否需要降采样
        
        Args:
            n_rows: 数据行数
            memory_mb: 数据内存占用（MB）
            
        Returns:
            bool
        """
        if n_rows <= self.min_samples:
            return False
        if n_rows > self.max_samples:
            return True
        # 内存过大也触发（即使行数不多，如宽表）
        if memory_mb and memory_mb > 2000:  # 2GB
            return True
        return False
    
    def sample(self,
               X: Union[pd.DataFrame, np.ndarray],
               y: Optional[Union[pd.Series, np.ndarray]] = None,
               force: bool = False) -> Tuple[Any, Optional[Any], SamplingReport]:
        """
        执行降采样
        
        Args:
            X: 特征
            y: 标签（可选）
            force: 强制采样（忽略 should_sample 判断）
            
        Returns:
            (X_sampled, y_sampled, SamplingReport)
        """
        n_rows = len(X)
        
        # 判断是否需要采样
        if not force and not self.should_sample(n_rows):
            report = SamplingReport(
                original_n=n_rows, sampled_n=n_rows,
                sample_ratio=1.0, strategy="none"
            )
            return X, y, report
        
        # 确定任务类型
        task_type = self._infer_task_type(y)
        
        # 计算采样比例
        target_n = min(self.max_samples, max(int(n_rows * 0.8), self.min_samples))
        sample_ratio = target_n / n_rows
        
        # 选择并执行采样策略
        if task_type == TaskType.CLASSIFICATION:
            X_s, y_s, strategy_name = self._stratified_sample(X, y, sample_ratio)
        elif task_type == TaskType.REGRESSION:
            X_s, y_s, strategy_name = self._quantile_stratified_sample(X, y, sample_ratio)
        else:
            # 无监督/聚类
            X_s, y_s, strategy_name = self._random_sample(X, y, sample_ratio)
        
        # 生成报告
        report = self._build_report(
            n_rows, len(X_s), sample_ratio, strategy_name,
            task_type, y, y_s
        )
        
        log_info(f"[AutoSampler] {report}")
        return X_s, y_s, report
    
    def _infer_task_type(self, y: Optional[Any]) -> TaskType:
        """推断任务类型"""
        if self.task_type:
            return self.task_type
        if y is not None:
            detected = TaskTypeDetector.detect(y)
            if detected != TaskType.UNKNOWN:
                return detected
        return TaskType.CLUSTERING
    
    def _stratified_sample(self, X: Any, y: Optional[Any], sample_ratio: float) -> Tuple[Any, Any, str]:
        """
        分类任务分层采样：保持类别比例
        """
        if y is None:
            return self._random_sample(X, y, sample_ratio)
        
        # 使用 sklearn 的 stratify 参数
        try:
            ratio = min(0.999, max(0.001, sample_ratio))
            X_s, _, y_s, _ = train_test_split(
                X, y,
                train_size=ratio,
                stratify=y,
                random_state=self.random_state
            )
            return X_s, y_s, "stratified"
        except ValueError as e:
            # 如果 stratify 失败（如某些类别样本太少），回退到随机
            log_warning(f"[AutoSampler] 分层采样失败: {e}，回退到随机采样")
            return self._random_sample(X, y, sample_ratio)
    
    def _quantile_stratified_sample(self, X: Any, y: Optional[Any], sample_ratio: float) -> Tuple[Any, Any, str]:
        """
        回归任务分位数分层采样：保持目标值分布
        
        思路：将目标值分成 n_quantiles 个等频分箱，
              然后在每个分箱内进行随机采样。
        """
        if y is None:
            return self._random_sample(X, y, sample_ratio)
        
        y_arr = np.array(y)
        n_quantiles = min(10, max(3, int(len(y) / 1000)))  # 动态分箱数
        
        try:
            # 等频分箱
            quantiles = pd.qcut(y_arr, q=n_quantiles, labels=False, duplicates='drop')
            n_bins = len(np.unique(quantiles))
            
            if n_bins <= 1:
                # 分箱失败，回退到随机
                return self._random_sample(X, y, sample_ratio)
            
            # 按分箱分层采样
            sampled_indices = []
            for bin_id in range(n_bins):
                bin_mask = quantiles == bin_id
                bin_indices = np.where(bin_mask)[0]
                
                if len(bin_indices) == 0:
                    continue
                
                # 计算该分箱应采样数量
                bin_target = max(1, int(len(bin_indices) * sample_ratio))
                
                # 随机采样（不放回）
                rng = np.random.RandomState(self.random_state + bin_id)
                chosen = rng.choice(bin_indices, size=min(bin_target, len(bin_indices)), replace=False)
                sampled_indices.extend(chosen)
            
            # 去重并排序（保持原始顺序）
            sampled_indices = np.array(sorted(set(sampled_indices)))
            
            if isinstance(X, pd.DataFrame):
                X_s = X.iloc[sampled_indices]
            else:
                X_s = X[sampled_indices]
            
            if isinstance(y, pd.Series):
                y_s = y.iloc[sampled_indices]
            else:
                y_s = y_arr[sampled_indices]
            
            return X_s, y_s, "quantile_stratified"
            
        except Exception as e:
            log_warning(f"[AutoSampler] 分位数分层采样失败: {e}，回退到随机采样")
            return self._random_sample(X, y, sample_ratio)
    
    def _random_sample(self, X: Any, y: Optional[Any], sample_ratio: float) -> Tuple[Any, Optional[Any], str]:
        """纯随机采样"""
        n = len(X)
        target_n = min(n, max(1, int(n * sample_ratio)))
        
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(n, size=target_n, replace=False)
        indices = np.sort(indices)
        
        if isinstance(X, pd.DataFrame):
            X_s = X.iloc[indices]
        else:
            X_s = X[indices]
        
        y_s = None
        if y is not None:
            if isinstance(y, pd.Series):
                y_s = y.iloc[indices]
            else:
                y_s = y[indices]
        
        return X_s, y_s, "random"
    
    def _build_report(self,
                      original_n: int,
                      sampled_n: int,
                      sample_ratio: float,
                      strategy: str,
                      task_type: TaskType,
                      y_orig: Optional[Any],
                      y_sampled: Optional[Any]) -> SamplingReport:
        """构建采样报告，计算分布保持度"""
        report = SamplingReport(
            original_n=original_n,
            sampled_n=sampled_n,
            sample_ratio=sample_ratio,
            strategy=strategy,
            task_type=task_type.value
        )
        
        # 计算分布保持度
        if y_orig is not None and y_sampled is not None:
            if task_type == TaskType.CLASSIFICATION:
                report.distribution_preservation = self._calc_class_preservation(y_orig, y_sampled)
            elif task_type == TaskType.REGRESSION:
                report.distribution_preservation = self._calc_distribution_similarity(y_orig, y_sampled)
        
        report.details = {
            'max_samples': self.max_samples,
            'min_samples': self.min_samples,
            'random_state': self.random_state,
        }
        
        return report
    
    def _calc_class_preservation(self, y_orig: Any, y_sampled: Any) -> float:
        """计算分类类别比例保持度（1-MAE）"""
        try:
            orig_counts = pd.Series(y_orig).value_counts(normalize=True).sort_index()
            samp_counts = pd.Series(y_sampled).value_counts(normalize=True).sort_index()
            
            # 对齐索引
            all_classes = sorted(set(orig_counts.index) | set(samp_counts.index))
            orig_p = np.array([orig_counts.get(c, 0) for c in all_classes])
            samp_p = np.array([samp_counts.get(c, 0) for c in all_classes])
            
            mae = np.mean(np.abs(orig_p - samp_p))
            return max(0, 1 - mae)
        except Exception:
            return 0.0
    
    def _calc_distribution_similarity(self, y_orig: Any, y_sampled: Any) -> float:
        """计算回归目标分布相似度（基于分位数重叠）"""
        try:
            y_o = np.array(y_orig).flatten()
            y_s = np.array(y_sampled).flatten()
            
            # 比较关键分位数
            quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
            orig_q = np.quantile(y_o, quantiles)
            samp_q = np.quantile(y_s, quantiles)
            
            # 归一化后比较
            scale = np.std(y_o) + 1e-8
            diff = np.mean(np.abs(orig_q - samp_q)) / scale
            return max(0, 1 - diff)
        except Exception:
            return 0.0


# =============================================================================
# 便捷函数
# =============================================================================

def auto_sample(X: Union[pd.DataFrame, np.ndarray],
                y: Optional[Union[pd.Series, np.ndarray]] = None,
                max_samples: int = 50_000,
                task_type: Optional[str] = None,
                random_state: int = 42) -> Tuple[Any, Optional[Any], SamplingReport]:
    """
    一键自动降采样
    
    Args:
        X: 特征
        y: 标签
        max_samples: 最大样本数
        task_type: 任务类型（'classification'/'regression'/'clustering'）
        random_state: 随机种子
        
    Returns:
        (X_sampled, y_sampled, report)
        
    示例：
        X_s, y_s, report = auto_sample(X, y, max_samples=10000)
        if report.sample_ratio < 1.0:
            print(f"已降采样: {report}")
    """
    sampler = AutoSampler(
        max_samples=max_samples,
        task_type=task_type,
        random_state=random_state
    )
    return sampler.sample(X, y)
