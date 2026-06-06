"""
数据集元特征提取器

提取数据集的统计特征，作为 AutoML 策略选择的输入。
元特征包括：规模、维度、类型分布、稀疏度、缺失率、不平衡度等。
"""

import math
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import entropy

from core.modeling_engine import TaskType

# _compute_complexity 用到的"规模桶"阈值
# 5 个样本桶和 4 个特征桶，从 5 个硬编码的 if/elif 块提取。
# 抽到模块常量便于调整且让复杂度的 100 分制评分分布可读。
_SAMPLE_BUCKETS = (
    (1_000,    5),    # < 1k 样本 → 5 分
    (10_000,  15),    # < 10k 样本 → 15 分
    (100_000, 20),    # < 100k 样本 → 20 分
    # >= 100k 样本 → 25 分
)
_FEATURE_BUCKETS = (
    (10,     5),      # < 10 特征 → 5 分
    (50,    10),      # < 50 特征 → 10 分
    (200,   15),      # < 200 特征 → 15 分
    # >= 200 特征 → 20 分
)
# 各项指标最大分值（cap 用）
_MAX_SAMPLE_SCORE = 25
_MAX_FEATURE_SCORE = 20
_MAX_MISSING_SCORE = 15
_MAX_IMBALANCE_SCORE = 15
_MAX_CORRELATION_SCORE = 15
_MAX_SPARSITY_SCORE = 10
_TOTAL_COMPLEXITY_CAP = 100


@dataclass
class MetaFeatures:
    """数据集元特征"""
    # 规模
    n_samples: int = 0
    n_features: int = 0
    n_numeric: int = 0
    n_categorical: int = 0
    
    # 比例
    sample_feature_ratio: float = 0.0       # n_samples / n_features
    numeric_ratio: float = 0.0
    categorical_ratio: float = 0.0
    
    # 质量
    missing_ratio: float = 0.0
    sparsity: float = 0.0                   # 数值列零值比例
    
    # 特征关系
    feature_correlation_mean: float = 0.0
    feature_correlation_max: float = 0.0
    
    # 目标变量
    n_classes: int = 0
    class_imbalance_ratio: float = 1.0
    target_entropy: float = 0.0
    target_std: float = 0.0
    
    # 综合指标
    complexity_score: float = 0.0           # 综合复杂度评分
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'n_numeric': self.n_numeric,
            'n_categorical': self.n_categorical,
            'sample_feature_ratio': round(self.sample_feature_ratio, 2),
            'numeric_ratio': round(self.numeric_ratio, 2),
            'categorical_ratio': round(self.categorical_ratio, 2),
            'missing_ratio': round(self.missing_ratio, 4),
            'sparsity': round(self.sparsity, 4),
            'feature_correlation_mean': round(self.feature_correlation_mean, 4),
            'feature_correlation_max': round(self.feature_correlation_max, 4),
            'n_classes': self.n_classes,
            'class_imbalance_ratio': round(self.class_imbalance_ratio, 2),
            'target_entropy': round(self.target_entropy, 4),
            'target_std': round(self.target_std, 4),
            'complexity_score': round(self.complexity_score, 2),
        }


class MetaFeatureExtractor:
    """
    元特征提取器
    
    使用方式：
        extractor = MetaFeatureExtractor()
        meta = extractor.extract(X, y, task_type)
        print(meta.to_dict())
    """
    
    def extract(self, X: pd.DataFrame, y: Optional[pd.Series], task_type: TaskType) -> MetaFeatures:
        meta = MetaFeatures()
        
        # 基本规模
        meta.n_samples, meta.n_features = X.shape
        # 向量化：使用 select_dtypes 替代逐列 dtype 检查
        meta.n_numeric = len(X.select_dtypes(include=[np.number]).columns)
        meta.n_categorical = meta.n_features - meta.n_numeric
        
        meta.sample_feature_ratio = meta.n_samples / max(meta.n_features, 1)
        meta.numeric_ratio = meta.n_numeric / max(meta.n_features, 1)
        meta.categorical_ratio = meta.n_categorical / max(meta.n_features, 1)
        
        # 缺失值：使用 numpy 一次性计算，避免 sum().sum() 双重遍历
        meta.missing_ratio = np.isnan(X.values).sum() / max(X.size, 1) if X.size > 0 else 0.0
        
        # 稀疏度（数值列零值比例）
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            numeric_values = X[numeric_cols].values
            meta.sparsity = (numeric_values == 0).sum() / max(numeric_values.size, 1)
        
        # 特征相关性
        if len(numeric_cols) >= 2:
            try:
                corr_matrix = X[numeric_cols].corr().abs().values
                # 取上三角（排除对角线）
                triu_idx = np.triu_indices_from(corr_matrix, k=1)
                if len(triu_idx[0]) > 0:
                    corrs = corr_matrix[triu_idx]
                    meta.feature_correlation_mean = float(np.nanmean(corrs))
                    meta.feature_correlation_max = float(np.nanmax(corrs))
            except Exception:
                pass
        
        # 目标变量特征
        if y is not None:
            if task_type == TaskType.CLASSIFICATION:
                class_counts = pd.Series(y).value_counts()
                meta.n_classes = len(class_counts)
                meta.class_imbalance_ratio = class_counts.max() / class_counts.min() if len(class_counts) > 1 else 1.0
                # 目标熵
                probs = class_counts / class_counts.sum()
                meta.target_entropy = float(entropy(probs))
            else:
                meta.target_std = float(np.std(y))
        
        # 综合复杂度评分
        meta.complexity_score = self._compute_complexity(meta)
        
        return meta
    
    def _compute_complexity(self, meta: MetaFeatures) -> float:
        """
        计算数据集综合复杂度评分（0-100）

        考虑因素：
        - 样本数（大数据更复杂）
        - 特征数（高维更复杂）
        - 缺失率
        - 类别不平衡度
        - 特征相关性
        """
        score = 0.0

        # 样本规模：按桶查表 + 兜底（>= 最大桶 → 最大分）
        sample_score = _MAX_SAMPLE_SCORE  # 兜底最大分
        for threshold, value in _SAMPLE_BUCKETS:
            if meta.n_samples < threshold:
                sample_score = value
                break
        score += sample_score

        # 特征维度：同样按桶查表
        feature_score = _MAX_FEATURE_SCORE
        for threshold, value in _FEATURE_BUCKETS:
            if meta.n_features < threshold:
                feature_score = value
                break
        score += feature_score

        # 缺失率 (0-15)
        score += min(meta.missing_ratio * 100, _MAX_MISSING_SCORE)

        # 类别不平衡 (0-15, 仅分类)
        if meta.n_classes > 0:
            imbalance_score = min(math.log2(meta.class_imbalance_ratio) * 3, _MAX_IMBALANCE_SCORE)
            score += imbalance_score

        # 特征相关性 (0-15)
        score += min(meta.feature_correlation_max * 15, _MAX_CORRELATION_SCORE)

        # 稀疏度 (0-10)
        score += min(meta.sparsity * 50, _MAX_SPARSITY_SCORE)

        return min(score, _TOTAL_COMPLEXITY_CAP)
