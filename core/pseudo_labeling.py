"""
伪标签 / 半监督学习引擎

用已训练模型对无标签测试集预测，选取高置信度样本作为伪标签，
扩充训练数据后重新训练。Kaggle 比赛中常用的提分技巧。
"""

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from core.logging_utils import log_info, log_warning

warnings.filterwarnings('ignore')


class PseudoLabeler:
    """
    伪标签生成器

    支持策略：
    - threshold: 置信度阈值（分类：最大概率；回归：预测方差/分位数）
    - top_k: 选取最自信的 top-k 样本
    - ratio: 按测试集比例选取
    """

    def __init__(self,
                 strategy: str = 'threshold',
                 threshold: float = 0.9,
                 top_k: Optional[int] = None,
                 ratio: float = 0.3,
                 random_state: int = 42) -> None:
        self.strategy = strategy
        self.threshold = threshold
        self.top_k = top_k
        self.ratio = ratio
        self.random_state = random_state

    def generate(self,
                 model: Any,
                 X_test: pd.DataFrame,
                 task_type: str = 'classification') -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        生成伪标签

        Returns:
            (X_pseudo, y_pseudo, confidences)
        """
        if task_type == 'classification':
            return self._generate_classification(model, X_test)
        else:
            return self._generate_regression(model, X_test)

    def _generate_classification(self, model: Any, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """分类任务伪标签：基于预测概率置信度"""
        if not hasattr(model, 'predict_proba'):
            log_warning("[PseudoLabeler] 模型不支持 predict_proba，跳过伪标签")
            return X_test.iloc[0:0], np.array([]), np.array([])

        proba = model.predict_proba(X_test)
        pred = model.predict(X_test)
        confidence = np.max(proba, axis=1)

        if self.strategy == 'threshold':
            mask = confidence >= self.threshold
        elif self.strategy == 'top_k':
            k = self.top_k or max(1, int(len(X_test) * 0.1))
            k = min(k, len(X_test))
            idx = np.argsort(confidence)[-k:]
            mask = np.zeros(len(X_test), dtype=bool)
            mask[idx] = True
        elif self.strategy == 'ratio':
            k = max(1, int(len(X_test) * self.ratio))
            idx = np.argsort(confidence)[-k:]
            mask = np.zeros(len(X_test), dtype=bool)
            mask[idx] = True
        else:
            mask = confidence >= self.threshold

        X_pseudo = X_test[mask]
        y_pseudo = pred[mask]
        conf_pseudo = confidence[mask]

        return X_pseudo, y_pseudo, conf_pseudo

    def _generate_regression(self, model: Any, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """回归任务伪标签：基于预测值的一致性（多次预测方差）"""
        pred = model.predict(X_test)

        # 回归任务的"置信度"：用预测值的绝对值归一化（简单的启发式）
        # 更好的方法：使用模型 ensemble 的方差，但这里简化为按预测值分位数选取
        if self.strategy == 'threshold':
            # 对于回归，threshold 解释为选取远离 0 的预测（假设目标已中心化）
            abs_pred = np.abs(pred)
            q = np.quantile(abs_pred, 1 - self.threshold)
            mask = abs_pred >= q
        elif self.strategy == 'top_k':
            k = self.top_k or max(1, int(len(X_test) * 0.1))
            k = min(k, len(X_test))
            idx = np.argsort(np.abs(pred))[-k:]
            mask = np.zeros(len(X_test), dtype=bool)
            mask[idx] = True
        elif self.strategy == 'ratio':
            k = max(1, int(len(X_test) * self.ratio))
            idx = np.argsort(np.abs(pred))[-k:]
            mask = np.zeros(len(X_test), dtype=bool)
            mask[idx] = True
        else:
            mask = np.ones(len(X_test), dtype=bool)

        X_pseudo = X_test[mask]
        y_pseudo = pred[mask]
        conf_pseudo = np.ones(len(y_pseudo))  # 回归任务无概率，用 1 占位

        return X_pseudo, y_pseudo, conf_pseudo


def apply_pseudo_labeling(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    task_type: str = 'classification',
    label_encoder: Any = None,
    **kwargs
) -> Tuple[Any, pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    端到端伪标签增强

    Args:
        model: 已训练的模型
        X_train, y_train: 原始训练数据
        X_test: 测试集（无标签）
        task_type: 'classification' 或 'regression'
        label_encoder: 分类任务的标签编码器
        **kwargs: PseudoLabeler 参数

    Returns:
        (retrained_model, X_combined, y_combined, report)
    """
    pl = PseudoLabeler(**kwargs)
    X_pseudo, y_pseudo, confidences = pl.generate(model, X_test, task_type)

    if len(X_pseudo) == 0:
        log_warning("[PseudoLabeling] 没有生成伪标签样本，跳过增强")
        return model, X_train, y_train, {'n_pseudo': 0, 'reason': 'no high-confidence samples'}

    # 标签编码（分类任务）
    if task_type == 'classification' and label_encoder is not None:
        y_pseudo = label_encoder.transform(pd.Series(y_pseudo).astype(str))

    # 合并数据
    X_combined = pd.concat([X_train, X_pseudo], ignore_index=True, copy=False)
    y_combined = pd.concat([pd.Series(y_train), pd.Series(y_pseudo)], ignore_index=True, copy=False)

    # 重新训练
    try:
        from sklearn.base import clone
        retrained = clone(model)
    except Exception:
        retrained = model

    retrained.fit(X_combined, y_combined)

    report = {
        'n_pseudo': len(X_pseudo),
        'n_original': len(X_train),
        'n_combined': len(X_combined),
        'mean_confidence': float(np.mean(confidences)) if len(confidences) > 0 else 0,
        'strategy': pl.strategy,
    }

    log_info(f"[PseudoLabeling] 伪标签增强: {report['n_pseudo']} 样本 (mean_conf={report['mean_confidence']:.3f})")
    return retrained, X_combined, y_combined, report
