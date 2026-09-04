"""
Permutation Importance 计算引擎

基于 sklearn.inspection.permutation_importance，
在比赛级特征选择中比树模型的内置重要性更可靠。
"""

import warnings
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

warnings.filterwarnings('ignore')


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: Optional[str] = None,
    n_repeats: int = 5,
    random_state: int = 42,
    test_size: float = 0.2,
    max_samples: Optional[int] = 5000,
    max_features: Optional[int] = 100,
) -> pd.DataFrame:
    """
    计算 Permutation Importance。

    Args:
        model: 已训练或待训练的模型
        X: 特征
        y: 标签
        scoring: sklearn scoring 字符串，None 时用 model.score
        n_repeats: 打乱次数
        test_size: 用于 PI 的 holdout 验证集比例
        max_samples: PI 专用样本上限，避免重复预测拖垮大数据流程
        max_features: PI 专用特征上限，宽表优先保留高方差特征

    Returns:
        DataFrame[feature, importance, std]
    """
    # 排列重要性需要约 n_features * n_repeats 次预测，其复杂度很容易
    # 超过主训练。它只用于解释，不应无限消耗建模资源。
    if max_samples is not None and len(X) > max_samples:
        rng = np.random.RandomState(random_state)
        positions = np.sort(rng.choice(len(X), size=max_samples, replace=False))
        X = X.iloc[positions]
        y = y.iloc[positions] if isinstance(y, pd.Series) else pd.Series(np.asarray(y)[positions])

    if max_features is not None and X.shape[1] > max_features:
        numeric = X.select_dtypes(include=[np.number])
        if numeric.shape[1] > 0:
            variances = numeric.var(axis=0).replace([np.inf, -np.inf], np.nan).fillna(-np.inf)
            selected_cols = variances.nlargest(min(max_features, len(variances))).index.tolist()
        else:
            selected_cols = list(X.columns[:max_features])
        X = X[selected_cols]

    # 分出 holdout 验证集
    y_series = pd.Series(y)
    value_counts = y_series.value_counts()
    can_stratify = 1 < len(value_counts) <= 20 and int(value_counts.min()) >= 2
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if can_stratify else None,
    )

    # 在训练集上拟合模型
    model_copy = _clone_model(model)
    model_copy.fit(X_train, y_train)

    # 计算 PI
    result = permutation_importance(
        model_copy, X_val, y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=1
    )

    df = pd.DataFrame({
        'feature': X.columns,
        'importance': result.importances_mean,
        'std': result.importances_std
    })
    df = df.sort_values('importance', ascending=False).reset_index(drop=True)
    return df


def _clone_model(model: Any) -> Any:
    """尝试克隆模型"""
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        # 对不支持 clone 的模型（如某些深度学习模型），返回原模型
        return model
