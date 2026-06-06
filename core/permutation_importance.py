"""
Permutation Importance 计算引擎

基于 sklearn.inspection.permutation_importance，
在比赛级特征选择中比树模型的内置重要性更可靠。
"""

import warnings
from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')


def compute_permutation_importance(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    scoring: Optional[str] = None,
    n_repeats: int = 5,
    random_state: int = 42,
    test_size: float = 0.2
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

    Returns:
        DataFrame[feature, importance, std]
    """
    from sklearn.inspection import permutation_importance as sk_pi

    # 分出 holdout 验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) <= 10 else None
    )

    # 在训练集上拟合模型
    model_copy = _clone_model(model)
    model_copy.fit(X_train, y_train)

    # 计算 PI
    result = sk_pi(
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
