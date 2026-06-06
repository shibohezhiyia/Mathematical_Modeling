"""
层次模型模块

核心思想：数据天然有层级结构时，用分层建模借用大盘信息。

数学形式：
    y_store = μ_global + β_city + γ_store + ε

层级示例：
    全局 → 城市 → 门店
    全局 → 品类 → 品牌 → 商品
    全局 → 用户群 → 用户

优势：
- 小样本层级也能借用大盘信息（Shrinkage）
- 避免小样本模型乱跳
- 可解释性强（全局效应 + 局部偏差）

实现策略：
1. 分层 Ridge：逐层拟合残差
2. 随机效应近似：用组内均值 + 全局收缩
3. 分层 LightGBM：加入层级标识作为类别特征
"""

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder

from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


class HierarchicalEstimator(BaseEstimator, RegressorMixin):
    """
    层次估计器

    用法：
        model = HierarchicalEstimator(
            hierarchy_cols=['city', 'store'],  # 从粗到细的层级
            global_estimator=Ridge(),
            local_estimator=LGBMRegressor(),
            shrinkage=0.5  # 收缩强度（0=完全局部，1=完全全局）
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    """

    def __init__(self,
                 hierarchy_cols: Optional[List[str]] = None,
                 global_estimator: Any = None,
                 group_estimators: Optional[Dict[str, Any]] = None,
                 local_estimator: Any = None,
                 shrinkage: float = 0.3,
                 random_state: int = 42) -> None:
        """
        Args:
            hierarchy_cols: 层级列名列表，从粗到细，如 ['region', 'city', 'store']
            global_estimator: 全局模型（捕捉大盘趋势）
            group_estimators: 每层分组模型，如 {'city': Ridge(), 'store': Ridge()}
            local_estimator: 最底层局部模型
            shrinkage: 收缩强度 [0, 1]，越大越偏向全局均值
            random_state: 随机种子
        """
        self.hierarchy_cols = hierarchy_cols
        self.global_estimator = global_estimator
        self.group_estimators = group_estimators or {}
        self.local_estimator = local_estimator
        self.shrinkage = shrinkage
        self.random_state = random_state

        self.global_model_: Optional[Any] = None
        self.group_models_: Dict[str, Dict[Any, Any]] = {}
        self.local_model_: Optional[Any] = None
        self.global_mean_: float = 0.0
        self.group_means_: Dict[str, Dict[Any, float]] = {}
        self.le_: Dict[str, LabelEncoder] = {}
        self._fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'HierarchicalEstimator':
        X = self._to_df(X)
        y = np.array(y).ravel()
        self.global_mean_ = np.mean(y)

        # 1. 全局模型
        global_est = self._create_global_estimator()
        global_est.fit(X, y)
        self.global_model_ = global_est
        global_pred = global_est.predict(X)

        # 2. 逐层分组模型（拟合残差）
        current_residual = y - global_pred
        self.group_models_ = {}
        self.group_means_ = {}

        if self.hierarchy_cols:
            for level, col in enumerate(self.hierarchy_cols):
                if col not in X.columns:
                    log_warning(f"[HierarchicalEstimator] 层级列 '{col}' 不在数据中，跳过")
                    continue

                # 编码类别
                le = LabelEncoder()
                groups = le.fit_transform(X[col].astype(str))
                self.le_[col] = le

                # 为每个组训练一个模型（拟合当前残差）
                level_models = {}
                level_means = {}

                for gid in np.unique(groups):
                    mask = groups == gid
                    group_name = le.inverse_transform([gid])[0]

                    if mask.sum() < 3:
                        # 样本过少：用组内均值
                        level_models[group_name] = None
                        level_means[group_name] = np.mean(current_residual[mask])
                        continue

                    group_est = self._create_group_estimator(col)
                    group_est.fit(X[mask], current_residual[mask])
                    level_models[group_name] = group_est
                    level_means[group_name] = np.mean(current_residual[mask])

                # 更新残差
                group_pred = np.zeros(len(y))
                for gid in np.unique(groups):
                    mask = groups == gid
                    group_name = le.inverse_transform([gid])[0]
                    model = level_models.get(group_name)
                    if model is not None:
                        group_pred[mask] = model.predict(X[mask])
                    else:
                        group_pred[mask] = level_means[group_name]

                current_residual = current_residual - group_pred
                self.group_models_[col] = level_models
                self.group_means_[col] = level_means

                log_info(f"[HierarchicalEstimator] 层级 '{col}': {len(level_models)} 个组")

        # 3. 局部模型（拟合最终残差）
        local_est = self._create_local_estimator()
        local_est.fit(X, current_residual)
        self.local_model_ = local_est

        self._fitted = True
        log_info(f"[HierarchicalEstimator] 拟合完成: global + {len(self.hierarchy_cols or [])} 层 + local")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self._fitted:
            raise ValueError("请先调用 fit()")

        X = self._to_df(X)

        # 1. 全局预测
        pred = self.global_model_.predict(X)

        # 2. 逐层修正
        if self.hierarchy_cols:
            for col in self.hierarchy_cols:
                if col not in X.columns or col not in self.group_models_:
                    continue

                le = self.le_.get(col)
                if le is None:
                    continue

                # 处理未见过的新类别
                groups = X[col].astype(str)
                known_mask = groups.isin(le.classes_)

                if known_mask.any():
                    known_groups = le.transform(groups[known_mask])
                    group_pred = np.zeros(len(X))

                    for gid in np.unique(known_groups):
                        mask = known_mask & (groups == le.inverse_transform([gid])[0])
                        group_name = le.inverse_transform([gid])[0]
                        model = self.group_models_[col].get(group_name)
                        group_mean = self.group_means_[col].get(group_name, 0.0)

                        if model is not None and mask.sum() > 0:
                            group_pred[mask] = model.predict(X[mask])
                        else:
                            group_pred[mask] = group_mean

                    # 收缩：混合全局和局部
                    pred = pred + self.shrinkage * group_pred

                # 未知类别：不衰减（保持全局预测）

        # 3. 局部修正
        if self.local_model_ is not None:
            pred = pred + self.local_model_.predict(X)

        return pred

    def predict_components(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """分别返回各层预测分量（用于分析）"""
        X = self._to_df(X)
        components = {'global': self.global_model_.predict(X)}

        if self.hierarchy_cols:
            for col in self.hierarchy_cols:
                if col not in X.columns or col not in self.group_models_:
                    continue
                le = self.le_.get(col)
                if le is None:
                    continue

                groups = X[col].astype(str)
                group_pred = np.zeros(len(X))
                for group_name in groups.unique():
                    if group_name not in self.group_models_[col]:
                        continue
                    mask = groups == group_name
                    model = self.group_models_[col][group_name]
                    if model is not None:
                        group_pred[mask] = model.predict(X[mask])
                    else:
                        group_pred[mask] = self.group_means_[col].get(group_name, 0.0)
                components[col] = group_pred * self.shrinkage

        components['local'] = self.local_model_.predict(X)
        components['total'] = sum(components.values())
        return components

    def _create_global_estimator(self) -> Any:
        if self.global_estimator is not None:
            return clone(self.global_estimator)
        return Ridge(alpha=1.0, random_state=self.random_state)

    def _create_group_estimator(self, level: str) -> Any:
        if level in self.group_estimators and self.group_estimators[level] is not None:
            return clone(self.group_estimators[level])
        return Ridge(alpha=1.0, random_state=self.random_state)

    def _create_local_estimator(self) -> Any:
        if self.local_estimator is not None:
            return clone(self.local_estimator)
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        except Exception:
            return Ridge(alpha=0.5, random_state=self.random_state)

    @staticmethod
    def _to_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class MixedEffectsApproximator(BaseEstimator, RegressorMixin):
    """
    混合效应近似器

    用简单方法近似随机效应模型：
    y_ij = β·x_ij + u_i + ε_ij
    其中 u_i 是组 i 的随机效应。

    近似策略：
    1. 先拟合固定效应（全局模型）
    2. 计算各组的平均残差作为随机效应估计
    3. 预测时：固定效应 + 组随机效应
    """

    def __init__(self,
                 group_col: str,
                 fixed_estimator: Any = None,
                 random_effect_shrinkage: float = 0.5,
                 random_state: int = 42) -> None:
        """
        Args:
            group_col: 分组列名
            fixed_estimator: 固定效应模型
            random_effect_shrinkage: 随机效应收缩强度
            random_state: 随机种子
        """
        self.group_col = group_col
        self.fixed_estimator = fixed_estimator
        self.random_effect_shrinkage = random_effect_shrinkage
        self.random_state = random_state

        self.fixed_model_: Optional[Any] = None
        self.random_effects_: Dict[Any, float] = {}
        self.global_residual_std_: float = 1.0

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'MixedEffectsApproximator':
        X = self._to_df(X)
        y = np.array(y).ravel()

        # 1. 固定效应
        fixed = self._create_fixed_estimator()
        fixed.fit(X, y)
        self.fixed_model_ = fixed
        fixed_pred = fixed.predict(X)
        residual = y - fixed_pred

        # 2. 随机效应（组内平均残差，向0收缩）
        if self.group_col in X.columns:
            groups = X[self.group_col]
            self.global_residual_std_ = max(residual.std(), 1e-6)

            for g in groups.unique():
                mask = groups == g
                group_residual = residual[mask]
                n_group = len(group_residual)

                # 经验贝叶斯收缩：
                # u_i = (n_i / (n_i + σ²/τ²)) · mean(residual_i)
                # 简化：用 shrinkage 参数控制
                raw_effect = np.mean(group_residual)
                self.random_effects_[g] = self.random_effect_shrinkage * raw_effect

            log_info(f"[MixedEffectsApproximator] 随机效应: {len(self.random_effects_)} 组")
        else:
            log_warning(f"[MixedEffectsApproximator] 分组列 '{self.group_col}' 不存在")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = self._to_df(X)
        pred = self.fixed_model_.predict(X)

        if self.group_col in X.columns:
            groups = X[self.group_col]
            for g in groups.unique():
                mask = groups == g
                effect = self.random_effects_.get(g, 0.0)
                pred[mask] += effect

        return pred

    def _create_fixed_estimator(self) -> Any:
        if self.fixed_estimator is not None:
            return clone(self.fixed_estimator)
        return Ridge(alpha=1.0, random_state=self.random_state)

    @staticmethod
    def _to_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)
