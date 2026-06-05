"""
分段建模模块 / Mixture of Experts

核心思想：把数据空间切成多个区域，每个区域训练一个专家模型。

数学形式：
    ŷ = Σ w_k(x) · f_k(x)

两种模式：
1. 硬切分（Hard Partition）：
   - 用聚类/分位数/决策树把数据分区
   - 预测时只使用所属区域的模型
   
2. 软切分（Soft Partition / MoE）：
   - 用门控网络 g(x) 输出各区域权重
   - 预测时加权融合所有专家

适用场景：
- 不同销量区间规律不同（低价区线性，高价区非线性）
- 不同用户群体行为差异大
- 数据分布明显多模态
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor

from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


class PiecewiseEstimator(BaseEstimator, RegressorMixin):
    """
    分段估计器（硬切分）

    用法：
        # 按目标值分位数切分
        model = PiecewiseEstimator(
            partition_method='quantile',
            n_partitions=3,
            base_estimator=Ridge(),
            expert_estimator=LGBMRegressor()
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    """

    def __init__(self,
                 partition_method: str = 'quantile',
                 n_partitions: int = 3,
                 partition_col: Optional[str] = None,
                 base_estimator: Any = None,
                 expert_estimator: Any = None,
                 random_state: int = 42) -> None:
        """
        Args:
            partition_method: 'quantile' | 'kmeans' | 'tree' | 'user_defined'
            n_partitions: 分区数
            partition_col: 分区依据列（None=用目标值 y）
            base_estimator: 分区器用的基础模型
            expert_estimator: 各区域的专家模型
            random_state: 随机种子
        """
        self.partition_method = partition_method
        self.n_partitions = n_partitions
        self.partition_col = partition_col
        self.base_estimator = base_estimator
        self.expert_estimator = expert_estimator
        self.random_state = random_state

        self.partition_boundaries_: Optional[List[float]] = None
        self.experts_: Dict[int, Any] = {}
        self.labels_: Optional[np.ndarray] = None
        self._global_fallback: Optional[Any] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'PiecewiseEstimator':
        X = self._to_df(X)
        y = np.array(y).ravel()

        # 确定分区依据
        if self.partition_col and self.partition_col in X.columns:
            partition_values = X[self.partition_col].values
        else:
            partition_values = y

        # 生成分区标签
        labels = self._create_partitions(partition_values, X)
        self.labels_ = labels

        # 每个分区训练专家模型
        self.experts_ = {}
        unique_labels = np.unique(labels)

        for label in unique_labels:
            mask = labels == label
            X_part = X[mask]
            y_part = y[mask]

            if len(y_part) < 10:
                log_warning(f"[PiecewiseEstimator] 分区 {label} 样本过少 ({len(y_part)})，跳过")
                continue

            expert = self._create_expert()
            expert.fit(X_part, y_part)
            self.experts_[int(label)] = expert
            log_info(f"[PiecewiseEstimator] 分区 {label}: {len(y_part)} 样本, "
                     f"y_range=[{y_part.min():.2f}, {y_part.max():.2f}]")

        # 全局回退模型（处理空分区）
        self._global_fallback = self._create_expert()
        self._global_fallback.fit(X, y)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = self._to_df(X)

        # 确定每个样本所属分区
        if self.partition_col and self.partition_col in X.columns:
            partition_values = X[self.partition_col].values
        else:
            # 无法确定分区时，用 KMeans 预测分区
            if hasattr(self, '_kmeans'):
                labels = self._kmeans.predict(X.values)
            else:
                # 回退到全局模型
                return self._global_fallback.predict(X)

        # 按分区预测
        predictions = np.zeros(len(X))
        for label, expert in self.experts_.items():
            if self.partition_col and self.partition_col in X.columns:
                if self.partition_method == 'quantile':
                    mask = self._quantile_mask(partition_values, label)
                else:
                    mask = labels == label
            else:
                mask = labels == label

            if mask.sum() > 0:
                predictions[mask] = expert.predict(X[mask])

        # 未覆盖的样本用全局模型
        uncovered = predictions == 0
        if uncovered.sum() > 0:
            predictions[uncovered] = self._global_fallback.predict(X[uncovered])

        return predictions

    def _create_partitions(self, values: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """生成分区标签"""
        if self.partition_method == 'quantile':
            # 按分位数切分
            quantiles = np.linspace(0, 1, self.n_partitions + 1)
            self.partition_boundaries_ = [np.quantile(values, q) for q in quantiles]
            # 优化：使用 np.searchsorted 替代逐循环赋值，O(n log n) vs O(n*k)
            labels = np.searchsorted(self.partition_boundaries_[1:], values, side='right').clip(0, self.n_partitions - 1)
            return labels

        elif self.partition_method == 'kmeans':
            # KMeans 聚类分区
            from sklearn.cluster import KMeans
            self._kmeans = KMeans(n_clusters=self.n_partitions, random_state=self.random_state, n_init=10)
            values_2d = values.reshape(-1, 1) if values.ndim == 1 else values
            return self._kmeans.fit_predict(values_2d)

        elif self.partition_method == 'tree':
            # 决策树分区
            tree = DecisionTreeRegressor(max_leaf_nodes=self.n_partitions, random_state=self.random_state)
            X_tree = X.values if hasattr(X, 'values') else X
            tree.fit(X_tree, values)
            return tree.apply(X_tree)

        else:
            raise ValueError(f"未知分区方法: {self.partition_method}")

    def _quantile_mask(self, values: np.ndarray, label: int) -> np.ndarray:
        """判断值是否属于某个分位区间"""
        lower = self.partition_boundaries_[label]
        upper = self.partition_boundaries_[label + 1]
        if label == self.n_partitions - 1:
            return (values >= lower) & (values <= upper)
        return (values >= lower) & (values < upper)

    def _create_expert(self) -> Any:
        if self.expert_estimator is not None:
            return clone(self.expert_estimator)
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        except Exception:
            return Ridge(alpha=1.0, random_state=self.random_state)

    @staticmethod
    def _to_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class MixtureOfExperts(BaseEstimator, RegressorMixin):
    """
    混合专家模型（软切分）

    门控网络输出各专家权重，预测时加权融合：
        ŷ = Σ softmax(g(x))_k · f_k(x)

    用法：
        moe = MixtureOfExperts(
            n_experts=3,
            gate_estimator=LogisticRegression(),
            expert_estimator=LGBMRegressor()
        )
        moe.fit(X_train, y_train)
        y_pred = moe.predict(X_test)
    """

    def __init__(self,
                 n_experts: int = 3,
                 gate_estimator: Any = None,
                 expert_estimator: Any = None,
                 partition_method: str = 'kmeans',
                 random_state: int = 42) -> None:
        """
        Args:
            n_experts: 专家数量
            gate_estimator: 门控网络（分类器）
            expert_estimator: 专家模型
            partition_method: 'kmeans' | 'quantile'
            random_state: 随机种子
        """
        self.n_experts = n_experts
        self.gate_estimator = gate_estimator
        self.expert_estimator = expert_estimator
        self.partition_method = partition_method
        self.random_state = random_state

        self.gate_: Optional[Any] = None
        self.experts_: Dict[int, Any] = {}
        self.expert_weights_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'MixtureOfExperts':
        X = self._to_df(X)
        y = np.array(y).ravel()

        # 1. 用分区方法生成伪标签（作为门控网络的训练目标）
        pseudo_labels = self._create_pseudo_labels(y, X)

        # 2. 训练门控网络
        gate = self._create_gate()
        gate.fit(X, pseudo_labels)
        self.gate_ = gate

        # 3. 为每个专家分配样本（软分配：用门控概率加权）
        gate_proba = gate.predict_proba(X)

        # 4. 训练每个专家（用门控权重加权样本）
        self.experts_ = {}
        for k in range(self.n_experts):
            weights = gate_proba[:, k]
            # 只使用权重大于阈值的样本
            mask = weights > 0.1
            if mask.sum() < 10:
                log_warning(f"[MoE] 专家 {k} 样本过少 ({mask.sum()})，使用全部样本")
                mask = np.ones(len(X), dtype=bool)
                weights = np.ones(len(X)) / self.n_experts

            expert = self._create_expert()
            # 加权训练（如果模型支持 sample_weight）
            if hasattr(expert, 'fit') and 'sample_weight' in expert.fit.__code__.co_varnames:
                expert.fit(X[mask], y[mask], sample_weight=weights[mask])
            else:
                expert.fit(X[mask], y[mask])
            self.experts_[k] = expert

        # 记录专家权重分布
        self.expert_weights_ = gate_proba.mean(axis=0)
        log_info(f"[MoE] 专家平均权重: {self.expert_weights_}")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = self._to_df(X)
        if self.gate_ is None:
            raise ValueError("请先调用 fit()")

        gate_proba = self.gate_.predict_proba(X)
        predictions = np.zeros(len(X))

        for k in range(self.n_experts):
            expert_pred = self.experts_[k].predict(X)
            predictions += gate_proba[:, k] * expert_pred

        return predictions

    def predict_expert_weights(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """返回每个样本的专家权重（用于分析）"""
        X = self._to_df(X)
        return self.gate_.predict_proba(X)

    def _create_pseudo_labels(self, y: np.ndarray, X: pd.DataFrame) -> np.ndarray:
        """生成专家伪标签"""
        if self.partition_method == 'quantile':
            quantiles = np.linspace(0, 1, self.n_experts + 1)
            boundaries = [np.quantile(y, q) for q in quantiles]
            labels = np.zeros(len(y), dtype=int)
            for i in range(self.n_experts):
                lower = boundaries[i]
                upper = boundaries[i + 1]
                if i == self.n_experts - 1:
                    mask = (y >= lower) & (y <= upper)
                else:
                    mask = (y >= lower) & (y < upper)
                labels[mask] = i
            return labels
        else:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_experts, random_state=self.random_state, n_init=10)
            return kmeans.fit_predict(y.reshape(-1, 1))

    def _create_gate(self) -> Any:
        if self.gate_estimator is not None:
            return clone(self.gate_estimator)
        return LogisticRegression(max_iter=1000, random_state=self.random_state, multi_class='multinomial')

    def _create_expert(self) -> Any:
        if self.expert_estimator is not None:
            return clone(self.expert_estimator)
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        except Exception:
            return Ridge(alpha=1.0, random_state=self.random_state)

    @staticmethod
    def _to_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class AutoPiecewiseRouter:
    """
    自动分段路由器

    自动决定是否需要分段、分几段、用什么模型。
    """

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.best_model_: Optional[Any] = None
        self.is_piecewise_: bool = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'AutoPiecewiseRouter':
        X = np.array(X)
        y = np.array(y).ravel()

        # 测试全局模型
        global_model = Ridge(alpha=1.0, random_state=self.random_state)
        global_score = cross_val_score(global_model, X, y, cv=3, scoring='r2').mean()

        # 测试分段模型
        piecewise = PiecewiseEstimator(n_partitions=3, random_state=self.random_state)
        piecewise_score = cross_val_score(piecewise, X, y, cv=3, scoring='r2').mean()

        log_info(f"[AutoPiecewiseRouter] 全局模型 R2={global_score:.4f}, 分段模型 R2={piecewise_score:.4f}")

        if piecewise_score > global_score + 0.02:
            self.is_piecewise_ = True
            self.best_model_ = piecewise
            log_info("[AutoPiecewiseRouter] 选择分段模型")
        else:
            self.is_piecewise_ = False
            self.best_model_ = global_model
            log_info("[AutoPiecewiseRouter] 选择全局模型")

        self.best_model_.fit(X, y)
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.best_model_.predict(X)
