"""
最优传输重加权模块 (Optimal Transport Reweighting)

核心思想：当训练集和测试集分布不同时，给训练样本加权，
         使加权后的训练分布接近测试分布。

数学形式：
    min Σ W_{ij} ||x_i^{train} - x_j^{test}||
    s.t. Σ_j W_{ij} = 1/N_train
         Σ_i W_{ij} = 1/N_test

然后每个训练样本的权重 = N_train * Σ_j W_{ij}

简化实现：
- 大数据：用 Sinkhorn 算法（熵正则化最优传输）
- 超大数据：用最近邻近似

用途：
- 领域适配（Domain Adaptation）
- Covariate Shift 修正
- 样本重要性加权
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


class OTReweighter(BaseEstimator):
    """
    最优传输重加权器

    用法：
        reweighter = OTReweighter(method='sinkhorn', reg=0.1)
        sample_weights = reweighter.fit(X_train, X_test).transform_weights()
        model.fit(X_train, y_train, sample_weight=sample_weights)
    """

    def __init__(self,
                 method: str = 'sinkhorn',
                 reg: float = 0.1,
                 max_samples: int = 5000,
                 standardize: bool = True,
                 random_state: int = 42) -> None:
        """
        Args:
            method: 'sinkhorn' | 'nearest_neighbor' | 'kernel_mean_matching'
            reg: Sinkhorn 正则化参数（越大越平滑，越小越精确）
            max_samples: 最大采样数（超大数据时子采样）
            standardize: 是否标准化
            random_state: 随机种子
        """
        self.method = method
        self.reg = reg
        self.max_samples = max_samples
        self.standardize = standardize
        self.random_state = random_state

        self.scaler_: Optional[Any] = None
        self.weights_: Optional[np.ndarray] = None
        self.transport_plan_: Optional[np.ndarray] = None

    def fit(self, X_source: Union[pd.DataFrame, np.ndarray],
            X_target: Union[pd.DataFrame, np.ndarray]) -> 'OTReweighter':
        """
        计算从 source（训练集）到 target（测试集）的最优传输权重。

        Returns:
            self (weights_ 已设置)
        """
        X_source = np.array(X_source, dtype=np.float64)
        X_target = np.array(X_target, dtype=np.float64)

        # 标准化
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_source = self.scaler_.fit_transform(X_source)
            X_target = self.scaler_.transform(X_target)

        # 大数据采样
        rng = np.random.RandomState(self.random_state)
        n_source = len(X_source)
        n_target = len(X_target)

        if n_source > self.max_samples:
            idx = rng.choice(n_source, self.max_samples, replace=False)
            X_source_sub = X_source[idx]
            source_indices = idx
        else:
            X_source_sub = X_source
            source_indices = np.arange(n_source)

        if n_target > self.max_samples:
            idx = rng.choice(n_target, self.max_samples, replace=False)
            X_target_sub = X_target[idx]
        else:
            X_target_sub = X_target

        # 计算权重
        if self.method == 'sinkhorn':
            weights_sub = self._sinkhorn_weights(X_source_sub, X_target_sub)
        elif self.method == 'nearest_neighbor':
            weights_sub = self._nn_weights(X_source_sub, X_target_sub)
        elif self.method == 'kernel_mean_matching':
            weights_sub = self._kmm_weights(X_source_sub, X_target_sub)
        else:
            raise ValueError(f"未知方法: {self.method}")

        # 映射回完整 source 集
        self.weights_ = np.ones(n_source)
        self.weights_[source_indices] = weights_sub

        # 裁剪极端权重
        q99 = np.quantile(self.weights_, 0.99)
        q01 = np.quantile(self.weights_, 0.01)
        self.weights_ = np.clip(self.weights_, q01, q99)
        self.weights_ = self.weights_ / self.weights_.mean()  # 归一化使均值为1

        log_info(f"[OTReweighter] {self.method}: 权重范围=[{self.weights_.min():.3f}, {self.weights_.max():.3f}], "
                 f"CV={self.weights_.std()/self.weights_.mean():.3f}")

        return self

    def transform_weights(self) -> np.ndarray:
        """返回训练样本权重"""
        if self.weights_ is None:
            raise ValueError("请先调用 fit()")
        return self.weights_

    def _sinkhorn_weights(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        """
        Sinkhorn 算法：熵正则化最优传输
        
        简化版实现（不依赖 POT 库，用 numpy 近似）。
        """
        n_source = len(X_source)
        n_target = len(X_target)

        # 计算代价矩阵（欧氏距离平方）
        C = self._pairwise_distances(X_source, X_target)
        C = C / C.max()  # 归一化

        # Sinkhorn 迭代
        a = np.ones(n_source) / n_source
        b = np.ones(n_target) / n_target
        K = np.exp(-C / self.reg)

        u = np.ones(n_source)
        v = np.ones(n_target)

        for _ in range(100):
            u = a / (K @ v)
            v = b / (K.T @ u)

        # 传输计划
        P = np.diag(u) @ K @ np.diag(v)

        # 每个 source 的权重 = 传输出去的质量总和 / (1/n_source)
        weights = P.sum(axis=1) * n_source
        return weights

    def _nn_weights(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        """
        最近邻近似：
        每个 source 样本的权重正比于 target 中邻居的密度。
        """
        n_source = len(X_source)

        # 在 target 上建 KNN
        knn = NearestNeighbors(n_neighbors=min(5, len(X_target)))
        knn.fit(X_target)

        # 查找 source 在 target 中的邻居距离
        distances, _ = knn.kneighbors(X_source)
        mean_dist = distances.mean(axis=1)

        # 距离越近 → 权重越高（指数衰减）
        weights = np.exp(-mean_dist / (mean_dist.mean() + 1e-6))
        weights = weights / weights.sum() * n_source
        return weights

    def _kmm_weights(self, X_source: np.ndarray, X_target: np.ndarray) -> np.ndarray:
        """
        Kernel Mean Matching (KMM)
        最小化：||E_source[w·φ(x)] - E_target[φ(x)]||
        """
        from sklearn.metrics.pairwise import rbf_kernel

        n_source = len(X_source)
        # 合并数据
        X_all = np.vstack([X_source, X_target])

        # 计算核矩阵（只计算 source-source 和 source-target）
        K_ss = rbf_kernel(X_source, X_source, gamma=0.1)
        K_st = rbf_kernel(X_source, X_target, gamma=0.1)

        # 目标：K_st @ 1 / n_target
        kappa = K_st.sum(axis=1) / len(X_target)

        # 近似求解：w = K_ss^{-1} @ kappa（加正则化）
        reg = 1e-3
        K_ss_reg = K_ss + reg * np.eye(n_source)
        try:
            # 数值稳定性：正则化 + 条件数检查
            n = K_ss.shape[0]
            K_ss_reg = K_ss + 1e-6 * np.eye(n)
            try:
                # 检查矩阵条件数
                s = np.linalg.svd(K_ss_reg, compute_uv=False)
                if s[-1] > 0:
                    cond = s[0] / s[-1]
                    if cond < 1e12:
                        # 条件良好，直接求解
                        weights = np.linalg.solve(K_ss_reg, kappa)
                    else:
                        # 条件数过高，使用最小二乘
                        weights = np.linalg.lstsq(K_ss_reg, kappa, rcond=None)[0]
                else:
                    # 完全奇异，回退到伪逆
                    weights = np.linalg.pinv(K_ss_reg) @ kappa
            except np.linalg.LinAlgError:
                weights = np.linalg.pinv(K_ss_reg) @ kappa
        except np.linalg.LinAlgError:
            weights = np.ones(n_source)

        # 裁剪和归一化
        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum() * n_source
        return weights

    @staticmethod
    def _pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """计算欧氏距离矩阵"""
        # 广播方式：(x-y)^2 = x^2 + y^2 - 2xy
        x2 = np.sum(X ** 2, axis=1).reshape(-1, 1)
        y2 = np.sum(Y ** 2, axis=1).reshape(1, -1)
        xy = X @ Y.T
        return np.maximum(x2 + y2 - 2 * xy, 0)


class DomainAdaptationSampler:
    """
    领域适配采样器

    从训练集中采样，使采样后的分布更接近测试集。
    """

    def __init__(self, n_samples: Optional[int] = None, method: str = 'ot',
                 random_state: int = 42) -> None:
        """
        Args:
            n_samples: 采样后的样本数，None=保持原数量
            method: 'ot' | 'importance'
            random_state: 随机种子
        """
        self.n_samples = n_samples
        self.method = method
        self.random_state = random_state
        self.weights_: Optional[np.ndarray] = None

    def fit_resample(self, X: Union[pd.DataFrame, np.ndarray],
                     y: Union[pd.Series, np.ndarray],
                     X_target: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        根据 target 分布重采样 source 数据。

        Returns:
            X_resampled, y_resampled
        """
        X = np.array(X)
        y = np.array(y).ravel()
        n_source = len(X)
        n_out = self.n_samples or n_source

        if self.method == 'ot':
            reweighter = OTReweighter(method='sinkhorn', random_state=self.random_state)
            weights = reweighter.fit(X, X_target).transform_weights()
        else:
            # 简单重要性采样：按权重概率抽取
            weights = np.ones(n_source)

        # 按权重概率采样（有放回）
        probs = weights / weights.sum()
        rng = np.random.RandomState(self.random_state)
        indices = rng.choice(n_source, size=n_out, replace=True, p=probs)

        return X[indices], y[indices]
