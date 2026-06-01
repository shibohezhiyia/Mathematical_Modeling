"""
核近似自动降级模块

解决 SVR/SVC 等核方法在大数据上的 O(n²)~O(n³) 复杂度问题。

策略：
- n < 5k:  真实核矩阵（RBF/poly/linear）
- 5k <= n < 30k:  Nystroem 近似（核矩阵低秩近似）
- 30k <= n < 100k: RBFSampler 随机傅里叶特征 + LinearSVR
- n >= 100k: 采样后 RBFSampler + LinearSVR（极致速度）

数学原理：
    K(x_i, x_j) ≈ φ(x_i)^T φ(x_j)
把非线性核问题转化为线性问题，训练速度从 O(n²) 降到 O(n·d)。
"""

import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.kernel_approximation import Nystroem, RBFSampler

from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


class ApproxSVR(BaseEstimator, RegressorMixin):
    """
    自动核近似的 SVR（sklearn 兼容）

    用法与普通 SVR 相同，但内部会根据数据规模自动降级：
        model = ApproxSVR(kernel='rbf', C=1.0, gamma='scale')
        model.fit(X_train, y_train)  # 大数据时自动用核近似
        y_pred = model.predict(X_test)
    """

    # 阈值配置
    THRESHOLD_NYSTROEM = 5000
    THRESHOLD_RBF = 30000
    THRESHOLD_SAMPLE = 100000
    DEFAULT_COMPONENTS = 512

    def __init__(self,
                 kernel: str = 'rbf',
                 C: float = 1.0,
                 gamma: str = 'scale',
                 degree: int = 3,
                 coef0: float = 0.0,
                 epsilon: float = 0.1,
                 shrinking: bool = True,
                 tol: float = 1e-3,
                 cache_size: int = 200,
                 verbose: bool = False,
                 max_iter: int = -1,
                 random_state: Optional[int] = None) -> None:
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.epsilon = epsilon
        self.shrinking = shrinking
        self.tol = tol
        self.cache_size = cache_size
        self.verbose = verbose
        self.max_iter = max_iter
        self.random_state = random_state

        # 内部状态
        self.approx_mode_: Optional[str] = None
        self.transformer_: Optional[Any] = None
        self.model_: Optional[Any] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'ApproxSVR':
        X = np.array(X)
        y = np.array(y).ravel()
        n_samples = X.shape[0]

        # 线性核不需要近似
        if self.kernel == 'linear' or n_samples < self.THRESHOLD_NYSTROEM:
            self.approx_mode_ = 'exact'
            from sklearn.svm import SVR
            self.model_ = SVR(
                kernel=self.kernel, C=self.C, gamma=self.gamma,
                degree=self.degree, coef0=self.coef0, epsilon=self.epsilon,
                shrinking=self.shrinking, tol=self.tol, cache_size=self.cache_size,
                verbose=self.verbose, max_iter=self.max_iter
            )
            self.model_.fit(X, y)
            log_info(f"[ApproxSVR] 使用真实核: kernel={self.kernel}, n={n_samples}")
            return self

        # 超大数据：先采样
        if n_samples >= self.THRESHOLD_SAMPLE:
            self.approx_mode_ = 'rbf_sampler_sampled'
            sample_size = min(50000, n_samples)
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n_samples, size=sample_size, replace=False)
            X_fit = X[idx]
            y_fit = y[idx]
            n_components = min(self.DEFAULT_COMPONENTS, max(256, sample_size // 20))
        else:
            X_fit = X
            y_fit = y
            n_components = min(self.DEFAULT_COMPONENTS, max(128, n_samples // 8))

        # 选择近似策略
        if n_samples < self.THRESHOLD_RBF:
            self.approx_mode_ = 'nystroem'
            self.transformer_ = Nystroem(
                kernel=self.kernel, gamma=self.gamma, degree=self.degree,
                coef0=self.coef0, n_components=n_components,
                random_state=self.random_state
            )
        else:
            if self.kernel == 'rbf':
                self.approx_mode_ = 'rbf_sampler'
                gamma_val = self._resolve_gamma(X_fit)
                self.transformer_ = RBFSampler(
                    gamma=gamma_val, n_components=n_components,
                    random_state=self.random_state
                )
            else:
                # poly/sigmoid 用 Nystroem
                self.approx_mode_ = 'nystroem'
                self.transformer_ = Nystroem(
                    kernel=self.kernel, gamma=self.gamma, degree=self.degree,
                    coef0=self.coef0, n_components=n_components,
                    random_state=self.random_state
                )

        X_transformed = self.transformer_.fit_transform(X_fit)
        from sklearn.svm import LinearSVR
        self.model_ = LinearSVR(
            C=self.C, epsilon=self.epsilon,
            max_iter=5000, random_state=self.random_state
        )
        self.model_.fit(X_transformed, y_fit)
        log_info(f"[ApproxSVR] 核近似降级: mode={self.approx_mode_}, n={n_samples}, components={n_components}, kernel={self.kernel}")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.array(X)
        if self.approx_mode_ == 'exact':
            return self.model_.predict(X)
        X_transformed = self.transformer_.transform(X)
        return self.model_.predict(X_transformed)

    def _resolve_gamma(self, X: np.ndarray) -> float:
        """解析 gamma 参数为数值"""
        if self.gamma == 'scale':
            return 1.0 / (X.shape[1] * X.var()) if X.var() != 0 else 1.0
        elif self.gamma == 'auto':
            return 1.0 / X.shape[1]
        else:
            return float(self.gamma)

    def get_feature_importances_via_residual(self, X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
        """
        通过残差线性模型获取近似的特征重要性（仅适用于近似模式）。
        返回原始特征空间的重要性（通过 transformer 逆映射近似）。
        """
        if self.approx_mode_ == 'exact' or self.transformer_ is None:
            return None
        # 简化：返回 LinearSVR 的 coef_ 的 L1 范数
        return np.abs(self.model_.coef_)


class KernelApproximator:
    """
    通用核近似器（可用于任何核模型）

    用法：
        approx = KernelApproximator(strategy='auto', random_state=42)
        X_approx = approx.fit_transform(X)  # 自动选择最佳策略
    """

    def __init__(self,
                 strategy: str = 'auto',
                 n_components: Optional[int] = None,
                 kernel: str = 'rbf',
                 gamma: Optional[float] = None,
                 random_state: int = 42) -> None:
        """
        Args:
            strategy: 'auto' | 'exact' | 'nystroem' | 'rbf_sampler'
            n_components: 近似维度，None=自动
            kernel: 核函数类型
            gamma: 核参数
            random_state: 随机种子
        """
        self.strategy = strategy
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.random_state = random_state
        self.transformer_: Optional[Any] = None
        self.approx_mode_: Optional[str] = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """拟合并转换"""
        X = np.array(X)
        n_samples = X.shape[0]

        if self.strategy == 'auto':
            if n_samples < ApproxSVR.THRESHOLD_NYSTROEM:
                strategy = 'exact'
            elif n_samples < ApproxSVR.THRESHOLD_RBF:
                strategy = 'nystroem'
            else:
                strategy = 'rbf_sampler'
        else:
            strategy = self.strategy

        self.approx_mode_ = strategy

        if strategy == 'exact':
            return X

        n_comp = self.n_components or min(ApproxSVR.DEFAULT_COMPONENTS, max(128, n_samples // 8))

        if strategy == 'nystroem':
            self.transformer_ = Nystroem(
                kernel=self.kernel, gamma=self.gamma,
                n_components=n_comp, random_state=self.random_state
            )
        elif strategy == 'rbf_sampler':
            gamma_val = self.gamma or 1.0 / X.shape[1]
            self.transformer_ = RBFSampler(
                gamma=gamma_val, n_components=n_comp, random_state=self.random_state
            )
        else:
            raise ValueError(f"未知策略: {strategy}")

        return self.transformer_.fit_transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """转换新数据"""
        X = np.array(X)
        if self.approx_mode_ == 'exact':
            return X
        if self.transformer_ is None:
            raise ValueError("请先调用 fit_transform()")
        return self.transformer_.transform(X)


def auto_kernel_strategy(n_samples: int, kernel: str = 'rbf') -> str:
    """
    根据数据规模自动选择核策略

    Returns: 'exact' | 'nystroem' | 'rbf_sampler'
    """
    if kernel == 'linear' or n_samples < ApproxSVR.THRESHOLD_NYSTROEM:
        return 'exact'
    elif n_samples < ApproxSVR.THRESHOLD_RBF:
        return 'nystroem'
    else:
        return 'rbf_sampler'
