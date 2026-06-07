"""
高级降维模块：稀疏化 + 低秩近似

解决高维数据（尤其 One-Hot 编码后）的维度灾难：
- 强线性相关：PCA / IncrementalPCA
- 稀疏高维类别：TruncatedSVD
- 非线性结构：KernelPCA
- 大规模流式数据：IncrementalPCA / MiniBatchDictionaryLearning
- 有监督降维：PLS / 基于模型的特征选择

数学原理：
    X ≈ U Σ V^T   (SVD 低秩近似)
    X_sparse ≈ W H   (NMF 非负矩阵分解)
"""

import warnings
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD, IncrementalPCA, KernelPCA, NMF
from sklearn.feature_selection import SelectKBest, mutual_info_regression, mutual_info_classif
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.utils.extmath import randomized_svd

from utils.helpers import log_info

warnings.filterwarnings('ignore')


class AutoDimensionReducer(BaseEstimator, TransformerMixin):
    """
    自动降维器

    根据数据特点自动选择最佳降维策略：
    - 稀疏矩阵 → TruncatedSVD
    - 大数据 → IncrementalPCA
    - 非线性结构 → KernelPCA
    - 一般数据 → PCA

    用法：
        reducer = AutoDimensionReducer(target_dim=50, strategy='auto')
        X_reduced = reducer.fit_transform(X)
    """

    def __init__(self,
                 target_dim: Optional[int] = None,
                 strategy: str = 'auto',
                 variance_threshold: float = 0.95,
                 random_state: int = 42) -> None:
        """
        Args:
            target_dim: 目标维度，None=按方差阈值自动决定
            strategy: 'auto' | 'pca' | 'truncated_svd' | 'incremental_pca' | 'kernel_pca' | 'nmf'
            variance_threshold: 保留的方差比例（用于 auto 策略）
            random_state: 随机种子
        """
        self.target_dim = target_dim
        self.strategy = strategy
        self.variance_threshold = variance_threshold
        self.random_state = random_state

        self.reducer_: Optional[Any] = None
        self.scaler_: Optional[Any] = None
        self.actual_strategy_: Optional[str] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) -> 'AutoDimensionReducer':
        X = np.array(X)
        n_samples, n_features = X.shape

        # 自动选择策略
        strategy = self._choose_strategy(X, self.strategy)
        self.actual_strategy_ = strategy

        # 确定目标维度
        target_dim = self.target_dim or min(n_samples, n_features)
        target_dim = min(target_dim, n_samples, n_features)

        if strategy == 'pca':
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            if self.target_dim is None:
                # 按方差阈值决定维度
                pca_full = PCA(random_state=self.random_state)
                pca_full.fit(X_scaled)
                cumsum = np.cumsum(pca_full.explained_variance_ratio_)
                target_dim = int(np.argmax(cumsum >= self.variance_threshold) + 1)
                target_dim = max(2, min(target_dim, n_features))
            self.reducer_ = PCA(n_components=target_dim, random_state=self.random_state)
            self.reducer_.fit(X_scaled)
            self.explained_variance_ratio_ = self.reducer_.explained_variance_ratio_

        elif strategy == 'incremental_pca':
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            batch_size = min(1000, n_samples)
            target_dim = self.target_dim or min(50, n_features)
            self.reducer_ = IncrementalPCA(n_components=target_dim, batch_size=batch_size)
            self.reducer_.fit(X_scaled)
            self.explained_variance_ratio_ = self.reducer_.explained_variance_ratio_

        elif strategy == 'truncated_svd':
            target_dim = self.target_dim or min(100, n_features)
            self.reducer_ = TruncatedSVD(n_components=target_dim, random_state=self.random_state)
            self.reducer_.fit(X)
            self.explained_variance_ratio_ = self.reducer_.explained_variance_ratio_

        elif strategy == 'kernel_pca':
            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X)
            target_dim = self.target_dim or min(50, n_features)
            self.reducer_ = KernelPCA(n_components=target_dim, kernel='rbf', random_state=self.random_state)
            self.reducer_.fit(X_scaled)

        elif strategy == 'nmf':
            # NMF 要求非负
            X_nonneg = np.clip(X, 0, None)
            target_dim = self.target_dim or min(50, n_features)
            self.reducer_ = NMF(n_components=target_dim, random_state=self.random_state, max_iter=500)
            self.reducer_.fit(X_nonneg)

        log_info(f"[AutoDimensionReducer] 策略={strategy}, 输入={n_features}, 输出={target_dim}")
        if self.explained_variance_ratio_ is not None:
            cumvar = np.sum(self.explained_variance_ratio_)
            log_info(f"[AutoDimensionReducer] 保留方差比例={cumvar:.2%}")

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.array(X)
        if self.reducer_ is None:
            raise ValueError("请先调用 fit()")

        if self.scaler_ is not None:
            X = self.scaler_.transform(X)

        if self.actual_strategy_ == 'nmf':
            X = np.clip(X, 0, None)

        return self.reducer_.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray], y: Optional[np.ndarray] = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def _choose_strategy(self, X: np.ndarray, strategy: str) -> str:
        if strategy != 'auto':
            return strategy

        n_samples, n_features = X.shape

        # 稀疏矩阵 → TruncatedSVD
        if hasattr(X, 'nnz'):
            return 'truncated_svd'
        # 对密集数组，采样估计稀疏度而非全量遍历（O(n*m) → O(sample))
        if isinstance(X, np.ndarray) and X.size > 0:
            sample_size = min(10000, X.size)
            flat = X.flat
            indices = np.random.choice(X.size, sample_size, replace=False)
            zero_ratio = np.sum(flat[indices] == 0) / sample_size
            if zero_ratio > 0.5:
                return 'truncated_svd'

        # 大数据 → IncrementalPCA
        if n_samples > 100000:
            return 'incremental_pca'

        # 高维但样本少 → TruncatedSVD
        if n_features > n_samples * 2:
            return 'truncated_svd'

        # 默认 PCA
        return 'pca'


class SparseFeatureSelector(BaseEstimator, TransformerMixin):
    """
    稀疏特征选择器

    用 L1 正则化自动筛选关键特征，输出稀疏模型。
    """

    def __init__(self,
                 task_type: str = 'regression',
                 alpha: float = 0.01,
                 max_features: Optional[int] = None,
                 random_state: int = 42) -> None:
        """
        Args:
            task_type: 'regression' | 'classification'
            alpha: L1 正则强度（越大越稀疏）
            max_features: 最大保留特征数
            random_state: 随机种子
        """
        self.task_type = task_type
        self.alpha = alpha
        self.max_features = max_features
        self.random_state = random_state
        self.selector_: Optional[Any] = None
        self.selected_features_: Optional[List[int]] = None
        self.feature_scores_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'SparseFeatureSelector':
        X = np.array(X)
        y = np.array(y).ravel()

        if self.task_type == 'regression':
            model = Lasso(alpha=self.alpha, random_state=self.random_state, max_iter=5000)
        else:
            model = LogisticRegression(
                penalty='l1', C=1.0 / (self.alpha + 1e-6),
                solver='saga', random_state=self.random_state, max_iter=5000
            )

        model.fit(X, y)
        coef = np.abs(model.coef_)
        if coef.ndim > 1:
            coef = coef.sum(axis=0)

        self.feature_scores_ = coef

        # 选择非零特征
        non_zero = np.where(coef > 1e-10)[0]

        # 如果非零特征太多，按重要性截断
        if self.max_features and len(non_zero) > self.max_features:
            top_idx = np.argsort(coef)[-self.max_features:]
            non_zero = np.sort(top_idx)

        self.selected_features_ = non_zero.tolist()

        log_info(f"[SparseFeatureSelector] 从 {X.shape[1]} 维筛选到 {len(self.selected_features_)} 维")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.array(X)
        if self.selected_features_ is None:
            raise ValueError("请先调用 fit()")
        return X[:, self.selected_features_]

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        return self.fit(X, y).transform(X)


class LowRankApproximator(BaseEstimator, TransformerMixin):
    """
    低秩近似器

    用随机 SVD 对大矩阵做低秩近似：X ≈ U_r Σ_r V_r^T
    比标准 SVD 快得多，适合大规模数据。
    """

    def __init__(self, n_components: int = 100, random_state: int = 42) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.U_: Optional[np.ndarray] = None
        self.S_: Optional[np.ndarray] = None
        self.Vt_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> 'LowRankApproximator':
        X = np.array(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # 随机 SVD
        self.U_, self.S_, self.Vt_ = randomized_svd(
            X_centered, n_components=self.n_components,
            random_state=self.random_state
        )
        log_info(f"[LowRankApproximator] 随机 SVD: {X.shape} → 秩{self.n_components}")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.array(X, dtype=np.float64)
        X_centered = X - self.mean_
        # 投影到低维空间: X @ V_r
        return X_centered @ self.Vt_.T

    def inverse_transform(self, X_reduced: np.ndarray) -> np.ndarray:
        """从低维重建"""
        return X_reduced @ self.Vt_ + self.mean_

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.fit(X).transform(X)


class SupervisedDimReducer(BaseEstimator, TransformerMixin):
    """
    有监督降维器

    结合目标变量 y 做降维：
    - 互信息选择
    - PLS（Partial Least Squares）
    """

    def __init__(self,
                 method: str = 'mutual_info',
                 n_components: int = 10,
                 task_type: str = 'regression',
                 random_state: int = 42) -> None:
        """
        Args:
            method: 'mutual_info' | 'pls'
            n_components: 输出维度
            task_type: 'regression' | 'classification'
        """
        self.method = method
        self.n_components = n_components
        self.task_type = task_type
        self.random_state = random_state
        self.reducer_: Optional[Any] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'SupervisedDimReducer':
        X = np.array(X)
        y = np.array(y).ravel()
        n_features = X.shape[1]
        k = min(self.n_components, n_features)

        if self.method == 'mutual_info':
            score_func = mutual_info_regression if self.task_type == 'regression' else mutual_info_classif
            self.reducer_ = SelectKBest(score_func=score_func, k=k)
            self.reducer_.fit(X, y)
        elif self.method == 'pls':
            from sklearn.cross_decomposition import PLSRegression
            self.reducer_ = PLSRegression(n_components=k)
            self.reducer_.fit(X, y)
        else:
            raise ValueError(f"未知方法: {self.method}")

        log_info(f"[SupervisedDimReducer] {self.method}: 保留 top-{k} 特征")
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.array(X)
        if self.reducer_ is None:
            raise ValueError("请先调用 fit()")
        return self.reducer_.transform(X)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray],
                      y: Union[pd.Series, np.ndarray]) -> np.ndarray:
        return self.fit(X, y).transform(X)
