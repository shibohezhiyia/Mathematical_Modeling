"""
图分解模块：KNN 相似度图 → 社区发现 → 子图分别建模

核心思想：
1. 用 KNN 构建样本相似度图
2. 用社区发现（谱聚类 / Louvain）将图分成若干子图
3. 每个子图训练一个专家模型
4. 预测时根据样本所属社区或社区相似度加权

数学流程：
    G = KNNGraph(X, k=10)
    Communities = SpectralClustering(G, n_clusters=K)
    for c in Communities:
        f_c = ExpertModel(X_c, y_c)
    ŷ(x) = f_{community(x)}(x)  [硬分配]
    ŷ(x) = Σ softmax(-dist(x, center_c)) · f_c(x)  [软加权]

适用场景：
- 数据天然聚类（不同客户群体、不同区域）
- 全局模型难以拟合所有局部模式
- 样本间相似度有意义（如用户画像、商品属性）
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


class KNNGraphBuilder:
    """
    KNN 图构建器

    从特征矩阵构建 k-近邻图（无向）。
    """

    def __init__(self, n_neighbors: int = 10, metric: str = 'euclidean',
                 weighted: bool = True, kernel: Optional[str] = None) -> None:
        """
        Args:
            n_neighbors: KNN 邻居数
            metric: 距离度量
            weighted: 是否用距离加权边
            kernel: 'rbf' | 'cosine' | None（用原始距离）
        """
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.weighted = weighted
        self.kernel = kernel

    def build(self, X: np.ndarray) -> np.ndarray:
        """
        构建邻接矩阵

        Returns:
            adjacency: (n_samples, n_samples) 对称矩阵
        """
        X = np.array(X)
        n_samples = X.shape[0]

        # KNN
        knn = NearestNeighbors(n_neighbors=min(self.n_neighbors + 1, n_samples),
                               metric=self.metric)
        knn.fit(X)
        distances, indices = knn.kneighbors(X)

        # 构建邻接矩阵
        adjacency = np.zeros((n_samples, n_samples))

        # 向量化构建邻接矩阵
        for i in range(n_samples):
            neighbors = indices[i][1:]  # 排除自身
            dists = distances[i][1:]

            if self.kernel == 'rbf':
                # 向量化：避免循环内的中位数计算，改用全局中位数
                gamma = 1.0 / (X.shape[1] * np.median(distances[distances > 0]) ** 2 + 1e-6)
                weights = np.exp(-gamma * dists ** 2)
            elif self.kernel == 'cosine':
                weights = np.ones(len(neighbors))
            elif self.weighted:
                # 距离倒数加权
                weights = 1.0 / (dists + 1e-6)
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(neighbors))

            adjacency[i, neighbors] = weights

        # 对称化：max(A, A^T)
        adjacency = np.maximum(adjacency, adjacency.T)
        return adjacency


class CommunityDetector:
    """
    社区发现器

    将图分成若干社区。
    """

    def __init__(self, method: str = 'spectral', n_communities: int = 5,
                 random_state: int = 42) -> None:
        """
        Args:
            method: 'spectral' | 'kmeans' | 'louvain'
            n_communities: 社区数量
            random_state: 随机种子
        """
        self.method = method
        self.n_communities = n_communities
        self.random_state = random_state
        self.labels_: Optional[np.ndarray] = None
        self.centers_: Optional[np.ndarray] = None

    def detect(self, X: np.ndarray, adjacency: Optional[np.ndarray] = None) -> np.ndarray:
        """
        检测社区

        Returns:
            labels: (n_samples,) 社区标签
        """
        X = np.array(X)
        n_samples = X.shape[0]

        if self.method == 'spectral':
            if adjacency is None:
                builder = KNNGraphBuilder(n_neighbors=min(10, n_samples - 1))
                adjacency = builder.build(X)
            # 拉普拉斯矩阵
            degree = np.diag(adjacency.sum(axis=1))
            laplacian = degree - adjacency
            # 谱聚类
            clustering = SpectralClustering(
                n_clusters=min(self.n_communities, n_samples),
                affinity='precomputed',
                random_state=self.random_state,
                assign_labels='kmeans'
            )
            self.labels_ = clustering.fit_predict(adjacency)

        elif self.method == 'kmeans':
            kmeans = KMeans(n_clusters=min(self.n_communities, n_samples),
                            random_state=self.random_state, n_init=10)
            self.labels_ = kmeans.fit_predict(X)
            self.centers_ = kmeans.cluster_centers_

        elif self.method == 'louvain':
            # 尝试用 networkx + python-louvain，如果没有则回退到 spectral
            try:
                import networkx as nx
                import community as community_louvain
                if adjacency is None:
                    builder = KNNGraphBuilder(n_neighbors=min(10, n_samples - 1))
                    adjacency = builder.build(X)
                G = nx.from_numpy_array(adjacency)
                partition = community_louvain.best_partition(G, random_state=self.random_state)
                self.labels_ = np.array([partition[i] for i in range(n_samples)])
                self.n_communities = len(np.unique(self.labels_))
            except ImportError:
                log_warning("[CommunityDetector] Louvain 不可用（缺少 networkx/python-louvain），回退到 Spectral")
                return self.detect(X, adjacency)

        # 计算社区中心
        self.centers_ = np.array([
            X[self.labels_ == c].mean(axis=0) for c in np.unique(self.labels_)
        ])
        
        # 新增：计算社区质量指标（轮廓系数）
        from sklearn.metrics import silhouette_score
        if len(np.unique(self.labels_)) > 1 and len(np.unique(self.labels_)) < n_samples:
            try:
                self.silhouette_score_ = silhouette_score(X, self.labels_)
                log_info(f"[CommunityDetector] 社区轮廓系数: {self.silhouette_score_:.3f}")
            except Exception:
                self.silhouette_score_ = None
        
        # 新增：自适应社区数调整 - 如果轮廓系数过低，尝试减少社区数
        if hasattr(self, 'silhouette_score_') and self.silhouette_score_ is not None and self.silhouette_score_ < 0.1:
            if self.n_communities > 2:
                log_warning(f"[CommunityDetector] 轮廓系数 {self.silhouette_score_:.3f} 过低，尝试减少社区数")
                self.n_communities = max(2, self.n_communities - 1)
                return self.detect(X, adjacency)

        log_info(f"[CommunityDetector] {self.method}: 发现 {self.n_communities} 个社区")
        for c in np.unique(self.labels_):
            log_info(f"  社区 {c}: {np.sum(self.labels_ == c)} 个样本")

        return self.labels_


class GraphDecomposer(BaseEstimator, RegressorMixin):
    """
    图分解估计器

    先建图 → 社区发现 → 每个社区训练专家 → 预测时硬分配或软加权。

    用法：
        model = GraphDecomposer(
            n_communities=5,
            community_method='spectral',
            expert_estimator=LGBMRegressor(),
            prediction_mode='soft'
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    """

    def __init__(self,
                 n_communities: int = 5,
                 community_method: str = 'spectral',
                 knn_neighbors: int = 10,
                 expert_estimator: Any = None,
                 prediction_mode: str = 'soft',
                 scaler: bool = True,
                 random_state: int = 42) -> None:
        """
        Args:
            n_communities: 社区数量
            community_method: 'spectral' | 'kmeans' | 'louvain'
            knn_neighbors: KNN 邻居数
            expert_estimator: 专家模型
            prediction_mode: 'hard'（硬分配） | 'soft'（按社区相似度加权）
            scaler: 是否标准化
            random_state: 随机种子
        """
        self.n_communities = n_communities
        self.community_method = community_method
        self.knn_neighbors = knn_neighbors
        self.expert_estimator = expert_estimator
        self.prediction_mode = prediction_mode
        self.scaler = scaler
        self.random_state = random_state

        self.scaler_: Optional[Any] = None
        self.detector_: Optional[CommunityDetector] = None
        self.experts_: Dict[int, Any] = {}
        self.global_model_: Optional[Any] = None
        self.labels_: Optional[np.ndarray] = None
        self.X_train_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'GraphDecomposer':
        X = np.array(X)
        y = np.array(y).ravel()
        self.X_train_ = X

        if self.scaler:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)

        # 1. 社区发现
        self.detector_ = CommunityDetector(
            method=self.community_method,
            n_communities=self.n_communities,
            random_state=self.random_state
        )
        self.labels_ = self.detector_.detect(X)
        self.n_communities = len(np.unique(self.labels_))  # 实际社区数

        # 2. 每个社区训练专家
        self.experts_ = {}
        for c in np.unique(self.labels_):
            mask = self.labels_ == c
            X_c = X[mask]
            y_c = y[mask]

            if len(y_c) < 5:
                log_warning(f"[GraphDecomposer] 社区 {c} 样本过少 ({len(y_c)})，跳过")
                continue

            expert = self._create_expert()
            expert.fit(X_c, y_c)
            self.experts_[int(c)] = expert
            log_info(f"[GraphDecomposer] 社区 {c} 专家训练完成: {len(y_c)} 样本")

        # 3. 全局回退模型
        self.global_model_ = self._create_expert()
        self.global_model_.fit(X, y)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.array(X)
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        if self.prediction_mode == 'hard':
            return self._predict_hard(X_scaled)
        else:
            return self._predict_soft(X_scaled)

    def _predict_hard(self, X: np.ndarray) -> np.ndarray:
        """硬分配：每个样本分配到最近的社区中心，用对应专家预测"""
        predictions = np.zeros(len(X))

        # 用 KMeans 的预测分配社区（如果 detector 支持）
        if hasattr(self.detector_, 'labels_') and self.detector_.centers_ is not None:
            # 计算到各社区中心的距离
            distances = np.linalg.norm(X[:, np.newaxis, :] - self.detector_.centers_[np.newaxis, :, :], axis=2)
            assigned_communities = np.argmin(distances, axis=1)
        else:
            # 回退：用 KNN 在训练集上找最近邻居的社区
            knn = NearestNeighbors(n_neighbors=min(5, len(self.X_train_)))
            knn.fit(self.X_train_)
            _, indices = knn.kneighbors(X)
            assigned_communities = np.array([
                np.bincount(self.labels_[idx]).argmax() for idx in indices
            ])

        for c in np.unique(assigned_communities):
            if c not in self.experts_:
                continue
            mask = assigned_communities == c
            if mask.sum() > 0:
                predictions[mask] = self.experts_[c].predict(X[mask])

        # 未覆盖的用全局模型
        uncovered = np.array([c not in self.experts_ for c in assigned_communities])
        if uncovered.sum() > 0:
            predictions[uncovered] = self.global_model_.predict(X[uncovered])

        return predictions

    def _predict_soft(self, X: np.ndarray) -> np.ndarray:
        """软加权：按到社区中心的距离加权所有专家的预测"""
        if self.detector_.centers_ is None:
            return self.global_model_.predict(X)

        # 计算到各社区中心的距离
        centers = self.detector_.centers_
        distances = np.linalg.norm(X[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)

        # 距离转相似度（softmax 负距离）
        similarities = np.exp(-distances / (distances.std() + 1e-6))
        similarities = similarities / (similarities.sum(axis=1, keepdims=True) + 1e-10)

        predictions = np.zeros(len(X))
        for c, expert in self.experts_.items():
            if c < similarities.shape[1]:
                pred_c = expert.predict(X)
                predictions += similarities[:, c] * pred_c

        return predictions

    def _create_expert(self) -> Any:
        if self.expert_estimator is not None:
            return clone(self.expert_estimator)
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        except Exception:
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0, random_state=self.random_state)


class GraphBasedEnsemble:
    """
    基于图的集成器

    结合图分解和集成学习：
    - 先图分解得到社区
    - 每个社区训练模型
    - 最终用社区表现加权融合
    """

    def __init__(self, base_estimators: Optional[List[Any]] = None,
                 n_communities: int = 3, random_state: int = 42) -> None:
        self.base_estimators = base_estimators
        self.n_communities = n_communities
        self.random_state = random_state
        self.models_: List[Any] = []
        self.weights_: Optional[np.ndarray] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'GraphBasedEnsemble':
        X = np.array(X)
        y = np.array(y).ravel()

        # 图分解
        decomposer = GraphDecomposer(
            n_communities=self.n_communities,
            prediction_mode='hard',
            random_state=self.random_state
        )
        decomposer.fit(X, y)

        # 收集各社区模型作为基学习器
        self.models_ = list(decomposer.experts_.values())
        if not self.models_:
            self.models_ = [decomposer.global_model_]

        # 计算权重（基于各社区在验证集上的表现）
        self.weights_ = np.ones(len(self.models_)) / len(self.models_)

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.array(X)
        preds = np.array([m.predict(X) for m in self.models_])
        return np.average(preds, axis=0, weights=self.weights_)
