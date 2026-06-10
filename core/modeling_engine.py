"""
建模引擎核心

提供完整的建模能力：
1. 自动任务类型判断（分类/回归/聚类）
2. 丰富的模型库（传统ML + 统计模型 + 树模型）
3. 统一模型接口
4. 自动编码（OneHot / Label / Target / Ordinal）
5. 自动特征选择
6. K折交叉验证
7. 多模型融合
8. 数据切分与评估
"""

import os
import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from copy import deepcopy
from enum import Enum

import numpy as np
from sklearn.metrics import pairwise_kernels
from sklearn.kernel_approximation import Nystroem, RBFSampler
from core.kernel_cache import KernelCache
import pandas as pd
from sklearn.model_selection import (
    StratifiedKFold, KFold
)
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder, LabelEncoder, OrdinalEncoder, PolynomialFeatures
)
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, mutual_info_regression,
    VarianceThreshold, SelectFromModel, RFE
)
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)
from sklearn.base import BaseEstimator, clone

from core.progress_bar import progress_iter
from core.smart_early_stopper import FoldEarlyStopper, FoldEarlyStopConfig
from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from joblib import Parallel, delayed
    _JOBLIB_AVAILABLE = True
except ImportError:
    _JOBLIB_AVAILABLE = False

# Module-level kernel cache (disk-backed under workspace if diskcache available)
_DEFAULT_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'cache', 'kernel_cache'))
KERNEL_CACHE = KernelCache(cache_dir=_DEFAULT_CACHE_DIR)
KERNEL_APPROX_THRESHOLD = 3000
KERNEL_APPROX_COMPONENTS = 512


def _create_kernel_approximation(X_tr, X_val, orig_kernel, kernel_params, n_components: int = KERNEL_APPROX_COMPONENTS):
    """
    Create approximate kernel feature maps for large datasets.
    
    自适应 n_components：根据数据有效秩（effective rank）调整组件数，
    避免过度近似或不足近似。
    """
    # 自适应调整 n_components
    n_samples = X_tr.shape[0]
    n_features = X_tr.shape[1]
    
    # 计算有效秩：基于数据的奇异值衰减
    if n_samples > 100 and n_features > 10:
        try:
            # 使用随机 SVD 快速估计有效秩
            from sklearn.utils.extmath import randomized_svd
            # 只关心奇异值 S：U/Vt 不需要，用 _ 占位避免一次性分配 (n, k) 和 (k, m) 数组
            _, S, _ = randomized_svd(X_tr.values, n_components=min(50, n_samples, n_features), random_state=42)
            # 有效秩：累积奇异值能量达到 90% 的位置
            total_energy = float(np.sum(S ** 2))
            cumsum = np.cumsum(S ** 2)
            effective_rank = int(np.searchsorted(cumsum, 0.9 * total_energy)) + 1
            # 自适应 n_components：有效秩的 1.5~3 倍，但不超过样本数的 1/4
            adaptive_n = min(max(int(effective_rank * 2), 128), n_samples // 4, KERNEL_APPROX_COMPONENTS)
            n_components = min(n_components, adaptive_n)
        except Exception:
            pass
    
    if orig_kernel == 'rbf':
        transformer = RBFSampler(
            gamma=kernel_params.get('gamma', 1.0),
            n_components=n_components,
            random_state=42
        )
    else:
        transformer = Nystroem(
            kernel=orig_kernel,
            degree=kernel_params.get('degree', 3),
            gamma=kernel_params.get('gamma', 1.0),
            coef0=kernel_params.get('coef0', 1.0),
            n_components=n_components,
            random_state=42
        )
    X_tr_approx = transformer.fit_transform(X_tr.values)
    X_val_approx = transformer.transform(X_val.values)
    return X_tr_approx, X_val_approx, transformer


# =============================================================================
# 枚举与常量
# =============================================================================

class TaskType(Enum):
    """任务类型"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    UNKNOWN = "unknown"


class EncodingType(Enum):
    """编码类型"""
    AUTO = "auto"
    ONEHOT = "onehot"
    LABEL = "label"
    ORDINAL = "ordinal"
    TARGET = "target"
    FREQUENCY = "frequency"
    NONE = "none"


class FeatureSelectionStrategy(Enum):
    """特征选择策略"""
    VARIANCE = "variance_threshold"
    MI = "mutual_information"
    MI_KNN = "mutual_information_knn"  # k-NN估计的互信息，更稳定
    RFE = "recursive_feature_elimination"
    MODEL_BASED = "model_based"
    CORRELATION = "correlation_filter"
    PCA_DIM = "pca_dimensionality"
    PCA_RANDOMIZED = "pca_randomized_svd"  # 随机SVD加速PCA
    NONE = "none"


class EnsembleMethod(Enum):
    """融合方法"""
    VOTING_HARD = "voting_hard"
    VOTING_SOFT = "voting_soft"
    WEIGHTED = "weighted_average"
    STACKING = "stacking"
    BEST_SINGLE = "best_single"


# =============================================================================
# 任务类型自动判断
# =============================================================================

class TaskTypeDetector:
    """自动判断任务类型：分类 / 回归 / 聚类"""
    
    @staticmethod
    def detect(y: Optional[Union[pd.Series, np.ndarray]] = None,
               X: Optional[Union[pd.DataFrame, np.ndarray]] = None,
               user_hint: Optional[str] = None) -> TaskType:
        """
        自动判断任务类型
        
        逻辑：
        1. 用户显式指定 → 直接返回
        2. y 为 None → 聚类
        3. y 数值型：
           - 唯一值 <= 10 或 唯一值比例 < 5% → 分类
           - 否则 → 回归
        4. y 非数值型 → 分类
        5. y 只有0/1 → 二分类
        """
        if user_hint:
            try:
                return TaskType(user_hint.lower())
            except ValueError:
                pass
        
        if y is None:
            return TaskType.CLUSTERING
        
        y_series = pd.Series(y).dropna()
        
        if len(y_series) == 0:
            return TaskType.UNKNOWN
        
        # 非数值型 → 分类
        if not pd.api.types.is_numeric_dtype(y_series):
            return TaskType.CLASSIFICATION
        
        n_unique = y_series.nunique()
        n_total = len(y_series)
        unique_ratio = n_unique / n_total
        
        # 只有0/1或整数且唯一值很少 → 分类
        if n_unique <= 2:
            return TaskType.CLASSIFICATION
        
        if n_unique <= 10:
            return TaskType.CLASSIFICATION
        
        # 唯一值比例极低 → 分类（可能是整数编码的多分类）
        if unique_ratio < 0.05 and n_unique <= 100:
            return TaskType.CLASSIFICATION
        
        # 否则 → 回归
        return TaskType.REGRESSION
    
    @staticmethod
    def get_metrics_dict(task_type: TaskType) -> Dict[str, Callable]:
        """获取默认评估指标"""
        if task_type == TaskType.CLASSIFICATION:
            return {
                'accuracy': accuracy_score,
                'f1_macro': lambda y, p: f1_score(y, p, average='macro', zero_division=0),
                'f1_weighted': lambda y, p: f1_score(y, p, average='weighted', zero_division=0),
                'precision_macro': lambda y, p: precision_score(y, p, average='macro', zero_division=0),
                'recall_macro': lambda y, p: recall_score(y, p, average='macro', zero_division=0),
            }
        elif task_type == TaskType.REGRESSION:
            return {
                'rmse': lambda y, p: np.sqrt(mean_squared_error(y, p)),
                'mae': mean_absolute_error,
                'r2': r2_score,
                'mse': mean_squared_error,
                'mape': lambda y, p: np.mean(np.abs((y - p) / (y + 1e-8))) * 100,
            }
        elif task_type == TaskType.CLUSTERING:
            return {
                'silhouette': silhouette_score,
                'calinski_harabasz': calinski_harabasz_score,
                'davies_bouldin': davies_bouldin_score,
            }
        return {}
    
    @staticmethod
    def get_primary_metric(task_type: TaskType) -> str:
        if task_type == TaskType.CLASSIFICATION:
            return 'f1_weighted'
        elif task_type == TaskType.REGRESSION:
            return 'rmse'
        elif task_type == TaskType.CLUSTERING:
            return 'silhouette'
        return ''


# =============================================================================
# 自动编码器
# =============================================================================

class AutoEncoder:
    """
    智能编码器
    
    根据基数和任务类型自动选择编码策略：
    - 二值类别 → Label Encoding
    - 低基数(<=10) → One-Hot Encoding
    - 中基数(11~50) → Ordinal Encoding（有序）或 One-Hot（无序）
    - 高基数(>50) → Target Encoding / Frequency Encoding
    - 聚类任务 → 必须数值化，Label Encoding
    """
    
    def __init__(self,
                 onehot_max_categories: int = 10,
                 target_encode_max: int = 500,
                 ordinal_categories: Optional[Dict[str, List]] = None) -> None:
        self.onehot_max = onehot_max_categories
        self.target_encode_max = target_encode_max
        self.ordinal_hint = ordinal_categories or {}
        
        self._encoders: Dict[str, Any] = {}
        self._encoding_map: Dict[str, EncodingType] = {}
        self._target_mean: Optional[Dict] = None
        # 预计算 LABEL/TARGET/FREQUENCY 在 transform 阶段要用的查表，避免每次 transform 重新构造
        self._label_value_maps: Dict[str, Dict[str, int]] = {}  # col -> {原始值: 整数编码}
        self._fallback_values: Dict[str, float] = {}  # col -> TARGET/FREQUENCY 未知值的全局均值
        self._fitted = False
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AutoEncoder':
        """学习编码规则"""
        self._encoders = {}
        self._encoding_map = {}
        self._target_mean = {}
        self._label_value_maps = {}
        self._fallback_values = {}

        for col in X.columns:
            # 单次 X[col] 取出 + 缓存 dtype：原代码 X[col].dtype 两次访问
            col_series = X[col]
            col_dtype = col_series.dtype
            if col_dtype == object or str(col_dtype) == 'category':
                n_unique = col_series.nunique()
                
                # 判断编码策略
                strategy = self._choose_strategy(col, n_unique, y is not None)
                self._encoding_map[col] = strategy
                
                if strategy == EncodingType.ONEHOT:
                    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                    enc.fit(X[[col]].astype(str))
                    self._encoders[col] = enc
                    
                elif strategy == EncodingType.LABEL:
                    enc = LabelEncoder()
                    enc.fit(X[col].astype(str))
                    self._encoders[col] = enc
                    # 预计算 {原始值: 整数编码} 查表，transform 时直接 dict 查 O(1)，
                    # 避免每列 N+1 次 enc.transform([v]) 反复调用
                    self._label_value_maps[col] = {str(v): int(i) for i, v in enumerate(enc.classes_)}
                    
                elif strategy == EncodingType.ORDINAL:
                    if col in self.ordinal_hint:
                        categories = [self.ordinal_hint[col]]
                        enc = OrdinalEncoder(categories=categories, handle_unknown='use_encoded_value', unknown_value=-1)
                    else:
                        enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
                    enc.fit(X[[col]].astype(str))
                    self._encoders[col] = enc
                    
                elif strategy == EncodingType.TARGET and y is not None:
                    # 目标编码：用每个类别的目标均值
                    df_tmp = pd.DataFrame({col: X[col].astype(str), 'target': y})
                    self._target_mean[col] = df_tmp.groupby(col)['target'].mean().to_dict()
                    # 预计算未知值的全局均值 fallback
                    vals = list(self._target_mean[col].values())
                    self._fallback_values[col] = float(np.mean(vals)) if vals else 0.0
                    
                elif strategy == EncodingType.FREQUENCY:
                    self._target_mean[col] = X[col].value_counts(normalize=True).to_dict()
                    vals = list(self._target_mean[col].values())
                    self._fallback_values[col] = float(np.mean(vals)) if vals else 0.0
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """执行编码
        
        优化：批量收集所有需要 drop 和 concat 的列，一次性合并，避免多次 pd.concat 内存开销。
        LABEL/TARGET/FREQUENCY 直接用 fit 阶段预计算的查表，避免 transform 时 N+1 次
        enc.transform / np.mean(list(mapping.values())) 重复工作。
        """
        if not self._fitted:
            raise ValueError("请先调用 fit()")
        
        X_out = X.copy()
        # 批量收集需要删除和新增的数据
        cols_to_drop = []
        dfs_to_concat = []
        
        for col, strategy in self._encoding_map.items():
            if col not in X_out.columns:
                continue
            
            if strategy == EncodingType.ONEHOT:
                enc = self._encoders[col]
                encoded = enc.transform(X_out[[col]].astype(str))
                names = [f"{col}_{cat}" for cat in enc.categories_[0]]
                df_enc = pd.DataFrame(encoded, columns=names, index=X_out.index)
                cols_to_drop.append(col)
                dfs_to_concat.append(df_enc)
                
            elif strategy == EncodingType.LABEL:
                # 直接用 fit 阶段预计算的 {原始值: 整数} 查表（O(1) 哈希查找）
                # 未知值（包括 NaN 和未出现过的类别）填 -1，与原 enc.transform 行为一致
                value_map = self._label_value_maps.get(col, {})
                X_out[col] = X_out[col].astype(str).map(value_map).fillna(-1).astype(int)
                
            elif strategy == EncodingType.ORDINAL:
                enc = self._encoders[col]
                encoded = enc.transform(X_out[[col]].astype(str))
                X_out[col] = encoded.flatten()
                
            elif strategy in (EncodingType.TARGET, EncodingType.FREQUENCY):
                mapping = self._target_mean.get(col, {})
                # 用 fit 阶段缓存的全局均值，避免每列重新构造 list + np.mean
                global_mean = self._fallback_values.get(col, 0.0)
                X_out[col] = X_out[col].astype(str).map(mapping).fillna(global_mean)
        
        # 一次性合并所有 one-hot 编码结果（避免多次 pd.concat 的内存碎片）
        if cols_to_drop:
            X_out = X_out.drop(columns=cols_to_drop)
        if dfs_to_concat:
            # 使用 copy=False 减少内存拷贝
            X_out = pd.concat([X_out] + dfs_to_concat, axis=1, copy=False)
        
        return X_out
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)
    
    def _choose_strategy(self, col: str, n_unique: int, has_target: bool) -> EncodingType:
        """选择编码策略"""
        if n_unique <= 2:
            return EncodingType.LABEL
        
        if col in self.ordinal_hint:
            return EncodingType.ORDINAL
        
        if n_unique <= self.onehot_max:
            return EncodingType.ONEHOT
        
        if n_unique <= 50:
            return EncodingType.ORDINAL
        
        if has_target and n_unique <= self.target_encode_max:
            return EncodingType.TARGET
        
        return EncodingType.FREQUENCY
    
    def get_encoding_report(self) -> pd.DataFrame:
        """获取编码报告"""
        if not self._encoding_map:
            return pd.DataFrame()
        
        rows = []
        for col, strategy in self._encoding_map.items():
            rows.append({
                'column': col,
                'strategy': strategy.value,
                'encoder_type': type(self._encoders.get(col)).__name__ if col in self._encoders else 'mapping'
            })
        return pd.DataFrame(rows)


# =============================================================================
# 自动特征选择
# =============================================================================

class AutoFeatureSelector:
    """
    自动特征选择器
    
    支持策略：
    - variance: 方差阈值（删除低方差特征）
    - mutual_info: 互信息选择
    - rfe: 递归特征消除
    - model_based: 基于模型重要性
    - correlation: 相关性过滤（高共线性）
    - pca: PCA降维
    """
    
    def __init__(self,
                 strategy: FeatureSelectionStrategy = FeatureSelectionStrategy.MI,
                 n_features: Optional[int] = None,
                 variance_threshold: float = 0.01,
                 correlation_threshold: float = 0.95) -> None:
        self.strategy = strategy
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        
        self._selector = None
        self._selected_features: List[str] = []
        self._feature_scores: Dict[str, float] = {}
        self._fitted = False
        self._datetime_info: Dict[str, List[str]] = {}  # {orig_col: [derived_cols]}
    
    def _convert_datetime_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """将 datetime64 列自动拆分为数值特征，避免 numpy dtype 提升错误"""
        X = X.copy()
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                dt = pd.to_datetime(X[col], errors='coerce')
                derived = {}
                derived[f'{col}_year'] = dt.dt.year.astype('float64')
                derived[f'{col}_month'] = dt.dt.month.astype('float64')
                derived[f'{col}_day'] = dt.dt.day.astype('float64')
                derived[f'{col}_dayofweek'] = dt.dt.dayofweek.astype('float64')
                derived[f'{col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype('float64')
                derived[f'{col}_quarter'] = dt.dt.quarter.astype('float64')
                if dt.dt.hour.nunique() > 1 or dt.dt.hour.notna().any():
                    derived[f'{col}_hour'] = dt.dt.hour.astype('float64')
                # 记录映射关系
                self._datetime_info[col] = list(derived.keys())
                for new_col, new_series in derived.items():
                    X[new_col] = new_series
                X = X.drop(columns=[col])
        return X
    
    def fit(self, X: pd.DataFrame, y: pd.Series, task_type: TaskType) -> 'AutoFeatureSelector':
        """学习特征选择规则"""
        X = self._convert_datetime_features(X)
        
        if self.strategy == FeatureSelectionStrategy.NONE:
            self._selected_features = list(X.columns)
            self._fitted = True
            return self
        
        n_features = self.n_features or max(int(X.shape[1] * 0.8), 1)
        
        if self.strategy == FeatureSelectionStrategy.VARIANCE:
            self._selector = VarianceThreshold(threshold=self.variance_threshold)
            self._selector.fit(X)
            mask = self._selector.get_support()
            self._selected_features = [c for c, m in zip(X.columns, mask) if m]
            
        elif self.strategy == FeatureSelectionStrategy.MI:
            # 前置检查：MI对类别数接近样本数的数据会崩溃（每个类别<2样本时KDTree报错）
            y_series = pd.Series(y).reset_index(drop=True)
            min_class_count = y_series.value_counts().min() if len(y_series) > 0 else 0
            n_classes = y_series.nunique()
            n_samples = len(y_series)
            
            # 回退条件：类别数过多或某类别样本不足，MI无法可靠计算
            should_fallback = (n_classes > 1 and min_class_count < 2) or (n_samples > 0 and n_classes / n_samples > 0.5)
            
            if should_fallback:
                log_warning(f"[AutoFeatureSelector] MI不适用（{n_samples}样本/{n_classes}类别，最少类仅{min_class_count}个），回退到方差阈值")
                self._selector = VarianceThreshold(threshold=self.variance_threshold)
                self._selector.fit(X)
                mask = self._selector.get_support()
                self._selected_features = [c for c, m in zip(X.columns, mask) if m]
            else:
                try:
                    if task_type == TaskType.REGRESSION:
                        score_func = mutual_info_regression
                    else:
                        score_func = mutual_info_classif
                    self._selector = SelectKBest(score_func=score_func, k=min(n_features, X.shape[1]))
                    self._selector.fit(X, y)
                    scores = self._selector.scores_
                    self._feature_scores = {c: float(s) for c, s in zip(X.columns, scores)}
                    mask = self._selector.get_support()
                    self._selected_features = [c for c, m in zip(X.columns, mask) if m]
                except ValueError as e:
                    if "0 sample" in str(e) or " Found array with" in str(e):
                        log_warning(f"[AutoFeatureSelector] MI计算失败: {e}，回退到方差阈值")
                        self._selector = VarianceThreshold(threshold=self.variance_threshold)
                        self._selector.fit(X)
                        mask = self._selector.get_support()
                        self._selected_features = [c for c, m in zip(X.columns, mask) if m]
                    else:
                        raise
            
        elif self.strategy == FeatureSelectionStrategy.CORRELATION:
            self._selected_features = self._correlation_filter(X)
            
        elif self.strategy == FeatureSelectionStrategy.PCA_DIM:
            self._selector = PCA(n_components=min(n_features, X.shape[0], X.shape[1]))
            self._selector.fit(X)
            self._selected_features = list(X.columns)  # PCA保持所有但降维
            
        elif self.strategy == FeatureSelectionStrategy.PCA_RANDOMIZED:
            # 随机SVD加速PCA：适合高维大数据，比标准PCA快数倍
            n_comp = min(n_features, X.shape[0], X.shape[1])
            try:
                from sklearn.utils.extmath import randomized_svd
                U, S, Vt = randomized_svd(X.values, n_components=n_comp, random_state=42)
                # 使用随机SVD结果构建近似PCA
                self._selector = PCA(n_components=n_comp, svd_solver='randomized', random_state=42)
                self._selector.fit(X)
                self._selected_features = list(X.columns)
                log_info(f"[AutoFeatureSelector] 随机SVD PCA: {n_comp} 组件")
            except Exception as e:
                log_warning(f"[AutoFeatureSelector] 随机SVD失败: {e}，回退到标准PCA")
                self._selector = PCA(n_components=n_comp)
                self._selector.fit(X)
                self._selected_features = list(X.columns)
            
        elif self.strategy == FeatureSelectionStrategy.MI_KNN:
            # k-NN估计的互信息：比直方图法更稳定，适合连续变量
            try:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                # 使用k-NN估计（通过设置n_neighbors参数）
                if task_type == TaskType.REGRESSION:
                    score_func = lambda X, y: mutual_info_regression(X, y, n_neighbors=5, random_state=42)
                else:
                    score_func = lambda X, y: mutual_info_classif(X, y, n_neighbors=5, random_state=42)
                self._selector = SelectKBest(score_func=score_func, k=min(n_features, X.shape[1]))
                self._selector.fit(X, y)
                scores = self._selector.scores_
                self._feature_scores = {c: float(s) for c, s in zip(X.columns, scores)}
                mask = self._selector.get_support()
                self._selected_features = [c for c, m in zip(X.columns, mask) if m]
                log_info(f"[AutoFeatureSelector] k-NN MI特征选择: {len(self._selected_features)} 特征")
            except Exception as e:
                log_warning(f"[AutoFeatureSelector] k-NN MI失败: {e}，回退到标准MI")
                if task_type == TaskType.REGRESSION:
                    score_func = mutual_info_regression
                else:
                    score_func = mutual_info_classif
                self._selector = SelectKBest(score_func=score_func, k=min(n_features, X.shape[1]))
                self._selector.fit(X, y)
                scores = self._selector.scores_
                self._feature_scores = {c: float(s) for c, s in zip(X.columns, scores)}
                mask = self._selector.get_support()
                self._selected_features = [c for c, m in zip(X.columns, mask) if m]
            
        elif self.strategy == FeatureSelectionStrategy.MODEL_BASED:
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            if task_type == TaskType.REGRESSION:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            self._selector = SelectFromModel(model, max_features=n_features)
            self._selector.fit(X, y)
            mask = self._selector.get_support()
            self._selected_features = [c for c, m in zip(X.columns, mask) if m]
            if hasattr(model, 'feature_importances_'):
                self._feature_scores = {c: float(s) for c, s in zip(X.columns, model.feature_importances_)}
        
        self._fitted = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """执行特征选择"""
        if not self._fitted:
            raise ValueError("请先调用 fit()")
        
        X = self._convert_datetime_features(X)
        
        if self.strategy == FeatureSelectionStrategy.PCA_DIM and self._selector:
            pca_result = self._selector.transform(X)
            cols = [f"pca_{i}" for i in range(pca_result.shape[1])]
            return pd.DataFrame(pca_result, columns=cols, index=X.index)
        
        return X[self._selected_features]
    
    def fit_transform(self, X: pd.DataFrame, y: pd.Series, task_type: TaskType) -> pd.DataFrame:
        return self.fit(X, y, task_type).transform(X)
    
    def _correlation_filter(self, X: pd.DataFrame) -> List[str]:
        """相关性过滤：删除高共线性特征
        
        优化：使用向量化 max 替代 any() 的 Python 循环。
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # 向量化：使用 max(axis=0) 替代 any() 的 Python 循环
        to_drop = upper.columns[upper.max(axis=0) > self.correlation_threshold].tolist()
        return [c for c in X.columns if c not in to_drop]
    
    def get_selected_features(self) -> List[str]:
        return self._selected_features
    
    def get_feature_importance(self) -> pd.DataFrame:
        """获取特征重要性"""
        if not self._feature_scores:
            return pd.DataFrame()
        df = pd.DataFrame([
            {'feature': k, 'score': v}
            for k, v in sorted(self._feature_scores.items(), key=lambda x: x[1], reverse=True)
        ])
        return df


# =============================================================================
# 模型库
# =============================================================================

@dataclass
class ModelSpec:
    """模型规格"""
    name: str
    key: str
    model_class: Any
    category: str  # 'tree', 'linear', 'svm', 'ensemble', 'statistical', 'neural', 'clustering'
    supports_gpu: bool = False
    supports_partial_fit: bool = False
    supports_sample_weight: bool = False
    is_probabilistic: bool = False
    default_params: Dict = field(default_factory=dict)
    hyperparam_space: Dict[str, List] = field(default_factory=dict)


class ModelLibrary:
    """
    模型库
    
    注册模型：
    - 分类：LogisticRegression, SVM, KNN, NaiveBayes, DecisionTree, 
            RandomForest, ExtraTrees, GradientBoosting,
            XGBoost, LightGBM, CatBoost, MLP
    - 回归：LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
            HuberRegressor, PLSRegression, TheilSenRegressor,
            SVR, KNN, DecisionTree, RandomForest, ExtraTrees,
            GradientBoosting, XGBoost, LightGBM, CatBoost, MLP
    - 聚类：KMeans, DBSCAN, Agglomerative, GaussianMixture, Spectral
    """
    
    _models: Dict[str, Dict[str, ModelSpec]] = {
        TaskType.CLASSIFICATION.value: {},
        TaskType.REGRESSION.value: {},
        TaskType.CLUSTERING.value: {},
    }
    _initialized = False
    _missing_modules: set = set()
    
    @classmethod
    def _warn_once(cls, module: str, message: str):
        """同一缺失模块只警告一次，避免重复日志淹没输出"""
        if module not in cls._missing_modules:
            cls._missing_modules.add(module)
            log_warning(message)
    
    @classmethod
    def refresh(cls) -> None:
        """重新初始化模型库（用于安装新依赖后刷新）"""
        cls._initialized = False
        cls._missing_modules.clear()
        cls._models = {
            TaskType.CLASSIFICATION.value: {},
            TaskType.REGRESSION.value: {},
            TaskType.CLUSTERING.value: {},
        }
        cls._init()
    
    @classmethod
    def _init(cls) -> None:
        if cls._initialized:
            return
        
        # ========== 分类模型 ==========
        
        # 线性/统计模型
        try:
            from sklearn.linear_model import LogisticRegression
            cls._register('classification', 'lr', 'LogisticRegression',
                          LogisticRegression, 'linear',
                          default_params={'max_iter': 1000, 'random_state': 42},
                          hyperparam_space={
                              'C': {'type': 'float', 'low': 1e-3, 'high': 1e3, 'scale': 'log'},
                              'penalty': ['l1', 'l2'],
                              'solver': ['liblinear', 'lbfgs'],
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] LogisticRegression 注册失败: {e}")
        
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
            cls._register('classification', 'lda', 'LDA',
                          LinearDiscriminantAnalysis, 'statistical')
            cls._register('classification', 'qda', 'QDA',
                          QuadraticDiscriminantAnalysis, 'statistical')
        except Exception as e:
            log_warning(f"[ModelLibrary] QDA 注册失败: {e}")
        
        try:
            from sklearn.naive_bayes import GaussianNB
            cls._register('classification', 'nb', 'NaiveBayes',
                          GaussianNB, 'statistical')
        except Exception as e:
            log_warning(f"[ModelLibrary] NaiveBayes 注册失败: {e}")
        
        # SVM
        try:
            from sklearn.svm import SVC
            cls._register('classification', 'svm', 'SVM',
                          SVC, 'svm',
                          default_params={
                              'probability': True,
                              'random_state': 42,
                              'cache_size': 2000,
                              'tol': 1e-2,
                              'max_iter': 10000,
                              'shrinking': True,
                          },
                          hyperparam_space={
                              'C': {'type': 'float', 'low': 1e-2, 'high': 1e2, 'scale': 'log'},
                              'kernel': ['rbf', 'linear'],
                              'gamma': {'type': 'float', 'low': 1e-4, 'high': 1.0, 'scale': 'log'},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] SVM 注册失败: {e}")
        
        # KNN
        try:
            from sklearn.neighbors import KNeighborsClassifier
            cls._register('classification', 'knn', 'KNN',
                          KNeighborsClassifier, 'ensemble',
                          default_params={'n_jobs': -1},
                          hyperparam_space={
                              'n_neighbors': {'type': 'int', 'low': 2, 'high': 20},
                              'weights': ['uniform', 'distance'],
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] KNN 注册失败: {e}")
        
        # 树模型
        try:
            from sklearn.tree import DecisionTreeClassifier
            cls._register('classification', 'dt', 'DecisionTree',
                          DecisionTreeClassifier, 'tree',
                          default_params={'random_state': 42},
                          hyperparam_space={
                              'max_depth': {'type': 'int', 'low': 2, 'high': 20},
                              'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] DecisionTree 注册失败: {e}")
        
        try:
            from sklearn.ensemble import (
                RandomForestClassifier, ExtraTreesClassifier,
                GradientBoostingClassifier
            )
            cls._register('classification', 'rf', 'RandomForest',
                          RandomForestClassifier, 'ensemble',
                          default_params={'n_estimators': 200, 'random_state': 42, 'n_jobs': -1},
                          hyperparam_space={
                              'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                              'max_depth': {'type': 'int', 'low': 3, 'high': 30},
                              'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                          })
            cls._register('classification', 'et', 'ExtraTrees',
                          ExtraTreesClassifier, 'ensemble',
                          default_params={'n_estimators': 200, 'random_state': 42, 'n_jobs': -1},
                          hyperparam_space={
                              'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                              'max_depth': {'type': 'int', 'low': 3, 'high': 30},
                          })
            cls._register('classification', 'gbdt', 'GradientBoosting',
                          GradientBoostingClassifier, 'ensemble',
                          default_params={'n_estimators': 200, 'random_state': 42},
                          hyperparam_space={
                              'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                              'max_depth': {'type': 'int', 'low': 2, 'high': 10},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] GradientBoosting 注册失败: {e}")
        
        # 梯度提升库
        try:
            from xgboost import XGBClassifier
            cls._register('classification', 'xgb', 'XGBoost',
                          XGBClassifier, 'ensemble',
                          default_params={'n_estimators': 1000, 'random_state': 42, 'n_jobs': -1},
                          hyperparam_space={
                              'max_depth': {'type': 'int', 'low': 2, 'high': 8},
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                              'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
                          },
                          supports_gpu=True)
        except Exception as e:
            cls._warn_once('xgboost', f"[ModelLibrary] XGBoost 未安装，跳过注册: {e}")
        
        try:
            from lightgbm import LGBMClassifier
            cls._register('classification', 'lgb', 'LightGBM',
                          LGBMClassifier, 'ensemble',
                          default_params={'n_estimators': 1000, 'random_state': 42, 'verbose': -1, 'n_jobs': -1},
                          hyperparam_space={
                              'num_leaves': {'type': 'int', 'low': 15, 'high': 64},
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                              'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
                          },
                          supports_gpu=True)
        except Exception as e:
            cls._warn_once('lightgbm', f"[ModelLibrary] LightGBM 未安装，跳过注册: {e}")
        
        try:
            from catboost import CatBoostClassifier
            cls._register('classification', 'catboost', 'CatBoost',
                          CatBoostClassifier, 'ensemble',
                          default_params={'iterations': 1000, 'verbose': False, 'random_seed': 42},
                          hyperparam_space={
                              'depth': {'type': 'int', 'low': 3, 'high': 8},
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                          },
                          supports_gpu=True)
        except Exception as e:
            cls._warn_once('catboost', f"[ModelLibrary] CatBoost 未安装，跳过注册: {e}")
        
        # 神经网络
        try:
            from sklearn.neural_network import MLPClassifier
            cls._register('classification', 'mlp', 'MLP',
                          MLPClassifier, 'neural',
                          default_params={'max_iter': 500, 'random_state': 42},
                          hyperparam_space={
                              'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                              'alpha': {'type': 'float', 'low': 1e-5, 'high': 0.1, 'scale': 'log'},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] MLP 注册失败: {e}")
        
        # HistGradientBoosting (sklearn)
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            cls._register('classification', 'hist_gb', 'HistGradientBoosting',
                          HistGradientBoostingClassifier, 'ensemble',
                          default_params={'random_state': 42},
                          hyperparam_space={
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                              'max_depth': {'type': 'int', 'low': 2, 'high': 8},
                              'max_iter': {'type': 'int', 'low': 50, 'high': 300},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] HistGradientBoosting 注册失败: {e}")
        
        # SGD
        try:
            from sklearn.linear_model import SGDClassifier
            cls._register('classification', 'sgd', 'SGD',
                          SGDClassifier, 'linear',
                          default_params={'max_iter': 1000, 'random_state': 42, 'n_jobs': -1},
                          hyperparam_space={
                              'loss': ['log_loss', 'hinge', 'modified_huber'],
                              'alpha': {'type': 'float', 'low': 1e-5, 'high': 0.1, 'scale': 'log'},
                              'penalty': ['l2', 'l1', 'elasticnet'],
                          },
                          supports_partial_fit=True)
        except Exception as e:
            log_warning(f"[ModelLibrary] SGD 注册失败: {e}")
        
        # ========== 回归模型 ==========
        
        # 线性/统计模型
        try:
            from sklearn.linear_model import (
                LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
            )
            cls._register('regression', 'linear', 'LinearRegression',
                          LinearRegression, 'linear')
            cls._register('regression', 'ridge', 'Ridge',
                          Ridge, 'linear',
                          default_params={'random_state': 42},
                          hyperparam_space={'alpha': {'type': 'float', 'low': 1e-3, 'high': 1e2, 'scale': 'log'}})
            cls._register('regression', 'lasso', 'Lasso',
                          Lasso, 'linear',
                          default_params={'random_state': 42, 'max_iter': 2000},
                          hyperparam_space={'alpha': {'type': 'float', 'low': 1e-4, 'high': 1.0, 'scale': 'log'}})
            cls._register('regression', 'elastic', 'ElasticNet',
                          ElasticNet, 'linear',
                          default_params={'random_state': 42, 'max_iter': 2000},
                          hyperparam_space={
                              'alpha': {'type': 'float', 'low': 1e-4, 'high': 1.0, 'scale': 'log'},
                              'l1_ratio': {'type': 'float', 'low': 0.1, 'high': 0.9}
                          })
            cls._register('regression', 'huber', 'HuberRegressor',
                          HuberRegressor, 'linear')
        except Exception as e:
            log_warning(f"[ModelLibrary] HuberRegressor 注册失败: {e}")
        
        try:
            from sklearn.linear_model import BayesianRidge, ARDRegression
            cls._register('regression', 'bayesian_ridge', 'BayesianRidge',
                          BayesianRidge, 'statistical')
            cls._register('regression', 'ard', 'ARDRegression',
                          ARDRegression, 'statistical')
        except Exception as e:
            log_warning(f"[ModelLibrary] ARDRegression 注册失败: {e}")
        
        try:
            from sklearn.cross_decomposition import PLSRegression
            cls._register('regression', 'pls', 'PLSRegression',
                          PLSRegression, 'statistical',
                          default_params={'n_components': 2},
                          hyperparam_space={'n_components': {'type': 'int', 'low': 1, 'high': 10}})
        except Exception as e:
            log_warning(f"[ModelLibrary] PLSRegression 注册失败: {e}")
        
        try:
            from sklearn.linear_model import TheilSenRegressor
            cls._register('regression', 'theilsen', 'TheilSenRegressor',
                          TheilSenRegressor, 'statistical')
        except Exception as e:
            log_warning(f"[ModelLibrary] TheilSenRegressor 注册失败: {e}")
        
        # SVM
        try:
            from sklearn.svm import SVR, LinearSVR
            from core.kernel_approx import ApproxSVR
            # ApproxSVR: 自动核近似的 SVR，大数据时自动降级为线性近似
            cls._register('regression', 'svm', 'SVR',
                          ApproxSVR, 'svm',
                          default_params={
                              'cache_size': 2000,
                              'tol': 1e-2,
                              'max_iter': 10000,
                          },
                          hyperparam_space={
                              'C': {'type': 'float', 'low': 1e-2, 'high': 1e2, 'scale': 'log'},
                              'kernel': ['rbf', 'linear'],
                              'gamma': {'type': 'float', 'low': 1e-4, 'high': 1.0, 'scale': 'log'},
                          })
            # LinearSVR: 纯线性 SVR（始终快，但无核能力）
            cls._register('regression', 'linear_svm', 'LinearSVR',
                          LinearSVR, 'linear',
                          default_params={'max_iter': 2000, 'tol': 1e-3},
                          hyperparam_space={
                              'C': {'type': 'float', 'low': 1e-3, 'high': 1e2, 'scale': 'log'},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] SVR 注册失败: {e}")
        
        # 残差模型（回归）
        try:
            from core.residual_modeling import ResidualEstimator
            cls._register('regression', 'residual_stack', 'ResidualStack',
                          ResidualEstimator, 'ensemble',
                          default_params={'cv': 5, 'random_state': 42},
                          hyperparam_space={
                              'cv': {'type': 'int', 'low': 2, 'high': 10},
                          })
            log_info("[ModelLibrary] ResidualEstimator 注册成功")
        except Exception as e:
            log_warning(f"[ModelLibrary] ResidualEstimator 注册失败: {e}")
        
        # KNN
        try:
            from sklearn.neighbors import KNeighborsRegressor
            cls._register('regression', 'knn', 'KNN',
                          KNeighborsRegressor, 'ensemble',
                          default_params={'n_jobs': -1},
                          hyperparam_space={
                              'n_neighbors': {'type': 'int', 'low': 2, 'high': 20},
                              'weights': ['uniform', 'distance'],
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] KNN 注册失败: {e}")
        
        # 树模型
        try:
            from sklearn.tree import DecisionTreeRegressor
            cls._register('regression', 'dt', 'DecisionTree',
                          DecisionTreeRegressor, 'tree',
                          default_params={'random_state': 42},
                          hyperparam_space={
                              'max_depth': {'type': 'int', 'low': 2, 'high': 20},
                              'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] DecisionTree 注册失败: {e}")
        
        try:
            from sklearn.ensemble import (
                RandomForestRegressor, ExtraTreesRegressor,
                GradientBoostingRegressor
            )
            cls._register('regression', 'rf', 'RandomForest',
                          RandomForestRegressor, 'ensemble',
                          default_params={'n_estimators': 200, 'random_state': 42, 'n_jobs': -1},
                          hyperparam_space={
                              'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                              'max_depth': {'type': 'int', 'low': 3, 'high': 30},
                              'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                          })
            cls._register('regression', 'et', 'ExtraTrees',
                          ExtraTreesRegressor, 'ensemble',
                          default_params={'n_estimators': 200, 'random_state': 42, 'n_jobs': -1},
                          hyperparam_space={
                              'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                              'max_depth': {'type': 'int', 'low': 3, 'high': 30},
                          })
            cls._register('regression', 'gbdt', 'GradientBoosting',
                          GradientBoostingRegressor, 'ensemble',
                          default_params={'n_estimators': 200, 'random_state': 42},
                          hyperparam_space={
                              'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                              'max_depth': {'type': 'int', 'low': 2, 'high': 10},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] GradientBoosting 注册失败: {e}")
        
        # 梯度提升库
        try:
            from xgboost import XGBRegressor
            cls._register('regression', 'xgb', 'XGBoost',
                          XGBRegressor, 'ensemble',
                          default_params={'n_estimators': 1000, 'random_state': 42, 'n_jobs': -1},
                          hyperparam_space={
                              'max_depth': {'type': 'int', 'low': 2, 'high': 8},
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                              'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
                          },
                          supports_gpu=True)
        except Exception as e:
            cls._warn_once('xgboost', f"[ModelLibrary] XGBoost 未安装，跳过注册: {e}")
        
        try:
            from lightgbm import LGBMRegressor
            cls._register('regression', 'lgb', 'LightGBM',
                          LGBMRegressor, 'ensemble',
                          default_params={'n_estimators': 1000, 'random_state': 42, 'verbose': -1, 'n_jobs': -1},
                          hyperparam_space={
                              'num_leaves': {'type': 'int', 'low': 15, 'high': 64},
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                              'subsample': {'type': 'float', 'low': 0.5, 'high': 1.0},
                          },
                          supports_gpu=True)
        except Exception as e:
            cls._warn_once('lightgbm', f"[ModelLibrary] LightGBM 未安装，跳过注册: {e}")
        
        try:
            from catboost import CatBoostRegressor
            cls._register('regression', 'catboost', 'CatBoost',
                          CatBoostRegressor, 'ensemble',
                          default_params={'iterations': 1000, 'verbose': False, 'random_seed': 42},
                          hyperparam_space={
                              'depth': {'type': 'int', 'low': 3, 'high': 8},
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                          },
                          supports_gpu=True)
        except Exception as e:
            cls._warn_once('catboost', f"[ModelLibrary] CatBoost 未安装，跳过注册: {e}")
        
        # 分段建模 / Mixture of Experts
        try:
            from core.piecewise_model import PiecewiseEstimator, MixtureOfExperts
            cls._register('regression', 'piecewise', 'Piecewise',
                          PiecewiseEstimator, 'ensemble',
                          default_params={'n_partitions': 3, 'random_state': 42},
                          hyperparam_space={
                              'n_partitions': {'type': 'int', 'low': 2, 'high': 5},
                          })
            cls._register('regression', 'moe', 'MixtureOfExperts',
                          MixtureOfExperts, 'ensemble',
                          default_params={'n_experts': 3, 'random_state': 42},
                          hyperparam_space={
                              'n_experts': {'type': 'int', 'low': 2, 'high': 5},
                          })
            log_info("[ModelLibrary] PiecewiseEstimator / MoE 注册成功")
        except Exception as e:
            log_warning(f"[ModelLibrary] 分段建模注册失败: {e}")
        
        # 图分解
        try:
            from core.graph_decomposer import GraphDecomposer
            cls._register('regression', 'graph_decomp', 'GraphDecomposer',
                          GraphDecomposer, 'ensemble',
                          default_params={'n_communities': 5, 'random_state': 42},
                          hyperparam_space={
                              'n_communities': {'type': 'int', 'low': 3, 'high': 10},
                          })
            log_info("[ModelLibrary] GraphDecomposer 注册成功")
        except Exception as e:
            log_warning(f"[ModelLibrary] GraphDecomposer 注册失败: {e}")
        
        # 层次模型
        try:
            from core.hierarchical_model import HierarchicalEstimator
            cls._register('regression', 'hierarchical', 'Hierarchical',
                          HierarchicalEstimator, 'ensemble',
                          default_params={'shrinkage': 0.3, 'random_state': 42})
            log_info("[ModelLibrary] HierarchicalEstimator 注册成功")
        except Exception as e:
            log_warning(f"[ModelLibrary] HierarchicalEstimator 注册失败: {e}")
        
        # 约束优化
        try:
            from core.constrained_optimizer import ConstrainedEstimator
            cls._register('regression', 'constrained', 'Constrained',
                          ConstrainedEstimator, 'linear',
                          default_params={'random_state': 42})
            log_info("[ModelLibrary] ConstrainedEstimator 注册成功")
        except Exception as e:
            log_warning(f"[ModelLibrary] ConstrainedEstimator 注册失败: {e}")
        
        # 神经网络
        try:
            from sklearn.neural_network import MLPRegressor
            cls._register('regression', 'mlp', 'MLP',
                          MLPRegressor, 'neural',
                          default_params={'max_iter': 500, 'random_state': 42},
                          hyperparam_space={
                              'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                              'alpha': {'type': 'float', 'low': 1e-5, 'high': 0.1, 'scale': 'log'},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] MLP 注册失败: {e}")
        
        # HistGradientBoosting (sklearn)
        try:
            from sklearn.ensemble import HistGradientBoostingRegressor
            cls._register('regression', 'hist_gb', 'HistGradientBoosting',
                          HistGradientBoostingRegressor, 'ensemble',
                          default_params={'random_state': 42},
                          hyperparam_space={
                              'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'scale': 'log'},
                              'max_depth': {'type': 'int', 'low': 2, 'high': 8},
                              'max_iter': {'type': 'int', 'low': 50, 'high': 300},
                          })
        except Exception as e:
            log_warning(f"[ModelLibrary] HistGradientBoosting 注册失败: {e}")
        
        # SGD
        try:
            from sklearn.linear_model import SGDRegressor
            cls._register('regression', 'sgd', 'SGD',
                          SGDRegressor, 'linear',
                          default_params={'max_iter': 1000, 'random_state': 42},
                          hyperparam_space={
                              'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
                              'alpha': {'type': 'float', 'low': 1e-5, 'high': 0.1, 'scale': 'log'},
                              'penalty': ['l2', 'l1', 'elasticnet'],
                          },
                          supports_partial_fit=True)
        except Exception as e:
            log_warning(f"[ModelLibrary] SGD 注册失败: {e}")
        
        # RANSAC
        try:
            from sklearn.linear_model import RANSACRegressor
            cls._register('regression', 'ransac', 'RANSAC',
                          RANSACRegressor, 'statistical',
                          default_params={'random_state': 42})
        except Exception as e:
            log_warning(f"[ModelLibrary] RANSAC 注册失败: {e}")
        
        # 时序模型: Prophet
        try:
            import logging as _prophet_logging
            from prophet import Prophet
            # 抑制 Prophet 的冗余日志（daily seasonality / chain processing）
            _prophet_logging.getLogger('prophet').setLevel(_prophet_logging.WARNING)
            _prophet_logging.getLogger('cmdstanpy').setLevel(_prophet_logging.WARNING)
            
            class _ProphetWrapper(BaseEstimator):
                """Prophet sklearn-compatible wrapper"""
                def __init__(self, yearly_seasonality: str = 'auto', weekly_seasonality: str = 'auto',
                             daily_seasonality: str = 'auto', changepoint_prior_scale: float = 0.05,
                             seasonality_prior_scale: float = 10.0, interval_width: float = 0.80,
                             random_state: Optional[int] = None) -> None:
                    self.yearly_seasonality = yearly_seasonality
                    self.weekly_seasonality = weekly_seasonality
                    self.daily_seasonality = daily_seasonality
                    self.changepoint_prior_scale = changepoint_prior_scale
                    self.seasonality_prior_scale = seasonality_prior_scale
                    self.interval_width = interval_width
                    self.random_state = random_state
                    self.model_ = None
                    self.date_col_ = None
                
                def get_params(self, deep=True):
                    return {
                        'yearly_seasonality': self.yearly_seasonality,
                        'weekly_seasonality': self.weekly_seasonality,
                        'daily_seasonality': self.daily_seasonality,
                        'changepoint_prior_scale': self.changepoint_prior_scale,
                        'seasonality_prior_scale': self.seasonality_prior_scale,
                        'interval_width': self.interval_width,
                        'random_state': self.random_state,
                    }
                
                def set_params(self, **params):
                    for key, value in params.items():
                        setattr(self, key, value)
                    return self
                
                def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> '_ProphetWrapper':
                    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
                    # 统一列名为字符串，避免 Prophet add_regressor 对 int 列名报错
                    X.columns = X.columns.astype(str)
                    y = pd.Series(y).values if not isinstance(y, pd.Series) else y.values
                    
                    # 自动检测日期列
                    date_cols = [c for c in X.columns if any(k in str(c).lower() for k in ['date', 'time', 'ds', 'timestamp', '年月'])]
                    if date_cols:
                        self.date_col_ = date_cols[0]
                        ds = pd.to_datetime(X[self.date_col_], errors='coerce')
                    else:
                        ds = pd.date_range(start='2020-01-01', periods=len(X), freq='D')
                    
                    df = pd.DataFrame({'ds': ds, 'y': y})
                    # 复用数值列判断（_fit / _add_regressor / predict 三处都用）
                    # 原代码每次都遍历 extra_cols 并重复调 pd.api.types.is_numeric_dtype，
                    # 改为一次判断后缓存结果。
                    extra_numeric_cols = [c for c in extra_cols if pd.api.types.is_numeric_dtype(X[c])]
                    for c in extra_numeric_cols:
                        df[c] = X[c].values

                    self.model_ = Prophet(
                        yearly_seasonality=self.yearly_seasonality,
                        weekly_seasonality=self.weekly_seasonality,
                        daily_seasonality=self.daily_seasonality,
                        changepoint_prior_scale=self.changepoint_prior_scale,
                        seasonality_prior_scale=self.seasonality_prior_scale,
                        interval_width=self.interval_width,
                    )
                    for c in extra_numeric_cols:
                        self.model_.add_regressor(c)
                    
                    self.model_.fit(df)
                    return self
                
                def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
                    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
                    X.columns = X.columns.astype(str)
                    if self.date_col_ and self.date_col_ in X.columns:
                        ds = pd.to_datetime(X[self.date_col_], errors='coerce')
                    else:
                        ds = pd.date_range(start='2020-01-01', periods=len(X), freq='D')
                    
                    df = pd.DataFrame({'ds': ds})
                    extra_cols = [c for c in X.columns if c != self.date_col_]
                    for c in extra_cols:
                        if pd.api.types.is_numeric_dtype(X[c]):
                            df[c] = X[c].values
                    
                    forecast = self.model_.predict(df)
                    return forecast['yhat'].values
            
            cls._register('regression', 'prophet', 'Prophet',
                          _ProphetWrapper, 'time_series',
                          default_params={'yearly_seasonality': 'auto', 'changepoint_prior_scale': 0.05},
                          hyperparam_space={
                              'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                              'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                          })
            log_info("[ModelLibrary] Prophet 已注册")
        except Exception as e:
            log_warning(f"[ModelLibrary] Prophet 注册失败: {e}")
        
        # ========== 聚类模型 ==========
        try:
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering, OPTICS, MiniBatchKMeans, Birch
            from sklearn.mixture import GaussianMixture
            cls._register('clustering', 'kmeans', 'KMeans',
                          KMeans, 'clustering',
                          default_params={'n_clusters': 5, 'random_state': 42, 'n_init': 10})
            cls._register('clustering', 'dbscan', 'DBSCAN',
                          DBSCAN, 'clustering',
                          default_params={'eps': 0.5, 'min_samples': 5})
            cls._register('clustering', 'agg', 'Agglomerative',
                          AgglomerativeClustering, 'clustering',
                          default_params={'n_clusters': 5})
            cls._register('clustering', 'gmm', 'GaussianMixture',
                          GaussianMixture, 'clustering',
                          default_params={'n_components': 5, 'random_state': 42})
            cls._register('clustering', 'spectral', 'Spectral',
                          SpectralClustering, 'clustering',
                          default_params={'n_clusters': 5, 'random_state': 42})
            cls._register('clustering', 'optics', 'OPTICS',
                          OPTICS, 'clustering',
                          default_params={'min_samples': 5})
            cls._register('clustering', 'minibatch_kmeans', 'MiniBatchKMeans',
                          MiniBatchKMeans, 'clustering',
                          default_params={'n_clusters': 5, 'random_state': 42, 'n_init': 3})
            cls._register('clustering', 'birch', 'Birch',
                          Birch, 'clustering',
                          default_params={'n_clusters': 5})
        except Exception as e:
            log_warning(f"[ModelLibrary] Birch 注册失败: {e}")
        
        cls._initialized = True
        
        # 统计
        for task in [TaskType.CLASSIFICATION.value, TaskType.REGRESSION.value, TaskType.CLUSTERING.value]:
            count = len(cls._models[task])
            if count > 0:
                log_info(f"[ModelLibrary] 已注册 {count} 个{task}模型")
    
    @classmethod
    def _register(cls, task_type: str, key: str, name: str, model_class: Any, category: str,
                  default_params: Optional[Dict] = None, hyperparam_space: Optional[Dict[str, List]] = None,
                  supports_gpu: bool = False, supports_partial_fit: bool = False,
                  supports_sample_weight: bool = False, is_probabilistic: bool = False) -> None:
        cls._models[task_type][key] = ModelSpec(
            name=name, key=key, model_class=model_class, category=category,
            supports_gpu=supports_gpu,
            supports_partial_fit=supports_partial_fit,
            supports_sample_weight=supports_sample_weight,
            is_probabilistic=is_probabilistic,
            default_params=default_params or {},
            hyperparam_space=hyperparam_space or {}
        )
    
    @classmethod
    def get_models(cls, task_type: TaskType, 
                   model_keys: Optional[List[str]] = None,
                   categories: Optional[List[str]] = None) -> Dict[str, ModelSpec]:
        """获取可用模型"""
        cls._init()
        task = task_type.value
        models = cls._models.get(task, {}).copy()
        
        if model_keys:
            models = {k: v for k, v in models.items() if k in model_keys}
        if categories:
            models = {k: v for k, v in models.items() if v.category in categories}
        
        return models
    
    @classmethod
    def create_model(cls, model_key: str, *args, use_gpu: bool = False, **override_params) -> BaseEstimator:
        """创建模型实例
        
        使用 *args 兼容 task_type 既作为位置参数又可能出现在 override_params 中的情况
        （例如 torch_mlp 的 default_params 包含 task_type）
        """
        cls._init()
        
        # 从位置参数或关键字参数中提取 task_type
        task_type = args[0] if args else None
        if 'task_type' in override_params:
            task_type = override_params.pop('task_type')
        if task_type is None:
            raise ValueError("task_type 必须提供")
        if isinstance(task_type, str):
            task_type = TaskType(task_type)
        
        spec = cls._models.get(task_type.value, {}).get(model_key)
        if not spec:
            raise ValueError(f"未知模型: {model_key} ({task_type.value})")
        
        params = deepcopy(spec.default_params)
        params.update(override_params)
        
        # GPU 加速支持
        if use_gpu and spec.supports_gpu:
            try:
                from core.accelerators import auto_gpu_model
                return auto_gpu_model(spec.model_class, use_gpu=True, **params)
            except Exception as e:
                log_warning(f"[ModelLibrary] GPU 加速失败 ({model_key}): {e}，回退到 CPU")
        
        return spec.model_class(**params)
    
    @classmethod
    def list_models(cls, task_type: TaskType) -> pd.DataFrame:
        """列出所有可用模型"""
        cls._init()
        models = cls._models.get(task_type.value, {})
        rows = []
        for key, spec in models.items():
            rows.append({
                'key': key,
                'name': spec.name,
                'category': spec.category,
                'gpu': spec.supports_gpu,
                'params': len(spec.hyperparam_space)
            })
        return pd.DataFrame(rows)


# =============================================================================
# K折交叉验证 + 训练评估
# =============================================================================

@dataclass
class CVResult:
    """交叉验证结果"""
    model_key: str
    model_name: str
    fold_scores: Dict[str, List[float]] = field(default_factory=dict)
    mean_scores: Dict[str, float] = field(default_factory=dict)
    std_scores: Dict[str, float] = field(default_factory=dict)
    oof_pred: Optional[np.ndarray] = None
    oof_proba: Optional[np.ndarray] = None
    fitted_models: List[Any] = field(default_factory=list)
    feature_importance: Optional[pd.DataFrame] = None
    train_time: float = 0.0


def _run_single_fold(fold_idx, train_idx, val_idx, X, y, model, task_type, metrics, n_classes,
                     enable_kernel_approximation: bool = True,
                     enable_precomputed_kernel_cache: bool = True):
    """单个fold的训练与评估（用于并行CV）"""
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # 分类任务：fold内类别重编码
    y_tr_orig = y_tr
    y_val_orig = y_val
    label_map = None
    if task_type == TaskType.CLASSIFICATION:
        y_tr_arr = np.array(y_tr)
        unique_labels = np.unique(y_tr_arr)
        if len(unique_labels) > 0 and not np.array_equal(unique_labels, np.arange(len(unique_labels))):
            # 向量化重编码：np.searchsorted O(n log k) 替代 Python list comp O(n) 哈希查找
            # unique_labels 已排序且唯一，索引位置就是新编码
            label_map = {old: new for new, old in enumerate(unique_labels)}
            y_tr = pd.Series(np.searchsorted(unique_labels, y_tr_arr), index=y_tr.index)

    model_fold = clone(model)
    approx_mode = False
    precomputed_mode = False
    approx_transformer = None
    orig_kernel = getattr(model_fold, 'kernel', None)
    kernel_params = {}

    # 如果是基于核的SVM/SVR，尝试使用预计算核并行/缓存计算或近似特征映射
    if orig_kernel is not None and orig_kernel != 'precomputed' and orig_kernel in ('rbf', 'linear', 'poly'):
        params = model_fold.get_params()
        for k in ('gamma', 'degree', 'coef0'):
            if k in params:
                kernel_params[k] = params[k]

        if enable_kernel_approximation and X_tr.shape[0] >= KERNEL_APPROX_THRESHOLD and orig_kernel in ('rbf', 'poly'):
            approx_mode = True
            n_components = min(KERNEL_APPROX_COMPONENTS, max(128, X_tr.shape[0] // 8))
            X_tr_input, X_val_input, approx_transformer = _create_kernel_approximation(
                X_tr, X_val, orig_kernel, kernel_params, n_components=n_components
            )
            try:
                model_fold.set_params(kernel='linear')
            except Exception:
                try:
                    model_fold.kernel = 'linear'
                except Exception:
                    pass
            model_fold.fit(X_tr_input, y_tr)
        elif enable_precomputed_kernel_cache:
            precomputed_mode = True
            a_key = (tuple(X_tr.columns) if hasattr(X_tr, 'columns') else None, tuple(train_idx))
            cache_key = (orig_kernel, tuple(sorted(kernel_params.items())), a_key)

            K_tr = KERNEL_CACHE.get(cache_key)
            if K_tr is None:
                K_tr = pairwise_kernels(X_tr.values, X_tr.values, metric=orig_kernel, n_jobs=min(os.cpu_count() or 1, 8), **kernel_params)
                try:
                    KERNEL_CACHE.set(cache_key, K_tr)
                except Exception:
                    pass

            try:
                model_fold.set_params(kernel='precomputed')
            except Exception:
                try:
                    model_fold.kernel = 'precomputed'
                except Exception:
                    pass

            model_fold.fit(K_tr, y_tr)
        else:
            model_fold.fit(X_tr, y_tr)
    else:
        model_fold.fit(X_tr, y_tr)

    if approx_mode:
        pred = model_fold.predict(X_val_input)
    elif precomputed_mode:
        # 关键：K_val 之前在 pred 和 proba 两个分支各算一次，pairwise_kernels 是 O(n*m) 矩阵乘
        # 改在前面算一次 K_val 缓存到局部变量，两个分支复用
        K_val = pairwise_kernels(X_val.values, X_tr.values, metric=orig_kernel, n_jobs=min(os.cpu_count() or 1, 8), **kernel_params)
        pred = model_fold.predict(K_val)
    else:
        pred = model_fold.predict(X_val)

    if task_type == TaskType.CLASSIFICATION and label_map is not None:
        # 向量化反编码：numpy 数组 O(1) 索引查找替代 Python list comp O(n) 哈希查找
        # label_map keys 是原始标签，values 是 0..n-1
        pred_int = pred.astype(np.int64)
        max_new = max(label_map.values())
        max_pred = int(pred_int.max()) if len(pred_int) > 0 else 0
        # 扩展到能容纳 pred 最大值的 size；超出 label_map 范围的位置用 self-mapping
        # （保持与 inv_map.get(p, p) 一致的 fallback 行为）
        inv_size = max(max_new, max_pred) + 1
        inv_arr = np.arange(inv_size, dtype=np.int64)  # 默认自映射
        for old, new in label_map.items():
            inv_arr[new] = old
        pred = inv_arr[pred_int]

    proba = None
    proba_aligned = None
    if task_type == TaskType.CLASSIFICATION and hasattr(model_fold, 'predict_proba'):
        if approx_mode:
            proba = model_fold.predict_proba(X_val_input)
        elif precomputed_mode:
            # 复用上面的 K_val（O(n*m) 矩阵乘只算一次）
            proba = model_fold.predict_proba(K_val)
        else:
            proba = model_fold.predict_proba(X_val)
        if proba.ndim > 1:
            fold_classes = getattr(model_fold, 'classes_', np.arange(proba.shape[1]))
            if n_classes == 2 and proba.shape[1] == 2:
                proba_aligned = proba[:, 1]
            elif proba.shape[1] == 1:
                proba_aligned = np.full(len(val_idx), 1.0 if fold_classes[0] == 1 else 0.0)
            elif n_classes > 2:
                if len(fold_classes) == n_classes:
                    proba_aligned = proba
                else:
                    # 向量化：aligned[:, fold_classes] = proba 一次赋值替代 n_classes 次循环
                    # 用 mask 过滤越界 cls（保持原 if 0 <= cls < n_classes 防御）
                    aligned = np.zeros((len(val_idx), n_classes))
                    fold_classes_arr = np.asarray(fold_classes, dtype=np.int64)
                    valid_mask = (fold_classes_arr >= 0) & (fold_classes_arr < n_classes)
                    if valid_mask.any():
                        aligned[:, fold_classes_arr[valid_mask]] = proba[:, valid_mask]
                    proba_aligned = aligned
            else:
                proba_aligned = proba
        else:
            proba_aligned = proba

    # 指标
    fold_scores = {}
    for name, func in metrics.items():
        try:
            if name == 'roc_auc' and proba_aligned is not None:
                score = func(y_val_orig, proba_aligned if proba_aligned.ndim == 1 or proba_aligned.shape[1] == 1 else proba_aligned)
            else:
                score = func(y_val_orig if task_type == TaskType.CLASSIFICATION else y_val, pred)
            fold_scores[name] = float(score)
        except Exception:
            pass

    # 特征重要性
    importances = {}
    if hasattr(model_fold, 'feature_importances_'):
        for i, col in enumerate(X.columns):
            importances[col] = float(model_fold.feature_importances_[i])

    return {
        'fold_idx': fold_idx,
        'val_idx': val_idx,
        'pred': pred,
        'proba': proba_aligned,
        'fold_scores': fold_scores,
        'importances': importances,
        'fitted_model': model_fold,
    }


class CrossValidator:
    """K折交叉验证器"""
    
    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42,
                 verbose: bool = True,
                 use_fold_early_stop: bool = False,
                 fold_early_stop_config: Optional[FoldEarlyStopConfig] = None,
                 fold_type: str = 'default',
                 n_jobs: int = 1,
                 enable_kernel_approximation: bool = True,
                 enable_precomputed_kernel_cache: bool = True) -> None:
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.verbose = verbose
        self.use_fold_early_stop = use_fold_early_stop
        self.fold_early_stop_config = fold_early_stop_config
        self.fold_type = fold_type
        self.n_jobs = n_jobs
        self.enable_kernel_approximation = enable_kernel_approximation
        self.enable_precomputed_kernel_cache = enable_precomputed_kernel_cache
        self._fold_stopper: Optional[FoldEarlyStopper] = None
        if self.use_fold_early_stop:
            direction = 'maximize'  # CV 默认最大化
            self._fold_stopper = FoldEarlyStopper(self.fold_early_stop_config or FoldEarlyStopConfig(), direction)
    
    def cross_validate(self,
                       model: BaseEstimator,
                       X: pd.DataFrame,
                       y: pd.Series,
                       task_type: TaskType,
                       metrics: Optional[Dict[str, Callable]] = None,
                       progress_callback: Optional[callable] = None,
                       model_key: Optional[str] = None,
                       model_name: Optional[str] = None,
                       groups: Optional[np.ndarray] = None) -> CVResult:
        """
        执行K折交叉验证
        
        Returns:
            CVResult 包含OOF预测、每折分数、特征重要性
        """
        start = time.time()
        
        if metrics is None:
            metrics = TaskTypeDetector.get_metrics_dict(task_type)
        
        # 分类任务：一次算清最小类别数（3 个分支都需要，避免重复 O(n) value_counts）
        _y_s = pd.Series(y)
        # 优化：把 value_counts 算一次，后面同时取 min（_min_class_count）
        # 和 len（n_classes），省一次独立的 np.unique 扫描
        _vc = _y_s.value_counts() if len(_y_s) > 0 else pd.Series(dtype=int)
        _min_class_count = int(_vc.min()) if len(_vc) > 0 else 0

        # 创建K折（支持高级CV策略）
        if self.fold_type == 'group' and groups is not None:
            from sklearn.model_selection import GroupKFold
            kfold = GroupKFold(n_splits=self.n_splits)
        elif self.fold_type == 'time':
            from sklearn.model_selection import TimeSeriesSplit
            kfold = TimeSeriesSplit(n_splits=self.n_splits)
        elif self.fold_type == 'repeated':
            # 重复交叉验证：多次KFold取平均，降低方差
            from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
            if task_type == TaskType.CLASSIFICATION and _min_class_count >= 2:
                kfold = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=3, random_state=self.random_state)
            else:
                kfold = RepeatedKFold(n_splits=self.n_splits, n_repeats=3, random_state=self.random_state)
        elif self.fold_type == 'stratified':
            # 显式分层KFold（类别不平衡数据优化）
            effective_splits = min(self.n_splits, max(2, _min_class_count))
            if effective_splits < self.n_splits:
                log_warning(f"[CrossValidator] 分层KFold自动降折数: {self.n_splits} → {effective_splits}")
            if _min_class_count < 2:
                log_warning(f"[CrossValidator] 某类别仅{_min_class_count}样本，回退到KFold")
                kfold = KFold(n_splits=max(2, min(self.n_splits, len(y)//2)), shuffle=self.shuffle,
                             random_state=self.random_state)
            else:
                kfold = StratifiedKFold(n_splits=effective_splits, shuffle=self.shuffle,
                                        random_state=self.random_state)
        elif task_type == TaskType.CLASSIFICATION:
            # 检查最小类别样本数，避免StratifiedKFold因某类样本不足而报错
            effective_splits = min(self.n_splits, max(2, _min_class_count))
            if effective_splits < self.n_splits:
                log_warning(f"[CrossValidator] 最小类别仅{_min_class_count}个样本，自动降低K折数: {self.n_splits} → {effective_splits}")
            # 如果某类别只有1个样本，StratifiedKFold无法使用（要求n_splits<=min_class_count），回退到KFold
            if _min_class_count < 2:
                log_warning(f"[CrossValidator] 某类别仅{_min_class_count}个样本，StratifiedKFold不可用，回退到KFold")
                kfold = KFold(n_splits=max(2, min(self.n_splits, len(y)//2)), shuffle=self.shuffle,
                             random_state=self.random_state)
            else:
                kfold = StratifiedKFold(n_splits=effective_splits, shuffle=self.shuffle, 
                                        random_state=self.random_state)
        else:
            kfold = KFold(n_splits=self.n_splits, shuffle=self.shuffle,
                         random_state=self.random_state)
        
        # 初始化结果容器
        fold_scores = {name: [] for name in metrics.keys()}
        oof_pred = np.zeros(len(y))
        oof_proba = None
        
        if task_type == TaskType.CLASSIFICATION:
            # 复用上面的 _vc：len(_vc) 已是类别数（之前是 np.unique(y) 又扫一次 O(n)）
            n_classes = len(_vc)
            if n_classes == 2:
                oof_proba = np.zeros(len(y))
            else:
                oof_proba = np.zeros((len(y), n_classes))
        
        fitted_models = []
        importances = defaultdict(list)
        
        if self.fold_type == 'group' and groups is not None:
            fold_iter = kfold.split(X, y, groups)
        else:
            fold_iter = kfold.split(X, y)
        if self.verbose:
            fold_iter = list(fold_iter)
        
        # 并行/串行双路径
        use_parallel = self.n_jobs > 1 and _JOBLIB_AVAILABLE
        if use_parallel:
            try:
                fold_results = Parallel(n_jobs=self.n_jobs, backend='threading')(
                    delayed(_run_single_fold)(
                        fold_idx, train_idx, val_idx, X, y, model, task_type, metrics, n_classes,
                        self.enable_kernel_approximation,
                        self.enable_precomputed_kernel_cache
                    )
                    for fold_idx, (train_idx, val_idx) in enumerate(fold_iter)
                )
                # 收集结果
                for res in fold_results:
                    fitted_models.append(res['fitted_model'])
                    oof_pred[res['val_idx']] = res['pred']
                    if res['proba'] is not None:
                        if n_classes == 2:
                            oof_proba[res['val_idx']] = res['proba']
                        else:
                            oof_proba[res['val_idx']] = res['proba']
                    for name, score in res['fold_scores'].items():
                        fold_scores[name].append(score)
                    for col, imp in res['importances'].items():
                        importances[col].append(imp)
                # 并行完成后统一回调
                if progress_callback:
                    progress_callback('cv_fold', self.n_splits, self.n_splits,
                                      f"{model_name or model_key or 'Model'} all folds done (parallel)")
            except Exception as e:
                log_warning(f"[CrossValidator] 并行CV失败，回退到串行: {e}")
                use_parallel = False
        
        if not use_parallel:
            for fold, (train_idx, val_idx) in enumerate(progress_iter(fold_iter, desc="CV", total=self.n_splits, disable=not self.verbose)):
                X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # 分类任务：某些模型（如XGBoost）要求fold内的y_tr类别连续（0~n-1）
                # 如果KFold导致某fold缺失某些类别，需要临时重编码
                y_tr_orig = y_tr
                y_val_orig = y_val
                label_map = None
                if task_type == TaskType.CLASSIFICATION:
                    y_tr_arr = np.array(y_tr)
                    unique_labels = np.unique(y_tr_arr)
                    if len(unique_labels) > 0 and not np.array_equal(unique_labels, np.arange(len(unique_labels))):
                        # 向量化重编码：np.searchsorted O(n log k) 替代 Python list comp O(n) 哈希查找
                        label_map = {old: new for new, old in enumerate(unique_labels)}
                        y_tr = pd.Series(
                            np.searchsorted(unique_labels, y_tr_arr),
                            index=y_tr.index
                        )
                
                res = _run_single_fold(
                    fold,
                    train_idx,
                    val_idx,
                    X,
                    y,
                    model,
                    task_type,
                    metrics,
                    n_classes,
                    self.enable_kernel_approximation,
                    self.enable_precomputed_kernel_cache
                )
                fitted_models.append(res['fitted_model'])
                oof_pred[res['val_idx']] = res['pred']
                if res['proba'] is not None:
                    if n_classes == 2:
                        oof_proba[res['val_idx']] = res['proba']
                    else:
                        oof_proba[res['val_idx']] = res['proba']
                for name, score in res['fold_scores'].items():
                    fold_scores[name].append(score)
                for col, imp in res['importances'].items():
                    importances[col].append(imp)
                if progress_callback:
                    fold_msg = f"{model_name or model_key or 'Model'} fold {fold + 1}/{self.n_splits} done"
                    progress_callback('cv_fold', fold + 1, self.n_splits, fold_msg)
        
        # 汇总
        mean_scores = {k: float(np.mean(v)) if v else 0.0 for k, v in fold_scores.items()}
        std_scores = {k: float(np.std(v)) if v else 0.0 for k, v in fold_scores.items()}
        
        # 特征重要性
        fi_df = None
        if importances:
            fi_data = []
            for col, scores in importances.items():
                fi_data.append({
                    'feature': col,
                    'importance': float(np.mean(scores)),
                    'std': float(np.std(scores))
                })
            fi_df = pd.DataFrame(fi_data).sort_values('importance', ascending=False)
        
        train_time = time.time() - start
        
        return CVResult(
            model_key='',
            model_name='',
            fold_scores=fold_scores,
            mean_scores=mean_scores,
            std_scores=std_scores,
            oof_pred=oof_pred,
            oof_proba=oof_proba,
            fitted_models=fitted_models,
            feature_importance=fi_df,
            train_time=train_time
        )


# =============================================================================
# 模型融合
# =============================================================================

class EnsembleBuilder:
    """
    模型融合构建器
    
    支持：
    - voting_hard: 硬投票（分类）
    - voting_soft: 软投票（分类，需概率）
    - weighted: 加权平均（按CV分数加权）
    - stacking: 堆叠（元学习器）
    """
    
    def __init__(self, method: EnsembleMethod = EnsembleMethod.WEIGHTED,
                 meta_model: Optional[Any] = None) -> None:
        self.method = method
        self.meta_model = meta_model
        self._weights: Optional[np.ndarray] = None
        self._meta_fitted = False
    
    def blend(self,
              cv_results: List[CVResult],
              X_test: Optional[pd.DataFrame] = None,
              task_type: TaskType = TaskType.CLASSIFICATION) -> Dict[str, Any]:
        """
        融合多个模型的OOF预测
        
        Returns:
            {'oof': oof_blend, 'test': test_blend, 'weights': weights}
        """
        if self.method == EnsembleMethod.BEST_SINGLE:
            best = cv_results[0]
            return {
                'oof': best.oof_pred,
                'test': None,
                'weights': {best.model_key: 1.0}
            }
        
        # 收集OOF预测
        # 鲁棒性：过滤掉 oof_pred 为 None 的模型，避免 np.column_stack 抛 TypeError；
        # 同步把 weights 对齐到有效 oof_preds 列表（按 cv_results 顺序）。之前 oof_predictions 拼写错误
        # 导致 _compute_weights 整个 negative-correlation 块被静默跳过，权重按 scores 简单归一化
        valid_cv = [r for r in cv_results if r.oof_pred is not None]
        if not valid_cv:
            raise ValueError("[EnsembleBuilder] 没有任何 CVResult 含 oof_pred，无法融合")
        oof_preds = np.column_stack([r.oof_pred for r in valid_cv])
        # 同步把 weights 对齐到 valid_cv 顺序（与 oof_preds 列顺序一致）
        if len(valid_cv) != len(cv_results):
            log_warning(f"[EnsembleBuilder] {len(cv_results) - len(valid_cv)} 个 CVResult 缺 oof_pred，已过滤")
            valid_keys = {r.model_key: i for i, r in enumerate(cv_results)}
            weights = np.array([weights[valid_keys[r.model_key]] for r in valid_cv], dtype=float)
            # 重新归一化 weights
            wsum = weights.sum()
            if wsum > 0:
                weights = weights / wsum
        
        # Stacking 分支
        if self.method == EnsembleMethod.STACKING:
            # Stacking 元学习器已在 fit_stacking 中训练
            if self._meta_fitted:
                # 关键：fit_stacking 训练时可能用 self._poly 把元特征展开为多项式交互项，
                # OOF 评估必须用同样的 transform 展开后再喂给 meta_model，
                # 否则特征维度对不上、OOF 分数也会是错的（之前是 bug）。
                meta_input = oof_preds
                if getattr(self, '_poly', None) is not None:
                    meta_input = self._poly.transform(meta_input)
                oof_blend = self.meta_model.predict(meta_input)
            else:
                log_warning("[EnsembleBuilder] STACKING 未训练元模型，回退到 WEIGHTED")
                weights = self._compute_weights(cv_results, task_type)
                oof_blend = np.average(oof_preds, axis=1, weights=weights)
            
            test_blend = None
            if X_test is not None and self._meta_fitted:
                test_blend = self.predict_stacking(cv_results, X_test, task_type)
            
            weight_dict = {r.model_key: 1.0 / len(cv_results) for r in cv_results}
            return {
                'oof': oof_blend,
                'test': test_blend,
                'weights': weight_dict,
                'meta_model': self.meta_model.__class__.__name__ if self.meta_model else None
            }
        
        # 计算权重
        if self.method == EnsembleMethod.WEIGHTED:
            weights = self._compute_weights(cv_results, task_type)
        else:
            weights = np.ones(len(cv_results)) / len(cv_results)
        
        self._weights = weights
        
        # OOF融合
        if task_type == TaskType.CLASSIFICATION and self.method in [EnsembleMethod.VOTING_HARD]:
            # 硬投票 - 向量化实现（比 np.apply_along_axis 快 10-100x）
            # 使用 one-hot 编码 + 求和替代逐行循环
            n_samples, n_models = oof_preds.shape
            n_classes = int(oof_preds.max()) + 1
            oof_preds_int = oof_preds.astype(int)
            # 构建 one-hot 矩阵: (n_samples, n_models, n_classes)
            one_hot = np.zeros((n_samples, n_models, n_classes), dtype=np.int32)
            rows = np.arange(n_samples)[:, None]
            cols = np.arange(n_models)[None, :]
            one_hot[rows, cols, oof_preds_int] = 1
            # 统计每类的票数
            votes = one_hot.sum(axis=1)  # (n_samples, n_classes)
            oof_blend = votes.argmax(axis=1)
        else:
            # 加权平均（概率或回归值）
            try:
                oof_blend = np.average(oof_preds, axis=1, weights=weights)
            except ValueError as e:
                log_warning(f"[EnsembleBuilder] 融合失败: {e}，回退到最佳单模型")
                oof_blend = oof_preds[:, 0] if oof_preds.ndim > 1 else oof_preds
        
        # 测试集融合（如果提供了X_test）
        test_blend = None
        if X_test is not None:
            test_blend = self._blend_test(cv_results, X_test, weights, task_type)
        
        weight_dict = {r.model_key: float(w) for r, w in zip(cv_results, weights)}
        
        return {
            'oof': oof_blend,
            'test': test_blend,
            'weights': weight_dict
        }
    
    def _compute_weights(self, cv_results: List[CVResult], task_type: TaskType) -> np.ndarray:
        """基于CV分数计算权重，添加负相关惩罚（diversity bonus）
        
        优化：使用平滑化的RMSE反转，避免除零和无穷大；使用向量化相关性计算。
        """
        if task_type == TaskType.CLASSIFICATION:
            scores = [r.mean_scores.get('f1_weighted', r.mean_scores.get('accuracy', 0.5)) 
                     for r in cv_results]
        else:
            scores = [r.mean_scores.get('r2', 0.0) for r in cv_results]
            # 回归：RMSE越小越好，需要反转，使用平滑化避免除零
            rmse_scores = [r.mean_scores.get('rmse', 1.0) for r in cv_results]
            if any(rmse_scores):
                # 数值稳定性：使用 np.divide 安全除法，避免 rmse=0 时产生无穷大
                rmse_arr = np.array(rmse_scores, dtype=np.float64)
                # 使用平滑化：1/(rmse + epsilon) 替代 1/rmse，避免极端值
                epsilon = 1e-6
                inv_rmse = 1.0 / (rmse_arr + epsilon)
                # 截断上限，避免单个模型权重过大
                inv_rmse = np.clip(inv_rmse, 0, 1e6)
                # 结合 R2 和反转 RMSE，以 R2 为主
                scores = [0.6 * max(s, 0) + 0.4 * inv for s, inv in zip(scores, inv_rmse)]
        
        scores = np.array([max(s, 0.01) for s in scores])
        
        # 负相关惩罚：如果模型间预测高度相关，降低其权重
        # 鼓励选择 diverse 的模型组合
        n_models = len(cv_results)
        if n_models > 1:
            # 收集每个模型的 OOF 预测（用 cv_results 的 oof_pred 字段，非拼写错误的 oof_predictions）
            oof_preds = []
            valid_indices = []  # 记录有效 oof_pred 的 cv_results 索引，与 scores 数组对齐
            for i, r in enumerate(cv_results):
                if r.oof_pred is not None and len(r.oof_pred) > 0:
                    oof_preds.append(np.asarray(r.oof_pred).ravel())
                    valid_indices.append(i)
            
            if len(oof_preds) > 1:
                # 向量化计算模型间预测相关性矩阵
                # 统一截断到相同长度
                min_len = min(len(p) for p in oof_preds)
                if min_len >= 2:
                    pred_matrix = np.vstack([p[:min_len] for p in oof_preds])  # (n_models, min_len)
                    # 计算相关性矩阵（向量化）
                    corr_matrix = np.corrcoef(pred_matrix)  # (n_models, n_models)
                    # 取绝对值并排除对角线
                    np.fill_diagonal(corr_matrix, 0)
                    corr_matrix = np.abs(corr_matrix)
                    # 每个模型与其他模型的平均相关性（只考虑有效的 oof_preds 对应的模型）
                    valid_n = len(oof_preds)
                    avg_corrs = corr_matrix[:valid_n, :valid_n].sum(axis=1) / (valid_n - 1)
                    # 把平均相关性写回到 scores 对应的索引位置（默认无多样性奖励）
                    corr_penalty = np.zeros(n_models)
                    for local_i, global_i in enumerate(valid_indices):
                        corr_penalty[global_i] = 0.3 * avg_corrs[local_i]
                    # 应用惩罚：权重与 (1 - corr_penalty) 成正比
                    scores = scores * (1.0 - corr_penalty)
        
        # 重新归一化
        total = scores.sum()
        if total > 0:
            return scores / total
        return np.ones(n_models) / n_models
    
    def _blend_test(self, cv_results: List[CVResult], X_test: pd.DataFrame,
                    weights: np.ndarray, task_type: TaskType) -> np.ndarray:
        """融合测试集预测"""
        test_preds = []
        
        for result in cv_results:
            # 用最后一个fold的模型预测
            model = result.fitted_models[-1] if result.fitted_models else None
            if model is None:
                continue
            
            pred = model.predict(X_test)
            test_preds.append(np.asarray(pred).ravel())
        
        if not test_preds:
            return None
        
        test_preds = np.column_stack(test_preds)
        
        if task_type == TaskType.CLASSIFICATION and self.method == EnsembleMethod.VOTING_HARD:
            # 硬投票 - 向量化实现（比 np.apply_along_axis 快 10-100x）
            # 使用 one-hot 编码 + 求和替代逐行循环
            n_samples, n_models = test_preds.shape
            n_classes = int(test_preds.max()) + 1
            test_preds_int = test_preds.astype(int)
            # 构建 one-hot 矩阵: (n_samples, n_models, n_classes)
            one_hot = np.zeros((n_samples, n_models, n_classes), dtype=np.int32)
            rows = np.arange(n_samples)[:, None]
            cols = np.arange(n_models)[None, :]
            one_hot[rows, cols, test_preds_int] = 1
            # 统计每类的票数
            votes = one_hot.sum(axis=1)  # (n_samples, n_classes)
            return votes.argmax(axis=1)
        
        try:
            return np.average(test_preds, axis=1, weights=weights)
        except ValueError as e:
            log_warning(f"[EnsembleBuilder] 测试集融合失败: {e}，回退到最佳单模型")
            return test_preds[:, 0] if test_preds.ndim > 1 else test_preds
    
    def fit_stacking(self, cv_results: List[CVResult], y: pd.Series,
                     task_type: TaskType) -> 'EnsembleBuilder':
        """
        训练 stacking 元学习器
        
        优化：使用 RidgeCV 自动选择正则化强度，添加多项式特征捕捉非线性交互。
        """
        if self.meta_model is None:
            if task_type == TaskType.CLASSIFICATION:
                from sklearn.linear_model import LogisticRegressionCV
                self.meta_model = LogisticRegressionCV(
                    Cs=10, cv=3, max_iter=1000, scoring='f1_weighted' if task_type == TaskType.CLASSIFICATION else 'r2',
                    random_state=42
                )
            else:
                from sklearn.linear_model import RidgeCV
                self.meta_model = RidgeCV(alphas=np.logspace(-3, 3, 13), cv=3)
        
        # 构建元特征
        # 鲁棒性：oof_pred 字段为 Optional，理论上可能为 None（用户手构 CVResult），
        # 直接 stack 会抛 TypeError。降级为拿不到 oof 的模型 → 全 0 占位。
        oof_arrays = []
        valid_mask = []
        for r in cv_results:
            arr = r.oof_pred
            if arr is None:
                # 用 0 填充；标记为 invalid，最后在训练时用一行空预测占位
                # 实际更安全的做法是抛错（让用户知道有 CVResult 缺 oof_pred）
                raise ValueError(
                    f"[EnsembleBuilder] CVResult {r.model_key} 缺少 oof_pred，无法 stacking"
                )
            oof_arrays.append(arr)
            valid_mask.append(True)
        meta_features = np.column_stack(oof_arrays)
        
        # 新增：添加多项式特征（捕捉模型间的非线性交互）
        if meta_features.shape[1] >= 2 and meta_features.shape[1] <= 20:
            self._poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
            meta_features_poly = self._poly.fit_transform(meta_features)
            # 限制特征数避免过拟合
            if meta_features_poly.shape[1] <= 100:
                meta_features = meta_features_poly
                log_info(f"[EnsembleBuilder] Stacking 使用多项式特征: {meta_features.shape[1]} 维")
        
        self.meta_model.fit(meta_features, y)
        self._meta_fitted = True
        
        return self
    
    def predict_stacking(self, cv_results: List[CVResult], X_test: pd.DataFrame,
                         task_type: TaskType) -> np.ndarray:
        """Stacking预测"""
        if not self._meta_fitted:
            raise ValueError("请先调用 fit_stacking()")
        
        # 构建测试集元特征
        meta_features = []
        for result in cv_results:
            model = result.fitted_models[-1]
            pred = model.predict(X_test)
            meta_features.append(pred)
        
        meta_features = np.column_stack(meta_features)
        # 关键：fit_stacking 训练时可能用了 self._poly 做特征展开，
        # 预测时必须用同样的 transform 把测试元特征展开到相同的维度，
        # 否则维度对不上、模型看到的特征不一致（之前是 bug）。
        if getattr(self, '_poly', None) is not None:
            meta_features = self._poly.transform(meta_features)
        return self.meta_model.predict(meta_features)


# =============================================================================
# 建模引擎（主入口）
# =============================================================================

@dataclass
class ModelingResult:
    """建模结果"""
    task_type: TaskType
    cv_results: List[CVResult] = field(default_factory=list)
    ensemble_result: Optional[Dict] = None
    best_model_key: Optional[str] = None
    best_cv_result: Optional[CVResult] = None
    feature_importance: Optional[pd.DataFrame] = None
    leaderboard: Optional[pd.DataFrame] = None
    encoding_report: Optional[pd.DataFrame] = None
    feature_selection_report: Optional[pd.DataFrame] = None
    preprocessing_info: Dict = field(default_factory=dict)
    train_time: float = 0.0
    explainability_results: Optional[Dict] = None
    optimized_params: Optional[Dict[str, Dict]] = None
    optimization_history: Optional[Dict[str, List[Dict]]] = None  # 每个模型的超参优化历史
    # 自动评估决策报告
    decision_report: Optional[Any] = None
    auto_recommended_model: Optional[str] = None  # 自动推荐的最优模型
    permutation_importance: Optional[pd.DataFrame] = None
    pseudo_label_report: Optional[Dict] = None
    
    # 降采样报告
    sampling_report: Optional[Any] = None
    
    # 不确定性量化：预测区间/概率校准
    prediction_intervals: Optional[Dict] = None
    conformal_report: Optional[Dict] = None


class ModelingEngine:
    """
    建模引擎主入口
    
    完整流程：
    1. 自动判断任务类型
    2. 自动编码分类变量
    3. 自动特征选择（可选）
    4. K折交叉验证训练
    5. 多模型融合
    6. 生成排行榜和报告
    """
    
    def __init__(self,
                 task_type: Optional[str] = None,
                 model_keys: Optional[List[str]] = None,
                 n_splits: int = 5,
                 encoding: Union[str, EncodingType] = EncodingType.AUTO,
                 feature_selection: Union[str, FeatureSelectionStrategy] = FeatureSelectionStrategy.MI,
                 ensemble: Union[str, EnsembleMethod] = EnsembleMethod.WEIGHTED,
                 random_state: int = 42,
                 n_jobs: int = -1,
                 optimize_hyperparams: bool = False,
                 hyperparam_trials: int = 20,
                 hyperparam_sampler: str = 'tpe',
                 explainability: bool = False,
                 auto_decision_mode: str = 'balanced',
                 user_override_model: Optional[str] = None,
                 auto_sample: bool = True,
                 max_samples: int = 50_000,
                 deep_learning: Optional[Dict] = None,
                 optimizer: str = 'bayesian',
                 dim_reduction: str = 'none',
                 feature_engineering: bool = False,
                 fold_type: str = 'default',
                 group_col: Optional[str] = None,
                 pseudo_labeling: bool = False,
                 pseudo_label_threshold: float = 0.9,
                 verbose: bool = True,
                 use_meta_learning: bool = False,
                 enable_kernel_approximation: bool = True,
                 enable_precomputed_kernel_cache: bool = True,
                 progress_callback: Optional[callable] = None,
                 pipeline_notify: Optional[callable] = None,
                 use_gpu: bool = False) -> None:
        """
        Args:
            task_type: 'classification' / 'regression' / 'clustering' / None=自动判断
            model_keys: 指定模型列表，None=全部
            n_splits: K折数
            encoding: 编码策略
            feature_selection: 特征选择策略，None=不选择
            ensemble: 融合策略
            random_state: 随机种子
            n_jobs: 并行数
            optimize_hyperparams: 是否启用超参优化
            hyperparam_trials: 超参搜索次数
            hyperparam_sampler: 'tpe', 'cmaes', 'random'
            explainability: 是否启用可解释性分析
            auto_decision_mode: 自动决策模式 ('accuracy_first', 'speed_first', 'stability_first', 'simplicity_first', 'balanced')
            user_override_model: 用户覆盖的模型key（None=接受自动推荐）
            auto_sample: 是否启用自动降采样
            max_samples: 自动降采样的最大样本数
            deep_learning: 深度学习配置 {'enabled': bool, 'models': List[str]}
            optimizer: 优化器 'bayesian', 'rl', 'both'
            dim_reduction: 降维 'none', 'pca', 'autoencoder'
            progress_callback: 进度回调函数，签名为 (step: str, current: int, total: int, message: str) -> None
            use_gpu: 是否启用GPU加速（XGBoost/LightGBM/CatBoost）
        """
        self.user_task_type = task_type
        self.model_keys = model_keys
        self.n_splits = n_splits
        self.encoding = encoding if isinstance(encoding, EncodingType) else EncodingType.AUTO
        self.feature_selection = feature_selection if isinstance(feature_selection, FeatureSelectionStrategy) else FeatureSelectionStrategy.MI
        self.ensemble = ensemble if isinstance(ensemble, EnsembleMethod) else EnsembleMethod.WEIGHTED
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.optimize_hyperparams = optimize_hyperparams
        self.hyperparam_trials = hyperparam_trials
        self.hyperparam_sampler = hyperparam_sampler
        self.explainability = explainability
        self.auto_decision_mode = auto_decision_mode
        self.user_override_model = user_override_model
        self.auto_sample = auto_sample
        self.max_samples = max_samples
        self.deep_learning = deep_learning or {'enabled': False, 'models': []}
        self.optimizer = optimizer
        self.dim_reduction = dim_reduction
        self.feature_engineering = feature_engineering
        self.fold_type = fold_type
        self.group_col = group_col
        self.pseudo_labeling = pseudo_labeling
        self.pseudo_label_threshold = pseudo_label_threshold
        self.verbose = verbose
        self.use_meta_learning = use_meta_learning
        self.enable_kernel_approximation = enable_kernel_approximation
        self.enable_precomputed_kernel_cache = enable_precomputed_kernel_cache
        self.progress_callback = progress_callback
        self.pipeline_notify = pipeline_notify
        self._meta_recommender: Optional[Any] = None
        if self.use_meta_learning:
            from core.meta_learning_recommender import MetaLearningModelRecommender
            self._meta_recommender = MetaLearningModelRecommender()
        
        self._encoder: Optional[AutoEncoder] = None
        self._feature_selector: Optional[AutoFeatureSelector] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._autoencoder: Optional[Any] = None
        self._feature_engineer: Optional[Any] = None
        self.result: Optional[ModelingResult] = None

    # 模型成本分类：这些模型在大数据上训练非常慢
    # 用 frozenset 替代普通 set 让 _is_expensive_model 成员查找 O(1) 且不可变
    _EXPENSIVE_MODELS = frozenset({
        'svr', 'svm', 'knn',
        'torch_mlp', 'torch_cnn1d', 'torch_lstm', 'torch_gru', 'torch_nas', 'tabnet',
    })

    def _apply_large_data_model_guards(self, models: Dict[str, Any], task_type: TaskType, n_samples: int) -> Dict[str, Any]:
        """针对大数据自动记录慢模型，保持全部模型但提示性能风险。"""
        if self.model_keys is not None or n_samples <= 30000:
            return models

        # 复用类级别 _EXPENSIVE_MODELS 常量，避免在两个方法里重复定义同一份 set
        slow_models = sorted(k for k in models if k in self._EXPENSIVE_MODELS)
        if slow_models:
            log_info(f"[ModelingEngine] 大数据检测到可能慢模型: {slow_models}，将保留训练但自动降低超参搜索/折数")
        return models

    def _is_expensive_model(self, model_key: str) -> bool:
        return model_key in self._EXPENSIVE_MODELS

    def _get_hyperopt_sample(self, X: pd.DataFrame, y: pd.Series, model_key: str) -> Tuple[pd.DataFrame, pd.Series]:
        """对慢模型的超参优化使用子样本，提高搜索速度。"""
        if not self._is_expensive_model(model_key) or len(X) <= 30000:
            return X, y

        sample_size = min(30000, len(X))
        X_hyp = X.sample(n=sample_size, random_state=self.random_state)
        y_hyp = y.loc[X_hyp.index]
        log_info(f"[ModelingEngine] {model_key} 超参搜索使用子样本: {len(X)} -> {sample_size}")
        return X_hyp, y_hyp

    def _get_effective_cv_folds(self, n_samples: int) -> int:
        """大数据时自动降低 CV 折数，提高训练速度。"""
        if self.n_splits > 3 and n_samples > 50000:
            log_info(f"[ModelingEngine] 大数据自动降低CV折数: {self.n_splits} -> 3")
            return 3
        return self.n_splits

    def _get_effective_hyperparam_trials(self, n_samples: int) -> int:
        """大数据时自动限制超参试验次数，避免过度搜索。"""
        if self.optimize_hyperparams and n_samples > 50000:
            effective = min(self.hyperparam_trials, 20)
            if effective != self.hyperparam_trials:
                log_info(f"[ModelingEngine] 大数据自动限制 hyperparam_trials: {self.hyperparam_trials} -> {effective}")
            return effective
        return self.hyperparam_trials
    
    @staticmethod
    def _sanitize_datetime_columns(X: pd.DataFrame) -> pd.DataFrame:
        """把 datetime64 列拆分为数值特征，防止 numpy dtype 提升错误"""
        X = X.copy()
        for col in list(X.columns):
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                dt = pd.to_datetime(X[col], errors='coerce')
                X[f'{col}_year'] = dt.dt.year.astype('float64')
                X[f'{col}_month'] = dt.dt.month.astype('float64')
                X[f'{col}_day'] = dt.dt.day.astype('float64')
                X[f'{col}_dayofweek'] = dt.dt.dayofweek.astype('float64')
                X[f'{col}_is_weekend'] = (dt.dt.dayofweek >= 5).astype('float64')
                X[f'{col}_quarter'] = dt.dt.quarter.astype('float64')
                X = X.drop(columns=[col])
        return X
    
    def fit(self,
            X: pd.DataFrame,
            y: Optional[pd.Series] = None,
            X_test: Optional[pd.DataFrame] = None) -> ModelingResult:
        """
        执行完整建模流程
        
        Args:
            X: 训练特征
            y: 训练标签（None=聚类）
            X_test: 测试特征（可选）
            
        Returns:
            ModelingResult
        """
        start_time = time.time()
        
        # 0. 统一清洗 datetime64 列（防止下游 sklearn 报错）
        X = self._sanitize_datetime_columns(X)
        if X_test is not None:
            X_test = self._sanitize_datetime_columns(X_test)
        
        # 1. 判断任务类型
        task_type = TaskTypeDetector.detect(y, X, self.user_task_type)
        log_info(f"[ModelingEngine] 任务类型: {task_type.value}")
        
        if task_type == TaskType.UNKNOWN:
            raise ValueError("无法判断任务类型，请显式指定")
        
        # 聚类特殊处理
        if task_type == TaskType.CLUSTERING:
            return self._fit_clustering(X)
        
        # 分类任务：如果 y 是字符串/对象类型，先进行 LabelEncoder 编码
        # 否则 ExtraTrees/GBDT/MLP/XGBoost 等模型会报 "could not convert string to float"
        if task_type == TaskType.CLASSIFICATION and y is not None:
            y_series = pd.Series(y).reset_index(drop=True)
            if not pd.api.types.is_numeric_dtype(y_series):
                self._label_encoder = LabelEncoder()
                y = pd.Series(
                    self._label_encoder.fit_transform(y_series.astype(str)),
                    index=y_series.index,
                    name=y_series.name
                )
                log_info(f"[ModelingEngine] 目标列编码: {len(self._label_encoder.classes_)} 个类别 → 整数标签")
        
        # 2. 自动降采样（在编码前执行，减少编码开销）
        sampling_report = None
        if self.auto_sample:
            try:
                from core.sampling_engine import AutoSampler
                sampler = AutoSampler(
                    max_samples=self.max_samples,
                    task_type=task_type,
                    random_state=self.random_state
                )
                X, y, sampling_report = sampler.sample(X, y)
                if sampling_report and sampling_report.sample_ratio < 1.0:
                    log_info(f"[ModelingEngine] 自动降采样: {sampling_report}")
            except Exception as e:
                log_warning(f"[ModelingEngine] 自动降采样失败: {e}，继续使用全量数据")
        
        # 3. 自动特征工程（比赛级）
        if self.feature_engineering:
            try:
                from core.feature_engineering import AutoFeatureEngineer
                fe = AutoFeatureEngineer(random_state=self.random_state)
                X = fe.fit_transform(X, y)
                if X_test is not None:
                    X_test = fe.transform(X_test)
                self._feature_engineer = fe
                log_info(f"[ModelingEngine] 特征工程: {len(fe._feature_names_in)} -> {X.shape[1]} 特征")
                if self.progress_callback:
                    self.progress_callback('feature_engineering', X.shape[1], len(fe._feature_names_in),
                        f"Feature engineering: {len(fe._feature_names_in)} -> {X.shape[1]} features")
            except Exception as e:
                log_warning(f"[ModelingEngine] 特征工程失败: {e}")
        
        # 4. 自动编码
        X_enc, X_test_enc, encoding_report = self._encode_features(X, X_test, y)
        if self.progress_callback:
            self.progress_callback('encoding', X_enc.shape[1], X.shape[1],
                f"Encoding: {X.shape[1]} -> {X_enc.shape[1]} columns")
        
        # 3. 自动特征选择
        X_sel, X_test_sel, fs_report = self._select_features(X_enc, y, task_type, X_test_enc)
        if self.progress_callback:
            self.progress_callback('feature_selection', X_sel.shape[1], X_enc.shape[1],
                f"Feature selection: {X_enc.shape[1]} -> {X_sel.shape[1]} columns")
        
        # 4. 降维/特征学习（可选）
        if self.dim_reduction == 'autoencoder':
            try:
                from core.deep_learning import TorchAutoEncoder
                ae_epochs = 20 if X_sel.shape[0] > 50000 else 30
                ae_batch_size = 128 if X_sel.shape[0] > 50000 else 64
                ae = TorchAutoEncoder(
                    encoding_dim=min(16, X_sel.shape[1] // 2),
                    epochs=ae_epochs,
                    batch_size=ae_batch_size,
                    verbose=False
                )
                ae.fit(X_sel)
                X_sel = pd.DataFrame(ae.transform(X_sel), index=X_sel.index)
                if X_test_sel is not None:
                    X_test_sel = pd.DataFrame(ae.transform(X_test_sel), index=X_test_sel.index)
                self._autoencoder = ae
                log_info(f"[ModelingEngine] AutoEncoder 降维: {X_sel.shape[1]} 维, epochs={ae_epochs}, batch_size={ae_batch_size}")
                if self.progress_callback:
                    self.progress_callback('dim_reduction', X_sel.shape[1], X_enc.shape[1],
                        f"AutoEncoder dim reduction: {X_enc.shape[1]} -> {X_sel.shape[1]} dims")
            except Exception as e:
                log_warning(f"[ModelingEngine] AutoEncoder 降维失败: {e}")
        
        # 5. 元学习模型推荐（如果启用且用户未指定模型）
        recommended_model_keys = self.model_keys
        if self.use_meta_learning and self.model_keys is None and self._meta_recommender is not None and y is not None:
            try:
                rec = self._meta_recommender.recommend(X_sel, y, task_type, preference=self.auto_decision_mode)
                if rec['model_keys']:
                    recommended_model_keys = rec['model_keys']
                    log_info(f"[MetaLearning] 推荐模型: {recommended_model_keys} (来源: {rec['source']}, {rec['reasoning']})")
            except Exception as e:
                log_warning(f"[MetaLearning] 推荐失败: {e}，回退到默认模型")
        
        # 5. 获取模型列表
        models = ModelLibrary.get_models(task_type, recommended_model_keys)
        if not models:
            raise ValueError(f"没有可用的{task_type.value}模型")

        models = self._apply_large_data_model_guards(models, task_type, X_sel.shape[0])

        # 深度学习模型过滤逻辑（含多模态模型）
        DL_MODEL_KEYS = {'torch_mlp', 'torch_cnn1d', 'torch_lstm', 'torch_gru', 'torch_nas', 'torch_ae', 'torch_resmlp', 'tabnet', 'image_resnet', 'text_bert'}
        if self.model_keys is None:
            # 用户未指定模型：默认排除深度学习模型，除非显式启用
            if not self.deep_learning.get('enabled', False):
                models = {k: v for k, v in models.items() if k not in DL_MODEL_KEYS}
                log_info("[ModelingEngine] 深度学习模型已禁用（默认），可在配置中启用")
            else:
                dl_models = set(self.deep_learning.get('models', []))
                if dl_models:
                    # 只启用指定的深度学习模型
                    models = {k: v for k, v in models.items() if k not in DL_MODEL_KEYS or k in dl_models}
                    log_info(f"[ModelingEngine] 启用深度学习模型: {sorted(dl_models & set(models.keys()))}")
        # 若 model_keys 已显式指定，则尊重用户选择（不过滤）
        
        if not models:
            raise ValueError(f"没有可用的{task_type.value}模型（检查深度学习配置）")
        
        log_info(f"[ModelingEngine] 将训练 {len(models)} 个模型")
        
        # 6. 超参数优化（可选）
        optimized_params: Dict[str, Dict] = {}
        optimization_history: Dict[str, List[Dict]] = {}
        if self.optimize_hyperparams:
            try:
                from core.optimizer_factory import OptimizerFactory
                effective_trials = self._get_effective_hyperparam_trials(X_sel.shape[0])
                effective_cv_folds = min(3, self._get_effective_cv_folds(X_sel.shape[0]))
                optimizer = OptimizerFactory.create(
                    self.optimizer,
                    n_trials=effective_trials,
                    cv_folds=effective_cv_folds,
                    random_state=self.random_state,
                    sampler=self.hyperparam_sampler,
                    n_jobs=self.n_jobs
                )
                log_info(f"[ModelingEngine] 启动 {self.optimizer} 超参数优化，每个模型 {self.hyperparam_trials} 次尝试")
                # 深度学习模型专用 trial 限制
                DL_KEYS = {'torch_mlp', 'torch_cnn1d', 'torch_lstm', 'torch_gru', 'torch_nas', 'torch_ae', 'torch_resmlp', 'tabnet'}
                
                for idx, key in enumerate(progress_iter(models.keys(), desc="超参优化", disable=not self.verbose)):
                    if self.progress_callback:
                        self.progress_callback('hyperopt', idx + 1, len(models), f"优化 {key} 中...")
                    
                    # 跳过时序模型：K-fold CV 对 Prophet 等时序模型无意义
                    spec = models[key]
                    if spec.category == 'time_series':
                        log_info(f"[ModelingEngine] {spec.name} 为时序模型，跳过超参优化")
                        continue
                    
                    try:
                        is_dl = key in DL_KEYS
                        is_expensive = self._is_expensive_model(key)
                        X_opt, y_opt = self._get_hyperopt_sample(X_sel, y, key)

                        if is_dl:
                            dl_trials = min(8, self.hyperparam_trials)
                            from core.optimizer_factory import OptimizerFactory
                            dl_optimizer = OptimizerFactory.create(
                                self.optimizer,
                                n_trials=dl_trials,
                                cv_folds=min(2, self._get_effective_cv_folds(X_sel.shape[0])),
                                random_state=self.random_state,
                                sampler=self.hyperparam_sampler,
                                n_jobs=1,  # DL 模型避免并行
                                trial_timeout=60  # DL 模型更短超时
                            )
                            opt_result = dl_optimizer.optimize(key, X_opt, y_opt, task_type)
                            log_info(f"[ModelingEngine] {key} DL hyperopt: {dl_trials} trials, {min(2, self._get_effective_cv_folds(X_sel.shape[0]))}-fold")
                        elif is_expensive:
                            exp_trials = min(15, self.hyperparam_trials)
                            from core.optimizer_factory import OptimizerFactory
                            exp_optimizer = OptimizerFactory.create(
                                self.optimizer,
                                n_trials=exp_trials,
                                cv_folds=min(2, self._get_effective_cv_folds(X_sel.shape[0])),
                                random_state=self.random_state,
                                sampler=self.hyperparam_sampler,
                                n_jobs=max(1, self.n_jobs // 2),
                                trial_timeout=60
                            )
                            opt_result = exp_optimizer.optimize(key, X_opt, y_opt, task_type)
                            log_info(f"[ModelingEngine] {key} expensive-model hyperopt: {exp_trials} trials, {min(2, self._get_effective_cv_folds(X_sel.shape[0]))}-fold")
                        else:
                            opt_result = optimizer.optimize(key, X_opt, y_opt, task_type)
                        
                        optimized_params[key] = opt_result.best_params
                        optimization_history[key] = opt_result.optimization_history
                        best_score = opt_result.best_score if hasattr(opt_result, 'best_score') else 0
                        log_info(f"[ModelingEngine] {key} 最优参数: {opt_result.best_params}")
                        if self.progress_callback:
                            trial_count = len(opt_result.optimization_history) if opt_result.optimization_history else 0
                            self.progress_callback('hyperopt_model', idx + 1, len(models),
                                f"Hyperopt {key}: {trial_count} trials, best_score={best_score:.4f}")
                    except Exception as e:
                        log_warning(f"[ModelingEngine] {key} 优化失败: {e}")
            except Exception as e:
                log_warning(f"[ModelingEngine] 优化器初始化失败: {e}，跳过超参优化")
        
        # 6. K折交叉验证训练
        effective_cv_folds = self._get_effective_cv_folds(X_sel.shape[0])
        cv = CrossValidator(
            n_splits=effective_cv_folds,
            random_state=self.random_state,
            verbose=self.verbose,
            fold_type=self.fold_type,
            n_jobs=self.n_jobs,
            enable_kernel_approximation=self.enable_kernel_approximation,
            enable_precomputed_kernel_cache=self.enable_precomputed_kernel_cache
        )
        cv_results = []
        
        # 提取分组列（GroupKFold 用）
        groups = None
        if self.fold_type == 'group' and self.group_col and self.group_col in X_sel.columns:
            groups = X_sel[self.group_col].values
            X_sel = X_sel.drop(columns=[self.group_col])
            if X_test_sel is not None and self.group_col in X_test_sel.columns:
                X_test_sel = X_test_sel.drop(columns=[self.group_col])
        
        # 多模态模型需要的特殊列
        MULTIModal_COLS = {'image_resnet': 'image_path', 'text_bert': 'text'}
        
        for idx, (key, spec) in enumerate(progress_iter(models.items(), desc="模型训练", total=len(models), disable=not self.verbose)):
            if self.progress_callback:
                self.progress_callback('training', idx + 1, len(models), f"训练 {spec.name} 中...")
            
            # 多模态模型：检查数据中是否有所需列
            required_col = MULTIModal_COLS.get(key)
            if required_col and required_col not in X_sel.columns:
                log_info(f"[ModelingEngine] 跳过 {spec.name}: 数据缺少 '{required_col}' 列")
                continue
            
            log_info(f"[ModelingEngine] 训练: {spec.name}")
            try:
                params = optimized_params.get(key, {})
                if key in DL_MODEL_KEYS and self.deep_learning.get('enabled', False):
                    params.setdefault('use_amp', self.deep_learning.get('use_amp', False))
                model = ModelLibrary.create_model(key, task_type, use_gpu=self.use_gpu, **params)
                result = cv.cross_validate(model, X_sel, y, task_type,
                    progress_callback=self.progress_callback,
                    model_key=key, model_name=spec.name,
                    groups=groups)
                result.model_key = key
                result.model_name = spec.name
                cv_results.append(result)
                primary_metric = TaskTypeDetector.get_primary_metric(task_type)
                score_str = f"{result.mean_scores.get(primary_metric, 0):.4f}"
                log_info(f"[ModelingEngine] {spec.name} CV完成: {result.mean_scores}")
                if self.progress_callback:
                    self.progress_callback('model_done', idx + 1, len(models),
                        f"{spec.name} CV done: {primary_metric}={score_str} ({result.train_time:.1f}s)")
            except Exception as e:
                log_warning(f"[ModelingEngine] {spec.name} 训练失败: {e}")
        
        if not cv_results:
            raise ValueError("所有模型训练失败")
        
        # 按主要指标排序
        primary_metric = TaskTypeDetector.get_primary_metric(task_type)
        if task_type == TaskType.CLASSIFICATION:
            cv_results.sort(key=lambda r: r.mean_scores.get(primary_metric, 0), reverse=True)
        else:
            # 回归：RMSE越小越好，其他越大越好
            if primary_metric == 'rmse':
                cv_results.sort(key=lambda r: r.mean_scores.get(primary_metric, float('inf')))
            else:
                cv_results.sort(key=lambda r: r.mean_scores.get(primary_metric, 0), reverse=True)
        
        # 6. 模型融合
        ensemble_result = None
        if len(cv_results) > 1 and self.ensemble != EnsembleMethod.BEST_SINGLE:
            builder = EnsembleBuilder(method=self.ensemble)
            # Stacking 需要先训练元学习器
            if self.ensemble == EnsembleMethod.STACKING and y is not None:
                builder.fit_stacking(cv_results, y, task_type)
            ensemble_result = builder.blend(cv_results, X_test_sel, task_type)
            log_info(f"[ModelingEngine] 融合完成，权重: {ensemble_result['weights']}")
            if self.progress_callback:
                weight_summary = ', '.join([f"{k}={v:.2f}" for k, v in list(ensemble_result['weights'].items())[:3]])
                self.progress_callback('ensemble', len(cv_results), len(cv_results),
                    f"Ensemble blended: {weight_summary}")
        
        # 7. 汇总特征重要性
        all_fi = []
        for r in cv_results:
            if r.feature_importance is not None:
                fi = r.feature_importance.copy()
                fi['model'] = r.model_name
                all_fi.append(fi)
        
        ensemble_fi = None
        if all_fi:
            combined = pd.concat(all_fi, copy=False)
            ensemble_fi = combined.groupby('feature')['importance'].mean().reset_index()
            ensemble_fi = ensemble_fi.sort_values('importance', ascending=False)
        
        # 8. 构建排行榜
        leaderboard = self._build_leaderboard(cv_results)
        
        # 9. Permutation Importance（比赛级特征重要性）
        pi_df = None
        if len(cv_results) > 0:
            try:
                from core.permutation_importance import compute_permutation_importance
                best_key = cv_results[0].model_key
                params = optimized_params.get(best_key, {})
                best_model = ModelLibrary.create_model(best_key, task_type, use_gpu=self.use_gpu, **params)
                best_model.fit(X_sel, y)
                scoring = 'accuracy' if task_type == TaskType.CLASSIFICATION else 'r2'
                pi_df = compute_permutation_importance(best_model, X_sel, y, scoring=scoring, n_repeats=3)
                log_info(f"[ModelingEngine] Permutation Importance 计算完成，Top3: {pi_df['feature'].iloc[:3].tolist()}")
            except Exception as e:
                log_warning(f"[ModelingEngine] Permutation Importance 计算失败: {e}")
        
        # 10. 伪标签增强（半监督学习）
        pseudo_report = None
        if self.pseudo_labeling and X_test_sel is not None and len(cv_results) > 0 and task_type != TaskType.CLUSTERING:
            try:
                from core.pseudo_labeling import PseudoLabeler
                best_key = cv_results[0].model_key
                params = optimized_params.get(best_key, {})
                best_model = ModelLibrary.create_model(best_key, task_type, use_gpu=self.use_gpu, **params)
                best_model.fit(X_sel, y)
                
                pl = PseudoLabeler(threshold=self.pseudo_label_threshold)
                X_pseudo, y_pseudo, confidences = pl.generate(best_model, X_test_sel, task_type.value)
                
                if len(X_pseudo) > 0:
                    if task_type == TaskType.CLASSIFICATION and self._label_encoder is not None:
                        y_pseudo = self._label_encoder.transform(pd.Series(y_pseudo).astype(str))
                    
                    X_combined = pd.concat([X_sel, X_pseudo], ignore_index=True, copy=False)
                    y_combined = pd.concat([pd.Series(y), pd.Series(y_pseudo)], ignore_index=True, copy=False)
                    best_model.fit(X_combined, y_combined)
                    
                    # 更新最佳模型的最后一个 fold 模型为增强版
                    if cv_results[0].fitted_models:
                        cv_results[0].fitted_models[-1] = best_model
                    
                    pseudo_report = {
                        'n_pseudo': int(len(X_pseudo)),
                        'n_original': int(len(X_sel)),
                        'n_combined': int(len(X_combined)),
                        'mean_confidence': float(np.mean(confidences)) if len(confidences) > 0 else 0.0,
                    }
                    log_info(f"[ModelingEngine] 伪标签增强完成: {pseudo_report['n_pseudo']} 伪标签样本")
            except Exception as e:
                log_warning(f"[ModelingEngine] 伪标签增强失败: {e}")
        
        # 11. 不确定性量化：共形预测区间（回归任务）
        prediction_intervals = None
        conformal_report = None
        if task_type == TaskType.REGRESSION and len(cv_results) > 0 and y is not None:
            try:
                best_result = cv_results[0]
                oof_pred = best_result.oof_pred
                if oof_pred is not None and len(oof_pred) == len(y):
                    # Split Conformal: 用 OOF 预测作为校准集
                    residuals = np.abs(np.array(y).ravel() - oof_pred)
                    n_calib = len(residuals)
                    q_level = np.ceil((n_calib + 1) * 0.9) / n_calib
                    q_hat = float(np.quantile(residuals, min(q_level, 1.0)))
                    
                    # 对测试集生成区间
                    if X_test_sel is not None:
                        best_key = best_result.model_key
                        params = optimized_params.get(best_key, {})
                        best_model = ModelLibrary.create_model(best_key, task_type, use_gpu=self.use_gpu, **params)
                        best_model.fit(X_sel, y)
                        test_pred = best_model.predict(X_test_sel)
                        lower = test_pred - q_hat
                        upper = test_pred + q_hat
                        prediction_intervals = {
                            'point': test_pred.tolist(),
                            'lower': lower.tolist(),
                            'upper': upper.tolist(),
                            'alpha': 0.1,
                            'confidence': '90%',
                        }
                        widths = upper - lower
                        conformal_report = {
                            'mean_width': float(np.mean(widths)),
                            'median_width': float(np.median(widths)),
                            'q_hat': q_hat,
                            'coverage_guarantee': f'保证覆盖率 ≥ 90%',
                        }
                        log_info(f"[Uncertainty] 共形预测区间: mean_width={conformal_report['mean_width']:.2f}, q_hat={q_hat:.4f}")
            except Exception as e:
                log_warning(f"[Uncertainty] 共形预测失败: {e}")
        
        self.result = ModelingResult(
            task_type=task_type,
            cv_results=cv_results,
            ensemble_result=ensemble_result,
            best_model_key=cv_results[0].model_key,
            best_cv_result=cv_results[0],
            feature_importance=ensemble_fi,
            permutation_importance=pi_df,
            pseudo_label_report=pseudo_report,
            leaderboard=leaderboard,
            encoding_report=encoding_report,
            feature_selection_report=fs_report,
            preprocessing_info={
                'original_features': X.shape[1],
                'encoded_features': X_enc.shape[1],
                'selected_features': X_sel.shape[1],
                'encoder': self._encoder,
                'feature_selector': self._feature_selector,
                'label_encoder': self._label_encoder,
                'autoencoder': self._autoencoder,
            },
            train_time=time.time() - start_time,
            optimized_params=optimized_params if self.optimize_hyperparams else None,
            optimization_history=optimization_history if self.optimize_hyperparams else None,
            sampling_report=sampling_report,
            prediction_intervals=prediction_intervals,
            conformal_report=conformal_report,
        )
        
        # 9. 可解释性分析（可选）
        if self.explainability and cv_results:
            try:
                from core.explainability import ExplainabilityEngine
                explainer = ExplainabilityEngine()
                exp_results = {}
                for cv_result in cv_results[:3]:  # 只解释Top3模型
                    if cv_result.fitted_models:
                        model = cv_result.fitted_models[-1]
                        exp = explainer.explain_model(
                            model, X_sel, y, cv_result.model_key, task_type,
                            feature_names=list(X_sel.columns)
                        )
                        exp_results[cv_result.model_key] = {
                            'method': exp.method,
                            'top_features': exp.global_importance.head(10).to_dict('records') 
                                if exp.global_importance is not None else []
                        }
                self.result.explainability_results = exp_results
                log_info(f"[ModelingEngine] 可解释性分析完成: {list(exp_results.keys())}")
            except Exception as e:
                log_warning(f"[ModelingEngine] 可解释性分析失败: {e}")
        
        # 10. 自动评估与决策
        try:
            from core.evaluation_engine import auto_select
            decision_report = auto_select(
                cv_results=cv_results,
                task_type=task_type,
                mode=self.auto_decision_mode,
                user_override=self.user_override_model,
                primary_metric=TaskTypeDetector.get_primary_metric(task_type)
            )
            self.result.decision_report = decision_report
            self.result.auto_recommended_model = decision_report.recommended_model
            
            # 如果有用户覆盖，更新 best_model_key
            if self.user_override_model:
                self.result.best_model_key = self.user_override_model
                for r in cv_results:
                    if r.model_key == self.user_override_model:
                        self.result.best_cv_result = r
                        break
            else:
                # 使用自动推荐的
                self.result.best_model_key = decision_report.recommended_model
                for r in cv_results:
                    if r.model_key == decision_report.recommended_model:
                        self.result.best_cv_result = r
                        break
            
            log_info(f"[ModelingEngine] 自动评估完成: 推荐模型={decision_report.recommended_name}, "
                     f"置信度={decision_report.confidence:.0%}, 模式={self.auto_decision_mode}")
            if self.user_override_model:
                log_info(f"[ModelingEngine] 用户覆盖了自动选择: {self.user_override_model}")
        except Exception as e:
            log_warning(f"[ModelingEngine] 自动评估失败: {e}")
        
        return self.result
    
    def _encode_features(self, X: pd.DataFrame, X_test: Optional[pd.DataFrame],
                         y: Optional[pd.Series]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """自动编码特征"""
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not cat_cols:
            log_info("[ModelingEngine] 无分类变量，跳过编码")
            return X, X_test, None
        
        log_info(f"[ModelingEngine] 发现 {len(cat_cols)} 个分类变量: {cat_cols}")
        
        self._encoder = AutoEncoder()
        X_enc = self._encoder.fit_transform(X, y)
        
        X_test_enc = None
        if X_test is not None:
            X_test_enc = self._encoder.transform(X_test)
        
        report = self._encoder.get_encoding_report()
        log_info(f"[ModelingEngine] 编码完成: {X.shape[1]} → {X_enc.shape[1]} 列")
        
        return X_enc, X_test_enc, report
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, task_type: TaskType,
                         X_test: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """自动特征选择"""
        if self.feature_selection == FeatureSelectionStrategy.NONE:
            return X, X_test, None
        
        self._feature_selector = AutoFeatureSelector(strategy=self.feature_selection)
        X_sel = self._feature_selector.fit_transform(X, y, task_type)
        
        X_test_sel = None
        if X_test is not None:
            X_test_sel = self._feature_selector.transform(X_test)
        
        report = self._feature_selector.get_feature_importance()
        n_selected = len(self._feature_selector.get_selected_features())
        log_info(f"[ModelingEngine] 特征选择: {X.shape[1]} → {n_selected} 列")
        
        return X_sel, X_test_sel, report
    
    def _fit_clustering(self, X: pd.DataFrame) -> ModelingResult:
        """聚类任务"""
        start_time = time.time()
        
        # 聚类前必须将所有特征数值化（字符串列编码 + 数值列标准化）
        X_proc = X.copy()
        cat_cols = X_proc.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            encoder = AutoEncoder()
            X_proc = encoder.fit_transform(X_proc)
            log_info(f"[ModelingEngine] 聚类编码: {len(cat_cols)}个分类变量 → {X_proc.shape[1]}列")
        
        # 标准化（KMeans/Spectral等对尺度敏感）
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X_proc),
            columns=X_proc.columns,
            index=X_proc.index
        )
        
        models = ModelLibrary.get_models(TaskType.CLUSTERING, self.model_keys)
        cv_results = []
        
        for key, spec in models.items():
            try:
                model = ModelLibrary.create_model(key, TaskType.CLUSTERING, use_gpu=self.use_gpu)
                t0 = time.time()
                model.fit(X_scaled)
                train_time = time.time() - t0
                labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_scaled)
                
                scores = {}
                try:
                    scores['silhouette'] = silhouette_score(X_scaled, labels)
                except Exception as e:
                    log_warning(f"[Clustering] {spec.name} silhouette_score 计算失败: {e}")
                try:
                    scores['calinski_harabasz'] = calinski_harabasz_score(X_scaled, labels)
                except Exception as e:
                    log_warning(f"[Clustering] {spec.name} calinski_harabasz_score 计算失败: {e}")
                try:
                    scores['davies_bouldin'] = davies_bouldin_score(X_scaled, labels)
                except Exception as e:
                    log_warning(f"[Clustering] {spec.name} davies_bouldin_score 计算失败: {e}")
                
                result = CVResult(
                    model_key=key,
                    model_name=spec.name,
                    mean_scores=scores,
                    oof_pred=labels,
                    train_time=train_time,
                    fitted_models=[model]
                )
                cv_results.append(result)
            except Exception as e:
                log_warning(f"[ModelingEngine] {spec.name} 聚类失败: {e}")
        
        # 按 silhouette 排序
        cv_results.sort(key=lambda r: r.mean_scores.get('silhouette', float('-inf')), reverse=True)
        
        rows = []
        for r in cv_results:
            row = {
                'rank': 0,
                'model_name': r.model_name,
                'model_key': r.model_key,
                'train_time': round(r.train_time, 1),
            }
            for metric, score in r.mean_scores.items():
                row[f'{metric}_mean'] = round(score, 4)
            rows.append(row)
        leaderboard = pd.DataFrame(rows)
        leaderboard['rank'] = range(1, len(leaderboard) + 1)
        
        # 自动评估与决策
        decision_report = None
        best_model_key = None
        best_cv_result = None
        try:
            from core.evaluation_engine import auto_select
            decision_report = auto_select(
                cv_results=cv_results,
                task_type=TaskType.CLUSTERING,
                mode=self.auto_decision_mode,
                user_override=self.user_override_model,
                primary_metric='silhouette'
            )
            best_model_key = decision_report.recommended_model
            for r in cv_results:
                if r.model_key == best_model_key:
                    best_cv_result = r
                    break
            log_info(f"[ModelingEngine] 聚类自动评估完成: 推荐模型={decision_report.recommended_name}, "
                     f"置信度={decision_report.confidence:.0%}")
        except Exception as e:
            log_warning(f"[ModelingEngine] 聚类自动评估失败: {e}")
            if cv_results:
                best_model_key = cv_results[0].model_key
                best_cv_result = cv_results[0]
        
        return ModelingResult(
            task_type=TaskType.CLUSTERING,
            cv_results=cv_results,
            leaderboard=leaderboard,
            best_model_key=best_model_key,
            best_cv_result=best_cv_result,
            decision_report=decision_report,
            train_time=time.time() - start_time,
            preprocessing_info={
                'original_features': X.shape[1],
                'encoded_features': X_proc.shape[1],
                'encoder': encoder if cat_cols else None,
                'scaler': scaler,
            }
        )
    
    def _build_leaderboard(self, cv_results: List[CVResult]) -> pd.DataFrame:
        """构建排行榜

        优化：每行循环内 r.mean_scores.items() + r.std_scores.get() + r.fold_scores.get()
        各做一次 dict 查找。把 std_scores / fold_scores 也走 .get + 默认值，行为等价。
        """
        rows = []
        for r in cv_results:
            row = {
                'rank': 0,
                'model_name': r.model_name,
                'model_key': r.model_key,
                'train_time': round(r.train_time, 1),
            }
            # 缓存到本地：避免每行重复走 3 个属性查找
            std_scores = r.std_scores
            fold_scores_map = r.fold_scores
            for metric, score in r.mean_scores.items():
                row[f'{metric}_mean'] = round(score, 4)
                row[f'{metric}_std'] = round(std_scores.get(metric, 0), 4)
                # 保留每折原始分数供箱线图使用
                fold_scores = fold_scores_map.get(metric)
                if fold_scores:
                    row[f'{metric}_fold_scores'] = [round(s, 4) for s in fold_scores]
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df['rank'] = range(1, len(df) + 1)
        return df
    
    def predict(self, X_test: pd.DataFrame) -> np.ndarray:
        """使用最优模型预测"""
        if self.result is None:
            raise ValueError("请先调用 fit()")
        
        X_proc = X_test.copy()
        if self._feature_engineer:
            X_proc = self._feature_engineer.transform(X_proc)
        if self._encoder:
            X_proc = self._encoder.transform(X_proc)
        if self._feature_selector:
            X_proc = self._feature_selector.transform(X_proc)
        
        # 优先使用融合结果
        if self.result.ensemble_result and self.result.ensemble_result.get('test') is not None:
            pred = self.result.ensemble_result['test']
        else:
            # 否则用最佳单模型
            best = self.result.best_cv_result
            if best and best.fitted_models:
                model = best.fitted_models[-1]
                pred = model.predict(X_proc)
            else:
                raise ValueError("无可用的预测模型")
        
        # 如果目标列做过 LabelEncoder 编码，预测结果需要逆变换回原始标签
        if self._label_encoder is not None:
            pred = self._label_encoder.inverse_transform(pred.astype(int))
        
        return pred
    
    def print_report(self) -> None:
        """打印建模报告"""
        if self.result is None:
            print("尚未建模，请先调用 fit()")
            return
        
        r = self.result
        print("\n" + "=" * 70)
        print("建模引擎报告".center(60))
        print("=" * 70)
        
        print(f"\n【任务信息】")
        print(f"  任务类型: {r.task_type.value}")
        print(f"  总耗时: {r.train_time:.1f}s")
        
        if r.preprocessing_info:
            print(f"\n【预处理】")
            for k, v in r.preprocessing_info.items():
                print(f"  {k}: {v}")
        
        if r.encoding_report is not None and not r.encoding_report.empty:
            print(f"\n【编码策略】")
            print(r.encoding_report.to_string(index=False))
        
        print(f"\n【模型排行榜】")
        if r.leaderboard is not None and not r.leaderboard.empty:
            print(r.leaderboard.to_string(index=False))
        
        if r.ensemble_result:
            print(f"\n【融合权重】")
            for model, weight in r.ensemble_result['weights'].items():
                print(f"  {model}: {weight:.3f}")
        
        if r.feature_importance is not None and not r.feature_importance.empty:
            print(f"\n【Top 10 重要特征】")
            for row in r.feature_importance.head(10).itertuples(index=False):
                print(f"  {row.feature:30s}: {row.importance:.4f}")
        
        print("\n" + "=" * 70)
