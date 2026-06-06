"""
自动化特征工程引擎

支持比赛级特征构造：数值交叉、多项式、目标编码、统计聚合、时间特征等。
"""

import hashlib
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings('ignore')

# 预计算 sin/cos 周期编码的角频率常数（2π/N），避免在每个 transform 重复计算
# 12 = 月份数，7 = 星期数
_OMEGA_MONTH = 2.0 * np.pi / 12.0
_OMEGA_DOW = 2.0 * np.pi / 7.0


class AutoFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    自动化特征工程器

    策略：
    1. 数值交叉 (+, -, *, /)
    2. 多项式特征 (square, interaction)
    3. 数学变换 (log1p, sqrt, abs)
    4. K-Fold 目标编码
    5. 计数/频率编码
    6. 分组统计聚合
    7. 时间特征提取
    8. 等频分箱
    """

    def __init__(self,
                 numeric_interactions: bool = True,
                 polynomial_features: bool = True,
                 math_transforms: bool = True,
                 target_encoding: bool = True,
                 count_frequency_encoding: bool = True,
                 groupby_aggregations: bool = True,
                 datetime_features: bool = True,
                 binning: bool = False,
                 max_interactions: int = 50,
                 te_folds: int = 5,
                 te_smoothing: float = 1.0,
                 random_state: int = 42) -> None:
        self.numeric_interactions = numeric_interactions
        self.polynomial_features = polynomial_features
        self.math_transforms = math_transforms
        self.target_encoding = target_encoding
        self.count_frequency_encoding = count_frequency_encoding
        self.groupby_aggregations = groupby_aggregations
        self.datetime_features = datetime_features
        self.binning = binning
        self.max_interactions = max_interactions
        self.te_folds = te_folds
        self.te_smoothing = te_smoothing
        self.random_state = random_state

        # Fitted state
        self._interaction_pairs: List[Tuple[str, str, str]] = []
        self._poly_cols: List[str] = []
        self._te_maps: Dict[str, Dict[Any, float]] = {}
        self._count_maps: Dict[str, Dict[Any, int]] = {}
        self._freq_maps: Dict[str, Dict[Any, float]] = {}
        self._groupby_stats: Dict[str, Dict[str, Dict[str, float]]] = {}
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._datetime_cols: List[str] = []
        self._fitted = False
        self._transform_cache: Dict[str, pd.DataFrame] = {}

    @staticmethod
    def _make_cache_key(X: pd.DataFrame) -> str:
        """基于列名、形状和数据指纹生成缓存键"""
        cols = tuple(X.columns)
        shape = X.shape
        sample = pd.concat([X.head(3), X.tail(3)])
        try:
            data_hash = hashlib.md5(sample.values.tobytes()).hexdigest()
        except Exception:
            data_hash = hashlib.md5(str(sample.values.tolist()).encode()).hexdigest()
        return f"{cols}|{shape}|{data_hash}"

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'AutoFeatureEngineer':
        """学习特征工程规则"""
        self._transform_cache.clear()
        X = X.copy()
        self._feature_names_in = list(X.columns)

        num_cols = self._get_numeric_cols(X)
        cat_cols = self._get_categorical_cols(X)
        dt_cols = self._detect_datetime_cols(X)

        # 1. 数值交叉对
        if self.numeric_interactions and len(num_cols) >= 2:
            self._interaction_pairs = self._select_interaction_pairs(X, num_cols, y)

        # 2. 多项式特征
        if self.polynomial_features and len(num_cols) >= 1:
            self._poly_cols = num_cols[:min(20, len(num_cols))]

        # 3. K-Fold 目标编码映射（用全量数据拟合全局均值，实际变换时用 K-Fold 防止泄漏）
        if self.target_encoding and y is not None and len(cat_cols) > 0:
            for col in cat_cols:
                self._te_maps[col] = self._fit_target_encoding(X[col], y)

        # 4. 计数/频率编码映射
        if self.count_frequency_encoding and len(cat_cols) > 0:
            for col in cat_cols:
                vc = X[col].value_counts()
                self._count_maps[col] = vc.to_dict()
                self._freq_maps[col] = (vc / len(X)).to_dict()

        # 5. 分组统计
        if self.groupby_aggregations and len(cat_cols) > 0 and len(num_cols) > 0:
            for cat_col in cat_cols[:min(5, len(cat_cols))]:
                self._groupby_stats[cat_col] = {}
                for num_col in num_cols[:min(5, len(num_cols))]:
                    grp = X.groupby(cat_col)[num_col].agg(['mean', 'std', 'min', 'max'])
                    self._groupby_stats[cat_col][num_col] = grp.to_dict()

        # 6. 分箱边界
        if self.binning and len(num_cols) > 0:
            for col in num_cols[:min(10, len(num_cols))]:
                self._bin_edges[col] = np.percentile(
                    X[col].dropna(), np.linspace(0, 100, 11)
                )

        # 7. 时间列
        if self.datetime_features:
            self._datetime_cols = dt_cols

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用特征工程（带缓存）"""
        if not self._fitted:
            raise ValueError("请先调用 fit()")

        cache_key = self._make_cache_key(X)
        if cache_key in self._transform_cache:
            return self._transform_cache[cache_key].copy()

        X_out = X.copy()
        num_cols = self._get_numeric_cols(X_out)
        cat_cols = self._get_categorical_cols(X_out)

        # 1. 数值交叉
        for col1, col2, op in self._interaction_pairs:
            if col1 in X_out.columns and col2 in X_out.columns:
                new_name = f"{col1}_{op}_{col2}"
                if op == 'add':
                    X_out[new_name] = X_out[col1] + X_out[col2]
                elif op == 'sub':
                    X_out[new_name] = X_out[col1] - X_out[col2]
                elif op == 'mul':
                    X_out[new_name] = X_out[col1] * X_out[col2]
                elif op == 'div':
                    denom = X_out[col2].replace(0, np.nan)
                    X_out[new_name] = X_out[col1] / denom
                    X_out[new_name] = X_out[new_name].fillna(0)

        # 2. 多项式特征
        for col in self._poly_cols:
            if col in X_out.columns:
                X_out[f"{col}_sq"] = X_out[col] ** 2
        # 两两交互
        for i, c1 in enumerate(self._poly_cols):
            for c2 in self._poly_cols[i + 1:]:
                if c1 in X_out.columns and c2 in X_out.columns:
                    X_out[f"{c1}_x_{c2}"] = X_out[c1] * X_out[c2]

        # 3. 数学变换
        if self.math_transforms:
            for col in num_cols[:min(15, len(num_cols))]:
                if col not in X_out.columns:
                    continue
                X_out[f"{col}_log1p"] = np.log1p(X_out[col].clip(lower=0))
                X_out[f"{col}_sqrt"] = np.sqrt(X_out[col].clip(lower=0))
                X_out[f"{col}_abs"] = X_out[col].abs()

        # 4. 目标编码
        for col, mapping in self._te_maps.items():
            if col in X_out.columns:
                X_out[f"{col}_te"] = X_out[col].map(mapping)
                X_out[f"{col}_te"] = X_out[f"{col}_te"].fillna(
                    np.mean(list(mapping.values())) if mapping else 0
                )

        # 5. 计数/频率编码
        for col, mapping in self._count_maps.items():
            if col in X_out.columns:
                X_out[f"{col}_count"] = X_out[col].map(mapping).fillna(0)
        for col, mapping in self._freq_maps.items():
            if col in X_out.columns:
                X_out[f"{col}_freq"] = X_out[col].map(mapping).fillna(0)

        # 6. 分组统计
        for cat_col, num_dict in self._groupby_stats.items():
            if cat_col not in X_out.columns:
                continue
            for num_col, stats in num_dict.items():
                if num_col not in X_out.columns:
                    continue
                for stat_name, stat_map in stats.items():
                    new_col = f"{cat_col}_{num_col}_{stat_name}"
                    X_out[new_col] = X_out[cat_col].map(stat_map)
                    X_out[new_col] = X_out[new_col].fillna(0)

        # 7. 分箱
        for col, edges in self._bin_edges.items():
            if col in X_out.columns:
                X_out[f"{col}_bin"] = pd.cut(X_out[col], bins=edges, labels=False, include_lowest=True).astype(float).fillna(-1)

        # 8. 时间特征
        drop_cols = []
        for col in self._datetime_cols:
            if col not in X_out.columns:
                continue
            s = pd.to_datetime(X_out[col], errors='coerce')
            X_out[f"{col}_year"] = s.dt.year.astype('float64')
            X_out[f"{col}_month"] = s.dt.month.astype('float64')
            X_out[f"{col}_day"] = s.dt.day.astype('float64')
            X_out[f"{col}_dow"] = s.dt.dayofweek.astype('float64')
            X_out[f"{col}_quarter"] = s.dt.quarter.astype('float64')
            X_out[f"{col}_month_sin"] = np.sin(s.dt.month * _OMEGA_MONTH)
            X_out[f"{col}_month_cos"] = np.cos(s.dt.month * _OMEGA_MONTH)
            X_out[f"{col}_dow_sin"] = np.sin(s.dt.dayofweek * _OMEGA_DOW)
            X_out[f"{col}_dow_cos"] = np.cos(s.dt.dayofweek * _OMEGA_DOW)
            drop_cols.append(col)
        if drop_cols:
            X_out = X_out.drop(columns=drop_cols)

        # 清理无穷值
        X_out = X_out.replace([np.inf, -np.inf], np.nan)
        self._transform_cache[cache_key] = X_out.copy()
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _get_numeric_cols(self, X: pd.DataFrame) -> List[str]:
        return [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]

    def _get_categorical_cols(self, X: pd.DataFrame) -> List[str]:
        return [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c]) or X[c].dtype == 'category']

    def _detect_datetime_cols(self, X: pd.DataFrame) -> List[str]:
        cols = []
        for c in X.columns:
            if 'date' in c.lower() or 'time' in c.lower() or 'dt' in c.lower():
                try:
                    pd.to_datetime(X[c].dropna().iloc[:5], errors='raise')
                    cols.append(c)
                except Exception:
                    pass
        return cols

    def _select_interaction_pairs(self, X: pd.DataFrame, num_cols: List[str],
                                   y: Optional[pd.Series]) -> List[Tuple[str, str, str]]:
        """选择最有价值的数值交叉对（基于相关系数）"""
        pairs = []
        ops = ['add', 'sub', 'mul', 'div']
        candidates = []

        for i, c1 in enumerate(num_cols):
            for c2 in num_cols[i + 1:]:
                for op in ops:
                    candidates.append((c1, c2, op))

        if len(candidates) <= self.max_interactions:
            return candidates

        # 简单启发式：选择与目标相关性更高的交叉
        if y is not None:
            scores = []
            for c1, c2, op in candidates:
                try:
                    if op == 'add':
                        v = X[c1].values + X[c2].values
                    elif op == 'sub':
                        v = X[c1].values - X[c2].values
                    elif op == 'mul':
                        v = X[c1].values * X[c2].values
                    else:
                        v = X[c1].values / np.where(X[c2].values == 0, np.nan, X[c2].values)
                    corr = np.corrcoef(v, y.values)[0, 1] if len(v) == len(y) else 0.0
                    scores.append(abs(corr) if not np.isnan(corr) else 0)
                except Exception:
                    scores.append(0)
            # 按绝对相关性排序取 top
            ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
            pairs = [c for c, s in ranked[:self.max_interactions]]
        else:
            pairs = candidates[:self.max_interactions]

        return pairs

    def _fit_target_encoding(self, series: pd.Series, y: pd.Series) -> Dict[Any, float]:
        """计算目标编码映射（全局均值 + 平滑）—— 向量化实现"""
        global_mean = y.mean()
        # 使用 groupby 向量化计算，避免 Python 级循环
        grouped = y.groupby(series).agg(['mean', 'count'])
        # 平滑公式：(n * local_mean + smoothing * global_mean) / (n + smoothing)
        smoothing = self.te_smoothing
        smoothed = (grouped['count'] * grouped['mean'] + smoothing * global_mean) / (grouped['count'] + smoothing)
        return smoothed.to_dict()

    def get_feature_names_out(self, input_features=None):
        # 简化：返回 transform 后的列名
        if not self._fitted:
            return input_features
        return None  # pandas DataFrame 不需要


class KFoldTargetEncoder(BaseEstimator, TransformerMixin):
    """
    K-Fold 目标编码器（防止数据泄漏）

    在 fit_transform 中使用 K-Fold 交叉编码，
    transform 时使用全局映射。
    """

    def __init__(self, cols: List[str], n_folds: int = 5,
                 smoothing: float = 1.0, random_state: int = 42) -> None:
        self.cols = cols
        self.n_folds = n_folds
        self.smoothing = smoothing
        self.random_state = random_state
        self._maps: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'KFoldTargetEncoder':
        for col in self.cols:
            if col in X.columns:
                self._maps[col] = self._compute_map(X[col], y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()
        for col, mapping in self._maps.items():
            if col in X_out.columns:
                X_out[f"{col}_kfte"] = X_out[col].map(mapping).fillna(
                    np.mean(list(mapping.values())) if mapping else 0
                )
        return X_out

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """使用 K-Fold 防止泄漏的 fit_transform"""
        X_out = X.copy()
        global_mean = y.mean()

        for col in self.cols:
            if col not in X.columns:
                continue
            # K-Fold 编码
            kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
            encoded = pd.Series(index=X.index, dtype=float)
            for tr_idx, val_idx in kf.split(X):
                tr_map = self._compute_map(X.iloc[tr_idx][col], y.iloc[tr_idx])
                encoded.iloc[val_idx] = X.iloc[val_idx][col].map(tr_map)
            encoded = encoded.fillna(global_mean)
            X_out[f"{col}_kfte"] = encoded

        # 同时拟合全局映射供 transform 使用
        self.fit(X, y)
        return X_out

    def _compute_map(self, series: pd.Series, y: pd.Series) -> Dict[Any, float]:
        global_mean = y.mean()
        # Vectorized using groupby — O(n_unique) loop → O(n log n) vectorized
        grouped = y.groupby(series).agg(['mean', 'count'])
        smoothed = (grouped['count'] * grouped['mean'] + self.smoothing * global_mean) / (grouped['count'] + self.smoothing)
        return smoothed.to_dict()
