"""
约束优化模块

在模型训练中加入业务约束，避免反直觉预测：
- 单调约束：价格↑ → 销量↓（不应预测为↑）
- 非负约束：库存、销量不能为负
- 有界预测：预测值必须在合理范围内
- 概率约束：概率总和为 1

支持模型：
- LightGBM: monotone_constraints
- XGBoost: monotone_constraints
- CatBoost: monotone_constraints
- 线性模型: 非负最小二乘

用法：
    model = ConstrainedEstimator(
        base_estimator=LGBMRegressor(),
        monotone_features={'price': -1, 'quality': 1},  # price 负相关, quality 正相关
        bounds=(0, None)  # 预测值非负
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)  # 自动满足约束
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import nnls

from utils.helpers import log_info

warnings.filterwarnings('ignore')


class ConstrainedEstimator(BaseEstimator, RegressorMixin):
    """
    约束估计器包装器

    自动为底层模型添加约束参数，并在预测后做后处理裁剪。
    """

    def __init__(self,
                 base_estimator: Any = None,
                 monotone_features: Optional[Dict[str, int]] = None,
                 bounds: Optional[Tuple[Optional[float], Optional[float]]] = None,
                 feature_names: Optional[List[str]] = None,
                 auto_detect_monotone: bool = False,
                 random_state: int = 42) -> None:
        """
        Args:
            base_estimator: 基础模型
            monotone_features: 单调约束，如 {'price': -1, 'quality': 1}
                               1 = 正单调（特征↑预测↑），-1 = 负单调
            bounds: (下界, 上界)，如 (0, None) 表示非负
            feature_names: 特征名列表（用于映射单调约束）
            auto_detect_monotone: 是否自动从数据检测单调性
            random_state: 随机种子
        """
        self.base_estimator = base_estimator
        self.monotone_features = monotone_features or {}
        self.bounds = bounds
        self.feature_names = feature_names
        self.auto_detect_monotone = auto_detect_monotone
        self.random_state = random_state

        self.model_: Optional[Any] = None
        self.detected_monotone_: Optional[Dict[str, int]] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'ConstrainedEstimator':
        X = self._to_df(X)
        y = np.array(y).ravel()

        # 自动检测单调性
        if self.auto_detect_monotone and not self.monotone_features:
            self.detected_monotone_ = self._detect_monotone_from_data(X, y)
            effective_monotone = self.detected_monotone_
        else:
            effective_monotone = self.monotone_features

        # 构建约束参数字符串
        constraints_str = self._build_monotone_constraints(X, effective_monotone)

        # 创建带约束的模型
        model = self._create_constrained_model(constraints_str)
        model.fit(X, y)
        self.model_ = model

        if effective_monotone:
            log_info(f"[ConstrainedEstimator] 单调约束: {effective_monotone}")
        if self.bounds:
            log_info(f"[ConstrainedEstimator] 边界约束: {self.bounds}")

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.model_ is None:
            raise ValueError("请先调用 fit()")

        X = self._to_df(X)
        pred = self.model_.predict(X)

        # 后处理：边界裁剪
        if self.bounds:
            lower, upper = self.bounds
            if lower is not None:
                pred = np.maximum(pred, lower)
            if upper is not None:
                pred = np.minimum(pred, upper)

        return pred

    def _build_monotone_constraints(self, X: pd.DataFrame,
                                    monotone_dict: Dict[str, int]) -> Optional[str]:
        """将特征名约束映射为模型需要的索引约束字符串"""
        if not monotone_dict:
            return None

        feature_names = list(X.columns)
        constraints = ['0'] * len(feature_names)
        # 建 dict 一次 O(n)，后续 O(1)：替代 feature_names.index(feat) O(n) 每次
        feat_to_idx = {name: i for i, name in enumerate(feature_names)}

        for feat, direction in monotone_dict.items():
            if feat in feat_to_idx:
                constraints[feat_to_idx[feat]] = str(int(direction))
            elif feat.isdigit() and int(feat) < len(feature_names):
                constraints[int(feat)] = str(int(direction))

        # LightGBM/XGBoost 格式: "0,1,-1,0,..."
        return '(' + ','.join(constraints) + ')'

    def _create_constrained_model(self, constraints_str: Optional[str]) -> Any:
        """创建带约束的模型副本"""
        if self.base_estimator is None:
            try:
                from lightgbm import LGBMRegressor
                base = LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
            except Exception:
                base = Ridge(alpha=1.0, random_state=self.random_state)
        else:
            base = clone(self.base_estimator)

        # 尝试设置单调约束
        if constraints_str:
            class_name = base.__class__.__name__.lower()
            if 'lgbm' in class_name or 'lightgbm' in class_name:
                try:
                    base.set_params(monotone_constraints=constraints_str)
                except Exception:
                    pass
            elif 'xgb' in class_name or 'xgboost' in class_name:
                try:
                    base.set_params(monotone_constraints=constraints_str)
                except Exception:
                    pass
            elif 'catboost' in class_name:
                try:
                    # CatBoost 格式不同：list of int
                    constraints_list = [int(x) for x in constraints_str.strip('()').split(',')]
                    base.set_params(monotone_constraints=constraints_list)
                except Exception:
                    pass

        return base

    def _detect_monotone_from_data(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, int]:
        """从数据自动检测单调关系"""
        monotone = {}
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                continue
            try:
                # Spearman 秩相关（对单调关系敏感，不受线性假设限制）
                corr = X[col].corr(pd.Series(y), method='spearman')
                if abs(corr) > 0.7:  # 强单调关系
                    direction = 1 if corr > 0 else -1
                    monotone[col] = direction
            except Exception:
                continue
        log_info(f"[ConstrainedEstimator] 自动检测到 {len(monotone)} 个单调特征")
        return monotone

    @staticmethod
    def _to_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class BoundedRegressor(BaseEstimator, RegressorMixin):
    """
    有界回归器

    确保预测值在 [lower, upper] 范围内。
    如果基础模型可能越界，用这个包装器兜底。
    """

    def __init__(self, base_estimator: Any = None,
                 lower: Optional[float] = None,
                 upper: Optional[float] = None) -> None:
        self.base_estimator = base_estimator
        self.lower = lower
        self.upper = upper
        self.model_: Optional[Any] = None

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'BoundedRegressor':
        model = clone(self.base_estimator) if self.base_estimator else Ridge(alpha=1.0)
        model.fit(X, y)
        self.model_ = model
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        pred = self.model_.predict(X)
        if self.lower is not None:
            pred = np.maximum(pred, self.lower)
        if self.upper is not None:
            pred = np.minimum(pred, self.upper)
        return pred


class NonNegativeRegressor(BaseEstimator, RegressorMixin):
    """
    非负回归器

    约束预测值 ≥ 0。适合销量、库存、人数等场景。
    使用 scipy.nnls（非负最小二乘）或后处理裁剪。
    """

    def __init__(self, method: str = 'clip', base_estimator: Any = None) -> None:
        """
        Args:
            method: 'clip'（后处理裁剪） | 'nnls'（非负最小二乘）
            base_estimator: 基础模型（仅 clip 模式使用）
        """
        self.method = method
        self.base_estimator = base_estimator
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: float = 0.0

    def fit(self, X: Union[pd.DataFrame, np.ndarray],
            y: Union[pd.Series, np.ndarray]) -> 'NonNegativeRegressor':
        X = np.array(X)
        y = np.array(y).ravel()

        if self.method == 'nnls':
            # 非负最小二乘：min ||Ax - b||, s.t. x >= 0
            # 添加偏置列
            X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
            coef, _ = nnls(X_with_bias, y)
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            model = clone(self.base_estimator) if self.base_estimator else Ridge(alpha=1.0)
            model.fit(X, y)
            self.model_ = model

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        X = np.array(X)
        if self.method == 'nnls':
            pred = X @ self.coef_ + self.intercept_
        else:
            pred = self.model_.predict(X)
        return np.maximum(pred, 0)


class MonotonicityEnforcer:
    """
    单调性强制器（后处理）

    如果模型本身不支持单调约束，可以用这个在预测后做等渗回归校准，
    确保输出关于某个特征单调。

    用法：
        enforcer = MonotonicityEnforcer(feature_idx=0, direction=-1)
        y_calibrated = enforcer.fit_transform(X_test[:, 0], raw_pred)
    """

    def __init__(self, feature_idx: int, direction: int = 1) -> None:
        """
        Args:
            feature_idx: 要强制单调的特征索引
            direction: 1（递增）或 -1（递减）
        """
        self.feature_idx = feature_idx
        self.direction = direction
        self.iso_: Optional[Any] = None

    def fit(self, feature_values: np.ndarray, predictions: np.ndarray) -> 'MonotonicityEnforcer':
        x = np.array(feature_values).ravel()
        y = np.array(predictions).ravel()

        # 按特征值排序
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]

        self.iso_ = IsotonicRegression(
            increasing=(self.direction > 0),
            out_of_bounds='clip'
        )
        self.iso_.fit(x_sorted, y_sorted)
        return self

    def transform(self, feature_values: np.ndarray) -> np.ndarray:
        return self.iso_.predict(np.array(feature_values).ravel())

    def fit_transform(self, feature_values: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        return self.fit(feature_values, predictions).transform(feature_values)


def auto_monotone_constraints(df: pd.DataFrame, target_col: str,
                              threshold: float = 0.6) -> Dict[str, int]:
    """
    自动推断单调约束

    用法：
        constraints = auto_monotone_constraints(df, 'sales_amount')
        # {'price': -1, 'ad_spend': 1, ...}
    """
    constraints = {}
    y = df[target_col]

    for col in df.columns:
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        try:
            corr = df[col].corr(y, method='spearman')
            if abs(corr) >= threshold:
                constraints[col] = 1 if corr > 0 else -1
        except Exception:
            continue

    log_info(f"[auto_monotone] 检测到 {len(constraints)} 个单调约束: {constraints}")
    return constraints
