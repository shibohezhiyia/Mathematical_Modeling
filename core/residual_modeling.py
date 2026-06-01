"""
残差建模模块

核心思想：
1. 用简单模型（线性/Ridge）捕捉主趋势
2. 用复杂模型（LightGBM/XGBoost）拟合残差
3. 最终预测 = 主趋势 + 残差修正

数学形式：
    ŷ = f_base(x) + f_residual(x)
    其中 r = y - f_base^OOF(x)  （OOF = Out-of-Fold，避免数据泄漏）

优势：
- 比单模型精度更高（尤其当数据同时有线性趋势和非线性波动时）
- 比单纯 Stacking 更易解释
- 训练速度快（base 模型通常很轻量）
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold

from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


class ResidualEstimator(BaseEstimator, RegressorMixin):
    """
    残差估计器（sklearn 兼容）

    用法：
        base = Ridge(alpha=1.0)
        residual = LGBMRegressor(n_estimators=100)
        model = ResidualEstimator(base_estimator=base, residual_estimator=residual, cv=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    """

    def __init__(self,
                 base_estimator: Any = None,
                 residual_estimator: Any = None,
                 cv: int = 5,
                 random_state: int = 42) -> None:
        """
        Args:
            base_estimator: 主模型，负责捕捉主趋势（建议用线性/Ridge/Lasso）
            residual_estimator: 残差模型，负责拟合 base 没捕捉到的部分（建议用树模型）
            cv: OOF 残差计算的折数，cv=1 表示不用 OOF（直接在训练集上算残差，更快但可能过拟合）
            random_state: 随机种子
        """
        self.base_estimator = base_estimator
        self.residual_estimator = residual_estimator
        self.cv = cv
        self.random_state = random_state

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'ResidualEstimator':
        """训练残差模型"""
        X = self._to_df(X)
        y = np.array(y).ravel()

        base = self._get_base_estimator()
        residual = self._get_residual_estimator()

        if self.cv > 1:
            # OOF 方式：用 CV 生成 base 的 out-of-fold 预测，避免数据泄漏
            oof_pred = self._compute_oof_prediction(base, X, y)
            residual_target = y - oof_pred
            log_info(f"[ResidualEstimator] OOF 残差统计: mean={residual_target.mean():.4f}, std={residual_target.std():.4f}")
        else:
            # 快速方式：直接在训练集上算残差（可能过拟合，但速度快）
            base.fit(X, y)
            residual_target = y - base.predict(X)
            log_info(f"[ResidualEstimator] 快速残差统计: mean={residual_target.mean():.4f}, std={residual_target.std():.4f}")

        # 训练残差模型
        residual.fit(X, residual_target)
        self.residual_estimator_ = residual

        # 用全量数据重新训练 base（用于最终预测）
        base.fit(X, y)
        self.base_estimator_ = base

        # 记录残差模型的特征重要性（如果有）
        if hasattr(self.residual_estimator_, 'feature_importances_'):
            self.residual_feature_importances_ = self.residual_estimator_.feature_importances_
        else:
            self.residual_feature_importances_ = None

        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """预测 = base 趋势 + 残差修正"""
        if not hasattr(self, 'base_estimator_') or not hasattr(self, 'residual_estimator_'):
            raise ValueError("请先调用 fit()")
        X = self._to_df(X)
        base_pred = self.base_estimator_.predict(X)
        residual_pred = self.residual_estimator_.predict(X)
        return base_pred + residual_pred

    def predict_components(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        """分别返回 base 和 residual 的预测分量（用于分析）"""
        X = self._to_df(X)
        return {
            'base': self.base_estimator_.predict(X),
            'residual': self.residual_estimator_.predict(X),
            'total': self.predict(X)
        }

    def _compute_oof_prediction(self, base, X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
        """计算 base_estimator 的 Out-of-Fold 预测"""
        oof_pred = np.zeros(len(y))
        kfold = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr = y[train_idx]
            model = clone(base)
            model.fit(X_tr, y_tr)
            oof_pred[val_idx] = model.predict(X_val)
        return oof_pred

    def _get_base_estimator(self) -> Any:
        if self.base_estimator is not None:
            return clone(self.base_estimator)
        from sklearn.linear_model import Ridge
        return Ridge(alpha=1.0, random_state=self.random_state)

    def _get_residual_estimator(self) -> Any:
        if self.residual_estimator is not None:
            return clone(self.residual_estimator)
        try:
            from lightgbm import LGBMRegressor
            return LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        except Exception:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)

    @staticmethod
    def _to_df(X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X)


class AutoResidualStacker:
    """
    自动残差堆叠器

    自动选择最佳 (base, residual) 组合，通过 CV 评估决定。
    """

    def __init__(self,
                 base_candidates: Optional[List[Any]] = None,
                 residual_candidates: Optional[List[Any]] = None,
                 cv: int = 5,
                 random_state: int = 42,
                 metric: str = 'rmse') -> None:
        """
        Args:
            base_candidates: 主模型候选列表，None = [Ridge, Lasso, ElasticNet, LinearRegression]
            residual_candidates: 残差模型候选列表，None = [LGBM, XGB, CatBoost, RF, GBDT]
            cv: 评估折数
            random_state: 随机种子
            metric: 评估指标 'rmse' | 'mae' | 'r2'
        """
        self.base_candidates = base_candidates
        self.residual_candidates = residual_candidates
        self.cv = cv
        self.random_state = random_state
        self.metric = metric
        self.best_estimator_: Optional[ResidualEstimator] = None
        self.best_score_: float = float('inf')
        self.best_combo_: Optional[Tuple[str, str]] = None
        self.results_: List[Dict] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]) -> 'AutoResidualStacker':
        """自动搜索最佳残差组合并训练"""
        X = ResidualEstimator._to_df(X)
        y = np.array(y).ravel()

        bases = self.base_candidates or self._default_bases()
        residuals = self.residual_candidates or self._default_residuals()

        best_score = float('inf')
        best_est = None
        best_combo = None

        for base_name, base in bases.items():
            for res_name, residual in residuals.items():
                try:
                    est = ResidualEstimator(base_estimator=base, residual_estimator=residual,
                                            cv=self.cv, random_state=self.random_state)
                    score = self._cv_score(est, X, y)
                    log_info(f"[AutoResidualStacker] {base_name} + {res_name}: {self.metric}={score:.4f}")
                    self.results_.append({
                        'base': base_name,
                        'residual': res_name,
                        'score': score
                    })
                    if score < best_score:
                        best_score = score
                        best_est = est
                        best_combo = (base_name, res_name)
                except Exception as e:
                    log_warning(f"[AutoResidualStacker] {base_name}+{res_name} 失败: {e}")

        if best_est is None:
            raise ValueError("没有成功训练任何残差组合")

        # 用最佳组合在全量数据上重新训练
        self.best_estimator_ = best_est
        self.best_estimator_.fit(X, y)
        self.best_score_ = best_score
        self.best_combo_ = best_combo
        log_info(f"[AutoResidualStacker] 最佳组合: {best_combo[0]} + {best_combo[1]}, {self.metric}={best_score:.4f}")
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if self.best_estimator_ is None:
            raise ValueError("请先调用 fit()")
        return self.best_estimator_.predict(X)

    def predict_components(self, X: Union[pd.DataFrame, np.ndarray]) -> Dict[str, np.ndarray]:
        if self.best_estimator_ is None:
            raise ValueError("请先调用 fit()")
        return self.best_estimator_.predict_components(X)

    def _cv_score(self, estimator, X: pd.DataFrame, y: np.ndarray) -> float:
        """交叉验证评估"""
        from sklearn.model_selection import cross_val_score
        if self.metric == 'rmse':
            scores = cross_val_score(estimator, X, y, cv=self.cv,
                                     scoring='neg_root_mean_squared_error',
                                     n_jobs=1)
            return -scores.mean()
        elif self.metric == 'mae':
            scores = cross_val_score(estimator, X, y, cv=self.cv,
                                     scoring='neg_mean_absolute_error',
                                     n_jobs=1)
            return -scores.mean()
        elif self.metric == 'r2':
            scores = cross_val_score(estimator, X, y, cv=self.cv,
                                     scoring='r2', n_jobs=1)
            return -scores.mean()  # 统一按越小越好
        else:
            raise ValueError(f"未知指标: {self.metric}")

    def _default_bases(self) -> Dict[str, Any]:
        from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
        return {
            'ridge': Ridge(alpha=1.0, random_state=self.random_state),
            'lasso': Lasso(alpha=0.1, random_state=self.random_state, max_iter=2000),
            'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state, max_iter=2000),
            'linear': LinearRegression(),
        }

    def _default_residuals(self) -> Dict[str, Any]:
        candidates = {}
        try:
            from lightgbm import LGBMRegressor
            candidates['lgb'] = LGBMRegressor(n_estimators=100, random_state=self.random_state, verbose=-1)
        except Exception:
            pass
        try:
            from xgboost import XGBRegressor
            candidates['xgb'] = XGBRegressor(n_estimators=100, random_state=self.random_state, verbosity=0)
        except Exception:
            pass
        try:
            from catboost import CatBoostRegressor
            candidates['catboost'] = CatBoostRegressor(iterations=100, random_seed=self.random_state, verbose=False)
        except Exception:
            pass
        if not candidates:
            from sklearn.ensemble import GradientBoostingRegressor
            candidates['gbdt'] = GradientBoostingRegressor(n_estimators=100, random_state=self.random_state)
        return candidates
