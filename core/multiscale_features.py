"""
多尺度时间特征分解模块

核心思想：把时间序列信号拆成多个尺度的分量：
    y = trend + seasonality + cyclical + holiday + residual

适用场景：
- 销量预测（长期趋势 + 月度周期 + 周末效应 + 节假日冲击）
- 流量预测（趋势 + 日周期 + 周周期 + 异常事件）
- 任何带时间戳的回归/分类任务

数学工具：
- 趋势：移动平均 / 线性拟合 / LOWESS
- 周期：傅里叶基 / 季节性分解
- 节假日：指示变量 / 冲击窗口
- 残差：去趋势去周期后的剩余信号
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import signal

from utils.helpers import log_info, log_warning

warnings.filterwarnings('ignore')


class MultiscaleTimeDecomposer(BaseEstimator, TransformerMixin):
    """
    多尺度时间分解器

    从 datetime 列中提取多尺度特征：
    - 趋势（长期方向）
    - 季节性（固定周期，如月/周/日）
    - 循环特征（正弦/余弦编码）
    - 节假日冲击
    - 滞后特征
    - 滑动统计

    用法：
        decomposer = MultiscaleTimeDecomposer(
            datetime_col='order_date',
            target_col='sales_amount',
            seasonal_periods=[7, 30, 365],  # 周、月、年周期
            holiday_calendar='CN'  # 中国节假日
        )
        X_enhanced = decomposer.fit_transform(df)
    """

    def __init__(self,
                 datetime_col: Optional[str] = None,
                 target_col: Optional[str] = None,
                 seasonal_periods: Optional[List[int]] = None,
                 trend_window: int = 30,
                 lag_periods: Optional[List[int]] = None,
                 rolling_windows: Optional[List[int]] = None,
                 holiday_calendar: Optional[str] = None,
                 fourier_terms: int = 3,
                 random_state: int = 42) -> None:
        """
        Args:
            datetime_col: 时间列名，None=自动检测
            target_col: 目标列名（用于趋势/滞后特征）
            seasonal_periods: 季节性周期列表，如 [7, 30, 365]
            trend_window: 趋势移动平均窗口
            lag_periods: 滞后周期列表，如 [1, 7, 14]
            rolling_windows: 滑动统计窗口，如 [7, 14, 30]
            holiday_calendar: 节假日日历 'CN' | 'US' | None
            fourier_terms: 傅里叶基函数数量
            random_state: 随机种子
        """
        self.datetime_col = datetime_col
        self.target_col = target_col
        self.seasonal_periods = seasonal_periods or [7, 30]
        self.trend_window = trend_window
        self.lag_periods = lag_periods or [1, 7]
        self.rolling_windows = rolling_windows or [7, 14]
        self.holiday_calendar = holiday_calendar
        self.fourier_terms = fourier_terms
        self.random_state = random_state

        # Fitted state
        self._dt_col: Optional[str] = None
        self._trend_model: Optional[Any] = None
        self._seasonal_medians: Dict[int, Dict] = {}
        self._feature_names: List[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'MultiscaleTimeDecomposer':
        """学习时间分解参数"""
        X = X.copy()

        # 自动检测时间列
        self._dt_col = self.datetime_col or self._detect_datetime_col(X)
        if self._dt_col is None:
            log_warning("[MultiscaleTimeDecomposer] 未检测到时间列，跳过多尺度分解")
            self._fitted = True
            return self

        # 转换为 datetime
        dt = pd.to_datetime(X[self._dt_col], errors='coerce')

        # 学习季节性中位数（用于季节性分解）
        if self.target_col and self.target_col in X.columns:
            for period in self.seasonal_periods:
                if period == 7:
                    # 周周期：按星期几分组
                    groups = dt.dt.dayofweek
                elif period == 30:
                    # 月周期：按月内日期分组
                    groups = dt.dt.day
                elif period == 12:
                    # 年周期：按月分组
                    groups = dt.dt.month
                elif period == 365:
                    # 年周期：按年内第几天分组
                    groups = dt.dt.dayofyear
                elif period == 24:
                    # 日周期：按小时分组
                    groups = dt.dt.hour
                else:
                    continue

                medians = X.groupby(groups)[self.target_col].median().to_dict()
                self._seasonal_medians[period] = medians

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """应用多尺度特征工程"""
        X = X.copy()

        if not self._fitted:
            raise ValueError("请先调用 fit()")

        if self._dt_col is None or self._dt_col not in X.columns:
            return X

        dt = pd.to_datetime(X[self._dt_col], errors='coerce')
        new_features: Dict[str, pd.Series] = {}

        # 1. 基础时间特征（已在 AutoFeatureEngineer 中部分实现，这里补充更精细的）
        new_features[f'{self._dt_col}_dayofyear'] = dt.dt.dayofyear.astype('float64')
        new_features[f'{self._dt_col}_weekofyear'] = dt.dt.isocalendar().week.astype('float64')
        new_features[f'{self._dt_col}_is_month_start'] = dt.dt.is_month_start.astype('float64')
        new_features[f'{self._dt_col}_is_month_end'] = dt.dt.is_month_end.astype('float64')
        new_features[f'{self._dt_col}_is_quarter_start'] = dt.dt.is_quarter_start.astype('float64')

        # 2. 循环傅里叶特征（比简单正弦余弦更丰富的周期表达）
        # 向量化生成：一次性计算所有 k 的正弦/余弦，避免 Python 循环
        dayofyear = dt.dt.dayofyear.astype(float)
        k_values = np.arange(1, self.fourier_terms + 1)
        # 使用广播机制：shape (n_samples, 1) * shape (1, k) = shape (n_samples, k)
        angle_matrix = (2 * np.pi * k_values / 365.25) * dayofyear.values[:, None]
        for idx, k in enumerate(k_values):
            new_features[f'{self._dt_col}_fourier_year_sin_{k}'] = np.sin(angle_matrix[:, idx])
            new_features[f'{self._dt_col}_fourier_year_cos_{k}'] = np.cos(angle_matrix[:, idx])

        dayofweek = dt.dt.dayofweek.astype(float)
        k_values_week = np.arange(1, min(self.fourier_terms + 1, 4))
        angle_matrix_week = (2 * np.pi * k_values_week / 7) * dayofweek.values[:, None]
        for idx, k in enumerate(k_values_week):
            new_features[f'{self._dt_col}_fourier_week_sin_{k}'] = np.sin(angle_matrix_week[:, idx])
            new_features[f'{self._dt_col}_fourier_week_cos_{k}'] = np.cos(angle_matrix_week[:, idx])

        # 3. 节假日特征
        if self.holiday_calendar:
            holiday_features = self._generate_holiday_features(dt)
            for name, series in holiday_features.items():
                new_features[name] = series

        # 4. 目标相关的滞后和滑动特征（仅当目标列存在时）
        if self.target_col and self.target_col in X.columns:
            y = X[self.target_col]

            # 滞后特征
            for lag in self.lag_periods:
                new_features[f'{self.target_col}_lag_{lag}'] = y.shift(lag)

            # 滑动统计
            for window in self.rolling_windows:
                new_features[f'{self.target_col}_rolling_mean_{window}'] = y.shift(1).rolling(window=window, min_periods=1).mean()
                new_features[f'{self.target_col}_rolling_std_{window}'] = y.shift(1).rolling(window=window, min_periods=1).std()
                new_features[f'{self.target_col}_rolling_max_{window}'] = y.shift(1).rolling(window=window, min_periods=1).max()
                new_features[f'{self.target_col}_rolling_min_{window}'] = y.shift(1).rolling(window=window, min_periods=1).min()

            # 指数加权移动平均（EWMA）
            new_features[f'{self.target_col}_ewma_{self.trend_window}'] = y.shift(1).ewm(span=self.trend_window, min_periods=1).mean()

            # 趋势分解（简单线性趋势残差）
            x_numeric = (dt.astype('int64') // 1e9).astype('float64')
            if len(x_numeric) > 1 and x_numeric.std() > 0:
                slope, intercept = np.polyfit(x_numeric.dropna().index, x_numeric.dropna(), 1)
                trend = slope * x_numeric + intercept
                new_features[f'{self.target_col}_trend_residual'] = y - trend

            # 季节性残差
            for period, medians in self._seasonal_medians.items():
                if period == 7:
                    groups = dt.dt.dayofweek
                elif period == 30:
                    groups = dt.dt.day
                elif period == 12:
                    groups = dt.dt.month
                elif period == 365:
                    groups = dt.dt.dayofyear
                elif period == 24:
                    groups = dt.dt.hour
                else:
                    continue

                seasonal_component = groups.map(medians).fillna(y.median())
                new_features[f'{self.target_col}_seasonal_residual_{period}'] = y - seasonal_component

        # 5. 合并新特征
        for name, series in new_features.items():
            X[name] = series

        # 6. 填充滞后/滑动特征中的 NaN
        num_new = [c for c in new_features.keys() if c in X.columns]
        X[num_new] = X[num_new].fillna(0)

        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def _detect_datetime_col(self, X: pd.DataFrame) -> Optional[str]:
        """自动检测时间列"""
        for col in X.columns:
            if pd.api.types.is_datetime64_any_dtype(X[col]):
                return col
        # 尝试转换常见时间列名
        for col in X.columns:
            if any(kw in col.lower() for kw in ['date', 'time', 'dt', 'timestamp', 'created', 'updated']):
                try:
                    pd.to_datetime(X[col], errors='raise')
                    return col
                except Exception:
                    continue
        return None

    def _generate_holiday_features(self, dt: pd.Series) -> Dict[str, pd.Series]:
        """生成节假日特征"""
        features = {}
        year_range = range(dt.dt.year.min(), dt.dt.year.max() + 1)

        if self.holiday_calendar == 'CN':
            # 中国主要节假日（简化版，每年固定日期）
            holidays = {
                'new_year': [(1, 1)],
                'labor_day': [(5, 1)],
                'national_day': [(10, 1), (10, 2), (10, 3)],
                'spring_festival': [],  # 农历，需要更复杂的计算
            }
            for name, dates in holidays.items():
                mask = pd.Series(False, index=dt.index)
                for y in year_range:
                    for month, day in dates:
                        mask |= (dt.dt.month == month) & (dt.dt.day == day)
                features[f'holiday_{name}'] = mask.astype('float64')

            # 周末 + 调休简化：标记周末
            features['is_weekend'] = (dt.dt.dayofweek >= 5).astype('float64')

        elif self.holiday_calendar == 'US':
            us_holidays = {
                'new_year': [(1, 1)],
                'independence_day': [(7, 4)],
                'christmas': [(12, 25)],
                'thanksgiving': [],  # 11月第4个周四
            }
            for name, dates in us_holidays.items():
                mask = pd.Series(False, index=dt.index)
                for y in year_range:
                    for month, day in dates:
                        mask |= (dt.dt.month == month) & (dt.dt.day == day)
                features[f'holiday_{name}'] = mask.astype('float64')

        # 节假日窗口（节前3天 + 节后3天）
        for col_name in features.keys():
            if col_name.startswith('holiday_'):
                # 节前节后冲击
                holiday_mask = features[col_name].astype(bool)
                window_mask = pd.Series(False, index=dt.index)
                for idx in holiday_mask[holiday_mask].index:
                    date = dt.loc[idx]
                    window = (dt >= date - pd.Timedelta(days=3)) & (dt <= date + pd.Timedelta(days=3))
                    window_mask |= window
                features[f'{col_name}_window'] = window_mask.astype('float64')

        return features


class TrendExtractor:
    """
    趋势提取器

    从时间序列中提取趋势分量（线性/非线性）。
    """

    def __init__(self, method: str = 'moving_average', window: int = 30) -> None:
        self.method = method
        self.window = window

    def fit_transform(self, y: pd.Series) -> pd.Series:
        if self.method == 'moving_average':
            return y.rolling(window=self.window, min_periods=1, center=True).mean()
        elif self.method == 'linear':
            x = np.arange(len(y))
            slope, intercept = np.polyfit(x, y.fillna(y.median()), 1)
            return pd.Series(slope * x + intercept, index=y.index)
        elif self.method == 'lowess':
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(y.dropna(), y.dropna().index, frac=0.1, return_sorted=False)
                return pd.Series(smoothed, index=y.dropna().index).reindex(y.index)
            except Exception:
                return self.fit_transform(y)  # 回退到 moving_average
        else:
            raise ValueError(f"未知趋势方法: {self.method}")


class SeasonalExtractor:
    """
    季节性提取器

    提取固定周期内的季节性模式。
    """

    def __init__(self, period: int = 7) -> None:
        self.period = period
        self.seasonal_profile_: Optional[pd.Series] = None

    def fit(self, y: pd.Series) -> 'SeasonalExtractor':
        """学习季节性 profile"""
        # 按周期内位置分组求中位数
        positions = np.arange(len(y)) % self.period
        profile = pd.Series(y.values).groupby(positions).median()
        # 补全所有位置
        full_profile = pd.Series(index=range(self.period), dtype=float)
        full_profile.update(profile)
        full_profile = full_profile.fillna(full_profile.median())
        self.seasonal_profile_ = full_profile
        return self

    def transform(self, y: pd.Series) -> pd.Series:
        """返回季节性分量"""
        if self.seasonal_profile_ is None:
            raise ValueError("请先调用 fit()")
        positions = pd.Series(np.arange(len(y)) % self.period, index=y.index)
        seasonal = positions.map(self.seasonal_profile_)
        return seasonal

    def fit_transform(self, y: pd.Series) -> pd.Series:
        return self.fit(y).transform(y)
