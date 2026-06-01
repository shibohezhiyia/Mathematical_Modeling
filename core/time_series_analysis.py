"""
Time Series Analysis Module

ACF/PACF, decomposition, stationarity tests, Prophet integration.
"""
import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def analyze_time_series(
    series: pd.Series,
    freq: Optional[str] = None,
    period: Optional[int] = None
) -> Dict[str, Any]:
    """Comprehensive time series analysis."""
    result = {}
    
    # Basic stats
    result['n_obs'] = len(series)
    result['mean'] = round(float(series.mean()), 4)
    result['std'] = round(float(series.std()), 4)
    result['min'] = round(float(series.min()), 4)
    result['max'] = round(float(series.max()), 4)
    
    # Stationarity test (ADF)
    try:
        from statsmodels.tsa.stattools import adfuller
        adf = adfuller(series.dropna())
        result['adf_statistic'] = round(float(adf[0]), 4)
        result['adf_pvalue'] = round(float(adf[1]), 4)
        result['is_stationary'] = adf[1] < 0.05
    except Exception:
        result['adf_statistic'] = None
        result['adf_pvalue'] = None
        result['is_stationary'] = None
    
    # Autocorrelation
    try:
        from statsmodels.tsa.stattools import acf, pacf
        acf_vals = acf(series.dropna(), nlags=min(20, len(series)//2), fft=True)
        pacf_vals = pacf(series.dropna(), nlags=min(20, len(series)//2))
        result['acf'] = [round(float(v), 4) for v in acf_vals[:10]]
        result['pacf'] = [round(float(v), 4) for v in pacf_vals[:10]]
    except Exception:
        result['acf'] = []
        result['pacf'] = []
    
    # Seasonal decomposition
    if period is None and freq is not None:
        freq_map = {'D': 7, 'W': 52, 'M': 12, 'Q': 4, 'H': 24}
        period = freq_map.get(freq, None)
    
    if period and len(series) >= period * 2:
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            decomp = seasonal_decompose(series.dropna(), model='additive', period=period)
            result['trend'] = [round(float(v), 4) for v in decomp.trend.dropna().values[:50]]
            result['seasonal'] = [round(float(v), 4) for v in decomp.seasonal.dropna().values[:50]]
            result['resid'] = [round(float(v), 4) for v in decomp.resid.dropna().values[:50]]
        except Exception:
            pass
    
    return result


def prophet_forecast(series: pd.Series, periods: int = 10) -> Optional[Dict]:
    """Run Prophet forecast if available."""
    try:
        from prophet import Prophet
        df = pd.DataFrame({'ds': series.index, 'y': series.values})
        m = Prophet(daily_seasonality=False)
        m.fit(df)
        future = m.make_future_dataframe(periods=periods)
        forecast = m.predict(future)
        return {
            'forecast': forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).to_dict('records'),
            'trend': forecast['trend'].tail(periods).tolist(),
        }
    except Exception:
        return None
