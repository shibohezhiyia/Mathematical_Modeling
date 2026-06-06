"""
Data Quality Report Engine

Generates a diagnostic report before training to help users
identify and fix data issues early.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def generate_data_quality_report(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    task_type: Optional[str] = None
) -> Dict[str, Any]:
    """Generate a comprehensive data quality report."""
    report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'memory_mb': round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        'missing_values': _missing_value_report(df),
        'duplicates': _duplicate_report(df),
        'outliers': _outlier_report(df),
        'correlations': _correlation_report(df, target_col),
        'constant_columns': _constant_columns_report(df),
        'high_cardinality': _high_cardinality_report(df),
        'target_leakage': _target_leakage_report(df, target_col),
    }
    if target_col and target_col in df.columns:
        report['target'] = _target_report(df[target_col], task_type)
    return report


def _missing_value_report(df: pd.DataFrame) -> Dict[str, Any]:
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    cols_with_missing = missing[missing > 0]
    result = {
        'total_missing_cells': int(missing.sum()),
        'columns_with_missing': int(len(cols_with_missing)),
        'details': []
    }
    for col in cols_with_missing.index:
        result['details'].append({
            'column': col,
            'missing_count': int(missing[col]),
            'missing_percent': float(missing_pct[col]),
            'suggestion': 'Consider imputation or drop' if missing_pct[col] < 50 else 'Consider dropping column'
        })
    # Sort by missing percent desc
    result['details'].sort(key=lambda x: x['missing_percent'], reverse=True)
    return result


def _duplicate_report(df: pd.DataFrame) -> Dict[str, Any]:
    n_duplicates = int(df.duplicated().sum())
    pct = round(n_duplicates / len(df) * 100, 2) if len(df) > 0 else 0.0
    return {
        'duplicate_rows': n_duplicates,
        'duplicate_percent': pct,
        'suggestion': 'Consider dropping duplicates' if n_duplicates > 0 else 'No duplicates found'
    }


def _outlier_report(df: pd.DataFrame) -> Dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    details = []
    total_outliers = 0
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) < 10:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = series[(series < lower) | (series > upper)]
        n_outliers = len(outliers)
        if n_outliers > 0:
            total_outliers += n_outliers
            details.append({
                'column': col,
                'outlier_count': int(n_outliers),
                'outlier_percent': round(n_outliers / len(series) * 100, 2),
                'bounds': [float(lower), float(upper)]
            })
    details.sort(key=lambda x: x['outlier_percent'], reverse=True)
    return {
        'total_outliers': total_outliers,
        'columns_with_outliers': len(details),
        'details': details[:10]  # top 10
    }


def _correlation_report(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return {'high_correlation_pairs': [], 'target_correlations': []}
    corr = numeric_df.corr().abs()
    # High correlation pairs (|r| > 0.95) — 向量化提取上三角
    mask = (corr.values > 0.95) & np.triu(np.ones_like(corr, dtype=bool), k=1)
    rows, cols_idx = np.where(mask)
    pairs = [
        {
            'col1': corr.index[i],
            'col2': corr.columns[j],
            'correlation': round(float(corr.iloc[i, j]), 4)
        }
        for i, j in zip(rows, cols_idx)
    ]
    # Target correlations — 向量化：直接从 corr 矩阵提取
    target_cors = []
    if target_col and target_col in numeric_df.columns:
        target_corr = corr[target_col].drop(target_col, errors='ignore')
        target_cors = [
            {'column': col, 'correlation': round(float(val), 4)}
            for col, val in target_corr.items()
            if not np.isnan(val)
        ]
        target_cors.sort(key=lambda x: x['correlation'], reverse=True)
    return {
        'high_correlation_pairs': pairs,
        'target_correlations': target_cors[:10]
    }


def _constant_columns_report(df: pd.DataFrame) -> Dict[str, Any]:
    # 向量化：使用 nunique() 一次性计算所有列的唯一值数
    nunique = df.nunique(dropna=False)
    constants = nunique[nunique <= 1]
    result = []
    for col in constants.index:
        val = str(df[col].iloc[0]) if len(df[col]) > 0 else 'N/A'
        result.append({'column': col, 'value': val})
    return {'count': len(result), 'columns': result}


def _high_cardinality_report(df: pd.DataFrame, threshold: float = 0.9) -> Dict[str, Any]:
    # 向量化：使用 nunique() 一次性计算所有列的唯一值数
    obj_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(obj_cols) == 0:
        return {'count': 0, 'columns': []}
    n_unique = df[obj_cols].nunique()
    ratios = n_unique / len(df) if len(df) > 0 else 0
    high_card = ratios[ratios >= threshold]
    details = [
        {'column': col, 'unique_count': int(n_unique[col]), 'unique_ratio': round(ratio, 4)}
        for col, ratio in high_card.items()
    ]
    return {'count': len(details), 'columns': details}


def _target_leakage_report(df: pd.DataFrame, target_col: Optional[str] = None) -> Dict[str, Any]:
    if not target_col or target_col not in df.columns:
        return {'count': 0, 'columns': []}
    y = df[target_col]
    leaks = []
    # 向量化：一次性计算所有数值列与目标的相关系数
    numeric_cols = df.select_dtypes(include=[np.number]).columns.difference([target_col])
    if len(numeric_cols) > 0:
        corrs = df[numeric_cols].corrwith(y).abs()
        perfect_corr = corrs[corrs > 0.999]
        for col in perfect_corr.index:
            leaks.append({'column': col, 'reason': 'perfect correlation'})
    # 检查非数值列的一对一映射
    for col in df.columns:
        if col == target_col or col in numeric_cols:
            continue
        if df[col].nunique() == y.nunique() and (df.groupby(col)[target_col].nunique() == 1).all():
            leaks.append({'column': col, 'reason': 'one-to-one mapping with target'})
    return {'count': len(leaks), 'columns': leaks}


def _target_report(y: pd.Series, task_type: Optional[str] = None) -> Dict[str, Any]:
    report = {}
    if task_type == 'classification' or (y.dtype == 'object' or y.nunique() <= 20):
        vc = y.value_counts()
        total = len(y)
        classes = []
        for cls, cnt in vc.items():
            classes.append({
                'class': str(cls),
                'count': int(cnt),
                'percent': round(cnt / total * 100, 2)
            })
        imbalance_ratio = round(vc.iloc[0] / vc.iloc[-1], 2) if len(vc) > 1 else 1.0
        report['type'] = 'classification'
        report['n_classes'] = int(y.nunique())
        report['class_distribution'] = classes
        report['imbalance_ratio'] = imbalance_ratio
        report['suggestion'] = 'Consider class balancing' if imbalance_ratio > 5 else 'Balanced'
    else:
        report['type'] = 'regression'
        report['mean'] = round(float(y.mean()), 4)
        report['std'] = round(float(y.std()), 4)
        report['min'] = round(float(y.min()), 4)
        report['max'] = round(float(y.max()), 4)
        report['skewness'] = round(float(y.skew()), 4)
        report['suggestion'] = 'Check for extreme skewness' if abs(y.skew()) > 3 else 'OK'
    return report
