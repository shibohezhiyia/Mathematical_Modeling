"""
Data Auto-Fix Engine

Automatically fixes common data quality issues based on
the data quality report recommendations.
"""
from typing import Any, Dict, Optional

import pandas as pd


def autofix_dataframe(
    df: pd.DataFrame,
    report: Optional[Dict[str, Any]] = None,
    target_col: Optional[str] = None,
    drop_high_missing: bool = True,
    missing_threshold: float = 50.0,
    fix_outliers: bool = False,
    drop_duplicates: bool = True
) -> pd.DataFrame:
    """Automatically fix common data quality issues."""
    df = df.copy()
    fixes = []

    if drop_duplicates:
        n_before = len(df)
        df = df.drop_duplicates()
        n_dropped = n_before - len(df)
        if n_dropped > 0:
            fixes.append(f'Dropped {n_dropped} duplicate rows')

    if report and 'missing_values' in report:
        missing_details = report['missing_values'].get('details', [])
        cols_to_drop = []
        for d in missing_details:
            col = d['column']
            if col == target_col:
                continue
            if drop_high_missing and d['missing_percent'] >= missing_threshold:
                cols_to_drop.append(col)
            else:
                if df[col].dtype.kind in 'iufc':
                    median = df[col].median()
                    df[col] = df[col].fillna(median)
                    fixes.append(f'Filled missing in {col} with median ({median:.4f})')
                else:
                    mode = df[col].mode()
                    if len(mode) > 0:
                        df[col] = df[col].fillna(mode.iloc[0])
                        fixes.append(f'Filled missing in {col} with mode ({mode.iloc[0]})')
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            fixes.append(f'Dropped columns with >= {missing_threshold}% missing: {cols_to_drop}')

    if fix_outliers and report and 'outliers' in report:
        outlier_details = report['outliers'].get('details', [])
        for d in outlier_details:
            col = d['column']
            if col == target_col:
                continue
            if col not in df.columns:
                continue
            bounds = d.get('bounds')
            if bounds and len(bounds) == 2:
                lower, upper = bounds
                before = (df[col] < lower).sum() + (df[col] > upper).sum()
                df[col] = df[col].clip(lower=lower, upper=upper)
                if before > 0:
                    fixes.append(f'Clipped {before} outliers in {col} to [{lower:.4f}, {upper:.4f}]')

    return df, fixes
