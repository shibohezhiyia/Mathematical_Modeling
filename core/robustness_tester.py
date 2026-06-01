"""
Robustness Tester

Tests model resilience against noise, missing values, distribution shift, etc.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

from sklearn.metrics import accuracy_score, mean_squared_error


def evaluate_robustness(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str = 'classification',
    metric_fn = None
) -> Dict[str, Any]:
    """Run a battery of robustness tests."""
    if metric_fn is None:
        metric_fn = accuracy_score if task_type == 'classification' else mean_squared_error
    
    baseline_pred = model.predict(X)
    baseline_score = metric_fn(y, baseline_pred)
    
    results = {
        'baseline': round(float(baseline_score), 4),
        'tests': []
    }
    
    # 1. Gaussian noise
    for noise_level in [0.01, 0.05, 0.1]:
        X_noisy = X.copy()
        numeric_cols = X_noisy.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            std = X_noisy[col].std()
            if std > 0:
                X_noisy[col] += np.random.normal(0, noise_level * std, len(X_noisy))
        pred = model.predict(X_noisy)
        score = metric_fn(y, pred)
        results['tests'].append({
            'name': f'Gaussian noise ({noise_level*100:.0f}%)',
            'score': round(float(score), 4),
            'drop': round(float(baseline_score - score), 4)
        })
    
    # 2. Missing values (MCAR)
    for missing_rate in [0.05, 0.1, 0.2]:
        X_miss = X.copy()
        numeric_cols = X_miss.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            mask = np.random.rand(len(X_miss)) < missing_rate
            X_miss.loc[mask, col] = np.nan
        X_miss = X_miss.fillna(X_miss.median())
        pred = model.predict(X_miss)
        score = metric_fn(y, pred)
        results['tests'].append({
            'name': f'Missing values ({missing_rate*100:.0f}%)',
            'score': round(float(score), 4),
            'drop': round(float(baseline_score - score), 4)
        })
    
    # 3. Feature permutation (shuffle one feature at a time)
    importance_drops = []
    for col in X.columns:
        X_perm = X.copy()
        X_perm[col] = np.random.permutation(X_perm[col].values)
        pred = model.predict(X_perm)
        score = metric_fn(y, pred)
        importance_drops.append({
            'feature': col,
            'drop': round(float(baseline_score - score), 4)
        })
    importance_drops.sort(key=lambda x: abs(x['drop']), reverse=True)
    results['feature_sensitivity'] = importance_drops[:10]
    
    # 4. Overall robustness score
    drops = [t['drop'] for t in results['tests']]
    avg_drop = np.mean(drops) if drops else 0
    results['robustness_score'] = round(max(0, 100 - avg_drop * 100), 1)
    results['summary'] = 'Robust' if results['robustness_score'] > 80 else ('Moderate' if results['robustness_score'] > 50 else 'Fragile')
    
    return results
