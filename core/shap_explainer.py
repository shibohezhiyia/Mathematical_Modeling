"""
SHAP Explainability Engine

Generates SHAP-based model explanations for better interpretability.
"""
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

import shap


def explain_model(
    model: Any,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    task_type: str = 'classification',
    sample_size: int = 100,
    plot_type: str = 'summary'
) -> Dict[str, Any]:
    """Generate SHAP explanations for a fitted model.
    
    Args:
        model: Fitted sklearn/pytorch model
        X: Feature DataFrame
        y: Optional target series
        task_type: 'classification' or 'regression'
        sample_size: Background sample size for SHAP
        plot_type: 'summary', 'bar', 'waterfall'
    
    Returns:
        Dict with shap_values, feature_importance, and plot data
    """
    X_sample = X.sample(min(sample_size, len(X)), random_state=42) if len(X) > sample_size else X
    
    # Select explainer based on model type
    explainer = _select_explainer(model, X_sample)
    
    # Compute SHAP values (limit to sample for speed)
    X_explain = X.sample(min(200, len(X)), random_state=43) if len(X) > 200 else X
    shap_values = explainer.shap_values(X_explain)
    
    # Handle multi-class SHAP values
    shap_arr = np.array(shap_values)
    if shap_arr.ndim == 3:
        # (n_samples, n_features, n_classes) -> take positive class for binary, mean for multi
        if shap_arr.shape[2] == 2:
            shap_arr = shap_arr[:, :, 1]
        else:
            shap_arr = shap_arr.mean(axis=2)
    elif isinstance(shap_values, list):
        shap_arr = np.abs(np.array(shap_values)).mean(axis=0)
    
    # Ensure shape is (n_samples, n_features)
    if shap_arr.ndim == 1:
        shap_arr = shap_arr.reshape(1, -1)
    
    # Feature importance from mean |SHAP|
    mean_shap = np.abs(shap_arr).mean(axis=0)
    feature_importance = []
    for i, col in enumerate(X.columns):
        feature_importance.append({
            'feature': col,
            'importance': round(float(mean_shap[i]), 6)
        })
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    # Generate sample explanations (top 5 instances)
    instance_explanations = []
    n_samples = min(5, len(X_explain))
    for idx in range(n_samples):
        instance_shap = shap_arr[idx]
        contributions = []
        for i, col in enumerate(X.columns):
            contributions.append({
                'feature': col,
                'value': round(float(X_explain.iloc[idx, i]), 4),
                'shap': round(float(instance_shap[i]), 6)
            })
        contributions.sort(key=lambda x: abs(x['shap']), reverse=True)
        instance_explanations.append({
            'index': int(X_explain.index[idx]),
            'top_features': contributions[:8]
        })
    
    return {
        'feature_importance': feature_importance,
        'instance_explanations': instance_explanations,
        'plot_type': plot_type,
        'n_background': len(X_sample),
        'n_explained': len(X_explain)
    }


def _select_explainer(model: Any, X_background: pd.DataFrame) -> Any:
    """Select appropriate SHAP explainer based on model type."""
    model_type = type(model).__name__.lower()
    module_name = type(model).__module__.lower()
    
    # Tree-based models
    if any(k in model_type for k in ('xgb', 'lgbm', 'catboost', 'randomforest', 'extratrees', 'gradientboosting', 'decisiontree')):
        try:
            return shap.TreeExplainer(model)
        except Exception:
            pass
    
    # Linear models
    if any(k in model_type for k in ('logisticregression', 'ridge', 'lasso', 'elasticnet', 'linearregression', 'sgd')):
        try:
            return shap.LinearExplainer(model, X_background)
        except Exception:
            pass
    
    # Deep learning models (PyTorch)
    if 'torch' in module_name or 'nn' in module_name:
        try:
            return shap.DeepExplainer(model, X_background.values)
        except Exception:
            pass
    
    # Fallback to KernelExplainer
    return shap.KernelExplainer(model.predict, X_background)
