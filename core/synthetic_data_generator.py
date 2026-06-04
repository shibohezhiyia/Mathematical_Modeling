"""
Synthetic Data Generator

Generates realistic datasets from problem descriptions for
modeling competitions without provided data.
"""
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from sklearn.datasets import make_classification, make_regression, make_blobs


def generate_synthetic_data(
    task_type: str = 'classification',
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: Optional[int] = None,
    n_classes: Optional[int] = None,
    noise: float = 0.1,
    random_state: int = 42,
    **kwargs
) -> pd.DataFrame:
    """Generate a synthetic dataset based on task type."""
    if task_type == 'classification':
        n_classes = n_classes or 2
        n_informative = n_informative or max(2, n_features // 2)
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=max(0, n_features - n_informative - 1),
            n_classes=n_classes,
            flip_y=noise,
            random_state=random_state,
            **{k: v for k, v in kwargs.items() if k not in ('n_samples', 'n_features', 'n_informative', 'n_classes', 'noise', 'random_state')}
        )
        target_name = 'target'
    elif task_type == 'regression':
        n_informative = n_informative or max(2, n_features // 2)
        X, y = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            noise=noise * 100,
            random_state=random_state,
            **{k: v for k, v in kwargs.items() if k not in ('n_samples', 'n_features', 'n_informative', 'noise', 'random_state')}
        )
        target_name = 'target'
    elif task_type == 'clustering':
        n_classes = n_classes or 3
        X, y = make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=n_classes,
            cluster_std=noise + 0.5,
            random_state=random_state,
            **{k: v for k, v in kwargs.items() if k not in ('n_samples', 'n_features', 'n_classes', 'noise', 'random_state')}
        )
        target_name = 'cluster'
    elif task_type == 'time_series':
        return _generate_time_series(n_samples, n_features, noise, random_state)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df[target_name] = y
    return df


def _generate_time_series(n_samples: int, n_features: int, noise: float, random_state: int) -> pd.DataFrame:
    rng = np.random.RandomState(random_state)
    t = np.arange(n_samples)
    df = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='D')})
    
    # Target: trend + seasonality + noise
    trend = 0.05 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 365.25)
    target = trend + seasonality + rng.normal(0, noise * 10, n_samples)
    df['target'] = target
    
    # Features: lag features and external regressors
    for i in range(min(n_features, 5)):
        df[f'feature_{i}'] = target + rng.normal(0, noise * 5, n_samples)
    for i in range(5, n_features):
        df[f'feature_{i}'] = rng.randn(n_samples)
    
    return df


def generate_from_description(description: str, **overrides) -> pd.DataFrame:
    """Parse a simple problem description and generate matching data.
    
    Examples:
        'binary classification with 5 features and 500 samples'
        'regression problem, 10 features, high noise'
    """
    desc = description.lower()
    params = {
        'task_type': 'classification',
        'n_samples': 1000,
        'n_features': 10,
        'noise': 0.1,
    }
    
    if 'regression' in desc:
        params['task_type'] = 'regression'
    elif 'cluster' in desc:
        params['task_type'] = 'clustering'
    elif 'time' in desc or 'series' in desc or 'forecast' in desc:
        params['task_type'] = 'time_series'
    
    # Extract numbers
    _NUMBER_RE = re.compile(r'(\d+)')
    numbers = _NUMBER_RE.findall(desc)
    if len(numbers) >= 1:
        params['n_samples'] = int(numbers[0])
    if len(numbers) >= 2:
        params['n_features'] = int(numbers[1])
    if len(numbers) >= 3:
        params['n_classes'] = int(numbers[2])
    
    if 'high noise' in desc or 'noisy' in desc:
        params['noise'] = 0.3
    elif 'low noise' in desc:
        params['noise'] = 0.05
    
    params.update(overrides)
    return generate_synthetic_data(**params)
